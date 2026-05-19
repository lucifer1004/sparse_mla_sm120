"""Reproducer for the MG_N_HG_T=1 + PAGE_BLOCK_SIZE_EXTRA=64 (C4A) bug.

Found via dsv4-workspace bench-replay against captured TP=4 production
shapes. See vllm dsv4-sm120 commit 80bec09b7 for the revert.

Symptoms at NUM_HEADS=16 + dual-cache + block_size_extra=64:
  - ~30% of output tokens have ULP-distance 40-100 vs torch-fp32 reference
  - No-sink path: kernel produces NaN in those tokens (sigmoid factor in
    the attn_sink path masks them to finite garbage, hiding the bug
    end-to-end until you compare against a reference)

Reproduces only at production-scale cache (n_blocks ≥ ~16384) with full
input range — small-cache unit tests (n_blocks=32) see clean output.
Trigger condition cross-product:

  num_heads=16  AND  block_size_extra=64  AND  n_blocks ≥ ~16384
                                          AND  inputs at clamp(-4,4) scale

Same kernel at num_heads ∈ {32, 64, 128} (MG_N_HG_T=2) is clean.
Same kernel at block_size_extra=2 (C128A) is clean.

Hypothesis (unverified): smem layout overlap in xv_rope_mma. The
BF16-weight stride is BI=64 while the FP8 layout used for XV nope MMA
is BI+16=80. The buffer is reused across phases via bar_sync — works
at MG_N_HG_T=2 (two groups iterate per tile, two synced writes), may
have a missed sync at MG_N_HG_T=1 (single group, one shot). Needs C++
deep-dive.

Usage:
  python -m benchmarks.repro_c4a_bug
"""
import argparse
import torch

import flash_mla_sm120
from tests.test_decode import quantize_kv_model1, dequantize_kv_model1

_unwrapped = flash_mla_sm120.interface.flash_mla_sparse_fwd
if hasattr(_unwrapped, "__wrapped__"):
    _unwrapped = _unwrapped.__wrapped__


def ref_attn(q, kv_main_dq, idx_main, kv_extra_dq, idx_extra,
             sm_scale, d_v, attn_sink=None):
    n_tokens, h_q, d_qk = q.shape
    q_f = q.float()
    main_flat = kv_main_dq.view(-1, d_qk).float()
    gathered = main_flat.index_select(
        0, idx_main.clamp(min=0).view(-1)
    ).view(n_tokens, idx_main.size(-1), d_qk)
    invalid = idx_main < 0
    extra_flat = kv_extra_dq.view(-1, d_qk).float()
    gathered_extra = extra_flat.index_select(
        0, idx_extra.clamp(min=0).view(-1)
    ).view(n_tokens, idx_extra.size(-1), d_qk)
    gathered = torch.cat([gathered, gathered_extra], dim=-2)
    invalid = torch.cat([invalid, idx_extra < 0], dim=-1)
    P = torch.einsum("nhd,ntd->nht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out = torch.einsum("nht,ntd->nhd", weights,
                        gathered[..., :d_v]).to(torch.bfloat16)
    if attn_sink is not None:
        factor = torch.sigmoid(lse - attn_sink.float().unsqueeze(0))
        out = (out.float() * factor.unsqueeze(-1)).to(torch.bfloat16)
    return out


def run_case(*, num_heads, n_tokens, n_blocks, block_size_extra,
             topk=128, topk_extra=512, block_size=64,
             input_clamp=4.0, with_sink=False, seed=0):
    torch.manual_seed(seed)
    d_qk = d_v = 512
    sm_scale = d_qk ** -0.5

    kv_main_bf16 = torch.randn(n_blocks, block_size, 1, d_qk,
                                device="cuda", dtype=torch.bfloat16
                                ).clamp(-input_clamp, input_clamp)
    kv_extra_bf16 = torch.randn(n_blocks, block_size_extra, 1, d_qk,
                                 device="cuda", dtype=torch.bfloat16
                                 ).clamp(-input_clamp, input_clamp)
    kv_main_packed = quantize_kv_model1(kv_main_bf16)
    kv_main_dq = dequantize_kv_model1(kv_main_packed)
    del kv_main_bf16
    kv_extra_packed = quantize_kv_model1(kv_extra_bf16)
    kv_extra_dq = dequantize_kv_model1(kv_extra_packed)
    del kv_extra_bf16

    s_main = n_blocks * block_size
    s_extra = n_blocks * block_size_extra
    q = torch.randn(n_tokens, num_heads, d_qk,
                    device="cuda", dtype=torch.bfloat16
                    ).clamp(-input_clamp, input_clamp)
    idx_main = torch.randint(0, s_main, (n_tokens, topk),
                              device="cuda", dtype=torch.int32)
    idx_main[:, -5:] = -1
    idx_extra = torch.randint(0, s_extra, (n_tokens, topk_extra),
                               device="cuda", dtype=torch.int32)
    idx_extra[:, -5:] = -1

    attn_sink = None
    if with_sink:
        attn_sink = torch.tensor(
            [(-1.0) ** i * (0.5 + 0.1 * (i % 7)) for i in range(num_heads)],
            device="cuda", dtype=torch.float32,
        )

    out_kernel = _unwrapped(
        q, kv_main_packed, idx_main, sm_scale, d_v=d_v,
        attn_sink=attn_sink, extra_k_cache=kv_extra_packed,
        extra_indices_in_kvcache=idx_extra,
    )[0]
    out_ref = ref_attn(q, kv_main_dq, idx_main, kv_extra_dq, idx_extra,
                       sm_scale, d_v, attn_sink=attn_sink)
    diff = (out_kernel.float() - out_ref.float()).abs()
    n_nan = int(out_kernel.isnan().sum().item())
    max_atol = diff.max().item()
    peak_ref = out_ref.abs().max().item()
    ulp = max_atol / max(peak_ref * 7.8125e-3, 1e-12)
    return dict(
        n_nan=n_nan,
        max_atol=max_atol,
        peak_ref=peak_ref,
        ulp=ulp,
        n_bad=int((diff.view(diff.shape[0], -1).max(dim=-1).values > 0.05).sum().item()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-blocks", type=int, default=16384,
                   help="cache size; bug only triggers at ≥ ~16384")
    p.add_argument("--n-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"=== axis sweep, n_blocks={args.n_blocks} n_tokens={args.n_tokens} ===")
    print(f"{'config':38s} {'n_nan':>6}  {'atol':>8}  {'peak':>6}  {'ULP':>6}  {'bad_tok':>7}")
    print("-" * 78)
    for num_heads in (16, 32, 64, 128):
        for bse in (2, 64):
            try:
                r = run_case(
                    num_heads=num_heads, n_tokens=args.n_tokens,
                    n_blocks=args.n_blocks, block_size_extra=bse,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"h={num_heads:3d} bse={bse:2d}: "
                      f"{type(e).__name__}: {e}")
                continue
            tag = " ← BUG" if r["ulp"] > 20 or r["n_nan"] > 0 else ""
            label = f"h={num_heads:3d} bse={bse:2d}"
            print(f"{label:38s} {r['n_nan']:6d}  "
                  f"{r['max_atol']:8.4e}  {r['peak_ref']:6.3f}  "
                  f"{r['ulp']:6.1f}  {r['n_bad']:7d}{tag}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
