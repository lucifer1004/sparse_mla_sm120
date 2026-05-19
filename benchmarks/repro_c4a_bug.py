"""Reproducer for the MG_N_HG_T=1 prefill race condition (FIXED).

Two races were caught by `compute-sanitizer --tool racecheck`:

(1) ``xv_rope_mma`` / XV-nope per-vc race in prefill_kernel.cuh

    The XV-nope MMA loop reused the SAME ``sm.w_fp8`` smem region across
    vc-iterations, with a barrier only BETWEEN write/read of one iter and
    AFTER the outer loop — not between vc=k's reads and vc=k+1's writes.
    With MG_N_HG_T=2 (NUM_HEADS=32/64/128) each vc iter processed two head
    groups, naturally widening the read window so the next iter's writes
    didn't catch the previous ldmatrix in flight. MG_N_HG_T=1 (NUM_HEADS=16)
    halved that work, narrowed the window, and exposed the race.

(2) ``io_gather_scales`` vs math read race in IO warps

    The IO warps issued ``io_bulk_gather_tile`` (cp.async.bulk, signals
    ``mbar_kv``) BEFORE ``io_gather_scales`` (synchronous plain stores, no
    mbar). The math warps woke up on ``mbar_kv`` after the bulk copy
    completed — which could happen before scales finished. Reordering
    scales-then-bulk + ``__threadfence_block`` places scales-visibility on
    the same side of mbar-signal as the bulk data. Latent in MG_N_HG_T=2
    because the longer math path absorbed the window.

Symptoms before the fix (with attn_sink, full input range, large grid):
  - ~0.05-7% of output elements were NaN (data-dependent and grid-size
    dependent — sharper at n_tokens in [128, 4096] with n_blocks≥16K)
  - Finite values reached ~4.0 (input clamp range) instead of ~0.5

Usage:
  python -m benchmarks.repro_c4a_bug             # axis sweep
  python -m benchmarks.repro_c4a_bug --quick    # one trigger config only
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
             input_clamp=4.0, with_sink=True, seed=0,
             n_tokens_corr=256):
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

    # Full-grid kernel call (the bug is grid-size-dependent), then
    # subset for the bf16 reference to keep host memory under control.
    out_kernel_full = _unwrapped(
        q, kv_main_packed, idx_main, sm_scale, d_v=d_v,
        attn_sink=attn_sink, extra_k_cache=kv_extra_packed,
        extra_indices_in_kvcache=idx_extra,
    )[0]
    n_tokens_corr = min(n_tokens_corr, n_tokens)
    out_kernel = out_kernel_full[:n_tokens_corr].contiguous()
    q_s = q[:n_tokens_corr].contiguous()
    idx_s = idx_main[:n_tokens_corr].contiguous()
    eidx_s = idx_extra[:n_tokens_corr].contiguous()
    out_ref = ref_attn(q_s, kv_main_dq, idx_s, kv_extra_dq, eidx_s,
                       sm_scale, d_v, attn_sink=attn_sink)
    diff = (out_kernel.float() - out_ref.float()).abs()
    n_nan_full = int(out_kernel_full.isnan().sum().item())
    n_nan_corr = int(out_kernel.isnan().sum().item())
    max_atol = diff.max().item() if n_nan_corr == 0 else float("nan")
    peak_ref = out_ref.abs().max().item()
    ulp = (max_atol / max(peak_ref * 7.8125e-3, 1e-12)
           if not torch.isnan(torch.tensor(max_atol)) else float("nan"))
    n_bad = int((diff.view(diff.shape[0], -1).max(dim=-1).values > 0.05
                 ).sum().item()) if n_nan_corr == 0 else -1
    return dict(
        n_nan_full=n_nan_full,
        n_nan_corr=n_nan_corr,
        max_atol=max_atol,
        peak_ref=peak_ref,
        ulp=ulp,
        n_bad=n_bad,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-blocks", type=int, default=16384)
    # Default n_tokens chosen to expose race pre-fix: small enough for
    # bf16 ref to fit in memory, large enough for grid ≥ 2 waves on
    # RTX PRO 6000 (188 SMs × REPLICATE_H=1).
    p.add_argument("--n-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Run the one trigger config that fired pre-fix")
    args = p.parse_args()

    if args.quick:
        # Most reliable pre-fix trigger.
        r = run_case(num_heads=16, n_tokens=args.n_tokens,
                     n_blocks=args.n_blocks, block_size_extra=64,
                     with_sink=True, seed=args.seed)
        print(f"h=16 bse=64 sink=True n_blocks={args.n_blocks} "
              f"n_tokens={args.n_tokens}: nan_full={r['n_nan_full']}, "
              f"nan_corr={r['n_nan_corr']}, ULP={r['ulp']:.1f}, "
              f"bad_tok={r['n_bad']}")
        return

    print(f"=== axis sweep, n_blocks={args.n_blocks} n_tokens={args.n_tokens} "
          f"with_sink=True ===")
    print(f"{'config':30s} {'nan(full)':>10}  {'nan(corr)':>10}  "
          f"{'atol':>10}  {'peak':>6}  {'ULP':>7}  {'bad_tok':>7}")
    print("-" * 96)
    for num_heads in (16, 32, 64, 128):
        for bse in (2, 64):
            try:
                r = run_case(
                    num_heads=num_heads, n_tokens=args.n_tokens,
                    n_blocks=args.n_blocks, block_size_extra=bse,
                    with_sink=True, seed=args.seed,
                )
            except Exception as e:
                print(f"h={num_heads:3d} bse={bse:2d}: "
                      f"{type(e).__name__}: {e}")
                continue
            tag = " ← BUG" if r["n_nan_full"] > 0 or (
                isinstance(r["ulp"], float) and r["ulp"] > 20) else ""
            ulp_str = (f"{r['ulp']:7.1f}" if not (
                isinstance(r["ulp"], float) and (r["ulp"] != r["ulp"]))
                else "    nan")
            atol_str = (f"{r['max_atol']:10.4e}"
                        if r["max_atol"] == r["max_atol"]
                        else "       nan")
            print(f"h={num_heads:3d} bse={bse:2d}                 "
                  f"{r['n_nan_full']:10d}  {r['n_nan_corr']:10d}  "
                  f"{atol_str}  {r['peak_ref']:6.3f}  "
                  f"{ulp_str}  {r['n_bad']:7d}{tag}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
