"""Dispatch-coverage test for the MG_N_HG_T=1 dual-cache prefill path.

Covers NUM_HEADS=16 (MG_N_HG_T=1) and NUM_HEADS=32 (MG_N_HG_T=2) across
the three DSv4-Flash dual-cache dispatch combinations:

    (topk, topk_extra, page_block_size_extra)
    (128,  128,        64)   — symmetric C4A-shape
    (128,  512,        64)   — production C4A (compress_ratio=4)
    (128,  512,         2)   — production C128A (compress_ratio=128)

The existing test_dual_cache_prefill.py only parametrizes num_heads ∈
{64, 128} — leaving NUM_HEADS=16 (the new MG_N_HG_T=1 dispatch added in
commit 88a613e) untested at the unit level. That gap let a production
bug ship: at production-scale cache (n_blocks ≥ 16384) with full-range
inputs, the MG_N_HG_T=1 + PAGE_BLOCK_SIZE_EXTRA=64 path produces ULP
40-100 vs reference. The unit-level part of that — dispatch wiring,
basic non-NaN output, small-scale precision — is in this file. The
production-scale precision regression is owned by the bench harness in
../bench/ (driver replay against captured shapes).
"""
import pytest
import torch

import flash_mla_sm120
from tests.test_decode import (
    quantize_kv_model1,
    dequantize_kv_model1,
)


# Small n_blocks matches the existing dual-cache test suite (fast). The
# kernel bug noted in the docstring is data-correlated and only fires at
# production cache scale (n_blocks ≥ 16384 with full-range inputs) — that
# case is owned by the bench harness in ../bench/, NOT this unit test.
# This file's job is dispatch-table coverage: ensure all three
# (topk_extra, page_block_size_extra) combos × num_heads ∈ {16, 32}
# instantiate, run, and produce non-NaN output within FP8-noise bounds.
_N_BLOCKS = 32
_BLOCK_SIZE = 64
_D_QK = 512
_D_V = 512


def _ref_dual_cache_attn(q, kv_main_dq, idx_main, kv_extra_dq, idx_extra,
                         sm_scale, d_v):
    """Torch-fp32 reference; matches ref_dual_cache_attn_prefill in
    test_dual_cache_prefill but lighter-weight (no attn_sink branch).
    """
    num_tokens, h_q, d_qk = q.shape
    q_f = q.float()

    main_flat = kv_main_dq.view(-1, d_qk).float()
    gathered_main = main_flat.index_select(0, idx_main.clamp(min=0).view(-1)) \
        .view(num_tokens, idx_main.size(-1), d_qk)
    extra_flat = kv_extra_dq.view(-1, d_qk).float()
    gathered_extra = extra_flat.index_select(0, idx_extra.clamp(min=0).view(-1)) \
        .view(num_tokens, idx_extra.size(-1), d_qk)
    gathered = torch.cat([gathered_main, gathered_extra], dim=-2)
    invalid = torch.cat([idx_main < 0, idx_extra < 0], dim=-1)

    P = torch.einsum("nhd,ntd->nht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    return torch.einsum("nht,ntd->nhd", weights, gathered[..., :d_v]).to(
        torch.bfloat16)


# (topk_extra, page_block_size_extra) — the three dispatch combos.
DUAL_VARIANTS = [
    pytest.param(128, 64, id="topk_ext_128_bse_64"),
    pytest.param(512, 64, id="topk_ext_512_bse_64_C4A"),
    pytest.param(512,  2, id="topk_ext_512_bse_2_C128A"),
]


@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("num_tokens", [128, 256])
@pytest.mark.parametrize("topk_extra,block_size_extra", DUAL_VARIANTS)
def test_mg_n_hg_1_dual_prefill(num_heads, num_tokens, topk_extra,
                                 block_size_extra):
    """num_heads=16 dispatches through MG_N_HG_T=1; 32 through MG_N_HG_T=2.

    The 32-variant rows verify the surrounding test setup is correct
    (those instantiations are known-good); the 16-variant rows are the
    new coverage missing from test_dual_cache_prefill.py.
    """
    torch.manual_seed(0)
    sm_scale = _D_QK ** -0.5
    topk_main = 128

    # KV caches — same allocation pattern as the existing dual-cache test
    # but at n_blocks=_N_BLOCKS so the failure-triggering memory layout
    # is reachable.
    # Match the input convention of test_dual_cache_prefill so the
    # threshold below stays comparable across the test suite.
    kv_main_bf16 = (torch.randn(_N_BLOCKS, _BLOCK_SIZE, 1, _D_QK,
                                 device="cuda", dtype=torch.bfloat16
                                 ) / 10).clamp(-1, 1)
    kv_extra_bf16 = (torch.randn(_N_BLOCKS, block_size_extra, 1, _D_QK,
                                  device="cuda", dtype=torch.bfloat16
                                  ) / 10).clamp(-1, 1)
    kv_main_packed = quantize_kv_model1(kv_main_bf16)
    kv_main_dq = dequantize_kv_model1(kv_main_packed)
    kv_extra_packed = quantize_kv_model1(kv_extra_bf16)
    kv_extra_dq = dequantize_kv_model1(kv_extra_packed)

    s_main = _N_BLOCKS * _BLOCK_SIZE
    s_extra = _N_BLOCKS * block_size_extra

    q = (torch.randn(num_tokens, num_heads, _D_QK,
                     device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    idx_main = torch.randint(0, s_main, (num_tokens, topk_main),
                              device="cuda", dtype=torch.int32)
    idx_extra = torch.randint(0, s_extra, (num_tokens, topk_extra),
                               device="cuda", dtype=torch.int32)
    idx_main[:, -5:] = -1
    idx_extra[:, -3:] = -1

    ref_out = _ref_dual_cache_attn(
        q, kv_main_dq, idx_main, kv_extra_dq, idx_extra,
        sm_scale, _D_V,
    )
    out, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, kv_main_packed, idx_main, sm_scale, _D_V,
        extra_kv_cache=kv_extra_packed,
        extra_indices=idx_extra,
    )

    n_nan = int(out.isnan().sum().item())
    max_err = (out.float() - ref_out.float()).abs().max().item()
    assert n_nan == 0, (
        f"kernel produced {n_nan} NaN elements "
        f"(h={num_heads}, tk_ext={topk_extra}, bse={block_size_extra})"
    )
    # Threshold matches test_dual_cache_prefill (0.01 — looser than the
    # 0.001 single-cache bound to absorb FP8 quant noise compounded over
    # two index sources).
    assert max_err < 0.01, (
        f"dual-cache prefill failed: h={num_heads} "
        f"tk_ext={topk_extra} bse={block_size_extra} "
        f"max_err={max_err:.4f}"
    )
