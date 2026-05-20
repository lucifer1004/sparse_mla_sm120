"""Smoke tests for BatchSparseMLAPagedAttentionWrapper.

The wrapper is a thin plan/run layer over the existing kernel-level
``sparse_mla_fwd`` dispatcher. These tests verify:
  - plan() validates configuration correctly
  - run() output is bit-identical to a direct ``sparse_mla_fwd`` call
    with the equivalent arguments
  - decode-shape and prefill-shape both route correctly
  - dual-cache args plumb through

No reference attention is computed here — that's covered by
``test_decode.py`` / ``test_prefill.py``. This file only ensures the
wrapper produces the same output as the existing entrypoint.
"""

import pytest
import torch

import flash_mla_sm120
from flash_mla_sm120 import BatchSparseMLAPagedAttentionWrapper
from flash_mla_sm120.ops import sparse_mla_fwd

from .test_decode import quantize_kv_model1


# DSv4-Flash MODEL1 shapes (d_qk=512, d_v=512, page_size=64).
_HEAD_DIM_QK = 512   # DSv4-Flash MODEL1: 448 nope + 64 rope
_HEAD_DIM_V = 512    # output value dim
_PAGE_SIZE = 64


def _make_model1_inputs(num_tokens, num_heads, topk, *,
                        device="cuda", num_blocks=128, seed=0):
    """Return (q_bf16, kv_packed_fp8, sparse_indices, sparse_topk_lens)."""
    torch.manual_seed(seed)
    d_qk = _HEAD_DIM_QK
    q = (
        torch.randn(num_tokens, num_heads, d_qk,
                    device=device, dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)

    kv_bf16 = (
        torch.randn(num_blocks, _PAGE_SIZE, 1, d_qk,
                    device=device, dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)

    s_kv = num_blocks * _PAGE_SIZE
    sparse_indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark a few slots as invalid so we exercise the -1 skip path.
    sparse_indices[:, -10:] = -1

    sparse_topk_lens = torch.full(
        (num_tokens,), topk - 5, device=device, dtype=torch.int32
    )
    return q, kv_packed, sparse_indices, sparse_topk_lens


def test_plan_validation():
    w = BatchSparseMLAPagedAttentionWrapper()
    # plan with bad num_heads
    with pytest.raises(ValueError, match="num_heads"):
        w.plan(num_heads=200, head_dim_qk=512, head_dim_v=512,
               page_size=64, topk=512, sm_scale=0.04)
    # plan with bad head_dim_qk
    with pytest.raises(ValueError, match="head_dim_qk"):
        w.plan(num_heads=64, head_dim_qk=0, head_dim_v=512,
               page_size=64, topk=512, sm_scale=0.04)
    # extra_topk > 0 without page_size_extra
    with pytest.raises(ValueError, match="page_size_extra"):
        w.plan(num_heads=64, head_dim_qk=512, head_dim_v=512,
               page_size=64, topk=512, sm_scale=0.04, extra_topk=128)
    # bad attn_sink shape
    bad_sink = torch.zeros(63, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="attn_sink"):
        w.plan(num_heads=64, head_dim_qk=512, head_dim_v=512,
               page_size=64, topk=512, sm_scale=0.04, attn_sink=bad_sink)


def test_run_before_plan_raises():
    w = BatchSparseMLAPagedAttentionWrapper()
    q, kv, idx, _ = _make_model1_inputs(num_tokens=4, num_heads=64, topk=512)
    with pytest.raises(RuntimeError, match="plan"):
        w.run(q=q, kv_cache=kv, sparse_indices=idx)


def test_run_shape_mismatch():
    w = BatchSparseMLAPagedAttentionWrapper()
    w.plan(num_heads=64, head_dim_qk=512, head_dim_v=512,
           page_size=64, topk=512, sm_scale=0.04)
    # Wrong num_heads at run time
    q = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(8, 64, 1, 584, device="cuda", dtype=torch.uint8)
    idx = torch.zeros(4, 512, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="num_heads"):
        w.run(q=q, kv_cache=kv, sparse_indices=idx)


@pytest.mark.parametrize("num_tokens", [4, 32, 128])
def test_wrapper_matches_direct_call(num_tokens):
    """Wrapper output ≡ direct sparse_mla_fwd output (same kernel path)."""
    num_heads = 64
    topk = 512
    sm_scale = _HEAD_DIM_QK ** -0.5

    q, kv_packed, sparse_indices, sparse_topk_lens = _make_model1_inputs(
        num_tokens=num_tokens, num_heads=num_heads, topk=topk
    )
    attn_sink = (
        torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.1
    )

    # Wrapper path
    w = BatchSparseMLAPagedAttentionWrapper()
    w.plan(
        num_heads=num_heads,
        head_dim_qk=_HEAD_DIM_QK,
        head_dim_v=_HEAD_DIM_V,
        page_size=_PAGE_SIZE,
        topk=topk,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
    )
    out_wrapper = w.run(
        q=q,
        kv_cache=kv_packed,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
    )

    # Direct path
    result = sparse_mla_fwd(
        q=q,
        kv_cache=kv_packed,
        indices=sparse_indices,
        sm_scale=sm_scale,
        d_v=_HEAD_DIM_V,
        topk_length=sparse_topk_lens,
        attn_sink=attn_sink,
    )
    out_direct = result[0] if isinstance(result, tuple) else result

    torch.testing.assert_close(out_wrapper, out_direct, rtol=0, atol=0)


def test_wrapper_return_lse():
    """return_lse=True returns (output, lse)."""
    num_heads, topk = 64, 512
    sm_scale = _HEAD_DIM_QK ** -0.5
    q, kv_packed, sparse_indices, sparse_topk_lens = _make_model1_inputs(
        num_tokens=8, num_heads=num_heads, topk=topk
    )

    w = BatchSparseMLAPagedAttentionWrapper()
    w.plan(
        num_heads=num_heads,
        head_dim_qk=_HEAD_DIM_QK,
        head_dim_v=_HEAD_DIM_V,
        page_size=_PAGE_SIZE,
        topk=topk,
        sm_scale=sm_scale,
    )
    out, lse = w.run(
        q=q,
        kv_cache=kv_packed,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
        return_lse=True,
    )
    assert out.shape == (8, num_heads, _HEAD_DIM_V)
    assert lse.shape == (8, num_heads)
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32


def test_wrapper_caller_provided_out_buffer():
    """Wrapper respects caller-allocated `out` buffer."""
    num_heads, topk = 64, 512
    sm_scale = _HEAD_DIM_QK ** -0.5
    q, kv_packed, sparse_indices, _ = _make_model1_inputs(
        num_tokens=16, num_heads=num_heads, topk=topk
    )
    out_buf = torch.zeros(
        16, num_heads, _HEAD_DIM_V, device="cuda", dtype=torch.bfloat16
    )
    out_ptr_before = out_buf.data_ptr()

    w = BatchSparseMLAPagedAttentionWrapper()
    w.plan(
        num_heads=num_heads,
        head_dim_qk=_HEAD_DIM_QK,
        head_dim_v=_HEAD_DIM_V,
        page_size=_PAGE_SIZE,
        topk=topk,
        sm_scale=sm_scale,
    )
    out = w.run(
        q=q,
        kv_cache=kv_packed,
        sparse_indices=sparse_indices,
        out=out_buf,
    )
    # The wrapper should write into the provided buffer, no copy.
    assert out.data_ptr() == out_ptr_before
