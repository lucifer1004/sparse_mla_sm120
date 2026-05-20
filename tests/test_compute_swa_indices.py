"""Correctness check: flash_mla_sm120 CUDA compute_swa_indices_and_lens
matches the Triton _compute_swa_indices_and_lens_kernel bit-for-bit.

The CUDA port exists so the SWA-indices kernel does not JIT-compile during
inference (suspected to participate in the intermittent c=4 n=16 IMA on
sm120). It must reproduce the Triton kernel's output exactly to be a
drop-in replacement.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )


def _make_fixture(
    num_reqs: int,
    query_lens: list[int],
    seq_lens: list[int],
    window_size: int,
    block_size: int,
    invalid_token_positions: list[int] | None = None,
    device: str = "cuda",
):
    assert len(query_lens) == num_reqs == len(seq_lens)
    num_tokens = sum(query_lens)
    max_blocks = max(s for s in seq_lens) // block_size + 4

    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32, device=device).cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    # Random block IDs per req — they just need to be deterministic.
    g = torch.Generator(device=device).manual_seed(42)
    block_table = torch.randint(
        100, 10_000, (num_reqs, max_blocks), dtype=torch.int32,
        device=device, generator=g,
    )

    t_to_r = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.tensor(query_lens, dtype=torch.int32, device=device),
    )

    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    for p in invalid_token_positions or []:
        is_valid[p] = False

    return qsl, seq_lens_t, t_to_r, is_valid, block_table


def _run_triton(qsl, seq_lens, t_to_r, is_valid, block_table,
                 window_size, block_size, num_tokens, token_offset):
    swa_indices = torch.full(
        (num_tokens, window_size), -42, dtype=torch.int32, device=qsl.device)
    swa_lens = torch.full(
        (num_tokens,), -42, dtype=torch.int32, device=qsl.device)
    _compute_swa_indices_and_lens_kernel[(num_tokens,)](
        swa_indices, swa_indices.stride(0), swa_lens, window_size,
        qsl, seq_lens, t_to_r, is_valid, block_table, block_table.stride(0),
        block_size, token_offset=token_offset, TRITON_BLOCK_SIZE=1024,
    )
    return swa_indices, swa_lens


def _run_cuda(qsl, seq_lens, t_to_r, is_valid, block_table,
              window_size, block_size, num_tokens, token_offset):
    import flash_mla_sm120.cuda as C
    swa_indices = torch.full(
        (num_tokens, window_size), -42, dtype=torch.int32, device=qsl.device)
    swa_lens = torch.full(
        (num_tokens,), -42, dtype=torch.int32, device=qsl.device)
    C.compute_swa_indices_and_lens(
        swa_indices=swa_indices, swa_lens=swa_lens,
        window_size=window_size,
        query_start_loc=qsl, seq_lens=seq_lens,
        token_to_req_indices=t_to_r, is_valid_token=is_valid,
        block_table=block_table,
        block_size=block_size,
        token_offset=token_offset,
        num_tokens=num_tokens,
    )
    return swa_indices, swa_lens


@pytest.mark.parametrize("case", [
    # decode-shape: 1 req, 1 token, long history
    dict(qlens=[1], slens=[2048], window=1024, bsz=256, t_off=0,
         invalid=None),
    # decode-shape: 8 reqs, 1 token each, various histories
    dict(qlens=[1]*8, slens=[1024, 2048, 5000, 256, 1, 8192, 4097, 33],
         window=1024, bsz=256, t_off=0, invalid=[3]),
    # prefill-shape: 4 reqs, prefix chunked, window smaller than block
    dict(qlens=[100, 50, 1024, 200], slens=[200, 50, 1024, 4096],
         window=512, bsz=256, t_off=0, invalid=None),
    # prefill with token_offset (simulating prefill_swa call after decode)
    dict(qlens=[3, 4097], slens=[10, 4097], window=1024, bsz=256, t_off=0,
         invalid=[0]),
    # large window, small block_size
    dict(qlens=[1, 1, 1], slens=[8192, 8192, 8192], window=2048,
         bsz=64, t_off=0, invalid=None),
])
def test_swa_indices_matches_triton(case):
    qsl, slens, t_to_r, is_valid, block_table = _make_fixture(
        num_reqs=len(case["qlens"]),
        query_lens=case["qlens"], seq_lens=case["slens"],
        window_size=case["window"], block_size=case["bsz"],
        invalid_token_positions=case["invalid"],
    )
    num_tokens = sum(case["qlens"])
    t_off = case["t_off"]

    ti_idx, ti_lens = _run_triton(
        qsl, slens, t_to_r, is_valid, block_table,
        case["window"], case["bsz"], num_tokens, t_off)
    cu_idx, cu_lens = _run_cuda(
        qsl, slens, t_to_r, is_valid, block_table,
        case["window"], case["bsz"], num_tokens, t_off)

    torch.testing.assert_close(cu_lens, ti_lens, rtol=0, atol=0)
    torch.testing.assert_close(cu_idx, ti_idx, rtol=0, atol=0)


def test_swa_indices_3d_buffer():
    """Mimic the prefill buffer layout: [max_tokens, 1, window_size]."""
    qsl, slens, t_to_r, is_valid, block_table = _make_fixture(
        num_reqs=3,
        query_lens=[64, 128, 256], seq_lens=[300, 500, 1000],
        window_size=1024, block_size=256,
    )
    num_tokens = 64 + 128 + 256
    max_tokens = num_tokens + 100  # over-allocated, like the real buffer

    # Triton: feed it the 3-D view, it uses stride(0) for row stride.
    ti_idx_full = torch.full(
        (max_tokens, 1, 1024), -42, dtype=torch.int32, device=qsl.device)
    ti_lens_full = torch.full(
        (max_tokens,), -42, dtype=torch.int32, device=qsl.device)
    ti_idx = ti_idx_full[:num_tokens]
    ti_lens = ti_lens_full[:num_tokens]
    _compute_swa_indices_and_lens_kernel[(num_tokens,)](
        ti_idx, ti_idx.stride(0), ti_lens, 1024,
        qsl, slens, t_to_r, is_valid, block_table, block_table.stride(0),
        256, token_offset=0, TRITON_BLOCK_SIZE=1024,
    )

    cu_idx_full = torch.full(
        (max_tokens, 1, 1024), -42, dtype=torch.int32, device=qsl.device)
    cu_lens_full = torch.full(
        (max_tokens,), -42, dtype=torch.int32, device=qsl.device)
    import flash_mla_sm120.cuda as C
    C.compute_swa_indices_and_lens(
        swa_indices=cu_idx_full[:num_tokens],
        swa_lens=cu_lens_full[:num_tokens],
        window_size=1024,
        query_start_loc=qsl, seq_lens=slens,
        token_to_req_indices=t_to_r, is_valid_token=is_valid,
        block_table=block_table,
        block_size=256, token_offset=0, num_tokens=num_tokens,
    )
    cu_idx = cu_idx_full[:num_tokens]
    cu_lens = cu_lens_full[:num_tokens]

    torch.testing.assert_close(cu_lens, ti_lens, rtol=0, atol=0)
    torch.testing.assert_close(cu_idx, ti_idx, rtol=0, atol=0)


def test_swa_indices_token_offset():
    """Mimic the prefill call: token_offset = num_decode_tokens, output
    written at index 0."""
    num_decode = 4
    decode_lens = [1, 1, 1, 1]
    prefill_lens = [3, 4]
    seq_lens = [256, 512, 1024, 2048, 50, 200]
    qsl, slens, t_to_r, is_valid, block_table = _make_fixture(
        num_reqs=6,
        query_lens=decode_lens + prefill_lens, seq_lens=seq_lens,
        window_size=1024, block_size=256,
    )
    num_prefill_tokens = sum(prefill_lens)

    ti_idx, ti_lens = _run_triton(
        qsl, slens, t_to_r, is_valid, block_table,
        window_size=1024, block_size=256,
        num_tokens=num_prefill_tokens, token_offset=num_decode)
    cu_idx, cu_lens = _run_cuda(
        qsl, slens, t_to_r, is_valid, block_table,
        window_size=1024, block_size=256,
        num_tokens=num_prefill_tokens, token_offset=num_decode)

    torch.testing.assert_close(cu_lens, ti_lens, rtol=0, atol=0)
    torch.testing.assert_close(cu_idx, ti_idx, rtol=0, atol=0)
