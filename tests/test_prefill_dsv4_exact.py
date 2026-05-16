"""Exact repro of vllm DSv4 prefill IMA: MODEL1 + h=64 + topk=128 + non-contig kv."""
import pytest
import torch
import flash_mla_sm120


@pytest.mark.parametrize("padded_block", [False, True])
def test_dsv4_exact_shape(padded_block):
    """Match vllm's call:
      q=(8192, 64, 512) bf16
      kv shape=(num_blocks, 64, 1, 584) uint8 — non-contig when padded_block=True
      idx=(8192, 128) int32
      topk_length=(8192,) int32 in [1, 128]
      attn_sink=(64,) float32
      out=(8192, 64, 512) bf16
    """
    torch.manual_seed(42)
    num_tokens = 8192
    num_heads = 64
    d_qk, d_v, topk = 512, 512, 128
    block_size = 64
    sm_scale = d_qk ** -0.5
    num_blocks = 9558

    # vllm allocates kv_cache as (num_blocks, block_size, head_bytes_padded)
    # and unsqueezes a h_kv axis. With padded_block=True we mimic the 64-byte
    # block stride padding observed in the dump (stride_0 = 37440, but the
    # data layout is block_size * 584 = 37376 — extra 64 bytes/block).
    bytes_per_token = d_qk + 8 * 9  # MODEL1 footer: D_NOPE(512) + D_ROPE*2(64) + scale(8) -> roughly 584
    # Match observed kv.shape[-1] = 584 directly:
    bytes_per_token = 584
    if padded_block:
        # Mimic vllm's observed stride: stride(0) = 37440 = 64 × 585, while
        # actual bytes_per_token = 584. Encode this by allocating an outer
        # tensor (num_blocks, block_size, bytes_per_token+1) and taking a
        # slice along last dim, so block stride becomes 64 × 585 and per-row
        # stride remains 585 (>1 byte padding per row).
        outer = torch.zeros(num_blocks, block_size, bytes_per_token + 1,
                            device="cuda", dtype=torch.uint8)
        kv_3d = outer[..., :bytes_per_token]  # (num_blocks, block_size, bytes_per_token), non-contig
        kv = kv_3d.unsqueeze(-2)  # (num_blocks, block_size, 1, bytes_per_token)
    else:
        kv = torch.randint(0, 256, (num_blocks, block_size, 1, bytes_per_token),
                           device="cuda", dtype=torch.uint8)
    print(f"\n  kv shape={tuple(kv.shape)} stride={kv.stride()} contig={kv.is_contiguous()}")

    q = (torch.randn(num_tokens, num_heads, d_qk,
                     device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    s_kv = num_blocks * block_size
    indices = torch.randint(0, s_kv, (num_tokens, topk),
                            device="cuda", dtype=torch.int32)
    indices[:, -3:] = -1
    topk_length = torch.full((num_tokens,), topk, device="cuda", dtype=torch.int32)
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32)

    out = torch.empty((num_tokens, num_heads, d_v), dtype=torch.bfloat16, device="cuda")
    result = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, kv, indices, sm_scale, d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )
    torch.cuda.synchronize()
    print(f"  OK padded_block={padded_block}")
