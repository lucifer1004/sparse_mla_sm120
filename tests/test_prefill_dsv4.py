"""Repro DSv4 prefill IMA: V32, h=64, num_tokens=8192, page_block_size=256.

DSv4-Flash inference config (matches vllm `just run` with --block-size 256):
  d_qk=576 (V32), num_heads=64, topk=2048, page_block_size=256, attn_sink present,
  topk_length-aware masking, dual-cache prefill (for non-swa-only layers).
"""
import pytest
import torch
import flash_mla_sm120
from test_decode import quantize_kv_v32, dequantize_kv_v32


def _build(num_tokens, num_heads, d_qk, topk, block_size, num_blocks, seed=42):
    torch.manual_seed(seed)
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                           device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_v32(kv_bf16)

    q = (torch.randn(num_tokens, num_heads, d_qk,
                     device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, topk),
                            device="cuda", dtype=torch.int32)
    indices[:, -10:] = -1
    return q, kv_packed, indices


@pytest.mark.parametrize("block_size", [64, 128, 256])
@pytest.mark.parametrize("num_tokens", [128, 1024, 8192])
def test_v32_h64_block_size(block_size, num_tokens):
    d_qk, d_v, topk, num_heads = 576, 512, 2048, 64
    sm_scale = d_qk ** -0.5
    # KV blocks scaled so total tokens > topk.
    num_blocks = max(64, (topk * 2) // block_size + 64)

    q, kv_packed, indices = _build(num_tokens, num_heads, d_qk, topk,
                                   block_size, num_blocks)

    out, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, kv_packed, indices, sm_scale, d_v)
    torch.cuda.synchronize()
    print(f"\n  V32 h=64 tokens={num_tokens} block_size={block_size}: out={out.shape}")
