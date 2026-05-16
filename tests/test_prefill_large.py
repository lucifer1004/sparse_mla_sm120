"""Repro of vllm DSv4 prefill IMA: V32 model, h=64, large num_tokens."""
import pytest
import torch
import flash_mla_sm120
from test_decode import quantize_kv_v32, dequantize_kv_v32, ref_sparse_attn_decode


@pytest.mark.parametrize("num_tokens", [256, 512, 1024, 2048, 4096, 8192])
def test_v32_h64_large_prefill(num_tokens):
    """Larger num_tokens to repro DSv4 chunked prefill shape (TP=1 == 64 heads)."""
    torch.manual_seed(42)
    d_qk, d_v, topk = 576, 512, 2048
    num_heads = 64
    block_size = 64
    sm_scale = d_qk ** -0.5
    # Make KV cache large enough so indices can legitimately reach far.
    num_blocks = max(64, (num_tokens + topk) // block_size + 64)
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                           device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_v32(kv_bf16)

    q = (torch.randn(num_tokens, 1, num_heads, d_qk,
                     device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, 1, topk),
                            device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    q_flat = q.view(-1, num_heads, d_qk)
    idx_flat = indices.view(-1, topk)

    out, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q_flat, kv_packed, idx_flat, sm_scale, d_v)
    torch.cuda.synchronize()
    print(f"\n  V32 h=64 tokens={num_tokens}: out={out.shape}")
