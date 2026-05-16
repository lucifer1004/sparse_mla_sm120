"""Standalone replay of dumped flash_mla_sparse_fwd inputs from vllm."""
import sys
import torch
import flash_mla_sm120


def main(path):
    print(f"Loading {path}...")
    d = torch.load(path, map_location="cuda", weights_only=False)
    q = d["q"].cuda()
    kv = d["kv"].cuda()
    indices = d["indices"].cuda()
    sm_scale = d["sm_scale"]
    d_v = d["d_v"]
    attn_sink = d["attn_sink"].cuda() if d["attn_sink"] is not None else None
    topk_length = d["topk_length"].cuda() if d["topk_length"] is not None else None
    extra_k_cache = d["extra_k_cache"].cuda() if d["extra_k_cache"] is not None else None
    extra_idx = d["extra_idx"].cuda() if d["extra_idx"] is not None else None
    extra_topk_length = d["extra_topk_length"].cuda() if d["extra_topk_length"] is not None else None
    out_shape = d["out_shape"]

    print(f"  q={tuple(q.shape)} {q.dtype}")
    print(f"  kv={tuple(kv.shape)} {kv.dtype}")
    print(f"  indices={tuple(indices.shape)} {indices.dtype} "
          f"min={indices.min().item()} max={indices.max().item()}")
    print(f"  sm_scale={sm_scale} d_v={d_v}")
    print(f"  attn_sink={None if attn_sink is None else tuple(attn_sink.shape)}")
    print(f"  topk_length={None if topk_length is None else (topk_length.min().item(), topk_length.max().item(), topk_length.shape)}")
    print(f"  extra_k_cache={None if extra_k_cache is None else tuple(extra_k_cache.shape)}")
    print(f"  extra_idx={None if extra_idx is None else tuple(extra_idx.shape)}")
    print(f"  out_shape={out_shape}")

    print("\nCalling sparse_mla_prefill_fwd...")
    out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda") if out_shape else None
    result = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, kv, indices, sm_scale, d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
        extra_kv_cache=extra_k_cache,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
    )
    torch.cuda.synchronize()
    print(f"  result returned: {type(result)}")
    if isinstance(result, tuple):
        for i, r in enumerate(result):
            if r is not None:
                print(f"    [{i}] {tuple(r.shape)} {r.dtype}")
    print("OK")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sparse_mla_dump.pt"
    main(path)
