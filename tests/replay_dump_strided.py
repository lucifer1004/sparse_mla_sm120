"""Replay dumped vllm inputs, reconstructing exact shape/stride from snapshot."""
import sys
import torch
import flash_mla_sm120


def _restore(snap):
    """Rebuild tensor preserving shape AND stride (handles non-contig views)."""
    if snap is None:
        return None
    flat = snap["data"].cuda()
    # Allocate destination with the correct total storage and stride; then
    # copy data into the correct elements. We do this by computing the
    # underlying storage size and creating a tensor with as_strided that
    # spans the original (shape, stride).
    shape = snap["shape"]
    stride = snap["stride"]
    storage_offset = snap["storage_offset"]
    # Required storage size = sum over dims of (size - 1) * stride + 1 + offset
    # (this gives the index of the last addressable element + 1).
    last = 0
    for s, st in zip(shape, stride):
        if s > 0:
            last += (s - 1) * st
    needed = last + 1 + storage_offset
    # If snapshot is contig (storage == prod(shape)), this matches numel.
    # If non-contig (padded), we need a larger storage.
    if flat.numel() < needed:
        bigger = torch.zeros(needed, dtype=flat.dtype, device=flat.device)
        # We can't recover the original padding bytes; fill data positions
        # using as_strided write — but we only have flat shape data, which
        # came from .contiguous().view(-1). So we need to write data into
        # the strided view positions.
        # Simpler: create an as_strided view that mirrors original shape,
        # then ASSIGN flat reshaped to original shape's contiguous version.
        view = bigger.as_strided(shape, stride, storage_offset)
        # data is contiguous flatten of original; reshape and copy.
        contig_view = flat.view(shape)
        view.copy_(contig_view)
        return view
    else:
        return flat.as_strided(shape, stride, storage_offset)


def main(path):
    print(f"Loading {path}...")
    d = torch.load(path, map_location="cpu", weights_only=False)
    q = _restore(d["q"])
    kv = _restore(d["kv"])
    indices = _restore(d["indices"])
    sm_scale = d["sm_scale"]
    d_v = d["d_v"]
    attn_sink = _restore(d.get("attn_sink"))
    topk_length = _restore(d.get("topk_length"))
    extra_k_cache = _restore(d.get("extra_k_cache"))
    extra_idx = _restore(d.get("extra_idx"))
    extra_topk_length = _restore(d.get("extra_topk_length"))
    out_shape = d["out_shape"]

    for name, t in [("q", q), ("kv", kv), ("indices", indices),
                    ("attn_sink", attn_sink), ("topk_length", topk_length)]:
        if t is None:
            print(f"  {name}=None")
        else:
            print(f"  {name} shape={tuple(t.shape)} stride={t.stride()} "
                  f"{t.dtype} contig={t.is_contiguous()}")
    if indices is not None:
        print(f"  indices min={indices.min().item()} max={indices.max().item()}")
    if topk_length is not None:
        print(f"  topk_length min={topk_length.min().item()} max={topk_length.max().item()}")

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
    print(f"  Done. result type={type(result)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sparse_mla_dump.pt"
    main(path)
