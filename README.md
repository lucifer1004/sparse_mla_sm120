# sparse_mla_sm120

CUDA kernel library for **DeepSeek-style sparse attention**, providing:

- **Sparse MLA**: **Prefill** and **decode** forwards with FP8-packed KV and top‑k indices (`sparse_mla_prefill_fwd`, `sparse_mla_decode_fwd`, etc.; see `sparse_mla_sm120/ops.py`).

## Python API

The canonical entrypoint is the plan/run wrapper
`flash_mla_sm120.BatchSparseMLAPagedAttentionWrapper`, modeled on
[`flashinfer.mla.BatchMLAPagedAttentionWrapper`](https://docs.flashinfer.ai/api/mla.html)
so that the API survives an eventual upstreaming into FlashInfer
without call-site changes:

```python
from flash_mla_sm120 import BatchSparseMLAPagedAttentionWrapper

wrapper = BatchSparseMLAPagedAttentionWrapper()
wrapper.plan(
    num_heads=64,
    head_dim_qk=512,    # DSv4-Flash MODEL1: nope(448) + rope(64)
    head_dim_v=512,
    page_size=64,
    topk=2048,
    sm_scale=1 / 24.0,
    attn_sink=attn_sink_per_head,  # optional [num_heads] float32
)
out = wrapper.run(
    q=q,                       # [num_tokens, num_heads, 512]
    kv_cache=swa_kv_paged,     # paged FP8 KV
    sparse_indices=indices,    # [num_tokens, topk]
    sparse_topk_lens=swa_lens, # [num_tokens]
)
```

The standalone functions (`flash_mla_sparse_fwd`,
`flash_mla_with_kvcache`, `sparse_mla_decode_fwd`,
`sparse_mla_prefill_fwd`) remain exported for backward compatibility
but are scheduled for deprecation once vLLM and other downstream
callers migrate.

## Target hardware

- **Architecture**: NVIDIA **compute capability 12.x (SM120 family)**. The build emits both **`sm_120a`** and **`sm_120f`** (see `-gencode` flags in `setup.py`).

## Requirements

- **Python** ≥ 3.10 (matches `python_requires` in `setup.py`)
- **PyTorch with CUDA** (the extension is built via `torch.utils.cpp_extension`)
- **CUDA Toolkit / nvcc** compatible with your GPU and able to compile for SM120

## Installation

From the repository root:

```bash
python3 setup.py bdist_wheel
```

The wheel is written under `dist/`. Install it with:

```bash
pip install dist/sparse_mla_sm120-*.whl
```

For development, an editable install is also fine:

```bash
pip install -e .
```

## Benchmarks

Run from the repository root (after installing the package or adding the root to `PYTHONPATH`).

| Script | Purpose |
|--------|---------|
| `benchmarks/benchmark_sparse_mla.py` | Sparse MLA **prefill / decode**: latency, effective bandwidth, TFLOP/s (DeepSeek V3.2–scale settings) |
| `scripts/bench.py` | Simple **prefill** benchmark (uses helpers under `tests/`; run from repo root) |

Examples:

```bash
python benchmarks/benchmark_sparse_mla.py
```

## Tests

With **pytest** installed:

```bash
pytest tests/test_sparse_mla.py -v -s
```

Tests compare CUDA outputs to PyTorch references; tolerances account for FP8 quantization.
