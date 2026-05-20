"""FlashInfer-style plan/run wrapper for sparse-MLA paged attention on sm120.

This is the **canonical** Python API going forward. The standalone
functions in :mod:`flash_mla_sm120.interface` and :mod:`flash_mla_sm120.ops`
remain for backward compatibility but will be deprecated once downstream
callers (notably vLLM) finish migrating.

The class signature deliberately mirrors
:class:`flashinfer.mla.BatchMLAPagedAttentionWrapper` so that an eventual
upstreaming into FlashInfer is a re-export of the class name, not a
rewrite of every call site:

  flash_mla_sm120.BatchSparseMLAPagedAttentionWrapper  →
  flashinfer.BatchSparseMLAPagedAttentionWrapper

Differences from FlashInfer's existing MLA wrapper (justified by what
DSv4-Flash actually needs and what FlashInfer's MLA paths currently
lack):

1. **Sparse paged ``indices`` per query token** instead of per-request
   ``kv_indptr``/``kv_indices``. DSv4's compressor-indexer emits paged
   slot IDs per query token; the kernel reads ``indices[token, k]`` to
   look up the k-th KV slot.
2. **Variable ``topk_lens`` per token** (the SWA window length / topk
   selection length differs per token).
3. **Dual cache** (main SWA + compressed C4A/C128A) with **different
   page block sizes**. FlashInfer concatenates ckv+kpe into a single
   cache; we keep them separate because the compressed cache has
   ``main_page_block_size / compress_ratio`` page block size, not equal.
4. **attn_sink** support. FlashInfer's XQA MLA backend (the only one
   that runs on sm120) does not currently accept sinks; ours does.
5. **Mixed decode/prefill dispatch** in a single call. The wrapper
   selects ``sparse_mla_decode_v2_fwd`` for ``num_tokens <= 64`` and
   ``sparse_mla_prefill_fwd`` otherwise via the existing
   ``sparse_mla_fwd`` dispatcher.

If/when FlashInfer's XQA/trtllm-gen MLA backend grows these features,
``BatchSparseMLAPagedAttentionWrapper`` becomes a thin wrapper that
calls into FlashInfer; the public signature here is the long-term ABI.
"""

from typing import Any, Literal, Optional, Tuple, Union, overload

import torch

from .ops import sparse_mla_fwd


_DEFAULT_BACKEND = "sm120"


class BatchSparseMLAPagedAttentionWrapper:
    """Plan/run wrapper for sparse-MLA paged attention on sm120.

    Example
    -------
    >>> wrapper = BatchSparseMLAPagedAttentionWrapper()
    >>> wrapper.plan(
    ...     num_heads=64,
    ...     head_dim_qk=512,     # DSv4-Flash MODEL1: nope(448) + rope(64)
    ...     head_dim_v=512,      # output dim
    ...     page_size=64,
    ...     topk=2048,
    ...     sm_scale=1 / 24.0,
    ...     attn_sink=attn_sink_per_head,  # [num_heads]
    ... )
    >>> out = wrapper.run(
    ...     q=q,                       # [num_tokens, num_heads, 512]
    ...     kv_cache=swa_kv_paged,     # [num_blocks, page_size, h_kv, bytes]
    ...     sparse_indices=indices,    # [num_tokens, topk]
    ...     sparse_topk_lens=swa_lens, # [num_tokens]
    ... )

    Migration note
    --------------
    FlashInfer's :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper`
    parameterises with ``head_dim_ckv`` (latent / value dim, == 512 for
    DSv3 and DSv4) and ``head_dim_kpe`` (rope dim, == 64). For DSv3
    (V32) the q tensor is ``[..., head_dim_ckv + head_dim_kpe]`` = 576.

    For DSv4-Flash (MODEL1) the q stays in compressed-nope space
    (``head_dim_nope`` = 448, ``head_dim_rope`` = 64, ``head_dim_qk``
    = 512), and the output is in the 512-d latent space — they don't
    decompose into FlashInfer's ckv+kpe contract. We therefore expose
    raw ``head_dim_qk`` / ``head_dim_v`` here; if/when FlashInfer
    integration requires the ckv+kpe split, the mapping is:
      head_dim_qk = head_dim_ckv + head_dim_kpe
      head_dim_v  = head_dim_ckv   (DSv3 only — DSv4 v is not ckv)
    """

    def __init__(
        self,
        workspace_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        """
        Parameters
        ----------
        workspace_buffer : Optional[torch.Tensor]
            Reserved for API compatibility with
            :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper`.
            Today the sparse-MLA kernels allocate split-K / scheduler
            buffers per call; this argument is currently unused but the
            wrapper accepts and stashes it so a future FlashInfer-style
            backend can use it without an API break.
        backend : str
            ``"auto"`` (default) or ``"sm120"``. Only the sm120 backend
            is implemented today.
        """
        if backend not in ("auto", _DEFAULT_BACKEND):
            raise ValueError(
                f"Unsupported backend {backend!r}. "
                f"Today only {_DEFAULT_BACKEND!r} (or 'auto') is supported."
            )
        self._workspace_buffer = workspace_buffer
        self._backend = _DEFAULT_BACKEND
        # Plan state — populated by plan(), consumed by run().
        self._planned = False
        self._num_heads: Optional[int] = None
        self._head_dim_qk: Optional[int] = None
        self._head_dim_v: Optional[int] = None
        self._page_size: Optional[int] = None
        self._topk: Optional[int] = None
        self._extra_topk: int = 0
        self._page_size_extra: Optional[int] = None
        self._sm_scale: Optional[float] = None
        self._attn_sink: Optional[torch.Tensor] = None
        self._q_data_type: Optional[torch.dtype] = None
        self._kv_data_type: Optional[torch.dtype] = None

    def plan(
        self,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        page_size: int,
        topk: int,
        sm_scale: float,
        attn_sink: Optional[torch.Tensor] = None,
        extra_topk: int = 0,
        page_size_extra: Optional[int] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        """Plan the sparse-MLA attention computation.

        Parameters
        ----------
        num_heads : int
            Number of query heads. Must be in ``(0, 128]``; downstream
            ours pads to one of {16, 32, 64, 128} for kernel dispatch.
        head_dim_qk : int
            Total q head dim (nope + rope concatenated). 512 for
            DSv4-Flash MODEL1 (448 nope + 64 rope); 576 for DeepSeek
            V3.2 / GLM 5.1 V32 (512 nope + 64 rope).
        head_dim_v : int
            Output value head dim. 512 for both DSv3 and DSv4 today.
        page_size : int
            Page block size of the *main* KV cache (e.g. 64).
        topk : int
            Maximum top-k per query for the main cache. Power of two and
            a multiple of the kernel's BI tile (typically 64).
        sm_scale : float
            Softmax scale (typically ``1 / sqrt(head_dim_qk)``).
        attn_sink : Optional[torch.Tensor]
            Optional per-head learnable sink, shape ``[num_heads]``,
            ``float32``. FlashMLA V4 convention: ``output *= sigmoid(lse
            - sink)`` and ``lse' = log(exp(lse) + exp(sink))``.
        extra_topk : int
            Optional secondary-cache top-k (DSv4 C4A / C128A layers).
            ``0`` disables the dual-cache path.
        page_size_extra : Optional[int]
            Page block size of the secondary cache, if used. For DSv4
            this is ``page_size / compress_ratio``. Required when
            ``extra_topk > 0``.
        q_data_type : torch.dtype
            Query dtype. Today only ``torch.bfloat16``.
        kv_data_type : torch.dtype
            KV-cache dtype. Today only ``torch.float8_e4m3fn``
            (sm120 sparse-MLA is FP8-only).
        """
        if num_heads <= 0 or num_heads > 128:
            raise ValueError(f"num_heads must be in (0, 128], got {num_heads}")
        if head_dim_qk <= 0:
            raise ValueError(f"head_dim_qk must be > 0, got {head_dim_qk}")
        if head_dim_v <= 0:
            raise ValueError(f"head_dim_v must be > 0, got {head_dim_v}")
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if topk <= 0:
            raise ValueError(f"topk must be > 0, got {topk}")
        if extra_topk < 0:
            raise ValueError(f"extra_topk must be >= 0, got {extra_topk}")
        if extra_topk > 0 and page_size_extra is None:
            raise ValueError(
                "page_size_extra is required when extra_topk > 0"
            )
        if attn_sink is not None:
            if attn_sink.shape != (num_heads,):
                raise ValueError(
                    f"attn_sink must have shape [{num_heads}], "
                    f"got {tuple(attn_sink.shape)}"
                )
            if attn_sink.dtype != torch.float32:
                raise ValueError(
                    f"attn_sink must be float32, got {attn_sink.dtype}"
                )
            if not attn_sink.is_contiguous():
                raise ValueError("attn_sink must be contiguous")

        self._num_heads = num_heads
        self._head_dim_qk = head_dim_qk
        self._head_dim_v = head_dim_v
        self._page_size = page_size
        self._topk = topk
        self._extra_topk = extra_topk
        self._page_size_extra = page_size_extra
        self._sm_scale = sm_scale
        self._attn_sink = attn_sink
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._planned = True

    @overload
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_topk_lens: Optional[torch.Tensor] = ...,
        extra_kv_cache: Optional[torch.Tensor] = ...,
        extra_indices: Optional[torch.Tensor] = ...,
        extra_topk_lens: Optional[torch.Tensor] = ...,
        out: Optional[torch.Tensor] = ...,
        return_lse: Literal[False] = ...,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_topk_lens: Optional[torch.Tensor] = ...,
        extra_kv_cache: Optional[torch.Tensor] = ...,
        extra_indices: Optional[torch.Tensor] = ...,
        extra_topk_lens: Optional[torch.Tensor] = ...,
        out: Optional[torch.Tensor] = ...,
        return_lse: Literal[True] = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_topk_lens: Optional[torch.Tensor] = None,
        extra_kv_cache: Optional[torch.Tensor] = None,
        extra_indices: Optional[torch.Tensor] = None,
        extra_topk_lens: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run sparse-MLA paged attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape ``[num_tokens, num_heads, head_dim_qk]``,
            dtype ``q_data_type`` from :meth:`plan`.
            ``head_dim_qk == head_dim_ckv + head_dim_kpe``.
        kv_cache : torch.Tensor
            Paged main KV cache.
            Shape ``[num_blocks, page_size, head_kv=1, bytes]`` (4-D, the
            ``h_kv=1`` axis kept for shape compatibility with
            FlashMLA-style kernels). The innermost dim is byte-packed FP8.
        sparse_indices : torch.Tensor
            Paged slot IDs per query token,
            shape ``[num_tokens, topk]``, dtype int32. ``-1`` marks
            invalid / out-of-window slots (kernel skips).
        sparse_topk_lens : Optional[torch.Tensor]
            Effective top-k length per query token,
            shape ``[num_tokens]``, dtype int32. Required for SWA where
            lengths vary; for uniform top-k pass ``None``.
        extra_kv_cache : Optional[torch.Tensor]
            Secondary (compressed) KV cache. Required when
            ``extra_topk > 0`` in :meth:`plan`. Same layout as
            ``kv_cache`` but with ``page_size_extra``.
        extra_indices : Optional[torch.Tensor]
            Top-k indices into ``extra_kv_cache``,
            shape ``[num_tokens, extra_topk]``, dtype int32.
        extra_topk_lens : Optional[torch.Tensor]
            Effective extra-top-k length per query token, shape
            ``[num_tokens]``, dtype int32.
        out : Optional[torch.Tensor]
            Pre-allocated output buffer, shape
            ``[num_tokens, num_heads, head_dim_v]``, dtype bfloat16.
            If ``None``, the kernel allocates internally.
        return_lse : bool
            If True, also returns the log-sum-exp of attention logits,
            shape ``[num_tokens, num_heads]``, dtype float32. FlashMLA V4
            convention: when ``attn_sink`` is set, the returned LSE
            already includes the sink merge.

        Returns
        -------
        Either the output tensor (default) or ``(output, lse)``.

        Notes
        -----
        Routes through :func:`flash_mla_sm120.ops.sparse_mla_fwd`, which
        selects between ``sparse_mla_decode_v2_fwd`` for
        ``num_tokens <= 64`` and ``sparse_mla_prefill_fwd`` otherwise.
        Both kernels honor ``attn_sink`` and ``topk_length`` uniformly.
        """
        if not self._planned:
            raise RuntimeError(
                "Call plan() before run() — wrapper has no layer config yet."
            )

        assert self._num_heads is not None
        assert self._head_dim_qk is not None
        assert self._head_dim_v is not None
        assert self._sm_scale is not None

        # Normalize q to 3-D: accept ``[num_tokens, num_heads, d_qk]`` (3-D)
        # or ``[batch, s_q, num_heads, d_qk]`` (FlashMLA-style 4-D). vLLM's
        # decode path passes 4-D with ``s_q=1``; the prefill path passes
        # 3-D directly.
        q_2dim = q if q.dim() == 3 else q.reshape(-1, q.size(-2), q.size(-1))
        num_tokens, num_heads, d_qk = q_2dim.shape
        if num_heads != self._num_heads:
            raise ValueError(
                f"q num_heads={num_heads} != plan's num_heads={self._num_heads}"
            )
        if d_qk != self._head_dim_qk:
            raise ValueError(
                f"q head_dim={d_qk} != plan's head_dim_qk={self._head_dim_qk}"
            )

        # Same flexibility for sparse_indices: accept ``[N, topk]`` (2-D)
        # or ``[N, 1, topk]`` (3-D, FlashMLA-style with s_q=1 axis).
        sparse_indices_2d = (
            sparse_indices
            if sparse_indices.dim() == 2
            else sparse_indices.squeeze(1)
        )
        if sparse_indices_2d.shape != (num_tokens, self._topk):
            raise ValueError(
                f"sparse_indices shape {tuple(sparse_indices.shape)} != "
                f"({num_tokens}, {self._topk}) (or 3-D equivalent)"
            )
        if self._extra_topk > 0:
            if extra_kv_cache is None or extra_indices is None:
                raise ValueError(
                    "plan() set extra_topk > 0 but run() got "
                    "extra_kv_cache=None or extra_indices=None"
                )
            extra_indices_2d = (
                extra_indices
                if extra_indices.dim() == 2
                else extra_indices.squeeze(1)
            )
            if extra_indices_2d.shape != (num_tokens, self._extra_topk):
                raise ValueError(
                    f"extra_indices shape {tuple(extra_indices.shape)} != "
                    f"({num_tokens}, {self._extra_topk}) (or 3-D equivalent)"
                )
        else:
            extra_indices_2d = None

        # Same flexibility for ``out``: caller may hand us 3-D or 4-D.
        # Since the underlying ops write into the storage we forward
        # ``out`` to, the dimensionality only matters for shape checks
        # done by the op layer (which expects 3-D), so squeeze a
        # singleton s_q axis if present.
        out_2dim = out
        if out is not None and out.dim() == 4 and out.size(1) == 1:
            out_2dim = out.squeeze(1)

        result = sparse_mla_fwd(
            q=q_2dim,
            kv_cache=kv_cache,
            indices=sparse_indices_2d,
            sm_scale=self._sm_scale,
            d_v=self._head_dim_v,
            out=out_2dim,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices_2d,
            topk_length=sparse_topk_lens,
            extra_topk_length=extra_topk_lens,
            attn_sink=self._attn_sink,
        )

        if isinstance(result, tuple):
            output, lse = result
        else:
            output, lse = result, None

        if return_lse:
            if lse is None:
                # sparse_mla_fwd's decode path returns (output, lse); the
                # prefill path also returns a tuple. If a future kernel
                # variant returns just output, we'd need a shape here —
                # fail loudly rather than silently fabricate.
                raise RuntimeError(
                    "return_lse=True but the underlying kernel did not "
                    "produce an LSE tensor."
                )
            return output, lse
        return output
