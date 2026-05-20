// Compute SWA (sliding-window-attention) paged slot IDs + window lengths.
//
// Port of vLLM's _compute_swa_indices_and_lens_kernel (Triton) — implemented
// as a plain CUDA kernel so it never JIT-compiles during inference. The
// inference-time JIT was suspected to participate in the intermittent c=4
// n=16 IMA on RTX PRO 6000 Blackwell (sm120).
//
// One block per output row. Threads in the block cooperate on the
// per-window-position writes (one int32 per position). `swa_lens[pid]` is
// written by thread 0.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int SWA_INDICES_THREADS = 128;

__global__ void compute_swa_indices_and_lens_kernel(
    int32_t* __restrict__ swa_indices,
    int swa_indices_stride,
    int32_t* __restrict__ swa_lens,
    int window_size,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ token_to_req_indices,
    const bool* __restrict__ is_valid_token,
    const int32_t* __restrict__ block_table,
    int block_table_stride,
    int block_size,
    int token_offset)
{
    const int pid = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_idx = pid + token_offset;

    // Per-block scalars — compute once on lane 0, broadcast via shared mem.
    __shared__ int s_swa_len;
    __shared__ int s_start_pos;
    __shared__ int s_req_idx;
    __shared__ int s_end_pos;
    __shared__ bool s_is_valid;

    if (tid == 0) {
        s_is_valid = is_valid_token[token_idx];
        if (!s_is_valid) {
            swa_lens[pid] = 0;
            s_swa_len = 0;
            s_start_pos = 0;
            s_req_idx = 0;
            s_end_pos = 0;
        } else {
            const int req_idx = token_to_req_indices[token_idx];
            const int query_start = query_start_loc[req_idx];
            const int query_end = query_start_loc[req_idx + 1];
            const int query_len = query_end - query_start;
            const int seq_len = seq_lens[req_idx];
            const int prefix_len = seq_len - query_len;

            const int pos = prefix_len + token_idx - query_start;
            const int start_pos = max(pos - window_size + 1, 0);
            const int end_pos = pos + 1;
            const int swa_len = end_pos - start_pos;

            swa_lens[pid] = swa_len;
            s_req_idx = req_idx;
            s_start_pos = start_pos;
            s_end_pos = end_pos;
            s_swa_len = swa_len;
        }
    }
    __syncthreads();

    if (!s_is_valid) return;

    const int req_idx = s_req_idx;
    const int start_pos = s_start_pos;
    const int end_pos = s_end_pos;
    const int swa_len = s_swa_len;

    int32_t* out_row = swa_indices + (size_t)pid * swa_indices_stride;
    const int32_t* block_row =
        block_table + (size_t)req_idx * block_table_stride;

    // Each thread strides through the window writing one int32 per step.
    for (int offset = tid; offset < window_size; offset += SWA_INDICES_THREADS) {
        const int pos_offset = start_pos + offset;
        int32_t slot;
        if (offset < swa_len) {
            // pos_offset < end_pos is implied by offset < swa_len.
            const int block_idx = pos_offset / block_size;
            const int block_off = pos_offset % block_size;
            const int32_t block_number = block_row[block_idx];
            slot = block_number * block_size + block_off;
        } else {
            slot = -1;
        }
        out_row[offset] = slot;
    }
}

}  // namespace

void compute_swa_indices_and_lens(
    torch::Tensor swa_indices,
    torch::Tensor swa_lens,
    int window_size,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor token_to_req_indices,
    torch::Tensor is_valid_token,
    torch::Tensor block_table,
    int block_size,
    int token_offset,
    int num_tokens)
{
    TORCH_CHECK(swa_indices.is_cuda(), "swa_indices must be CUDA");
    TORCH_CHECK(swa_indices.dtype() == torch::kInt32);
    // Accept both [N, W] (decode buffer) and [N, 1, W] (prefill buffer).
    // For 3-D the singleton at dim 1 contributes stride*1 to the row-stride,
    // so swa_indices.stride(0) is still the correct row-to-row int32 stride.
    TORCH_CHECK(swa_indices.dim() == 2
            || (swa_indices.dim() == 3 && swa_indices.size(1) == 1),
        "swa_indices must be [N, W] or [N, 1, W]");
    TORCH_CHECK(swa_indices.stride(-1) == 1,
        "swa_indices innermost stride must be 1");
    TORCH_CHECK(swa_indices.size(-1) >= window_size,
        "swa_indices last dim must be >= window_size");

    TORCH_CHECK(swa_lens.dtype() == torch::kInt32 && swa_lens.is_contiguous());

    TORCH_CHECK(query_start_loc.dtype() == torch::kInt32);
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(token_to_req_indices.dtype() == torch::kInt32);
    TORCH_CHECK(is_valid_token.dtype() == torch::kBool);
    TORCH_CHECK(block_table.dtype() == torch::kInt32);
    TORCH_CHECK(block_table.dim() == 2);
    TORCH_CHECK(block_table.stride(-1) == 1,
        "block_table innermost stride must be 1");

    TORCH_CHECK(num_tokens >= 0);
    if (num_tokens == 0) return;

    TORCH_CHECK(window_size > 0, "window_size must be > 0");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(token_offset >= 0, "token_offset must be >= 0");

    const int swa_indices_stride = (int)swa_indices.stride(0);
    const int block_table_stride = (int)block_table.stride(0);

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(swa_indices.get_device()).stream();

    compute_swa_indices_and_lens_kernel<<<num_tokens, SWA_INDICES_THREADS, 0, stream>>>(
        swa_indices.data_ptr<int32_t>(),
        swa_indices_stride,
        swa_lens.data_ptr<int32_t>(),
        window_size,
        query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        token_to_req_indices.data_ptr<int32_t>(),
        is_valid_token.data_ptr<bool>(),
        block_table.data_ptr<int32_t>(),
        block_table_stride,
        block_size,
        token_offset);
}
