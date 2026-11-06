#include "tir/cuda_check.h"
#include "tir/kernels.h"

namespace tir {

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // Global index of the element this thread handles.
    // Each block has blockDim.x threads, and this is block number blockIdx.x.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The total thread count is a multiple of the block size, so it can exceed n.
    // Keep the extra threads from touching memory past the end of the array.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;  // ceiling division
    vector_add_kernel<<<grid_size, kBlockSize, 0, stream>>>(a, b, c, n);
    TIR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tir
