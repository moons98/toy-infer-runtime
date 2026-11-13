#include "tir/cuda_check.h"
#include "tir/kernels.h"

namespace tir {

// Elementwise, like vector_add: one thread per element. The shape does not matter, only the
// total number of elements.
__global__ void relu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fmaxf(x[i], 0.0f);
    }
}

void launch_relu(const float* x, float* y, int n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;
    relu_kernel<<<grid_size, kBlockSize, 0, stream>>>(x, y, n);
    TIR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tir
