#include "tir/cuda_check.h"
#include "tir/kernels.h"

namespace tir {

// Elementwise with a per-channel lookup, one thread per element.
// In inference, mean and var are fixed numbers learned during training, not computed from x.
__global__ void batchnorm_kernel(const float* x, const float* gamma, const float* beta,
                                 const float* mean, const float* var, float* y, int n, int channels,
                                 int inner, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Channel of flat index i: `inner` elements per channel, wrapping every `channels` steps
        int c = (i / inner) % channels;
        y[i] = (x[i] - mean[c]) * rsqrtf(var[c] + eps) * gamma[c] + beta[c];
    }
}

void launch_batchnorm(const float* x, const float* gamma, const float* beta, const float* mean,
                      const float* var, float* y, int n, int channels, int inner, float eps,
                      cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;
    batchnorm_kernel<<<grid_size, kBlockSize, 0, stream>>>(x, gamma, beta, mean, var, y, n,
                                                           channels, inner, eps);
    TIR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tir
