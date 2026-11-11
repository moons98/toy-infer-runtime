#pragma once

// Launch functions for the CUDA kernels in src/kernels/.
//
// This header is plain C++, so .cpp files can include it. The kernels themselves
// (__global__ xxx_kernel) stay inside their .cu files; callers only see launch_xxx,
// which picks the grid/block size and launches the kernel with <<<grid, block>>>.
//
// Rules:
//   - Kernels take only raw pointers and sizes. They never allocate or copy memory.
//   - launch_xxx is asynchronous: it returns before the kernel finishes.

#include <cuda_runtime.h>

namespace tir {

// c[i] = a[i] + b[i]
void launch_vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0);

// C[M, N] = A[M, K] * B[K, N]   (all row-major)
// naive: each thread computes one element of C.
void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream = 0);

// 2D convolution (NCHW layout, no bias)
//   x: input   [N, C, H, W]
//   w: weight  [K, C, R, S]   (K output channels, each a C x R x S filter)
//   y: output  [N, K, P, Q]   (P and Q come from conv2d_out_size)
// naive: each thread computes one output element.
void launch_conv2d_naive(const float* x, const float* w, float* y, int N, int C, int H, int W, int K, int R, int S,
                         int pad, int stride, cudaStream_t stream = 0);

// Output size for input size `in` and filter size `filter`
inline int conv2d_out_size(int in, int filter, int pad, int stride) {
    return (in + 2 * pad - filter) / stride + 1;
}

// y[i] = max(x[i], 0)
void launch_relu(const float* x, float* y, int n, cudaStream_t stream = 0);

// Batch normalization for inference, per channel c along dimension 1 of x:
//   y = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
//   x, y: [N, C, H, W] (or [N, C]), n elements in total
//   inner: elements per channel slice (H * W, or 1 for [N, C])
void launch_batchnorm(const float* x, const float* gamma, const float* beta, const float* mean, const float* var,
                      float* y, int n, int channels, int inner, float eps, cudaStream_t stream = 0);

}  // namespace tir
