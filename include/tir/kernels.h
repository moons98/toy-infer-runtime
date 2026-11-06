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
void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t stream = 0);

}  // namespace tir
