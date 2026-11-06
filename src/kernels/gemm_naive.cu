#include "tir/cuda_check.h"
#include "tir/kernels.h"

namespace tir {

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 2D grid: x maps to columns, y maps to rows.
    // Threads in a warp have consecutive threadIdx.x, so they handle neighboring columns.
    // Reads of B[k * N + col] and writes of C[row * N + col] then hit consecutive addresses
    // (coalescing).
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // Dot product of one row of A and one column of B.
            // Slow because every value is read from global memory each time (improved in week 3).
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K,
                       cudaStream_t stream) {
    constexpr int kBlock = 16;
    dim3 block(kBlock, kBlock);  // 16 x 16 = 256 threads per block
    dim3 grid((N + kBlock - 1) / kBlock, (M + kBlock - 1) / kBlock);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    TIR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tir
