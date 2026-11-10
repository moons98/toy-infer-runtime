#pragma once

// CPU implementations that compute the same thing as the GPU kernels.
// They are written as the simplest possible loops so we can trust them to be correct.
// Speed does not matter here.

namespace tir::ref {

void vector_add(const float* a, const float* b, float* c, int n);

// C[M, N] = A[M, K] * B[K, N]   (row-major)
void gemm(const float* A, const float* B, float* C, int M, int N, int K);

// x [N, C, H, W], w [K, C, R, S] -> y [N, K, P, Q]
void conv2d(const float* x, const float* w, float* y, int N, int C, int H, int W, int K, int R,
            int S, int pad, int stride);

void relu(const float* x, float* y, int n);

}  // namespace tir::ref
