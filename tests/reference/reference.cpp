#include "reference/reference.h"

#include <cmath>

namespace tir::ref {

void vector_add(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

void conv2d(const float* x, const float* w, float* y, int N, int C, int H, int W, int K, int R,
            int S, int pad, int stride) {
    int P = (H + 2 * pad - R) / stride + 1;
    int Q = (W + 2 * pad - S) / stride + 1;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int p = 0; p < P; ++p) {
                for (int q = 0; q < Q; ++q) {
                    float acc = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int h = p * stride - pad + r;
                                int w_idx = q * stride - pad + s;
                                if (h < 0 || h >= H || w_idx < 0 || w_idx >= W) {
                                    continue;  // padding is zero
                                }
                                acc += x[((n * C + c) * H + h) * W + w_idx] *
                                       w[((k * C + c) * R + r) * S + s];
                            }
                        }
                    }
                    y[((n * K + k) * P + p) * Q + q] = acc;
                }
            }
        }
    }
}

void relu(const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void batchnorm(const float* x, const float* gamma, const float* beta, const float* mean,
               const float* var, float* y, int n, int channels, int inner, float eps) {
    for (int i = 0; i < n; ++i) {
        int c = (i / inner) % channels;
        y[i] = (x[i] - mean[c]) / std::sqrt(var[c] + eps) * gamma[c] + beta[c];
    }
}

}  // namespace tir::ref
