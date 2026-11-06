#include "tir/cuda_check.h"
#include "tir/kernels.h"

namespace tir {

__global__ void conv2d_naive_kernel(const float* x, const float* w, float* y, int N, int C, int H,
                                    int W, int K, int R, int S, int pad, int stride, int P, int Q) {
    // Flatten the output [N, K, P, Q] to 1D; each thread handles one element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * P * Q;
    if (idx >= total) {
        return;
    }

    // Recover (n, k, p, q) from the flat index, peeling off the innermost dimension (q) first.
    int q = idx % Q;
    int p = (idx / Q) % P;
    int k = (idx / (P * Q)) % K;
    int n = idx / (K * P * Q);

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int h = p * stride - pad + r;  // input row seen by this filter tap
            if (h < 0 || h >= H) {
                continue;  // padding is zero, so skip it
            }
            for (int s = 0; s < S; ++s) {
                int w_col = q * stride - pad + s;  // input column seen by this filter tap
                if (w_col < 0 || w_col >= W) {
                    continue;
                }
                // NCHW: W is innermost (contiguous) in memory, then H, C, N
                acc += x[((n * C + c) * H + h) * W + w_col] * w[((k * C + c) * R + r) * S + s];
            }
        }
    }
    y[idx] = acc;  // idx is exactly the NCHW offset of [n, k, p, q]
}

void launch_conv2d_naive(const float* x, const float* w, float* y, int N, int C, int H, int W,
                         int K, int R, int S, int pad, int stride, cudaStream_t stream) {
    int P = conv2d_out_size(H, R, pad, stride);
    int Q = conv2d_out_size(W, S, pad, stride);
    int total = N * K * P * Q;

    constexpr int kBlockSize = 256;
    int grid_size = (total + kBlockSize - 1) / kBlockSize;
    conv2d_naive_kernel<<<grid_size, kBlockSize, 0, stream>>>(x, w, y, N, C, H, W, K, R, S, pad,
                                                              stride, P, Q);
    TIR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tir
