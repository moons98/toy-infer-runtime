// tir: the main program of this repo.
//
// This is plain C++. The CUDA kernels live in src/kernels/*.cu and are called through the
// launch_* functions declared in tir/kernels.h.
//
// Week 0 has no runtime yet, so every step of running a kernel is written out by hand:
// allocate device memory, copy inputs, launch, copy results back, check against the CPU, free.
// As the runtime grows in later weeks (Tensor, Op, Graph), this file gets shorter.
//
// cudaMalloc/cudaFree are paired by hand on purpose (no RAII).
// The point is to feel what gets repeated and what is easy to get wrong.
//
// To change what runs, edit the configuration below (or comment out a run_* call in main)
// and rebuild.

#include <cmath>
#include <cstdio>
#include <vector>

#include "reference/reference.h"
#include "tir/cuda_check.h"
#include "tir/kernels.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// vector_add: c[i] = a[i] + b[i]
constexpr int kVectorN = 1 << 20;

// gemm: C[M, N] = A[M, K] * B[K, N]
struct GemmConfig {
    int M, N, K;
};
constexpr GemmConfig kGemm = {256, 256, 256};

// conv2d: x[N, C, H, W] * w[K, C, R, S] -> y[N, K, P, Q]
struct Conv2dConfig {
    int N, C, H, W;  // input
    int K, R, S;     // filter
    int pad, stride;
};
constexpr Conv2dConfig kConv2d = {1, 1, 28, 28, 6, 5, 5, 0, 1};  // LeNet first layer

// ---------------------------------------------------------------------------

static bool run_vector_add(int n) {
    std::printf("[vector_add] n = %d\n", n);
    size_t bytes = n * sizeof(float);

    // 1. Prepare inputs in host (CPU) memory
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // 2. Allocate device (GPU) memory. The GPU cannot read host memory directly.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&d_a, bytes));
    TIR_CUDA_CHECK(cudaMalloc(&d_b, bytes));
    TIR_CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 3. Copy inputs host -> device
    TIR_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    TIR_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 4. Launch the kernel. launch_vector_add (src/kernels/vector_add.cu) chooses how many
    //    threads to run and launches vector_add_kernel<<<grid, block>>>.
    //    It only enqueues work on the GPU and returns immediately (asynchronous).
    //    It does check that the launch itself succeeded, but errors *during* kernel
    //    execution only surface at the next synchronization point.
    tir::launch_vector_add(d_a, d_b, d_c, n);

    // 5. Copy the result device -> host.
    //    Plain cudaMemcpy waits for preceding GPU work, so the kernel is guaranteed done here.
    TIR_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // 6. Compute the same thing on the CPU and compare
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            ++mismatches;
        }
    }

    // 7. Free device memory. If forgotten, it stays allocated until the process exits.
    TIR_CUDA_CHECK(cudaFree(d_a));
    TIR_CUDA_CHECK(cudaFree(d_b));
    TIR_CUDA_CHECK(cudaFree(d_c));

    std::printf("  %s (%d mismatches)\n", mismatches == 0 ? "OK" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool run_gemm(int M, int N, int K) {
    std::printf("[gemm_naive] M = %d, N = %d, K = %d\n", M, N, K);
    size_t bytes_A = size_t(M) * K * sizeof(float);
    size_t bytes_B = size_t(K) * N * sizeof(float);
    size_t bytes_C = size_t(M) * N * sizeof(float);

    // Same sequence as vector_add again. Only the sizes and the launch function differ.
    std::vector<float> h_A(size_t(M) * K), h_B(size_t(K) * N), h_C(size_t(M) * N);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>(i % 7) * 0.1f;
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = static_cast<float>(i % 5) * 0.1f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    TIR_CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    TIR_CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    TIR_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    TIR_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // C is 2D, so launch_gemm_naive lays threads out in 2D too: x for columns, y for rows
    tir::launch_gemm_naive(d_A, d_B, d_C, M, N, K);

    TIR_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    std::vector<float> expected(size_t(M) * N);
    tir::ref::gemm(h_A.data(), h_B.data(), expected.data(), M, N, K);
    int mismatches = 0;
    for (size_t i = 0; i < h_C.size(); ++i) {
        if (std::fabs(h_C[i] - expected[i]) > 1e-3f * (1.0f + std::fabs(expected[i]))) {
            ++mismatches;
        }
    }

    TIR_CUDA_CHECK(cudaFree(d_A));
    TIR_CUDA_CHECK(cudaFree(d_B));
    TIR_CUDA_CHECK(cudaFree(d_C));

    std::printf("  %s (%d mismatches)\n", mismatches == 0 ? "OK" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool run_conv2d(int N, int C, int H, int W, int K, int R, int S, int pad, int stride) {
    // We have to compute the output size ourselves (to allocate y) and pass every size and
    // setting down to the kernel.
    int P = (H + 2 * pad - R) / stride + 1;
    int Q = (W + 2 * pad - S) / stride + 1;
    std::printf("[conv2d_naive] x[%d,%d,%d,%d] w[%d,%d,%d,%d] pad=%d stride=%d -> y[%d,%d,%d,%d]\n",
                N, C, H, W, K, C, R, S, pad, stride, N, K, P, Q);
    size_t x_count = size_t(N) * C * H * W;
    size_t w_count = size_t(K) * C * R * S;
    size_t y_count = size_t(N) * K * P * Q;

    std::vector<float> h_x(x_count), h_w(w_count), h_y(y_count);
    for (size_t i = 0; i < x_count; ++i) {
        h_x[i] = static_cast<float>(i % 11) * 0.1f;
    }
    for (size_t i = 0; i < w_count; ++i) {
        h_w[i] = static_cast<float>(i % 3) * 0.1f;
    }

    float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&d_x, x_count * sizeof(float)));
    TIR_CUDA_CHECK(cudaMalloc(&d_w, w_count * sizeof(float)));
    TIR_CUDA_CHECK(cudaMalloc(&d_y, y_count * sizeof(float)));

    TIR_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_count * sizeof(float), cudaMemcpyHostToDevice));
    TIR_CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), w_count * sizeof(float), cudaMemcpyHostToDevice));

    // launch_conv2d_naive runs one thread per output element
    tir::launch_conv2d_naive(d_x, d_w, d_y, N, C, H, W, K, R, S, pad, stride);

    TIR_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected(y_count);
    tir::ref::conv2d(h_x.data(), h_w.data(), expected.data(), N, C, H, W, K, R, S, pad, stride);
    int mismatches = 0;
    for (size_t i = 0; i < y_count; ++i) {
        if (std::fabs(h_y[i] - expected[i]) > 1e-3f * (1.0f + std::fabs(expected[i]))) {
            ++mismatches;
        }
    }

    TIR_CUDA_CHECK(cudaFree(d_x));
    TIR_CUDA_CHECK(cudaFree(d_w));
    TIR_CUDA_CHECK(cudaFree(d_y));

    std::printf("  %s (%d mismatches)\n", mismatches == 0 ? "OK" : "FAIL", mismatches);
    return mismatches == 0;
}

int main() {
    bool ok = true;
    ok &= run_vector_add(kVectorN);
    ok &= run_gemm(kGemm.M, kGemm.N, kGemm.K);
    ok &= run_conv2d(kConv2d.N, kConv2d.C, kConv2d.H, kConv2d.W, kConv2d.K, kConv2d.R, kConv2d.S,
                     kConv2d.pad, kConv2d.stride);
    return ok ? 0 : 1;
}
