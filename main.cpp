// tir: the main program of this repo.
//
// This is plain C++. The CUDA kernels live in src/kernels/*.cu and are called through the
// launch_* functions declared in tir/kernels.h.
//
// Without a runtime, every step of running a kernel is written by hand: allocate device
// memory, copy inputs, launch, copy results back, check against the CPU, free.
// Here device memory lives in Tensors. Tensor::from_host allocates and copies,
// to_host copies back, and the memory is freed automatically when the Tensor goes away.
//
// To change what runs, edit the configuration below (or comment out a run_* call in main)
// and rebuild.

#include <cmath>
#include <cstdio>
#include <vector>

#include "reference/reference.h"
#include "tir/kernels.h"
#include "tir/tensor.h"

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

    // 1. Prepare inputs in host (CPU) memory
    std::vector<float> h_a(n), h_b(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // 2. Allocate device memory and copy the inputs host -> device
    tir::Tensor a = tir::Tensor::from_host(h_a, {n});
    tir::Tensor b = tir::Tensor::from_host(h_b, {n});
    tir::Tensor c = tir::Tensor::empty({n});

    // 3. Launch the kernel (asynchronous: returns before the kernel finishes)
    tir::launch_vector_add(a.data(), b.data(), c.data(), n);

    // 4. Copy the result device -> host. This waits for the kernel to finish.
    std::vector<float> h_c = c.to_host();

    // 5. Compute the same thing on the CPU and compare
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            ++mismatches;
        }
    }

    // No cudaFree: a, b, and c release their memory when this function returns.
    std::printf("  %s (%d mismatches)\n", mismatches == 0 ? "OK" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool run_gemm(int M, int N, int K) {
    std::printf("[gemm_naive] M = %d, N = %d, K = %d\n", M, N, K);

    // Same sequence as vector_add again. Only the shapes and the launch function differ.
    std::vector<float> h_A(size_t(M) * K), h_B(size_t(K) * N);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>(i % 7) * 0.1f;
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = static_cast<float>(i % 5) * 0.1f;
    }

    tir::Tensor A = tir::Tensor::from_host(h_A, {M, K});
    tir::Tensor B = tir::Tensor::from_host(h_B, {K, N});
    tir::Tensor C = tir::Tensor::empty({M, N});

    tir::launch_gemm_naive(A.data(), B.data(), C.data(), M, N, K);
    std::vector<float> h_C = C.to_host();

    std::vector<float> expected(size_t(M) * N);
    tir::ref::gemm(h_A.data(), h_B.data(), expected.data(), M, N, K);
    int mismatches = 0;
    for (size_t i = 0; i < h_C.size(); ++i) {
        if (std::fabs(h_C[i] - expected[i]) > 1e-3f * (1.0f + std::fabs(expected[i]))) {
            ++mismatches;
        }
    }

    std::printf("  %s (%d mismatches)\n", mismatches == 0 ? "OK" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool run_conv2d(int N, int C, int H, int W, int K, int R, int S, int pad, int stride) {
    // We still compute the output size ourselves (to allocate y) and pass every size and
    // setting down to the kernel. The op registry takes this over next.
    int P = (H + 2 * pad - R) / stride + 1;
    int Q = (W + 2 * pad - S) / stride + 1;
    std::printf("[conv2d_naive] x[%d,%d,%d,%d] w[%d,%d,%d,%d] pad=%d stride=%d -> y[%d,%d,%d,%d]\n",
                N, C, H, W, K, C, R, S, pad, stride, N, K, P, Q);
    std::vector<float> h_x(size_t(N) * C * H * W), h_w(size_t(K) * C * R * S);
    for (size_t i = 0; i < h_x.size(); ++i) {
        h_x[i] = static_cast<float>(i % 11) * 0.1f;
    }
    for (size_t i = 0; i < h_w.size(); ++i) {
        h_w[i] = static_cast<float>(i % 3) * 0.1f;
    }

    tir::Tensor x = tir::Tensor::from_host(h_x, {N, C, H, W});
    tir::Tensor w = tir::Tensor::from_host(h_w, {K, C, R, S});
    tir::Tensor y = tir::Tensor::empty({N, K, P, Q});

    // launch_conv2d_naive runs one thread per output element
    tir::launch_conv2d_naive(x.data(), w.data(), y.data(), N, C, H, W, K, R, S, pad, stride);
    std::vector<float> h_y = y.to_host();

    std::vector<float> expected(h_y.size());
    tir::ref::conv2d(h_x.data(), h_w.data(), expected.data(), N, C, H, W, K, R, S, pad, stride);
    int mismatches = 0;
    for (size_t i = 0; i < h_y.size(); ++i) {
        if (std::fabs(h_y[i] - expected[i]) > 1e-3f * (1.0f + std::fabs(expected[i]))) {
            ++mismatches;
        }
    }

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
