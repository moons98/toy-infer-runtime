// tir: the main program of this repo.
//
// This is plain C++. The CUDA kernels live in src/kernels/*.cu and are called through the
// launch_* functions declared in tir/kernels.h.
//
// Without a runtime, every step of running a kernel is written by hand: allocate device
// memory, copy inputs, launch, copy results back, check against the CPU, free.
// Here device memory lives in Tensors, and kernels are run by name through the op registry.
// run_op("gemm", {A, B}) computes the output shape, allocates the output, and launches the
// kernel, so main no longer deals with sizes, pointers, or launch functions.
//
// To change what runs, edit the configuration below (or comment out a run_* call in main)
// and rebuild.

#include <cmath>
#include <cstdio>
#include <vector>

#include "reference/reference.h"
#include "tir/op.h"
#include "tir/tensor.h"

using tir::Tensor;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

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

// Simple deterministic input values
static std::vector<float> make_input(size_t n, int period) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i % period) * 0.1f;
    }
    return v;
}

static bool check(const char* name, const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    int mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-3f * (1.0f + std::fabs(expected[i]))) {
            ++mismatches;
        }
    }
    std::printf("  %s: %s (%d mismatches)\n", name, mismatches == 0 ? "OK" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool run_gemm(int M, int N, int K) {
    auto h_A = make_input(size_t(M) * K, 7);
    auto h_B = make_input(size_t(K) * N, 5);

    Tensor A = Tensor::from_host(h_A, {M, K});
    Tensor B = Tensor::from_host(h_B, {K, N});
    Tensor C = tir::run_op("gemm", {A, B});
    std::printf("[gemm] %s x %s -> %s\n", tir::to_string(A.shape()).c_str(),
                tir::to_string(B.shape()).c_str(), tir::to_string(C.shape()).c_str());

    std::vector<float> expected(C.numel());
    tir::ref::gemm(h_A.data(), h_B.data(), expected.data(), M, N, K);
    return check("gemm", C.to_host(), expected);
}

static bool run_conv2d(int N, int C, int H, int W, int K, int R, int S, int pad, int stride) {
    auto h_x = make_input(size_t(N) * C * H * W, 11);
    auto h_w = make_input(size_t(K) * C * R * S, 3);

    // The op computes the output size [N, K, P, Q] itself
    Tensor x = Tensor::from_host(h_x, {N, C, H, W});
    Tensor w = Tensor::from_host(h_w, {K, C, R, S});
    Tensor y = tir::run_op("conv2d", {x, w}, {{"pad", pad}, {"stride", stride}});
    std::printf("[conv2d] x%s w%s pad=%d stride=%d -> y%s\n", tir::to_string(x.shape()).c_str(),
                tir::to_string(w.shape()).c_str(), pad, stride, tir::to_string(y.shape()).c_str());

    std::vector<float> expected(y.numel());
    tir::ref::conv2d(h_x.data(), h_w.data(), expected.data(), N, C, H, W, K, R, S, pad, stride);
    return check("conv2d", y.to_host(), expected);
}

int main() {
    bool ok = true;
    ok &= run_gemm(kGemm.M, kGemm.N, kGemm.K);
    ok &= run_conv2d(kConv2d.N, kConv2d.C, kConv2d.H, kConv2d.W, kConv2d.K, kConv2d.R, kConv2d.S,
                     kConv2d.pad, kConv2d.stride);
    return ok ? 0 : 1;
}
