#pragma once

// Shared test utilities: random inputs, a device memory helper, result comparison.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "tir/cuda_check.h"

namespace tir::test {

// Uniform random values in [-1, 1]. The seed is fixed so every run sees the same input.
inline std::vector<float> random_vector(size_t n, unsigned seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dist(gen);
    }
    return v;
}

// Minimal device memory helper for tests.
// cudaMalloc (+ copy) on construction, cudaFree on destruction (RAII).
// The runtime's real Tensor type comes in a later week.
class DeviceArray {
  public:
    // ': n_(n)' initializes the member n_ before the body runs.
    explicit DeviceArray(size_t n) : n_(n) { TIR_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(float))); }
    // ': DeviceArray(...)' first runs the constructor above (allocate), then this body copies.
    explicit DeviceArray(const std::vector<float>& host) : DeviceArray(host.size()) {
        TIR_CUDA_CHECK(cudaMemcpy(ptr_, host.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { cudaFree(ptr_); }

    // Copying would free the same pointer twice, so it is disabled
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    float* data() { return ptr_; }
    std::vector<float> to_host() const {
        std::vector<float> host(n_);
        TIR_CUDA_CHECK(cudaMemcpy(host.data(), ptr_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        return host;
    }

  private:
    float* ptr_ = nullptr;
    size_t n_ = 0;
};

// Compare GPU and CPU results element by element.
// Floating-point results differ slightly with evaluation order, so instead of exact equality
// we check |actual - expected| <= atol + rtol * |expected|.
inline bool all_close(const std::vector<float>& actual, const std::vector<float>& expected,
                      float rtol = 1e-4f, float atol = 1e-5f) {
    if (actual.size() != expected.size()) {
        std::printf("  size mismatch: %zu vs %zu\n", actual.size(), expected.size());
        return false;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        if (diff > atol + rtol * std::fabs(expected[i])) {
            if (mismatches < 5) {  // print only the first few
                std::printf("  mismatch at [%zu]: got %f, expected %f\n", i, actual[i],
                            expected[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches > 0) {
        std::printf("  %zu / %zu mismatches\n", mismatches, actual.size());
    }
    return mismatches == 0;
}

}  // namespace tir::test
