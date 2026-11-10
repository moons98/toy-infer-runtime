#pragma once

// Timers for measuring CPU and GPU time, plus a median helper.
// The chrono / cudaEvent code from bench/timing_methods.cpp, packaged for reuse by main and the
// benchmarks.

#include <algorithm>
#include <chrono>
#include <vector>

#include "tir/cuda_check.h"

namespace tir {

// CPU wall-clock time (us)
class CpuTimer {
  public:
    void start() { start_ = Clock::now(); }
    double stop_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }

  private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

// GPU execution time (us) of the work enqueued on the stream between start() and stop().
// Creates the events on construction and destroys them on destruction (RAII).
class GpuTimer {
  public:
    GpuTimer() {
        TIR_CUDA_CHECK(cudaEventCreate(&start_));
        TIR_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = 0) { TIR_CUDA_CHECK(cudaEventRecord(start_, stream)); }
    void stop(cudaStream_t stream = 0) { TIR_CUDA_CHECK(cudaEventRecord(stop_, stream)); }

    // Wait for the GPU to reach the stop marker, then read the time
    double elapsed_us() const {
        TIR_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        TIR_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms * 1000.0;
    }

  private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// Measurements jitter, so use the median of several runs.
// A single outlier can drag the mean far off; the median is robust to it.
inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace tir
