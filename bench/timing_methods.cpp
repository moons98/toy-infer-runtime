// Time the same GEMM kernel three different ways and compare.
//
//   (a) chrono      : CPU time around the launch call
//   (b) chrono+sync : CPU time from launch through cudaDeviceSynchronize
//   (c) cudaEvent   : difference between start/end timestamps recorded by the GPU
//
// Usage: ./timing_methods [repeats]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tir/cuda_check.h"
#include "tir/kernels.h"

using Clock = std::chrono::steady_clock;

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// Measurements jitter from run to run, so take the median of several
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void measure(int size, int repeats) {
    int M = size, N = size, K = size;
    float *A = nullptr, *B = nullptr, *C = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&A, size_t(M) * K * sizeof(float)));
    TIR_CUDA_CHECK(cudaMalloc(&B, size_t(K) * N * sizeof(float)));
    TIR_CUDA_CHECK(cudaMalloc(&C, size_t(M) * N * sizeof(float)));
    // Values do not affect timing, so just zero-fill
    TIR_CUDA_CHECK(cudaMemset(A, 0, size_t(M) * K * sizeof(float)));
    TIR_CUDA_CHECK(cudaMemset(B, 0, size_t(K) * N * sizeof(float)));

    // The first launch includes one-time setup cost, so run it once up front
    // (first_call measures this separately)
    tir::launch_gemm_naive(A, B, C, M, N, K);
    TIR_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_event, stop_event;
    TIR_CUDA_CHECK(cudaEventCreate(&start_event));
    TIR_CUDA_CHECK(cudaEventCreate(&stop_event));

    std::vector<double> t_chrono, t_chrono_sync, t_event;
    for (int i = 0; i < repeats; ++i) {
        // (a) Wrap only the launch call.
        //     Launch just enqueues work and returns, so kernel execution time is not included.
        auto t0 = Clock::now();
        tir::launch_gemm_naive(A, B, C, M, N, K);
        auto t1 = Clock::now();
        t_chrono.push_back(elapsed_us(t0, t1));
        TIR_CUDA_CHECK(cudaDeviceSynchronize());  // drain so the next measurement is clean

        // (b) Wait from launch until the GPU work finishes.
        //     Includes kernel time + launch cost + synchronization cost.
        t0 = Clock::now();
        tir::launch_gemm_naive(A, B, C, M, N, K);
        TIR_CUDA_CHECK(cudaDeviceSynchronize());
        t1 = Clock::now();
        t_chrono_sync.push_back(elapsed_us(t0, t1));

        // (c) Put "record a timestamp" markers into the GPU queue before and after the kernel.
        //     The GPU stamps the time as it passes each marker, so only GPU execution is measured.
        TIR_CUDA_CHECK(cudaEventRecord(start_event));
        tir::launch_gemm_naive(A, B, C, M, N, K);
        TIR_CUDA_CHECK(cudaEventRecord(stop_event));
        TIR_CUDA_CHECK(cudaEventSynchronize(stop_event));  // wait until stop is recorded
        float ms = 0.0f;
        TIR_CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        t_event.push_back(ms * 1000.0);
    }

    std::printf("%6d  %14.1f  %14.1f  %14.1f\n", size, median(t_chrono), median(t_chrono_sync), median(t_event));

    TIR_CUDA_CHECK(cudaEventDestroy(start_event));
    TIR_CUDA_CHECK(cudaEventDestroy(stop_event));
    TIR_CUDA_CHECK(cudaFree(A));
    TIR_CUDA_CHECK(cudaFree(B));
    TIR_CUDA_CHECK(cudaFree(C));
}

int main(int argc, char** argv) {
    constexpr int kDefaultRepeats = 20;
    int repeats = argc > 1 ? std::atoi(argv[1]) : kDefaultRepeats;
    if (repeats < 1) {
        repeats = kDefaultRepeats;  // atoi gives 0 for non-numbers; median() needs a run
    }

    std::printf("gemm_naive (M = N = K = size), median of %d runs, unit: us\n\n", repeats);
    std::printf("%6s  %14s  %14s  %14s\n", "size", "(a) chrono", "(b) chrono+sync", "(c) cudaEvent");
    for (int size : {32, 128, 512, 1024, 2048}) {
        measure(size, repeats);
    }
    return 0;
}
