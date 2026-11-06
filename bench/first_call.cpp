// Measure why the first few calls are slower than later ones.
//
//   1. First CUDA API call: creates the GPU context (driver init, memory mapping, ...)
//   2. First launch of each kernel: loads that kernel's GPU code (lazy loading is the default
//      since CUDA 12.2)
//   3. Later calls: the "real" time once setup is done
//
// That is why benchmarks run a few warmup iterations before measuring.
//
// Try: CUDA_MODULE_LOADING=EAGER ./first_call
//   Loads every kernel at context creation. How do the times for 1 and 2 change?

#include <cstdio>

#include "bench_util.h"
#include "tir/kernels.h"

using namespace tir;

int main() {
    bench::CpuTimer cpu;

    // 1. Context creation. cudaFree(nullptr) frees nothing but forces the context to be created.
    cpu.start();
    TIR_CUDA_CHECK(cudaFree(nullptr));
    std::printf("context creation (first CUDA API call): %10.1f us\n\n", cpu.stop_us());

    constexpr int kSize = 512;
    constexpr int kCalls = 5;
    float *A = nullptr, *B = nullptr, *C = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&A, kSize * kSize * sizeof(float)));
    TIR_CUDA_CHECK(cudaMalloc(&B, kSize * kSize * sizeof(float)));
    TIR_CUDA_CHECK(cudaMalloc(&C, kSize * kSize * sizeof(float)));
    TIR_CUDA_CHECK(cudaMemset(A, 0, kSize * kSize * sizeof(float)));
    TIR_CUDA_CHECK(cudaMemset(B, 0, kSize * kSize * sizeof(float)));
    TIR_CUDA_CHECK(cudaDeviceSynchronize());

    bench::GpuTimer gpu;

    // 2, 3. Call the same kernel repeatedly and time each call.
    //   host time = how long the CPU waited from launch through synchronization
    //   GPU time  = how long the kernel ran on the GPU
    std::printf("gemm_naive %dx%dx%d\n", kSize, kSize, kSize);
    std::printf("%6s  %14s  %14s\n", "call", "host (us)", "GPU (us)");
    for (int i = 1; i <= kCalls; ++i) {
        cpu.start();
        gpu.start();
        launch_gemm_naive(A, B, C, kSize, kSize, kSize);
        gpu.stop();
        TIR_CUDA_CHECK(cudaDeviceSynchronize());
        double host_us = cpu.stop_us();
        std::printf("%6d  %14.1f  %14.1f\n", i, host_us, gpu.elapsed_us());
    }

    // A different kernel is slow again on its own first launch, even though gemm is warm,
    // because each kernel is loaded onto the GPU separately.
    constexpr int kN = kSize * kSize;
    std::printf("\nvector_add n=%d (gemm already warmed up)\n", kN);
    std::printf("%6s  %14s  %14s\n", "call", "host (us)", "GPU (us)");
    for (int i = 1; i <= kCalls; ++i) {
        cpu.start();
        gpu.start();
        launch_vector_add(A, B, C, kN);
        gpu.stop();
        TIR_CUDA_CHECK(cudaDeviceSynchronize());
        double host_us = cpu.stop_us();
        std::printf("%6d  %14.1f  %14.1f\n", i, host_us, gpu.elapsed_us());
    }

    TIR_CUDA_CHECK(cudaFree(A));
    TIR_CUDA_CHECK(cudaFree(B));
    TIR_CUDA_CHECK(cudaFree(C));
    return 0;
}
