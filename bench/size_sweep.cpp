// Grow the GEMM size and compare copy time against kernel time.
//
//   H2D    : time to copy inputs A, B host -> device
//   kernel : gemm_naive execution time
//   D2H    : time to copy output C device -> host
//   GFLOPS : floating-point ops per second. GEMM does K multiply-adds per element -> 2*M*N*K
//
// What to look for
//   - The small-size range where time barely drops (launch overhead is the floor)
//   - Which takes longer, copy or kernel, and where they cross over
//
// Things to try
//   - Allocate host buffers with cudaMallocHost (pinned memory) instead of std::vector.
//     How does copy speed change?
//
// Usage: ./size_sweep [repeats]

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_util.h"
#include "tir/kernels.h"

using namespace tir;

static void measure(int size, int repeats) {
    int M = size, N = size, K = size;
    size_t bytes_A = size_t(M) * K * sizeof(float);
    size_t bytes_B = size_t(K) * N * sizeof(float);
    size_t bytes_C = size_t(M) * N * sizeof(float);

    std::vector<float> h_A(bytes_A / sizeof(float), 1.0f);
    std::vector<float> h_B(bytes_B / sizeof(float), 1.0f);
    std::vector<float> h_C(bytes_C / sizeof(float));

    float *A = nullptr, *B = nullptr, *C = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&A, bytes_A));
    TIR_CUDA_CHECK(cudaMalloc(&B, bytes_B));
    TIR_CUDA_CHECK(cudaMalloc(&C, bytes_C));

    bench::GpuTimer timer;
    std::vector<double> t_h2d, t_kernel, t_d2h;

    // Discard the first iteration as warmup
    for (int i = 0; i <= repeats; ++i) {
        timer.start();
        TIR_CUDA_CHECK(cudaMemcpy(A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
        TIR_CUDA_CHECK(cudaMemcpy(B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
        timer.stop();
        double h2d = timer.elapsed_us();

        timer.start();
        launch_gemm_naive(A, B, C, M, N, K);
        timer.stop();
        double kernel = timer.elapsed_us();

        timer.start();
        TIR_CUDA_CHECK(cudaMemcpy(h_C.data(), C, bytes_C, cudaMemcpyDeviceToHost));
        timer.stop();
        double d2h = timer.elapsed_us();

        if (i > 0) {
            t_h2d.push_back(h2d);
            t_kernel.push_back(kernel);
            t_d2h.push_back(d2h);
        }
    }

    double kernel_us = bench::median(t_kernel);
    double gflops = 2.0 * M * N * K / (kernel_us * 1e-6) / 1e9;
    std::printf("%6d  %12.1f  %12.1f  %12.1f  %10.1f\n", size, bench::median(t_h2d), kernel_us,
                bench::median(t_d2h), gflops);

    TIR_CUDA_CHECK(cudaFree(A));
    TIR_CUDA_CHECK(cudaFree(B));
    TIR_CUDA_CHECK(cudaFree(C));
}

int main(int argc, char** argv) {
    constexpr int kDefaultRepeats = 10;
    int repeats = argc > 1 ? std::atoi(argv[1]) : kDefaultRepeats;
    if (repeats < 1) {
        repeats = kDefaultRepeats;  // atoi gives 0 for non-numbers; median() needs a run
    }

    std::printf("gemm_naive (M = N = K = size), median of %d runs\n\n", repeats);
    std::printf("%6s  %12s  %12s  %12s  %10s\n", "size", "H2D (us)", "kernel (us)", "D2H (us)",
                "GFLOPS");
    for (int size : {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
        measure(size, repeats);
    }
    return 0;
}
