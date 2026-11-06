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

#include "tir/cuda_check.h"
#include "tir/kernels.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// vector_add: c[i] = a[i] + b[i]
constexpr int kVectorN = 1 << 20;

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

int main() {
    bool ok = true;
    ok &= run_vector_add(kVectorN);
    return ok ? 0 : 1;
}
