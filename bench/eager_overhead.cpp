// What does eager run_op cost on top of the kernel itself?
//
// Runs a chain of kChain relu ops on n elements three ways and reports the time per op:
//   (a) direct     : launch_relu on two preallocated buffers (no runtime)
//   (b) run_op     : each op allocates its output; the previous output is freed right away
//                    (what main's eager forward pass does)
//   (c) run_op keep: same, but every output is kept until the chain is done, so no cudaFree
//                    happens inside the timed region
//
// What to look for
//   - Small n: (b) is several times slower than (a). Where does the time go?
//   - (b) vs (c): the gap is cudaFree. It waits for the GPU to finish all queued work, so the
//     host cannot run ahead and queue the next ops while the GPU is busy.
//   - Large n: the gap does not close. cudaMalloc itself gets slower as the size grows, and
//     soon costs more than the kernel. This is why real runtimes reuse memory instead of
//     calling cudaMalloc for every op.
//
// Usage: ./eager_overhead [repeats]

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_util.h"
#include "tir/kernels.h"
#include "tir/op.h"

using namespace tir;

constexpr int kChain = 100;

static double time_direct(int n) {
    Tensor a = Tensor::empty({n}), b = Tensor::empty({n});
    bench::CpuTimer timer;
    timer.start();
    for (int i = 0; i < kChain; ++i) {
        // ping-pong between the two buffers
        if (i % 2 == 0) {
            launch_relu(a.data(), b.data(), n);
        } else {
            launch_relu(b.data(), a.data(), n);
        }
    }
    TIR_CUDA_CHECK(cudaDeviceSynchronize());
    return timer.stop_us() / kChain;
}

static double time_run_op(int n) {
    Tensor x = Tensor::empty({n});
    bench::CpuTimer timer;
    timer.start();
    for (int i = 0; i < kChain; ++i) {
        x = run_op("relu", {x});  // the old x is freed here (cudaFree)
    }
    TIR_CUDA_CHECK(cudaDeviceSynchronize());
    return timer.stop_us() / kChain;
}

static double time_run_op_keep(int n) {
    std::vector<Tensor> keep;
    keep.reserve(kChain + 1);
    keep.push_back(Tensor::empty({n}));
    bench::CpuTimer timer;
    timer.start();
    for (int i = 0; i < kChain; ++i) {
        keep.push_back(run_op("relu", {keep.back()}));  // nothing is freed inside the loop
    }
    TIR_CUDA_CHECK(cudaDeviceSynchronize());
    return timer.stop_us() / kChain;
}  // all outputs are freed here, outside the timed region

int main(int argc, char** argv) {
    int repeats = argc > 1 ? std::atoi(argv[1]) : 10;
    if (repeats <= 0) {
        repeats = 10;
    }

    // Warm up: context creation, kernel loading, the registry
    time_direct(1024);
    time_run_op(1024);

    std::printf("chain of %d relu ops, time per op (us), median of %d runs\n\n", kChain, repeats);
    std::printf("%10s  %12s  %12s  %14s  %10s\n", "n", "(a) direct", "(b) run_op", "(c) run_op keep", "(b) / (a)");
    for (int n : {1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 22}) {
        std::vector<double> a, b, c;
        for (int r = 0; r < repeats; ++r) {
            a.push_back(time_direct(n));
            b.push_back(time_run_op(n));
            c.push_back(time_run_op_keep(n));
        }
        double ma = bench::median(a), mb = bench::median(b), mc = bench::median(c);
        std::printf("%10d  %12.2f  %12.2f  %14.2f  %9.1fx\n", n, ma, mb, mc, mb / ma);
    }
    return 0;
}
