// tir: the main program of this repo.
//
// This is plain C++. The CUDA kernels live in src/kernels/*.cu; main only talks to the runtime
// (Tensor and run_op).
//
// Runs one conv -> batchnorm -> relu block eagerly, the basic building block of ResNet.
// Each layer is one run_op() call, which finds the op by name in the registry, infers the output
// shape, allocates the output, and launches the kernel right away. Weights are random; the result
// is checked against the CPU reference.
//
// It then reports where the time goes: the block end to end vs. the kernels alone.
// The difference is the runtime's own overhead (shape inference, allocation, launch).
//
// To change what runs, edit the configuration below and rebuild.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "reference/reference.h"
#include "tir/cuda_check.h"
#include "tir/op.h"
#include "tir/tensor.h"
#include "tir/timer.h"

using tir::Attrs;
using tir::Shape;
using tir::Tensor;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Defaults match the first residual block of ResNet18: [N, 64, 56, 56], 3x3 conv, pad 1
constexpr int kBatch = 1;
constexpr int kChannels = 64;  // input and output channels
constexpr int kSize = 56;      // height and width
constexpr int kFilter = 3;     // conv filter size (kFilter x kFilter)
constexpr int kPad = 1;        // conv padding; 1 keeps 56x56 for a 3x3 filter
constexpr int kRepeats = 20;   // timed runs of the block (after one warmup run)
constexpr int kReplays = 100;  // launches per op when timing a kernel on its own

// ---------------------------------------------------------------------------

// A value kept on both sides: a host copy for the CPU reference and a Tensor for the GPU
struct Param {
    std::vector<float> host;
    Tensor tensor;
};

static Param make_param(const Shape& shape, float lo, float hi, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> host(tir::numel(shape));
    for (auto& v : host) {
        v = dist(gen);
    }
    return {host, Tensor::from_host(host, shape)};
}

struct Block {
    Param conv_w;                  // [K, C, R, S]
    Param gamma, beta, mean, var;  // batchnorm, each [K]
};

static Block make_block() {
    float w_scale = 1.0f / std::sqrt(static_cast<float>(kChannels * kFilter * kFilter));
    Block b;
    b.conv_w = make_param({kChannels, kChannels, kFilter, kFilter}, -w_scale, w_scale, 1);
    b.gamma = make_param({kChannels}, 0.5f, 1.5f, 2);
    b.beta = make_param({kChannels}, -0.5f, 0.5f, 3);
    b.mean = make_param({kChannels}, -0.5f, 0.5f, 4);
    b.var = make_param({kChannels}, 0.5f, 1.5f, 5);  // variance must be positive
    return b;
}

// One executed op, kept so its kernel can be timed again on its own afterwards
struct Step {
    std::string op;
    std::vector<Tensor> inputs;
    Attrs attrs;
    Tensor output;
};

// run_op, plus recording the call in `trace` if one is given
static Tensor op(std::vector<Step>* trace, const std::string& name,
                 const std::vector<Tensor>& inputs, const Attrs& attrs = {}) {
    Tensor out = tir::run_op(name, inputs, attrs);
    if (trace) {
        trace->push_back({name, inputs, attrs, out});
    }
    return out;
}

// The block on the GPU. Eager: each line launches its kernel immediately.
// Assigning to `y` drops the previous intermediate, whose memory is freed right there.
static Tensor run_block(const Tensor& x, const Block& b, std::vector<Step>* trace = nullptr) {
    Tensor y = op(trace, "conv2d", {x, b.conv_w.tensor}, {{"pad", kPad}});
    y = op(trace, "batchnorm", {y, b.gamma.tensor, b.beta.tensor, b.mean.tensor, b.var.tensor});
    return op(trace, "relu", {y});
}

// The same block on the CPU, with the reference implementations
static std::vector<float> run_block_cpu(const std::vector<float>& x, const Block& b) {
    namespace ref = tir::ref;
    int out_size = kSize + 2 * kPad - kFilter + 1;
    int inner = out_size * out_size;
    std::vector<float> y(size_t(kBatch) * kChannels * inner);
    int n = static_cast<int>(y.size());
    ref::conv2d(x.data(), b.conv_w.host.data(), y.data(), kBatch, kChannels, kSize, kSize,
                kChannels, kFilter, kFilter, kPad, 1);
    ref::batchnorm(y.data(), b.gamma.host.data(), b.beta.host.data(), b.mean.host.data(),
                   b.var.host.data(), y.data(), n, kChannels, inner, 1e-5f);
    ref::relu(y.data(), y.data(), n);
    return y;
}

static bool check(const std::vector<float>& actual, const std::vector<float>& expected) {
    int mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-3f * (1.0f + std::fabs(expected[i]))) {
            ++mismatches;
        }
    }
    std::printf("check vs CPU reference: %s (%d mismatches)\n", mismatches == 0 ? "OK" : "FAIL",
                mismatches);
    return mismatches == 0;
}

// Time one op's kernel alone: launch it kReplays times back to back on the tensors it used in
// the block. Nothing else runs in between, so the GPU never waits for the host.
static double kernel_time_us(const Step& step) {
    const tir::OpDef& def = tir::registry().get(step.op);
    Tensor out = step.output;
    tir::GpuTimer timer;
    timer.start();
    for (int i = 0; i < kReplays; ++i) {
        def.launch(step.inputs, out, step.attrs, tir::default_context());
    }
    timer.stop();
    return timer.elapsed_us() / kReplays;
}

int main() {
    Param x = make_param({kBatch, kChannels, kSize, kSize}, -1.0f, 1.0f, 0);
    Block block = make_block();

    // 1. Correctness: one traced run, compared with the CPU
    std::vector<Step> trace;
    Tensor y = run_block(x.tensor, block, &trace);
    bool ok = check(y.to_host(), run_block_cpu(x.host, block));

    // 2. End-to-end time of the block (CPU clock, until the GPU is done)
    auto& stats = tir::default_allocator().stats();
    std::vector<double> block_us;
    size_t allocs = 0, frees = 0;
    for (int i = 0; i < kRepeats; ++i) {
        tir::default_allocator().reset_stats();
        tir::CpuTimer timer;
        timer.start();
        {
            Tensor out = run_block(x.tensor, block);
            TIR_CUDA_CHECK(cudaDeviceSynchronize());
        }  // out is dropped here, so every allocation of this run has been freed
        block_us.push_back(timer.stop_us());
        allocs = stats.num_allocs;
        frees = stats.num_frees;
    }

    // 3. Kernel-only time of each op
    std::printf("\nconv -> batchnorm -> relu, x[%d,%d,%d,%d], fp32, eager\n\n", kBatch, kChannels,
                kSize, kSize);
    std::printf("%-10s %-16s %12s\n", "op", "output", "kernel (us)");
    double kernel_sum = 0.0;
    for (const Step& step : trace) {
        double us = kernel_time_us(step);
        kernel_sum += us;
        std::printf("%-10s %-16s %12.1f\n", step.op.c_str(),
                    tir::to_string(step.output.shape()).c_str(), us);
    }

    double total = tir::median(block_us);
    std::printf("\nsum of kernel times : %8.1f us\n", kernel_sum);
    std::printf("block end to end    : %8.1f us  (median of %d)\n", total, kRepeats);
    std::printf("runtime overhead    : %8.1f us  (%.0f%% of the block)\n", total - kernel_sum,
                100.0 * (total - kernel_sum) / total);
    std::printf("per run             : %zu run_op calls, %zu cudaMalloc, %zu cudaFree\n\n",
                trace.size(), allocs, frees);
    return ok ? 0 : 1;
}
