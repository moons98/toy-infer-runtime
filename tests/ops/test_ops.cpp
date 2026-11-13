// Runs each registered op through run_op() and compares with the CPU reference.
// The kernel tests (test_gemm.cpp, ...) check kernels directly; this checks the op layer:
// shape inference, attrs, and passing the right sizes to the kernel.

#include <cstdio>
#include <stdexcept>

#include "common.h"
#include "reference/reference.h"
#include "tir/op.h"

using namespace tir;

static bool report(const std::string& name, bool ok) {
    std::printf("%-50s %s\n", name.c_str(), ok ? "OK" : "FAIL");
    return ok;
}

template <typename F>
static bool throws(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static bool check_gemm(int M, int N, int K) {
    auto a = test::random_vector(size_t(M) * K, 1);
    auto b = test::random_vector(size_t(K) * N, 2);
    std::vector<float> expected(size_t(M) * N);
    ref::gemm(a.data(), b.data(), expected.data(), M, N, K);

    Tensor y = run_op("gemm", {Tensor::from_host(a, {M, K}), Tensor::from_host(b, {K, N})});
    bool ok = y.shape() == Shape{M, N} && test::all_close(y.to_host(), expected, 1e-4f, 1e-4f);
    return report("gemm " + to_string({M, K}) + " x " + to_string({K, N}), ok);
}

static bool check_conv2d(int N, int C, int H, int W, int K, int R, int S, int pad, int stride) {
    auto x = test::random_vector(size_t(N) * C * H * W, 1);
    auto w = test::random_vector(size_t(K) * C * R * S, 2);
    Tensor y = run_op("conv2d", {Tensor::from_host(x, {N, C, H, W}), Tensor::from_host(w, {K, C, R, S})},
                      {{"pad", pad}, {"stride", stride}});

    std::vector<float> expected(y.numel());
    ref::conv2d(x.data(), w.data(), expected.data(), N, C, H, W, K, R, S, pad, stride);
    bool ok = test::all_close(y.to_host(), expected, 1e-4f, 1e-4f);
    return report("conv2d x" + to_string({N, C, H, W}) + " -> " + to_string(y.shape()), ok);
}

static bool check_relu(const Shape& shape) {
    auto x = test::random_vector(numel(shape), 1);
    std::vector<float> expected(x.size());
    ref::relu(x.data(), expected.data(), static_cast<int>(x.size()));

    Tensor y = run_op("relu", {Tensor::from_host(x, shape)});
    bool ok = y.shape() == shape && test::all_close(y.to_host(), expected);
    return report("relu " + to_string(shape), ok);
}

static bool check_batchnorm(const Shape& shape) {
    int channels = shape[1];
    int inner = static_cast<int>(numel(shape) / (shape[0] * shape[1]));
    auto x = test::random_vector(numel(shape), 1);
    std::vector<std::vector<float>> p;  // gamma, beta, mean, var
    for (unsigned s = 2; s <= 5; ++s) {
        p.push_back(test::random_vector(channels, s));
    }
    for (auto& v : p[3]) {
        v = v + 1.5f;  // variance must be positive
    }
    std::vector<float> expected(x.size());
    ref::batchnorm(x.data(), p[0].data(), p[1].data(), p[2].data(), p[3].data(), expected.data(),
                   static_cast<int>(x.size()), channels, inner, 1e-5f);

    std::vector<Tensor> inputs = {Tensor::from_host(x, shape)};
    for (const auto& v : p) {
        inputs.push_back(Tensor::from_host(v, {channels}));
    }
    Tensor y = run_op("batchnorm", inputs);
    bool ok = y.shape() == shape && test::all_close(y.to_host(), expected);
    return report("batchnorm " + to_string(shape), ok);
}

int main() {
    bool ok = true;

    ok &= check_gemm(17, 33, 9);
    ok &= check_gemm(1, 10, 84);
    ok &= report("gemm rejects mismatched K",
                 throws([] { run_op("gemm", {Tensor::empty({2, 3}), Tensor::empty({4, 5})}); }));

    ok &= check_conv2d(2, 3, 8, 8, 4, 3, 3, 1, 1);
    ok &= check_conv2d(1, 1, 28, 28, 6, 5, 5, 0, 1);
    ok &= check_conv2d(2, 4, 17, 13, 6, 3, 3, 1, 2);
    ok &= report("conv2d rejects mismatched channels",
                 throws([] { run_op("conv2d", {Tensor::empty({1, 3, 8, 8}), Tensor::empty({4, 2, 3, 3})}); }));

    ok &= check_relu({2, 6, 24, 24});
    ok &= check_relu({1, 120});

    ok &= check_batchnorm({1, 64, 56, 56});
    ok &= check_batchnorm({4, 120});
    ok &= report("batchnorm rejects a wrong parameter length", throws([] {
                     Tensor p = Tensor::empty({5});
                     run_op("batchnorm", {Tensor::empty({1, 6, 4, 4}), p, p, p, p});
                 }));

    return ok ? 0 : 1;
}
