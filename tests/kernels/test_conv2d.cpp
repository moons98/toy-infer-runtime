#include <cstdio>

#include "common.h"
#include "reference/reference.h"
#include "tir/kernels.h"

using namespace tir;

struct ConvCase {
    int N, C, H, W, K, R, S, pad, stride;
};

static bool check(const ConvCase& c) {
    int P = conv2d_out_size(c.H, c.R, c.pad, c.stride);
    int Q = conv2d_out_size(c.W, c.S, c.pad, c.stride);
    size_t x_size = size_t(c.N) * c.C * c.H * c.W;
    size_t w_size = size_t(c.K) * c.C * c.R * c.S;
    size_t y_size = size_t(c.N) * c.K * P * Q;

    auto x = test::random_vector(x_size, 1);
    auto w = test::random_vector(w_size, 2);

    std::vector<float> expected(y_size);
    ref::conv2d(x.data(), w.data(), expected.data(), c.N, c.C, c.H, c.W, c.K, c.R, c.S, c.pad, c.stride);

    test::DeviceArray d_x(x), d_w(w), d_y(y_size);
    launch_conv2d_naive(d_x.data(), d_w.data(), d_y.data(), c.N, c.C, c.H, c.W, c.K, c.R, c.S, c.pad, c.stride);
    auto actual = d_y.to_host();

    bool ok = test::all_close(actual, expected, 1e-4f, 1e-4f);
    std::printf("conv2d x[%d,%d,%d,%d] w[%d,%d,%d,%d] pad=%d stride=%d -> y[%d,%d,%d,%d] %s\n", c.N, c.C, c.H, c.W, c.K,
                c.C, c.R, c.S, c.pad, c.stride, c.N, c.K, P, Q, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    //                N   C   H   W   K  R  S pad stride
    ok &= check({1, 1, 5, 5, 1, 3, 3, 0, 1});      // smallest case
    ok &= check({2, 3, 8, 8, 4, 3, 3, 1, 1});      // padding keeps the size
    ok &= check({2, 4, 17, 13, 6, 3, 3, 1, 2});    // non-square input, stride 2
    ok &= check({1, 8, 7, 7, 16, 1, 1, 0, 1});     // 1x1 conv
    ok &= check({1, 3, 32, 32, 8, 5, 5, 2, 1});    // 5x5 filter
    ok &= check({4, 16, 28, 28, 32, 3, 3, 1, 1});  // somewhat larger
    return ok ? 0 : 1;
}
