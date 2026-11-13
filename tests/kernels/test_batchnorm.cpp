#include <cstdio>

#include "common.h"
#include "reference/reference.h"
#include "tir/kernels.h"

using namespace tir;

static bool check(int batch, int channels, int inner) {
    int n = batch * channels * inner;
    auto x = test::random_vector(n, 1);
    auto gamma = test::random_vector(channels, 2);
    auto beta = test::random_vector(channels, 3);
    auto mean = test::random_vector(channels, 4);
    auto var = test::random_vector(channels, 5);
    for (auto& v : var) {
        v = v + 1.5f;  // variance must be positive
    }
    std::vector<float> expected(n);
    ref::batchnorm(x.data(), gamma.data(), beta.data(), mean.data(), var.data(), expected.data(), n, channels, inner,
                   1e-5f);

    test::DeviceArray d_x(x), d_g(gamma), d_b(beta), d_m(mean), d_v(var), d_y(n);
    launch_batchnorm(d_x.data(), d_g.data(), d_b.data(), d_m.data(), d_v.data(), d_y.data(), n, channels, inner, 1e-5f);
    bool ok = test::all_close(d_y.to_host(), expected);
    std::printf("batchnorm batch=%d channels=%-4d inner=%-5d %s\n", batch, channels, inner, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= check(1, 64, 56 * 56);  // ResNet18 first block
    ok &= check(2, 3, 7 * 5);
    ok &= check(4, 120, 1);  // [N, C]
    return ok ? 0 : 1;
}
