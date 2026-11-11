#include <cstdio>

#include "common.h"
#include "reference/reference.h"
#include "tir/kernels.h"

using namespace tir;

static bool check(int n) {
    auto x = test::random_vector(n, 1);
    std::vector<float> expected(n);
    ref::relu(x.data(), expected.data(), n);

    test::DeviceArray d_x(x), d_y(n);
    launch_relu(d_x.data(), d_y.data(), n);
    bool ok = test::all_close(d_y.to_host(), expected);
    std::printf("relu n=%-8d %s\n", n, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    for (int n : {1, 1000, 1 << 20}) {
        ok &= check(n);
    }
    return ok ? 0 : 1;
}
