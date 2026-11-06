#include <cstdio>

#include "common.h"
#include "reference/reference.h"
#include "tir/kernels.h"

using namespace tir;

static bool check(int n) {
    auto a = test::random_vector(n, 1);
    auto b = test::random_vector(n, 2);

    std::vector<float> expected(n);
    ref::vector_add(a.data(), b.data(), expected.data(), n);

    test::DeviceArray d_a(a), d_b(b), d_c(n);
    launch_vector_add(d_a.data(), d_b.data(), d_c.data(), n);
    auto actual = d_c.to_host();  // waits for the kernel to finish

    bool ok = test::all_close(actual, expected);
    std::printf("vector_add n=%-8d %s\n", n, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    // 1, smaller than the block size (256), not a multiple of it, and large
    for (int n : {1, 100, 256, 1000, 1 << 20}) {
        ok &= check(n);
    }
    return ok ? 0 : 1;
}
