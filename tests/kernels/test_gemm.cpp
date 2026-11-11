#include <cstdio>

#include "common.h"
#include "reference/reference.h"
#include "tir/kernels.h"

using namespace tir;

static bool check(int M, int N, int K) {
    auto A = test::random_vector(size_t(M) * K, 1);
    auto B = test::random_vector(size_t(K) * N, 2);

    std::vector<float> expected(size_t(M) * N);
    ref::gemm(A.data(), B.data(), expected.data(), M, N, K);

    test::DeviceArray d_A(A), d_B(B), d_C(size_t(M) * N);
    launch_gemm_naive(d_A.data(), d_B.data(), d_C.data(), M, N, K);
    auto actual = d_C.to_host();

    // Error accumulates over K additions, so the tolerance is looser than for vector_add
    bool ok = test::all_close(actual, expected, 1e-4f, 1e-4f);
    std::printf("gemm M=%-4d N=%-4d K=%-4d %s\n", M, N, K, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= check(1, 1, 1);
    ok &= check(16, 16, 16);   // exactly one block
    ok &= check(17, 33, 9);    // not a multiple of the block size
    ok &= check(1, 512, 256);  // M = 1 (a single row)
    ok &= check(128, 64, 256);
    ok &= check(300, 200, 500);
    return ok ? 0 : 1;
}
