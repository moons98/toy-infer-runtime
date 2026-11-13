#include <cstdio>
#include <stdexcept>

#include "common.h"
#include "tir/tensor.h"

using namespace tir;

static bool report(const char* name, bool ok) {
    std::printf("%-40s %s\n", name, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    auto& stats = default_allocator().stats();
    bool ok = true;

    // shape, stride, numel
    Tensor t = Tensor::empty({2, 3, 4, 5});
    ok &=
        report("shape / stride / numel", t.ndim() == 4 && t.numel() == 120 && t.bytes() == 480 &&
                                             t.stride() == Shape{60, 20, 5, 1} && to_string(t.shape()) == "[2,3,4,5]");

    // host -> device -> host keeps the data
    auto host = test::random_vector(6, 1);
    Tensor a = Tensor::from_host(host, {2, 3});
    ok &= report("from_host / to_host round trip", a.to_host() == host);

    // copying a Tensor shares the buffer instead of allocating
    size_t allocs_before = stats.num_allocs;
    Tensor b = a;
    ok &= report("copy shares the buffer", b.data() == a.data() && stats.num_allocs == allocs_before);

    // the buffer is freed only when the last copy goes away
    size_t in_use = stats.bytes_in_use;
    {
        Tensor c = Tensor::empty({256});
        Tensor d = c;
    }
    ok &= report("freed when the last copy is gone", stats.bytes_in_use == in_use);

    // wrong number of values is rejected
    bool threw = false;
    try {
        Tensor::from_host({1.0f, 2.0f}, {3});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ok &= report("from_host rejects a size mismatch", threw);

    return ok ? 0 : 1;
}
