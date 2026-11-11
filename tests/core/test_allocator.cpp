#include <cstdio>

#include "tir/allocator.h"

using namespace tir;

int main() {
    CudaAllocator allocator;
    bool ok = true;

    {
        Buffer a(1000, allocator);
        Buffer b(500, allocator);
        const auto& s = allocator.stats();
        ok &= s.num_allocs == 2 && s.bytes_in_use == 1500 && s.peak_bytes == 1500;
        ok &= a.data() != nullptr && a.bytes() == 1000;
    }  // a and b are destroyed here, which frees their memory

    const auto& s = allocator.stats();
    ok &= s.num_frees == 2 && s.bytes_in_use == 0 && s.peak_bytes == 1500;

    allocator.reset_stats();
    ok &= allocator.stats().num_allocs == 0 && allocator.stats().peak_bytes == 0;

    std::printf("allocator stats and RAII release %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
