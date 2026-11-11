#include <cstdio>
#include <stdexcept>

#include "common.h"
#include "tir/cuda_check.h"
#include "tir/op.h"

using namespace tir;

static bool report(const char* name, bool ok) {
    std::printf("%-40s %s\n", name, ok ? "OK" : "FAIL");
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

int main() {
    // A test-only op: copies its input. It needs no kernel of its own.
    OpDef copy;
    copy.name = "test_copy";
    copy.num_inputs = 1;
    copy.infer_shape = [](const std::vector<Shape>& in, const Attrs&) { return in[0]; };
    copy.launch = [](const std::vector<Tensor>& in, Tensor& out, const Attrs&, ExecContext& ctx) {
        TIR_CUDA_CHECK(cudaMemcpyAsync(out.data(), in[0].data(), in[0].bytes(),
                                       cudaMemcpyDeviceToDevice, ctx.stream));
    };
    registry().add(copy);

    bool ok = true;
    auto host = test::random_vector(10, 1);
    Tensor x = Tensor::from_host(host, {2, 5});

    size_t allocs = default_allocator().stats().num_allocs;
    Tensor y = run_op("test_copy", {x});
    ok &= report("run_op finds and runs the op", y.shape() == x.shape() && y.to_host() == host);
    ok &=
        report("run_op allocates one output", default_allocator().stats().num_allocs == allocs + 1);

    ok &= report("unknown op is rejected", throws([&] { run_op("no_such_op", {x}); }));
    ok &= report("wrong input count is rejected", throws([&] { run_op("test_copy", {x, x}); }));
    ok &= report("duplicate name is rejected", throws([&] { registry().add(copy); }));

    Attrs attrs = {{"pad", 2}};
    ok &= report("get_attr with default",
                 get_attr(attrs, "pad", 0) == 2 && get_attr(attrs, "stride", 1) == 1);

    return ok ? 0 : 1;
}
