// gemm: C[M, N] = A[M, K] * B[K, N]

#include <stdexcept>

#include "ops.h"
#include "tir/kernels.h"

namespace tir {

void register_gemm(OpRegistry& registry) {
    OpDef op;
    op.name = "gemm";
    op.num_inputs = 2;

    op.infer_shape = [](const std::vector<Shape>& in, const Attrs&) {
        const Shape& a = in[0];
        const Shape& b = in[1];
        if (a.size() != 2 || b.size() != 2 || a[1] != b[0]) {
            throw std::invalid_argument("gemm: expected [M, K] x [K, N], got " + to_string(a) + " x " + to_string(b));
        }
        return Shape{a[0], b[1]};
    };

    // The op's job is to unpack tensors into the raw pointers and sizes the kernel takes
    op.launch = [](const std::vector<Tensor>& in, Tensor& out, const Attrs&, ExecContext& ctx) {
        int M = in[0].dim(0), K = in[0].dim(1), N = in[1].dim(1);
        launch_gemm_naive(in[0].data(), in[1].data(), out.data(), M, N, K, ctx.stream);
    };

    registry.add(op);
}

}  // namespace tir
