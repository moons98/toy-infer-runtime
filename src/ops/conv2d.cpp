// conv2d: x[N, C, H, W] * w[K, C, R, S] -> y[N, K, P, Q]
// attrs: pad (default 0), stride (default 1)

#include <stdexcept>

#include "ops.h"
#include "tir/kernels.h"

namespace tir {

void register_conv2d(OpRegistry& registry) {
    OpDef op;
    op.name = "conv2d";
    op.num_inputs = 2;

    op.infer_shape = [](const std::vector<Shape>& in, const Attrs& attrs) {
        const Shape& x = in[0];
        const Shape& w = in[1];
        if (x.size() != 4 || w.size() != 4 || x[1] != w[1]) {
            throw std::invalid_argument("conv2d: expected x[N, C, H, W] and w[K, C, R, S], got " + to_string(x) +
                                        " and " + to_string(w));
        }
        int pad = get_attr(attrs, "pad", 0);
        int stride = get_attr(attrs, "stride", 1);
        int P = conv2d_out_size(x[2], w[2], pad, stride);
        int Q = conv2d_out_size(x[3], w[3], pad, stride);
        if (P <= 0 || Q <= 0) {
            throw std::invalid_argument("conv2d: filter larger than padded input");
        }
        return Shape{x[0], w[0], P, Q};
    };

    op.launch = [](const std::vector<Tensor>& in, Tensor& out, const Attrs& attrs, ExecContext& ctx) {
        const Tensor& x = in[0];
        const Tensor& w = in[1];
        launch_conv2d_naive(x.data(), w.data(), out.data(), x.dim(0), x.dim(1), x.dim(2), x.dim(3), w.dim(0), w.dim(2),
                            w.dim(3), get_attr(attrs, "pad", 0), get_attr(attrs, "stride", 1), ctx.stream);
    };

    registry.add(op);
}

}  // namespace tir
