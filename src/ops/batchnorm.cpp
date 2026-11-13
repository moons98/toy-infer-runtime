// batchnorm (inference): inputs x[N, C, ...], gamma[C], beta[C], mean[C], var[C]
//   y = (x - mean) / sqrt(var + eps) * gamma + beta, per channel (dimension 1)
// eps is fixed at 1e-5 (PyTorch's default) because attrs hold integers only.

#include <stdexcept>

#include "ops.h"
#include "tir/kernels.h"

namespace tir {

constexpr float kBatchNormEps = 1e-5f;

void register_batchnorm(OpRegistry& registry) {
    OpDef op;
    op.name = "batchnorm";
    op.num_inputs = 5;

    op.infer_shape = [](const std::vector<Shape>& in, const Attrs&) {
        const Shape& x = in[0];
        if (x.size() < 2) {
            throw std::invalid_argument("batchnorm: expected x[N, C, ...], got " + to_string(x));
        }
        for (int i = 1; i < 5; ++i) {
            if (in[i] != Shape{x[1]}) {
                throw std::invalid_argument("batchnorm: gamma, beta, mean, var must be [" + std::to_string(x[1]) +
                                            "], got " + to_string(in[i]));
            }
        }
        return x;
    };

    op.launch = [](const std::vector<Tensor>& in, Tensor& out, const Attrs&, ExecContext& ctx) {
        const Tensor& x = in[0];
        int inner = static_cast<int>(x.numel() / (x.dim(0) * x.dim(1)));
        launch_batchnorm(x.data(), in[1].data(), in[2].data(), in[3].data(), in[4].data(), out.data(),
                         static_cast<int>(x.numel()), x.dim(1), inner, kBatchNormEps, ctx.stream);
    };

    registry.add(op);
}

}  // namespace tir
