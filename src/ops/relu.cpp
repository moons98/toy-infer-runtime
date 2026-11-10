// relu: y = max(x, 0), elementwise. Output has the input's shape.

#include "ops.h"
#include "tir/kernels.h"

namespace tir {

void register_relu(OpRegistry& registry) {
    OpDef op;
    op.name = "relu";
    op.num_inputs = 1;
    op.infer_shape = [](const std::vector<Shape>& in, const Attrs&) { return in[0]; };
    op.launch = [](const std::vector<Tensor>& in, Tensor& out, const Attrs&, ExecContext& ctx) {
        launch_relu(in[0].data(), out.data(), static_cast<int>(in[0].numel()), ctx.stream);
    };
    registry.add(op);
}

}  // namespace tir
