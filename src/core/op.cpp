#include "tir/op.h"

#include <stdexcept>

namespace tir {

int get_attr(const Attrs& attrs, const std::string& name, int default_value) {
    auto it = attrs.find(name);
    return it == attrs.end() ? default_value : it->second;
}

ExecContext& default_context() {
    static ExecContext ctx;
    return ctx;
}

void OpRegistry::add(OpDef op) {
    if (ops_.count(op.name)) {
        throw std::invalid_argument("op already registered: " + op.name);
    }
    std::string name = op.name;
    ops_.emplace(name, std::move(op));
}

const OpDef& OpRegistry::get(const std::string& name) const {
    auto it = ops_.find(name);
    if (it == ops_.end()) {
        throw std::invalid_argument("unknown op: " + name);
    }
    return it->second;
}

std::vector<std::string> OpRegistry::names() const {
    std::vector<std::string> out;
    for (const auto& [name, op] : ops_) {
        out.push_back(name);
    }
    return out;
}

OpRegistry& registry() {
    // Built and filled once, the first time anyone asks for it
    static OpRegistry r = [] {
        OpRegistry fresh;
        register_all_ops(fresh);
        return fresh;
    }();
    return r;
}

Tensor run_op(const std::string& name, const std::vector<Tensor>& inputs, const Attrs& attrs, ExecContext& ctx) {
    const OpDef& op = registry().get(name);
    if (static_cast<int>(inputs.size()) != op.num_inputs) {
        throw std::invalid_argument(name + ": expected " + std::to_string(op.num_inputs) + " inputs, got " +
                                    std::to_string(inputs.size()));
    }

    std::vector<Shape> input_shapes;
    for (const Tensor& t : inputs) {
        input_shapes.push_back(t.shape());
    }
    Shape out_shape = op.infer_shape(input_shapes, attrs);

    Tensor out = Tensor::empty(out_shape, *ctx.allocator);
    op.launch(inputs, out, attrs, ctx);
    return out;
}

}  // namespace tir
