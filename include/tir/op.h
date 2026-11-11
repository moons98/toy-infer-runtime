#pragma once

// Ops and the op registry.
//
// An OpDef describes one kind of operation ("gemm", "relu", ...): how to compute its output
// shape and how to launch it. The registry maps each op name to its OpDef. run_op() looks an
// op up by name, allocates the output, and launches it right away (eager execution).
//
//   run_op("relu", {x})
//     -> registry().get("relu")          find the OpDef by name
//     -> infer_shape({x.shape()}, attrs)  output shape
//     -> Tensor::empty(shape)             allocate the output
//     -> launch({x}, out, attrs, ctx)     enqueue the kernel (asynchronous)

#include <cuda_runtime.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tir/tensor.h"

namespace tir {

// Integer settings of an op, by name. e.g. conv2d: {{"pad", 1}, {"stride", 1}}
using Attrs = std::map<std::string, int>;

// Value of attribute `name`, or `default_value` if it is not set.
int get_attr(const Attrs& attrs, const std::string& name, int default_value);

// Where and how ops run. For now everything uses the default stream and allocator; later the
// graph executor passes its own stream and memory planning passes an arena allocator.
struct ExecContext {
    cudaStream_t stream = 0;
    Allocator* allocator = &default_allocator();
};
ExecContext& default_context();

struct OpDef {
    std::string name;
    int num_inputs = 0;

    // Output shape from the input shapes and attrs. Throws std::invalid_argument if the inputs
    // do not fit (e.g. mismatched gemm inner dimensions). Takes shapes, not tensors, so it can
    // also run before any memory exists (the graph executor needs that).
    std::function<Shape(const std::vector<Shape>& inputs, const Attrs& attrs)> infer_shape;

    // Launch the kernel(s) that read `inputs` and write the already allocated `output`.
    std::function<void(const std::vector<Tensor>& inputs, Tensor& output, const Attrs& attrs, ExecContext& ctx)> launch;
};

class OpRegistry {
  public:
    void add(OpDef op);                               // throws if the name is taken
    const OpDef& get(const std::string& name) const;  // throws if the name is unknown
    std::vector<std::string> names() const;

  private:
    std::map<std::string, OpDef> ops_;
};

// The program-wide registry. On first use it is filled by register_all_ops().
OpRegistry& registry();

// Registers every built-in op. Defined in src/ops/register_all.cpp.
void register_all_ops(OpRegistry& registry);

// Run one op now: look it up, check inputs, allocate the output, launch.
// Returns as soon as the work is queued; the output is ready once the GPU gets there.
Tensor run_op(const std::string& name, const std::vector<Tensor>& inputs, const Attrs& attrs = {},
              ExecContext& ctx = default_context());

}  // namespace tir
