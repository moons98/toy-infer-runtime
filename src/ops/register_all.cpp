#include "ops.h"

namespace tir {

// Adding an op: write src/ops/<op>.cpp with a register_<op>() function, declare it in ops.h,
// call it here, and add the file to src/CMakeLists.txt.
void register_all_ops(OpRegistry& registry) {
    register_gemm(registry);
    register_conv2d(registry);
    register_relu(registry);
}

}  // namespace tir
