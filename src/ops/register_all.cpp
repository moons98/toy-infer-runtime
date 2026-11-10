#include "tir/op.h"

namespace tir {

// Adding an op: write src/ops/<op>.cpp with a register_<op>() function, declare it in ops.h,
// call it here, and add the file to src/CMakeLists.txt.
void register_all_ops(OpRegistry& registry) {
    (void)registry;
}

}  // namespace tir
