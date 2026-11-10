#pragma once

// One register_xxx function per op, each defined in src/ops/xxx.cpp.
// register_all_ops() (register_all.cpp) calls them all.

#include "tir/op.h"

namespace tir {

void register_gemm(OpRegistry& registry);
void register_conv2d(OpRegistry& registry);

}  // namespace tir
