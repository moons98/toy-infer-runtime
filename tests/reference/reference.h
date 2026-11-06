#pragma once

// CPU implementations that compute the same thing as the GPU kernels.
// They are written as the simplest possible loops so we can trust them to be correct.
// Speed does not matter here.

namespace tir::ref {

void vector_add(const float* a, const float* b, float* c, int n);

}  // namespace tir::ref
