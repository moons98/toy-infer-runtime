#pragma once

// Tensor: a device buffer plus its shape.
//
// Without it, code carries a raw pointer and its sizes around separately and pairs every
// cudaMalloc with a cudaFree by hand. A Tensor keeps them together and frees the memory
// automatically.
//
// Limits (on purpose, to keep things small):
//   - fp32 only
//   - contiguous, row-major layout only (stride is derived from shape, never set by hand)

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tir/allocator.h"

namespace tir {

// Size of each dimension, outermost first. e.g. {N, C, H, W}
// Real runtimes use int64_t; int keeps the code short and matches our kernels' arguments.
using Shape = std::vector<int>;

int64_t numel(const Shape& shape);          // number of elements (product of dims)
std::string to_string(const Shape& shape);  // "[1,6,24,24]"

class Tensor {
  public:
    // An empty handle with no memory. Check with defined().
    Tensor() = default;

    // New tensor with uninitialized device memory.
    static Tensor empty(const Shape& shape, Allocator& allocator = default_allocator());
    // New tensor initialized with a copy of host data (host -> device copy).
    static Tensor from_host(const std::vector<float>& data, const Shape& shape,
                            Allocator& allocator = default_allocator());
    // Copy the contents back to the host. Waits for preceding GPU work, like cudaMemcpy.
    std::vector<float> to_host() const;

    bool defined() const { return buffer_ != nullptr; }
    const Shape& shape() const { return shape_; }
    // Elements to skip to move one step along each dimension. For [N, C, H, W] that is
    // [C*H*W, H*W, W, 1]: the last dimension is contiguous in memory.
    const Shape& stride() const { return stride_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int dim(int i) const { return shape_.at(i); }
    int64_t numel() const { return tir::numel(shape_); }
    size_t bytes() const { return numel() * sizeof(float); }

    float* data() { return static_cast<float*>(buffer_->data()); }
    const float* data() const { return static_cast<const float*>(buffer_->data()); }

  private:
    // shared_ptr: copying a Tensor copies the handle, not the memory. All copies point at the
    // same Buffer, and the Buffer is freed when the last copy goes away.
    std::shared_ptr<Buffer> buffer_;
    Shape shape_;
    Shape stride_;
};

}  // namespace tir
