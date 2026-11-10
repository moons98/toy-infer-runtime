#include "tir/tensor.h"

#include <stdexcept>

#include "tir/cuda_check.h"

namespace tir {

int64_t numel(const Shape& shape) {
    int64_t n = 1;
    for (int d : shape) {
        n *= d;
    }
    return n;
}

std::string to_string(const Shape& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += (i ? "," : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

// Row-major strides: the last dimension moves by 1, each earlier one by the product of the
// dimensions after it.
static Shape contiguous_stride(const Shape& shape) {
    Shape stride(shape.size());
    int s = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        stride[i] = s;
        s *= shape[i];
    }
    return stride;
}

Tensor Tensor::empty(const Shape& shape, Allocator& allocator) {
    for (int d : shape) {
        if (d <= 0) {
            throw std::invalid_argument("Tensor::empty: bad shape " + to_string(shape));
        }
    }
    Tensor t;
    t.shape_ = shape;
    t.stride_ = contiguous_stride(shape);
    t.buffer_ = std::make_shared<Buffer>(tir::numel(shape) * sizeof(float), allocator);
    return t;
}

Tensor Tensor::from_host(const std::vector<float>& data, const Shape& shape, Allocator& allocator) {
    if (static_cast<int64_t>(data.size()) != tir::numel(shape)) {
        throw std::invalid_argument("Tensor::from_host: " + std::to_string(data.size()) +
                                    " values for shape " + to_string(shape));
    }
    Tensor t = empty(shape, allocator);
    TIR_CUDA_CHECK(cudaMemcpy(t.data(), data.data(), t.bytes(), cudaMemcpyHostToDevice));
    return t;
}

std::vector<float> Tensor::to_host() const {
    std::vector<float> host(numel());
    TIR_CUDA_CHECK(cudaMemcpy(host.data(), data(), bytes(), cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace tir
