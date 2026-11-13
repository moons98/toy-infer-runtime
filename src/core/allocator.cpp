#include "tir/allocator.h"

#include <algorithm>

#include "tir/cuda_check.h"

namespace tir {

void* CudaAllocator::allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    TIR_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    sizes_[ptr] = bytes;
    stats_.num_allocs += 1;
    stats_.bytes_in_use += bytes;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.bytes_in_use);
    return ptr;
}

void CudaAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    // cudaFree waits for all work already queued on the GPU to finish before it returns.
    // That hidden synchronization is one reason runtimes avoid freeing memory inside a model run.
    TIR_CUDA_CHECK(cudaFree(ptr));
    stats_.num_frees += 1;
    stats_.bytes_in_use -= sizes_.at(ptr);
    sizes_.erase(ptr);
}

void CudaAllocator::reset_stats() {
    stats_.num_allocs = 0;
    stats_.num_frees = 0;
    stats_.peak_bytes = stats_.bytes_in_use;
}

CudaAllocator& default_allocator() {
    // Created on first use and shared by the whole program (singleton)
    static CudaAllocator allocator;
    return allocator;
}

Buffer::Buffer(size_t bytes, Allocator& allocator)
    : ptr_(allocator.allocate(bytes)), bytes_(bytes), allocator_(&allocator) {}

Buffer::~Buffer() {
    allocator_->deallocate(ptr_);
}

}  // namespace tir
