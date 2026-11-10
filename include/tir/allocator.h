#pragma once

// Device memory allocation.
//
// Every device buffer in the runtime goes through an Allocator. The allocation
// strategy can then be swapped (memory planning replaces one cudaMalloc per tensor with an
// arena) without touching Tensor or the ops.

#include <cstddef>
#include <unordered_map>

namespace tir {

class Allocator {
  public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
};

// Calls cudaMalloc/cudaFree for every request and keeps simple statistics,
// so we can see how many allocations a program makes and how much memory it holds.
class CudaAllocator : public Allocator {
  public:
    struct Stats {
        size_t num_allocs = 0;    // cudaMalloc calls
        size_t num_frees = 0;     // cudaFree calls
        size_t bytes_in_use = 0;  // currently allocated
        size_t peak_bytes = 0;    // largest bytes_in_use seen so far
    };

    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;

    const Stats& stats() const { return stats_; }
    // Zero the counters and restart peak tracking from the current usage.
    void reset_stats();

  private:
    Stats stats_;
    std::unordered_map<void*, size_t> sizes_;  // size of each live allocation, for the stats
};

// The allocator used when none is given explicitly.
CudaAllocator& default_allocator();

// A block of device memory that returns itself to its allocator when destroyed (RAII).
class Buffer {
  public:
    Buffer(size_t bytes, Allocator& allocator);
    ~Buffer();

    // Copying would free the same memory twice, so it is disabled. (Buffer b2 = b1)
    // Sharing one buffer between tensors is done with std::shared_ptr<Buffer> instead. (b2 = b1)
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* data() const { return ptr_; }
    size_t bytes() const { return bytes_; }

  private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    Allocator* allocator_ = nullptr;
};

}  // namespace tir
