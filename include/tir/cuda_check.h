#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// CUDA API calls do not throw on failure; they only return an error code.
// If it goes unchecked, the error passes silently and shows up much later somewhere unrelated.
// So every CUDA API call is wrapped in this macro, which prints the location and exits on failure.
#define TIR_CUDA_CHECK(call)                                                                        \
    do {                                                                                            \
        cudaError_t err_ = (call);                                                                  \
        if (err_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error at %s:%d\n  %s\n  -> %s\n", __FILE__, __LINE__, #call, \
                         cudaGetErrorString(err_));                                                 \
            std::exit(EXIT_FAILURE);                                                                \
        }                                                                                           \
    } while (0)
