add_library(tir_kernels STATIC
    kernels/vector_add.cu
    kernels/gemm_naive.cu
    kernels/conv2d_naive.cu
)
target_include_directories(tir_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tir_kernels PUBLIC CUDA::cudart)

# Runtime core (plain C++): device memory, tensors, and the op registry
add_library(tir_core STATIC
    core/allocator.cpp
    core/tensor.cpp
)
target_link_libraries(tir_core PUBLIC tir_kernels)
