add_library(tir_kernels STATIC
    kernels/vector_add.cu
    kernels/gemm_naive.cu
)
target_include_directories(tir_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tir_kernels PUBLIC CUDA::cudart)
