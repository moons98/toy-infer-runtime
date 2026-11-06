add_executable(device_info device_info.cpp)
target_link_libraries(device_info PRIVATE CUDA::cudart)

# Time the same kernel three ways (chrono only, chrono + sync, cudaEvent)
add_executable(timing_methods timing_methods.cpp)
target_link_libraries(timing_methods PRIVATE tir_kernels)
