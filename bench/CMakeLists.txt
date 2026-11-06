add_executable(device_info device_info.cpp)
target_link_libraries(device_info PRIVATE CUDA::cudart)
