cmake_minimum_required(VERSION 3.22)

# GPU architecture: defaults to RTX PRO 4000 Blackwell (sm_120).
# For other GPUs, pass e.g. -DCMAKE_CUDA_ARCHITECTURES=86.
# Must be set before project() to take effect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(toy_infer_runtime LANGUAGES CXX CUDA)

# CUDA::cudart lets plain C++ (.cpp) targets call the CUDA runtime API
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -lineinfo: lets Nsight Compute map kernel metrics to source lines (no performance cost)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(bench)

# tir: the main program. What it runs is configured at the top of main.cpp.
add_executable(tir main.cpp)
target_link_libraries(tir PRIVATE tir_kernels)
