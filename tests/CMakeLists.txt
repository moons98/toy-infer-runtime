# CPU reference implementations, shared by tests and benchmarks.
add_library(tir_reference STATIC
    reference/reference.cpp
)
# Also puts tests/ on the include path, so tests in subfolders can include "common.h"
target_include_directories(tir_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# One test = one small program with its own main().
# It returns 0 if the results match and 1 otherwise; ctest uses the exit code as pass/fail.
# Tests live in folders that mirror src/ (kernels, core, ops). The folder name is also the
# test's label, so one group can be run alone: ctest --test-dir build -L kernels
function(tir_add_test dir name)
    add_executable(${name} ${dir}/${name}.cpp)
    target_link_libraries(${name} PRIVATE tir_core tir_reference)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS ${dir})
endfunction()

# Kernels called directly with raw pointers, checked against the CPU reference
tir_add_test(kernels test_vector_add)
tir_add_test(kernels test_gemm)
tir_add_test(kernels test_conv2d)
tir_add_test(kernels test_relu)
tir_add_test(kernels test_batchnorm)

# Runtime core: allocator, tensor, op registry
tir_add_test(core test_allocator)
tir_add_test(core test_tensor)
tir_add_test(core test_registry)

# Ops run through run_op: shape inference, attrs, kernel arguments
tir_add_test(ops test_ops)
