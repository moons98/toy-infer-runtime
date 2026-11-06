# CPU reference implementations, shared by tests and benchmarks.
add_library(tir_reference STATIC
    reference/reference.cpp
)
target_include_directories(tir_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# One test = one small program with its own main().
# It returns 0 if the results match and 1 otherwise; ctest uses the exit code as pass/fail.
function(tir_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE tir_kernels tir_reference)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

tir_add_test(test_vector_add)
tir_add_test(test_gemm)
