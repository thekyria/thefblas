add_executable(thefblas_example_fixed_point_ops fixed_point_ops.cpp)
target_link_libraries(thefblas_example_fixed_point_ops PRIVATE thefblas::thefblas)

# Embedded-style example: Q1.15, no iostream, checks itself through its exit
# status. On bare-metal GCC builds unused code is dropped at link time, so its
# size (reported by CI for each Cortex-M target) reflects the code it uses.
add_executable(thefblas_example_embedded_q15 embedded_q15.cpp)
target_link_libraries(thefblas_example_embedded_q15 PRIVATE thefblas::thefblas)
if(CMAKE_SYSTEM_NAME STREQUAL "Generic" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(thefblas_example_embedded_q15 PRIVATE -ffunction-sections
                                                               -fdata-sections)
  target_link_options(thefblas_example_embedded_q15 PRIVATE -Wl,--gc-sections)
endif()

if(THEFBLAS_BUILD_TESTS)
  add_test(NAME thefblas_example_embedded_q15 COMMAND thefblas_example_embedded_q15)
endif()
