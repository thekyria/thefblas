add_executable(thefblas_bench bench_blas.cpp)
target_link_libraries(thefblas_bench PRIVATE thefblas::thefblas)

# A single pass over every benchmark, so that the code is exercised in CI
# without timing anything.
if(THEFBLAS_BUILD_TESTS)
  add_test(NAME thefblas_bench_smoke COMMAND thefblas_bench --quick)
endif()
