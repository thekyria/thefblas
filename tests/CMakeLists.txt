add_executable(thefblas_tests test_fixed.cpp)
target_link_libraries(thefblas_tests PRIVATE thefblas::thefblas)

add_test(NAME thefblas_tests COMMAND thefblas_tests)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_level1.cpp)
  add_executable(thefblas_test_level1 test_level1.cpp)
  target_link_libraries(thefblas_test_level1 PRIVATE thefblas::thefblas)
  add_test(NAME thefblas_test_level1 COMMAND thefblas_test_level1)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_level2.cpp)
  add_executable(thefblas_test_level2 test_level2.cpp)
  target_link_libraries(thefblas_test_level2 PRIVATE thefblas::thefblas)
  add_test(NAME thefblas_test_level2 COMMAND thefblas_test_level2)
endif()
