cmake_minimum_required(VERSION 3.21)
project(test_thefblas_conan CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(thefblas CONFIG REQUIRED)

add_executable(test_thefblas_conan main.cpp)
target_link_libraries(test_thefblas_conan PRIVATE thefblas::thefblas)
