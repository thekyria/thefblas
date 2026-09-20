cmake_minimum_required(VERSION 3.21)

project(
  thefblas
  VERSION 0.1.0
  DESCRIPTION
    "Minimal, value-type-generic (templated) fixed-point C++17 BLAS-style library"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(THEFBLAS_BUILD_TESTS "Build thefblas unit tests" ${PROJECT_IS_TOP_LEVEL})
option(THEFBLAS_BUILD_EXAMPLES "Build thefblas examples" ${PROJECT_IS_TOP_LEVEL})

add_library(thefblas INTERFACE)
add_library(thefblas::thefblas ALIAS thefblas)

target_include_directories(
  thefblas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)

target_compile_features(thefblas INTERFACE cxx_std_17)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS thefblas EXPORT thefblasTargets)
install(
  EXPORT thefblasTargets
  FILE thefblasTargets.cmake
  NAMESPACE thefblas::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/thefblas)

if(THEFBLAS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(THEFBLAS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
