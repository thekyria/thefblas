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
option(THEFBLAS_BUILD_EXAMPLES "Build thefblas examples"
       ${PROJECT_IS_TOP_LEVEL})
option(THEFBLAS_ENABLE_STRICT_WARNINGS "Enable strict compiler warnings" ON)
option(THEFBLAS_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(THEFBLAS_ENABLE_SANITIZERS
       "Enable ASan/UBSan for Debug builds (Clang/GCC)" OFF)
option(THEFBLAS_ENABLE_COVERAGE
       "Enable gcov/lcov code coverage instrumentation (GCC/Clang Debug only)"
       OFF)
option(THEFBLAS_ENABLE_IPO "Enable IPO/LTO for release builds when supported"
       ON)
option(THEFBLAS_ENABLE_RELEASE_HARDENING
       "Enable additional hardening for release configurations on Clang/GCC" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIPOSupported)

# --- The header-only interface library -------------------------------------
add_library(thefblas INTERFACE)
add_library(thefblas::thefblas ALIAS thefblas)

target_include_directories(
  thefblas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(thefblas INTERFACE cxx_std_17)

# --- Internal build options for tests/examples -----------------------------
# The public interface library MUST NOT carry warning/sanitizer/coverage flags
# (that would poison consumers). Instead, collect them in a private INTERFACE
# target that is linked into the project's own executables (tests, examples)
# via a directory-scoped link_libraries() call below. This target is never
# installed or exported.
add_library(thefblas_build_options INTERFACE)

if(THEFBLAS_ENABLE_STRICT_WARNINGS)
  if(MSVC)
    target_compile_options(
      thefblas_build_options
      INTERFACE /W4
                /permissive-
                /w14242
                /w14254
                /w14263
                /w14265
                /w14287
                /we4289
                /w14296
                /w14311
                /w14545
                /w14546
                /w14547
                /w14549
                /w14555
                /w14619
                /w14640
                /w14826
                /w14905
                /w14906
                /w14928)
    if(THEFBLAS_WARNINGS_AS_ERRORS)
      target_compile_options(thefblas_build_options INTERFACE /WX)
    endif()
  else()
    target_compile_options(
      thefblas_build_options
      INTERFACE -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
                -Wshadow
                -Wnon-virtual-dtor
                -Wold-style-cast
                -Woverloaded-virtual
                -Wnull-dereference
                -Wdouble-promotion
                -Wformat=2)
    if(THEFBLAS_WARNINGS_AS_ERRORS)
      target_compile_options(thefblas_build_options INTERFACE -Werror)
    endif()
  endif()
endif()

if(THEFBLAS_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      thefblas_build_options
      INTERFACE -fno-omit-frame-pointer
                $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(
      thefblas_build_options INTERFACE -fno-omit-frame-pointer
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  else()
    message(
      WARNING "THEFBLAS_ENABLE_SANITIZERS is ON, but sanitizers are configured"
              " only for Clang/GNU.")
  endif()
endif()

if(THEFBLAS_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(thefblas_build_options INTERFACE --coverage -O0 -g)
    target_link_options(thefblas_build_options INTERFACE --coverage)
  else()
    message(WARNING "THEFBLAS_ENABLE_COVERAGE is ON, but coverage is only"
                    " supported for GCC/Clang.")
  endif()
endif()

if(THEFBLAS_ENABLE_RELEASE_HARDENING AND CMAKE_CXX_COMPILER_ID MATCHES
                                         "Clang|GNU")
  target_compile_definitions(
    thefblas_build_options
    INTERFACE $<$<CONFIG:Release>:_FORTIFY_SOURCE=3>
              $<$<CONFIG:RelWithDebInfo>:_FORTIFY_SOURCE=3>
              $<$<CONFIG:MinSizeRel>:_FORTIFY_SOURCE=3>)
  target_compile_options(
    thefblas_build_options
    INTERFACE $<$<CONFIG:Release>:-fstack-protector-strong>
              $<$<CONFIG:RelWithDebInfo>:-fstack-protector-strong>
              $<$<CONFIG:MinSizeRel>:-fstack-protector-strong>)
endif()

if(THEFBLAS_ENABLE_IPO)
  set(THEFBLAS_ENABLE_IPO_FOR_TARGET TRUE)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)
    find_program(THEFBLAS_LLD NAMES ld.lld lld)
    if(NOT THEFBLAS_LLD)
      set(THEFBLAS_ENABLE_IPO_FOR_TARGET FALSE)
      message(STATUS "IPO/LTO disabled for Clang: no compatible LLVM linker"
                     " (ld.lld) was found.")
    endif()
  endif()

  if(THEFBLAS_ENABLE_IPO_FOR_TARGET)
    check_ipo_supported(RESULT THEFBLAS_IPO_SUPPORTED OUTPUT THEFBLAS_IPO_ERROR)
  else()
    set(THEFBLAS_IPO_SUPPORTED FALSE)
  endif()

  if(THEFBLAS_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
  elseif(THEFBLAS_IPO_ERROR)
    message(STATUS "IPO/LTO not enabled: ${THEFBLAS_IPO_ERROR}")
  endif()
endif()

# --- Install / export -------------------------------------------------------
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS thefblas EXPORT thefblasTargets)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/thefblasConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/thefblasConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/thefblasConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/thefblas)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/thefblasConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/thefblasConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/thefblas)

install(
  EXPORT thefblasTargets
  FILE thefblasTargets.cmake
  NAMESPACE thefblas::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/thefblas)

# Link the internal build options into every executable defined in the
# subdirectories below (tests and examples) without modifying their
# CMakeLists.txt files.
link_libraries(thefblas_build_options)

if(THEFBLAS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(THEFBLAS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
