# The tests check results with tests/test_support.hpp, which does not depend on
# assert(). The library's own assertions (the `checked` overflow policy) are
# kept active in every configuration, including Release presets that define
# NDEBUG, because some tests rely on them being enabled.
add_compile_options($<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>)
# Bit-identity checks against reference Netlib-order loops need the compiler not
# to fuse multiplies and adds differently in the two (GCC's GNU-mode default).
add_compile_options($<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)

add_executable(thefblas_tests test_fixed.cpp)
target_link_libraries(thefblas_tests PRIVATE thefblas::thefblas)

add_test(NAME thefblas_tests COMMAND thefblas_tests)

add_executable(thefblas_test_coverage_edges test_coverage_edges.cpp)
target_link_libraries(thefblas_test_coverage_edges PRIVATE thefblas::thefblas)
add_test(NAME thefblas_test_coverage_edges COMMAND thefblas_test_coverage_edges)

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

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_level2_banded.cpp)
  add_executable(thefblas_test_level2_banded test_level2_banded.cpp)
  target_link_libraries(thefblas_test_level2_banded PRIVATE thefblas::thefblas)
  add_test(NAME thefblas_test_level2_banded COMMAND thefblas_test_level2_banded)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_level2_packed.cpp)
  add_executable(thefblas_test_level2_packed test_level2_packed.cpp)
  target_link_libraries(thefblas_test_level2_packed PRIVATE thefblas::thefblas)
  add_test(NAME thefblas_test_level2_packed COMMAND thefblas_test_level2_packed)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_accumulators.cpp)
  add_executable(thefblas_test_accumulators test_accumulators.cpp)
  target_link_libraries(thefblas_test_accumulators PRIVATE thefblas::thefblas)
  add_test(NAME thefblas_test_accumulators COMMAND thefblas_test_accumulators)
endif()

# The same checks with compiler extensions disabled (-std=c++17 rather than
# -std=gnu++17), where the standard traits do not recognise __int128.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_accumulators.cpp)
  add_executable(thefblas_test_accumulators_strict test_accumulators.cpp)
  target_link_libraries(thefblas_test_accumulators_strict PRIVATE thefblas::thefblas)
  set_target_properties(thefblas_test_accumulators_strict PROPERTIES CXX_EXTENSIONS OFF)
  add_test(NAME thefblas_test_accumulators_strict COMMAND thefblas_test_accumulators_strict)
endif()

# The release-build (NDEBUG) fallback of the checked policy; the source defines
# NDEBUG itself, overriding the -UNDEBUG above for this target only.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_checked_release.cpp)
  add_executable(thefblas_test_checked_release test_checked_release.cpp)
  target_link_libraries(thefblas_test_checked_release PRIVATE thefblas::thefblas)
  add_test(NAME thefblas_test_checked_release COMMAND thefblas_test_checked_release)
endif()

# The test helper itself, also without GNU extensions (see above).
add_executable(thefblas_test_support test_support_selftest.cpp)
target_link_libraries(thefblas_test_support PRIVATE thefblas::thefblas)
add_test(NAME thefblas_test_support COMMAND thefblas_test_support)
add_executable(thefblas_test_support_strict test_support_selftest.cpp)
target_link_libraries(thefblas_test_support_strict PRIVATE thefblas::thefblas)
set_target_properties(thefblas_test_support_strict PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME thefblas_test_support_strict COMMAND thefblas_test_support_strict)

add_executable(thefblas_test_api test_api.cpp)
target_link_libraries(thefblas_test_api PRIVATE thefblas::thefblas)
add_test(NAME thefblas_test_api COMMAND thefblas_test_api)

# Randomised comparison against a reference CBLAS (Netlib, OpenBLAS, ...). Opt-in
# because it needs a system BLAS that provides the CBLAS interface and cblas.h.
if(THEFBLAS_ENABLE_REFERENCE_TESTS)
  find_package(BLAS REQUIRED)
  find_path(THEFBLAS_CBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES openblas REQUIRED)
  add_executable(thefblas_test_reference_blas test_reference_blas.cpp)
  target_include_directories(thefblas_test_reference_blas SYSTEM
                             PRIVATE ${THEFBLAS_CBLAS_INCLUDE_DIR})
  target_link_libraries(thefblas_test_reference_blas PRIVATE thefblas::thefblas BLAS::BLAS)
  add_test(NAME thefblas_test_reference_blas COMMAND thefblas_test_reference_blas)
endif()

add_executable(thefblas_test_level3 test_level3.cpp)
target_link_libraries(thefblas_test_level3 PRIVATE thefblas::thefblas)
add_test(NAME thefblas_test_level3 COMMAND thefblas_test_level3)
