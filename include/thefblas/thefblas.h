// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

/**
 * @file thefblas.h
 * @brief Convenience umbrella header for the thefblas library.
 *
 * thefblas is a value-type-generic (templated), header-only C++17 BLAS-style
 * library. Unlike classic BLAS naming (s/d/c/z precision prefixes), every
 * routine is a single template parameterized on the value type `T`, which may
 * be `float`, `double`, `thefblas::fixed<IntType, FracBits>` (see fixed.hpp),
 * or `std::complex<T>` of any of the above.
 *
 * Including this header pulls in the fixed-point type plus all Level 1 and
 * Level 2 routines (dense, banded and packed storage).
 */

#include "thefblas/fixed.hpp"
#include "thefblas/level1.hpp"
#include "thefblas/level2.hpp"
#include "thefblas/level2_banded.hpp"
#include "thefblas/level2_packed.hpp"
