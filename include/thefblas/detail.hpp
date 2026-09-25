// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include <complex>

namespace thefblas {

/**
 * @file detail.hpp
 * @brief Internal helpers shared by the Level 1 and Level 2 headers.
 *
 * These live in a single header so that the routine headers can be included
 * together (see thefblas.h) without redefining the helpers.
 */

namespace detail {

/// Index of the first processed element for a vector of length `n` and stride
/// `inc`; negative strides start at the far end, as in Netlib BLAS.
inline int start_index(int n, int inc) {
  return (inc > 0) ? 0 : (1 - n) * inc;
}

template <typename T>
inline T conj_value(const T& value) {
  return value;
}

template <typename T>
inline std::complex<T> conj_value(const std::complex<T>& value) {
  using std::conj;
  return conj(value);
}

}  // namespace detail

}  // namespace thefblas
