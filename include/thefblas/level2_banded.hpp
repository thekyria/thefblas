// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/fixed.hpp"
#include "thefblas/level2.hpp"

#include <complex>

namespace thefblas {

/**
 * @file level2_banded.hpp
 * @brief Header-only generic Level-2 BLAS-style banded matrix-vector routines.
 *
 * Banded matrices use the classic BLAS band storage in column-major order.
 *
 * For a general band matrix with `kl` sub-diagonals and `ku` super-diagonals,
 * the element `a(i, j)` is stored at `ab[(ku + i - j) + j * ldab]` and is only
 * referenced for `max(0, j - ku) <= i <= min(m - 1, j + kl)`; `ldab` must be at
 * least `kl + ku + 1`.
 *
 * For a symmetric/Hermitian/triangular band matrix with `k` off-diagonals:
 * - `uplo == 'U'`: `a(i, j)` is stored at `ab[(k + i - j) + j * ldab]` for
 *   `max(0, j - k) <= i <= j`.
 * - `uplo == 'L'`: `a(i, j)` is stored at `ab[(i - j) + j * ldab]` for
 *   `j <= i <= min(n - 1, j + k)`.
 *
 * `ldab` must be at least `k + 1`. All other parameter conventions (`trans`,
 * `uplo`, `diag`, `incx`, `incy`) match level2.hpp, as does the early-return
 * behavior on invalid character parameters, non-positive sizes, negative
 * bandwidths or zero strides.
 */

namespace detail {

template <typename T>
inline void gbmv_impl(char trans, int m, int n, int kl, int ku, T alpha, const T* a, int lda,
                      const T* x, int incx, T beta, T* y, int incy) {
  const char tr = to_upper(trans);
  if (!valid_trans(trans) || m <= 0 || n <= 0 || kl < 0 || ku < 0 || lda < kl + ku + 1 ||
      incx == 0 || incy == 0) {
    return;
  }

  const int leny = (tr == 'N') ? m : n;
  const int lenx = (tr == 'N') ? n : m;
  scale_vector(leny, beta, y, incy);
  if (alpha == value_constants<T>::zero()) {
    return;
  }

  if (tr == 'N') {
    int jx = start_index(lenx, incx);
    int ky = start_index(leny, incy);
    for (int j = 0; j < n; ++j) {
      const T temp = alpha * x[jx];
      const int first = (j - ku > 0) ? (j - ku) : 0;
      const int last = (j + kl < m - 1) ? (j + kl) : (m - 1);
      int iy = ky;
      for (int i = first; i <= last; ++i) {
        y[iy] += temp * a[(ku + i - j) + j * lda];
        iy += incy;
      }
      jx += incx;
      if (j >= ku) {
        ky += incy;
      }
    }
  } else {
    const bool conjugate = (tr == 'C');
    int jy = start_index(leny, incy);
    int kx = start_index(lenx, incx);
    for (int j = 0; j < n; ++j) {
      accumulator_t<T> temp = value_constants<accumulator_t<T>>::zero();
      const int first = (j - ku > 0) ? (j - ku) : 0;
      const int last = (j + kl < m - 1) ? (j + kl) : (m - 1);
      int ix = kx;
      for (int i = first; i <= last; ++i) {
        const T value = a[(ku + i - j) + j * lda];
        temp += acc_mul(conjugate ? conj_value(value) : value, x[ix]);
        ix += incx;
      }
      y[jy] += alpha * from_accumulator<T>(temp);
      jy += incy;
      if (j >= ku) {
        kx += incx;
      }
    }
  }
}

template <typename T>
inline void sbmv_impl(char uplo, int n, int k, T alpha, const T* a, int lda, const T* x,
                      int incx, T beta, T* y, int incy) {
  const char ul = to_upper(uplo);
  if (!valid_uplo(uplo) || n <= 0 || k < 0 || lda < k + 1 || incx == 0 || incy == 0) {
    return;
  }

  scale_vector(n, beta, y, incy);
  if (alpha == value_constants<T>::zero()) {
    return;
  }

  if (ul == 'U') {
    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    int kx = jx;
    int ky = jy;
    for (int j = 0; j < n; ++j) {
      const T temp1 = alpha * x[jx];
      accumulator_t<T> temp2 = value_constants<accumulator_t<T>>::zero();
      const int first = (j - k > 0) ? (j - k) : 0;
      int ix = kx;
      int iy = ky;
      for (int i = first; i < j; ++i) {
        const T value = a[(k + i - j) + j * lda];
        y[iy] += temp1 * value;
        temp2 += acc_mul(value, x[ix]);
        ix += incx;
        iy += incy;
      }
      y[jy] += temp1 * a[k + j * lda] + alpha * from_accumulator<T>(temp2);
      jx += incx;
      jy += incy;
      if (j >= k) {
        kx += incx;
        ky += incy;
      }
    }
  } else {
    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
      const T temp1 = alpha * x[jx];
      accumulator_t<T> temp2 = value_constants<accumulator_t<T>>::zero();
      y[jy] += temp1 * a[j * lda];
      const int last = (j + k < n - 1) ? (j + k) : (n - 1);
      int ix = jx;
      int iy = jy;
      for (int i = j + 1; i <= last; ++i) {
        ix += incx;
        iy += incy;
        const T value = a[(i - j) + j * lda];
        y[iy] += temp1 * value;
        temp2 += acc_mul(value, x[ix]);
      }
      y[jy] += alpha * from_accumulator<T>(temp2);
      jx += incx;
      jy += incy;
    }
  }
}

template <typename T>
inline void hbmv_impl(char uplo, int n, int k, std::complex<T> alpha, const std::complex<T>* a,
                      int lda, const std::complex<T>* x, int incx, std::complex<T> beta,
                      std::complex<T>* y, int incy) {
  using C = std::complex<T>;
  const char ul = to_upper(uplo);
  if (!valid_uplo(uplo) || n <= 0 || k < 0 || lda < k + 1 || incx == 0 || incy == 0) {
    return;
  }

  scale_vector(n, beta, y, incy);
  if (alpha == value_constants<C>::zero()) {
    return;
  }

  if (ul == 'U') {
    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    int kx = jx;
    int ky = jy;
    for (int j = 0; j < n; ++j) {
      const C temp1 = alpha * x[jx];
      accumulator_t<C> temp2 = value_constants<accumulator_t<C>>::zero();
      const int first = (j - k > 0) ? (j - k) : 0;
      int ix = kx;
      int iy = ky;
      for (int i = first; i < j; ++i) {
        const C value = a[(k + i - j) + j * lda];
        y[iy] += temp1 * value;
        temp2 += acc_mul(conj_value(value), x[ix]);
        ix += incx;
        iy += incy;
      }
      y[jy] += temp1 * C(a[k + j * lda].real(), T{}) + alpha * from_accumulator<C>(temp2);
      jx += incx;
      jy += incy;
      if (j >= k) {
        kx += incx;
        ky += incy;
      }
    }
  } else {
    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
      const C temp1 = alpha * x[jx];
      accumulator_t<C> temp2 = value_constants<accumulator_t<C>>::zero();
      y[jy] += temp1 * C(a[j * lda].real(), T{});
      const int last = (j + k < n - 1) ? (j + k) : (n - 1);
      int ix = jx;
      int iy = jy;
      for (int i = j + 1; i <= last; ++i) {
        ix += incx;
        iy += incy;
        const C value = a[(i - j) + j * lda];
        y[iy] += temp1 * value;
        temp2 += acc_mul(conj_value(value), x[ix]);
      }
      y[jy] += alpha * from_accumulator<C>(temp2);
      jx += incx;
      jy += incy;
    }
  }
}

template <typename T>
inline void tbmv_impl(char uplo, char trans, char diag, int n, int k, const T* a, int lda,
                      T* x, int incx) {
  const char ul = to_upper(uplo);
  const char tr = to_upper(trans);
  const char dg = to_upper(diag);
  if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || k < 0 ||
      lda < k + 1 || incx == 0) {
    return;
  }
  const bool unit = (dg == 'U');
  const bool conjugate = (tr == 'C');
  const bool upper = (ul == 'U');

  // Offset of the diagonal within a band column, and the band row of a(i, j).
  const int diag_row = upper ? k : 0;
  const auto band = [upper, k](int i, int j) { return upper ? (k + i - j) : (i - j); };

  if (tr == 'N') {
    if (upper) {
      int jx = start_index(n, incx);
      for (int j = 0; j < n; ++j) {
        if (x[jx] != value_constants<T>::zero()) {
          const T temp = x[jx];
          const int first = (j - k > 0) ? (j - k) : 0;
          int ix = jx - (j - first) * incx;
          for (int i = first; i < j; ++i) {
            x[ix] += temp * a[band(i, j) + j * lda];
            ix += incx;
          }
          if (!unit) {
            x[jx] *= a[diag_row + j * lda];
          }
        }
        jx += incx;
      }
    } else {
      int jx = start_index(n, incx) + (n - 1) * incx;
      for (int j = n - 1; j >= 0; --j) {
        if (x[jx] != value_constants<T>::zero()) {
          const T temp = x[jx];
          const int last = (j + k < n - 1) ? (j + k) : (n - 1);
          int ix = jx + (last - j) * incx;
          for (int i = last; i > j; --i) {
            x[ix] += temp * a[band(i, j) + j * lda];
            ix -= incx;
          }
          if (!unit) {
            x[jx] *= a[diag_row + j * lda];
          }
        }
        jx -= incx;
      }
    }
  } else {
    if (upper) {
      int jx = start_index(n, incx) + (n - 1) * incx;
      for (int j = n - 1; j >= 0; --j) {
        accumulator_t<T> temp = to_accumulator(x[jx]);
        if (!unit) {
          const T d = a[diag_row + j * lda];
          temp *= to_accumulator(conjugate ? conj_value(d) : d);
        }
        const int first = (j - k > 0) ? (j - k) : 0;
        int ix = jx;
        for (int i = j - 1; i >= first; --i) {
          ix -= incx;
          const T value = a[band(i, j) + j * lda];
          temp += acc_mul(conjugate ? conj_value(value) : value, x[ix]);
        }
        x[jx] = from_accumulator<T>(temp);
        jx -= incx;
      }
    } else {
      int jx = start_index(n, incx);
      for (int j = 0; j < n; ++j) {
        accumulator_t<T> temp = to_accumulator(x[jx]);
        if (!unit) {
          const T d = a[diag_row + j * lda];
          temp *= to_accumulator(conjugate ? conj_value(d) : d);
        }
        const int last = (j + k < n - 1) ? (j + k) : (n - 1);
        int ix = jx;
        for (int i = j + 1; i <= last; ++i) {
          ix += incx;
          const T value = a[band(i, j) + j * lda];
          temp += acc_mul(conjugate ? conj_value(value) : value, x[ix]);
        }
        x[jx] = from_accumulator<T>(temp);
        jx += incx;
      }
    }
  }
}

template <typename T>
inline void tbsv_impl(char uplo, char trans, char diag, int n, int k, const T* a, int lda,
                      T* x, int incx) {
  const char ul = to_upper(uplo);
  const char tr = to_upper(trans);
  const char dg = to_upper(diag);
  if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || k < 0 ||
      lda < k + 1 || incx == 0) {
    return;
  }
  const bool unit = (dg == 'U');
  const bool conjugate = (tr == 'C');
  const bool upper = (ul == 'U');

  const int diag_row = upper ? k : 0;
  const auto band = [upper, k](int i, int j) { return upper ? (k + i - j) : (i - j); };

  if (tr == 'N') {
    if (upper) {
      int jx = start_index(n, incx) + (n - 1) * incx;
      for (int j = n - 1; j >= 0; --j) {
        if (!unit) {
          x[jx] /= a[diag_row + j * lda];
        }
        const T temp = x[jx];
        const int first = (j - k > 0) ? (j - k) : 0;
        int ix = jx;
        for (int i = j - 1; i >= first; --i) {
          ix -= incx;
          x[ix] -= temp * a[band(i, j) + j * lda];
        }
        jx -= incx;
      }
    } else {
      int jx = start_index(n, incx);
      for (int j = 0; j < n; ++j) {
        if (!unit) {
          x[jx] /= a[diag_row + j * lda];
        }
        const T temp = x[jx];
        const int last = (j + k < n - 1) ? (j + k) : (n - 1);
        int ix = jx;
        for (int i = j + 1; i <= last; ++i) {
          ix += incx;
          x[ix] -= temp * a[band(i, j) + j * lda];
        }
        jx += incx;
      }
    }
  } else {
    if (upper) {
      int jx = start_index(n, incx);
      for (int j = 0; j < n; ++j) {
        accumulator_t<T> temp = to_accumulator(x[jx]);
        const int first = (j - k > 0) ? (j - k) : 0;
        int ix = jx - (j - first) * incx;
        for (int i = first; i < j; ++i) {
          const T value = a[band(i, j) + j * lda];
          temp -= acc_mul(conjugate ? conj_value(value) : value, x[ix]);
          ix += incx;
        }
        if (!unit) {
          const T d = a[diag_row + j * lda];
          temp /= to_accumulator(conjugate ? conj_value(d) : d);
        }
        x[jx] = from_accumulator<T>(temp);
        jx += incx;
      }
    } else {
      int jx = start_index(n, incx) + (n - 1) * incx;
      for (int j = n - 1; j >= 0; --j) {
        accumulator_t<T> temp = to_accumulator(x[jx]);
        const int last = (j + k < n - 1) ? (j + k) : (n - 1);
        int ix = jx + (last - j) * incx;
        for (int i = last; i > j; --i) {
          const T value = a[band(i, j) + j * lda];
          temp -= acc_mul(conjugate ? conj_value(value) : value, x[ix]);
          ix -= incx;
        }
        if (!unit) {
          const T d = a[diag_row + j * lda];
          temp /= to_accumulator(conjugate ? conj_value(d) : d);
        }
        x[jx] = from_accumulator<T>(temp);
        jx -= incx;
      }
    }
  }
}

}  // namespace detail

/**
 * @brief General banded matrix-vector multiply: `y <- alpha * op(a) * x + beta * y`.
 */
template <typename T>
inline void gbmv(char trans, int m, int n, int kl, int ku, T alpha, const T* a, int lda,
                 const T* x, int incx, T beta, T* y, int incy) {
  detail::gbmv_impl(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

/**
 * @brief Symmetric banded matrix-vector multiply: `y <- alpha * a * x + beta * y`.
 */
template <typename T>
inline void sbmv(char uplo, int n, int k, T alpha, const T* a, int lda, const T* x, int incx,
                 T beta, T* y, int incy) {
  detail::sbmv_impl(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/**
 * @brief Hermitian banded matrix-vector multiply: `y <- alpha * a * x + beta * y`.
 *
 * The diagonal of `a` is treated as real; stored imaginary parts are ignored.
 */
template <typename T>
inline void hbmv(char uplo, int n, int k, std::complex<T> alpha, const std::complex<T>* a,
                 int lda, const std::complex<T>* x, int incx, std::complex<T> beta,
                 std::complex<T>* y, int incy) {
  detail::hbmv_impl(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/**
 * @brief Triangular banded matrix-vector multiply: `x <- op(a) * x`.
 */
template <typename T>
inline void tbmv(char uplo, char trans, char diag, int n, int k, const T* a, int lda, T* x,
                 int incx) {
  detail::tbmv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}

/**
 * @brief Triangular banded solve: solve `op(a) * x = b` in place.
 */
template <typename T>
inline void tbsv(char uplo, char trans, char diag, int n, int k, const T* a, int lda, T* x,
                 int incx) {
  detail::tbsv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}

}  // namespace thefblas
