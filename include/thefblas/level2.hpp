// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/detail.hpp"
#include "thefblas/fixed.hpp"

#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>

namespace thefblas {

/**
 * @file level2.hpp
 * @brief Header-only generic Level-2 BLAS-style dense matrix-vector routines.
 *
 * Matrices use column-major (Fortran) layout: `a(i,j)` resides at offset
 * `i + j * lda`.
 *
 * Parameter conventions:
 * - `trans`: `'N'` no-transpose, `'T'` transpose, `'C'` conjugate-transpose.
 * - `uplo`: `'U'` upper triangle, `'L'` lower triangle.
 * - `diag`: `'U'` unit diagonal, `'N'` non-unit diagonal.
 * - `incx`, `incy`: vector strides; zero stride causes a no-op.
 *
 * Behavior follows theblas/Netlib BLAS dense Level-2 semantics:
 * invalid character parameters or non-positive problem sizes return early.
 */

namespace detail {

inline char to_upper(char value) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(value)));
}

inline bool valid_trans(char value) {
    const char upper = to_upper(value);
    return upper == 'N' || upper == 'T' || upper == 'C';
}

inline bool valid_uplo(char value) {
    const char upper = to_upper(value);
    return upper == 'U' || upper == 'L';
}

inline bool valid_diag(char value) {
    const char upper = to_upper(value);
    return upper == 'U' || upper == 'N';
}

template <typename T> inline void scale_vector(int n, T beta, T *y, int incy) {
    int iy = start_index(n, incy);
    if (beta == value_constants<T>::zero()) {
        for (int i = 0; i < n; ++i) {
            y[iy] = value_constants<T>::zero();
            iy += incy;
        }
    } else if (beta != value_constants<T>::one()) {
        for (int i = 0; i < n; ++i) {
            y[iy] *= beta;
            iy += incy;
        }
    }
}

/// `y += alpha * A * x` for a symmetric (or, with `Hermitian`, Hermitian)
/// matrix of half-bandwidth `k` (`n - 1` for a full matrix), in the Netlib
/// BLAS column-update order, with `y` already `beta`-scaled. Used for the
/// floating-point element types so that their results stay bit-identical to
/// Netlib; `elem(i, j)` returns the stored element `a(i, j)` of the referenced
/// triangle (`i <= j` for upper, `i >= j` for lower).
template <bool Hermitian, typename T, typename Elem>
inline void symmetric_update_netlib(bool upper, int n, int k, T alpha, Elem elem, const T *x,
                                    int incx, T *y, int incy) {
    const auto diag_product = [](const T &temp1, const T &ajj) -> T {
        if constexpr (Hermitian) {
            return temp1 * ajj.real();
        } else {
            return temp1 * ajj;
        }
    };
    const auto off_product = [](const T &aij, const T &xi) -> T {
        if constexpr (Hermitian) {
            return conj_value(aij) * xi;
        } else {
            return aij * xi;
        }
    };

    const int start_x = start_index(n, incx);
    const int start_y = start_index(n, incy);
    int jx = start_x;
    int jy = start_y;
    for (int j = 0; j < n; ++j) {
        const T temp1 = alpha * x[jx];
        T temp2 = value_constants<T>::zero();
        if (upper) {
            const int first = (j - k > 0) ? (j - k) : 0;
            int ix = start_x + first * incx;
            int iy = start_y + first * incy;
            for (int i = first; i < j; ++i) {
                const T value = elem(i, j);
                y[iy] += temp1 * value;
                temp2 += off_product(value, x[ix]);
                ix += incx;
                iy += incy;
            }
            y[jy] = y[jy] + diag_product(temp1, elem(j, j)) + alpha * temp2;
        } else {
            y[jy] += diag_product(temp1, elem(j, j));
            const int last = (j + k < n - 1) ? (j + k) : (n - 1);
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i <= last; ++i) {
                ix += incx;
                iy += incy;
                const T value = elem(i, j);
                y[iy] += temp1 * value;
                temp2 += off_product(value, x[ix]);
            }
            y[jy] += alpha * temp2;
        }
        jx += incx;
        jy += incy;
    }
}

template <typename T>
inline void gemv_impl(char trans, int m, int n, T alpha, const T *a, int lda, const T *x, int incx,
                      T beta, T *y, int incy) {
    const char tr = to_upper(trans);
    if (!valid_trans(trans) || m <= 0 || n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    const int leny = (tr == 'N') ? m : n;
    const int lenx = (tr == 'N') ? n : m;
    if (alpha == value_constants<T>::zero()) {
        scale_vector(leny, beta, y, incy);
        return;
    }

    if (tr == 'N') {
        scale_vector(leny, beta, y, incy);
        int jx = start_index(lenx, incx);
        for (int j = 0; j < n; ++j) {
            const T temp = alpha * x[jx];
            int iy = start_index(leny, incy);
            for (int i = 0; i < m; ++i) {
                y[iy] += temp * a[i + j * lda];
                iy += incy;
            }
            jx += incx;
        }
    } else if (tr == 'T') {
        int jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            mac_t<T> temp;
            int ix = start_index(lenx, incx);
            for (int i = 0; i < m; ++i) {
                temp.add_product(a[i + j * lda], x[ix]);
                ix += incx;
            }
            y[jy] = acc_scale_add_and_narrow(beta, y[jy], alpha, temp.wide_value());
            jy += incy;
        }
    } else {
        int jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            mac_t<T> temp;
            int ix = start_index(lenx, incx);
            for (int i = 0; i < m; ++i) {
                temp.add_conj_product(a[i + j * lda], x[ix]);
                ix += incx;
            }
            y[jy] = acc_scale_add_and_narrow(beta, y[jy], alpha, temp.wide_value());
            jy += incy;
        }
    }
}

template <typename T>
inline void symv_impl(char uplo, int n, T alpha, const T *a, int lda, const T *x, int incx, T beta,
                      T *y, int incy) {
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    if (alpha == value_constants<T>::zero()) {
        scale_vector(n, beta, y, incy);
        return;
    }
    const bool upper = (ul == 'U');

    if constexpr (is_floating_v<T>) {
        scale_vector(n, beta, y, incy);
        symmetric_update_netlib<false>(
            upper, n, n - 1, alpha, [a, lda](int i, int j) { return a[i + j * lda]; }, x, incx, y,
            incy);
    } else {
        // Each output element is a single widened reduction over its full matrix
        // row (gathering the mirrored element from the stored triangle), scaled by
        // alpha and added to the beta-scaled y in the accumulator type, and narrowed
        // exactly once, so intermediate terms cannot saturate or wrap before later
        // terms cancel them.
        int jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            mac_t<T> temp;
            int ix = start_index(n, incx);
            for (int i = 0; i < n; ++i) {
                const int row = upper ? (i < j ? i : j) : (i < j ? j : i);
                const int col = upper ? (i < j ? j : i) : (i < j ? i : j);
                temp.add_product(a[row + col * lda], x[ix]);
                ix += incx;
            }
            y[jy] = acc_scale_add_and_narrow(beta, y[jy], alpha, temp.wide_value());
            jy += incy;
        }
    }
}

template <typename T>
inline void hemv_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T> *a, int lda,
                      const std::complex<T> *x, int incx, std::complex<T> beta, std::complex<T> *y,
                      int incy) {
    using C = std::complex<T>;
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    if (alpha == value_constants<C>::zero()) {
        scale_vector(n, beta, y, incy);
        return;
    }
    const bool upper = (ul == 'U');

    if constexpr (is_floating_v<C>) {
        scale_vector(n, beta, y, incy);
        symmetric_update_netlib<true>(
            upper, n, n - 1, alpha, [a, lda](int i, int j) { return a[i + j * lda]; }, x, incx, y,
            incy);
    } else {
        // As in symv_impl, each output element is a single widened reduction over
        // its full matrix row; elements from the opposite triangle are conjugated
        // and the diagonal is taken as real, per the Hermitian storage convention.
        int jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            mac_t<C> temp;
            int ix = start_index(n, incx);
            for (int i = 0; i < n; ++i) {
                if (i == j) {
                    temp.add_product(C(a[j + j * lda].real(), T{}), x[ix]);
                } else if (upper ? (j < i) : (j > i)) {
                    temp.add_product(a[j + i * lda], x[ix]);
                } else {
                    temp.add_conj_product(a[i + j * lda], x[ix]);
                }
                ix += incx;
            }
            y[jy] = acc_scale_add_and_narrow(beta, y[jy], alpha, temp.wide_value());
            jy += incy;
        }
    }
}

template <typename T>
inline void trmv_impl(char uplo, char trans, char diag, int n, const T *a, int lda, T *x,
                      int incx) {
    const char ul = to_upper(uplo);
    const char tr = to_upper(trans);
    const char dg = to_upper(diag);
    if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
        return;
    }
    const bool unit = (dg == 'U');

    if (tr == 'N') {
        // Each x(i) is formed as one widened dot product of row i with x and
        // overwritten in an order that leaves the x(j) it still needs intact.
        if (ul == 'U') {
            int ix = start_index(n, incx);
            for (int i = 0; i < n; ++i) {
                mac_t<T> temp;
                if (unit) {
                    temp.add(x[ix]);
                } else {
                    temp.add_product(a[i + i * lda], x[ix]);
                }
                int jx = ix;
                for (int j = i + 1; j < n; ++j) {
                    jx += incx;
                    temp.add_product(a[i + j * lda], x[jx]);
                }
                x[ix] = temp.value();
                ix += incx;
            }
        } else {
            int ix = start_index(n, incx) + (n - 1) * incx;
            for (int i = n - 1; i >= 0; --i) {
                mac_t<T> temp;
                if (unit) {
                    temp.add(x[ix]);
                } else {
                    temp.add_product(a[i + i * lda], x[ix]);
                }
                int jx = ix;
                for (int j = i - 1; j >= 0; --j) {
                    jx -= incx;
                    temp.add_product(a[i + j * lda], x[jx]);
                }
                x[ix] = temp.value();
                ix -= incx;
            }
        }
    } else {
        const bool conjugate = (tr == 'C');
        if (ul == 'U') {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                mac_t<T> temp;
                if (unit) {
                    temp.add(x[jx]);
                } else if (conjugate) {
                    temp.add_conj_product(a[j + j * lda], x[jx]);
                } else {
                    temp.add_product(a[j + j * lda], x[jx]);
                }
                int ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    if (conjugate) {
                        temp.add_conj_product(a[i + j * lda], x[ix]);
                    } else {
                        temp.add_product(a[i + j * lda], x[ix]);
                    }
                }
                x[jx] = temp.value();
                jx -= incx;
            }
        } else {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                mac_t<T> temp;
                if (unit) {
                    temp.add(x[jx]);
                } else if (conjugate) {
                    temp.add_conj_product(a[j + j * lda], x[jx]);
                } else {
                    temp.add_product(a[j + j * lda], x[jx]);
                }
                int ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    if (conjugate) {
                        temp.add_conj_product(a[i + j * lda], x[ix]);
                    } else {
                        temp.add_product(a[i + j * lda], x[ix]);
                    }
                }
                x[jx] = temp.value();
                jx += incx;
            }
        }
    }
}

template <typename T>
inline void trsv_impl(char uplo, char trans, char diag, int n, const T *a, int lda, T *x,
                      int incx) {
    const char ul = to_upper(uplo);
    const char tr = to_upper(trans);
    const char dg = to_upper(diag);
    if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
        return;
    }
    const bool unit = (dg == 'U');

    if (tr == 'N') {
        if (ul == 'U') {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                if (!unit) {
                    x[jx] /= a[j + j * lda];
                }
                const T temp = x[jx];
                int ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    x[ix] -= temp * a[i + j * lda];
                }
                jx -= incx;
            }
        } else {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (!unit) {
                    x[jx] /= a[j + j * lda];
                }
                const T temp = x[jx];
                int ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    x[ix] -= temp * a[i + j * lda];
                }
                jx += incx;
            }
        }
    } else {
        const bool conjugate = (tr == 'C');
        if (ul == 'U') {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                mac_t<T> sum;
                sum.add(x[jx]);
                int ix = start_index(n, incx);
                for (int i = 0; i < j; ++i) {
                    if (conjugate) {
                        sum.subtract_conj_product(a[i + j * lda], x[ix]);
                    } else {
                        sum.subtract_product(a[i + j * lda], x[ix]);
                    }
                    ix += incx;
                }
                accumulator_t<T> temp = sum.wide_value();
                if (!unit) {
                    temp /= to_accumulator(conjugate ? conj_value(a[j + j * lda]) : a[j + j * lda]);
                }
                x[jx] = from_accumulator<T>(temp);
                jx += incx;
            }
        } else {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                mac_t<T> sum;
                sum.add(x[jx]);
                int ix = start_index(n, incx) + (n - 1) * incx;
                for (int i = n - 1; i > j; --i) {
                    if (conjugate) {
                        sum.subtract_conj_product(a[i + j * lda], x[ix]);
                    } else {
                        sum.subtract_product(a[i + j * lda], x[ix]);
                    }
                    ix -= incx;
                }
                accumulator_t<T> temp = sum.wide_value();
                if (!unit) {
                    temp /= to_accumulator(conjugate ? conj_value(a[j + j * lda]) : a[j + j * lda]);
                }
                x[jx] = from_accumulator<T>(temp);
                jx -= incx;
            }
        }
    }
}

template <typename T>
inline void ger_impl(int m, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a,
                     int lda) {
    if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
        return;
    }
    if (alpha == value_constants<T>::zero()) {
        return;
    }

    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        const T temp = alpha * y[jy];
        int ix = start_index(m, incx);
        for (int i = 0; i < m; ++i) {
            a[i + j * lda] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}

template <typename T>
inline void geru_impl(int m, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                      const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
        return;
    }
    if (alpha == value_constants<C>::zero()) {
        return;
    }

    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        const C temp = alpha * y[jy];
        int ix = start_index(m, incx);
        for (int i = 0; i < m; ++i) {
            a[i + j * lda] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}

template <typename T>
inline void gerc_impl(int m, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                      const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
        return;
    }
    if (alpha == value_constants<C>::zero()) {
        return;
    }

    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        const C temp = alpha * conj_value(y[jy]);
        int ix = start_index(m, incx);
        for (int i = 0; i < m; ++i) {
            a[i + j * lda] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}

template <typename T>
inline void syr_impl(char uplo, int n, T alpha, const T *x, int incx, T *a, int lda) {
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
        return;
    }
    if (alpha == value_constants<T>::zero()) {
        return;
    }

    int jx = start_index(n, incx);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            const T temp = alpha * x[jx];
            int ix = start_index(n, incx);
            for (int i = 0; i <= j; ++i) {
                a[i + j * lda] += x[ix] * temp;
                ix += incx;
            }
            jx += incx;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const T temp = alpha * x[jx];
            int ix = jx;
            for (int i = j; i < n; ++i) {
                a[i + j * lda] += x[ix] * temp;
                ix += incx;
            }
            jx += incx;
        }
    }
}

template <typename T>
inline void her_impl(char uplo, int n, T alpha, const std::complex<T> *x, int incx,
                     std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
        return;
    }
    if (alpha == value_constants<T>::zero()) {
        return;
    }

    int jx = start_index(n, incx);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            const C temp = C(alpha) * conj_value(x[jx]);
            int ix = start_index(n, incx);
            for (int i = 0; i < j; ++i) {
                a[i + j * lda] += x[ix] * temp;
                ix += incx;
            }
            a[j + j * lda] = C(a[j + j * lda].real() + (x[jx] * temp).real(), T{});
            jx += incx;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const C temp = C(alpha) * conj_value(x[jx]);
            a[j + j * lda] = C(a[j + j * lda].real() + (x[jx] * temp).real(), T{});
            int ix = jx;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                a[i + j * lda] += x[ix] * temp;
            }
            jx += incx;
        }
    }
}

template <typename T>
inline void syr2_impl(char uplo, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a,
                      int lda) {
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
        return;
    }
    if (alpha == value_constants<T>::zero()) {
        return;
    }

    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            const T temp1 = alpha * y[jy];
            const T temp2 = alpha * x[jx];
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i <= j; ++i) {
                a[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
                ix += incx;
                iy += incy;
            }
            jx += incx;
            jy += incy;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const T temp1 = alpha * y[jy];
            const T temp2 = alpha * x[jx];
            int ix = jx;
            int iy = jy;
            for (int i = j; i < n; ++i) {
                a[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
                ix += incx;
                iy += incy;
            }
            jx += incx;
            jy += incy;
        }
    }
}

template <typename T>
inline void her2_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                      const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
        return;
    }
    if (alpha == value_constants<C>::zero()) {
        return;
    }

    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            const C temp1 = alpha * conj_value(y[jy]);
            const C temp2 = conj_value(alpha * x[jx]);
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                a[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
                ix += incx;
                iy += incy;
            }
            a[j + j * lda] = C((a[j + j * lda] + x[jx] * temp1 + y[jy] * temp2).real(), T{});
            jx += incx;
            jy += incy;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const C temp1 = alpha * conj_value(y[jy]);
            const C temp2 = conj_value(alpha * x[jx]);
            a[j + j * lda] = C((a[j + j * lda] + x[jx] * temp1 + y[jy] * temp2).real(), T{});
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                a[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
            }
            jx += incx;
            jy += incy;
        }
    }
}

} // namespace detail

/**
 * @brief General matrix-vector multiply: `y <- alpha * op(a) * x + beta * y`.
 */
template <typename T>
inline void gemv(char trans, int m, int n, T alpha, const T *a, int lda, const T *x, int incx,
                 T beta, T *y, int incy) {
    detail::gemv_impl(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

/**
 * @brief General rank-1 update: `a <- alpha * x * y^T + a`.
 */
template <typename T>
inline void ger(int m, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a, int lda) {
    detail::ger_impl(m, n, alpha, x, incx, y, incy, a, lda);
}

/**
 * @brief Complex unconjugated rank-1 update: `a <- alpha * x * y^T + a`.
 */
template <typename T>
inline void geru(int m, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                 const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    detail::geru_impl(m, n, alpha, x, incx, y, incy, a, lda);
}

/**
 * @brief Complex conjugated rank-1 update: `a <- alpha * x * y^H + a`.
 */
template <typename T>
inline void gerc(int m, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                 const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    detail::gerc_impl(m, n, alpha, x, incx, y, incy, a, lda);
}

/**
 * @brief Symmetric matrix-vector multiply: `y <- alpha * a * x + beta * y`.
 */
template <typename T>
inline void symv(char uplo, int n, T alpha, const T *a, int lda, const T *x, int incx, T beta, T *y,
                 int incy) {
    detail::symv_impl(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/**
 * @brief Hermitian matrix-vector multiply: `y <- alpha * a * x + beta * y`.
 *
 * The diagonal of `a` is treated as real; stored imaginary parts are ignored.
 */
template <typename T>
inline void hemv(char uplo, int n, std::complex<T> alpha, const std::complex<T> *a, int lda,
                 const std::complex<T> *x, int incx, std::complex<T> beta, std::complex<T> *y,
                 int incy) {
    detail::hemv_impl(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/**
 * @brief Symmetric rank-1 update: `a <- alpha * x * x^T + a`.
 */
template <typename T>
inline void syr(char uplo, int n, T alpha, const T *x, int incx, T *a, int lda) {
    detail::syr_impl(uplo, n, alpha, x, incx, a, lda);
}

/**
 * @brief Hermitian rank-1 update: `a <- alpha * x * x^H + a`.
 *
 * `alpha` is real-valued and the updated diagonal is forced real.
 */
template <typename T>
inline void her(char uplo, int n, T alpha, const std::complex<T> *x, int incx, std::complex<T> *a,
                int lda) {
    detail::her_impl(uplo, n, alpha, x, incx, a, lda);
}

/**
 * @brief Symmetric rank-2 update: `a <- alpha * x * y^T + alpha * y * x^T + a`.
 */
template <typename T>
inline void syr2(char uplo, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a,
                 int lda) {
    detail::syr2_impl(uplo, n, alpha, x, incx, y, incy, a, lda);
}

/**
 * @brief Hermitian rank-2 update: `a <- alpha * x * y^H + conj(alpha) * y * x^H + a`.
 */
template <typename T>
inline void her2(char uplo, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                 const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    detail::her2_impl(uplo, n, alpha, x, incx, y, incy, a, lda);
}

/**
 * @brief Triangular matrix-vector multiply: `x <- op(a) * x`.
 */
template <typename T>
inline void trmv(char uplo, char trans, char diag, int n, const T *a, int lda, T *x, int incx) {
    detail::trmv_impl(uplo, trans, diag, n, a, lda, x, incx);
}

/**
 * @brief Triangular solve: solve `op(a) * x = b` in place.
 */
template <typename T>
inline void trsv(char uplo, char trans, char diag, int n, const T *a, int lda, T *x, int incx) {
    detail::trsv_impl(uplo, trans, diag, n, a, lda, x, incx);
}

} // namespace thefblas
