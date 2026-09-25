// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/fixed.hpp"
#include "thefblas/level2.hpp"

#include <complex>

namespace thefblas {

/**
 * @file level2_packed.hpp
 * @brief Header-only generic Level-2 BLAS-style packed matrix-vector routines.
 *
 * Packed matrices store only the referenced triangle, column by column, in the
 * classic BLAS packed format:
 * - `uplo == 'U'`: `a(i, j)` with `i <= j` is stored at `ap[i + j * (j + 1) / 2]`,
 *   so the array holds `a(0,0), a(0,1), a(1,1), a(0,2), ...`.
 * - `uplo == 'L'`: `a(i, j)` with `i >= j` is stored at
 *   `ap[(i - j) + j * (2 * n - j + 1) / 2]`, so the array holds
 *   `a(0,0), a(1,0), ..., a(n-1,0), a(1,1), ...`.
 *
 * In both cases `ap` must hold `n * (n + 1) / 2` elements. All other parameter
 * conventions (`trans`, `uplo`, `diag`, `incx`, `incy`) match level2.hpp, as
 * does the early-return behavior on invalid character parameters, non-positive
 * sizes or zero strides.
 */

namespace detail {

/// Index of `a(i, j)` (`i <= j`) within an upper-triangular packed array.
inline int packed_index_upper(int i, int j) {
    return i + (j * (j + 1)) / 2;
}

/// Index of `a(i, j)` (`i >= j`) within a lower-triangular packed array.
inline int packed_index_lower(int i, int j, int n) {
    return (i - j) + (j * (2 * n - j + 1)) / 2;
}

/// Index of `a(i, j)` within a packed array of either triangle.
inline int packed_index(bool upper, int i, int j, int n) {
    return upper ? packed_index_upper(i, j) : packed_index_lower(i, j, n);
}

template <typename T>
inline void spmv_impl(char uplo, int n, T alpha, const T *ap, const T *x, int incx, T beta, T *y,
                      int incy) {
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    scale_vector(n, beta, y, incy);
    if (alpha == value_constants<T>::zero()) {
        return;
    }
    const bool upper = (ul == 'U');

    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    if (upper) {
        for (int j = 0; j < n; ++j) {
            const T temp1 = alpha * x[jx];
            accumulator_t<T> temp2 = value_constants<accumulator_t<T>>::zero();
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                const T value = ap[packed_index_upper(i, j)];
                y[iy] += temp1 * value;
                temp2 += acc_mul(value, x[ix]);
                ix += incx;
                iy += incy;
            }
            y[jy] += temp1 * ap[packed_index_upper(j, j)] + alpha * from_accumulator<T>(temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const T temp1 = alpha * x[jx];
            accumulator_t<T> temp2 = value_constants<accumulator_t<T>>::zero();
            y[jy] += temp1 * ap[packed_index_lower(j, j, n)];
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                const T value = ap[packed_index_lower(i, j, n)];
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
inline void hpmv_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T> *ap,
                      const std::complex<T> *x, int incx, std::complex<T> beta, std::complex<T> *y,
                      int incy) {
    using C = std::complex<T>;
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    scale_vector(n, beta, y, incy);
    if (alpha == value_constants<C>::zero()) {
        return;
    }
    const bool upper = (ul == 'U');

    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    if (upper) {
        for (int j = 0; j < n; ++j) {
            const C temp1 = alpha * x[jx];
            accumulator_t<C> temp2 = value_constants<accumulator_t<C>>::zero();
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                const C value = ap[packed_index_upper(i, j)];
                y[iy] += temp1 * value;
                temp2 += acc_mul(conj_value(value), x[ix]);
                ix += incx;
                iy += incy;
            }
            y[jy] += temp1 * C(ap[packed_index_upper(j, j)].real(), T{}) +
                     alpha * from_accumulator<C>(temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const C temp1 = alpha * x[jx];
            accumulator_t<C> temp2 = value_constants<accumulator_t<C>>::zero();
            y[jy] += temp1 * C(ap[packed_index_lower(j, j, n)].real(), T{});
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                const C value = ap[packed_index_lower(i, j, n)];
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
inline void tpmv_impl(char uplo, char trans, char diag, int n, const T *ap, T *x, int incx) {
    const char ul = to_upper(uplo);
    const char tr = to_upper(trans);
    const char dg = to_upper(diag);
    if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
        return;
    }
    const bool unit = (dg == 'U');
    const bool upper = (ul == 'U');
    const bool conjugate = (tr == 'C');

    if (tr == 'N') {
        if (upper) {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (x[jx] != value_constants<T>::zero()) {
                    const T temp = x[jx];
                    int ix = start_index(n, incx);
                    for (int i = 0; i < j; ++i) {
                        x[ix] += temp * ap[packed_index_upper(i, j)];
                        ix += incx;
                    }
                    if (!unit) {
                        x[jx] *= ap[packed_index_upper(j, j)];
                    }
                }
                jx += incx;
            }
        } else {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                if (x[jx] != value_constants<T>::zero()) {
                    const T temp = x[jx];
                    int ix = start_index(n, incx) + (n - 1) * incx;
                    for (int i = n - 1; i > j; --i) {
                        x[ix] += temp * ap[packed_index_lower(i, j, n)];
                        ix -= incx;
                    }
                    if (!unit) {
                        x[jx] *= ap[packed_index_lower(j, j, n)];
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
                    const T d = ap[packed_index_upper(j, j)];
                    temp *= to_accumulator(conjugate ? conj_value(d) : d);
                }
                int ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    const T value = ap[packed_index_upper(i, j)];
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
                    const T d = ap[packed_index_lower(j, j, n)];
                    temp *= to_accumulator(conjugate ? conj_value(d) : d);
                }
                int ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    const T value = ap[packed_index_lower(i, j, n)];
                    temp += acc_mul(conjugate ? conj_value(value) : value, x[ix]);
                }
                x[jx] = from_accumulator<T>(temp);
                jx += incx;
            }
        }
    }
}

template <typename T>
inline void tpsv_impl(char uplo, char trans, char diag, int n, const T *ap, T *x, int incx) {
    const char ul = to_upper(uplo);
    const char tr = to_upper(trans);
    const char dg = to_upper(diag);
    if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
        return;
    }
    const bool unit = (dg == 'U');
    const bool upper = (ul == 'U');
    const bool conjugate = (tr == 'C');

    if (tr == 'N') {
        if (upper) {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                if (!unit) {
                    x[jx] /= ap[packed_index_upper(j, j)];
                }
                const T temp = x[jx];
                int ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    x[ix] -= temp * ap[packed_index_upper(i, j)];
                }
                jx -= incx;
            }
        } else {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (!unit) {
                    x[jx] /= ap[packed_index_lower(j, j, n)];
                }
                const T temp = x[jx];
                int ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    x[ix] -= temp * ap[packed_index_lower(i, j, n)];
                }
                jx += incx;
            }
        }
    } else {
        if (upper) {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                accumulator_t<T> temp = to_accumulator(x[jx]);
                int ix = start_index(n, incx);
                for (int i = 0; i < j; ++i) {
                    const T value = ap[packed_index_upper(i, j)];
                    temp -= acc_mul(conjugate ? conj_value(value) : value, x[ix]);
                    ix += incx;
                }
                if (!unit) {
                    const T d = ap[packed_index_upper(j, j)];
                    temp /= to_accumulator(conjugate ? conj_value(d) : d);
                }
                x[jx] = from_accumulator<T>(temp);
                jx += incx;
            }
        } else {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                accumulator_t<T> temp = to_accumulator(x[jx]);
                int ix = start_index(n, incx) + (n - 1) * incx;
                for (int i = n - 1; i > j; --i) {
                    const T value = ap[packed_index_lower(i, j, n)];
                    temp -= acc_mul(conjugate ? conj_value(value) : value, x[ix]);
                    ix -= incx;
                }
                if (!unit) {
                    const T d = ap[packed_index_lower(j, j, n)];
                    temp /= to_accumulator(conjugate ? conj_value(d) : d);
                }
                x[jx] = from_accumulator<T>(temp);
                jx -= incx;
            }
        }
    }
}

template <typename T> inline void spr_impl(char uplo, int n, T alpha, const T *x, int incx, T *ap) {
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || alpha == value_constants<T>::zero()) {
        return;
    }
    const bool upper = (ul == 'U');

    int jx = start_index(n, incx);
    for (int j = 0; j < n; ++j) {
        const T temp = alpha * x[jx];
        const int first = upper ? 0 : j;
        const int last = upper ? j : (n - 1);
        int ix = upper ? start_index(n, incx) : jx;
        for (int i = first; i <= last; ++i) {
            ap[packed_index(upper, i, j, n)] += x[ix] * temp;
            ix += incx;
        }
        jx += incx;
    }
}

template <typename T>
inline void hpr_impl(char uplo, int n, T alpha, const std::complex<T> *x, int incx,
                     std::complex<T> *ap) {
    using C = std::complex<T>;
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || alpha == value_constants<T>::zero()) {
        return;
    }
    const bool upper = (ul == 'U');

    int jx = start_index(n, incx);
    for (int j = 0; j < n; ++j) {
        const C temp = C(alpha) * conj_value(x[jx]);
        const int diag = packed_index(upper, j, j, n);
        if (upper) {
            int ix = start_index(n, incx);
            for (int i = 0; i < j; ++i) {
                ap[packed_index_upper(i, j)] += x[ix] * temp;
                ix += incx;
            }
            ap[diag] = C(ap[diag].real() + (x[jx] * temp).real(), T{});
        } else {
            ap[diag] = C(ap[diag].real() + (x[jx] * temp).real(), T{});
            int ix = jx;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                ap[packed_index_lower(i, j, n)] += x[ix] * temp;
            }
        }
        jx += incx;
    }
}

template <typename T>
inline void spr2_impl(char uplo, int n, T alpha, const T *x, int incx, const T *y, int incy,
                      T *ap) {
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0 ||
        alpha == value_constants<T>::zero()) {
        return;
    }
    const bool upper = (ul == 'U');

    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        const T temp1 = alpha * y[jy];
        const T temp2 = alpha * x[jx];
        const int first = upper ? 0 : j;
        const int last = upper ? j : (n - 1);
        int ix = upper ? start_index(n, incx) : jx;
        int iy = upper ? start_index(n, incy) : jy;
        for (int i = first; i <= last; ++i) {
            ap[packed_index(upper, i, j, n)] += x[ix] * temp1 + y[iy] * temp2;
            ix += incx;
            iy += incy;
        }
        jx += incx;
        jy += incy;
    }
}

template <typename T>
inline void hpr2_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                      const std::complex<T> *y, int incy, std::complex<T> *ap) {
    using C = std::complex<T>;
    const char ul = to_upper(uplo);
    if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0 ||
        alpha == value_constants<C>::zero()) {
        return;
    }
    const bool upper = (ul == 'U');

    int jx = start_index(n, incx);
    int jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        const C temp1 = alpha * conj_value(y[jy]);
        const C temp2 = conj_value(alpha * x[jx]);
        const int diag = packed_index(upper, j, j, n);
        if (upper) {
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                ap[packed_index_upper(i, j)] += x[ix] * temp1 + y[iy] * temp2;
                ix += incx;
                iy += incy;
            }
            ap[diag] = C((ap[diag] + x[jx] * temp1 + y[jy] * temp2).real(), T{});
        } else {
            ap[diag] = C((ap[diag] + x[jx] * temp1 + y[jy] * temp2).real(), T{});
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                ap[packed_index_lower(i, j, n)] += x[ix] * temp1 + y[iy] * temp2;
            }
        }
        jx += incx;
        jy += incy;
    }
}

} // namespace detail

/**
 * @brief Symmetric packed matrix-vector multiply: `y <- alpha * a * x + beta * y`.
 */
template <typename T>
inline void spmv(char uplo, int n, T alpha, const T *ap, const T *x, int incx, T beta, T *y,
                 int incy) {
    detail::spmv_impl(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

/**
 * @brief Hermitian packed matrix-vector multiply: `y <- alpha * a * x + beta * y`.
 *
 * The diagonal of `ap` is treated as real; stored imaginary parts are ignored.
 */
template <typename T>
inline void hpmv(char uplo, int n, std::complex<T> alpha, const std::complex<T> *ap,
                 const std::complex<T> *x, int incx, std::complex<T> beta, std::complex<T> *y,
                 int incy) {
    detail::hpmv_impl(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

/**
 * @brief Triangular packed matrix-vector multiply: `x <- op(a) * x`.
 */
template <typename T>
inline void tpmv(char uplo, char trans, char diag, int n, const T *ap, T *x, int incx) {
    detail::tpmv_impl(uplo, trans, diag, n, ap, x, incx);
}

/**
 * @brief Triangular packed solve: solve `op(a) * x = b` in place.
 */
template <typename T>
inline void tpsv(char uplo, char trans, char diag, int n, const T *ap, T *x, int incx) {
    detail::tpsv_impl(uplo, trans, diag, n, ap, x, incx);
}

/**
 * @brief Symmetric packed rank-1 update: `a <- alpha * x * x^T + a`.
 */
template <typename T> inline void spr(char uplo, int n, T alpha, const T *x, int incx, T *ap) {
    detail::spr_impl(uplo, n, alpha, x, incx, ap);
}

/**
 * @brief Hermitian packed rank-1 update: `a <- alpha * x * x^H + a`.
 *
 * `alpha` is real-valued and the updated diagonal is forced real.
 */
template <typename T>
inline void hpr(char uplo, int n, T alpha, const std::complex<T> *x, int incx,
                std::complex<T> *ap) {
    detail::hpr_impl(uplo, n, alpha, x, incx, ap);
}

/**
 * @brief Symmetric packed rank-2 update: `a <- alpha * x * y^T + alpha * y * x^T + a`.
 */
template <typename T>
inline void spr2(char uplo, int n, T alpha, const T *x, int incx, const T *y, int incy, T *ap) {
    detail::spr2_impl(uplo, n, alpha, x, incx, y, incy, ap);
}

/**
 * @brief Hermitian packed rank-2 update: `a <- alpha * x * y^H + conj(alpha) * y * x^H + a`.
 */
template <typename T>
inline void hpr2(char uplo, int n, std::complex<T> alpha, const std::complex<T> *x, int incx,
                 const std::complex<T> *y, int incy, std::complex<T> *ap) {
    detail::hpr2_impl(uplo, n, alpha, x, incx, y, incy, ap);
}

} // namespace thefblas
