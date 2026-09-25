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

template <typename T> struct value_constants {
    static T zero() { return T{}; }
    static T one() { return T{1}; }
};

template <typename T> struct value_constants<std::complex<T>> {
    static std::complex<T> zero() { return std::complex<T>(T{}, T{}); }
    static std::complex<T> one() { return std::complex<T>(T{1}, T{}); }
};

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

template <typename T>
inline void gemv_impl(char trans, int m, int n, T alpha, const T *a, int lda, const T *x, int incx,
                      T beta, T *y, int incy) {
    const char tr = to_upper(trans);
    if (!valid_trans(trans) || m <= 0 || n <= 0 || incx == 0 || incy == 0) {
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
            accumulator_t<T> temp = value_constants<accumulator_t<T>>::zero();
            int ix = start_index(lenx, incx);
            for (int i = 0; i < m; ++i) {
                temp += acc_mul(a[i + j * lda], x[ix]);
                ix += incx;
            }
            y[jy] += alpha * from_accumulator<T>(temp);
            jy += incy;
        }
    } else {
        int jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            accumulator_t<T> temp = value_constants<accumulator_t<T>>::zero();
            int ix = start_index(lenx, incx);
            for (int i = 0; i < m; ++i) {
                temp += acc_mul(conj_value(a[i + j * lda]), x[ix]);
                ix += incx;
            }
            y[jy] += alpha * from_accumulator<T>(temp);
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

    scale_vector(n, beta, y, incy);
    if (alpha == value_constants<T>::zero()) {
        return;
    }

    if (ul == 'U') {
        int jx = start_index(n, incx);
        int jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            const T temp1 = alpha * x[jx];
            accumulator_t<T> temp2 = value_constants<accumulator_t<T>>::zero();
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                y[iy] += temp1 * a[i + j * lda];
                temp2 += acc_mul(a[i + j * lda], x[ix]);
                ix += incx;
                iy += incy;
            }
            y[jy] += temp1 * a[j + j * lda] + alpha * from_accumulator<T>(temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        int jx = start_index(n, incx);
        int jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            const T temp1 = alpha * x[jx];
            accumulator_t<T> temp2 = value_constants<accumulator_t<T>>::zero();
            y[jy] += temp1 * a[j + j * lda];
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * a[i + j * lda];
                temp2 += acc_mul(a[i + j * lda], x[ix]);
            }
            y[jy] += alpha * from_accumulator<T>(temp2);
            jx += incx;
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

    scale_vector(n, beta, y, incy);
    if (alpha == value_constants<C>::zero()) {
        return;
    }

    if (ul == 'U') {
        int jx = start_index(n, incx);
        int jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            const C temp1 = alpha * x[jx];
            accumulator_t<C> temp2 = value_constants<accumulator_t<C>>::zero();
            int ix = start_index(n, incx);
            int iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                y[iy] += temp1 * a[i + j * lda];
                temp2 += acc_mul(conj_value(a[i + j * lda]), x[ix]);
                ix += incx;
                iy += incy;
            }
            y[jy] += temp1 * C(a[j + j * lda].real(), T{}) + alpha * from_accumulator<C>(temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        int jx = start_index(n, incx);
        int jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            const C temp1 = alpha * x[jx];
            accumulator_t<C> temp2 = value_constants<accumulator_t<C>>::zero();
            y[jy] += temp1 * C(a[j + j * lda].real(), T{});
            int ix = jx;
            int iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * a[i + j * lda];
                temp2 += acc_mul(conj_value(a[i + j * lda]), x[ix]);
            }
            y[jy] += alpha * from_accumulator<C>(temp2);
            jx += incx;
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
        if (ul == 'U') {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (x[jx] != value_constants<T>::zero()) {
                    const T temp = x[jx];
                    int ix = start_index(n, incx);
                    for (int i = 0; i < j; ++i) {
                        x[ix] += temp * a[i + j * lda];
                        ix += incx;
                    }
                    if (!unit) {
                        x[jx] *= a[j + j * lda];
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
                        x[ix] += temp * a[i + j * lda];
                        ix -= incx;
                    }
                    if (!unit) {
                        x[jx] *= a[j + j * lda];
                    }
                }
                jx -= incx;
            }
        }
    } else {
        const bool conjugate = (tr == 'C');
        if (ul == 'U') {
            int jx = start_index(n, incx) + (n - 1) * incx;
            for (int j = n - 1; j >= 0; --j) {
                accumulator_t<T> temp = to_accumulator(x[jx]);
                if (!unit) {
                    temp *= to_accumulator(conjugate ? conj_value(a[j + j * lda]) : a[j + j * lda]);
                }
                int ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    temp += acc_mul(conjugate ? conj_value(a[i + j * lda]) : a[i + j * lda], x[ix]);
                }
                x[jx] = from_accumulator<T>(temp);
                jx -= incx;
            }
        } else {
            int jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                accumulator_t<T> temp = to_accumulator(x[jx]);
                if (!unit) {
                    temp *= to_accumulator(conjugate ? conj_value(a[j + j * lda]) : a[j + j * lda]);
                }
                int ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    temp += acc_mul(conjugate ? conj_value(a[i + j * lda]) : a[i + j * lda], x[ix]);
                }
                x[jx] = from_accumulator<T>(temp);
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
                accumulator_t<T> temp = to_accumulator(x[jx]);
                int ix = start_index(n, incx);
                for (int i = 0; i < j; ++i) {
                    temp -= acc_mul(conjugate ? conj_value(a[i + j * lda]) : a[i + j * lda], x[ix]);
                    ix += incx;
                }
                if (!unit) {
                    temp /= to_accumulator(conjugate ? conj_value(a[j + j * lda]) : a[j + j * lda]);
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
                    temp -= acc_mul(conjugate ? conj_value(a[i + j * lda]) : a[i + j * lda], x[ix]);
                    ix -= incx;
                }
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
