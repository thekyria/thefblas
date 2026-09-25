// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/detail.hpp"
#include "thefblas/fixed.hpp"

#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace thefblas {

/**
 * @file level1.hpp
 * @brief Header-only templated Level-1 BLAS-style vector routines.
 *
 * The API follows Netlib BLAS Level-1 naming and parameter conventions, but
 * collapses the classic s/d/c/z precision families into C++ templates:
 * callers choose the element type `T` directly (`float`, `double`,
 * `fixed<IntType, FracBits>`, or `std::complex<T>` thereof where applicable).
 *
 * Parameter conventions used by all routines:
 * - `n`: number of logical vector elements to process
 * - `x`, `y`: input/output vectors
 * - `incx`, `incy`: element strides (can be negative unless otherwise noted)
 *
 * Behavior notes:
 * - If `n <= 0`, routines are no-ops (or return zero / index 0).
 * - If a stride is zero, most routines are treated as no-ops (or return zero).
 * - For `iamax`, `incx` must be strictly positive; otherwise 0 is returned.
 * - `iamax` return values use Netlib BLAS indexing (1-based index).
 */

namespace detail {

// is_complex, is_fixed, enable_if_real_t and the accumulator machinery live in
// detail.hpp, which both level1.hpp and level2.hpp include.

template <typename T> inline T abs_value(const T &value) {
    using std::abs;
    return abs(value);
}

template <typename T> inline T sqrt_value(const T &value) {
    using std::sqrt;
    return sqrt(value);
}

template <typename T> inline T abs1(const std::complex<T> &value) {
    return abs_value(value.real()) + abs_value(value.imag());
}

template <typename T> inline T norm_square(const T &value) {
    return value * value;
}

template <typename T> inline T norm_square(const std::complex<T> &value) {
    return (value.real() * value.real()) + (value.imag() * value.imag());
}

template <typename T> inline T complex_abs(const std::complex<T> &value) {
    return sqrt_value(norm_square(value));
}

template <typename T> inline T signed_magnitude(T magnitude, const T &sign_source) {
    return sign_source < T(0) ? -magnitude : magnitude;
}

template <typename T> inline T tiny_value() {
    if constexpr (is_fixed_v<T>) {
        return T::from_raw(1);
    } else {
        return T(1e-30);
    }
}

} // namespace detail

/**
 * @brief Swap two vectors element-wise.
 * @param n Number of elements to process.
 * @param x First vector, updated in place.
 * @param incx Stride between elements of x.
 * @param y Second vector, updated in place.
 * @param incy Stride between elements of y.
 */
template <typename T> inline void swap(int n, T *x, int incx, T *y, int incy) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        const T tmp = x[ix];
        x[ix] = y[iy];
        y[iy] = tmp;
        ix += incx;
        iy += incy;
    }
}

/**
 * @brief Copy one vector into another.
 * @param n Number of elements to process.
 * @param x Source vector.
 * @param incx Stride between elements of x.
 * @param y Destination vector.
 * @param incy Stride between elements of y.
 */
template <typename T> inline void copy(int n, const T *x, int incx, T *y, int incy) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        y[iy] = x[ix];
        ix += incx;
        iy += incy;
    }
}

/**
 * @brief Compute `y <- alpha * x + y`.
 * @param n Number of elements to process.
 * @param alpha Scalar multiplier.
 * @param x Input vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 */
template <typename T> inline void axpy(int n, T alpha, const T *x, int incx, T *y, int incy) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}

/**
 * @brief Scale a vector in place: `x <- alpha * x`.
 * @param n Number of elements to process.
 * @param alpha Scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
template <typename T> inline void scal(int n, T alpha, T *x, int incx) {
    if (n <= 0 || incx == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    for (int i = 0; i < n; ++i) {
        x[ix] *= alpha;
        ix += incx;
    }
}

/**
 * @brief Scale a complex vector by a real scalar in place.
 * @param n Number of elements to process.
 * @param alpha Real scale factor.
 * @param x Complex vector to scale in place.
 * @param incx Stride between elements of x.
 */
template <typename T> inline void scal(int n, T alpha, std::complex<T> *x, int incx) {
    if (n <= 0 || incx == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    for (int i = 0; i < n; ++i) {
        x[ix] *= alpha;
        ix += incx;
    }
}

/**
 * @brief Dot product of two real vectors.
 * @param n Number of elements to process.
 * @param x First input vector.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Dot product value.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline T dot(int n, const T *x, int incx, const T *y, int incy) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return T(0);
    }

    detail::real_mac<T> acc;
    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        acc.add_product(x[ix], y[iy]);
        ix += incx;
        iy += incy;
    }
    return acc.value();
}

/**
 * @brief Complex dot product without conjugation.
 * @param n Number of elements to process.
 * @param x First input vector.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Complex dot product value.
 */
template <typename T>
inline std::complex<T> dotu(int n, const std::complex<T> *x, int incx, const std::complex<T> *y,
                            int incy) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return std::complex<T>(T(0), T(0));
    }

    detail::complex_mac<T> acc;
    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        acc.add_product(x[ix], y[iy]);
        ix += incx;
        iy += incy;
    }
    return acc.value();
}

/**
 * @brief Complex dot product with conjugated first argument.
 * @param n Number of elements to process.
 * @param x First input vector, conjugated in the product.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Complex dot product value.
 */
template <typename T>
inline std::complex<T> dotc(int n, const std::complex<T> *x, int incx, const std::complex<T> *y,
                            int incy) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return std::complex<T>(T(0), T(0));
    }

    detail::complex_mac<T> acc;
    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        acc.add_conj_product(x[ix], y[iy]);
        ix += incx;
        iy += incy;
    }
    return acc.value();
}

/**
 * @brief Euclidean norm of a real vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
template <typename T, detail::enable_if_real_t<T> = 0> inline T nrm2(int n, const T *x, int incx) {
    if (n <= 0 || incx == 0) {
        return T(0);
    }

    detail::real_mac<T> sum;
    int ix = detail::start_index(n, incx);
    for (int i = 0; i < n; ++i) {
        sum.add_product(x[ix], x[ix]);
        ix += incx;
    }
    // The sum of squares can legitimately exceed the range of T even when the
    // norm itself does not, so the square root is taken in the wide accumulator
    // type and only its result is narrowed back to T.
    return detail::from_accumulator<T>(detail::sqrt_value(sum.wide_value()));
}

/**
 * @brief Euclidean norm of a complex vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
template <typename T> inline T nrm2(int n, const std::complex<T> *x, int incx) {
    if (n <= 0 || incx == 0) {
        return T(0);
    }

    detail::real_mac<T> sum;
    int ix = detail::start_index(n, incx);
    for (int i = 0; i < n; ++i) {
        sum.add_product(x[ix].real(), x[ix].real());
        sum.add_product(x[ix].imag(), x[ix].imag());
        ix += incx;
    }
    return detail::from_accumulator<T>(detail::sqrt_value(sum.wide_value()));
}

/**
 * @brief Sum of absolute values of a real vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Sum of absolute values.
 */
template <typename T, detail::enable_if_real_t<T> = 0> inline T asum(int n, const T *x, int incx) {
    if (n <= 0 || incx == 0) {
        return T(0);
    }

    detail::real_mac<T> sum;
    int ix = detail::start_index(n, incx);
    for (int i = 0; i < n; ++i) {
        sum.add(detail::abs_value(x[ix]));
        ix += incx;
    }
    return sum.value();
}

/**
 * @brief Sum of `|Re(x_i)| + |Im(x_i)|` for a complex vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Sum of absolute component values.
 */
template <typename T> inline T asum(int n, const std::complex<T> *x, int incx) {
    if (n <= 0 || incx == 0) {
        return T(0);
    }

    detail::real_mac<T> sum;
    int ix = detail::start_index(n, incx);
    for (int i = 0; i < n; ++i) {
        sum.add(detail::abs_value(x[ix].real()));
        sum.add(detail::abs_value(x[ix].imag()));
        ix += incx;
    }
    return sum.value();
}

/**
 * @brief Apply a real Givens rotation to two real vectors.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param c Cosine-like rotation coefficient.
 * @param s Sine-like rotation coefficient.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline void rot(int n, T *x, int incx, T *y, int incy, T c, T s) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        const T w = x[ix];
        const T z = y[iy];
        x[ix] = (c * w) + (s * z);
        y[iy] = (c * z) - (s * w);
        ix += incx;
        iy += incy;
    }
}

/**
 * @brief Apply a real Givens rotation to two complex vectors.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param c Real cosine-like rotation coefficient.
 * @param s Real sine-like rotation coefficient.
 */
template <typename T>
inline void rot(int n, std::complex<T> *x, int incx, std::complex<T> *y, int incy, T c, T s) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        const std::complex<T> w = x[ix];
        const std::complex<T> z = y[iy];
        x[ix] = (c * w) + (s * z);
        y[iy] = (c * z) - (s * w);
        ix += incx;
        iy += incy;
    }
}

/**
 * @brief Construct real Givens rotation parameters in place.
 * @param a On input: first scalar; on output: rotation radius.
 * @param b On input: second scalar; on output: implementation-defined auxiliary value.
 * @param c Output cosine coefficient.
 * @param s Output sine coefficient.
 */
template <typename T, detail::enable_if_real_t<T> = 0> inline void rotg(T *a, T *b, T *c, T *s) {
    const T abs_a = detail::abs_value(*a);
    const T abs_b = detail::abs_value(*b);
    const T roe = abs_a > abs_b ? *a : *b;
    const T scale = abs_a + abs_b;
    if (scale == T(0)) {
        *c = T(1);
        *s = T(0);
        *a = T(0);
        *b = T(0);
        return;
    }

    T r = scale * detail::sqrt_value(((*a / scale) * (*a / scale)) + ((*b / scale) * (*b / scale)));
    r = detail::signed_magnitude(r, roe);
    *c = *a / r;
    *s = *b / r;
    T z = T(1);
    if (abs_a > abs_b) {
        z = *s;
    } else if (*c != T(0)) {
        z = T(1) / *c;
    }
    *a = r;
    *b = z;
}

/**
 * @brief Construct complex Givens rotation parameters in place.
 * @param a On input: first scalar; on output: rotation radius-like value.
 * @param b Second scalar.
 * @param c Output real cosine coefficient.
 * @param s Output complex sine-like coefficient.
 */
template <typename T>
inline void rotg(std::complex<T> *a, std::complex<T> b, T *c, std::complex<T> *s) {
    const T abs_a = detail::complex_abs(*a);
    if (abs_a == T(0)) {
        *c = T(0);
        *s = std::complex<T>(T(1), T(0));
        *a = b;
        return;
    }

    const T abs_b = detail::complex_abs(b);
    const T scale = abs_a + abs_b;
    const T norm = scale * detail::sqrt_value(detail::norm_square(*a / scale) +
                                              detail::norm_square(b / scale));
    const std::complex<T> alpha = *a / abs_a;
    *c = abs_a / norm;
    *s = alpha * detail::conj_value(b) / norm;
    *a = alpha * norm;
}

/**
 * @brief Apply modified Givens rotation to real vectors using `param`.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param param Pointer to a 5-element modified Givens parameter array.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline void rotm(int n, T *x, int incx, T *y, int incy, const T *param) {
    if (n <= 0 || incx == 0 || incy == 0) {
        return;
    }

    const T flag = param[0];
    if (flag == T(-2)) {
        return;
    }

    const T h11 = param[1];
    const T h21 = param[2];
    const T h12 = param[3];
    const T h22 = param[4];

    int ix = detail::start_index(n, incx);
    int iy = detail::start_index(n, incy);
    for (int i = 0; i < n; ++i) {
        const T w = x[ix];
        const T z = y[iy];
        if (flag < T(0)) {
            x[ix] = (w * h11) + (z * h12);
            y[iy] = (w * h21) + (z * h22);
        } else if (flag == T(0)) {
            x[ix] = w + (z * h12);
            y[iy] = (w * h21) + z;
        } else {
            x[ix] = (w * h11) + z;
            y[iy] = -w + (z * h22);
        }
        ix += incx;
        iy += incy;
    }
}

/**
 * @brief Construct modified Givens parameters for real values.
 * @param d1 Scale factor component, updated in place.
 * @param d2 Scale factor component, updated in place.
 * @param b1 Input/output vector component.
 * @param b2 Input vector component.
 * @param param Output 5-element modified Givens parameter array.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline void rotmg(T *d1, T *d2, T *b1, T b2, T *param) {
    if (*d1 <= T(0) || *d2 <= T(0)) {
        param[0] = T(-2);
        param[1] = T(0);
        param[2] = T(0);
        param[3] = T(0);
        param[4] = T(0);
        return;
    }

    T a = detail::sqrt_value(*d1) * (*b1);
    T b = detail::sqrt_value(*d2) * b2;
    T c = T(0);
    T s = T(0);
    rotg(&a, &b, &c, &s);

    param[0] = T(-1);
    param[1] = c;
    param[2] = -s;
    param[3] = s;
    param[4] = c;

    const T c2 = c * c;
    const T s2 = s * s;
    const T d1_old = *d1;
    const T d2_old = *d2;
    *d1 = (d1_old * c2) + (d2_old * s2);
    *d2 = (d1_old * s2) + (d2_old * c2);
    const T floor = *d1 > detail::tiny_value<T>() ? *d1 : detail::tiny_value<T>();
    *b1 = a / detail::sqrt_value(floor);
}

/**
 * @brief Index of the element with maximum absolute value in a real vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline int iamax(int n, const T *x, int incx) {
    if (n <= 0 || incx <= 0) {
        return 0;
    }

    int best_logical = 0;
    int ix = 0;
    T best = detail::abs_value(x[ix]);
    for (int i = 1; i < n; ++i) {
        ix += incx;
        const T cand = detail::abs_value(x[ix]);
        if (cand > best) {
            best = cand;
            best_logical = i;
        }
    }

    return best_logical + 1;
}

/**
 * @brief Index of the element with maximum `|Re| + |Im|` in a complex vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
template <typename T> inline int iamax(int n, const std::complex<T> *x, int incx) {
    if (n <= 0 || incx <= 0) {
        return 0;
    }

    int best_logical = 0;
    int ix = 0;
    T best = detail::abs1(x[ix]);
    for (int i = 1; i < n; ++i) {
        ix += incx;
        const T cand = detail::abs1(x[ix]);
        if (cand > best) {
            best = cand;
            best_logical = i;
        }
    }

    return best_logical + 1;
}

} // namespace thefblas
