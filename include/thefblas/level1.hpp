// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/detail.hpp"
#include "thefblas/fixed.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

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

/// True when `T` holds the `rotm`/`rotmg` flag values -2, -1, 0 and 1: every
/// supported type except fixed-point formats without an integer bit (`q7`,
/// `q15`, `q31`, whose range is [-1, 1)).
template <typename T> constexpr bool holds_rotm_flags() noexcept {
    if constexpr (is_fixed_v<T>) {
        return std::numeric_limits<typename T::rep>::digits - T::frac_bits >= 1;
    } else {
        return true;
    }
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

/// |z|. For floating-point parts this uses std::hypot, so tiny or huge finite
/// parts neither underflow to 0 nor overflow to infinity; for fixed point the
/// squares are summed exactly in the widened accumulator and the root is
/// rounded once.
template <typename T> inline T complex_abs(const std::complex<T> &value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::hypot(value.real(), value.imag());
    } else {
        sum_of_squares<T> sum;
        sum.add(value.real());
        sum.add(value.imag());
        return sum.root();
    }
}

/// `x * y / z` for fixed point, with the product exact in the wide type and a
/// single rounding (to nearest, ties away from zero, as in `fixed::operator/`).
/// `z` must be nonzero.
template <typename T> inline T mul_div(const T &x, const T &y, const T &z) {
    using wide = widen_t<typename T::rep>;
    using Policy = typename T::policy;
    const auto numerator =
        static_cast<wide>(static_cast<wide>(x.raw()) * static_cast<wide>(y.raw()));
    const auto divisor = static_cast<wide>(z.raw());
    const auto half = static_cast<wide>(divisor / 2);
    const wide biased = ((numerator >= 0) == (divisor >= 0)) ? static_cast<wide>(numerator + half)
                                                             : static_cast<wide>(numerator - half);
    return T::from_raw(
        Policy::template narrow<typename T::rep>(static_cast<wide>(biased / divisor)));
}

template <typename T> inline T signed_magnitude(T magnitude, const T &sign_source) {
    return sign_source < T(0) ? -magnitude : magnitude;
}

/// Overflow- and underflow-safe Euclidean norm accumulator for floating-point
/// types, using Blue's algorithm as in Netlib reference BLAS 3.10+
/// (Anderson, "Algorithm 978: Safe Scaling in the Level 1 BLAS", 2017).
///
/// Values are split into three bins by magnitude. Mid-range values are summed
/// unscaled, so for vectors whose elements are all in the mid range (the
/// common case) the result is bit-identical to `sqrt(sum of x_i * x_i)`.
/// Very large and very small values are scaled by powers of two (exact) into
/// the safe range before squaring, so the result is finite and accurate
/// whenever the norm itself is representable.
template <typename T> class blue_nrm2 {
    static_assert(std::numeric_limits<T>::radix == 2, "blue_nrm2 requires a binary type");

    static int floor_half(int v) { return (v >= 0) ? v / 2 : -((-v + 1) / 2); }
    static int ceil_half(int v) { return -floor_half(-v); }

    using limits = std::numeric_limits<T>;
    static T pow2(int e) { return std::ldexp(T(1), e); }

  public:
    void add(const T &value) {
        const T ax = std::fabs(value);
        if (ax > tbig()) {
            const T scaled = ax * sbig();
            abig_ += scaled * scaled;
            notbig_ = false;
        } else if (ax < tsml()) {
            if (notbig_) {
                const T scaled = ax * ssml();
                asml_ += scaled * scaled;
            }
        } else {
            // Also reached by NaN, which then propagates through amed_.
            amed_ += ax * ax;
        }
    }

    [[nodiscard]] T value() const {
        const T one(1);
        T amed = amed_;
        const bool amed_used = amed > T(0) || amed > (limits::max)() || amed != amed;
        if (abig_ > T(0)) {
            T abig = abig_;
            if (amed_used) {
                abig += (amed * sbig()) * sbig();
            }
            return (one / sbig()) * std::sqrt(abig);
        }
        if (asml_ > T(0)) {
            if (amed_used) {
                amed = std::sqrt(amed);
                const T asml = std::sqrt(asml_) / ssml();
                const T ymin = (asml > amed) ? amed : asml;
                const T ymax = (asml > amed) ? asml : amed;
                const T ratio = ymin / ymax;
                return std::sqrt(ymax * ymax * (one + ratio * ratio));
            }
            return (one / ssml()) * std::sqrt(asml_);
        }
        return std::sqrt(amed);
    }

  private:
    // Blue's thresholds and scaling factors, all exact powers of two.
    // The medium/big cutoff leaves headroom for up to 2 * INT_MAX squares
    // (complex vectors of maximal int length), which is stricter than Netlib's
    // 2^(digits - 1) for types with fewer than 33 significand bits (float).
    static constexpr int headroom_bits() {
        return (limits::digits - 1 > std::numeric_limits<int>::digits + 2)
                   ? limits::digits - 1
                   : std::numeric_limits<int>::digits + 2;
    }
    static T tsml() { return pow2(ceil_half(limits::min_exponent - 1)); }
    static T tbig() { return pow2(floor_half(limits::max_exponent - headroom_bits())); }
    static T ssml() { return pow2(-floor_half(limits::min_exponent - limits::digits)); }
    static T sbig() { return pow2(-ceil_half(limits::max_exponent + limits::digits - 1)); }

    T asml_{};
    T amed_{};
    T abig_{};
    bool notbig_ = true;
};

/// The wider type in which `sdsdot` and `dsdot` accumulate: `double` for
/// `float`, and for `fixed<I, F, P>` the accumulator type `fixed<wider(I), F,
/// P>`. Fixed-point types are supported only where an integer wider than the
/// products exists for the running sum (8- and 16-bit storage, and 32-bit
/// storage with a 128-bit integer), so that the sum is exact (for 8-bit
/// storage up to 2^17 maximal products, as in `real_mac`); 64-bit storage, and
/// 32-bit storage without `__int128`, are rejected at compile time.
template <typename T, typename = void> struct has_extended : std::is_same<T, float> {};
template <typename T>
struct has_extended<T, typename std::enable_if<is_fixed_v<T>>::type>
    : std::integral_constant<bool, can_widen_v<typename T::rep>> {};
template <typename T> constexpr bool has_extended_v = has_extended<T>::value;

template <typename T, typename = void> struct extended {
    static_assert(has_extended_v<T>,
                  "sdsdot/dsdot support float, and fixed-point types whose products have a "
                  "wider integer for the running sum (not 64-bit storage, nor 32-bit storage "
                  "without __int128)");
};
template <> struct extended<float> {
    using type = double;
};
template <typename T>
struct extended<T, typename std::enable_if<has_extended_v<T> && is_fixed_v<T>>::type> {
    using type = accumulator_t<T>;
};
template <typename T> using extended_t = typename extended<T>::type;

/// `init + x^T y` summed in `extended_t<T>`; just `init` for `n <= 0` or a
/// zero stride. For `float`, `init` starts the `double` sum, as in Netlib's
/// `sdsdot`. For fixed point, `init` and the unshifted products are summed
/// exactly at `2 * FracBits` and rounded once.
template <typename T>
inline extended_t<T> extended_dot(const T &init, int n, const T *x, int incx, const T *y,
                                  int incy) {
    const bool empty = n <= 0 || incx == 0 || incy == 0;
    int ix = empty ? 0 : start_index(n, incx);
    int iy = empty ? 0 : start_index(n, incy);
    const int count = empty ? 0 : n;
    if constexpr (std::is_same_v<T, float>) {
        double sum = static_cast<double>(init);
        for (int i = 0; i < count; ++i) {
            sum += static_cast<double>(x[ix]) * static_cast<double>(y[iy]);
            ix += incx;
            iy += incy;
        }
        return sum;
    } else {
        real_mac<T> acc;
        acc.add(init);
        for (int i = 0; i < count; ++i) {
            acc.add_product(x[ix], y[iy]);
            ix += incx;
            iy += incy;
        }
        return acc.wide_value();
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
template <typename T> inline void swap(int n, T *x, int incx, T *y, int incy) noexcept {
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
template <typename T> inline void copy(int n, const T *x, int incx, T *y, int incy) noexcept {
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
template <typename T>
inline void axpy(int n, T alpha, const T *x, int incx, T *y, int incy) noexcept {
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
 * @param incx Stride between elements of x; must be positive (otherwise, as in
 *        Netlib BLAS, the call is a no-op).
 */
template <typename T> inline void scal(int n, T alpha, T *x, int incx) noexcept {
    if (n <= 0 || incx <= 0) {
        return;
    }

    int ix = 0;
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
 * @param incx Stride between elements of x; must be positive (otherwise, as in
 *        Netlib BLAS, the call is a no-op).
 */
template <typename T> inline void scal(int n, T alpha, std::complex<T> *x, int incx) noexcept {
    if (n <= 0 || incx <= 0) {
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
inline T dot(int n, const T *x, int incx, const T *y, int incy) noexcept {
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
                            int incy) noexcept {
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
                            int incy) noexcept {
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
 * @brief Dot product accumulated in extended precision (Netlib `dsdot`).
 *
 * Returns `x^T y` in `detail::extended_t<T>`: `double` for `float` input, and
 * the widened fixed-point type (same `FracBits`, twice the integer width) for
 * `fixed` input, without narrowing back to `T`. Supports `float` and the
 * fixed-point types with a wider accumulator (see `detail::extended`).
 * @return The sum, or zero when `n <= 0` or a stride is zero.
 */
template <typename T>
inline detail::extended_t<T> dsdot(int n, const T *x, int incx, const T *y, int incy) noexcept {
    return detail::extended_dot(T{}, n, x, incx, y, incy);
}

/**
 * @brief `sb + x^T y` accumulated in extended precision (Netlib `sdsdot`).
 *
 * `sb` and the products are summed in `detail::extended_t<T>` (see `dsdot`) and
 * the result is narrowed to `T` once, through the overflow policy for fixed
 * point, which rounds `sb + x^T y` exactly once.
 * @return The result, or `sb` when `n <= 0` or a stride is zero.
 */
template <typename T>
inline T sdsdot(int n, T sb, const T *x, int incx, const T *y, int incy) noexcept {
    const detail::extended_t<T> sum = detail::extended_dot(sb, n, x, incx, y, incy);
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(sum);
    } else {
        return detail::from_accumulator<T>(sum);
    }
}

/**
 * @brief Euclidean norm of a real vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline T nrm2(int n, const T *x, int incx) noexcept {
    if (n <= 0 || incx == 0) {
        return T(0);
    }

    int ix = detail::start_index(n, incx);
    if constexpr (std::is_floating_point_v<T>) {
        detail::blue_nrm2<T> norm;
        for (int i = 0; i < n; ++i) {
            norm.add(x[ix]);
            ix += incx;
        }
        return norm.value();
    } else {
        detail::sum_of_squares<T> sum;
        for (int i = 0; i < n; ++i) {
            sum.add(x[ix]);
            ix += incx;
        }
        // The sum of squares can legitimately exceed the range of T even when the
        // norm itself does not, so the square root is taken of the exact sum in the
        // accumulator, rounded once, and only its result is narrowed back to T.
        return sum.root();
    }
}

/**
 * @brief Euclidean norm of a complex vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
template <typename T> inline T nrm2(int n, const std::complex<T> *x, int incx) noexcept {
    if (n <= 0 || incx == 0) {
        return T(0);
    }

    int ix = detail::start_index(n, incx);
    if constexpr (std::is_floating_point_v<T>) {
        detail::blue_nrm2<T> norm;
        for (int i = 0; i < n; ++i) {
            norm.add(x[ix].real());
            norm.add(x[ix].imag());
            ix += incx;
        }
        return norm.value();
    } else {
        detail::sum_of_squares<T> sum;
        for (int i = 0; i < n; ++i) {
            sum.add(x[ix].real());
            sum.add(x[ix].imag());
            ix += incx;
        }
        return sum.root();
    }
}

/**
 * @brief Sum of absolute values of a real vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Sum of absolute values, or 0 when `n <= 0` or `incx <= 0` (as in
 *         Netlib BLAS).
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline T asum(int n, const T *x, int incx) noexcept {
    if (n <= 0 || incx <= 0) {
        return T(0);
    }

    detail::real_mac<T> sum;
    int ix = 0;
    for (int i = 0; i < n; ++i) {
        sum.add_abs(x[ix]);
        ix += incx;
    }
    return sum.value();
}

/**
 * @brief Sum of `|Re(x_i)| + |Im(x_i)|` for a complex vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Sum of absolute component values, or 0 when `n <= 0` or `incx <= 0`
 *         (as in Netlib BLAS).
 */
template <typename T> inline T asum(int n, const std::complex<T> *x, int incx) noexcept {
    if (n <= 0 || incx <= 0) {
        return T(0);
    }

    detail::real_mac<T> sum;
    int ix = 0;
    for (int i = 0; i < n; ++i) {
        sum.add_abs(x[ix].real());
        sum.add_abs(x[ix].imag());
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
inline void rot(int n, T *x, int incx, T *y, int incy, T c, T s) noexcept {
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
inline void rot(int n, std::complex<T> *x, int incx, std::complex<T> *y, int incy, T c,
                T s) noexcept {
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
template <typename T, detail::enable_if_real_t<T> = 0>
inline void rotg(T *a, T *b, T *c, T *s) noexcept {
    const T abs_a = detail::abs_value(*a);
    const T abs_b = detail::abs_value(*b);
    const T roe = abs_a > abs_b ? *a : *b;
    if (*a == T(0) && *b == T(0)) {
        *c = T(1);
        *s = T(0);
        *a = T(0);
        *b = T(0);
        return;
    }

    T r{};
    if constexpr (std::is_floating_point_v<T>) {
        const T scale = abs_a + abs_b;
        r = scale *
            detail::sqrt_value(((*a / scale) * (*a / scale)) + ((*b / scale) * (*b / scale)));
    } else {
        // a^2 + b^2 is exact in the widened accumulator, so the radius is its
        // square root rounded once, without the scaling floating point needs.
        detail::sum_of_squares<T> sum;
        sum.add(*a);
        sum.add(*b);
        r = sum.root();
    }
    r = detail::signed_magnitude(r, roe);
    *c = *a / r;
    *s = *b / r;
    // 1 is only constructed on the branches that return it (or its quotient):
    // fixed-point formats without integer bits (q15) cannot represent it, and
    // there the policy resolves such a z, while z = s is always representable.
    T z{};
    if (abs_a > abs_b) {
        z = *s;
    } else if (*c != T(0)) {
        z = T(1) / *c;
    } else {
        z = T(1);
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
inline void rotg(std::complex<T> *a, std::complex<T> b, T *c, std::complex<T> *s) noexcept {
    const T abs_a = detail::complex_abs(*a);
    const T abs_b = detail::complex_abs(b);
    if (abs_a == T(0)) {
        // As in Netlib reference BLAS 3.10+: r = |b| is real and s = conj(b)/|b|,
        // and the rotation is the identity when b is zero too.
        if (abs_b == T(0)) {
            *c = T(1);
            *s = std::complex<T>(T(0), T(0));
            return;
        }
        *c = T(0);
        *s = detail::conj_value(b) / abs_b;
        *a = std::complex<T>(abs_b, T(0));
        return;
    }

    T norm{};
    if constexpr (std::is_floating_point_v<T>) {
        const T scale = abs_a + abs_b;
        norm = scale *
               detail::sqrt_value(detail::norm_square(*a / scale) + detail::norm_square(b / scale));
    } else {
        // As for real rotg: the sum of squares is exact in the accumulator.
        detail::sum_of_squares<T> sum;
        sum.add(a->real());
        sum.add(a->imag());
        sum.add(b.real());
        sum.add(b.imag());
        norm = sum.root();
    }
    const std::complex<T> alpha = *a / abs_a;
    *c = abs_a / norm;
    *s = alpha * detail::conj_value(b) / norm;
    if constexpr (std::is_floating_point_v<T>) {
        *a = alpha * norm;
    } else {
        // a * (norm / |a|) with one rounding per part: multiplying the rounded
        // alpha by norm would scale its rounding error by norm.
        *a = std::complex<T>(detail::mul_div(a->real(), norm, abs_a),
                             detail::mul_div(a->imag(), norm, abs_a));
    }
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
inline void rotm(int n, T *x, int incx, T *y, int incy, const T *param) noexcept {
    static_assert(detail::holds_rotm_flags<T>(),
                  "rotm: the element type must represent the flag values -2 and 1; "
                  "use a fixed-point format with at least one integer bit");
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
 *
 * Follows Netlib reference BLAS `srotmg`/`drotmg`: on return, the matrix `H`
 * encoded in `param` (see rotm) satisfies `H * (b1, b2)^T = (b1', 0)^T`, and
 * `d1`, `d2`, `b1` hold the updated scale factors and first component.
 * `param[0]` is the flag: -2 (H is the identity, nothing else is written),
 * -1 (full H), 0 (unit diagonal: only `param[2]` and `param[3]` are written)
 * or 1 (only `param[1]` and `param[4]` are written). A negative `d1`, or a
 * negative `d2` that makes the problem ill-posed, gives flag -1 with H, d1, d2
 * and b1 all zero.
 *
 * For floating-point types the magnitudes of nonzero `d1` and `d2` are kept in
 * [2^-24, 2^24] by exact rescaling, as in Netlib (a valid negative `d2` keeps
 * its sign). Fixed-point types skip that step: their range is already bounded
 * and the rescaling constants would not be representable in most Q formats.
 *
 * @param d1 Scale factor component, updated in place.
 * @param d2 Scale factor component, updated in place.
 * @param b1 Input/output vector component.
 * @param b2 Input vector component.
 * @param param Output 5-element modified Givens parameter array.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline void rotmg(T *d1, T *d2, T *b1, T b2, T *param) noexcept {
    static_assert(detail::holds_rotm_flags<T>(),
                  "rotmg: the element type must represent the flag values -2 and 1; "
                  "use a fixed-point format with at least one integer bit");
    const T zero(0);
    const T one(1);
    T flag = -one;
    T h11 = zero;
    T h21 = zero;
    T h12 = zero;
    T h22 = zero;

    bool reject = *d1 < zero;
    if (!reject) {
        const T p2 = *d2 * b2;
        if (p2 == zero) {
            param[0] = T(-2);
            return;
        }
        const T p1 = *d1 * (*b1);
        const T q2 = p2 * b2;
        const T q1 = p1 * (*b1);
        if (detail::abs_value(q1) > detail::abs_value(q2)) {
            h21 = -b2 / *b1;
            h12 = p2 / p1;
            const T u = one - (h12 * h21);
            // Netlib: u <= 0 can only come from rounding.
            reject = !(u > zero);
            if (!reject) {
                flag = zero;
                *d1 /= u;
                *d2 /= u;
                *b1 *= u;
            }
        } else {
            reject = q2 < zero;
            if (!reject) {
                flag = one;
                h11 = p1 / p2;
                h22 = *b1 / b2;
                const T u = one + (h11 * h22);
                const T temp = *d2 / u;
                *d2 = *d1 / u;
                *d1 = temp;
                *b1 = b2 * u;
            }
        }
    }

    if (reject) {
        flag = -one;
        h11 = zero;
        h21 = zero;
        h12 = zero;
        h22 = zero;
        *d1 = zero;
        *d2 = zero;
        *b1 = zero;
    } else if constexpr (std::is_floating_point_v<T>) {
        const T gam(4096);
        const T gamsq(16777216);
        const T rgamsq = static_cast<T>(5.9604645e-8); // Netlib RGAMSQ, about 2^-24
        // Switch to the full form of H before rescaling its entries.
        const auto make_full = [&]() {
            if (flag == zero) {
                h11 = one;
                h22 = one;
            } else if (flag > zero) {
                h21 = -one;
                h12 = one;
            }
            flag = -one;
        };
        if (*d1 != zero) {
            while (*d1 <= rgamsq || *d1 >= gamsq) {
                make_full();
                if (*d1 <= rgamsq) {
                    *d1 *= gam * gam;
                    *b1 /= gam;
                    h11 /= gam;
                    h12 /= gam;
                } else {
                    *d1 /= gam * gam;
                    *b1 *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }
        if (*d2 != zero) {
            while (detail::abs_value(*d2) <= rgamsq || detail::abs_value(*d2) >= gamsq) {
                make_full();
                if (detail::abs_value(*d2) <= rgamsq) {
                    *d2 *= gam * gam;
                    h21 /= gam;
                    h22 /= gam;
                } else {
                    *d2 /= gam * gam;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }

    if (flag < zero) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == zero) {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
    param[0] = flag;
}

/**
 * @brief Index of the element with maximum absolute value in a real vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
template <typename T, detail::enable_if_real_t<T> = 0>
inline int iamax(int n, const T *x, int incx) noexcept {
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
template <typename T> inline int iamax(int n, const std::complex<T> *x, int incx) noexcept {
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

// ---------------------------------------------------------------------------
// Contiguous-range overloads
// ---------------------------------------------------------------------------

namespace detail {

template <typename R> using range_pointer_t = decltype(std::data(std::declval<R &>()));

/// True for anything with `std::data` returning a pointer and `std::size`:
/// `std::vector`, `std::array`, C arrays, and C++20 `std::span`.
template <typename R, typename = void> struct is_contiguous_range : std::false_type {};

template <typename R>
struct is_contiguous_range<
    R, std::void_t<range_pointer_t<R>, decltype(std::size(std::declval<R &>()))>>
    : std::is_pointer<range_pointer_t<R>> {};

template <typename... R>
using enable_if_ranges_t =
    typename std::enable_if<(is_contiguous_range<std::remove_reference_t<R>>::value && ...),
                            int>::type;

/// Number of elements of a range as the `int` count of the pointer API. A size
/// above `INT_MAX` asserts and becomes 0, which makes every routine a no-op.
template <typename R> constexpr int range_count(const R &r) noexcept {
    const auto size = std::size(r);
    constexpr auto int_max = static_cast<std::size_t>((std::numeric_limits<int>::max)());
    assert(static_cast<std::size_t>(size) <= int_max && "thefblas: range larger than INT_MAX");
    return static_cast<std::size_t>(size) <= int_max ? static_cast<int>(size) : 0;
}

/// Common element count of two ranges. Mismatched sizes assert and become 0.
template <typename X, typename Y> constexpr int range_count(const X &x, const Y &y) noexcept {
    const bool same =
        static_cast<std::size_t>(std::size(x)) == static_cast<std::size_t>(std::size(y));
    assert(same && "thefblas: ranges have different sizes");
    return same ? range_count(x) : 0;
}

} // namespace detail

/**
 * @name Contiguous-range overloads
 * Level-1 routines over whole contiguous ranges with unit stride, e.g.
 * `dot(x, y)` for two `std::vector<double>`. The element count comes from
 * `std::size`. Two ranges of different sizes, or a size above `INT_MAX`, assert
 * in debug builds and make the call a no-op (or return zero) otherwise. Scalars
 * and element types follow the pointer overloads, which these forward to.
 * There is deliberately no range `swap`, which would compete with `std::swap`
 * through argument-dependent lookup.
 * @{
 */

/// `y := x` over two ranges of equal size.
template <typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline void copy(X &&x, Y &&y) noexcept {
    copy(detail::range_count(x, y), std::data(x), 1, std::data(y), 1);
}

/// `y := alpha * x + y` over two ranges of equal size.
template <typename A, typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline void axpy(A alpha, X &&x, Y &&y) noexcept {
    axpy(detail::range_count(x, y), alpha, std::data(x), 1, std::data(y), 1);
}

/// `x := alpha * x`; a real `alpha` with a complex range scales both parts.
template <typename A, typename X, detail::enable_if_ranges_t<X> = 0>
inline void scal(A alpha, X &&x) noexcept {
    scal(detail::range_count(x), alpha, std::data(x), 1);
}

/// Dot product `x^T y` of two ranges of equal size.
template <typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline auto dot(X &&x, Y &&y) noexcept {
    return dot(detail::range_count(x, y), std::data(x), 1, std::data(y), 1);
}

/// Unconjugated complex dot product `x^T y`.
template <typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline auto dotu(X &&x, Y &&y) noexcept {
    return dotu(detail::range_count(x, y), std::data(x), 1, std::data(y), 1);
}

/// Conjugated complex dot product `x^H y`.
template <typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline auto dotc(X &&x, Y &&y) noexcept {
    return dotc(detail::range_count(x, y), std::data(x), 1, std::data(y), 1);
}

/// Dot product in extended precision, as `dsdot`.
template <typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline auto dsdot(X &&x, Y &&y) noexcept {
    return dsdot(detail::range_count(x, y), std::data(x), 1, std::data(y), 1);
}

/// `sb + x^T y` in extended precision, as `sdsdot`.
template <typename T, typename X, typename Y, detail::enable_if_ranges_t<X, Y> = 0>
inline T sdsdot(T sb, X &&x, Y &&y) noexcept {
    return sdsdot(detail::range_count(x, y), sb, std::data(x), 1, std::data(y), 1);
}

/// Euclidean norm of a range.
template <typename X, detail::enable_if_ranges_t<X> = 0> inline auto nrm2(X &&x) noexcept {
    return nrm2(detail::range_count(x), std::data(x), 1);
}

/// Sum of absolute values (`|Re| + |Im|` for complex) of a range.
template <typename X, detail::enable_if_ranges_t<X> = 0> inline auto asum(X &&x) noexcept {
    return asum(detail::range_count(x), std::data(x), 1);
}

/// Netlib-style **1-based** index of the largest element, as in the pointer
/// overload; 0 for an empty range.
template <typename X, detail::enable_if_ranges_t<X> = 0> inline int iamax(X &&x) noexcept {
    return iamax(detail::range_count(x), std::data(x), 1);
}

/// Apply a plane rotation to two ranges of equal size.
template <typename X, typename Y, typename C, typename S, detail::enable_if_ranges_t<X, Y> = 0>
inline void rot(X &&x, Y &&y, C c, S s) noexcept {
    rot(detail::range_count(x, y), std::data(x), 1, std::data(y), 1, c, s);
}

/// Apply a modified Givens rotation (`param` as produced by `rotmg`) to two
/// ranges of equal size.
template <typename X, typename Y, typename T, detail::enable_if_ranges_t<X, Y> = 0>
inline void rotm(X &&x, Y &&y, const T *param) noexcept {
    rotm(detail::range_count(x, y), std::data(x), 1, std::data(y), 1, param);
}

/** @} */

} // namespace thefblas
