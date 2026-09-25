// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/fixed.hpp"

#include <complex>
#include <type_traits>

namespace thefblas {

/**
 * @file detail.hpp
 * @brief Internal helpers shared by the Level 1 and Level 2 headers.
 *
 * These live in a single header so that the routine headers can be included
 * together (see thefblas.h) without redefining the helpers.
 *
 * Besides the small index/conjugation utilities, this header defines the
 * **accumulator** machinery that the BLAS routines use for their internal
 * reductions. Fixed-point arithmetic has no exponent, so a naive running sum
 * overflows the caller's Q format long before the mathematical result does:
 * in Q1.15 (range [-1, 1)) the inner product of `[0.9, 0.9]` with itself is
 * 1.62 and cannot be represented at all, even though every input and every
 * individual product can. Accumulating in a type twice as wide removes that
 * failure mode for the intermediate values, and only the final, user-visible
 * result is narrowed back through the element type's overflow policy.
 */

namespace detail {

/// Index of the first processed element for a vector of length `n` and stride
/// `inc`; negative strides start at the far end, as in Netlib BLAS.
inline int start_index(int n, int inc) {
    return (inc > 0) ? 0 : (1 - n) * inc;
}

template <typename T> inline T conj_value(const T &value) {
    return value;
}

template <typename T> inline std::complex<T> conj_value(const std::complex<T> &value) {
    using std::conj;
    return conj(value);
}

template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

template <typename T> using enable_if_real_t = typename std::enable_if<!is_complex_v<T>, int>::type;

// ---------------------------------------------------------------------------
// Widened accumulators
// ---------------------------------------------------------------------------

/// Type used for internal reductions over elements of type `T`.
///
/// - `float`/`double` map to themselves, so floating-point results stay
///   bit-identical to Netlib BLAS (which accumulates `sdot` in `float`).
/// - `fixed<I, F, P>` maps to `fixed<wider(I), F, P>`.
/// - `std::complex<T>` lifts element-wise.
template <typename T> struct accumulator {
    using type = typename wider_fixed<T>::type;
};

template <typename T> struct accumulator<std::complex<T>> {
    using type = std::complex<typename accumulator<T>::type>;
};

template <typename T> using accumulator_t = typename accumulator<T>::type;

/// Widens an element value into its accumulator type. Always exact.
template <typename T> inline accumulator_t<T> to_accumulator(const T &value) {
    if constexpr (is_complex_v<T>) {
        using R = typename T::value_type;
        return accumulator_t<T>(to_accumulator<R>(value.real()), to_accumulator<R>(value.imag()));
    } else if constexpr (is_fixed_v<T>) {
        using Rep = typename accumulator_t<T>::rep;
        return accumulator_t<T>::from_raw(static_cast<Rep>(value.raw()));
    } else {
        return value;
    }
}

/// Narrows an accumulator back to the element type, applying the element
/// type's overflow policy exactly once.
template <typename T> inline T from_accumulator(const accumulator_t<T> &value) {
    if constexpr (is_complex_v<T>) {
        using R = typename T::value_type;
        return T(from_accumulator<R>(value.real()), from_accumulator<R>(value.imag()));
    } else if constexpr (is_fixed_v<T>) {
        using Policy = typename T::policy;
        return T::from_raw(Policy::template narrow<typename T::rep>(value.raw()));
    } else {
        return value;
    }
}

/// Product of two element values evaluated in the accumulator type, for use as
/// a drop-in replacement for `a * b` inside a reduction loop.
template <typename T> inline accumulator_t<T> acc_mul(const T &a, const T &b) {
    return to_accumulator(a) * to_accumulator(b);
}

// ---------------------------------------------------------------------------
// Exact multiply-accumulate for the Level 1 reductions
// ---------------------------------------------------------------------------

/// Running sum of products of real values.
///
/// For floating point this is the plain `sum += a * b` loop of Netlib BLAS.
/// For fixed point the products are accumulated **unshifted**, that is at
/// `2 * FracBits` fractional bits in the widened integer type, so neither the
/// per-product rounding nor the intermediate overflow of the naive loop
/// occurs; the single rounding and the single policy narrowing happen in
/// `value()`.
template <typename T, typename Enable = void> class real_mac {
  public:
    void add_product(const T &a, const T &b) { sum_ = sum_ + a * b; }
    void subtract_product(const T &a, const T &b) { sum_ = sum_ - a * b; }
    void add(const T &a) { sum_ = sum_ + a; }
    T value() const { return sum_; }
    T wide_value() const { return sum_; }

  private:
    T sum_{};
};

template <typename T> class real_mac<T, typename std::enable_if<is_fixed_v<T>>::type> {
    using wide = widen_t<typename T::rep>;

  public:
    void add_product(const T &a, const T &b) {
        sum_ = static_cast<wide>(sum_ + static_cast<wide>(a.raw()) * static_cast<wide>(b.raw()));
    }
    void subtract_product(const T &a, const T &b) {
        sum_ = static_cast<wide>(sum_ - static_cast<wide>(a.raw()) * static_cast<wide>(b.raw()));
    }
    /// Adds a plain value, lifting it to the accumulator's `2 * FracBits` scale.
    void add(const T &a) {
        sum_ = static_cast<wide>(sum_ + (static_cast<wide>(a.raw()) << T::frac_bits));
    }

    /// The sum as a value of the *widened* fixed-point type, rounded once from
    /// `2 * FracBits` down to `FracBits`. Use this when the mathematical result
    /// may legitimately exceed the range of `T`, as for a sum of squares.
    accumulator_t<T> wide_value() const {
        using Acc = accumulator_t<T>;
        return Acc::from_raw(static_cast<typename Acc::rep>(round_shift(sum_, T::frac_bits)));
    }

    T value() const { return from_accumulator<T>(wide_value()); }

  private:
    wide sum_{};
};

/// Running sum of products of complex values, built from two real accumulators
/// so that it inherits the exactness described above.
template <typename T> class complex_mac {
  public:
    using value_type = std::complex<T>;

    /// `sum += a * b`
    void add_product(const value_type &a, const value_type &b) {
        real_.add_product(a.real(), b.real());
        real_.subtract_product(a.imag(), b.imag());
        imag_.add_product(a.real(), b.imag());
        imag_.add_product(a.imag(), b.real());
    }

    /// `sum += conj(a) * b`
    void add_conj_product(const value_type &a, const value_type &b) {
        real_.add_product(a.real(), b.real());
        real_.add_product(a.imag(), b.imag());
        imag_.add_product(a.real(), b.imag());
        imag_.subtract_product(a.imag(), b.real());
    }

    value_type value() const { return value_type(real_.value(), imag_.value()); }

  private:
    real_mac<T> real_;
    real_mac<T> imag_;
};

} // namespace detail

} // namespace thefblas
