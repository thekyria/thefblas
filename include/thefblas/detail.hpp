// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/fixed.hpp"

#include <climits>
#include <complex>
#include <limits>
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

/// True for `float`/`double` and for `std::complex` of them: the element types
/// whose results must stay bit-identical to Netlib BLAS.
template <typename T> struct is_floating : std::is_floating_point<T> {};

template <typename T> struct is_floating<std::complex<T>> : std::is_floating_point<T> {};

template <typename T> constexpr bool is_floating_v = is_floating<T>::value;

template <typename T> struct value_constants {
    static T zero() { return T{}; }
    static T one() { return T{1}; }
};

template <typename T> struct value_constants<std::complex<T>> {
    static std::complex<T> zero() { return std::complex<T>(T{}, T{}); }
    static std::complex<T> one() { return std::complex<T>(T{1}, T{}); }
};

/// True when `value` is exactly 1. Unlike a comparison with
/// `value_constants<T>::one()`, this does not construct 1, which fixed-point
/// formats without integer bits (`q7`, `q15`, `q31`) cannot represent: there
/// `checked` would assert, although such a `value` simply is never 1.
template <typename T> inline bool is_one(const T &value) {
    if constexpr (is_complex_v<T>) {
        return is_one(value.real()) && value.imag() == typename T::value_type{};
    } else if constexpr (is_fixed_v<T>) {
        using Rep = typename T::rep;
        if constexpr (T::frac_bits >= std::numeric_limits<Rep>::digits) {
            return false;
        } else {
            return value.raw() == static_cast<Rep>(Rep(1) << T::frac_bits);
        }
    } else {
        return value == T(1);
    }
}

/// `numerator / denominator`, also for `std::complex` of fixed-point types.
///
/// The standard only specifies `std::complex<T>` for floating-point `T`, and
/// MSVC's generic complex division does not compile for `fixed` (it compares
/// the parts against `int` literals). For complex non-floating types this uses
/// Smith's algorithm, which scales by the larger part of the denominator so
/// intermediate values stay small, a property fixed-point needs as well.
template <typename T> inline T divide_value(const T &numerator, const T &denominator) {
    if constexpr (is_complex_v<T> && !is_floating_v<T>) {
        using R = typename T::value_type;
        const R ar = numerator.real();
        const R ai = numerator.imag();
        const R br = denominator.real();
        const R bi = denominator.imag();
        const R abs_br = (br < R{}) ? -br : br;
        const R abs_bi = (bi < R{}) ? -bi : bi;
        if (abs_br >= abs_bi) {
            const R ratio = bi / br;
            const R scale = br + bi * ratio;
            return T((ar + ai * ratio) / scale, (ai - ar * ratio) / scale);
        }
        const R ratio = br / bi;
        const R scale = bi + br * ratio;
        return T((ar * ratio + ai) / scale, (ai * ratio - ar) / scale);
    } else {
        return numerator / denominator;
    }
}

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

/// Computes `beta * y + alpha * value` with the `beta`-scaled output element
/// and the scaled sum held together in the accumulator type, and narrows back
/// to the element type, applying the overflow policy exactly once. `y` must be
/// the original (not yet `beta`-scaled) output element; as in Netlib BLAS it is
/// not read when `beta` is zero.
template <typename T>
inline T acc_scale_add_and_narrow(const T &beta, const T &y, const T &alpha,
                                  const accumulator_t<T> &value) {
    using Acc = accumulator_t<T>;
    Acc scaled_y = value_constants<Acc>::zero();
    if (is_one(beta)) {
        scaled_y = to_accumulator(y);
    } else if (beta != value_constants<T>::zero()) {
        scaled_y = to_accumulator(y) * to_accumulator(beta);
    }
    return from_accumulator<T>(scaled_y + to_accumulator(alpha) * value);
}

// ---------------------------------------------------------------------------
// Exact multiply-accumulate for the Level 1 and Level 2 reductions
// ---------------------------------------------------------------------------

/// Adds two integers with two's-complement wrapping instead of undefined
/// signed overflow. Overflow of the running sums below is resolved by the
/// single policy narrowing in `wide_value()`, never by the addition itself.
template <typename I> constexpr I wrap_add(I a, I b) noexcept {
    using U = make_unsigned_integer_t<I>;
    return wrap_narrow<I>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
}

/// Subtracts two integers with two's-complement wrapping; see `wrap_add`.
template <typename I> constexpr I wrap_sub(I a, I b) noexcept {
    using U = make_unsigned_integer_t<I>;
    return wrap_narrow<I>(static_cast<U>(static_cast<U>(a) - static_cast<U>(b)));
}

/// Square root of an unsigned integer, rounded to nearest. Ties cannot occur:
/// `(r + 1/2)^2 = r^2 + r + 1/4` is never an integer.
template <typename U> constexpr U isqrt_rounded(U value) noexcept {
    U rem = value;
    U root = 0;
    U bit = static_cast<U>(U(1) << (sizeof(U) * CHAR_BIT - 2));
    while (bit > rem) {
        bit = static_cast<U>(bit >> 2U);
    }
    while (bit != 0) {
        if (rem >= root + bit) {
            rem = static_cast<U>(rem - (root + bit));
            root = static_cast<U>((root >> 1U) + bit);
        } else {
            root = static_cast<U>(root >> 1U);
        }
        bit = static_cast<U>(bit >> 2U);
    }
    // root = floor(sqrt(value)) and rem = value - root^2; round up when
    // value > root^2 + root, i.e. when value exceeds (root + 1/2)^2.
    return rem > root ? static_cast<U>(root + 1) : root;
}

/// Running sum of products of real values.
///
/// For floating point this is the plain `sum += a * b` loop of Netlib BLAS.
/// For fixed point the products are accumulated **unshifted**, that is at
/// `2 * FracBits` fractional bits, so neither the per-product rounding nor the
/// intermediate overflow of the naive loop occurs; the single rounding and the
/// single policy narrowing happen in `value()`.
template <typename T, typename Enable = void> class real_mac {
  public:
    void add_product(const T &a, const T &b) { sum_ = sum_ + a * b; }
    void subtract_product(const T &a, const T &b) { sum_ = sum_ - a * b; }
    void add_conj_product(const T &a, const T &b) { add_product(a, b); }
    void subtract_conj_product(const T &a, const T &b) { subtract_product(a, b); }
    void add(const T &a) { sum_ = sum_ + a; }
    void add_abs(const T &a) { sum_ = sum_ + (a < T(0) ? -a : a); }
    T value() const { return sum_; }
    T wide_value() const { return sum_; }

  private:
    T sum_{};
};

template <typename T> class real_mac<T, typename std::enable_if<is_fixed_v<T>>::type> {
    using wide = widen_t<typename T::rep>;
    // Every product is exact in `wide`. The running sum takes one more
    // widening step where the platform provides one, so that no sum over a
    // vector whose length an `int` can express overflows it. Where `wide` is
    // already the widest available integer (Q1.31 products in int64 without
    // __int128), additions past its range wrap modulo 2^bits — well defined,
    // never UB — and are then resolved by the policy narrowing in
    // `wide_value()`; sums are exact only while `|sum|` stays below
    // `2^(bits(acc) - 1 - 2 * FracBits)`.
    using acc = widen_or_same_t<wide>;

  public:
    void add_product(const T &a, const T &b) {
        sum_ =
            wrap_add(sum_, static_cast<acc>(static_cast<acc>(a.raw()) * static_cast<acc>(b.raw())));
    }
    void subtract_product(const T &a, const T &b) {
        sum_ =
            wrap_sub(sum_, static_cast<acc>(static_cast<acc>(a.raw()) * static_cast<acc>(b.raw())));
    }
    void add_conj_product(const T &a, const T &b) { add_product(a, b); }
    void subtract_conj_product(const T &a, const T &b) { subtract_product(a, b); }
    /// Adds a plain value, lifting it to the accumulator's `2 * FracBits` scale.
    void add(const T &a) {
        sum_ = wrap_add(sum_, scale_pow2(static_cast<acc>(a.raw()), T::frac_bits));
    }
    /// Adds `|a|` computed on the raw representation in the accumulator type,
    /// so that `abs(min)` — not representable in `T` — is summed exactly
    /// instead of being clamped or wrapped by the element policy first.
    void add_abs(const T &a) {
        const acc raw = static_cast<acc>(a.raw());
        sum_ = wrap_add(sum_, scale_pow2(raw < 0 ? static_cast<acc>(-raw) : raw, T::frac_bits));
    }

    /// The sum as a value of the *widened* fixed-point type, rounded once from
    /// `2 * FracBits` down to `FracBits` and narrowed once through the element
    /// policy, which resolves any accumulator overflow (assert / saturate /
    /// wrap) rather than leaving it implementation-defined. Use this when the
    /// mathematical result may legitimately exceed the range of `T`, as for a
    /// sum of squares.
    accumulator_t<T> wide_value() const {
        using Acc = accumulator_t<T>;
        using Policy = typename T::policy;
        return Acc::from_raw(
            Policy::template narrow<typename Acc::rep>(round_shift(sum_, T::frac_bits)));
    }

    T value() const { return from_accumulator<T>(wide_value()); }

  private:
    acc sum_{};
};

/// Exact sum of squares of fixed-point values, for norms. The squares are
/// accumulated unshifted (at `2 * FracBits` fractional bits) in the unsigned
/// form of the accumulator type, where an overflow is detected instead of
/// wrapping. An overflowing sum means a norm of at least 2^(bits(acc) / 2) raw
/// units, beyond the range of `T` for every supported width.
template <typename T> class sum_of_squares {
    using acc = widen_or_same_t<widen_t<typename T::rep>>;
    using U = make_unsigned_integer_t<acc>;

  public:
    void add(const T &a) {
        const auto raw = static_cast<acc>(a.raw());
        const U magnitude =
            raw < 0 ? static_cast<U>(U(0) - static_cast<U>(raw)) : static_cast<U>(raw);
        const auto next = static_cast<U>(sum_ + static_cast<U>(magnitude * magnitude));
        overflow_ = overflow_ || next < sum_;
        sum_ = next;
    }

    /// The square root of the sum, rounded once (the root of the raw sum at
    /// `2 * FracBits` is the raw root at `FracBits`) and narrowed through the
    /// policy of `T`. A sum that overflowed the accumulator is out of range:
    /// it is clamped to the largest value under every policy, and `checked`
    /// asserts.
    [[nodiscard]] T root() const {
        using Policy = typename T::policy;
        using Rep = typename T::rep;
        if (overflow_) {
            return T::from_raw(Policy::template from_out_of_range<Rep>(true));
        }
        return T::from_raw(Policy::template narrow<Rep>(static_cast<acc>(isqrt_rounded(sum_))));
    }

  private:
    U sum_{};
    bool overflow_ = false;
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

    /// `sum -= a * b`
    void subtract_product(const value_type &a, const value_type &b) {
        real_.subtract_product(a.real(), b.real());
        real_.add_product(a.imag(), b.imag());
        imag_.subtract_product(a.real(), b.imag());
        imag_.subtract_product(a.imag(), b.real());
    }

    /// `sum += conj(a) * b`
    void add_conj_product(const value_type &a, const value_type &b) {
        real_.add_product(a.real(), b.real());
        real_.add_product(a.imag(), b.imag());
        imag_.add_product(a.real(), b.imag());
        imag_.subtract_product(a.imag(), b.real());
    }

    /// `sum += conj(a) * conj(b)`, i.e. `conj(a * b)`
    void add_conj_conj_product(const value_type &a, const value_type &b) {
        real_.add_product(a.real(), b.real());
        real_.subtract_product(a.imag(), b.imag());
        imag_.subtract_product(a.real(), b.imag());
        imag_.subtract_product(a.imag(), b.real());
    }

    /// `sum -= conj(a) * b`
    void subtract_conj_product(const value_type &a, const value_type &b) {
        real_.subtract_product(a.real(), b.real());
        real_.subtract_product(a.imag(), b.imag());
        imag_.subtract_product(a.real(), b.imag());
        imag_.add_product(a.imag(), b.real());
    }

    /// `sum += a`
    void add(const value_type &a) {
        real_.add(a.real());
        imag_.add(a.imag());
    }

    value_type value() const { return value_type(real_.value(), imag_.value()); }

    accumulator_t<value_type> wide_value() const {
        return accumulator_t<value_type>(real_.wide_value(), imag_.wide_value());
    }

  private:
    real_mac<T> real_;
    real_mac<T> imag_;
};

/// The multiply-accumulate helper for element type `T`: `real_mac<T>` for real
/// element types and `complex_mac<U>` for `std::complex<U>`.
template <typename T> struct mac_type {
    using type = real_mac<T>;
};

template <typename T> struct mac_type<std::complex<T>> {
    using type = complex_mac<T>;
};

template <typename T> using mac_t = typename mac_type<T>::type;

/// `beta * y + alpha1 * w1 + alpha2 * w2` held in the accumulator type and
/// narrowed once; `y` is not read when `beta` is zero.
template <typename T>
inline T acc_scale_add2_and_narrow(const T &beta, const T &y, const T &alpha1,
                                   const accumulator_t<T> &w1, const T &alpha2,
                                   const accumulator_t<T> &w2) {
    using Acc = accumulator_t<T>;
    Acc scaled_y = value_constants<Acc>::zero();
    if (is_one(beta)) {
        scaled_y = to_accumulator(y);
    } else if (beta != value_constants<T>::zero()) {
        scaled_y = to_accumulator(y) * to_accumulator(beta);
    }
    return from_accumulator<T>(scaled_y + to_accumulator(alpha1) * w1 +
                               to_accumulator(alpha2) * w2);
}

// ---------------------------------------------------------------------------
// Elements of the rank-1 and rank-2 updates for fixed point
// ---------------------------------------------------------------------------
//
// Netlib forms temp = alpha * y(j) once per column and then adds x(i) * temp to
// each element. In fixed point that rounds temp, and the rounding error is then
// scaled by |x(i)|. These helpers instead form the products exactly in the
// multiply-accumulate, round the sum once into the accumulator type, scale it by
// alpha there and add the element, narrowing once: the error no longer depends
// on the magnitude of the operands. The floating-point paths keep the Netlib
// order, so their results stay bit-identical.

/// `a + alpha * x * y`, or `a + alpha * x * conj(y)` with `ConjY`.
template <bool ConjY, typename T>
inline T rank1_element(const T &a, const T &alpha, const T &x, const T &y) {
    mac_t<T> product;
    if constexpr (ConjY) {
        product.add_conj_product(y, x);
    } else {
        product.add_product(x, y);
    }
    return from_accumulator<T>(to_accumulator(a) + to_accumulator(alpha) * product.wide_value());
}

/// `a + alpha * (x1 * y1 + x2 * y2)`, the symmetric rank-2 update element.
template <typename T>
inline T rank2_element(const T &a, const T &alpha, const T &x1, const T &y1, const T &x2,
                       const T &y2) {
    mac_t<T> sum;
    sum.add_product(x1, y1);
    sum.add_product(x2, y2);
    return from_accumulator<T>(to_accumulator(a) + to_accumulator(alpha) * sum.wide_value());
}

/// `a + alpha * x1 * conj(y1) + conj(alpha) * x2 * conj(y2)`, the Hermitian
/// rank-2 update element. The second term is formed as
/// `conj(alpha * conj(x2) * y2)` in the accumulator type, so `conj(alpha)` is
/// never materialized in the element type, where it would overflow for an
/// imaginary part equal to the minimum.
template <typename T>
inline T hermitian_rank2_element(const T &a, const T &alpha, const T &x1, const T &y1, const T &x2,
                                 const T &y2) {
    mac_t<T> first;
    first.add_conj_product(y1, x1);
    mac_t<T> second;
    second.add_conj_product(x2, y2);
    const auto scaled_alpha = to_accumulator(alpha);
    return from_accumulator<T>(to_accumulator(a) + scaled_alpha * first.wide_value() +
                               conj_value(scaled_alpha * second.wide_value()));
}

} // namespace detail

} // namespace thefblas
