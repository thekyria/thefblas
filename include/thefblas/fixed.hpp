// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/overflow.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

namespace thefblas {

/**
 * @file fixed.hpp
 * @brief Generic templated Q-format fixed-point number type.
 *
 * `fixed<IntType, FracBits, Policy>` stores a signed fixed-point value in the
 * underlying integer type `IntType`, using `FracBits` bits for the fractional
 * part. The remaining bits (sign + integer part) follow from the width of
 * `IntType`. For example `fixed<std::int16_t, 15>` is Q1.15, and
 * `fixed<std::int32_t, 31>` is Q1.31.
 *
 * Overflow policy: every operation — `+`, `-`, unary `-`, `*`, `/`, the
 * compound assignments, `abs` and construction from a floating-point value — is
 * evaluated in an intermediate type at least twice as wide as `IntType` and
 * narrowed back through `Policy` (see overflow.hpp). `Policy` defaults to
 * `checked`, which asserts in debug builds and wraps in release builds; use
 * `saturate` (or the `*_sat` aliases below) for sign-preserving clamping, or
 * `wrap` for plain modular arithmetic. There is no undefined behaviour under
 * any policy.
 *
 * Rounding policy: multiplication and division round to nearest, ties away from
 * zero, as does conversion from a floating-point value.
 *
 * No external dependencies are used; only the STL (`<cstdint>`, `<cassert>`,
 * `<limits>`) is required, keeping the library portable to bare-metal targets
 * such as ARM Cortex-M without an FPU.
 */

namespace detail {

// Selects an integer type at least twice as wide as IntType, used for
// intermediate multiply/divide results so overflow can be detected before
// narrowing back down to IntType. Falls back to a compiler __int128
// extension (widely available on GCC/Clang, including bare-metal ARM
// toolchains) when IntType is already 64-bit wide; if that extension is
// unavailable, `widen<std::int64_t>` maps onto itself and `fixed<std::int64_t,
// ...>` is rejected at compile time (see the static_assert in `fixed`), because
// evaluating its arithmetic at the original width would risk undefined signed
// overflow before the policy could resolve it.
template <typename IntType> struct widen;
template <> struct widen<std::int8_t> {
    using type = std::int16_t;
};
template <> struct widen<std::int16_t> {
    using type = std::int32_t;
};
template <> struct widen<std::int32_t> {
    using type = std::int64_t;
};
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
template <> struct widen<std::int64_t> {
    using type = __int128;
};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
template <> struct widen<std::int64_t> {
    using type = std::int64_t;
};
#endif

template <typename IntType> using widen_t = typename widen<IntType>::type;

/// `widen_t<IntType>` when `widen` is specialised for `IntType`, otherwise
/// `IntType` itself. Used to give reduction sums one more widening step than
/// the products they accumulate.
template <typename IntType, typename = void> struct widen_or_same {
    using type = IntType;
};
template <typename IntType>
struct widen_or_same<IntType, std::void_t<typename widen<IntType>::type>> {
    using type = typename widen<IntType>::type;
};
template <typename IntType> using widen_or_same_t = typename widen_or_same<IntType>::type;

/// True when `widen_t<IntType>` exists and is genuinely wider than `IntType`.
template <typename IntType>
constexpr bool has_wider_type_v = sizeof(widen_or_same_t<IntType>) > sizeof(IntType);

/// Divides by 2^shift with round-to-nearest, ties away from zero, avoiding the
/// implementation-defined behaviour of right-shifting a negative value.
template <typename Wide> constexpr Wide round_shift(Wide value, int shift) noexcept {
    if (shift <= 0) {
        return value;
    }
    const Wide half = static_cast<Wide>(Wide(1) << (shift - 1));
    if (value >= 0) {
        return static_cast<Wide>((value + half) >> shift);
    }
    return static_cast<Wide>(-static_cast<Wide>((-value + half) >> shift));
}

/// Scales by 2^shift without left-shifting a signed negative value.
template <typename Wide> constexpr Wide scale_pow2(Wide value, int shift) noexcept {
    static_assert(is_signed_integer<Wide>::value, "scale_pow2 requires a signed integral type");
    if (shift <= 0) {
        return value;
    }
    using Unsigned = make_unsigned_integer_t<Wide>;
    const Unsigned bits = static_cast<Unsigned>(value);
    const Unsigned magnitude = value < 0 ? static_cast<Unsigned>(Unsigned(0) - bits) : bits;
    const Unsigned scaled = static_cast<Unsigned>(magnitude << shift);
    return value < 0 ? static_cast<Wide>(Unsigned(0) - scaled) : static_cast<Wide>(scaled);
}

} // namespace detail

template <typename IntType, int FracBits, typename Policy = checked> class fixed {
    static_assert(std::is_integral<IntType>::value && std::is_signed<IntType>::value,
                  "fixed<IntType, FracBits, Policy> requires a signed integral IntType");
    static_assert(FracBits >= 0 && FracBits <= std::numeric_limits<IntType>::digits,
                  "FracBits must not exceed the number of value bits of IntType");
    static_assert(detail::has_wider_type_v<IntType>,
                  "fixed<IntType, FracBits, Policy> requires an intermediate type wider than "
                  "IntType; this compiler provides no 128-bit integer, so 64-bit IntType is "
                  "not supported");

    using wide = detail::widen_t<IntType>;

  public:
    using rep = IntType;
    using policy = Policy;
    static constexpr int frac_bits = FracBits;

    constexpr fixed() noexcept : value_(0) {}

    /// Construct from a raw underlying representation (no scaling).
    static constexpr fixed from_raw(IntType raw) noexcept {
        fixed f;
        f.value_ = raw;
        return f;
    }

    /// Construct by scaling a floating-point value into the fixed-point grid
    /// (round to nearest, ties away from zero), or from an integral value.
    ///
    /// NaN converts to zero. Values whose magnitude exceeds the representable
    /// range are clamped to the nearest limit under every policy, and `checked`
    /// additionally asserts; see overflow.hpp for why modular reduction is not
    /// offered here.
    template <typename Number,
              typename = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
    explicit constexpr fixed(Number v) noexcept
        : value_(std::is_floating_point<Number>::value ? from_float(static_cast<double>(v))
                                                       : from_integer(v)) {}

    constexpr IntType raw() const noexcept { return value_; }

    template <typename FloatType = double> constexpr FloatType to_float() const noexcept {
        using unsigned_rep = typename std::make_unsigned<IntType>::type;
        const auto scale = static_cast<unsigned_rep>(unsigned_rep(1) << FracBits);
        return static_cast<FloatType>(value_) / static_cast<FloatType>(scale);
    }

    constexpr fixed operator-() const noexcept {
        return from_raw(
            Policy::template narrow<IntType>(static_cast<wide>(-static_cast<wide>(value_))));
    }

    friend constexpr fixed operator+(fixed a, fixed b) noexcept {
        return from_raw(Policy::template narrow<IntType>(
            static_cast<wide>(static_cast<wide>(a.value_) + static_cast<wide>(b.value_))));
    }
    friend constexpr fixed operator-(fixed a, fixed b) noexcept {
        return from_raw(Policy::template narrow<IntType>(
            static_cast<wide>(static_cast<wide>(a.value_) - static_cast<wide>(b.value_))));
    }

    friend constexpr fixed operator*(fixed a, fixed b) noexcept {
        const wide product =
            static_cast<wide>(static_cast<wide>(a.value_) * static_cast<wide>(b.value_));
        return from_raw(Policy::template narrow<IntType>(detail::round_shift(product, FracBits)));
    }

    friend constexpr fixed operator/(fixed a, fixed b) noexcept {
        if (b.value_ == 0) {
            if constexpr (std::is_same<Policy, checked>::value) {
                assert(false && "thefblas::fixed<>: division by zero");
            }
            return from_raw(Policy::template from_out_of_range<IntType>(a.value_ >= 0));
        }
        const wide numerator = detail::scale_pow2(static_cast<wide>(a.value_), FracBits);
        const wide divisor = static_cast<wide>(b.value_);
        // Round to nearest, ties away from zero: bias the numerator by half the
        // divisor in the direction of the quotient's sign, then truncate.
        const wide half = static_cast<wide>(divisor / 2);
        const wide biased = ((numerator >= 0) == (divisor >= 0))
                                ? static_cast<wide>(numerator + half)
                                : static_cast<wide>(numerator - half);
        return from_raw(Policy::template narrow<IntType>(static_cast<wide>(biased / divisor)));
    }

    fixed &operator+=(fixed o) noexcept { return *this = *this + o; }
    fixed &operator-=(fixed o) noexcept { return *this = *this - o; }
    fixed &operator*=(fixed o) noexcept { return *this = *this * o; }
    fixed &operator/=(fixed o) noexcept { return *this = *this / o; }

    friend constexpr bool operator==(fixed a, fixed b) noexcept { return a.value_ == b.value_; }
    friend constexpr bool operator!=(fixed a, fixed b) noexcept { return a.value_ != b.value_; }
    friend constexpr bool operator<(fixed a, fixed b) noexcept { return a.value_ < b.value_; }
    friend constexpr bool operator>(fixed a, fixed b) noexcept { return a.value_ > b.value_; }
    friend constexpr bool operator<=(fixed a, fixed b) noexcept { return a.value_ <= b.value_; }
    friend constexpr bool operator>=(fixed a, fixed b) noexcept { return a.value_ >= b.value_; }

    friend std::ostream &operator<<(std::ostream &os, fixed f) {
        return os << f.template to_float<double>();
    }

  private:
    static constexpr double scale() noexcept {
        using unsigned_rep = typename std::make_unsigned<IntType>::type;
        return static_cast<double>(static_cast<unsigned_rep>(unsigned_rep(1) << FracBits));
    }

    // NaN (the only value not equal to itself) maps to zero. Comparing `v != v`
    // avoids <cmath>, whose classification helpers are not constexpr.
    static constexpr IntType from_float(double v) noexcept {
        return v != v ? IntType(0) : from_scaled(v * scale() + (v >= 0.0 ? 0.5 : -0.5));
    }

    // `scaled` is already rounded; clamp before converting so no out-of-range
    // floating-point-to-integer cast is attempted, including at 64-bit limits.
    static constexpr IntType from_scaled(double scaled) noexcept {
        using unsigned_rep = typename std::make_unsigned<IntType>::type;
        const double abs_limit = static_cast<double>(
            static_cast<unsigned_rep>(unsigned_rep(1) << std::numeric_limits<IntType>::digits));
        return scaled < -abs_limit
                   ? Policy::template from_out_of_range<IntType>(false)
                   : (scaled >= abs_limit ? Policy::template from_out_of_range<IntType>(true)
                                          : static_cast<IntType>(scaled));
    }

    // Scaling `v` by 2^FracBits could overflow `wide` before the policy sees
    // it (e.g. `fixed<int8_t, 7>(1000)` would compute `1000 << 7` in int16), so
    // out-of-range sources are resolved by the policy before any scaling.
    template <typename Number> static constexpr IntType from_integer(Number v) noexcept {
        // Largest whole number representable in this Q format; the smallest is
        // `-(max_int + 1)`, since the raw range is asymmetric two's complement.
        constexpr IntType max_int =
            static_cast<IntType>(static_cast<typename std::make_unsigned<IntType>::type>(
                                     (std::numeric_limits<IntType>::max)()) >>
                                 FracBits);
        if constexpr (std::is_signed<Number>::value) {
            if (v < Number(0)) {
                constexpr std::intmax_t min_int = -static_cast<std::intmax_t>(max_int) - 1;
                return static_cast<std::intmax_t>(v) < min_int
                           ? Policy::template from_out_of_range<IntType>(false)
                           : Policy::template narrow<IntType>(
                                 detail::scale_pow2(static_cast<wide>(v), FracBits));
            }
        }
        return static_cast<std::uintmax_t>(v) > static_cast<std::uintmax_t>(max_int)
                   ? Policy::template from_out_of_range<IntType>(true)
                   : Policy::template narrow<IntType>(
                         detail::scale_pow2(static_cast<wide>(v), FracBits));
    }

    IntType value_;
};

/// Convenience aliases for the common Q formats, with the default `checked`
/// policy and with the recommended `saturate` policy.
using q7 = fixed<std::int8_t, 7>;
using q15 = fixed<std::int16_t, 15>;
using q31 = fixed<std::int32_t, 31>;
using q16_16 = fixed<std::int32_t, 16>;
using q7_sat = fixed<std::int8_t, 7, saturate>;
using q15_sat = fixed<std::int16_t, 15, saturate>;
using q31_sat = fixed<std::int32_t, 31, saturate>;
using q16_16_sat = fixed<std::int32_t, 16, saturate>;

/// Absolute value. Note that `abs(min)` is not representable and is therefore
/// resolved by the overflow policy (clamped under `saturate`, wrapped back to
/// `min` under `wrap`, asserted under `checked`).
template <typename IntType, int FracBits, typename Policy>
constexpr fixed<IntType, FracBits, Policy> abs(fixed<IntType, FracBits, Policy> f) noexcept {
    return f < fixed<IntType, FracBits, Policy>::from_raw(0) ? -f : f;
}

/// Integer-only Newton's method square root on the fixed-point grid.
template <typename IntType, int FracBits, typename Policy>
fixed<IntType, FracBits, Policy> sqrt(fixed<IntType, FracBits, Policy> f) {
    using F = fixed<IntType, FracBits, Policy>;
    assert(f >= F::from_raw(0) && "thefblas::fixed<>: sqrt of negative value");
    if (f <= F::from_raw(0)) {
        return F::from_raw(0);
    }
    // Initial guess: the value itself (or one raw unit if it's tiny), refined via Newton
    // iterations x_{k+1} = (x_k + f / x_k) / 2. The halving is done on the (non-negative)
    // raw values with round-half-up, so it neither depends on 0.5 being representable
    // (FracBits == 0) nor overflows when x_k + f / x_k exceeds the representable range.
    F x = f > F::from_raw(1) ? f : F::from_raw(1);
    for (int i = 0; i < 32; ++i) {
        const IntType a = x.raw();
        const IntType b = (f / x).raw();
        const F next = F::from_raw(static_cast<IntType>(a / 2 + b / 2 + ((a % 2) + (b % 2) + 1) / 2));
        if (next == x) {
            break;
        }
        x = next;
    }
    return x;
}

namespace detail {

template <typename T> struct is_fixed : std::false_type {};

template <typename IntType, int FracBits, typename Policy>
struct is_fixed<fixed<IntType, FracBits, Policy>> : std::true_type {};

template <typename T> constexpr bool is_fixed_v = is_fixed<T>::value;

/// True when `widen_t<IntType>` is genuinely wider than `IntType` and is itself
/// a valid storage type of a `fixed<>` (with a further wider type for its own
/// arithmetic). This excludes `__int128`, which has no wider type, and
/// `std::int64_t` when the compiler provides no `__int128`. In those cases the
/// accumulator falls back to the original width.
template <typename IntType, bool = has_wider_type_v<IntType>> struct can_widen : std::false_type {};

template <typename IntType>
struct can_widen<IntType, true> : std::integral_constant<bool, has_wider_type_v<widen_t<IntType>>> {
};

template <typename IntType> constexpr bool can_widen_v = can_widen<IntType>::value;

/// The same Q format in a wider storage type, used for internal accumulators.
/// `FracBits` is deliberately unchanged, so converting to it is a plain sign
/// extension with no shift and no rounding. The widened type is only named in
/// the selected specialisation, so an unsupported `fixed<>` is never formed.
template <typename T, bool Widen> struct wider_fixed_select {
    using type = T;
};

template <typename IntType, int FracBits, typename Policy>
struct wider_fixed_select<fixed<IntType, FracBits, Policy>, true> {
    using type = fixed<widen_t<IntType>, FracBits, Policy>;
};

template <typename T> struct wider_fixed {
    using type = T;
};

template <typename IntType, int FracBits, typename Policy>
struct wider_fixed<fixed<IntType, FracBits, Policy>>
    : wider_fixed_select<fixed<IntType, FracBits, Policy>, can_widen_v<IntType>> {};

} // namespace detail

} // namespace thefblas
