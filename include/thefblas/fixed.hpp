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
// unavailable, `widen<std::int64_t>` maps onto itself and the intermediates
// are computed at the original width.
template <typename IntType>
struct widen;
template <>
struct widen<std::int8_t> {
  using type = std::int16_t;
};
template <>
struct widen<std::int16_t> {
  using type = std::int32_t;
};
template <>
struct widen<std::int32_t> {
  using type = std::int64_t;
};
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
template <>
struct widen<std::int64_t> {
  using type = __int128;
};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
template <>
struct widen<std::int64_t> {
  using type = std::int64_t;
};
#endif

template <typename IntType>
using widen_t = typename widen<IntType>::type;

/// True when `widen_t<IntType>` is genuinely wider than `IntType`.
template <typename IntType>
constexpr bool has_wider_type_v = sizeof(widen_t<IntType>) > sizeof(IntType);

/// Divides by 2^shift with round-to-nearest, ties away from zero, avoiding the
/// implementation-defined behaviour of right-shifting a negative value.
template <typename Wide>
constexpr Wide round_shift(Wide value, int shift) noexcept {
  if (shift <= 0) {
    return value;
  }
  const Wide half = static_cast<Wide>(Wide(1) << (shift - 1));
  if (value >= 0) {
    return static_cast<Wide>((value + half) >> shift);
  }
  return static_cast<Wide>(-static_cast<Wide>((-value + half) >> shift));
}

}  // namespace detail

template <typename IntType, int FracBits, typename Policy = checked>
class fixed {
  static_assert(std::is_integral<IntType>::value && std::is_signed<IntType>::value,
                "fixed<IntType, FracBits, Policy> requires a signed integral IntType");
  static_assert(FracBits >= 0 && FracBits <= std::numeric_limits<IntType>::digits,
                "FracBits must not exceed the number of value bits of IntType");

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

  template <typename FloatType = double>
  constexpr FloatType to_float() const noexcept {
    return static_cast<FloatType>(value_) / static_cast<FloatType>(IntType(1) << FracBits);
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
    assert(b.value_ != 0 && "thefblas::fixed<>: division by zero");
    const wide numerator = static_cast<wide>(static_cast<wide>(a.value_) << FracBits);
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
    return static_cast<double>(IntType(1) << FracBits);
  }

  // NaN (the only value not equal to itself) maps to zero. Comparing `v != v`
  // avoids <cmath>, whose classification helpers are not constexpr.
  static constexpr IntType from_float(double v) noexcept {
    return v != v ? IntType(0) : from_scaled(v * scale() + (v >= 0.0 ? 0.5 : -0.5));
  }

  // `scaled` is already rounded; reject it when it is outside the range that
  // can be converted back to IntType without undefined behaviour. The +/- 1.0
  // slack keeps the comparison usable for every width, including 64-bit, where
  // the exact limits are not representable as a double.
  static constexpr IntType from_scaled(double scaled) noexcept {
    return scaled < static_cast<double>((std::numeric_limits<IntType>::min)()) - 1.0
               ? Policy::template from_out_of_range<IntType>(false)
               : (scaled > static_cast<double>((std::numeric_limits<IntType>::max)()) + 1.0
                      ? Policy::template from_out_of_range<IntType>(true)
                      : static_cast<IntType>(scaled));
  }

  template <typename Number>
  static constexpr IntType from_integer(Number v) noexcept {
    return Policy::template narrow<IntType>(
        static_cast<wide>(static_cast<wide>(v) << FracBits));
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
  // Initial guess: the value itself (or 1 if it's tiny), refined via Newton
  // iterations x_{k+1} = (x_k + f / x_k) / 2.
  F x = f > F(1) ? f : F(1);
  for (int i = 0; i < 32; ++i) {
    const F next = (x + f / x) / F(2);
    if (next == x) {
      break;
    }
    x = next;
  }
  return x;
}

namespace detail {

template <typename T>
struct is_fixed : std::false_type {};

template <typename IntType, int FracBits, typename Policy>
struct is_fixed<fixed<IntType, FracBits, Policy>> : std::true_type {};

template <typename T>
constexpr bool is_fixed_v = is_fixed<T>::value;

/// True when `widen_t<IntType>` is both genuinely wider and usable as the
/// storage type of a `fixed<>`. The second condition matters for `__int128`,
/// which only satisfies `std::is_integral` (and only has a
/// `std::numeric_limits` specialisation) when compiler extensions are enabled;
/// in strict mode the accumulator silently falls back to the original width.
template <typename IntType>
constexpr bool can_widen_v =
    has_wider_type_v<IntType> && std::is_integral<widen_t<IntType>>::value;

/// The same Q format in a wider storage type, used for internal accumulators.
/// `FracBits` is deliberately unchanged, so converting to it is a plain sign
/// extension with no shift and no rounding.
template <typename T>
struct wider_fixed {
  using type = T;
};

template <typename IntType, int FracBits, typename Policy>
struct wider_fixed<fixed<IntType, FracBits, Policy>> {
  using type = typename std::conditional<can_widen_v<IntType>,
                                         fixed<widen_t<IntType>, FracBits, Policy>,
                                         fixed<IntType, FracBits, Policy>>::type;
};

}  // namespace detail

}  // namespace thefblas
