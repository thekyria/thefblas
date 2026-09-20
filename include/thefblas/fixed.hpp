// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

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
 * `fixed<IntType, FracBits>` stores a signed fixed-point value in the underlying
 * integer type `IntType`, using `FracBits` bits for the fractional part. The
 * remaining bits (sign + integer part) follow from the width of `IntType`. For
 * example `fixed<std::int16_t, 15>` is Q1.15, and `fixed<std::int32_t, 31>` is
 * Q1.31.
 *
 * Overflow policy: arithmetic that overflows the range representable by
 * `IntType` is undefined behaviour, guarded by `assert()` in debug builds
 * (`NDEBUG` disables the checks, matching `<cassert>` semantics). Callers are
 * responsible for choosing a representation wide enough to avoid overflow, or
 * for saturating/scaling their own data before calling into thefblas.
 *
 * Rounding policy: multiplication and division round to nearest (ties away
 * from zero); shifts used internally are on quantities already rounded.
 *
 * No external dependencies are used; only the STL (`<cstdint>`, `<cassert>`,
 * `<limits>`) is required, keeping the library portable to bare-metal targets
 * such as ARM Cortex-M without an FPU.
 */

namespace detail {

// Selects an integer type at least twice as wide as IntType, used for
// intermediate multiply/divide results so overflow can be detected before
// truncating back down to IntType. Falls back to a compiler __int128
// extension (widely available on GCC/Clang, including bare-metal ARM
// toolchains) when IntType is already 64-bit wide.
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
template <>
struct widen<std::int64_t> {
  using type = __int128;
};
#endif

}  // namespace detail

template <typename IntType, int FracBits>
class fixed {
  static_assert(std::is_integral<IntType>::value && std::is_signed<IntType>::value,
                "fixed<IntType, FracBits> requires a signed integral IntType");
  static_assert(FracBits >= 0 && FracBits < std::numeric_limits<IntType>::digits,
                "FracBits must be within the number of value bits of IntType");

 public:
  using rep = IntType;
  static constexpr int frac_bits = FracBits;

  constexpr fixed() noexcept : value_(0) {}

  /// Construct from a raw underlying representation (no scaling).
  static constexpr fixed from_raw(IntType raw) noexcept {
    fixed f;
    f.value_ = raw;
    return f;
  }

  /// Construct by scaling a floating-point value into the fixed-point grid,
  /// or from an integral value (integer part only, no fractional bits set).
  template <typename Number,
            typename = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
  explicit constexpr fixed(Number v) noexcept
      : value_(std::is_floating_point<Number>::value
                    ? static_cast<IntType>(
                          static_cast<double>(v) * static_cast<double>(IntType(1) << FracBits) +
                          (static_cast<double>(v) >= 0.0 ? 0.5 : -0.5))
                    : static_cast<IntType>(static_cast<IntType>(v) << FracBits)) {}

  constexpr IntType raw() const noexcept { return value_; }

  template <typename FloatType = double>
  constexpr FloatType to_float() const noexcept {
    return static_cast<FloatType>(value_) / static_cast<FloatType>(IntType(1) << FracBits);
  }

  constexpr fixed operator-() const noexcept { return from_raw(static_cast<IntType>(-value_)); }

  friend constexpr fixed operator+(fixed a, fixed b) noexcept {
    return from_raw(static_cast<IntType>(a.value_ + b.value_));
  }
  friend constexpr fixed operator-(fixed a, fixed b) noexcept {
    return from_raw(static_cast<IntType>(a.value_ - b.value_));
  }

  friend constexpr fixed operator*(fixed a, fixed b) noexcept {
    using wide = typename detail::widen<IntType>::type;
    wide product = static_cast<wide>(a.value_) * static_cast<wide>(b.value_);
    // Round to nearest, ties away from zero, then shift back down to FracBits.
    wide half = wide(1) << (FracBits - 1);
    wide rounded = product >= 0 ? (product + half) : (product - half);
    wide shifted = rounded >> FracBits;
    assert(shifted >= static_cast<wide>(std::numeric_limits<IntType>::min()) &&
           shifted <= static_cast<wide>(std::numeric_limits<IntType>::max()) &&
           "fixed<>: multiplication overflow");
    return from_raw(static_cast<IntType>(shifted));
  }

  friend constexpr fixed operator/(fixed a, fixed b) noexcept {
    assert(b.value_ != 0 && "fixed<>: division by zero");
    using wide = typename detail::widen<IntType>::type;
    wide numerator = static_cast<wide>(a.value_) << FracBits;
    wide half = static_cast<wide>(b.value_) / 2;
    wide rounded = (numerator >= 0) == (b.value_ >= 0) ? numerator + half : numerator - half;
    wide quotient = rounded / static_cast<wide>(b.value_);
    assert(quotient >= static_cast<wide>(std::numeric_limits<IntType>::min()) &&
           quotient <= static_cast<wide>(std::numeric_limits<IntType>::max()) &&
           "fixed<>: division overflow");
    return from_raw(static_cast<IntType>(quotient));
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
  IntType value_;
};

// Select a wide integer type at least twice the width of IntType for
// intermediate multiply/divide results, without any external dependency.

/// Absolute value, saturated at the representable magnitude for the minimum value.
template <typename IntType, int FracBits>
constexpr fixed<IntType, FracBits> abs(fixed<IntType, FracBits> f) noexcept {
  return f < fixed<IntType, FracBits>::from_raw(0) ? -f : f;
}

/// Integer-only Newton's method square root on the fixed-point grid.
template <typename IntType, int FracBits>
fixed<IntType, FracBits> sqrt(fixed<IntType, FracBits> f) {
  using F = fixed<IntType, FracBits>;
  assert(f >= F::from_raw(0) && "fixed<>: sqrt of negative value");
  if (f == F::from_raw(0)) {
    return f;
  }
  // Initial guess: the value itself (or 1 if it's tiny), refined via Newton
  // iterations x_{k+1} = (x_k + f / x_k) / 2.
  F x = f > F(1) ? f : F(1);
  for (int i = 0; i < 32; ++i) {
    F next = (x + f / x) / F(2);
    if (next == x) {
      break;
    }
    x = next;
  }
  return x;
}

}  // namespace thefblas
