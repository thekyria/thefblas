// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include <cassert>
#include <limits>
#include <type_traits>

namespace thefblas {

/**
 * @file overflow.hpp
 * @brief Compile-time overflow policies for thefblas::fixed.
 *
 * Every arithmetic operation on `fixed<IntType, FracBits, Policy>` is evaluated
 * in an intermediate type at least twice as wide as `IntType` and then narrowed
 * back to `IntType` through `Policy`. The policy therefore decides, once and for
 * the whole type, what happens when a result does not fit the chosen Q format:
 *
 * | Policy     | Behaviour on overflow                                          |
 * |------------|----------------------------------------------------------------|
 * | `wrap`     | modular (two's-complement) reduction; always well defined       |
 * | `saturate` | clamps to the representable range of `IntType`                  |
 * | `checked`  | `assert()` in debug builds, then behaves like `wrap`            |
 *
 * `checked` is the default, preserving the historical behaviour of the type.
 * `saturate` is the recommended choice for production/embedded use because it
 * is sign preserving and monotone: a value that is too large stays large rather
 * than flipping sign. Note however that saturating arithmetic is **neither
 * associative nor distributive** — `(a + b) + c` and `a + (b + c)` can differ —
 * so it must not be relied upon to make an ill-scaled computation correct. Use
 * a Q format with enough integer bits (and the wide accumulators that the BLAS
 * routines use internally) for that.
 *
 * Conversion from a floating-point value whose magnitude is outside the
 * representable range is a separate case: modular reduction of, say, an
 * infinity is meaningless, so **all** policies clamp to the nearest limit there
 * and `checked` additionally asserts. NaN converts to zero under every policy.
 *
 * Policies are empty tag types; selecting one costs nothing at run time beyond
 * the operation it performs.
 */

namespace detail {

/// Reduces `value` modulo 2^bits(IntType) and reinterprets the result as a
/// two's-complement signed value, without relying on implementation-defined
/// signed conversion or on signed overflow.
template <typename IntType, typename Wide>
constexpr IntType wrap_narrow(Wide value) noexcept {
  using Unsigned = typename std::make_unsigned<IntType>::type;
  const Unsigned bits = static_cast<Unsigned>(value);
  if (bits <= static_cast<Unsigned>((std::numeric_limits<IntType>::max)())) {
    return static_cast<IntType>(bits);
  }
  const Unsigned offset =
      static_cast<Unsigned>(bits - static_cast<Unsigned>((std::numeric_limits<IntType>::min)()));
  return static_cast<IntType>(static_cast<IntType>(offset) +
                              (std::numeric_limits<IntType>::min)());
}

template <typename IntType>
constexpr IntType clamp_limit(bool positive) noexcept {
  return positive ? (std::numeric_limits<IntType>::max)() : (std::numeric_limits<IntType>::min)();
}

}  // namespace detail

/// Modular overflow policy: results wrap around, as on raw DSP hardware.
struct wrap {
  template <typename IntType, typename Wide>
  static constexpr IntType narrow(Wide value) noexcept {
    return detail::wrap_narrow<IntType>(value);
  }

  /// Floating-point sources outside the representable range cannot be reduced
  /// modularly in a meaningful way and are clamped instead.
  template <typename IntType>
  static constexpr IntType from_out_of_range(bool positive) noexcept {
    return detail::clamp_limit<IntType>(positive);
  }
};

/// Saturating overflow policy: results clamp to the representable range.
struct saturate {
  template <typename IntType, typename Wide>
  static constexpr IntType narrow(Wide value) noexcept {
    return value > static_cast<Wide>((std::numeric_limits<IntType>::max)())
               ? (std::numeric_limits<IntType>::max)()
               : (value < static_cast<Wide>((std::numeric_limits<IntType>::min)())
                      ? (std::numeric_limits<IntType>::min)()
                      : static_cast<IntType>(value));
  }

  template <typename IntType>
  static constexpr IntType from_out_of_range(bool positive) noexcept {
    return detail::clamp_limit<IntType>(positive);
  }
};

/// Debug-checked overflow policy (the default): asserts on overflow, and
/// otherwise behaves like `wrap` so that release builds stay well defined.
struct checked {
  template <typename IntType, typename Wide>
  static constexpr IntType narrow(Wide value) noexcept {
    return (assert(value >= static_cast<Wide>((std::numeric_limits<IntType>::min)()) &&
                   value <= static_cast<Wide>((std::numeric_limits<IntType>::max)()) &&
                   "thefblas::fixed<>: arithmetic overflow"),
            detail::wrap_narrow<IntType>(value));
  }

  template <typename IntType>
  static constexpr IntType from_out_of_range(bool positive) noexcept {
    return (assert(false && "thefblas::fixed<>: value out of range for this Q format"),
            detail::clamp_limit<IntType>(positive));
  }
};

}  // namespace thefblas
