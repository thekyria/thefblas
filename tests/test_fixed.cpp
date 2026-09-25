// NOLINTNEXTLINE(portability-avoid-pragma-once)
#include "thefblas/fixed.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

using thefblas::fixed;

namespace {

template <typename IntType, int FracBits>
bool almost_equal(fixed<IntType, FracBits> a, fixed<IntType, FracBits> b, double eps = 1e-3) {
  return std::fabs(a.template to_float<double>() - b.template to_float<double>()) < eps;
}

void test_construction_and_conversion() {
  using F = fixed<std::int32_t, 24>;
  F a(1.5);
  assert(almost_equal(a, F(1.5)));
  F b(-2);
  assert(almost_equal(b, F(-2.0)));
  F zero;
  assert(zero.raw() == 0);
}

void test_arithmetic() {
  using F = fixed<std::int32_t, 20>;
  F a(3.25);
  F b(1.5);
  assert(almost_equal(a + b, F(4.75)));
  assert(almost_equal(a - b, F(1.75)));
  assert(almost_equal(a * b, F(4.875)));
  assert(almost_equal(a / b, F(3.25 / 1.5)));
  F c = a;
  c += b;
  assert(almost_equal(c, F(4.75)));
  c -= b;
  assert(almost_equal(c, a));
  c *= b;
  assert(almost_equal(c, F(3.25 * 1.5)));
}

void test_comparisons() {
  using F = fixed<std::int16_t, 12>;
  F a(1.0);
  F b(2.0);
  assert(a < b);
  assert(b > a);
  assert(a <= a);
  assert(a >= a);
  assert(a == F(1.0));
  assert(a != b);
}

void test_negation_and_abs() {
  using F = fixed<std::int32_t, 16>;
  F a(4.5);
  F neg = -a;
  assert(almost_equal(neg, F(-4.5)));
  assert(almost_equal(thefblas::abs(neg), a));
  assert(almost_equal(thefblas::abs(a), a));
}

void test_sqrt() {
  using F = fixed<std::int32_t, 20>;
  assert(almost_equal(thefblas::sqrt(F(16.0)), F(4.0)));
  assert(almost_equal(thefblas::sqrt(F(2.0)), F(1.41421356), 1e-2));
  assert(thefblas::sqrt(F(0.0)) == F(0.0));
}

void test_different_widths() {
  fixed<std::int8_t, 4> small(2.0);
  assert(almost_equal(small + small, fixed<std::int8_t, 4>(4.0)));

  fixed<std::int64_t, 32> big(123.456);
  assert(std::fabs(big.to_float<double>() - 123.456) < 1e-6);
}

// ---------------------------------------------------------------------------
// Overflow policy matrix
// ---------------------------------------------------------------------------

template <typename F>
constexpr F raw(typename F::rep v) {
  return F::from_raw(v);
}

void test_wrap_policy() {
  // Q1.7: raw range [-128, 127], one unit = 1/128.
  using F = fixed<std::int8_t, 7, thefblas::wrap>;
  constexpr std::int8_t lo = -128;
  constexpr std::int8_t hi = 127;

  // Addition past the top wraps around to the bottom.
  assert((raw<F>(hi) + raw<F>(1)).raw() == lo);
  assert((raw<F>(hi) + raw<F>(hi)).raw() == -2);
  // Subtraction past the bottom wraps around to the top.
  assert((raw<F>(lo) - raw<F>(1)).raw() == hi);
  // Negating the most negative value is its own negation.
  assert((-raw<F>(lo)).raw() == lo);
  assert(thefblas::abs(raw<F>(lo)).raw() == lo);
  // Division overflow wraps: 65/128 divided by 1/128 is 65, whose scaled raw
  // value 8320 reduces to -128 in eight bits.
  assert((raw<F>(65) / raw<F>(1)).raw() == lo);
  // Out-of-range construction is clamped under every policy.
  assert(F(10.0).raw() == hi);
  assert(F(-10.0).raw() == lo);
}

void test_saturate_policy() {
  using F = fixed<std::int8_t, 7, thefblas::saturate>;
  constexpr std::int8_t lo = -128;
  constexpr std::int8_t hi = 127;

  assert((raw<F>(hi) + raw<F>(1)).raw() == hi);
  assert((raw<F>(hi) + raw<F>(hi)).raw() == hi);
  assert((raw<F>(lo) - raw<F>(1)).raw() == lo);
  assert((-raw<F>(lo)).raw() == hi);
  assert(thefblas::abs(raw<F>(lo)).raw() == hi);
  assert((raw<F>(64) / raw<F>(1)).raw() == hi);
  assert((raw<F>(-64) / raw<F>(1)).raw() == lo);
  assert(F(10.0).raw() == hi);
  assert(F(-10.0).raw() == lo);

  // In-range results are unaffected by the policy.
  assert((raw<F>(40) + raw<F>(40)).raw() == 80);

  // Saturating arithmetic is not associative: clamping happens per operation.
  const F a = raw<F>(100);
  const F b = raw<F>(100);
  const F c = raw<F>(-100);
  assert(((a + b) + c).raw() == 27);   // clamped to 127 first
  assert((a + (b + c)).raw() == 100);  // no intermediate overflow
}

void test_checked_policy_in_range() {
  // The default policy asserts on overflow, so only in-range behaviour can be
  // exercised portably here; the defined release fallback is wrap.
  using F = fixed<std::int8_t, 7>;
  static_assert(std::is_same<typename F::policy, thefblas::checked>::value,
                "checked must be the default overflow policy");
  assert((raw<F>(60) + raw<F>(60)).raw() == 120);
  assert((raw<F>(-60) - raw<F>(60)).raw() == -120);
  assert(thefblas::abs(raw<F>(-120)).raw() == 120);
}

void test_nan_construction() {
  using F = fixed<std::int32_t, 16>;
  const double nan = std::nan("");
  assert(F(nan).raw() == 0);
}

void test_rounding() {
  // Round to nearest, ties away from zero, symmetric about zero.
  using F = fixed<std::int32_t, 2>;
  assert((F(0.75) * F(0.75)).raw() == 2);    // 0.5625 -> 0.5
  assert((F(-0.75) * F(0.75)).raw() == -2);  // symmetric
  assert((F(0.25) * F(0.25)).raw() == 0);
  assert((F(-0.25) * F(-0.25)).raw() == 0);
}

void test_policy_aliases() {
  static_assert(std::is_same<thefblas::q15, fixed<std::int16_t, 15>>::value, "q15");
  static_assert(
      std::is_same<thefblas::q15_sat, fixed<std::int16_t, 15, thefblas::saturate>>::value,
      "q15_sat");
}

}  // namespace

int main() {
  test_construction_and_conversion();
  test_arithmetic();
  test_comparisons();
  test_negation_and_abs();
  test_sqrt();
  test_different_widths();
  test_wrap_policy();
  test_saturate_policy();
  test_checked_policy_in_range();
  test_nan_construction();
  test_rounding();
  test_policy_aliases();
  return 0;
}
