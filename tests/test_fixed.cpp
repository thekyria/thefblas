// NOLINTNEXTLINE(portability-avoid-pragma-once)
#include "thefblas/fixed.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>

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

}  // namespace

int main() {
  test_construction_and_conversion();
  test_arithmetic();
  test_comparisons();
  test_negation_and_abs();
  test_sqrt();
  test_different_widths();
  return 0;
}
