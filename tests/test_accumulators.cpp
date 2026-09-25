// NOLINTNEXTLINE(portability-avoid-pragma-once)
#include "thefblas/thefblas.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>

using thefblas::fixed;

namespace {

// Q1.15: range [-1, 1), one unit = 2^-15.
using Q15 = thefblas::q15;
// Q9.7 in 16 bits: range [-256, 256), one unit = 2^-7.
using Q7_8 = fixed<std::int16_t, 7>;

bool close(double a, double b, double eps) {
  return std::fabs(a - b) < eps;
}

// Example A: the dot product of [0.9, 0.9] with itself is 1.62, which is not
// representable in Q1.15 at all. Without a wide accumulator the running sum
// overflows the 16-bit format after the second product and, under the checked
// policy, would trip an assertion; with the accumulator the sum is formed
// exactly in 32 bits and only the final narrowing is subject to the policy.
void test_dot_wide_accumulator() {
  using F = fixed<std::int16_t, 15, thefblas::saturate>;
  const F x[2] = {F(0.9), F(0.9)};
  const F result = thefblas::dot(2, x, 1, x, 1);
  // 1.62 saturates to the largest representable value, not to a wrapped
  // negative number, and not to the doubly-wrapped value a naive loop gives.
  assert(result.raw() == 32767);
}

// The same reduction in a format that can hold the result must be exact to
// within one quantisation step, with no per-product rounding loss.
void test_dot_exactness() {
  const Q7_8 x[4] = {Q7_8(0.9), Q7_8(0.9), Q7_8(0.9), Q7_8(0.9)};
  const Q7_8 result = thefblas::dot(4, x, 1, x, 1);
  assert(close(result.to_float<double>(), 4.0 * 0.9 * 0.9, 0.02));
}

// Example C: a sum of squares can exceed the range of the element type even
// when the norm itself does not. nrm2 must therefore take the square root in
// the accumulator type. Here the squares sum to 1.62 but the norm is 1.27,
// which is still outside Q1.15, so check the in-range Q9.7 case instead.
void test_nrm2_wide_accumulator() {
  const Q7_8 x[2] = {Q7_8(0.9), Q7_8(0.9)};
  const Q7_8 result = thefblas::nrm2(2, x, 1);
  assert(close(result.to_float<double>(), std::sqrt(1.62), 0.05));

  // In Q1.15 the sum of squares (1.62) is unrepresentable but the norm
  // (1.2728) is too, so saturation at the very end is the correct answer.
  using F = fixed<std::int16_t, 15, thefblas::saturate>;
  const F y[2] = {F(0.9), F(0.9)};
  assert(thefblas::nrm2(2, y, 1).raw() == 32767);

  // Whereas a vector whose squares overflow but whose norm does not still
  // gives the right answer: 0.8^2 + 0.5^2 = 0.89, norm 0.9434.
  const F z[2] = {F(0.8), F(0.5)};
  assert(close(thefblas::nrm2(2, z, 1).to_float<double>(), std::sqrt(0.89), 0.01));
}

// asum accumulates absolute values, which overflow just as easily.
void test_asum_wide_accumulator() {
  const Q7_8 x[4] = {Q7_8(0.9), Q7_8(-0.9), Q7_8(0.9), Q7_8(-0.9)};
  assert(close(thefblas::asum(4, x, 1).to_float<double>(), 3.6, 0.02));
}

// Level 2 reductions use the same accumulator. A 4x4 symmetric product whose
// intermediate row sums leave the element range must still come back correct
// when the final result is in range.
void test_gemv_transpose_accumulator() {
  const int n = 4;
  Q7_8 a[16];
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[i + j * n] = Q7_8(0.9);
    }
  }
  const Q7_8 x[4] = {Q7_8(0.9), Q7_8(0.9), Q7_8(0.9), Q7_8(0.9)};
  Q7_8 y[4] = {Q7_8(0.0), Q7_8(0.0), Q7_8(0.0), Q7_8(0.0)};
  thefblas::gemv('T', n, n, Q7_8(1.0), a, n, x, 1, Q7_8(0.0), y, 1);
  for (int i = 0; i < n; ++i) {
    assert(close(y[i].to_float<double>(), 4.0 * 0.81, 0.05));
  }
}

// Floating-point results must be untouched by the accumulator machinery: the
// accumulator type for float is float, so the reductions stay bit-identical to
// the straightforward loop that Netlib BLAS specifies.
void test_float_bit_identical() {
  static_assert(std::is_same<thefblas::detail::accumulator_t<float>, float>::value,
                "float must accumulate in float");
  static_assert(std::is_same<thefblas::detail::accumulator_t<double>, double>::value,
                "double must accumulate in double");

  const float x[5] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
  const float y[5] = {1.5F, 2.5F, 3.5F, 4.5F, 5.5F};
  float expected = 0.0F;
  for (int i = 0; i < 5; ++i) {
    expected += x[i] * y[i];
  }
  assert(thefblas::dot(5, x, 1, y, 1) == expected);

  float sum = 0.0F;
  for (int i = 0; i < 5; ++i) {
    sum += x[i] * x[i];
  }
  assert(thefblas::nrm2(5, x, 1) == std::sqrt(sum));

  float abs_sum = 0.0F;
  for (int i = 0; i < 5; ++i) {
    abs_sum += std::fabs(x[i]);
  }
  assert(thefblas::asum(5, x, 1) == abs_sum);
}

// Complex reductions build on two real accumulators.
void test_complex_accumulator() {
  using C = std::complex<Q7_8>;
  const C x[2] = {C(Q7_8(0.9), Q7_8(0.9)), C(Q7_8(0.9), Q7_8(0.9))};
  const C d = thefblas::dotc(2, x, 1, x, 1);
  // conj(x) . x = 2 * (0.81 + 0.81) = 3.24, purely real.
  assert(close(d.real().to_float<double>(), 3.24, 0.05));
  assert(close(d.imag().to_float<double>(), 0.0, 0.02));
}

}  // namespace

int main() {
  test_dot_wide_accumulator();
  test_dot_exactness();
  test_nrm2_wide_accumulator();
  test_asum_wide_accumulator();
  test_gemv_transpose_accumulator();
  test_float_bit_identical();
  test_complex_accumulator();
  return 0;
}
