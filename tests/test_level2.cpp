#include "thefblas/fixed.hpp"
#include "thefblas/level2.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>

namespace {

using fixed32 = thefblas::fixed<std::int32_t, 20>;
using cfloat = std::complex<float>;
using cfixed = std::complex<fixed32>;

template <typename T>
double to_double(T value) {
  return static_cast<double>(value);
}

template <typename IntType, int FracBits>
double to_double(thefblas::fixed<IntType, FracBits> value) {
  return value.template to_float<double>();
}

template <typename T>
bool almost_equal_scalar(T lhs, T rhs, double eps = 1e-5) {
  const double diff = to_double(lhs - rhs);
  return diff <= eps && diff >= -eps;
}

template <typename T>
bool almost_equal_scalar(std::complex<T> lhs, std::complex<T> rhs, double eps = 1e-5) {
  return almost_equal_scalar(lhs.real(), rhs.real(), eps) &&
         almost_equal_scalar(lhs.imag(), rhs.imag(), eps);
}

template <typename T>
void assert_complex_real(T expected, std::complex<T> actual, double eps = 1e-5) {
  assert(almost_equal_scalar(actual.real(), expected, eps));
  assert(almost_equal_scalar(actual.imag(), T(0), eps));
}

void test_gemv_float() {
  std::array<float, 6> a = {1.0F, 3.0F, 5.0F, 2.0F, 4.0F, 6.0F};
  std::array<float, 2> x = {1.0F, 1.0F};
  std::array<float, 3> y = {0.0F, 0.0F, 0.0F};
  thefblas::gemv('N', 3, 2, 1.0F, a.data(), 3, x.data(), 1, 0.0F, y.data(), 1);
  assert(almost_equal_scalar(y[0], 3.0F));
  assert(almost_equal_scalar(y[1], 7.0F));
  assert(almost_equal_scalar(y[2], 11.0F));
}

void test_gemv_fixed_transpose() {
  std::array<fixed32, 6> a = {fixed32(1), fixed32(3), fixed32(5), fixed32(2), fixed32(4), fixed32(6)};
  std::array<fixed32, 3> x = {fixed32(1), fixed32(1), fixed32(1)};
  std::array<fixed32, 2> y = {fixed32(0), fixed32(0)};
  thefblas::gemv('T', 3, 2, fixed32(1), a.data(), 3, x.data(), 1, fixed32(0), y.data(), 1);
  assert(almost_equal_scalar(y[0], fixed32(9)));
  assert(almost_equal_scalar(y[1], fixed32(12)));
}

void test_gemv_complex_conjugate_transpose() {
  std::array<cfloat, 4> a = {cfloat(1.0F, 1.0F), cfloat(0.0F, 0.0F), cfloat(2.0F, 0.0F),
                             cfloat(0.0F, 1.0F)};
  std::array<cfloat, 2> x = {cfloat(1.0F, 0.0F), cfloat(1.0F, 0.0F)};
  std::array<cfloat, 2> y = {cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F)};
  thefblas::gemv('C', 2, 2, cfloat(1.0F, 0.0F), a.data(), 2, x.data(), 1, cfloat(0.0F, 0.0F),
                 y.data(), 1);
  assert(almost_equal_scalar(y[0], cfloat(1.0F, -1.0F)));
  assert(almost_equal_scalar(y[1], cfloat(2.0F, -1.0F)));
}

void test_gemv_complex_fixed_and_invalid_char() {
  std::array<cfixed, 4> a = {cfixed(fixed32(1), fixed32(0)), cfixed(fixed32(0), fixed32(0)),
                             cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(1), fixed32(0))};
  std::array<cfixed, 2> x = {cfixed(fixed32(3), fixed32(1)), cfixed(fixed32(4), fixed32(-2))};
  std::array<cfixed, 2> y = {cfixed(fixed32(9), fixed32(9)), cfixed(fixed32(8), fixed32(8))};
  thefblas::gemv('X', 2, 2, cfixed(fixed32(1), fixed32(0)), a.data(), 2, x.data(), 1,
                 cfixed(fixed32(0), fixed32(0)), y.data(), 1);
  assert(almost_equal_scalar(y[0], cfixed(fixed32(9), fixed32(9))));
  assert(almost_equal_scalar(y[1], cfixed(fixed32(8), fixed32(8))));
  thefblas::gemv('N', 2, 2, cfixed(fixed32(1), fixed32(0)), a.data(), 2, x.data(), 1,
                 cfixed(fixed32(0), fixed32(0)), y.data(), 1);
  assert(almost_equal_scalar(y[0], x[0]));
  assert(almost_equal_scalar(y[1], x[1]));
}

void test_ger_float() {
  std::array<float, 2> x = {1.0F, 2.0F};
  std::array<float, 2> y = {3.0F, 4.0F};
  std::array<float, 4> a = {0.0F, 0.0F, 0.0F, 0.0F};
  thefblas::ger(2, 2, 1.0F, x.data(), 1, y.data(), 1, a.data(), 2);
  assert(almost_equal_scalar(a[0], 3.0F));
  assert(almost_equal_scalar(a[1], 6.0F));
  assert(almost_equal_scalar(a[2], 4.0F));
  assert(almost_equal_scalar(a[3], 8.0F));
}

void test_geru_and_gerc_complex() {
  std::array<cfloat, 2> x = {cfloat(1.0F, 1.0F), cfloat(2.0F, 0.0F)};
  std::array<cfloat, 2> y = {cfloat(1.0F, -1.0F), cfloat(0.0F, 1.0F)};
  std::array<cfloat, 4> au = {cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F),
                              cfloat(0.0F, 0.0F)};
  std::array<cfloat, 4> ac = au;
  thefblas::geru(2, 2, cfloat(1.0F, 0.0F), x.data(), 1, y.data(), 1, au.data(), 2);
  thefblas::gerc(2, 2, cfloat(1.0F, 0.0F), x.data(), 1, y.data(), 1, ac.data(), 2);
  assert(almost_equal_scalar(au[0], cfloat(2.0F, 0.0F)));
  assert(almost_equal_scalar(ac[0], cfloat(0.0F, 2.0F)));
}

void test_symv_upper_float() {
  std::array<float, 4> a = {2.0F, 0.0F, 1.0F, 3.0F};
  std::array<float, 2> x = {1.0F, 2.0F};
  std::array<float, 2> y = {0.0F, 0.0F};
  thefblas::symv('U', 2, 1.0F, a.data(), 2, x.data(), 1, 0.0F, y.data(), 1);
  assert(almost_equal_scalar(y[0], 4.0F));
  assert(almost_equal_scalar(y[1], 7.0F));
}

void test_symv_lower_fixed() {
  std::array<fixed32, 4> a = {fixed32(2), fixed32(1), fixed32(0), fixed32(3)};
  std::array<fixed32, 2> x = {fixed32(1), fixed32(2)};
  std::array<fixed32, 2> y = {fixed32(0), fixed32(0)};
  thefblas::symv('L', 2, fixed32(1), a.data(), 2, x.data(), 1, fixed32(0), y.data(), 1);
  assert(almost_equal_scalar(y[0], fixed32(4)));
  assert(almost_equal_scalar(y[1], fixed32(7)));
}

void test_hemv_upper_complex_float() {
  std::array<cfloat, 4> a = {cfloat(2.0F, 5.0F), cfloat(0.0F, 0.0F), cfloat(1.0F, 1.0F),
                             cfloat(3.0F, -9.0F)};
  std::array<cfloat, 2> x = {cfloat(1.0F, 0.0F), cfloat(1.0F, 0.0F)};
  std::array<cfloat, 2> y = {cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F)};
  thefblas::hemv('U', 2, cfloat(1.0F, 0.0F), a.data(), 2, x.data(), 1, cfloat(0.0F, 0.0F),
                 y.data(), 1);
  assert(almost_equal_scalar(y[0], cfloat(3.0F, 1.0F)));
  assert(almost_equal_scalar(y[1], cfloat(4.0F, -1.0F)));
}

void test_hemv_lower_complex_fixed() {
  std::array<cfixed, 4> a = {cfixed(fixed32(2), fixed32(7)), cfixed(fixed32(1), fixed32(-1)),
                             cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(3), fixed32(4))};
  std::array<cfixed, 2> x = {cfixed(fixed32(1), fixed32(0)), cfixed(fixed32(1), fixed32(0))};
  std::array<cfixed, 2> y = {cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(0), fixed32(0))};
  thefblas::hemv('L', 2, cfixed(fixed32(1), fixed32(0)), a.data(), 2, x.data(), 1,
                 cfixed(fixed32(0), fixed32(0)), y.data(), 1);
  assert(almost_equal_scalar(y[0], cfixed(fixed32(3), fixed32(1))));
  assert(almost_equal_scalar(y[1], cfixed(fixed32(4), fixed32(-1))));
}

void test_syr_float_and_fixed() {
  std::array<float, 4> a1 = {0.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 2> x1 = {1.0F, 2.0F};
  thefblas::syr('U', 2, 1.0F, x1.data(), 1, a1.data(), 2);
  assert(almost_equal_scalar(a1[0], 1.0F));
  assert(almost_equal_scalar(a1[2], 2.0F));
  assert(almost_equal_scalar(a1[3], 4.0F));

  std::array<fixed32, 4> a2 = {fixed32(0), fixed32(0), fixed32(0), fixed32(0)};
  std::array<fixed32, 2> x2 = {fixed32(1), fixed32(2)};
  thefblas::syr('L', 2, fixed32(1), x2.data(), 1, a2.data(), 2);
  assert(almost_equal_scalar(a2[0], fixed32(1)));
  assert(almost_equal_scalar(a2[1], fixed32(2)));
  assert(almost_equal_scalar(a2[3], fixed32(4)));
}

void test_her_complex_float_and_fixed() {
  std::array<cfloat, 4> a1 = {cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F),
                              cfloat(0.0F, 0.0F)};
  std::array<cfloat, 2> x1 = {cfloat(1.0F, 1.0F), cfloat(2.0F, 0.0F)};
  thefblas::her('U', 2, 1.0F, x1.data(), 1, a1.data(), 2);
  assert_complex_real(2.0F, a1[0]);
  assert(almost_equal_scalar(a1[2], cfloat(2.0F, 2.0F)));
  assert_complex_real(4.0F, a1[3]);

  std::array<cfixed, 4> a2 = {cfixed(fixed32(0), fixed32(1)), cfixed(fixed32(0), fixed32(0)),
                              cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(0), fixed32(2))};
  std::array<cfixed, 2> x2 = {cfixed(fixed32(1), fixed32(1)), cfixed(fixed32(2), fixed32(0))};
  thefblas::her('L', 2, fixed32(1), x2.data(), 1, a2.data(), 2);
  assert_complex_real(fixed32(2), a2[0]);
  assert(almost_equal_scalar(a2[1], cfixed(fixed32(2), fixed32(-2))));
  assert_complex_real(fixed32(4), a2[3]);
}

void test_syr2_float_and_fixed() {
  std::array<float, 4> a1 = {0.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 2> x1 = {1.0F, 0.0F};
  std::array<float, 2> y1 = {0.0F, 1.0F};
  thefblas::syr2('U', 2, 1.0F, x1.data(), 1, y1.data(), 1, a1.data(), 2);
  assert(almost_equal_scalar(a1[0], 0.0F));
  assert(almost_equal_scalar(a1[2], 1.0F));
  assert(almost_equal_scalar(a1[3], 0.0F));

  std::array<fixed32, 4> a2 = {fixed32(0), fixed32(0), fixed32(0), fixed32(0)};
  std::array<fixed32, 2> x2 = {fixed32(1), fixed32(0)};
  std::array<fixed32, 2> y2 = {fixed32(0), fixed32(1)};
  thefblas::syr2('L', 2, fixed32(1), x2.data(), 1, y2.data(), 1, a2.data(), 2);
  assert(almost_equal_scalar(a2[0], fixed32(0)));
  assert(almost_equal_scalar(a2[1], fixed32(1)));
  assert(almost_equal_scalar(a2[3], fixed32(0)));
}

void test_her2_complex_float_and_fixed() {
  std::array<cfloat, 4> a1 = {cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F), cfloat(0.0F, 0.0F),
                              cfloat(0.0F, 0.0F)};
  std::array<cfloat, 2> x1 = {cfloat(1.0F, 0.0F), cfloat(0.0F, 0.0F)};
  std::array<cfloat, 2> y1 = {cfloat(0.0F, 0.0F), cfloat(1.0F, 0.0F)};
  thefblas::her2('U', 2, cfloat(1.0F, 0.0F), x1.data(), 1, y1.data(), 1, a1.data(), 2);
  assert(almost_equal_scalar(a1[2], cfloat(1.0F, 0.0F)));
  assert_complex_real(0.0F, a1[0]);
  assert_complex_real(0.0F, a1[3]);

  std::array<cfixed, 4> a2 = {cfixed(fixed32(0), fixed32(1)), cfixed(fixed32(0), fixed32(0)),
                              cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(0), fixed32(2))};
  std::array<cfixed, 2> x2 = {cfixed(fixed32(1), fixed32(0)), cfixed(fixed32(0), fixed32(0))};
  std::array<cfixed, 2> y2 = {cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(1), fixed32(0))};
  thefblas::her2('L', 2, cfixed(fixed32(1), fixed32(0)), x2.data(), 1, y2.data(), 1,
                 a2.data(), 2);
  assert(almost_equal_scalar(a2[1], cfixed(fixed32(1), fixed32(0))));
  assert_complex_real(fixed32(0), a2[0]);
  assert_complex_real(fixed32(0), a2[3]);
}

void test_trmv_float() {
  std::array<float, 4> a = {2.0F, 0.0F, 3.0F, 4.0F};
  std::array<float, 2> x = {1.0F, 2.0F};
  thefblas::trmv('U', 'N', 'N', 2, a.data(), 2, x.data(), 1);
  assert(almost_equal_scalar(x[0], 8.0F));
  assert(almost_equal_scalar(x[1], 8.0F));
}

void test_trmv_fixed_unit_transpose() {
  std::array<fixed32, 4> a = {fixed32(1), fixed32(5), fixed32(0), fixed32(1)};
  std::array<fixed32, 2> x = {fixed32(2), fixed32(3)};
  thefblas::trmv('L', 'T', 'U', 2, a.data(), 2, x.data(), 1);
  assert(almost_equal_scalar(x[0], fixed32(17)));
  assert(almost_equal_scalar(x[1], fixed32(3)));
}

void test_trmv_complex_conjugate_transpose() {
  std::array<cfloat, 4> a = {cfloat(1.0F, 0.0F), cfloat(0.0F, 0.0F), cfloat(2.0F, 1.0F),
                             cfloat(3.0F, 0.0F)};
  std::array<cfloat, 2> x = {cfloat(1.0F, 0.0F), cfloat(1.0F, 0.0F)};
  thefblas::trmv('U', 'C', 'N', 2, a.data(), 2, x.data(), 1);
  assert(almost_equal_scalar(x[0], cfloat(1.0F, 0.0F)));
  assert(almost_equal_scalar(x[1], cfloat(5.0F, -1.0F)));
}

void test_trsv_float() {
  std::array<float, 4> a = {2.0F, 0.0F, 3.0F, 4.0F};
  std::array<float, 2> x = {8.0F, 8.0F};
  thefblas::trsv('U', 'N', 'N', 2, a.data(), 2, x.data(), 1);
  assert(almost_equal_scalar(x[0], 1.0F));
  assert(almost_equal_scalar(x[1], 2.0F));
}

void test_trsv_fixed() {
  std::array<fixed32, 4> a = {fixed32(2), fixed32(3), fixed32(0), fixed32(4)};
  std::array<fixed32, 2> x = {fixed32(2), fixed32(11)};
  thefblas::trsv('L', 'N', 'N', 2, a.data(), 2, x.data(), 1);
  assert(almost_equal_scalar(x[0], fixed32(1)));
  assert(almost_equal_scalar(x[1], fixed32(2)));
}

void test_trsv_complex_conjugate_transpose() {
  std::array<cfloat, 4> a = {cfloat(2.0F, 0.0F), cfloat(0.0F, 0.0F), cfloat(1.0F, 1.0F),
                             cfloat(3.0F, 0.0F)};
  std::array<cfloat, 2> x = {cfloat(2.0F, 0.0F), cfloat(4.0F, -1.0F)};
  thefblas::trsv('U', 'C', 'N', 2, a.data(), 2, x.data(), 1);
  assert(almost_equal_scalar(x[0], cfloat(1.0F, 0.0F)));
  assert(almost_equal_scalar(x[1], cfloat(1.0F, 0.0F)));
}

}  // namespace

int main() {
  test_gemv_float();
  test_gemv_fixed_transpose();
  test_gemv_complex_conjugate_transpose();
  test_gemv_complex_fixed_and_invalid_char();
  test_ger_float();
  test_geru_and_gerc_complex();
  test_symv_upper_float();
  test_symv_lower_fixed();
  test_hemv_upper_complex_float();
  test_hemv_lower_complex_fixed();
  test_syr_float_and_fixed();
  test_her_complex_float_and_fixed();
  test_syr2_float_and_fixed();
  test_her2_complex_float_and_fixed();
  test_trmv_float();
  test_trmv_fixed_unit_transpose();
  test_trmv_complex_conjugate_transpose();
  test_trsv_float();
  test_trsv_fixed();
  test_trsv_complex_conjugate_transpose();
}
