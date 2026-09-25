#include "thefblas/fixed.hpp"
#include "thefblas/level2.hpp"
#include "thefblas/level2_banded.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <vector>

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

// Expands general band storage into a dense column-major m-by-n matrix.
template <typename T>
std::vector<T> expand_general_band(int m, int n, int kl, int ku, const std::vector<T>& ab,
                                   int ldab) {
  std::vector<T> dense(static_cast<std::size_t>(m) * n, T{});
  for (int j = 0; j < n; ++j) {
    const int first = (j - ku > 0) ? (j - ku) : 0;
    const int last = (j + kl < m - 1) ? (j + kl) : (m - 1);
    for (int i = first; i <= last; ++i) {
      dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m] =
          ab[static_cast<std::size_t>(ku + i - j) + static_cast<std::size_t>(j) * ldab];
    }
  }
  return dense;
}

// Expands symmetric/Hermitian/triangular band storage into a dense n-by-n matrix,
// filling only the triangle selected by `uplo` (as the dense routines expect).
template <typename T>
std::vector<T> expand_triangle_band(char uplo, int n, int k, const std::vector<T>& ab,
                                    int ldab) {
  std::vector<T> dense(static_cast<std::size_t>(n) * n, T{});
  for (int j = 0; j < n; ++j) {
    if (uplo == 'U') {
      const int first = (j - k > 0) ? (j - k) : 0;
      for (int i = first; i <= j; ++i) {
        dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] =
            ab[static_cast<std::size_t>(k + i - j) + static_cast<std::size_t>(j) * ldab];
      }
    } else {
      const int last = (j + k < n - 1) ? (j + k) : (n - 1);
      for (int i = j; i <= last; ++i) {
        dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] =
            ab[static_cast<std::size_t>(i - j) + static_cast<std::size_t>(j) * ldab];
      }
    }
  }
  return dense;
}

void test_gbmv_float_matches_dense() {
  // 4x4 matrix with kl = 1, ku = 1, stored in a 3-row band array.
  const int n = 4;
  const int kl = 1;
  const int ku = 1;
  const int ldab = kl + ku + 1;
  std::vector<float> ab(static_cast<std::size_t>(ldab) * n, 0.0F);
  for (int j = 0; j < n; ++j) {
    const int first = (j - ku > 0) ? (j - ku) : 0;
    const int last = (j + kl < n - 1) ? (j + kl) : (n - 1);
    for (int i = first; i <= last; ++i) {
      ab[static_cast<std::size_t>(ku + i - j) + static_cast<std::size_t>(j) * ldab] =
          static_cast<float>(i + 1) + 0.5F * static_cast<float>(j);
    }
  }
  const std::vector<float> dense = expand_general_band(n, n, kl, ku, ab, ldab);

  const std::array<float, 4> x = {1.0F, -2.0F, 3.0F, 0.5F};
  for (const char trans : {'N', 'T', 'C'}) {
    std::array<float, 4> y_band = {1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 4> y_dense = y_band;
    thefblas::gbmv(trans, n, n, kl, ku, 2.0F, ab.data(), ldab, x.data(), 1, 0.5F,
                   y_band.data(), 1);
    thefblas::gemv(trans, n, n, 2.0F, dense.data(), n, x.data(), 1, 0.5F, y_dense.data(), 1);
    for (int i = 0; i < n; ++i) {
      assert(almost_equal_scalar(y_band[i], y_dense[i]));
    }
  }
}

void test_gbmv_rectangular_negative_stride() {
  // 3x4, kl = 2, ku = 0 (lower banded rectangular matrix).
  const int m = 3;
  const int n = 4;
  const int kl = 2;
  const int ku = 0;
  const int ldab = kl + ku + 1;
  std::vector<float> ab(static_cast<std::size_t>(ldab) * n, 0.0F);
  for (int j = 0; j < n; ++j) {
    const int last = (j + kl < m - 1) ? (j + kl) : (m - 1);
    for (int i = j; i <= last; ++i) {
      ab[static_cast<std::size_t>(ku + i - j) + static_cast<std::size_t>(j) * ldab] =
          static_cast<float>(1 + i + 2 * j);
    }
  }
  const std::vector<float> dense = expand_general_band(m, n, kl, ku, ab, ldab);

  const std::array<float, 4> x = {1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 3> y_band = {0.0F, 0.0F, 0.0F};
  std::array<float, 3> y_dense = y_band;
  thefblas::gbmv('N', m, n, kl, ku, 1.0F, ab.data(), ldab, x.data(), -1, 0.0F, y_band.data(),
                 -1);
  thefblas::gemv('N', m, n, 1.0F, dense.data(), m, x.data(), -1, 0.0F, y_dense.data(), -1);
  for (int i = 0; i < m; ++i) {
    assert(almost_equal_scalar(y_band[i], y_dense[i]));
  }
}

void test_gbmv_fixed_and_invalid_args() {
  // Tridiagonal 3x3 identity-like band in fixed point.
  const int n = 3;
  const int ldab = 3;
  std::vector<fixed32> ab(static_cast<std::size_t>(ldab) * n, fixed32(0));
  for (int j = 0; j < n; ++j) {
    ab[static_cast<std::size_t>(1) + static_cast<std::size_t>(j) * ldab] = fixed32(2);
  }
  const std::array<fixed32, 3> x = {fixed32(1), fixed32(2), fixed32(3)};
  std::array<fixed32, 3> y = {fixed32(0), fixed32(0), fixed32(0)};
  thefblas::gbmv('N', n, n, 1, 1, fixed32(1), ab.data(), ldab, x.data(), 1, fixed32(0),
                 y.data(), 1);
  assert(almost_equal_scalar(y[0], fixed32(2)));
  assert(almost_equal_scalar(y[1], fixed32(4)));
  assert(almost_equal_scalar(y[2], fixed32(6)));

  // Invalid trans, negative bandwidth and an lda that is too small are no-ops.
  thefblas::gbmv('X', n, n, 1, 1, fixed32(1), ab.data(), ldab, x.data(), 1, fixed32(1),
                 y.data(), 1);
  thefblas::gbmv('N', n, n, -1, 1, fixed32(1), ab.data(), ldab, x.data(), 1, fixed32(1),
                 y.data(), 1);
  thefblas::gbmv('N', n, n, 1, 1, fixed32(1), ab.data(), 1, x.data(), 1, fixed32(1), y.data(),
                 1);
  assert(almost_equal_scalar(y[0], fixed32(2)));
  assert(almost_equal_scalar(y[1], fixed32(4)));
  assert(almost_equal_scalar(y[2], fixed32(6)));
}

void test_sbmv_matches_symv() {
  const int n = 4;
  const int k = 1;
  const int ldab = k + 1;
  for (const char uplo : {'U', 'L'}) {
    std::vector<float> ab(static_cast<std::size_t>(ldab) * n, 0.0F);
    for (int j = 0; j < n; ++j) {
      if (uplo == 'U') {
        ab[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * ldab] =
            static_cast<float>(j + 1);
        if (j > 0) {
          ab[static_cast<std::size_t>(k - 1) + static_cast<std::size_t>(j) * ldab] = 0.5F;
        }
      } else {
        ab[static_cast<std::size_t>(j) * ldab] = static_cast<float>(j + 1);
        if (j < n - 1) {
          ab[static_cast<std::size_t>(1) + static_cast<std::size_t>(j) * ldab] = 0.5F;
        }
      }
    }
    const std::vector<float> dense = expand_triangle_band(uplo, n, k, ab, ldab);
    const std::array<float, 4> x = {1.0F, 2.0F, -1.0F, 3.0F};
    std::array<float, 4> y_band = {1.0F, 0.0F, -1.0F, 2.0F};
    std::array<float, 4> y_dense = y_band;
    thefblas::sbmv(uplo, n, k, 1.5F, ab.data(), ldab, x.data(), 1, 0.25F, y_band.data(), 1);
    thefblas::symv(uplo, n, 1.5F, dense.data(), n, x.data(), 1, 0.25F, y_dense.data(), 1);
    for (int i = 0; i < n; ++i) {
      assert(almost_equal_scalar(y_band[i], y_dense[i]));
    }
  }
}

void test_sbmv_fixed() {
  // Diagonal band (k = 0) in fixed point: y <- alpha * diag(a) * x.
  const int n = 3;
  std::vector<fixed32> ab = {fixed32(1), fixed32(2), fixed32(4)};
  const std::array<fixed32, 3> x = {fixed32(1), fixed32(1), fixed32(1)};
  std::array<fixed32, 3> y = {fixed32(0), fixed32(0), fixed32(0)};
  thefblas::sbmv('U', n, 0, fixed32(2), ab.data(), 1, x.data(), 1, fixed32(0), y.data(), 1);
  assert(almost_equal_scalar(y[0], fixed32(2)));
  assert(almost_equal_scalar(y[1], fixed32(4)));
  assert(almost_equal_scalar(y[2], fixed32(8)));
}

void test_hbmv_matches_hemv() {
  const int n = 3;
  const int k = 1;
  const int ldab = k + 1;
  for (const char uplo : {'U', 'L'}) {
    std::vector<cfloat> ab(static_cast<std::size_t>(ldab) * n, cfloat(0.0F, 0.0F));
    for (int j = 0; j < n; ++j) {
      if (uplo == 'U') {
        ab[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * ldab] =
            cfloat(static_cast<float>(j + 1), 0.0F);
        if (j > 0) {
          ab[static_cast<std::size_t>(k - 1) + static_cast<std::size_t>(j) * ldab] =
              cfloat(0.5F, 1.0F);
        }
      } else {
        ab[static_cast<std::size_t>(j) * ldab] = cfloat(static_cast<float>(j + 1), 0.0F);
        if (j < n - 1) {
          ab[static_cast<std::size_t>(1) + static_cast<std::size_t>(j) * ldab] =
              cfloat(0.5F, -1.0F);
        }
      }
    }
    const std::vector<cfloat> dense = expand_triangle_band(uplo, n, k, ab, ldab);
    const std::array<cfloat, 3> x = {cfloat(1.0F, 1.0F), cfloat(2.0F, 0.0F),
                                     cfloat(0.0F, -1.0F)};
    std::array<cfloat, 3> y_band = {cfloat(1.0F, 0.0F), cfloat(0.0F, 1.0F),
                                    cfloat(-1.0F, 0.0F)};
    std::array<cfloat, 3> y_dense = y_band;
    thefblas::hbmv(uplo, n, k, cfloat(2.0F, 1.0F), ab.data(), ldab, x.data(), 1,
                   cfloat(0.5F, 0.0F), y_band.data(), 1);
    thefblas::hemv(uplo, n, cfloat(2.0F, 1.0F), dense.data(), n, x.data(), 1,
                   cfloat(0.5F, 0.0F), y_dense.data(), 1);
    for (int i = 0; i < n; ++i) {
      assert(almost_equal_scalar(y_band[i], y_dense[i]));
    }
  }
}

void test_hbmv_complex_fixed() {
  const int n = 2;
  const int k = 1;
  const int ldab = k + 1;
  // Upper band: diag = (1, 2), super-diagonal a(0,1) = i.
  std::vector<cfixed> ab = {cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(1), fixed32(0)),
                            cfixed(fixed32(0), fixed32(1)), cfixed(fixed32(2), fixed32(0))};
  const std::array<cfixed, 2> x = {cfixed(fixed32(1), fixed32(0)),
                                   cfixed(fixed32(1), fixed32(0))};
  std::array<cfixed, 2> y = {cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(0), fixed32(0))};
  thefblas::hbmv('U', n, k, cfixed(fixed32(1), fixed32(0)), ab.data(), ldab, x.data(), 1,
                 cfixed(fixed32(0), fixed32(0)), y.data(), 1);
  // y = [1 + i, -i + 2]
  assert(almost_equal_scalar(y[0], cfixed(fixed32(1), fixed32(1))));
  assert(almost_equal_scalar(y[1], cfixed(fixed32(2), fixed32(-1))));
}

void test_tbmv_and_tbsv_match_dense() {
  const int n = 4;
  const int k = 1;
  const int ldab = k + 1;
  for (const char uplo : {'U', 'L'}) {
    for (const char diag : {'N', 'U'}) {
      std::vector<float> ab(static_cast<std::size_t>(ldab) * n, 0.0F);
      for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
          ab[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * ldab] = 2.0F;
          if (j > 0) {
            ab[static_cast<std::size_t>(k - 1) + static_cast<std::size_t>(j) * ldab] = 0.5F;
          }
        } else {
          ab[static_cast<std::size_t>(j) * ldab] = 2.0F;
          if (j < n - 1) {
            ab[static_cast<std::size_t>(1) + static_cast<std::size_t>(j) * ldab] = 0.5F;
          }
        }
      }
      const std::vector<float> dense = expand_triangle_band(uplo, n, k, ab, ldab);
      for (const char trans : {'N', 'T', 'C'}) {
        const std::array<float, 4> x0 = {1.0F, -2.0F, 3.0F, 0.5F};
        std::array<float, 4> x_band = x0;
        std::array<float, 4> x_dense = x0;
        thefblas::tbmv(uplo, trans, diag, n, k, ab.data(), ldab, x_band.data(), 1);
        thefblas::trmv(uplo, trans, diag, n, dense.data(), n, x_dense.data(), 1);
        for (int i = 0; i < n; ++i) {
          assert(almost_equal_scalar(x_band[i], x_dense[i]));
        }

        // Solving with the same triangular matrix undoes the multiply.
        thefblas::tbsv(uplo, trans, diag, n, k, ab.data(), ldab, x_band.data(), 1);
        for (int i = 0; i < n; ++i) {
          assert(almost_equal_scalar(x_band[i], x0[i], 1e-4));
        }
      }
    }
  }
}

void test_tbmv_negative_stride_and_fixed() {
  const int n = 3;
  const int k = 1;
  const int ldab = k + 1;
  std::vector<float> ab(static_cast<std::size_t>(ldab) * n, 0.0F);
  for (int j = 0; j < n; ++j) {
    ab[static_cast<std::size_t>(j) * ldab] = 2.0F;
    if (j < n - 1) {
      ab[static_cast<std::size_t>(1) + static_cast<std::size_t>(j) * ldab] = 1.0F;
    }
  }
  const std::vector<float> dense = expand_triangle_band('L', n, k, ab, ldab);
  const std::array<float, 3> x0 = {1.0F, 2.0F, 3.0F};
  std::array<float, 3> x_band = x0;
  std::array<float, 3> x_dense = x0;
  thefblas::tbmv('L', 'N', 'N', n, k, ab.data(), ldab, x_band.data(), -1);
  thefblas::trmv('L', 'N', 'N', n, dense.data(), n, x_dense.data(), -1);
  for (int i = 0; i < n; ++i) {
    assert(almost_equal_scalar(x_band[i], x_dense[i]));
  }

  // Fixed-point bidiagonal solve: [[2,1],[0,2]] * x = [4, 4] -> x = [1, 2].
  std::vector<fixed32> ab_fixed = {fixed32(0), fixed32(2), fixed32(1), fixed32(2)};
  std::array<fixed32, 2> x_fixed = {fixed32(4), fixed32(4)};
  thefblas::tbsv('U', 'N', 'N', 2, 1, ab_fixed.data(), 2, x_fixed.data(), 1);
  assert(almost_equal_scalar(x_fixed[0], fixed32(1)));
  assert(almost_equal_scalar(x_fixed[1], fixed32(2)));

  // Invalid characters and sizes leave the vector untouched.
  thefblas::tbmv('X', 'N', 'N', 2, 1, ab_fixed.data(), 2, x_fixed.data(), 1);
  thefblas::tbsv('U', 'N', 'Z', 2, 1, ab_fixed.data(), 2, x_fixed.data(), 1);
  thefblas::tbsv('U', 'N', 'N', 0, 1, ab_fixed.data(), 2, x_fixed.data(), 1);
  assert(almost_equal_scalar(x_fixed[0], fixed32(1)));
  assert(almost_equal_scalar(x_fixed[1], fixed32(2)));
}

}  // namespace

int main() {
  test_gbmv_float_matches_dense();
  test_gbmv_rectangular_negative_stride();
  test_gbmv_fixed_and_invalid_args();
  test_sbmv_matches_symv();
  test_sbmv_fixed();
  test_hbmv_matches_hemv();
  test_hbmv_complex_fixed();
  test_tbmv_and_tbsv_match_dense();
  test_tbmv_negative_stride_and_fixed();
}
