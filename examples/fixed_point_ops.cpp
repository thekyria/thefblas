#include "thefblas/thefblas.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

// Q11.20 fixed-point: 32-bit storage, 20 fractional bits.
using q20 = thefblas::fixed<std::int32_t, 20>;

namespace {

void level1_example() {
  std::array<q20, 4> x = {q20(1.5), q20(-2.0), q20(0.25), q20(3.0)};
  std::array<q20, 4> y = {q20(1.0), q20(1.0), q20(1.0), q20(1.0)};

  // y <- 2 * x + y
  thefblas::axpy(4, q20(2), x.data(), 1, y.data(), 1);
  std::cout << "axpy y = [";
  for (const q20 value : y) {
    std::cout << ' ' << value;
  }
  std::cout << " ]\n";

  std::cout << "dot(x, y) = " << thefblas::dot(4, x.data(), 1, y.data(), 1) << '\n';
  std::cout << "nrm2(x)   = " << thefblas::nrm2(4, x.data(), 1) << '\n';
  std::cout << "asum(x)   = " << thefblas::asum(4, x.data(), 1) << '\n';
  std::cout << "iamax(x)  = " << thefblas::iamax(4, x.data(), 1) << '\n';
}

void level2_dense_example() {
  // Column-major 2x3 matrix [[1, 2, 3], [4, 5, 6]].
  const std::array<q20, 6> a = {q20(1), q20(4), q20(2), q20(5), q20(3), q20(6)};
  const std::array<q20, 3> x = {q20(1), q20(1), q20(1)};
  std::array<q20, 2> y = {q20(0), q20(0)};

  // y <- a * x
  thefblas::gemv('N', 2, 3, q20(1), a.data(), 2, x.data(), 1, q20(0), y.data(), 1);
  std::cout << "gemv y = [ " << y[0] << ' ' << y[1] << " ]\n";
}

void level2_banded_example() {
  // Tridiagonal 4x4 matrix with 2 on the diagonal and 1 off-diagonal,
  // in band storage with kl = ku = 1 and ldab = kl + ku + 1 = 3.
  const int n = 4;
  const int kl = 1;
  const int ku = 1;
  const int ldab = kl + ku + 1;
  std::vector<q20> ab(static_cast<std::size_t>(ldab) * n, q20(0));
  for (int j = 0; j < n; ++j) {
    ab[static_cast<std::size_t>(ku) + static_cast<std::size_t>(j) * ldab] = q20(2);  // a(j, j)
    if (j > 0) {
      ab[static_cast<std::size_t>(ku - 1) + static_cast<std::size_t>(j) * ldab] =
          q20(1);  // a(j - 1, j)
    }
    if (j < n - 1) {
      ab[static_cast<std::size_t>(ku + 1) + static_cast<std::size_t>(j) * ldab] =
          q20(1);  // a(j + 1, j)
    }
  }

  const std::array<q20, 4> x = {q20(1), q20(2), q20(3), q20(4)};
  std::array<q20, 4> y = {q20(0), q20(0), q20(0), q20(0)};
  thefblas::gbmv('N', n, n, kl, ku, q20(1), ab.data(), ldab, x.data(), 1, q20(0), y.data(), 1);
  std::cout << "gbmv y = [";
  for (const q20 value : y) {
    std::cout << ' ' << value;
  }
  std::cout << " ]\n";
}

void level2_packed_example() {
  // Upper-triangular packed 3x3 matrix with 2 on the diagonal and 1 above it:
  // ap holds a(0,0), a(0,1), a(1,1), a(0,2), a(1,2), a(2,2).
  std::vector<q20> ap = {q20(2), q20(1), q20(2), q20(1), q20(1), q20(2)};
  std::array<q20, 3> x = {q20(4), q20(6), q20(2)};

  // Solve a * x = b in place with the packed triangular solver.
  thefblas::tpsv('U', 'N', 'N', 3, ap.data(), x.data(), 1);
  std::cout << "tpsv x = [ " << x[0] << ' ' << x[1] << ' ' << x[2] << " ]\n";

  // Symmetric packed matrix-vector product with the same storage.
  const std::array<q20, 3> ones = {q20(1), q20(1), q20(1)};
  std::array<q20, 3> y = {q20(0), q20(0), q20(0)};
  thefblas::spmv('U', 3, q20(1), ap.data(), ones.data(), 1, q20(0), y.data(), 1);
  std::cout << "spmv y = [ " << y[0] << ' ' << y[1] << ' ' << y[2] << " ]\n";
}

}  // namespace

int main() {
  level1_example();
  level2_dense_example();
  level2_banded_example();
  level2_packed_example();
}
