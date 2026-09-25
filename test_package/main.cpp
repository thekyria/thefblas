#include "thefblas/thefblas.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

int main() {
  // Level-1 axpy on float: y <- 2 * x + y
  std::array<float, 3> xf = {1.0F, 2.0F, 3.0F};
  std::array<float, 3> yf = {1.0F, 1.0F, 1.0F};
  thefblas::axpy(3, 2.0F, xf.data(), 1, yf.data(), 1);
  const bool float_ok = std::fabs(yf[0] - 3.0F) < 1e-5F &&
                        std::fabs(yf[1] - 5.0F) < 1e-5F &&
                        std::fabs(yf[2] - 7.0F) < 1e-5F;

  // Level-1 axpy on a fixed-point value type (Q11.20).
  using q20 = thefblas::fixed<std::int32_t, 20>;
  std::array<q20, 3> xq = {q20(1.0), q20(2.0), q20(3.0)};
  std::array<q20, 3> yq = {q20(1.0), q20(1.0), q20(1.0)};
  thefblas::axpy(3, q20(2.0), xq.data(), 1, yq.data(), 1);
  const bool fixed_ok = std::fabs(yq[0].to_float<double>() - 3.0) < 1e-3 &&
                        std::fabs(yq[2].to_float<double>() - 7.0) < 1e-3;

  std::printf("thefblas test_package: float_ok=%d fixed_ok=%d\n",
              static_cast<int>(float_ok), static_cast<int>(fixed_ok));
  return (float_ok && fixed_ok) ? 0 : 1;
}
