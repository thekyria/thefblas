#include "thefblas/thefblas.h"

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <limits>

namespace {

using fixed32 = thefblas::fixed<std::int32_t, 20>;
using fixed16 = thefblas::fixed<std::int16_t, 7>;
using complex32 = std::complex<fixed32>;

void test_level1_empty_inputs() {
    std::array<double, 2> x = {1.0, 2.0};
    std::array<double, 2> y = {3.0, 4.0};
    thefblas::copy(0, x.data(), 1, y.data(), 1);
    thefblas::copy(2, x.data(), 0, y.data(), 1);
    thefblas::axpy(0, 1.0, x.data(), 1, y.data(), 1);
    thefblas::axpy(2, 1.0, x.data(), 1, y.data(), 0);
    thefblas::scal(0, 2.0, x.data(), 1);
    thefblas::scal(2, 2.0, x.data(), 0);
    assert(thefblas::dot(0, x.data(), 1, y.data(), 1) == 0.0);
    assert(thefblas::dot(2, x.data(), 0, y.data(), 1) == 0.0);
    assert(thefblas::nrm2(0, x.data(), 1) == 0.0);
    assert(thefblas::nrm2(2, x.data(), 0) == 0.0);
    assert(thefblas::asum(0, x.data(), 1) == 0.0);
    assert(thefblas::asum(2, x.data(), 0) == 0.0);
    assert(thefblas::iamax(2, x.data(), 0) == 0);

    using complex = std::complex<double>;
    const std::array<complex, 2> cx = {complex(1.0, 2.0), complex(3.0, 4.0)};
    const std::array<complex, 2> cy = {complex(2.0, 1.0), complex(4.0, 3.0)};
    assert(thefblas::dotu(0, cx.data(), 1, cy.data(), 1) == complex{});
    assert(thefblas::dotc(0, cx.data(), 1, cy.data(), 1) == complex{});
    assert(thefblas::iamax(2, cx.data(), 0) == 0);
    assert(thefblas::iamax(2, cx.data(), 1) == 2);

    std::array<complex, 2> cy_scaled = cy;
    thefblas::scal(2, 2.0, cy_scaled.data(), 0);
    assert(cy_scaled == cy);
}

void test_fixed_arithmetic_paths() {
    const fixed16 a(2);
    const fixed16 b(1);
    assert((a + b) == fixed16(3));
    assert((a - b) == fixed16(1));
    assert((a * b) == fixed16(2));
    assert((a / b) == fixed16(2));

    using saturated32 = thefblas::fixed<std::int32_t, 20, thefblas::saturate>;
    const saturated32 raw_min =
        saturated32::from_raw((std::numeric_limits<std::int32_t>::min)());
    assert(thefblas::asum(1, &raw_min, 1).raw() ==
           (std::numeric_limits<std::int32_t>::max)());
    assert(thefblas::nrm2(1, &a, 1) == a);
}

void test_level2_zero_scalars() {
    const double zero = 0.0;
    const double one = 1.0;
    const double matrix[4] = {1.0, 2.0, 3.0, 4.0};
    const double vector[2] = {2.0, 3.0};
    double output[2] = {5.0, 6.0};

    thefblas::gemv('N', 2, 2, zero, matrix, 2, vector, 1, 2.0, output, 1);
    assert(output[0] == 10.0 && output[1] == 12.0);
    thefblas::symv('U', 2, zero, matrix, 2, vector, 1, one, output, 1);
    thefblas::gbmv('N', 2, 2, 1, 1, zero, matrix, 3, vector, 1, one, output, 1);
    thefblas::sbmv('U', 2, 1, zero, matrix, 2, vector, 1, one, output, 1);
    thefblas::spmv('U', 2, zero, matrix, vector, 1, one, output, 1);

    using complex = std::complex<double>;
    const complex czero{};
    const complex cone(1.0, 0.0);
    const complex cmatrix[4] = {complex(1.0, 0.0), complex(2.0, 1.0),
                                complex(3.0, -1.0), complex(4.0, 0.0)};
    const complex cvector[2] = {complex(2.0, 1.0), complex(3.0, -1.0)};
    complex coutput[2] = {complex(5.0, 2.0), complex(6.0, 3.0)};
    thefblas::gemv('C', 2, 2, czero, cmatrix, 2, cvector, 1, cone, coutput, 1);
    thefblas::hemv('U', 2, czero, cmatrix, 2, cvector, 1, cone, coutput, 1);
    thefblas::hbmv('L', 2, 1, czero, cmatrix, 2, cvector, 1, cone, coutput, 1);
    thefblas::hpmv('L', 2, czero, cmatrix, cvector, 1, cone, coutput, 1);
}

void test_fixed_triangular_transposes() {
    const fixed32 zero(0);
    const fixed32 one(1);
    const fixed32 dense[4] = {one, fixed32(0), fixed32(2), one};
    fixed32 x[2] = {fixed32(3), fixed32(5)};
    thefblas::trsv('U', 'T', 'N', 2, dense, 2, x, 1);
    x[0] = fixed32(3);
    x[1] = fixed32(5);
    thefblas::trsv('L', 'T', 'U', 2, dense, 2, x, 1);

    const complex32 cdense[4] = {complex32(one, one), complex32(zero, zero),
                                 complex32(one, fixed32(-1)), complex32(one, zero)};
    complex32 cx[2] = {complex32(one, one), complex32(fixed32(2), zero)};
    thefblas::trsv('U', 'C', 'N', 2, cdense, 2, cx, 1);

    const fixed32 band[6] = {zero, one, one, one, one, zero};
    fixed32 bx[2] = {one, one};
    thefblas::tbmv('U', 'T', 'N', 2, 1, band, 2, bx, 1);
    bx[0] = one;
    bx[1] = one;
    thefblas::tbmv('L', 'N', 'U', 2, 1, band, 2, bx, 1);

    const fixed32 packed[3] = {one, one, one};
    fixed32 px[2] = {one, one};
    thefblas::tpmv('U', 'T', 'N', 2, packed, px, 1);
    px[0] = one;
    px[1] = one;
    thefblas::tpsv('L', 'N', 'U', 2, packed, px, 1);
}

} // namespace

int main() {
    test_level1_empty_inputs();
    test_fixed_arithmetic_paths();
    test_level2_zero_scalars();
    test_fixed_triangular_transposes();
    return 0;
}
