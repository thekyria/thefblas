#include "thefblas/thefblas.h"

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <limits>

namespace {

using fixed32 = thefblas::fixed<std::int32_t, 20>;
using fixed16 = thefblas::fixed<std::int16_t, 7>;
using fixed8 = thefblas::fixed<std::int16_t, 0>;
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
    std::array<complex, 2> cx = {complex(1.0, 2.0), complex(3.0, 4.0)};
    const std::array<complex, 2> cy = {complex(2.0, 1.0), complex(4.0, 3.0)};
    assert(thefblas::dotu(0, cx.data(), 1, cy.data(), 1) == complex{});
    assert(thefblas::dotc(0, cx.data(), 1, cy.data(), 1) == complex{});
    assert(thefblas::iamax(2, cx.data(), 0) == 0);
    assert(thefblas::iamax(2, cx.data(), 1) == 2);
    assert(thefblas::nrm2(0, cx.data(), 1) == 0.0);
    assert(thefblas::asum(0, cx.data(), 1) == 0.0);

    std::array<complex, 2> cy_scaled = cy;
    thefblas::scal(2, 2.0, cy_scaled.data(), 0);
    assert(cy_scaled == cy);
    thefblas::rot(0, cx.data(), 1, cy_scaled.data(), 1, 1.0, 0.0);

    const double givens[5] = {-2.0, 0.0, 0.0, 0.0, 0.0};
    thefblas::rotm(2, x.data(), 1, y.data(), 1, givens);
    thefblas::rotm(0, x.data(), 1, y.data(), 1, givens);

    double a = 3.0;
    double b = 4.0;
    double c = 0.0;
    double s = 0.0;
    thefblas::rotg(&a, &b, &c, &s);
    assert(c > 0.0 && s > 0.0);
}

void test_fixed_arithmetic_paths() {
    const fixed16 a(2);
    const fixed16 b(1);
    assert((a + b) == fixed16(3));
    assert((a - b) == fixed16(1));
    assert((a * b) == fixed16(2));
    assert((a / b) == fixed16(2));
    fixed16 accum = a;
    accum += b;
    accum -= b;
    accum *= b;
    accum /= b;
    assert(accum == a);

    using saturated32 = thefblas::fixed<std::int32_t, 20, thefblas::saturate>;
    const saturated32 raw_min =
        saturated32::from_raw((std::numeric_limits<std::int32_t>::min)());
    assert(thefblas::asum(1, &raw_min, 1).raw() ==
           (std::numeric_limits<std::int32_t>::max)());
    assert(thefblas::nrm2(1, &a, 1) == a);
    const fixed8 whole(2);
    assert((whole * whole) == fixed8(4));

    using saturating16 = thefblas::fixed<std::int16_t, 15, thefblas::saturate>;
    const saturating16 half(0.5);
    assert((half + half) == saturating16(1));
    assert((half - half) == saturating16(0));
    assert((half * half) == saturating16(0.25));
    saturating16 accumulated = half;
    accumulated += half;
    accumulated -= half;
    accumulated *= half;
    accumulated /= half;
    assert(accumulated == half);

    using saturating8 = thefblas::fixed<std::int16_t, 8, thefblas::saturate>;
    const saturating8 eighth(0.125);
    assert((eighth + eighth) == saturating8(0.25));
    assert((eighth - eighth) == saturating8(0));
    assert((eighth * eighth) == saturating8(0.015625));
    saturating8 accumulated8 = eighth;
    accumulated8 += eighth;
    accumulated8 -= eighth;
    accumulated8 *= eighth;
    accumulated8 /= eighth;
    assert(accumulated8 == eighth);

    const fixed32 checked64(3);
    fixed32 mutable64 = checked64;
    mutable64 += fixed32(1);
    mutable64 -= fixed32(1);
    mutable64 *= fixed32(1);
    mutable64 /= fixed32(1);
    assert(mutable64 != fixed32(0));
#if defined(__SIZEOF_INT128__)
    using fixed64 = thefblas::fixed<std::int64_t, 32>;
    fixed64 accum64(3);
    accum64 += fixed64(1);
    accum64 -= fixed64(1);
    accum64 *= fixed64(1);
    accum64 /= fixed64(1);
    assert(accum64 != fixed64(0));
#endif

    using saturating32 = thefblas::fixed<std::int32_t, 20, thefblas::saturate>;
    assert(saturating32(10000).raw() == (std::numeric_limits<std::int32_t>::max)());

    thefblas::detail::real_mac<saturating16> mac;
    mac.add(half);
    assert(mac.value() == half);
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

    const double symmetric[4] = {1.0, 2.0, 2.0, 4.0};
    thefblas::symv('U', 2, one, symmetric, 2, vector, 1, zero, output, 1);
    const double band[4] = {0.0, 1.0, 2.0, 0.0};
    thefblas::sbmv('U', 2, 1, one, band, 2, vector, 1, zero, output, 1);
    thefblas::sbmv('U', 2, 1, one, band, 1, vector, 1, zero, output, 1);
    const double packed[3] = {1.0, 2.0, 4.0};
    thefblas::spmv('L', 2, one, packed, vector, 1, zero, output, 1);

    const fixed32 fx[2] = {fixed32(1), fixed32(2)};
    fixed32 fy[2] = {fixed32(3), fixed32(4)};
    const fixed32 fmatrix[4] = {fixed32(1), fixed32(0), fixed32(0), fixed32(1)};
    thefblas::gemv('N', 2, 2, fixed32(0), fmatrix, 2, fx, 1, fixed32(2), fy, 1);
    thefblas::symv('U', 2, fixed32(0), fmatrix, 2, fx, 1, fixed32(1), fy, 1);
    thefblas::gemv('C', 2, 2, fixed32(1), fmatrix, 2, fx, 1, fixed32(0), fy, 1);

    fixed16 f16y[2] = {fixed16(2), fixed16(3)};
    const fixed16 f16x[2] = {fixed16(1), fixed16(2)};
    const fixed16 f16matrix[4] = {fixed16(1), fixed16(0), fixed16(0), fixed16(1)};
    thefblas::gemv('N', 2, 2, fixed16(0), f16matrix, 2, f16x, 1, fixed16(2), f16y, 1);
    thefblas::gemv('C', 2, 2, fixed16(1), f16matrix, 2, f16x, 1, fixed16(0), f16y, 1);

    using saturating16 = thefblas::fixed<std::int16_t, 15, thefblas::saturate>;
    saturating16 saty[2] = {saturating16(2), saturating16(3)};
    const saturating16 satmatrix[4] = {saturating16(1), saturating16(0),
                                       saturating16(0), saturating16(1)};
    thefblas::gemv('N', 2, 2, saturating16(0), satmatrix, 2, saty, 1, saturating16(2), saty, 1);
    thefblas::gemv('C', 2, 2, saturating16(1), satmatrix, 2, saty, 1, saturating16(0), saty, 1);

    using complex = std::complex<double>;
    const complex czero{};
    const complex cone(1.0, 0.0);
    const complex cmatrix[4] = {complex(1.0, 0.0), complex(2.0, 1.0),
                                complex(3.0, -1.0), complex(4.0, 0.0)};
    const complex cvector[2] = {complex(2.0, 1.0), complex(3.0, -1.0)};
    complex coutput[2] = {complex(5.0, 2.0), complex(6.0, 3.0)};
    thefblas::gemv('C', 2, 2, czero, cmatrix, 2, cvector, 1, cone, coutput, 1);
    thefblas::gemv('T', 2, 2, one, matrix, 2, vector, 1, zero, output, 1);
    thefblas::gemv('C', 2, 2, one, matrix, 2, vector, 1, zero, output, 1);
    thefblas::gemv('C', 2, 2, cone, cmatrix, 2, cvector, 1, czero, coutput, 1);
    thefblas::hemv('U', 2, czero, cmatrix, 2, cvector, 1, cone, coutput, 1);
    thefblas::hbmv('L', 2, 1, czero, cmatrix, 2, cvector, 1, cone, coutput, 1);
    thefblas::hpmv('L', 2, czero, cmatrix, cvector, 1, cone, coutput, 1);
    thefblas::hemv('U', 2, cone, cmatrix, 2, cvector, 1, czero, coutput, 1);
    thefblas::hbmv('L', 2, 1, cone, cmatrix, 2, cvector, 1, czero, coutput, 1);
    thefblas::hpmv('U', 2, cone, cmatrix, cvector, 1, czero, coutput, 1);
    thefblas::hbmv('U', 2, 1, cone, cmatrix, 1, cvector, 1, czero, coutput, 1);
    thefblas::hpmv('U', 2, cone, cmatrix, cvector, 0, czero, coutput, 1);

    double ger_matrix[4] = {};
    thefblas::ger(2, 2, 1.0, vector, 1, vector, 1, ger_matrix, 2);
    thefblas::ger(2, 2, 0.0, vector, 1, vector, 1, ger_matrix, 2);
    thefblas::syr('U', 2, 0.0, vector, 1, ger_matrix, 2);
    thefblas::syr2('L', 2, 0.0, vector, 1, vector, 1, ger_matrix, 2);
    thefblas::her('U', 2, 0.0, cvector, 1, coutput, 2);
    thefblas::her2('L', 2, czero, cvector, 1, cvector, 1, coutput, 2);
    complex cger_matrix[4] = {};
    thefblas::geru(2, 2, cone, cvector, 1, cvector, 1, cger_matrix, 2);
    thefblas::geru(2, 2, czero, cvector, 1, cvector, 1, cger_matrix, 2);
    thefblas::gerc(2, 2, cone, cvector, 1, cvector, 1, cger_matrix, 2);
    thefblas::gerc(2, 2, czero, cvector, 1, cvector, 1, cger_matrix, 2);

    complex packed_complex[3] = {cone, cone, cone};
    thefblas::hpr('U', 2, 0.0, cvector, 1, packed_complex);
    thefblas::hpr2('L', 2, czero, cvector, 1, cvector, 1,
                   packed_complex);
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
    x[0] = fixed32(3);
    x[1] = fixed32(5);
    thefblas::trsv('U', 'C', 'N', 2, dense, 2, x, 1);

    const complex32 cdense[4] = {complex32(one, one), complex32(zero, zero),
                                 complex32(one, fixed32(-1)), complex32(one, zero)};
    complex32 cx[2] = {complex32(one, one), complex32(fixed32(2), zero)};
    thefblas::trsv('U', 'C', 'N', 2, cdense, 2, cx, 1);
    cx[0] = complex32(one, one);
    cx[1] = complex32(fixed32(2), zero);
    thefblas::trsv('L', 'C', 'N', 2, cdense, 2, cx, 1);

    const double real_dense[4] = {2.0, 0.0, 1.0, 3.0};
    double real_x[2] = {5.0, 6.0};
    thefblas::trsv('U', 'T', 'N', 2, real_dense, 2, real_x, 1);
    real_x[0] = 5.0;
    real_x[1] = 6.0;
    thefblas::trsv('U', 'C', 'N', 2, real_dense, 2, real_x, 1);
    const std::complex<double> double_complex[4] = {
        {2.0, 1.0}, {0.0, 0.0}, {1.0, -1.0}, {3.0, 0.0}};
    std::complex<double> double_cx[2] = {{1.0, 2.0}, {3.0, 4.0}};
    thefblas::trsv('U', 'N', 'N', 2, double_complex, 2, double_cx, 1);
    double_cx[0] = {1.0, 2.0};
    double_cx[1] = {3.0, 4.0};
    thefblas::trsv('U', 'T', 'N', 2, double_complex, 2, double_cx, 1);
    double_cx[0] = {1.0, 2.0};
    double_cx[1] = {3.0, 4.0};
    thefblas::trsv('L', 'T', 'N', 2, double_complex, 2, double_cx, 1);
    double_cx[0] = {1.0, 2.0};
    double_cx[1] = {3.0, 4.0};
    thefblas::trsv('U', 'C', 'N', 2, double_complex, 2, double_cx, 1);

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

void test_remaining_fixed_types_and_zero_alpha() {
    using saturating15 = thefblas::fixed<std::int16_t, 15, thefblas::saturate>;
    using saturating8 = thefblas::fixed<std::int16_t, 8, thefblas::saturate>;
    using Q15Complex = std::complex<saturating15>;
    using Q8Complex = std::complex<saturating8>;
    const saturating15 one15(1);
    const saturating8 one8(1);
    const saturating15 zero15(0);
    const saturating8 zero8(0);

    saturating15 y15[1] = {one15};
    const saturating15 a15[1] = {one15};
    thefblas::gemv('C', 1, 1, one15, a15, 1, a15, 1, zero15, y15, 1);
    thefblas::gemv('N', 1, 1, zero15, a15, 1, a15, 1, one15, y15, 1);
    thefblas::symv('L', 1, one15, a15, 1, a15, 1, zero15, y15, 1);
    thefblas::sbmv('U', 1, 0, one15, a15, 1, a15, 1, zero15, y15, 1);
    thefblas::spmv('L', 1, one15, a15, a15, 1, zero15, y15, 1);

    saturating8 y8[1] = {one8};
    const saturating8 a8[1] = {one8};
    thefblas::gemv('C', 1, 1, one8, a8, 1, a8, 1, zero8, y8, 1);
    thefblas::symv('L', 1, one8, a8, 1, a8, 1, zero8, y8, 1);
    thefblas::gemv('N', 1, 1, zero8, a8, 1, a8, 1, one8, y8, 1);

    const Q15Complex c15one(one15, zero15);
    const Q15Complex c15zero(zero15, zero15);
    const Q15Complex c15a[1] = {c15one};
    Q15Complex c15y[1] = {c15one};
    thefblas::hemv('L', 1, c15one, c15a, 1, c15a, 1, c15zero, c15y, 1);
    thefblas::hbmv('U', 1, 0, c15one, c15a, 1, c15a, 1, c15zero, c15y, 1);
    thefblas::hpmv('L', 1, c15one, c15a, c15a, 1, c15zero, c15y, 1);
    thefblas::hemv('U', 1, c15zero, c15a, 1, c15a, 1, c15one, c15y, 1);

    const Q8Complex c8one(one8, zero8);
    const Q8Complex c8zero(zero8, zero8);
    const Q8Complex c8a[1] = {c8one};
    Q8Complex c8y[1] = {c8one};
    thefblas::hemv('L', 1, c8one, c8a, 1, c8a, 1, c8zero, c8y, 1);
    thefblas::hpmv('U', 1, c8one, c8a, c8a, 1, c8zero, c8y, 1);
    thefblas::gemv('C', 1, 1, c8one, c8a, 1, c8a, 1, c8zero, c8y, 1);
    thefblas::hemv('U', 1, c8zero, c8a, 1, c8a, 1, c8one, c8y, 1);

    const std::complex<double> x[2] = {{1.0, 1.0}, {2.0, -1.0}};
    const std::complex<double> y[2] = {{2.0, 1.0}, {1.0, 2.0}};
    (void)thefblas::dotu(2, x, 1, y, 1);
    (void)thefblas::dotc(2, x, 1, y, 1);
    (void)thefblas::asum(2, x, 1);

    double rx[2] = {1.0, 2.0};
    double ry[2] = {3.0, 4.0};
    double rmat[4] = {2.0, 0.0, 1.0, 2.0};
    thefblas::rot(0, rx, 1, ry, 1, 1.0, 0.0);
    const double plus_flag[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double zero_flag[5] = {0.0, 2.0, 3.0, 4.0, 5.0};
    thefblas::rotm(2, rx, 1, ry, 1, plus_flag);
    thefblas::rotm(2, rx, 1, ry, 1, zero_flag);
    thefblas::ger(2, 2, 0.0, rx, 1, ry, 1, rmat, 2);
    thefblas::syr('U', 2, 0.0, rx, 1, rmat, 2);
    thefblas::syr2('L', 2, 0.0, rx, 1, ry, 1, rmat, 2);
    std::complex<double> cmat[4] = {};
    thefblas::her('L', 2, 0.0, x, 1, cmat, 2);
    thefblas::her2('U', 2, std::complex<double>{}, x, 1, y, 1, cmat, 2);

#if defined(__SIZEOF_INT128__)
    using fixed64 = thefblas::fixed<std::int64_t, 32>;
    const fixed64 one64(1);
    fixed64 output64[1] = {one64};
    thefblas::gemv('C', 1, 1, one64, &one64, 1, &one64, 1, fixed64(0), output64, 1);
    thefblas::gemv('N', 1, 1, fixed64(0), &one64, 1, &one64, 1, one64, output64, 1);
#endif

    const complex32 c32one(fixed32(1), fixed32(0));
    const complex32 c32matrix[4] = {
        c32one, complex32{}, complex32(fixed32(1), fixed32(1)), c32one};
    complex32 c32output[2] = {c32one, complex32(fixed32(2), fixed32(1))};
    thefblas::gemv('C', 1, 1, c32one, &c32one, 1, &c32one, 1, complex32{}, c32output, 1);
    thefblas::trsv('U', 'T', 'N', 2, c32matrix, 2, c32output, 1);
    c32output[0] = c32one;
    c32output[1] = complex32(fixed32(2), fixed32(1));
    thefblas::trsv('L', 'C', 'N', 2, c32matrix, 2, c32output, 1);

    const std::complex<float> float_matrix[4] = {
        {2.0F, 1.0F}, {0.0F, 0.0F}, {1.0F, -1.0F}, {3.0F, 0.0F}};
    std::complex<float> float_x[2] = {{1.0F, 2.0F}, {3.0F, 4.0F}};
    thefblas::trsv('U', 'T', 'N', 2, float_matrix, 2, float_x, 1);
}

} // namespace

int main() {
    test_level1_empty_inputs();
    test_fixed_arithmetic_paths();
    test_level2_zero_scalars();
    test_fixed_triangular_transposes();
    test_remaining_fixed_types_and_zero_alpha();
    return 0;
}
