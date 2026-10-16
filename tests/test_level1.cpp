#include "test_support.hpp"
#include "thefblas/fixed.hpp"
#include "thefblas/level1.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

using fixed32 = thefblas::fixed<std::int32_t, 20>;

template <typename T> double to_double(const T &value) {
    if constexpr (std::is_same_v<T, fixed32>) {
        return value.template to_float<double>();
    } else {
        return static_cast<double>(value);
    }
}

template <typename T>
bool almost_equal_scalar(const T &lhs, const T &rhs, double tolerance = 1e-5) {
    const double diff = to_double(lhs - rhs);
    return diff <= tolerance && diff >= -tolerance;
}

template <typename T>
bool almost_equal(const std::complex<T> &lhs, const std::complex<T> &rhs, double tolerance = 1e-5) {
    return almost_equal_scalar(lhs.real(), rhs.real(), tolerance) &&
           almost_equal_scalar(lhs.imag(), rhs.imag(), tolerance);
}

template <typename T> bool almost_equal(const T &lhs, const T &rhs, double tolerance = 1e-5) {
    return almost_equal_scalar(lhs, rhs, tolerance);
}

void test_real_float() {
    {
        std::array<float, 4> x = {1.0F, 2.0F, 3.0F, 4.0F};
        std::array<float, 4> y = {5.0F, 6.0F, 7.0F, 8.0F};
        thefblas::swap(4, x.data(), 1, y.data(), 1);
        CHECK_PRED(almost_equal, x[0], 5.0F);
        CHECK_PRED(almost_equal, y[3], 4.0F);
    }

    {
        const std::array<float, 5> x = {1.0F, 9.0F, 2.0F, 9.0F, 3.0F};
        std::array<float, 3> y = {0.0F, 0.0F, 0.0F};
        thefblas::copy(3, x.data(), 2, y.data(), 1);
        CHECK_PRED(almost_equal, y[0], 1.0F);
        CHECK_PRED(almost_equal, y[1], 2.0F);
        CHECK_PRED(almost_equal, y[2], 3.0F);
    }

    {
        std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        thefblas::axpy(3, 0.5F, x.data(), 1, y.data(), 1);
        CHECK_PRED(almost_equal, y[0], 4.5F);
        CHECK_PRED(almost_equal, y[1], 6.0F);
        CHECK_PRED(almost_equal, y[2], 7.5F);
    }

    {
        std::array<float, 3> x = {1.0F, -2.0F, 3.0F};
        thefblas::scal(3, 2.0F, x.data(), 1);
        CHECK_PRED(almost_equal, x[0], 2.0F);
        CHECK_PRED(almost_equal, x[1], -4.0F);
        CHECK_PRED(almost_equal, x[2], 6.0F);
    }

    {
        const std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        const std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        CHECK_PRED(almost_equal, thefblas::dot(3, x.data(), 1, y.data(), 1), 32.0F);
    }

    {
        const std::array<float, 2> x = {3.0F, 4.0F};
        CHECK_PRED(almost_equal, thefblas::nrm2(2, x.data(), 1), 5.0F);
    }

    {
        const std::array<float, 3> x = {-1.0F, 2.0F, -3.0F};
        CHECK_PRED(almost_equal, thefblas::asum(3, x.data(), 1), 6.0F);
        CHECK(thefblas::iamax(3, x.data(), 1) == 3);
    }

    {
        std::array<float, 3> x = {1.0F, 0.0F, -1.0F};
        std::array<float, 3> y = {0.0F, 1.0F, 2.0F};
        thefblas::rot(3, x.data(), 1, y.data(), 1, 0.0F, 1.0F);
        CHECK_PRED(almost_equal, x[0], 0.0F);
        CHECK_PRED(almost_equal, y[0], -1.0F);
        CHECK_PRED(almost_equal, x[1], 1.0F);
        CHECK_PRED(almost_equal, y[1], 0.0F);
        CHECK_PRED(almost_equal, x[2], 2.0F);
        CHECK_PRED(almost_equal, y[2], 1.0F);
    }

    {
        float a = 3.0F;
        float b = 4.0F;
        float c = 0.0F;
        float s = 0.0F;
        thefblas::rotg(&a, &b, &c, &s);
        CHECK_PRED(almost_equal, a, 5.0F);
        CHECK_PRED(almost_equal, b, 1.0F / 0.6F, 1e-4);
        CHECK_PRED(almost_equal, c, 0.6F, 1e-4);
        CHECK_PRED(almost_equal, s, 0.8F, 1e-4);
    }

    {
        std::array<float, 2> x = {1.0F, 2.0F};
        std::array<float, 2> y = {3.0F, 4.0F};
        const std::array<float, 5> p = {-1.0F, 1.0F, 2.0F, 3.0F, 4.0F};
        thefblas::rotm(2, x.data(), 1, y.data(), 1, p.data());
        CHECK_PRED(almost_equal, x[0], 10.0F);
        CHECK_PRED(almost_equal, y[0], 14.0F);
        CHECK_PRED(almost_equal, x[1], 14.0F);
        CHECK_PRED(almost_equal, y[1], 20.0F);
    }

    {
        float d1 = 1.0F;
        float d2 = 4.0F;
        float b1 = 2.0F;
        const float b2 = 0.0F;
        std::array<float, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, b2, p.data());
        CHECK_PRED(almost_equal, p[0], -2.0F);
    }

    {
        float d1 = -1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        std::array<float, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, 1.0F, p.data());
        CHECK_PRED(almost_equal, p[0], -1.0F);
        CHECK(d1 == 0.0F && d2 == 0.0F && b1 == 0.0F);
        CHECK(p[1] == 0.0F && p[2] == 0.0F && p[3] == 0.0F && p[4] == 0.0F);
    }

    {
        const std::array<float, 5> x = {1.0F, 99.0F, 2.0F, 99.0F, 3.0F};
        std::array<float, 3> y = {0.0F, 0.0F, 0.0F};
        thefblas::copy(3, x.data(), -2, y.data(), 1);
        CHECK_PRED(almost_equal, y[0], 3.0F);
        CHECK_PRED(almost_equal, y[1], 2.0F);
        CHECK_PRED(almost_equal, y[2], 1.0F);
    }

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        thefblas::swap(1, x.data(), 0, y.data(), 1);
        CHECK_PRED(almost_equal, x[0], 1.0F);
        CHECK_PRED(almost_equal, y[0], 2.0F);
        CHECK(thefblas::iamax(1, x.data(), 0) == 0);
    }
}

void test_real_double() {
    std::array<double, 3> x = {1.0, -2.0, 3.0};
    std::array<double, 3> y = {4.0, 5.0, 6.0};
    thefblas::axpy(3, 2.0, x.data(), 1, y.data(), 1);
    CHECK_PRED(almost_equal, y[0], 6.0);
    CHECK_PRED(almost_equal, y[1], 1.0);
    CHECK_PRED(almost_equal, y[2], 12.0);
    CHECK_PRED(almost_equal, thefblas::dot(3, x.data(), 1, y.data(), 1), 40.0);

    double a = 0.0;
    double b = 0.0;
    double c = -1.0;
    double s = -1.0;
    thefblas::rotg(&a, &b, &c, &s);
    CHECK_PRED(almost_equal, a, 0.0);
    CHECK_PRED(almost_equal, b, 0.0);
    CHECK_PRED(almost_equal, c, 1.0);
    CHECK_PRED(almost_equal, s, 0.0);

    const std::array<double, 4> values = {1.0, -5.0, 3.0, 4.0};
    CHECK(thefblas::iamax(4, values.data(), 1) == 2);
}

void test_real_fixed() {
    {
        std::array<fixed32, 3> x = {fixed32(1.0), fixed32(2.0), fixed32(3.0)};
        std::array<fixed32, 3> y = {fixed32(4.0), fixed32(5.0), fixed32(6.0)};
        thefblas::swap(3, x.data(), 1, y.data(), 1);
        CHECK_PRED(almost_equal, x[0], fixed32(4.0));
        CHECK_PRED(almost_equal, y[2], fixed32(3.0));
    }

    {
        const std::array<fixed32, 3> x = {fixed32(1.0), fixed32(2.0), fixed32(3.0)};
        std::array<fixed32, 3> y = {fixed32(1.0), fixed32(1.0), fixed32(1.0)};
        thefblas::axpy(3, fixed32(0.5), x.data(), 1, y.data(), 1);
        CHECK_PRED(almost_equal, y[0], fixed32(1.5), 1e-4);
        CHECK_PRED(almost_equal, y[1], fixed32(2.0), 1e-4);
        CHECK_PRED(almost_equal, y[2], fixed32(2.5), 1e-4);
    }

    {
        std::array<fixed32, 3> x = {fixed32(1.0), fixed32(-2.0), fixed32(3.0)};
        thefblas::scal(3, fixed32(2.0), x.data(), 1);
        CHECK_PRED(almost_equal, x[0], fixed32(2.0));
        CHECK_PRED(almost_equal, x[1], fixed32(-4.0));
        CHECK_PRED(almost_equal, x[2], fixed32(6.0));
        CHECK_PRED(almost_equal, thefblas::dot(3, x.data(), 1, x.data(), 1), fixed32(56.0), 1e-3);
        CHECK_PRED(almost_equal, thefblas::nrm2(3, x.data(), 1), fixed32(7.483314), 5e-3);
        CHECK_PRED(almost_equal, thefblas::asum(3, x.data(), 1), fixed32(12.0), 1e-4);
        CHECK(thefblas::iamax(3, x.data(), 1) == 3);
    }

    {
        std::array<fixed32, 2> x = {fixed32(1.0), fixed32(0.0)};
        std::array<fixed32, 2> y = {fixed32(0.0), fixed32(1.0)};
        thefblas::rot(2, x.data(), 1, y.data(), 1, fixed32(0.0), fixed32(1.0));
        CHECK_PRED(almost_equal, x[0], fixed32(0.0));
        CHECK_PRED(almost_equal, y[0], fixed32(-1.0));
        CHECK_PRED(almost_equal, x[1], fixed32(1.0));
        CHECK_PRED(almost_equal, y[1], fixed32(0.0));
    }

    {
        fixed32 a(3.0);
        fixed32 b(4.0);
        fixed32 c(0.0);
        fixed32 s(0.0);
        thefblas::rotg(&a, &b, &c, &s);
        CHECK_PRED(almost_equal, a, fixed32(5.0), 2e-3);
        CHECK_PRED(almost_equal, c, fixed32(0.6), 2e-3);
        CHECK_PRED(almost_equal, s, fixed32(0.8), 2e-3);
    }

    {
        fixed32 d1(1.0);
        fixed32 d2(1.0);
        fixed32 b1(1.0);
        std::array<fixed32, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, fixed32(0.0), p.data());
        CHECK_PRED(almost_equal, p[0], fixed32(-2.0));
    }

    {
        const std::array<fixed32, 5> x = {fixed32(1.0), fixed32(9.0), fixed32(2.0), fixed32(9.0),
                                          fixed32(3.0)};
        std::array<fixed32, 3> y = {fixed32(0.0), fixed32(0.0), fixed32(0.0)};
        thefblas::copy(3, x.data(), -2, y.data(), 1);
        CHECK_PRED(almost_equal, y[0], fixed32(3.0));
        CHECK_PRED(almost_equal, y[1], fixed32(2.0));
        CHECK_PRED(almost_equal, y[2], fixed32(1.0));
    }
}

void test_complex_float() {
    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, 4.0F}}};
        std::array<std::complex<float>, 2> y = {{{5.0F, 6.0F}, {7.0F, 8.0F}}};
        thefblas::swap(2, x.data(), 1, y.data(), 1);
        CHECK(almost_equal(x[0], {5.0F, 6.0F}));
        CHECK(almost_equal(y[1], {3.0F, 4.0F}));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 1.0F}, {2.0F, -1.0F}}};
        std::array<std::complex<float>, 2> y = {{{3.0F, 0.0F}, {1.0F, 1.0F}}};
        thefblas::axpy(2, std::complex<float>(2.0F, 0.0F), x.data(), 1, y.data(), 1);
        CHECK(almost_equal(y[0], {5.0F, 2.0F}));
        CHECK(almost_equal(y[1], {5.0F, -1.0F}));
    }

    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 0.0F}, {0.0F, 1.0F}}};
        thefblas::scal(2, std::complex<float>(0.0F, 1.0F), x.data(), 1);
        CHECK(almost_equal(x[0], {0.0F, 1.0F}));
        CHECK(almost_equal(x[1], {-1.0F, 0.0F}));
        thefblas::scal(2, 0.5F, x.data(), 1);
        CHECK(almost_equal(x[0], {0.0F, 0.5F}));
        CHECK(almost_equal(x[1], {-0.5F, 0.0F}));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, -1.0F}}};
        const std::array<std::complex<float>, 2> y = {{{2.0F, 0.0F}, {0.0F, 1.0F}}};
        CHECK_PRED(almost_equal, thefblas::dotu(2, x.data(), 1, y.data(), 1),
                   std::complex<float>(3.0F, 7.0F));
        CHECK_PRED(almost_equal, thefblas::dotc(2, x.data(), 1, y.data(), 1),
                   std::complex<float>(1.0F, -1.0F));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{3.0F, 4.0F}, {1.0F, -2.0F}}};
        CHECK_PRED(almost_equal, thefblas::nrm2(2, x.data(), 1), 5.477225F, 1e-4);
        CHECK_PRED(almost_equal, thefblas::asum(2, x.data(), 1), 10.0F);
        CHECK(thefblas::iamax(2, x.data(), 1) == 1);
    }

    {
        std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{0.0F, 1.0F}}};
        thefblas::rot(1, x.data(), 1, y.data(), 1, 0.0F, 1.0F);
        CHECK(almost_equal(x[0], {0.0F, 1.0F}));
        CHECK(almost_equal(y[0], {-1.0F, 0.0F}));
    }

    {
        std::complex<float> a = {0.0F, 0.0F};
        const std::complex<float> b = {3.0F, 4.0F};
        float c = 0.0F;
        std::complex<float> s = {0.0F, 0.0F};
        thefblas::rotg(&a, b, &c, &s);
        CHECK_PRED(almost_equal, c, 0.0F);
        CHECK(almost_equal(s, {0.6F, -0.8F}));
        CHECK(almost_equal(a, {5.0F, 0.0F}));
    }

    {
        std::complex<float> a = {0.0F, 0.0F};
        float c = 0.0F;
        std::complex<float> s = {2.0F, 2.0F};
        thefblas::rotg(&a, {0.0F, 0.0F}, &c, &s);
        CHECK_PRED(almost_equal, c, 1.0F);
        CHECK(almost_equal(s, {0.0F, 0.0F}));
        CHECK(almost_equal(a, {0.0F, 0.0F}));
    }
}

void test_complex_double() {
    const std::array<std::complex<double>, 5> x = {
        {{1.0, 1.0}, {9.0, 9.0}, {2.0, -1.0}, {9.0, 9.0}, {3.0, 0.5}}};
    std::array<std::complex<double>, 3> y = {};
    thefblas::copy(3, x.data(), -2, y.data(), 1);
    CHECK_PRED(almost_equal, y[0], std::complex<double>(3.0, 0.5));
    CHECK_PRED(almost_equal, y[1], std::complex<double>(2.0, -1.0));
    CHECK_PRED(almost_equal, y[2], std::complex<double>(1.0, 1.0));
}

void test_complex_fixed() {
    using cfixed = std::complex<fixed32>;

    {
        std::array<cfixed, 2> x = {cfixed(fixed32(1.0), fixed32(2.0)),
                                   cfixed(fixed32(3.0), fixed32(-1.0))};
        std::array<cfixed, 2> y = {cfixed(fixed32(2.0), fixed32(0.0)),
                                   cfixed(fixed32(0.0), fixed32(1.0))};
        CHECK_PRED(almost_equal, thefblas::dotu(2, x.data(), 1, y.data(), 1),
                   cfixed(fixed32(3.0), fixed32(7.0)), 2e-3);
        CHECK_PRED(almost_equal, thefblas::dotc(2, x.data(), 1, y.data(), 1),
                   cfixed(fixed32(1.0), fixed32(-1.0)), 2e-3);
    }

    {
        std::array<cfixed, 2> x = {cfixed(fixed32(3.0), fixed32(4.0)),
                                   cfixed(fixed32(1.0), fixed32(-2.0))};
        thefblas::scal(2, fixed32(0.5), x.data(), 1);
        CHECK_PRED(almost_equal, x[0], cfixed(fixed32(1.5), fixed32(2.0)), 2e-3);
        CHECK_PRED(almost_equal, thefblas::nrm2(2, x.data(), 1), fixed32(2.738612), 5e-3);
        CHECK_PRED(almost_equal, thefblas::asum(2, x.data(), 1), fixed32(5.0), 2e-3);
        CHECK(thefblas::iamax(2, x.data(), 1) == 1);
    }

    {
        std::array<cfixed, 1> x = {cfixed(fixed32(1.0), fixed32(0.0))};
        std::array<cfixed, 1> y = {cfixed(fixed32(0.0), fixed32(1.0))};
        thefblas::rot(1, x.data(), 1, y.data(), 1, fixed32(0.0), fixed32(1.0));
        CHECK_PRED(almost_equal, x[0], cfixed(fixed32(0.0), fixed32(1.0)), 2e-3);
        CHECK_PRED(almost_equal, y[0], cfixed(fixed32(-1.0), fixed32(0.0)), 2e-3);
    }

    {
        std::complex<fixed32> a(fixed32(1.0), fixed32(0.0));
        std::complex<fixed32> b(fixed32(0.0), fixed32(1.0));
        fixed32 c(0.0);
        std::complex<fixed32> s(fixed32(0.0), fixed32(0.0));
        thefblas::rotg(&a, b, &c, &s);
        CHECK_PRED(almost_equal, c, fixed32(0.707106), 5e-3);
        CHECK_PRED(almost_equal, a, std::complex<fixed32>(fixed32(1.414213), fixed32(0.0)), 5e-3);
    }
}

// nrm2 must not overflow or underflow when the norm itself is representable,
// even though the plain sum of squares would (Blue's algorithm).
template <typename T> bool rel_close(T actual, T expected, T rel) {
    return std::fabs(actual - expected) <= rel * std::fabs(expected);
}

void test_nrm2_scaling() {
    {
        // 3e20^2 overflows float; the norm 5e20 does not.
        const std::array<float, 2> x = {3e20F, -4e20F};
        CHECK_PRED(rel_close, thefblas::nrm2(2, x.data(), 1), 5e20F, 1e-6F);
        CHECK_PRED(rel_close, thefblas::nrm2(2, x.data(), -1), 5e20F, 1e-6F);
    }
    {
        // 3e-30^2 underflows float to zero.
        const std::array<float, 2> x = {3e-30F, 4e-30F};
        CHECK_PRED(rel_close, thefblas::nrm2(2, x.data(), 1), 5e-30F, 1e-6F);
    }
    {
        const std::array<double, 2> x = {3e200, 4e200};
        CHECK_PRED(rel_close, thefblas::nrm2(2, x.data(), 1), 5e200, 1e-12);
        const std::array<double, 2> y = {3e-200, 4e-200};
        CHECK_PRED(rel_close, thefblas::nrm2(2, y.data(), 1), 5e-200, 1e-12);
    }
    {
        // Mixed big, medium and small magnitudes.
        const std::array<double, 3> x = {3e200, 1.0, 4e200};
        CHECK_PRED(rel_close, thefblas::nrm2(3, x.data(), 1), 5e200, 1e-12);
        const std::array<float, 2> y = {1e-25F, 1.0F};
        CHECK(thefblas::nrm2(2, y.data(), 1) == 1.0F);
    }
    {
        const std::array<std::complex<float>, 2> x = {std::complex<float>(3e20F, 0.0F),
                                                      std::complex<float>(0.0F, -4e20F)};
        CHECK_PRED(rel_close, thefblas::nrm2(2, x.data(), 1), 5e20F, 1e-6F);
        const std::array<std::complex<double>, 1> y = {std::complex<double>(3e-200, 4e-200)};
        CHECK_PRED(rel_close, thefblas::nrm2(1, y.data(), 1), 5e-200, 1e-12);
    }
    {
        constexpr float inf = std::numeric_limits<float>::infinity();
        const std::array<float, 3> x = {1.0F, inf, -inf};
        CHECK(std::isinf(thefblas::nrm2(3, x.data(), 1)));
        const std::array<float, 2> y = {1.0F, std::numeric_limits<float>::quiet_NaN()};
        CHECK(std::isnan(thefblas::nrm2(2, y.data(), 1)));
    }
    {
        // 2^24 float elements of 2^52: the true norm 2^64 is representable,
        // so the medium bin must not overflow for long vectors.
        thefblas::detail::blue_nrm2<float> norm;
        const float big = std::ldexp(1.0F, 52);
        for (int i = 0; i < (1 << 24); ++i) {
            norm.add(big);
        }
        CHECK(norm.value() == std::ldexp(1.0F, 64));
    }
}

} // namespace

// H * (b1, b2)^T must equal (b1', 0)^T for the H that rotmg encodes.
void check_rotmg(double d1, double d2, double b1, double b2, double expected_flag) {
    const double weighted = d1 * b1 * b1 + d2 * b2 * b2;
    std::array<double, 5> p = {};
    double x = b1;
    double y = b2;
    thefblas::rotmg(&d1, &d2, &b1, b2, p.data());
    CHECK(p[0] == expected_flag);
    thefblas::rotm(1, &x, 1, &y, 1, p.data());
    CHECK(std::fabs(x - b1) <= 1e-12 * std::fabs(b1));
    CHECK(std::fabs(y) <= 1e-12 * std::fabs(b1));
    // The weighted norm d1 * b1^2 + d2 * b2^2 is preserved.
    CHECK(std::fabs(d1 * b1 * b1 - weighted) <= 1e-12 * std::fabs(weighted));
    // Netlib keeps the scale factors within [2^-24, 2^24].
    CHECK(d1 > 5.9e-8 && d1 < 16777216.0);
}

// Behaviour aligned with Netlib reference BLAS by the randomised comparison in
// test_reference_blas.cpp (which needs a system CBLAS and is opt-in).
void test_netlib_conventions() {
    // scal and asum treat a non-positive stride as an empty vector.
    std::array<float, 3> x = {1.0F, -2.0F, 3.0F};
    thefblas::scal(3, 2.0F, x.data(), -1);
    CHECK(x[0] == 1.0F && x[1] == -2.0F && x[2] == 3.0F);
    CHECK(thefblas::asum(3, x.data(), -1) == 0.0F);
    std::array<std::complex<float>, 2> cx = {{{1.0F, 1.0F}, {2.0F, -2.0F}}};
    thefblas::scal(2, 2.0F, cx.data(), -1);
    thefblas::scal(2, std::complex<float>(0.0F, 1.0F), cx.data(), -1);
    CHECK(cx[0] == std::complex<float>(1.0F, 1.0F));
    CHECK(thefblas::asum(2, cx.data(), -1) == 0.0F);

    // rotmg: flag 0 (|q1| > |q2|), flag 1 (|q1| <= |q2|) and the rescaling of
    // d1 and d2 into [2^-24, 2^24] from both sides, starting from either form.
    check_rotmg(2.0, 1.0, 3.0, 1.0, 0.0);
    check_rotmg(1.0, 2.0, 1.0, 3.0, 1.0);
    check_rotmg(1e-12, 1e-14, 1.0, 1.0, -1.0);
    check_rotmg(1e12, 1e10, 1.0, 1.0, -1.0);
    check_rotmg(1e-14, 1e-12, 1.0, 1.0, -1.0);
    check_rotmg(1e10, 1e12, 1.0, 1.0, -1.0);
    check_rotmg(1.0, 1e-12, 1.0, 1e3, -1.0);
    check_rotmg(1.0, -1e12, 1.0, 1e-9, -1.0);
    check_rotmg(1.0, 1e12, 1.0, 1e-8, -1.0);
    check_rotmg(1.0, -1e-12, 1.0, 1e-3, -1.0);

    // The same rescaling for float (each element type has its own instantiation).
    float fd1_small = 1e-12F;
    float fd2_small = 1e-14F;
    float fb1_small = 1.0F;
    std::array<float, 5> fp_small = {};
    thefblas::rotmg(&fd1_small, &fd2_small, &fb1_small, 1.0F, fp_small.data());
    CHECK(fp_small[0] == -1.0F);
    CHECK(fd1_small > 5.9e-8F && fd2_small > 5.9e-8F);

    // Complex rotg must not underflow or overflow |a| and |b| for finite extremes.
    {
        std::complex<float> a = {0.0F, 0.0F};
        float c = 1.0F;
        std::complex<float> s = {};
        thefblas::rotg(&a, {1e-30F, 0.0F}, &c, &s);
        CHECK(c == 0.0F && s == std::complex<float>(1.0F, 0.0F));
        CHECK(std::fabs(a.real() - 1e-30F) <= 1e-36F && a.imag() == 0.0F);

        a = {0.0F, 0.0F};
        thefblas::rotg(&a, {3e30F, 4e30F}, &c, &s);
        CHECK(c == 0.0F && almost_equal(s, {0.6F, -0.8F}));
        CHECK(std::fabs(a.real() - 5e30F) <= 1e25F);

        a = {1e-30F, 0.0F};
        thefblas::rotg(&a, {1.0F, 0.0F}, &c, &s);
        CHECK(c > 0.0F && c < 2e-30F);
        CHECK(almost_equal(s, {1.0F, 0.0F}) && almost_equal(a, {1.0F, 0.0F}));
    }

    // Real rotg with |a| > |b|: z = s (with |a| <= |b|, z = 1/c is covered above).
    double ra = 4.0;
    double rb = 3.0;
    double rc = 0.0;
    double rs = 0.0;
    thefblas::rotg(&ra, &rb, &rc, &rs);
    CHECK(std::fabs(ra - 5.0) < 1e-12 && std::fabs(rc - 0.8) < 1e-12);
    CHECK(std::fabs(rs - 0.6) < 1e-12 && std::fabs(rb - 0.6) < 1e-12);

    // rotmg rejects a negative d2 that makes |q2| dominate: everything is zeroed.
    double d1 = 1.0;
    double d2 = -4.0;
    double b1 = 1.0;
    std::array<double, 5> p = {9.0, 9.0, 9.0, 9.0, 9.0};
    thefblas::rotmg(&d1, &d2, &b1, 1.0, p.data());
    CHECK(p[0] == -1.0 && p[1] == 0.0 && p[2] == 0.0 && p[3] == 0.0 && p[4] == 0.0);
    CHECK(d1 == 0.0 && d2 == 0.0 && b1 == 0.0);

    // Fixed point skips the rescaling but follows the same flags.
    fixed32 fd1(2.0);
    fixed32 fd2(1.0);
    fixed32 fb1(3.0);
    std::array<fixed32, 5> fp = {};
    thefblas::rotmg(&fd1, &fd2, &fb1, fixed32(1.0), fp.data());
    CHECK(fp[0] == fixed32(0.0));
    CHECK_PRED(almost_equal, fp[2], fixed32(-1.0 / 3.0), 1e-5);
    CHECK_PRED(almost_equal, fp[3], fixed32(1.0 / 6.0), 1e-5);
    CHECK_PRED(almost_equal, fb1, fixed32(19.0 / 6.0), 1e-5);
}

// Formats without integer bits (range [-1, 1)) cannot represent 1.
void test_rotations_without_integer_bits() {
    // |a| > |b|: every output, z = s included, is representable, so the default
    // checked policy must not assert.
    {
        thefblas::q15 a(0.4);
        thefblas::q15 b(0.3);
        thefblas::q15 c;
        thefblas::q15 s;
        thefblas::rotg(&a, &b, &c, &s);
        CHECK(std::fabs(a.to_float() - 0.5) < 1e-4);
        CHECK(std::fabs(c.to_float() - 0.8) < 1e-4);
        CHECK(std::fabs(s.to_float() - 0.6) < 1e-4);
        CHECK(b == s);
    }
    // |a| <= |b|: z = 1/c exceeds the range and is resolved by the policy.
    {
        thefblas::q15_sat a(0.3);
        thefblas::q15_sat b(0.4);
        thefblas::q15_sat c;
        thefblas::q15_sat s;
        thefblas::rotg(&a, &b, &c, &s);
        CHECK(std::fabs(a.to_float() - 0.5) < 1e-4);
        CHECK(std::fabs(c.to_float() - 0.6) < 1e-4);
        CHECK(std::fabs(s.to_float() - 0.8) < 1e-4);
        CHECK(b == (std::numeric_limits<thefblas::q15_sat>::max)());
    }

    // rotm/rotmg reject (at compile time) the formats that cannot hold their
    // flag values -2 and 1.
    static_assert(!thefblas::detail::holds_rotm_flags<thefblas::q7>(), "Q1.7");
    static_assert(!thefblas::detail::holds_rotm_flags<thefblas::q15_sat>(), "Q1.15");
    static_assert(!thefblas::detail::holds_rotm_flags<thefblas::q31>(), "Q1.31");
    static_assert(thefblas::detail::holds_rotm_flags<thefblas::fixed<std::int16_t, 14>>(),
                  "Q2.14 holds [-2, 2)");
    static_assert(thefblas::detail::holds_rotm_flags<fixed32>(), "Q12.20");
    static_assert(thefblas::detail::holds_rotm_flags<float>(), "float");
}

void test_extended_dot() {
    // float: 1e8 + 1 - 1e8 loses the 1 when summed in float, not in double.
    const float x[3] = {1e8F, 1.0F, -1e8F};
    const float ones[3] = {1.0F, 1.0F, 1.0F};
    static_assert(std::is_same_v<decltype(thefblas::dsdot(3, x, 1, ones, 1)), double>,
                  "dsdot of float returns double");
    CHECK(thefblas::dot(3, x, 1, ones, 1) == 0.0F);
    CHECK(thefblas::dsdot(3, x, 1, ones, 1) == 1.0);
    CHECK(thefblas::sdsdot(3, 0.5F, x, 1, ones, 1) == 1.5F);
    // Negative strides walk the vectors backwards, as in dot.
    const float a[2] = {1.0F, 2.0F};
    const float b[2] = {3.0F, 4.0F};
    CHECK(thefblas::dsdot(2, a, -1, b, 1) == 2.0 * 3.0 + 1.0 * 4.0);
    // n <= 0 or a zero stride: zero, and sb returned unchanged.
    CHECK(thefblas::dsdot(0, a, 1, b, 1) == 0.0);
    CHECK(std::signbit(thefblas::sdsdot(0, -0.0F, a, 1, b, 1)));
    CHECK(thefblas::sdsdot(2, 2.5F, a, 0, b, 1) == 2.5F);

    // fixed: 0.75^2 + 0.75^2 = 1.125 is outside Q1.15. dsdot returns it in the
    // widened type; sdsdot adds sb = -0.5 there and narrows 0.625 once, where
    // dot followed by an addition saturates first.
    using Q15 = thefblas::fixed<std::int16_t, 15, thefblas::saturate>;
    using Wide = thefblas::fixed<std::int32_t, 15, thefblas::saturate>;
    const Q15 q[2] = {Q15(0.75), Q15(0.75)};
    static_assert(std::is_same_v<decltype(thefblas::dsdot(2, q, 1, q, 1)), Wide>,
                  "dsdot of fixed returns the widened fixed type");
    CHECK(thefblas::dsdot(2, q, 1, q, 1) == Wide(1.125));
    CHECK(thefblas::sdsdot(2, Q15(-0.5), q, 1, q, 1) == Q15(0.625));
    CHECK(thefblas::dot(2, q, 1, q, 1) + Q15(-0.5) != Q15(0.625));
    CHECK(thefblas::dsdot(0, q, 1, q, 1) == Wide(0.0));
    CHECK(thefblas::sdsdot(0, Q15(0.25), q, 1, q, 1) == Q15(0.25));
    // sb takes part in the single rounding: in Q11.4, sb = 1 LSB and the product
    // (-2 LSB) * (4 LSB) = -0.5 LSB sum to exactly +0.5 LSB, which rounds to 1
    // LSB; rounding the product first (to -1 LSB) would give 0.
    using Q4 = thefblas::fixed<std::int16_t, 4>;
    const Q4 neg[1] = {Q4::from_raw(-2)};
    const Q4 pos[1] = {Q4::from_raw(4)};
    const Q4 one_lsb = Q4::from_raw(1);
    CHECK(thefblas::sdsdot(1, one_lsb, neg, 1, pos, 1).raw() == 1);

    // Only fixed-point types with a wider accumulator for the running sum are
    // accepted: not 64-bit storage, and 32-bit storage only with __int128.
    static_assert(thefblas::detail::has_extended_v<float>, "float");
    static_assert(!thefblas::detail::has_extended_v<double>, "no portable wider type");
    static_assert(thefblas::detail::has_extended_v<thefblas::fixed<std::int8_t, 7>>, "8-bit");
    static_assert(thefblas::detail::has_extended_v<Q15>, "16-bit");
#if defined(__SIZEOF_INT128__)
    static_assert(thefblas::detail::has_extended_v<thefblas::q31>, "32-bit with __int128");
    static_assert(!thefblas::detail::has_extended_v<thefblas::fixed<std::int64_t, 32>>, "64-bit");
#else
    static_assert(!thefblas::detail::has_extended_v<thefblas::q31>, "32-bit without __int128");
#endif

    // Range overloads.
    const std::array<float, 3> vx = {1e8F, 1.0F, -1e8F};
    const std::array<float, 3> vy = {1.0F, 1.0F, 1.0F};
    CHECK(thefblas::dsdot(vx, vy) == 1.0);
    CHECK(thefblas::sdsdot(0.25F, vx, vy) == 1.25F);
}

int main() {
    test_real_float();
    test_real_double();
    test_real_fixed();
    test_complex_float();
    test_complex_double();
    test_complex_fixed();
    test_nrm2_scaling();
    test_netlib_conventions();
    test_rotations_without_integer_bits();
    test_extended_dot();
    return thefblas_test::finish();
}
