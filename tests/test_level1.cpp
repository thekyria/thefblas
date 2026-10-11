#include "thefblas/fixed.hpp"
#include "thefblas/level1.hpp"

#include <array>
#include <cassert>
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
        assert(almost_equal(x[0], 5.0F));
        assert(almost_equal(y[3], 4.0F));
    }

    {
        const std::array<float, 5> x = {1.0F, 9.0F, 2.0F, 9.0F, 3.0F};
        std::array<float, 3> y = {0.0F, 0.0F, 0.0F};
        thefblas::copy(3, x.data(), 2, y.data(), 1);
        assert(almost_equal(y[0], 1.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 3.0F));
    }

    {
        std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        thefblas::axpy(3, 0.5F, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], 4.5F));
        assert(almost_equal(y[1], 6.0F));
        assert(almost_equal(y[2], 7.5F));
    }

    {
        std::array<float, 3> x = {1.0F, -2.0F, 3.0F};
        thefblas::scal(3, 2.0F, x.data(), 1);
        assert(almost_equal(x[0], 2.0F));
        assert(almost_equal(x[1], -4.0F));
        assert(almost_equal(x[2], 6.0F));
    }

    {
        const std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        const std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        assert(almost_equal(thefblas::dot(3, x.data(), 1, y.data(), 1), 32.0F));
    }

    {
        const std::array<float, 2> x = {3.0F, 4.0F};
        assert(almost_equal(thefblas::nrm2(2, x.data(), 1), 5.0F));
    }

    {
        const std::array<float, 3> x = {-1.0F, 2.0F, -3.0F};
        assert(almost_equal(thefblas::asum(3, x.data(), 1), 6.0F));
        assert(thefblas::iamax(3, x.data(), 1) == 3);
    }

    {
        std::array<float, 3> x = {1.0F, 0.0F, -1.0F};
        std::array<float, 3> y = {0.0F, 1.0F, 2.0F};
        thefblas::rot(3, x.data(), 1, y.data(), 1, 0.0F, 1.0F);
        assert(almost_equal(x[0], 0.0F));
        assert(almost_equal(y[0], -1.0F));
        assert(almost_equal(x[1], 1.0F));
        assert(almost_equal(y[1], 0.0F));
        assert(almost_equal(x[2], 2.0F));
        assert(almost_equal(y[2], 1.0F));
    }

    {
        float a = 3.0F;
        float b = 4.0F;
        float c = 0.0F;
        float s = 0.0F;
        thefblas::rotg(&a, &b, &c, &s);
        assert(almost_equal(a, 5.0F));
        assert(almost_equal(b, 1.0F / 0.6F, 1e-4));
        assert(almost_equal(c, 0.6F, 1e-4));
        assert(almost_equal(s, 0.8F, 1e-4));
    }

    {
        std::array<float, 2> x = {1.0F, 2.0F};
        std::array<float, 2> y = {3.0F, 4.0F};
        const std::array<float, 5> p = {-1.0F, 1.0F, 2.0F, 3.0F, 4.0F};
        thefblas::rotm(2, x.data(), 1, y.data(), 1, p.data());
        assert(almost_equal(x[0], 10.0F));
        assert(almost_equal(y[0], 14.0F));
        assert(almost_equal(x[1], 14.0F));
        assert(almost_equal(y[1], 20.0F));
    }

    {
        float d1 = 1.0F;
        float d2 = 4.0F;
        float b1 = 2.0F;
        const float b2 = 0.0F;
        std::array<float, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, b2, p.data());
        assert(almost_equal(p[0], -2.0F));
    }

    {
        float d1 = -1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        std::array<float, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, 1.0F, p.data());
        assert(almost_equal(p[0], -1.0F));
        assert(d1 == 0.0F && d2 == 0.0F && b1 == 0.0F);
        assert(p[1] == 0.0F && p[2] == 0.0F && p[3] == 0.0F && p[4] == 0.0F);
    }

    {
        const std::array<float, 5> x = {1.0F, 99.0F, 2.0F, 99.0F, 3.0F};
        std::array<float, 3> y = {0.0F, 0.0F, 0.0F};
        thefblas::copy(3, x.data(), -2, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 1.0F));
    }

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        thefblas::swap(1, x.data(), 0, y.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 2.0F));
        assert(thefblas::iamax(1, x.data(), 0) == 0);
    }
}

void test_real_double() {
    std::array<double, 3> x = {1.0, -2.0, 3.0};
    std::array<double, 3> y = {4.0, 5.0, 6.0};
    thefblas::axpy(3, 2.0, x.data(), 1, y.data(), 1);
    assert(almost_equal(y[0], 6.0));
    assert(almost_equal(y[1], 1.0));
    assert(almost_equal(y[2], 12.0));
    assert(almost_equal(thefblas::dot(3, x.data(), 1, y.data(), 1), 40.0));

    double a = 0.0;
    double b = 0.0;
    double c = -1.0;
    double s = -1.0;
    thefblas::rotg(&a, &b, &c, &s);
    assert(almost_equal(a, 0.0));
    assert(almost_equal(b, 0.0));
    assert(almost_equal(c, 1.0));
    assert(almost_equal(s, 0.0));

    const std::array<double, 4> values = {1.0, -5.0, 3.0, 4.0};
    assert(thefblas::iamax(4, values.data(), 1) == 2);
}

void test_real_fixed() {
    {
        std::array<fixed32, 3> x = {fixed32(1.0), fixed32(2.0), fixed32(3.0)};
        std::array<fixed32, 3> y = {fixed32(4.0), fixed32(5.0), fixed32(6.0)};
        thefblas::swap(3, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], fixed32(4.0)));
        assert(almost_equal(y[2], fixed32(3.0)));
    }

    {
        const std::array<fixed32, 3> x = {fixed32(1.0), fixed32(2.0), fixed32(3.0)};
        std::array<fixed32, 3> y = {fixed32(1.0), fixed32(1.0), fixed32(1.0)};
        thefblas::axpy(3, fixed32(0.5), x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], fixed32(1.5), 1e-4));
        assert(almost_equal(y[1], fixed32(2.0), 1e-4));
        assert(almost_equal(y[2], fixed32(2.5), 1e-4));
    }

    {
        std::array<fixed32, 3> x = {fixed32(1.0), fixed32(-2.0), fixed32(3.0)};
        thefblas::scal(3, fixed32(2.0), x.data(), 1);
        assert(almost_equal(x[0], fixed32(2.0)));
        assert(almost_equal(x[1], fixed32(-4.0)));
        assert(almost_equal(x[2], fixed32(6.0)));
        assert(almost_equal(thefblas::dot(3, x.data(), 1, x.data(), 1), fixed32(56.0), 1e-3));
        assert(almost_equal(thefblas::nrm2(3, x.data(), 1), fixed32(7.483314), 5e-3));
        assert(almost_equal(thefblas::asum(3, x.data(), 1), fixed32(12.0), 1e-4));
        assert(thefblas::iamax(3, x.data(), 1) == 3);
    }

    {
        std::array<fixed32, 2> x = {fixed32(1.0), fixed32(0.0)};
        std::array<fixed32, 2> y = {fixed32(0.0), fixed32(1.0)};
        thefblas::rot(2, x.data(), 1, y.data(), 1, fixed32(0.0), fixed32(1.0));
        assert(almost_equal(x[0], fixed32(0.0)));
        assert(almost_equal(y[0], fixed32(-1.0)));
        assert(almost_equal(x[1], fixed32(1.0)));
        assert(almost_equal(y[1], fixed32(0.0)));
    }

    {
        fixed32 a(3.0);
        fixed32 b(4.0);
        fixed32 c(0.0);
        fixed32 s(0.0);
        thefblas::rotg(&a, &b, &c, &s);
        assert(almost_equal(a, fixed32(5.0), 2e-3));
        assert(almost_equal(c, fixed32(0.6), 2e-3));
        assert(almost_equal(s, fixed32(0.8), 2e-3));
    }

    {
        fixed32 d1(1.0);
        fixed32 d2(1.0);
        fixed32 b1(1.0);
        std::array<fixed32, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, fixed32(0.0), p.data());
        assert(almost_equal(p[0], fixed32(-2.0)));
    }

    {
        const std::array<fixed32, 5> x = {fixed32(1.0), fixed32(9.0), fixed32(2.0), fixed32(9.0),
                                          fixed32(3.0)};
        std::array<fixed32, 3> y = {fixed32(0.0), fixed32(0.0), fixed32(0.0)};
        thefblas::copy(3, x.data(), -2, y.data(), 1);
        assert(almost_equal(y[0], fixed32(3.0)));
        assert(almost_equal(y[1], fixed32(2.0)));
        assert(almost_equal(y[2], fixed32(1.0)));
    }
}

void test_complex_float() {
    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, 4.0F}}};
        std::array<std::complex<float>, 2> y = {{{5.0F, 6.0F}, {7.0F, 8.0F}}};
        thefblas::swap(2, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], {5.0F, 6.0F}));
        assert(almost_equal(y[1], {3.0F, 4.0F}));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 1.0F}, {2.0F, -1.0F}}};
        std::array<std::complex<float>, 2> y = {{{3.0F, 0.0F}, {1.0F, 1.0F}}};
        thefblas::axpy(2, std::complex<float>(2.0F, 0.0F), x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {5.0F, 2.0F}));
        assert(almost_equal(y[1], {5.0F, -1.0F}));
    }

    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 0.0F}, {0.0F, 1.0F}}};
        thefblas::scal(2, std::complex<float>(0.0F, 1.0F), x.data(), 1);
        assert(almost_equal(x[0], {0.0F, 1.0F}));
        assert(almost_equal(x[1], {-1.0F, 0.0F}));
        thefblas::scal(2, 0.5F, x.data(), 1);
        assert(almost_equal(x[0], {0.0F, 0.5F}));
        assert(almost_equal(x[1], {-0.5F, 0.0F}));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, -1.0F}}};
        const std::array<std::complex<float>, 2> y = {{{2.0F, 0.0F}, {0.0F, 1.0F}}};
        assert(almost_equal(thefblas::dotu(2, x.data(), 1, y.data(), 1),
                            std::complex<float>(3.0F, 7.0F)));
        assert(almost_equal(thefblas::dotc(2, x.data(), 1, y.data(), 1),
                            std::complex<float>(1.0F, -1.0F)));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{3.0F, 4.0F}, {1.0F, -2.0F}}};
        assert(almost_equal(thefblas::nrm2(2, x.data(), 1), 5.477225F, 1e-4));
        assert(almost_equal(thefblas::asum(2, x.data(), 1), 10.0F));
        assert(thefblas::iamax(2, x.data(), 1) == 1);
    }

    {
        std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{0.0F, 1.0F}}};
        thefblas::rot(1, x.data(), 1, y.data(), 1, 0.0F, 1.0F);
        assert(almost_equal(x[0], {0.0F, 1.0F}));
        assert(almost_equal(y[0], {-1.0F, 0.0F}));
    }

    {
        std::complex<float> a = {0.0F, 0.0F};
        const std::complex<float> b = {3.0F, 4.0F};
        float c = 0.0F;
        std::complex<float> s = {0.0F, 0.0F};
        thefblas::rotg(&a, b, &c, &s);
        assert(almost_equal(c, 0.0F));
        assert(almost_equal(s, {0.6F, -0.8F}));
        assert(almost_equal(a, {5.0F, 0.0F}));
    }

    {
        std::complex<float> a = {0.0F, 0.0F};
        float c = 0.0F;
        std::complex<float> s = {2.0F, 2.0F};
        thefblas::rotg(&a, {0.0F, 0.0F}, &c, &s);
        assert(almost_equal(c, 1.0F));
        assert(almost_equal(s, {0.0F, 0.0F}));
        assert(almost_equal(a, {0.0F, 0.0F}));
    }
}

void test_complex_double() {
    const std::array<std::complex<double>, 5> x = {
        {{1.0, 1.0}, {9.0, 9.0}, {2.0, -1.0}, {9.0, 9.0}, {3.0, 0.5}}};
    std::array<std::complex<double>, 3> y = {};
    thefblas::copy(3, x.data(), -2, y.data(), 1);
    assert(almost_equal(y[0], std::complex<double>(3.0, 0.5)));
    assert(almost_equal(y[1], std::complex<double>(2.0, -1.0)));
    assert(almost_equal(y[2], std::complex<double>(1.0, 1.0)));
}

void test_complex_fixed() {
    using cfixed = std::complex<fixed32>;

    {
        std::array<cfixed, 2> x = {cfixed(fixed32(1.0), fixed32(2.0)),
                                   cfixed(fixed32(3.0), fixed32(-1.0))};
        std::array<cfixed, 2> y = {cfixed(fixed32(2.0), fixed32(0.0)),
                                   cfixed(fixed32(0.0), fixed32(1.0))};
        assert(almost_equal(thefblas::dotu(2, x.data(), 1, y.data(), 1),
                            cfixed(fixed32(3.0), fixed32(7.0)), 2e-3));
        assert(almost_equal(thefblas::dotc(2, x.data(), 1, y.data(), 1),
                            cfixed(fixed32(1.0), fixed32(-1.0)), 2e-3));
    }

    {
        std::array<cfixed, 2> x = {cfixed(fixed32(3.0), fixed32(4.0)),
                                   cfixed(fixed32(1.0), fixed32(-2.0))};
        thefblas::scal(2, fixed32(0.5), x.data(), 1);
        assert(almost_equal(x[0], cfixed(fixed32(1.5), fixed32(2.0)), 2e-3));
        assert(almost_equal(thefblas::nrm2(2, x.data(), 1), fixed32(2.738612), 5e-3));
        assert(almost_equal(thefblas::asum(2, x.data(), 1), fixed32(5.0), 2e-3));
        assert(thefblas::iamax(2, x.data(), 1) == 1);
    }

    {
        std::array<cfixed, 1> x = {cfixed(fixed32(1.0), fixed32(0.0))};
        std::array<cfixed, 1> y = {cfixed(fixed32(0.0), fixed32(1.0))};
        thefblas::rot(1, x.data(), 1, y.data(), 1, fixed32(0.0), fixed32(1.0));
        assert(almost_equal(x[0], cfixed(fixed32(0.0), fixed32(1.0)), 2e-3));
        assert(almost_equal(y[0], cfixed(fixed32(-1.0), fixed32(0.0)), 2e-3));
    }

    {
        std::complex<fixed32> a(fixed32(1.0), fixed32(0.0));
        std::complex<fixed32> b(fixed32(0.0), fixed32(1.0));
        fixed32 c(0.0);
        std::complex<fixed32> s(fixed32(0.0), fixed32(0.0));
        thefblas::rotg(&a, b, &c, &s);
        assert(almost_equal(c, fixed32(0.707106), 5e-3));
        assert(almost_equal(a, std::complex<fixed32>(fixed32(1.414213), fixed32(0.0)), 5e-3));
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
        assert(rel_close(thefblas::nrm2(2, x.data(), 1), 5e20F, 1e-6F));
        assert(rel_close(thefblas::nrm2(2, x.data(), -1), 5e20F, 1e-6F));
    }
    {
        // 3e-30^2 underflows float to zero.
        const std::array<float, 2> x = {3e-30F, 4e-30F};
        assert(rel_close(thefblas::nrm2(2, x.data(), 1), 5e-30F, 1e-6F));
    }
    {
        const std::array<double, 2> x = {3e200, 4e200};
        assert(rel_close(thefblas::nrm2(2, x.data(), 1), 5e200, 1e-12));
        const std::array<double, 2> y = {3e-200, 4e-200};
        assert(rel_close(thefblas::nrm2(2, y.data(), 1), 5e-200, 1e-12));
    }
    {
        // Mixed big, medium and small magnitudes.
        const std::array<double, 3> x = {3e200, 1.0, 4e200};
        assert(rel_close(thefblas::nrm2(3, x.data(), 1), 5e200, 1e-12));
        const std::array<float, 2> y = {1e-25F, 1.0F};
        assert(thefblas::nrm2(2, y.data(), 1) == 1.0F);
    }
    {
        const std::array<std::complex<float>, 2> x = {std::complex<float>(3e20F, 0.0F),
                                                      std::complex<float>(0.0F, -4e20F)};
        assert(rel_close(thefblas::nrm2(2, x.data(), 1), 5e20F, 1e-6F));
        const std::array<std::complex<double>, 1> y = {std::complex<double>(3e-200, 4e-200)};
        assert(rel_close(thefblas::nrm2(1, y.data(), 1), 5e-200, 1e-12));
    }
    {
        constexpr float inf = std::numeric_limits<float>::infinity();
        const std::array<float, 3> x = {1.0F, inf, -inf};
        assert(std::isinf(thefblas::nrm2(3, x.data(), 1)));
        const std::array<float, 2> y = {1.0F, std::numeric_limits<float>::quiet_NaN()};
        assert(std::isnan(thefblas::nrm2(2, y.data(), 1)));
    }
    {
        // 2^24 float elements of 2^52: the true norm 2^64 is representable,
        // so the medium bin must not overflow for long vectors.
        thefblas::detail::blue_nrm2<float> norm;
        const float big = std::ldexp(1.0F, 52);
        for (int i = 0; i < (1 << 24); ++i) {
            norm.add(big);
        }
        assert(norm.value() == std::ldexp(1.0F, 64));
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
    assert(p[0] == expected_flag);
    thefblas::rotm(1, &x, 1, &y, 1, p.data());
    assert(std::fabs(x - b1) <= 1e-12 * std::fabs(b1));
    assert(std::fabs(y) <= 1e-12 * std::fabs(b1));
    // The weighted norm d1 * b1^2 + d2 * b2^2 is preserved.
    assert(std::fabs(d1 * b1 * b1 - weighted) <= 1e-12 * std::fabs(weighted));
    // Netlib keeps the scale factors within [2^-24, 2^24].
    assert(d1 > 5.9e-8 && d1 < 16777216.0);
}

// Behaviour aligned with Netlib reference BLAS by the randomised comparison in
// test_reference_blas.cpp (which needs a system CBLAS and is opt-in).
void test_netlib_conventions() {
    // scal and asum treat a non-positive stride as an empty vector.
    std::array<float, 3> x = {1.0F, -2.0F, 3.0F};
    thefblas::scal(3, 2.0F, x.data(), -1);
    assert(x[0] == 1.0F && x[1] == -2.0F && x[2] == 3.0F);
    assert(thefblas::asum(3, x.data(), -1) == 0.0F);
    std::array<std::complex<float>, 2> cx = {{{1.0F, 1.0F}, {2.0F, -2.0F}}};
    thefblas::scal(2, 2.0F, cx.data(), -1);
    thefblas::scal(2, std::complex<float>(0.0F, 1.0F), cx.data(), -1);
    assert(cx[0] == std::complex<float>(1.0F, 1.0F));
    assert(thefblas::asum(2, cx.data(), -1) == 0.0F);

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
    assert(fp_small[0] == -1.0F);
    assert(fd1_small > 5.9e-8F && fd2_small > 5.9e-8F);

    // Real rotg with |a| > |b|: z = s (with |a| <= |b|, z = 1/c is covered above).
    double ra = 4.0;
    double rb = 3.0;
    double rc = 0.0;
    double rs = 0.0;
    thefblas::rotg(&ra, &rb, &rc, &rs);
    assert(std::fabs(ra - 5.0) < 1e-12 && std::fabs(rc - 0.8) < 1e-12);
    assert(std::fabs(rs - 0.6) < 1e-12 && std::fabs(rb - 0.6) < 1e-12);

    // rotmg rejects a negative d2 that makes |q2| dominate: everything is zeroed.
    double d1 = 1.0;
    double d2 = -4.0;
    double b1 = 1.0;
    std::array<double, 5> p = {9.0, 9.0, 9.0, 9.0, 9.0};
    thefblas::rotmg(&d1, &d2, &b1, 1.0, p.data());
    assert(p[0] == -1.0 && p[1] == 0.0 && p[2] == 0.0 && p[3] == 0.0 && p[4] == 0.0);
    assert(d1 == 0.0 && d2 == 0.0 && b1 == 0.0);

    // Fixed point skips the rescaling but follows the same flags.
    fixed32 fd1(2.0);
    fixed32 fd2(1.0);
    fixed32 fb1(3.0);
    std::array<fixed32, 5> fp = {};
    thefblas::rotmg(&fd1, &fd2, &fb1, fixed32(1.0), fp.data());
    assert(fp[0] == fixed32(0.0));
    assert(almost_equal(fp[2], fixed32(-1.0 / 3.0), 1e-5));
    assert(almost_equal(fp[3], fixed32(1.0 / 6.0), 1e-5));
    assert(almost_equal(fb1, fixed32(19.0 / 6.0), 1e-5));
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
    return 0;
}
