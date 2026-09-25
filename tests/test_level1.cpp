#include "thefblas/fixed.hpp"
#include "thefblas/level1.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
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
        assert(almost_equal(p[0], -1.0F));
    }

    {
        float d1 = -1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        std::array<float, 5> p = {};
        thefblas::rotmg(&d1, &d2, &b1, 1.0F, p.data());
        assert(almost_equal(p[0], -2.0F));
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
        assert(almost_equal(p[0], fixed32(-1.0)));
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
        assert(almost_equal(s, {1.0F, 0.0F}));
        assert(almost_equal(a, b));
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

} // namespace

int main() {
    test_real_float();
    test_real_double();
    test_real_fixed();
    test_complex_float();
    test_complex_double();
    test_complex_fixed();
    return 0;
}
