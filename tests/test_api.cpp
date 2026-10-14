// API-level properties: every public routine is noexcept, the Level-1
// contiguous-range overloads forward to the pointer API, and fixed-point sqrt
// is usable in constant expressions.
#include "thefblas/thefblas.h"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace {

using C = std::complex<double>;

#define THEFBLAS_CHECK_NOEXCEPT(...) static_assert(noexcept(__VA_ARGS__), #__VA_ARGS__)

// Arguments for the unevaluated noexcept checks below; they are never used at
// run time.
constexpr double *p = nullptr;
constexpr C *q = nullptr;
constexpr const double *cp = nullptr;
constexpr const C *cq = nullptr;
constexpr double d = 0.0;
constexpr C w{};

void test_noexcept() {

    // Level 1
    THEFBLAS_CHECK_NOEXCEPT(thefblas::swap(1, p, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::copy(1, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::axpy(1, 1.0, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::scal(1, 1.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::scal(1, 1.0, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::dot(1, cp, 1, cp, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::dotu(1, cq, 1, cq, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::dotc(1, cq, 1, cq, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::nrm2(1, cp, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::nrm2(1, cq, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::asum(1, cp, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::asum(1, cq, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::iamax(1, cp, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::iamax(1, cq, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::rot(1, p, 1, p, 1, 1.0, 0.0));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::rot(1, q, 1, q, 1, 1.0, 0.0));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::rotg(p, p, p, p));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::rotg(q, w, p, q));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::rotm(1, p, 1, p, 1, cp));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::rotmg(p, p, p, d, p));

    // Level 2, dense
    THEFBLAS_CHECK_NOEXCEPT(thefblas::gemv('N', 1, 1, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::ger(1, 1, 1.0, cp, 1, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::geru(1, 1, w, cq, 1, cq, 1, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::gerc(1, 1, w, cq, 1, cq, 1, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::symv('U', 1, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::hemv('U', 1, w, cq, 1, cq, 1, w, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::syr('U', 1, 1.0, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::her('U', 1, 1.0, cq, 1, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::syr2('U', 1, 1.0, cp, 1, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::her2('U', 1, w, cq, 1, cq, 1, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::trmv('U', 'N', 'N', 1, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::trsv('U', 'N', 'N', 1, cp, 1, p, 1));

    // Level 2, banded
    THEFBLAS_CHECK_NOEXCEPT(thefblas::gbmv('N', 1, 1, 0, 0, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::sbmv('U', 1, 0, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::hbmv('U', 1, 0, w, cq, 1, cq, 1, w, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::tbmv('U', 'N', 'N', 1, 0, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::tbsv('U', 'N', 'N', 1, 0, cp, 1, p, 1));

    // Level 2, packed
    THEFBLAS_CHECK_NOEXCEPT(thefblas::spmv('U', 1, 1.0, cp, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::hpmv('U', 1, w, cq, cq, 1, w, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::tpmv('U', 'N', 'N', 1, cp, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::tpsv('U', 'N', 'N', 1, cp, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::spr('U', 1, 1.0, cp, 1, p));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::hpr('U', 1, 1.0, cq, 1, q));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::spr2('U', 1, 1.0, cp, 1, cp, 1, p));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::hpr2('U', 1, w, cq, 1, cq, 1, q));

    // Level 3
    THEFBLAS_CHECK_NOEXCEPT(thefblas::gemm('N', 'N', 1, 1, 1, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::syrk('U', 'N', 1, 1, 1.0, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::herk('U', 'N', 1, 1, 1.0, cq, 1, 0.0, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::trsm('L', 'U', 'N', 'N', 1, 1, 1.0, cp, 1, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::symm('L', 'U', 1, 1, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::hemm('L', 'U', 1, 1, w, cq, 1, cq, 1, w, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::syr2k('U', 'N', 1, 1, 1.0, cp, 1, cp, 1, 0.0, p, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::her2k('U', 'N', 1, 1, w, cq, 1, cq, 1, 0.0, q, 1));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::trmm('L', 'U', 'N', 'N', 1, 1, 1.0, cp, 1, p, 1));

    // Range overloads
    THEFBLAS_CHECK_NOEXCEPT(thefblas::dot(std::declval<std::vector<double> &>(),
                                          std::declval<std::vector<double> &>()));
    THEFBLAS_CHECK_NOEXCEPT(thefblas::axpy(1.0, std::declval<std::vector<double> &>(),
                                           std::declval<std::vector<double> &>()));
}

void test_range_overloads_real() {
    std::vector<double> x = {1.0, -3.0, 2.0};
    std::vector<double> y = {4.0, 5.0, 6.0};
    assert(thefblas::dot(x, y) == 1.0 * 4.0 - 3.0 * 5.0 + 2.0 * 6.0);
    assert(thefblas::asum(x) == 6.0);
    assert(thefblas::iamax(x) == 2); // 1-based, as in the pointer API
    assert(std::fabs(thefblas::nrm2(x) - std::sqrt(14.0)) < 1e-12);

    thefblas::axpy(2.0, x, y);
    assert((y == std::vector<double>{6.0, -1.0, 10.0}));
    thefblas::scal(0.5, y);
    assert((y == std::vector<double>{3.0, -0.5, 5.0}));
    thefblas::copy(x, y);
    assert(y == x);

    // Rotation by 90 degrees: (x, y) -> (y, -x).
    std::array<double, 2> a = {1.0, 2.0};
    double b[2] = {3.0, 4.0};
    thefblas::rot(a, b, 0.0, 1.0);
    assert(a[0] == 3.0 && a[1] == 4.0);
    assert(b[0] == -1.0 && b[1] == -2.0);

    // rotm with flag -1: full H = [[2, 0], [0, 3]].
    const double param[5] = {-1.0, 2.0, 0.0, 0.0, 3.0};
    thefblas::rotm(a, b, param);
    assert(a[0] == 6.0 && a[1] == 8.0);
    assert(b[0] == -3.0 && b[1] == -6.0);

    // Read-only ranges may be const.
    const std::array<float, 2> f = {3.0F, 4.0F};
    assert(thefblas::nrm2(f) == 5.0F);

    // Empty ranges are no-ops, as with n == 0.
    std::vector<double> empty;
    assert(thefblas::dot(empty, empty) == 0.0);
    assert(thefblas::iamax(empty) == 0);
}

void test_range_overloads_complex() {
    std::vector<C> x = {C(1.0, 2.0), C(3.0, -1.0)};
    std::vector<C> y = {C(2.0, 0.0), C(0.0, 1.0)};
    assert(thefblas::dotu(x, y) == x[0] * y[0] + x[1] * y[1]);
    assert(thefblas::dotc(x, y) == std::conj(x[0]) * y[0] + std::conj(x[1]) * y[1]);
    assert(thefblas::asum(x) == 7.0);
    assert(thefblas::iamax(x) == 2);
    assert(std::fabs(thefblas::nrm2(x) - std::sqrt(15.0)) < 1e-12);

    // A real alpha scales both parts; a complex alpha multiplies.
    thefblas::scal(2.0, x);
    assert(x[0] == C(2.0, 4.0));
    thefblas::scal(C(0.0, 1.0), x);
    assert(x[0] == C(-4.0, 2.0));
}

void test_range_overloads_fixed() {
    using F = thefblas::q16_16;
    const std::vector<F> x = {F(1.5), F(-2.0)};
    const std::vector<F> y = {F(2.0), F(0.25)};
    assert(thefblas::dot(x, y) == F(2.5));
    const std::vector<F> none;
    assert(thefblas::dot(none, none) == F(0.0));
}

void test_constexpr_sqrt() {
    using F = thefblas::q16_16;
    static_assert(thefblas::sqrt(F(16.0)) == F(4.0), "sqrt is constexpr");
    static_assert(thefblas::sqrt(F(0.0)) == F(0.0), "sqrt of zero");
    constexpr F two = [] {
        F v(1.0);
        v += F(1.0);
        v *= F(3.0);
        v -= F(2.0);
        v /= F(2.0);
        return v;
    }();
    static_assert(two == F(2.0), "compound assignments are constexpr");
}

} // namespace

int main() {
    test_noexcept();
    test_range_overloads_real();
    test_range_overloads_complex();
    test_range_overloads_fixed();
    test_constexpr_sqrt();
    return 0;
}
