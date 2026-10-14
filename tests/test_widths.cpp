// Every routine on 8-bit and 64-bit fixed-point storage, compared with the
// same routine evaluated in double on the same inputs.
//
// The other tests use mostly 32-bit (and some 16-bit) formats. The narrowest
// and widest storage take different paths: 8-bit values have little range,
// and 64-bit values need a 128-bit intermediate type (so that part is skipped
// on compilers without one). Running every routine on both catches
// width-specific mistakes such as a missing widening step or a wrong cast.
//
// Inputs lie on a grid that both fixed point and double represent exactly,
// with magnitudes chosen so that no result leaves the range of the format.
// The default `checked` policy asserts on any overflow, including in an
// intermediate value. Sums and products of grid values are then (nearly) exact
// in double, so the double result is the reference, and the fixed-point
// result must match it within the rounding of its own format. Each routine
// gets the bound that follows from where it rounds (see the tolerances below),
// in units in the last place (ulps) of the format: half an ulp for an exact
// reduction rounded once, a few ulps where alpha, beta or complex products
// round separately, and more for the solves, rotg and rotmg.

#include "test_support.hpp"
#include "thefblas/thefblas.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

/// Q3.4: real inputs in [-1, 1] on the 1/16 grid use every fractional bit, so
/// products must be rounded. Sizes stay small to keep every result below 8,
/// the limit of the format (syr2k is the tightest, at about 7).
struct format8 {
    using F = thefblas::fixed<std::int8_t, 4>;
    static constexpr double step = 1.0 / 16;
    static constexpr double bound = 1.0;
    static constexpr double scalar_step = 1.0 / 16;
    static constexpr int max_size = 3;
};

#if defined(__SIZEOF_INT128__)
/// Q31.32 with inputs up to 256 on a 2^-16 grid: the products use all 32
/// fractional bits, and alpha and beta on a 2^-24 grid make their products
/// round, while the results stay far below 2^31.
struct format64 {
    using F = thefblas::fixed<std::int64_t, 32>;
    static constexpr double step = 1.0 / 65536;
    static constexpr double bound = 256.0;
    static constexpr double scalar_step = 1.0 / (1 << 24);
    static constexpr int max_size = 6;
};
#endif

constexpr int trials = 150;

// Tolerances in ulps of the fixed-point format, on top of which the double
// reference may be off by a few units of 2^-53 relative.
//
// - One rounding: a reduction accumulated exactly (dot, asum, nrm2, trmv, ...)
//   or a single real product.
constexpr double ulps_once = 0.5;
// - Two rounded products added exactly (rot, rotm).
constexpr double ulps_two_products = 1.0;
// - Triangular solves round at every step; with well-conditioned matrices the
//   errors stay within a few ulps.
constexpr double ulps_solve = 4.0;
// - rotmg rounds at every step and divides by products of its inputs; for
//   the inputs drawn below the observed maximum is under 3 ulps.
constexpr double ulps_rotmg = 8.0;

// ---------------------------------------------------------------------------
// Random inputs (a fixed generator, so every platform sees the same inputs)
// ---------------------------------------------------------------------------

class generator {
  public:
    void seed(std::uint64_t value) { state_ = value; }
    std::uint32_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state_ >> 33U);
    }
    /// Uniform in [lo, hi] (the modulo bias is irrelevant here).
    int uniform(int lo, int hi) {
        const auto span = static_cast<std::uint32_t>(hi - lo) + 1U;
        return lo + static_cast<int>(next() % span);
    }

  private:
    std::uint64_t state_ = 20261014U;
};

generator rng; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

int random_inc() {
    static const int incs[] = {-3, -2, -1, 1, 2, 3};
    return incs[rng.uniform(0, 5)];
}

char random_of(const char *choices) {
    int count = 0;
    while (choices[count] != '\0') {
        ++count;
    }
    return choices[rng.uniform(0, count - 1)];
}

int random_ld(int rows) {
    return (rows > 1 ? rows : 1) + rng.uniform(0, 2);
}

int packed_size(int n) {
    return n * (n + 1) / 2;
}

std::size_t vector_length(int n, int inc) {
    return n > 0 ? 1 + static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(std::abs(inc))
                 : 1;
}

std::size_t matrix_size(int ld, int cols) {
    return static_cast<std::size_t>(ld) * static_cast<std::size_t>(cols > 0 ? cols : 1);
}

// ---------------------------------------------------------------------------
// Conversion to the double reference
// ---------------------------------------------------------------------------

template <typename F> double to_double(const F &value) {
    return value.template to_float<double>();
}

template <typename F> std::complex<double> to_double(const std::complex<F> &value) {
    return {to_double(value.real()), to_double(value.imag())};
}

template <typename T> auto to_double(const std::vector<T> &values) {
    std::vector<decltype(to_double(values[0]))> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(),
                   [](const T &v) { return to_double(v); });
    return out;
}

double component_error(double got, double ref) {
    return std::fabs(got - ref);
}

/// The larger of the two component errors; NaN if either is NaN.
double component_error(std::complex<double> got, std::complex<double> ref) {
    const double re = std::fabs(got.real() - ref.real());
    const double im = std::fabs(got.imag() - ref.imag());
    return std::isnan(re) || re > im ? re : im;
}

double magnitude(double v) {
    return std::fabs(v);
}

double magnitude(std::complex<double> v) {
    return std::fabs(v.real()) + std::fabs(v.imag());
}

// ---------------------------------------------------------------------------
// The tests, for one format and one element type (F or complex<F>)
// ---------------------------------------------------------------------------

template <typename Fmt, typename T> struct suite {
    using F = typename Fmt::F;
    using D = std::conditional_t<is_complex<T>::value, std::complex<double>, double>;
    static constexpr bool complex = is_complex<T>::value;
    /// Complex products have two terms per component, so complex inputs are
    /// half as large to keep the same headroom.
    static constexpr double bound = complex ? Fmt::bound / 2 : Fmt::bound;

    // --- inputs --------------------------------------------------------------

    static double grid(double limit) {
        const auto k = static_cast<int>(limit / Fmt::step);
        return rng.uniform(-k, k) * Fmt::step;
    }

    static T make(double re, double im) {
        if constexpr (complex) {
            return T(F(re), F(im));
        } else {
            static_cast<void>(im);
            return F(re);
        }
    }

    /// A multiple of Fmt::scalar_step in [-1, 1].
    static double grid_scalar() {
        const auto k = static_cast<int>(1.0 / Fmt::scalar_step);
        return rng.uniform(-k, k) * Fmt::scalar_step;
    }

    static T value(double limit = bound) {
        // Two statements: the order of evaluation of function arguments is
        // unspecified, and every compiler should draw the same inputs.
        const double re = grid(limit);
        return make(re, grid(limit));
    }

    /// alpha/beta: 0 and 1 often enough to reach their special cases,
    /// otherwise a multiple of Fmt::scalar_step in [-1, 1].
    template <typename S> static S scalar() {
        const int pick = rng.uniform(0, 7);
        const auto component = [pick]() {
            return pick == 0 ? 0.0 : pick == 1 ? 1.0 : grid_scalar();
        };
        if constexpr (is_complex<S>::value) {
            const double im = pick <= 1 ? 0.0 : component();
            const double re = component();
            return S(F(re), F(im));
        } else {
            return S(component());
        }
    }

    static std::vector<T> buffer(std::size_t size, double limit = bound) {
        std::vector<T> out(size > 0 ? size : 1);
        std::generate(out.begin(), out.end(), [limit]() { return value(limit); });
        return out;
    }

    /// A triangular matrix that is well conditioned: diagonal magnitudes in
    /// [1.5, 2.5], the other entries at most 1 / (n + 1), so that the solves
    /// stay within range (also with a unit diagonal) and lose only a few ulps.
    template <typename DiagIndex>
    static std::vector<T> well_conditioned(std::size_t size, int n, DiagIndex diag_index) {
        std::vector<T> a = buffer(size, 1.0 / (n + 1));
        for (int j = 0; j < n; ++j) {
            const double sign = rng.uniform(0, 1) == 0 ? 1.0 : -1.0;
            a[static_cast<std::size_t>(diag_index(j))] =
                make(sign * (1.5 + rng.uniform(0, 16) / 16.0), 0.0);
        }
        return a;
    }

    static double grid_round(double v) { return std::floor(v / Fmt::step + 0.5) * Fmt::step; }

    // --- comparison ----------------------------------------------------------

    /// A complex product rounds its two real products separately.
    static constexpr double ulps_product = complex ? 2 * ulps_once : ulps_once;

    /// `beta * y + sum of alpha_t * s_t` over `terms` exactly accumulated sums
    /// s_t: each s_t is rounded (1/2 ulp, scaled by |alpha_t| <= 1 per real
    /// part), and each scalar product rounds once per real product.
    static constexpr double ulps_scaled(int terms) {
        return complex ? 2.0 * terms + 1.0 : terms + 0.5;
    }

    static double tolerance(double ref_magnitude, double ulps) {
        const double eps = std::numeric_limits<F>::epsilon().template to_float<double>();
        // The double reference itself is exact to about 2^-53 relative.
        return ulps * eps + 4.0 * std::numeric_limits<double>::epsilon() * ref_magnitude;
    }

    template <typename G, typename R>
    static void check_close(const std::vector<G> &got, const std::vector<R> &ref, double ulps) {
        std::size_t index = 0;
        bool within_tolerance = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const auto g = to_double(got[i]);
            // Written so that a NaN error or tolerance (a non-finite double
            // reference) fails the check rather than passing it.
            if (!(component_error(g, ref[i]) <= tolerance(magnitude(ref[i]), ulps))) {
                index = i;
                within_tolerance = false;
                break;
            }
        }
        TEST_CONTEXT("element ", index, " of ", got.size(), ", tolerance ", ulps, " ulp");
        CHECK_VALUES(within_tolerance, got[index], ref[index]);
    }

    template <typename G, typename R> static void check_close(const G &got, R ref, double ulps) {
        check_close(std::vector<G>{got}, std::vector<R>{ref}, ulps);
    }

    // --- Level 1 -------------------------------------------------------------

    static void level1() {
        for (int trial = 0; trial < trials; ++trial) {
            const int n = rng.uniform(0, Fmt::max_size);
            const int incx = random_inc();
            const int incy = random_inc();
            const std::vector<T> x = buffer(vector_length(n, incx));
            const std::vector<T> y = buffer(vector_length(n, incy));
            const std::vector<D> xd = to_double(x);
            const std::vector<D> yd = to_double(y);
            const T alpha = scalar<T>();
            const D alphad = to_double(alpha);
            TEST_CONTEXT("n=", n, " incx=", incx, " incy=", incy, " alpha=", alpha);
            {
                TEST_CONTEXT("swap");
                std::vector<T> x1 = x;
                std::vector<T> y1 = y;
                thefblas::swap(n, x1.data(), incx, y1.data(), incy);
                std::vector<D> x2 = xd;
                std::vector<D> y2 = yd;
                thefblas::swap(n, x2.data(), incx, y2.data(), incy);
                check_close(x1, x2, 0.0);
                check_close(y1, y2, 0.0);
            }
            {
                TEST_CONTEXT("copy");
                std::vector<T> y1 = y;
                thefblas::copy(n, x.data(), incx, y1.data(), incy);
                std::vector<D> y2 = yd;
                thefblas::copy(n, xd.data(), incx, y2.data(), incy);
                check_close(y1, y2, 0.0);
            }
            {
                TEST_CONTEXT("axpy");
                std::vector<T> y1 = y;
                thefblas::axpy(n, alpha, x.data(), incx, y1.data(), incy);
                std::vector<D> y2 = yd;
                thefblas::axpy(n, alphad, xd.data(), incx, y2.data(), incy);
                check_close(y1, y2, ulps_product);
            }
            {
                TEST_CONTEXT("scal");
                std::vector<T> x1 = x;
                thefblas::scal(n, alpha, x1.data(), incx);
                std::vector<D> x2 = xd;
                thefblas::scal(n, alphad, x2.data(), incx);
                check_close(x1, x2, ulps_product);
            }
            if constexpr (complex) {
                {
                    const F real_alpha = scalar<F>();
                    TEST_CONTEXT("scal (real alpha ", real_alpha, ")");
                    std::vector<T> x1 = x;
                    thefblas::scal(n, real_alpha, x1.data(), incx);
                    std::vector<D> x2 = xd;
                    thefblas::scal(n, to_double(real_alpha), x2.data(), incx);
                    check_close(x1, x2, ulps_once);
                }
                TEST_CONTEXT("dotu/dotc");
                check_close(thefblas::dotu(n, x.data(), incx, y.data(), incy),
                            thefblas::dotu(n, xd.data(), incx, yd.data(), incy), ulps_once);
                check_close(thefblas::dotc(n, x.data(), incx, y.data(), incy),
                            thefblas::dotc(n, xd.data(), incx, yd.data(), incy), ulps_once);
            } else {
                TEST_CONTEXT("dot");
                const double ref = thefblas::dot(n, xd.data(), incx, yd.data(), incy);
                check_close(thefblas::dot(n, x.data(), incx, y.data(), incy), ref, ulps_once);
                if constexpr (thefblas::detail::has_extended_v<F>) {
                    // dsdot returns the sum in the wider accumulator type, with
                    // the same fractional bits; sdsdot adds sb. Both round once.
                    check_close(thefblas::dsdot(n, x.data(), incx, y.data(), incy), ref, ulps_once);
                    check_close(thefblas::sdsdot(n, alpha, x.data(), incx, y.data(), incy),
                                alphad + ref, ulps_once);
                }
            }
            {
                TEST_CONTEXT("nrm2/asum/iamax");
                check_close(thefblas::nrm2(n, x.data(), incx), thefblas::nrm2(n, xd.data(), incx),
                            ulps_once);
                check_close(thefblas::asum(n, x.data(), incx), thefblas::asum(n, xd.data(), incx),
                            ulps_once);
                CHECK_EQ(thefblas::iamax(n, x.data(), incx), thefblas::iamax(n, xd.data(), incx));
            }
            {
                // c and s on the grid; c^2 + s^2 is then only close to 1, which
                // does not matter for comparing the two evaluations.
                const double angle = rng.uniform(-314, 314) / 100.0;
                const F c(grid_round(std::cos(angle)));
                const F s(grid_round(std::sin(angle)));
                TEST_CONTEXT("rot c=", c, " s=", s);
                std::vector<T> x1 = x;
                std::vector<T> y1 = y;
                thefblas::rot(n, x1.data(), incx, y1.data(), incy, c, s);
                std::vector<D> x2 = xd;
                std::vector<D> y2 = yd;
                thefblas::rot(n, x2.data(), incx, y2.data(), incy, to_double(c), to_double(s));
                check_close(x1, x2, ulps_two_products);
                check_close(y1, y2, ulps_two_products);
            }
            if constexpr (!complex) {
                // Entries of H at most 1, so the results stay below 2 * bound.
                F param[5] = {F(rng.uniform(-2, 1)), F(grid(1.0)), F(grid(1.0)), F(grid(1.0)),
                              F(grid(1.0))};
                double paramd[5] = {};
                for (int i = 0; i < 5; ++i) {
                    paramd[i] = to_double(param[i]);
                }
                TEST_CONTEXT("rotm flag=", param[0]);
                std::vector<T> x1 = x;
                std::vector<T> y1 = y;
                thefblas::rotm(n, x1.data(), incx, y1.data(), incy, param);
                std::vector<D> x2 = xd;
                std::vector<D> y2 = yd;
                thefblas::rotm(n, x2.data(), incx, y2.data(), incy, paramd);
                check_close(x1, x2, ulps_two_products);
                check_close(y1, y2, ulps_two_products);
            }
        }
    }

    static void rotg() {
        const double max = (std::numeric_limits<F>::max)().template to_float<double>();
        for (int trial = 0; trial < trials; ++trial) {
            // Zeros select the special branches. For real types, rotg returns
            // z = 1/c when |a| <= |b|, which leaves the range of a format with
            // few integer bits when c is small (an overflow the policy
            // resolves), so such inputs are redrawn.
            T a0{};
            T b0{};
            for (;;) {
                a0 = rng.uniform(0, 5) == 0 ? T() : value();
                b0 = rng.uniform(0, 5) == 0 ? T() : value();
                const double ad = magnitude(to_double(a0));
                const double r = std::hypot(std::abs(to_double(a0)), std::abs(to_double(b0)));
                if (complex || ad == 0.0 || r / ad < max / 2) {
                    break;
                }
            }
            TEST_CONTEXT("rotg a=", a0, " b=", b0);
            T a1 = a0;
            F c1{};
            T s1{};
            D a2 = to_double(a0);
            double c2 = 0;
            D s2{};
            if constexpr (complex) {
                thefblas::rotg(&a1, b0, &c1, &s1);
                thefblas::rotg(&a2, to_double(b0), &c2, &s2);
            } else {
                T b1 = b0;
                D b2 = to_double(b0);
                thefblas::rotg(&a1, &b1, &c1, &s1);
                thefblas::rotg(&a2, &b2, &c2, &s2);
                // z is 0 when a = b = 0, otherwise s when |a| > |b|, else 1/c
                // (1 when c is zero). 1/c amplifies any error in c, so it is
                // checked against the computed c.
                const bool s_case = std::fabs(to_double(a0)) > std::fabs(to_double(b0));
                const bool zero = a0 == F() && b0 == F();
                const double z = zero        ? 0.0
                                 : s_case    ? to_double(s1)
                                 : c1 == F() ? 1.0
                                             : 1.0 / to_double(c1);
                check_close(b1, z, s_case ? 0.0 : ulps_once);
            }
            // c and s are quotients by the rounded radius, whose error of 1/2
            // ulp is relative to the divisor, so the tolerance grows as
            // 1 / divisor. Complex s also scales a / |a| (rounded) by
            // conj(b) / norm, adding up to about one ulp, and the complex
            // radius a * norm / |a| inherits the error of |a| times norm / |a|.
            const double norm = std::abs(a2);
            if (norm == 0.0) {
                check_close(a1, a2, 0.0);
                check_close(c1, c2, 0.0);
                check_close(s1, s2, 0.0);
            } else if constexpr (complex) {
                const double abs_a = std::abs(to_double(a0));
                const double divisor = abs_a == 0.0 ? std::abs(to_double(b0)) : abs_a;
                check_close(a1, a2, 1.0 + norm / divisor);
                check_close(c1, c2, 2.0 + 2.0 / divisor);
                check_close(s1, s2, 2.0 + 2.0 / divisor);
            } else {
                check_close(a1, a2, ulps_once);
                check_close(c1, c2, 1.0 + 1.0 / norm);
                check_close(s1, s2, 1.0 + 1.0 / norm);
            }
        }
    }

    static void rotmg() {
        for (int trial = 0; trial < trials; ++trial) {
            // Magnitudes in [0.5, 1.5] keep every intermediate of rotmg within
            // the range of the 8-bit format. Inputs where |d1 b1^2| and
            // |d2 b2^2| are within a factor 4 of each other are redrawn, so
            // that rounding cannot select a different branch than in double
            // and the divisor u stays within [0.75, 1.25].
            const auto mag = []() { return (8 + rng.uniform(0, 16)) / 16.0; };
            const auto sign = []() { return rng.uniform(0, 1) == 0 ? 1.0 : -1.0; };
            double d1 = 0;
            double d2 = 0;
            double b1 = 0;
            double b2 = 0;
            for (;;) {
                d1 = rng.uniform(0, 9) == 0 ? -mag() : mag();
                d2 = rng.uniform(0, 9) == 0 ? -mag() : mag();
                b1 = sign() * mag();
                b2 = rng.uniform(0, 9) == 0 ? 0.0 : sign() * mag();
                const double q1 = std::fabs(d1 * b1 * b1);
                const double q2 = std::fabs(d2 * b2 * b2);
                if (q1 > 4 * q2 || q2 > 4 * q1) {
                    break;
                }
            }
            TEST_CONTEXT("rotmg d1=", d1, " d2=", d2, " b1=", b1, " b2=", b2);
            F fd1(d1);
            F fd2(d2);
            F fb1(b1);
            F fparam[5] = {};
            thefblas::rotmg(&fd1, &fd2, &fb1, F(b2), fparam);
            double param[5] = {};
            thefblas::rotmg(&d1, &d2, &b1, b2, param);
            CHECK_EQ(to_double(fparam[0]), param[0]);
            check_close(fd1, d1, ulps_rotmg);
            check_close(fd2, d2, ulps_rotmg);
            check_close(fb1, b1, ulps_rotmg);
            // Only the entries the flag marks as meaningful are compared.
            const int flag = static_cast<int>(param[0]);
            for (int i = 1; i < 5; ++i) {
                const bool used = flag == -1 || (flag == 0 && (i == 2 || i == 3)) ||
                                  (flag == 1 && (i == 1 || i == 4));
                if (used) {
                    TEST_CONTEXT("h index ", i);
                    check_close(fparam[i], param[i], ulps_rotmg);
                }
            }
        }
    }

    // --- Level 2 -------------------------------------------------------------

    static void gemv_gbmv() {
        for (int trial = 0; trial < trials; ++trial) {
            const char trans = random_of("NTC");
            const int m = rng.uniform(0, Fmt::max_size);
            const int n = rng.uniform(0, Fmt::max_size);
            const int kl = rng.uniform(0, 2);
            const int ku = rng.uniform(0, 2);
            const int lda = random_ld(m);
            const int ldab = kl + ku + 1 + rng.uniform(0, 1);
            const int incx = random_inc();
            const int incy = random_inc();
            const int lenx = trans == 'N' ? n : m;
            const int leny = trans == 'N' ? m : n;
            const std::vector<T> a = buffer(matrix_size(lda, n));
            const std::vector<T> ab = buffer(matrix_size(ldab, n));
            const std::vector<T> x = buffer(vector_length(lenx, incx));
            const std::vector<T> y = buffer(vector_length(leny, incy));
            const T alpha = scalar<T>();
            const T beta = scalar<T>();
            const std::vector<D> ad = to_double(a);
            const std::vector<D> abd = to_double(ab);
            const std::vector<D> xd = to_double(x);
            const D alphad = to_double(alpha);
            const D betad = to_double(beta);
            TEST_CONTEXT("trans=", trans, " m=", m, " n=", n, " incx=", incx, " incy=", incy,
                         " alpha=", alpha, " beta=", beta);
            {
                TEST_CONTEXT("gemv lda=", lda);
                std::vector<T> y1 = y;
                thefblas::gemv(trans, m, n, alpha, a.data(), lda, x.data(), incx, beta, y1.data(),
                               incy);
                std::vector<D> y2 = to_double(y);
                thefblas::gemv(trans, m, n, alphad, ad.data(), lda, xd.data(), incx, betad,
                               y2.data(), incy);
                check_close(y1, y2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT("gbmv kl=", kl, " ku=", ku, " lda=", ldab);
                std::vector<T> y1 = y;
                thefblas::gbmv(trans, m, n, kl, ku, alpha, ab.data(), ldab, x.data(), incx, beta,
                               y1.data(), incy);
                std::vector<D> y2 = to_double(y);
                thefblas::gbmv(trans, m, n, kl, ku, alphad, abd.data(), ldab, xd.data(), incx,
                               betad, y2.data(), incy);
                check_close(y1, y2, ulps_scaled(1));
            }
        }
    }

    /// symv/sbmv/spmv for real types, hemv/hbmv/hpmv for complex types.
    static void symmetric_mv() {
        for (int trial = 0; trial < trials; ++trial) {
            const char uplo = random_of("UL");
            const int n = rng.uniform(0, Fmt::max_size);
            const int k = rng.uniform(0, 2);
            const int lda = random_ld(n);
            const int ldab = k + 1 + rng.uniform(0, 1);
            const int incx = random_inc();
            const int incy = random_inc();
            const std::vector<T> a = buffer(matrix_size(lda, n));
            const std::vector<T> ab = buffer(matrix_size(ldab, n));
            const std::vector<T> ap = buffer(static_cast<std::size_t>(packed_size(n)));
            const std::vector<T> x = buffer(vector_length(n, incx));
            const std::vector<T> y = buffer(vector_length(n, incy));
            const T alpha = scalar<T>();
            const T beta = scalar<T>();
            const std::vector<D> ad = to_double(a);
            const std::vector<D> abd = to_double(ab);
            const std::vector<D> apd = to_double(ap);
            const std::vector<D> xd = to_double(x);
            const D alphad = to_double(alpha);
            const D betad = to_double(beta);
            TEST_CONTEXT("uplo=", uplo, " n=", n, " incx=", incx, " incy=", incy, " alpha=", alpha,
                         " beta=", beta);
            {
                TEST_CONTEXT(complex ? "hemv" : "symv", " lda=", lda);
                std::vector<T> y1 = y;
                std::vector<D> y2 = to_double(y);
                if constexpr (complex) {
                    thefblas::hemv(uplo, n, alpha, a.data(), lda, x.data(), incx, beta, y1.data(),
                                   incy);
                    thefblas::hemv(uplo, n, alphad, ad.data(), lda, xd.data(), incx, betad,
                                   y2.data(), incy);
                } else {
                    thefblas::symv(uplo, n, alpha, a.data(), lda, x.data(), incx, beta, y1.data(),
                                   incy);
                    thefblas::symv(uplo, n, alphad, ad.data(), lda, xd.data(), incx, betad,
                                   y2.data(), incy);
                }
                check_close(y1, y2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT(complex ? "hbmv" : "sbmv", " k=", k, " lda=", ldab);
                std::vector<T> y1 = y;
                std::vector<D> y2 = to_double(y);
                if constexpr (complex) {
                    thefblas::hbmv(uplo, n, k, alpha, ab.data(), ldab, x.data(), incx, beta,
                                   y1.data(), incy);
                    thefblas::hbmv(uplo, n, k, alphad, abd.data(), ldab, xd.data(), incx, betad,
                                   y2.data(), incy);
                } else {
                    thefblas::sbmv(uplo, n, k, alpha, ab.data(), ldab, x.data(), incx, beta,
                                   y1.data(), incy);
                    thefblas::sbmv(uplo, n, k, alphad, abd.data(), ldab, xd.data(), incx, betad,
                                   y2.data(), incy);
                }
                check_close(y1, y2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT(complex ? "hpmv" : "spmv");
                std::vector<T> y1 = y;
                std::vector<D> y2 = to_double(y);
                if constexpr (complex) {
                    thefblas::hpmv(uplo, n, alpha, ap.data(), x.data(), incx, beta, y1.data(),
                                   incy);
                    thefblas::hpmv(uplo, n, alphad, apd.data(), xd.data(), incx, betad, y2.data(),
                                   incy);
                } else {
                    thefblas::spmv(uplo, n, alpha, ap.data(), x.data(), incx, beta, y1.data(),
                                   incy);
                    thefblas::spmv(uplo, n, alphad, apd.data(), xd.data(), incx, betad, y2.data(),
                                   incy);
                }
                check_close(y1, y2, ulps_scaled(1));
            }
        }
    }

    /// trmv/trsv, tbmv/tbsv and tpmv/tpsv.
    static void triangular_mv() {
        for (int trial = 0; trial < trials; ++trial) {
            const char uplo = random_of("UL");
            const char trans = random_of("NTC");
            const char diag = random_of("NU");
            const int n = rng.uniform(0, Fmt::max_size);
            const int k = rng.uniform(0, 2);
            const int lda = random_ld(n);
            const int ldab = k + 1 + rng.uniform(0, 1);
            const int incx = random_inc();
            const bool upper = uplo == 'U';
            const std::vector<T> a =
                well_conditioned(matrix_size(lda, n), n, [lda](int j) { return j + j * lda; });
            const std::vector<T> ab = well_conditioned(
                matrix_size(ldab, n), n, [=](int j) { return (upper ? k : 0) + j * ldab; });
            const std::vector<T> ap =
                well_conditioned(static_cast<std::size_t>(packed_size(n)), n, [=](int j) {
                    return upper ? j + j * (j + 1) / 2 : j * (2 * n - j + 1) / 2;
                });
            const std::vector<T> x = buffer(vector_length(n, incx));
            const std::vector<D> ad = to_double(a);
            const std::vector<D> abd = to_double(ab);
            const std::vector<D> apd = to_double(ap);
            TEST_CONTEXT("uplo=", uplo, " trans=", trans, " diag=", diag, " n=", n, " incx=", incx);

            const auto compare = [&](const char *name, double ulps, auto fixed_call,
                                     auto double_call) {
                TEST_CONTEXT(name);
                std::vector<T> x1 = x;
                fixed_call(x1.data());
                std::vector<D> x2 = to_double(x);
                double_call(x2.data());
                check_close(x1, x2, ulps);
            };
            compare(
                "trmv", ulps_once,
                [&](T *v) { thefblas::trmv(uplo, trans, diag, n, a.data(), lda, v, incx); },
                [&](D *v) { thefblas::trmv(uplo, trans, diag, n, ad.data(), lda, v, incx); });
            compare(
                "trsv", ulps_solve,
                [&](T *v) { thefblas::trsv(uplo, trans, diag, n, a.data(), lda, v, incx); },
                [&](D *v) { thefblas::trsv(uplo, trans, diag, n, ad.data(), lda, v, incx); });
            compare(
                "tbmv", ulps_once,
                [&](T *v) { thefblas::tbmv(uplo, trans, diag, n, k, ab.data(), ldab, v, incx); },
                [&](D *v) { thefblas::tbmv(uplo, trans, diag, n, k, abd.data(), ldab, v, incx); });
            compare(
                "tbsv", ulps_solve,
                [&](T *v) { thefblas::tbsv(uplo, trans, diag, n, k, ab.data(), ldab, v, incx); },
                [&](D *v) { thefblas::tbsv(uplo, trans, diag, n, k, abd.data(), ldab, v, incx); });
            compare(
                "tpmv", ulps_once,
                [&](T *v) { thefblas::tpmv(uplo, trans, diag, n, ap.data(), v, incx); },
                [&](D *v) { thefblas::tpmv(uplo, trans, diag, n, apd.data(), v, incx); });
            compare(
                "tpsv", ulps_solve,
                [&](T *v) { thefblas::tpsv(uplo, trans, diag, n, ap.data(), v, incx); },
                [&](D *v) { thefblas::tpsv(uplo, trans, diag, n, apd.data(), v, incx); });
        }
    }

    /// ger for real types, geru and gerc for complex types.
    static void ger() {
        for (int trial = 0; trial < trials; ++trial) {
            const int m = rng.uniform(0, Fmt::max_size);
            const int n = rng.uniform(0, Fmt::max_size);
            const int lda = random_ld(m);
            const int incx = random_inc();
            const int incy = random_inc();
            const std::vector<T> a = buffer(matrix_size(lda, n));
            const std::vector<T> x = buffer(vector_length(m, incx));
            const std::vector<T> y = buffer(vector_length(n, incy));
            const T alpha = scalar<T>();
            const std::vector<D> xd = to_double(x);
            const std::vector<D> yd = to_double(y);
            const D alphad = to_double(alpha);
            TEST_CONTEXT("m=", m, " n=", n, " lda=", lda, " incx=", incx, " incy=", incy,
                         " alpha=", alpha);
            std::vector<T> a1 = a;
            std::vector<D> a2 = to_double(a);
            if constexpr (complex) {
                {
                    TEST_CONTEXT("geru");
                    thefblas::geru(m, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
                    thefblas::geru(m, n, alphad, xd.data(), incx, yd.data(), incy, a2.data(), lda);
                    check_close(a1, a2, ulps_scaled(1));
                }
                TEST_CONTEXT("gerc");
                a1 = a;
                a2 = to_double(a);
                thefblas::gerc(m, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
                thefblas::gerc(m, n, alphad, xd.data(), incx, yd.data(), incy, a2.data(), lda);
                check_close(a1, a2, ulps_scaled(1));
            } else {
                TEST_CONTEXT("ger");
                thefblas::ger(m, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
                thefblas::ger(m, n, alphad, xd.data(), incx, yd.data(), incy, a2.data(), lda);
                check_close(a1, a2, ulps_scaled(1));
            }
        }
    }

    /// syr/syr2/spr/spr2 for real types, her/her2/hpr/hpr2 for complex types.
    static void symmetric_update() {
        for (int trial = 0; trial < trials; ++trial) {
            const char uplo = random_of("UL");
            const int n = rng.uniform(0, Fmt::max_size);
            const int lda = random_ld(n);
            const int incx = random_inc();
            const int incy = random_inc();
            const std::vector<T> a = buffer(matrix_size(lda, n));
            const std::vector<T> ap = buffer(static_cast<std::size_t>(packed_size(n)));
            const std::vector<T> x = buffer(vector_length(n, incx));
            const std::vector<T> y = buffer(vector_length(n, incy));
            const F real_alpha = scalar<F>();
            const T alpha = scalar<T>();
            const std::vector<D> xd = to_double(x);
            const std::vector<D> yd = to_double(y);
            const double real_alphad = to_double(real_alpha);
            const D alphad = to_double(alpha);
            TEST_CONTEXT("uplo=", uplo, " n=", n, " lda=", lda, " incx=", incx, " incy=", incy,
                         " real alpha=", real_alpha, " alpha=", alpha);
            {
                TEST_CONTEXT(complex ? "her" : "syr");
                std::vector<T> a1 = a;
                std::vector<D> a2 = to_double(a);
                if constexpr (complex) {
                    thefblas::her(uplo, n, real_alpha, x.data(), incx, a1.data(), lda);
                    thefblas::her(uplo, n, real_alphad, xd.data(), incx, a2.data(), lda);
                } else {
                    thefblas::syr(uplo, n, real_alpha, x.data(), incx, a1.data(), lda);
                    thefblas::syr(uplo, n, real_alphad, xd.data(), incx, a2.data(), lda);
                }
                check_close(a1, a2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT(complex ? "her2" : "syr2");
                std::vector<T> a1 = a;
                std::vector<D> a2 = to_double(a);
                if constexpr (complex) {
                    thefblas::her2(uplo, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
                    thefblas::her2(uplo, n, alphad, xd.data(), incx, yd.data(), incy, a2.data(),
                                   lda);
                } else {
                    thefblas::syr2(uplo, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
                    thefblas::syr2(uplo, n, alphad, xd.data(), incx, yd.data(), incy, a2.data(),
                                   lda);
                }
                check_close(a1, a2, ulps_scaled(complex ? 2 : 1));
            }
            {
                TEST_CONTEXT(complex ? "hpr" : "spr");
                std::vector<T> ap1 = ap;
                std::vector<D> ap2 = to_double(ap);
                if constexpr (complex) {
                    thefblas::hpr(uplo, n, real_alpha, x.data(), incx, ap1.data());
                    thefblas::hpr(uplo, n, real_alphad, xd.data(), incx, ap2.data());
                } else {
                    thefblas::spr(uplo, n, real_alpha, x.data(), incx, ap1.data());
                    thefblas::spr(uplo, n, real_alphad, xd.data(), incx, ap2.data());
                }
                check_close(ap1, ap2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT(complex ? "hpr2" : "spr2");
                std::vector<T> ap1 = ap;
                std::vector<D> ap2 = to_double(ap);
                if constexpr (complex) {
                    thefblas::hpr2(uplo, n, alpha, x.data(), incx, y.data(), incy, ap1.data());
                    thefblas::hpr2(uplo, n, alphad, xd.data(), incx, yd.data(), incy, ap2.data());
                } else {
                    thefblas::spr2(uplo, n, alpha, x.data(), incx, y.data(), incy, ap1.data());
                    thefblas::spr2(uplo, n, alphad, xd.data(), incx, yd.data(), incy, ap2.data());
                }
                check_close(ap1, ap2, ulps_scaled(complex ? 2 : 1));
            }
        }
    }

    // --- Level 3 -------------------------------------------------------------

    static void gemm() {
        for (int trial = 0; trial < trials; ++trial) {
            const char transa = random_of("NTC");
            const char transb = random_of("NTC");
            const int m = rng.uniform(0, Fmt::max_size);
            const int n = rng.uniform(0, Fmt::max_size);
            const int k = rng.uniform(0, Fmt::max_size);
            const int lda = random_ld(transa == 'N' ? m : k);
            const int ldb = random_ld(transb == 'N' ? k : n);
            const int ldc = random_ld(m);
            const std::vector<T> a = buffer(matrix_size(lda, transa == 'N' ? k : m));
            const std::vector<T> b = buffer(matrix_size(ldb, transb == 'N' ? n : k));
            const std::vector<T> c = buffer(matrix_size(ldc, n));
            const T alpha = scalar<T>();
            const T beta = scalar<T>();
            TEST_CONTEXT("gemm transa=", transa, " transb=", transb, " m=", m, " n=", n, " k=", k,
                         " alpha=", alpha, " beta=", beta);
            std::vector<T> c1 = c;
            thefblas::gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta,
                           c1.data(), ldc);
            std::vector<D> c2 = to_double(c);
            thefblas::gemm(transa, transb, m, n, k, to_double(alpha), to_double(a).data(), lda,
                           to_double(b).data(), ldb, to_double(beta), c2.data(), ldc);
            check_close(c1, c2, ulps_scaled(1));
        }
    }

    /// symm for every type, and hemm for complex types.
    static void symm() {
        for (int trial = 0; trial < trials; ++trial) {
            const char side = random_of("LR");
            const char uplo = random_of("UL");
            const int m = rng.uniform(0, Fmt::max_size);
            const int n = rng.uniform(0, Fmt::max_size);
            const int dim = side == 'L' ? m : n;
            const int lda = random_ld(dim);
            const int ldb = random_ld(m);
            const int ldc = random_ld(m);
            const std::vector<T> a = buffer(matrix_size(lda, dim));
            const std::vector<T> b = buffer(matrix_size(ldb, n));
            const std::vector<T> c = buffer(matrix_size(ldc, n));
            const T alpha = scalar<T>();
            const T beta = scalar<T>();
            const std::vector<D> ad = to_double(a);
            const std::vector<D> bd = to_double(b);
            const D alphad = to_double(alpha);
            const D betad = to_double(beta);
            TEST_CONTEXT("side=", side, " uplo=", uplo, " m=", m, " n=", n, " alpha=", alpha,
                         " beta=", beta);
            {
                TEST_CONTEXT("symm");
                std::vector<T> c1 = c;
                thefblas::symm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta,
                               c1.data(), ldc);
                std::vector<D> c2 = to_double(c);
                thefblas::symm(side, uplo, m, n, alphad, ad.data(), lda, bd.data(), ldb, betad,
                               c2.data(), ldc);
                check_close(c1, c2, ulps_scaled(1));
            }
            if constexpr (complex) {
                TEST_CONTEXT("hemm");
                std::vector<T> c1 = c;
                thefblas::hemm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta,
                               c1.data(), ldc);
                std::vector<D> c2 = to_double(c);
                thefblas::hemm(side, uplo, m, n, alphad, ad.data(), lda, bd.data(), ldb, betad,
                               c2.data(), ldc);
                check_close(c1, c2, ulps_scaled(1));
            }
        }
    }

    /// syrk and syr2k for every type, herk and her2k for complex types.
    static void rank_k() {
        for (int trial = 0; trial < trials; ++trial) {
            const char uplo = random_of("UL");
            const int n = rng.uniform(0, Fmt::max_size);
            const int k = rng.uniform(0, Fmt::max_size);
            const int ldc = random_ld(n);
            const std::vector<T> c = buffer(matrix_size(ldc, n));
            const T alpha = scalar<T>();
            const T beta = scalar<T>();
            const F real_alpha = scalar<F>();
            const F real_beta = scalar<F>();
            TEST_CONTEXT("uplo=", uplo, " n=", n, " k=", k, " alpha=", alpha, " beta=", beta);

            // Complex syrk/syr2k have no 'C' form; real ones treat 'C' as 'T'.
            const char trans = complex ? random_of("NT") : random_of("NTC");
            const int lda = random_ld(trans == 'N' ? n : k);
            const std::vector<T> a = buffer(matrix_size(lda, trans == 'N' ? k : n));
            const std::vector<T> b = buffer(matrix_size(lda, trans == 'N' ? k : n));
            const std::vector<D> ad = to_double(a);
            const std::vector<D> bd = to_double(b);
            {
                TEST_CONTEXT("syrk trans=", trans);
                std::vector<T> c1 = c;
                thefblas::syrk(uplo, trans, n, k, alpha, a.data(), lda, beta, c1.data(), ldc);
                std::vector<D> c2 = to_double(c);
                thefblas::syrk(uplo, trans, n, k, to_double(alpha), ad.data(), lda, to_double(beta),
                               c2.data(), ldc);
                check_close(c1, c2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT("syr2k trans=", trans);
                std::vector<T> c1 = c;
                thefblas::syr2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), lda, beta,
                                c1.data(), ldc);
                std::vector<D> c2 = to_double(c);
                thefblas::syr2k(uplo, trans, n, k, to_double(alpha), ad.data(), lda, bd.data(), lda,
                                to_double(beta), c2.data(), ldc);
                check_close(c1, c2, ulps_scaled(2));
            }
            if constexpr (complex) {
                const char htrans = random_of("NC");
                const int ldh = random_ld(htrans == 'N' ? n : k);
                const std::vector<T> ha = buffer(matrix_size(ldh, htrans == 'N' ? k : n));
                const std::vector<T> hb = buffer(matrix_size(ldh, htrans == 'N' ? k : n));
                const std::vector<D> had = to_double(ha);
                const std::vector<D> hbd = to_double(hb);
                {
                    TEST_CONTEXT("herk trans=", htrans, " alpha=", real_alpha, " beta=", real_beta);
                    std::vector<T> c1 = c;
                    thefblas::herk(uplo, htrans, n, k, real_alpha, ha.data(), ldh, real_beta,
                                   c1.data(), ldc);
                    std::vector<D> c2 = to_double(c);
                    thefblas::herk(uplo, htrans, n, k, to_double(real_alpha), had.data(), ldh,
                                   to_double(real_beta), c2.data(), ldc);
                    check_close(c1, c2, ulps_scaled(1));
                }
                TEST_CONTEXT("her2k trans=", htrans, " beta=", real_beta);
                std::vector<T> c1 = c;
                thefblas::her2k(uplo, htrans, n, k, alpha, ha.data(), ldh, hb.data(), ldh,
                                real_beta, c1.data(), ldc);
                std::vector<D> c2 = to_double(c);
                thefblas::her2k(uplo, htrans, n, k, to_double(alpha), had.data(), ldh, hbd.data(),
                                ldh, to_double(real_beta), c2.data(), ldc);
                check_close(c1, c2, ulps_scaled(2));
            }
        }
    }

    /// trmm with a general triangle, trsm with a well-conditioned one.
    static void triangular_mm() {
        for (int trial = 0; trial < trials; ++trial) {
            const char side = random_of("LR");
            const char uplo = random_of("UL");
            const char trans = random_of("NTC");
            const char diag = random_of("NU");
            const int m = rng.uniform(0, Fmt::max_size);
            const int n = rng.uniform(0, Fmt::max_size);
            const int dim = side == 'L' ? m : n;
            const int lda = random_ld(dim);
            const int ldb = random_ld(m);
            const std::vector<T> a = buffer(matrix_size(lda, dim));
            const std::vector<T> as =
                well_conditioned(matrix_size(lda, dim), dim, [lda](int j) { return j + j * lda; });
            const std::vector<T> b = buffer(matrix_size(ldb, n));
            const T alpha = scalar<T>();
            const D alphad = to_double(alpha);
            TEST_CONTEXT("side=", side, " uplo=", uplo, " trans=", trans, " diag=", diag, " m=", m,
                         " n=", n, " alpha=", alpha);
            {
                TEST_CONTEXT("trmm");
                std::vector<T> b1 = b;
                thefblas::trmm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b1.data(), ldb);
                std::vector<D> b2 = to_double(b);
                thefblas::trmm(side, uplo, trans, diag, m, n, alphad, to_double(a).data(), lda,
                               b2.data(), ldb);
                check_close(b1, b2, ulps_scaled(1));
            }
            {
                TEST_CONTEXT("trsm");
                std::vector<T> b1 = b;
                thefblas::trsm(side, uplo, trans, diag, m, n, alpha, as.data(), lda, b1.data(),
                               ldb);
                std::vector<D> b2 = to_double(b);
                thefblas::trsm(side, uplo, trans, diag, m, n, alphad, to_double(as).data(), lda,
                               b2.data(), ldb);
                check_close(b1, b2, ulps_solve);
            }
        }
    }

    static void run(const char *name) {
        TEST_CONTEXT(name);
        level1();
        rotg();
        if constexpr (!complex) {
            rotmg();
        }
        gemv_gbmv();
        symmetric_mv();
        triangular_mv();
        ger();
        symmetric_update();
        gemm();
        symm();
        rank_k();
        triangular_mm();
    }
};

} // namespace

int main() {
    // THEFBLAS_TEST_SEED selects other inputs, e.g. to reproduce a failure.
    if (const char *env = std::getenv("THEFBLAS_TEST_SEED")) {
        rng.seed(std::strtoull(env, nullptr, 10));
    }
    suite<format8, format8::F>::run("fixed<int8_t, 4>");
    suite<format8, std::complex<format8::F>>::run("complex<fixed<int8_t, 4>>");
#if defined(__SIZEOF_INT128__)
    suite<format64, format64::F>::run("fixed<int64_t, 32>");
    suite<format64, std::complex<format64::F>>::run("complex<fixed<int64_t, 32>>");
#endif
    return thefblas_test::finish();
}
