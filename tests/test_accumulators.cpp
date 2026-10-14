// NOLINTNEXTLINE(portability-avoid-pragma-once)
#include "test_support.hpp"
#include "thefblas/thefblas.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using thefblas::fixed;

namespace {

// Q1.15: range [-1, 1), one unit = 2^-15.
using Q15 = thefblas::q15;
// Q9.7 in 16 bits: range [-256, 256), one unit = 2^-7.
using Q7_8 = fixed<std::int16_t, 7>;

bool close(double a, double b, double eps) {
    return std::fabs(a - b) < eps;
}

// Example A: the dot product of [0.9, 0.9] with itself is 1.62, which is not
// representable in Q1.15 at all. Without a wide accumulator the running sum
// overflows the 16-bit format after the second product and, under the checked
// policy, would trip an assertion; with the accumulator the sum is formed
// exactly in 32 bits and only the final narrowing is subject to the policy.
void test_dot_wide_accumulator() {
    using F = fixed<std::int16_t, 15, thefblas::saturate>;
    const F x[2] = {F(0.9), F(0.9)};
    const F result = thefblas::dot(2, x, 1, x, 1);
    // 1.62 saturates to the largest representable value, not to a wrapped
    // negative number, and not to the doubly-wrapped value a naive loop gives.
    CHECK(result.raw() == 32767);
}

// The same reduction in a format that can hold the result must be exact to
// within one quantisation step, with no per-product rounding loss.
void test_dot_exactness() {
    const Q7_8 x[4] = {Q7_8(0.9), Q7_8(0.9), Q7_8(0.9), Q7_8(0.9)};
    const Q7_8 result = thefblas::dot(4, x, 1, x, 1);
    CHECK_PRED(close, result.to_float<double>(), 4.0 * 0.9 * 0.9, 0.02);
}

// Example C: a sum of squares can exceed the range of the element type even
// when the norm itself does not. nrm2 must therefore take the square root in
// the accumulator type. Here the squares sum to 1.62 but the norm is 1.27,
// which is still outside Q1.15, so check the in-range Q9.7 case instead.
void test_nrm2_wide_accumulator() {
    const Q7_8 x[2] = {Q7_8(0.9), Q7_8(0.9)};
    const Q7_8 result = thefblas::nrm2(2, x, 1);
    CHECK_PRED(close, result.to_float<double>(), std::sqrt(1.62), 0.05);

    // In Q1.15 the sum of squares (1.62) is unrepresentable but the norm
    // (1.2728) is too, so saturation at the very end is the correct answer.
    using F = fixed<std::int16_t, 15, thefblas::saturate>;
    const F y[2] = {F(0.9), F(0.9)};
    CHECK(thefblas::nrm2(2, y, 1).raw() == 32767);

    // Whereas a vector whose squares overflow but whose norm does not still
    // gives the right answer: 0.8^2 + 0.5^2 = 0.89, norm 0.9434.
    const F z[2] = {F(0.8), F(0.5)};
    CHECK_PRED(close, thefblas::nrm2(2, z, 1).to_float<double>(), std::sqrt(0.89), 0.01);
}

// asum accumulates absolute values, which overflow just as easily.
void test_asum_wide_accumulator() {
    const Q7_8 x[4] = {Q7_8(0.9), Q7_8(-0.9), Q7_8(0.9), Q7_8(-0.9)};
    CHECK_PRED(close, thefblas::asum(4, x, 1).to_float<double>(), 3.6, 0.02);
}

// Level 2 reductions use the same accumulator. A 4x4 symmetric product whose
// intermediate row sums leave the element range must still come back correct
// when the final result is in range.
void test_gemv_transpose_accumulator() {
    const int n = 4;
    Q7_8 a[16];
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a[i + j * n] = Q7_8(0.9);
        }
    }
    const Q7_8 x[4] = {Q7_8(0.9), Q7_8(0.9), Q7_8(0.9), Q7_8(0.9)};
    Q7_8 y[4] = {Q7_8(0.0), Q7_8(0.0), Q7_8(0.0), Q7_8(0.0)};
    thefblas::gemv('T', n, n, Q7_8(1.0), a, n, x, 1, Q7_8(0.0), y, 1);
    for (int i = 0; i < n; ++i) {
        CHECK_PRED(close, y[i].to_float<double>(), 4.0 * 0.81, 0.05);
    }
}

// Floating-point results must be untouched by the accumulator machinery: the
// accumulator type for float is float, so the reductions stay bit-identical to
// the straightforward loop that Netlib BLAS specifies.
void test_float_bit_identical() {
    static_assert(std::is_same<thefblas::detail::accumulator_t<float>, float>::value,
                  "float must accumulate in float");
    static_assert(std::is_same<thefblas::detail::accumulator_t<double>, double>::value,
                  "double must accumulate in double");

    const float x[5] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
    const float y[5] = {1.5F, 2.5F, 3.5F, 4.5F, 5.5F};
    float expected = 0.0F;
    for (int i = 0; i < 5; ++i) {
        expected += x[i] * y[i];
    }
    CHECK(thefblas::dot(5, x, 1, y, 1) == expected);

    float sum = 0.0F;
    for (int i = 0; i < 5; ++i) {
        sum += x[i] * x[i];
    }
    CHECK(thefblas::nrm2(5, x, 1) == std::sqrt(sum));

    float abs_sum = 0.0F;
    for (int i = 0; i < 5; ++i) {
        abs_sum += std::fabs(x[i]);
    }
    CHECK(thefblas::asum(5, x, 1) == abs_sum);
}

// Complex reductions build on two real accumulators.
void test_complex_accumulator() {
    using C = std::complex<Q7_8>;
    const C x[2] = {C(Q7_8(0.9), Q7_8(0.9)), C(Q7_8(0.9), Q7_8(0.9))};
    const C d = thefblas::dotc(2, x, 1, x, 1);
    // conj(x) . x = 2 * (0.81 + 0.81) = 3.24, purely real.
    CHECK_PRED(close, d.real().to_float<double>(), 3.24, 0.05);
    CHECK_PRED(close, d.imag().to_float<double>(), 0.0, 0.02);
}

// The absolute value of the most negative raw value is not representable in
// the element type, so asum must widen before taking magnitudes.
void test_asum_raw_min() {
    using F = fixed<std::int16_t, 8, thefblas::saturate>;
    const F x[2] = {F::from_raw(-32768), F::from_raw(-32768)};
    // |min| + |min| = 256.0 exceeds the Q7.8 range and saturates only at the
    // final narrowing rather than wrapping per element.
    CHECK(thefblas::asum(2, x, 1).raw() == 32767);
}

// Constructing from an integer whose scaled value exceeds even the widened
// intermediate range must resolve through the policy, not overflow.
void test_from_integer_overflow() {
    using S = fixed<std::int8_t, 4, thefblas::saturate>;
    CHECK(S(1000).raw() == 127);
    CHECK(S(-1000).raw() == -128);
    using W = fixed<std::int8_t, 4, thefblas::wrap>;
    CHECK(W(8).raw() == 127); // out-of-range sources clamp under every policy.
}

// symv sums whole output rows in the accumulator, so partial sums that leave
// the element range must not saturate before later terms cancel them.
void test_symv_cancellation() {
    using F = fixed<std::int16_t, 15, thefblas::saturate>; // Q1.15
    const int n = 3;
    // Upper-triangle column-major storage of row 0 = [0.9, 0.9, -0.9].
    const F a[9] = {F(0.9), F(0.0), F(0.0), F(0.9), F(0.0), F(0.0), F(-0.9), F(0.0), F(0.0)};
    const F x[3] = {F(0.9), F(0.9), F(0.9)};
    F y[3] = {F(0.0), F(0.0), F(0.0)};
    thefblas::symv('U', n, F(0.5), a, n, x, 1, F(0.0), y, 1);
    // y[0] = 0.5 * (0.81 + 0.81 - 0.81) = 0.405; the intermediate 1.62 must
    // not saturate the accumulation.
    CHECK_PRED(close, y[0].to_float<double>(), 0.405, 0.01);
}

// The Hermitian, banded and packed symmetric/Hermitian variants reduce whole
// output rows in the accumulator as well.
void test_symmetric_variants_cancellation() {
    using F = fixed<std::int16_t, 15, thefblas::saturate>; // Q1.15
    using C = std::complex<F>;
    const int n = 3;
    const F zero(0.0);
    const F x[3] = {F(0.9), F(0.9), F(0.9)};

    // Row 0 of the matrix is [0.9, 0.9, -0.9]; the partial sum 1.62 saturates
    // unless the whole row is reduced before narrowing.
    const F ap[6] = {F(0.9), F(0.9), zero, F(-0.9), zero, zero};
    F y_packed[3] = {zero, zero, zero};
    thefblas::spmv('U', n, F(0.5), ap, x, 1, zero, y_packed, 1);
    CHECK_PRED(close, y_packed[0].to_float<double>(), 0.405, 0.01);

    const F ab[9] = {zero, zero, F(0.9), zero, F(0.9), zero, F(-0.9), zero, zero};
    F y_banded[3] = {zero, zero, zero};
    thefblas::sbmv('U', n, 2, F(0.5), ab, 3, x, 1, zero, y_banded, 1);
    CHECK_PRED(close, y_banded[0].to_float<double>(), 0.405, 0.01);

    const C cx[3] = {C(F(0.9), zero), C(F(0.9), zero), C(F(0.9), zero)};
    const C ca[9] = {C(F(0.9), zero),  C(zero, zero), C(zero, zero),
                     C(F(0.9), zero),  C(zero, zero), C(zero, zero),
                     C(F(-0.9), zero), C(zero, zero), C(zero, zero)};
    C cy[3] = {C(zero, zero), C(zero, zero), C(zero, zero)};
    thefblas::hemv('U', n, C(F(0.5), zero), ca, n, cx, 1, C(zero, zero), cy, 1);
    CHECK_PRED(close, cy[0].real().to_float<double>(), 0.405, 0.01);

    const C cap[6] = {C(F(0.9), zero),  C(F(0.9), zero), C(zero, zero),
                      C(F(-0.9), zero), C(zero, zero),   C(zero, zero)};
    C cy_packed[3] = {C(zero, zero), C(zero, zero), C(zero, zero)};
    thefblas::hpmv('U', n, C(F(0.5), zero), cap, cx, 1, C(zero, zero), cy_packed, 1);
    CHECK_PRED(close, cy_packed[0].real().to_float<double>(), 0.405, 0.01);
}

// The beta-scaled output element is added to alpha * (row sum) in the
// accumulator, so a scaled row sum outside the element range can still be
// cancelled by the existing value of y before the single narrowing.
void test_output_cancellation() {
    using F = fixed<std::int16_t, 15, thefblas::saturate>; // Q1.15
    const F a[2] = {F(0.9), F(0.9)};
    const F x[2] = {F(0.9), F(0.9)};
    const F almost_one = F::from_raw(32767);
    F y[1] = {F(-0.9)};
    // y = 1.62 * alpha + (-0.9) * beta ~= 0.72; narrowing 1.62 first would
    // saturate it to ~1.0 and give ~0.1 instead.
    thefblas::gemv('T', 2, 1, almost_one, a, 2, x, 1, almost_one, y, 1);
    CHECK_PRED(close, y[0].to_float<double>(), 0.72, 0.01);
}

// beta * y is formed in the accumulator together with alpha * (row sum), not
// narrowed on its own first: in Q7.8 (range [-128, 128)) beta * y = 160 must
// not saturate before alpha * sum = -100 brings the result back to 60.
void test_beta_scaling_in_accumulator() {
    using F = fixed<std::int16_t, 8, thefblas::saturate>; // Q7.8
    using C = std::complex<F>;
    const F one(1.0);
    const F beta(16.0);
    const F a[1] = {one};
    const F x[1] = {F(-100.0)};
    const auto check = [](F value) { CHECK_PRED(close, value.to_float<double>(), 60.0, 0.01); };

    F y[1] = {F(10.0)};
    thefblas::gemv('T', 1, 1, one, a, 1, x, 1, beta, y, 1);
    check(y[0]);
    y[0] = F(10.0);
    thefblas::gbmv('T', 1, 1, 0, 0, one, a, 1, x, 1, beta, y, 1);
    check(y[0]);
    y[0] = F(10.0);
    thefblas::symv('U', 1, one, a, 1, x, 1, beta, y, 1);
    check(y[0]);
    y[0] = F(10.0);
    thefblas::sbmv('L', 1, 0, one, a, 1, x, 1, beta, y, 1);
    check(y[0]);
    y[0] = F(10.0);
    thefblas::spmv('U', 1, one, a, x, 1, beta, y, 1);
    check(y[0]);

    const F zero(0.0);
    const C cone(one, zero);
    const C cbeta(beta, zero);
    const C ca[1] = {cone};
    const C cx[1] = {C(F(-100.0), zero)};
    C cy[1] = {C(F(10.0), zero)};
    thefblas::hemv('U', 1, cone, ca, 1, cx, 1, cbeta, cy, 1);
    check(cy[0].real());
    cy[0] = C(F(10.0), zero);
    thefblas::hbmv('L', 1, 0, cone, ca, 1, cx, 1, cbeta, cy, 1);
    check(cy[0].real());
    cy[0] = C(F(10.0), zero);
    thefblas::hpmv('U', 1, cone, ca, cx, 1, cbeta, cy, 1);
    check(cy[0].real());
}

// Reference y := alpha * A * x + beta * y for a dense symmetric (or Hermitian)
// matrix, written in the Netlib BLAS column-update order of SSYMV/CHEMV.
template <bool Hermitian, typename T>
void netlib_symmetric_reference(bool upper, int n, T alpha, const T *a, const T *x, T beta, T *y) {
    const auto diag_product = [](const T &temp1, const T &ajj) -> T {
        if constexpr (Hermitian) {
            return temp1 * ajj.real();
        } else {
            return temp1 * ajj;
        }
    };
    const auto off_product = [](const T &aij, const T &xi) -> T {
        if constexpr (Hermitian) {
            return std::conj(aij) * xi;
        } else {
            return aij * xi;
        }
    };
    for (int i = 0; i < n; ++i) {
        y[i] *= beta;
    }
    for (int j = 0; j < n; ++j) {
        const T temp1 = alpha * x[j];
        T temp2 = T{};
        if (upper) {
            for (int i = 0; i < j; ++i) {
                y[i] += temp1 * a[i + j * n];
                temp2 += off_product(a[i + j * n], x[i]);
            }
            y[j] = y[j] + diag_product(temp1, a[j + j * n]) + alpha * temp2;
        } else {
            y[j] += diag_product(temp1, a[j + j * n]);
            for (int i = j + 1; i < n; ++i) {
                y[i] += temp1 * a[i + j * n];
                temp2 += off_product(a[i + j * n], x[i]);
            }
            y[j] += alpha * temp2;
        }
    }
}

// Floating-point symmetric/Hermitian products, in dense, banded (full
// bandwidth) and packed storage, must reproduce the Netlib update order bit for
// bit rather than the row-gathering order used for fixed point.
template <bool Hermitian, typename T> void check_symmetric_netlib_order(T alpha, T beta) {
    constexpr int n = 5;
    T a[n * n];
    T x[n];
    T y0[n];
    for (int j = 0; j < n; ++j) {
        x[j] = T(0.1F + 0.37F * static_cast<float>(j));
        y0[j] = T(1.0F / static_cast<float>(j + 3));
        for (int i = 0; i < n; ++i) {
            a[i + j * n] =
                T(0.3F / static_cast<float>(i + 2 * j + 1) - 0.07F * static_cast<float>(i));
        }
    }
    if constexpr (Hermitian) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                a[i + j * n] += T(0.0F, 0.11F * static_cast<float>(i - j) + 0.03F);
            }
        }
    }

    for (const char uplo : {'U', 'L'}) {
        const bool upper = (uplo == 'U');
        T expected[n];
        T y[n];
        for (int i = 0; i < n; ++i) {
            expected[i] = y0[i];
        }
        netlib_symmetric_reference<Hermitian>(upper, n, alpha, a, x, beta, expected);
        const auto check = [&expected](const T *result) {
            for (int i = 0; i < n; ++i) {
                CHECK(result[i] == expected[i]);
            }
        };

        T ab[n * n];
        T ap[n * (n + 1) / 2];
        int p = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (upper ? i <= j : i >= j) {
                    ab[(upper ? (n - 1 + i - j) : (i - j)) + j * n] = a[i + j * n];
                    ap[p++] = a[i + j * n];
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            y[i] = y0[i];
        }
        if constexpr (Hermitian) {
            thefblas::hemv(uplo, n, alpha, a, n, x, 1, beta, y, 1);
        } else {
            thefblas::symv(uplo, n, alpha, a, n, x, 1, beta, y, 1);
        }
        check(y);

        for (int i = 0; i < n; ++i) {
            y[i] = y0[i];
        }
        if constexpr (Hermitian) {
            thefblas::hbmv(uplo, n, n - 1, alpha, ab, n, x, 1, beta, y, 1);
        } else {
            thefblas::sbmv(uplo, n, n - 1, alpha, ab, n, x, 1, beta, y, 1);
        }
        check(y);

        for (int i = 0; i < n; ++i) {
            y[i] = y0[i];
        }
        if constexpr (Hermitian) {
            thefblas::hpmv(uplo, n, alpha, ap, x, 1, beta, y, 1);
        } else {
            thefblas::spmv(uplo, n, alpha, ap, x, 1, beta, y, 1);
        }
        check(y);
    }
}

void test_float_symmetric_netlib_order() {
    check_symmetric_netlib_order<false>(0.7F, 0.3F);
    check_symmetric_netlib_order<true>(std::complex<float>(0.7F, -0.2F),
                                       std::complex<float>(0.3F, 0.1F));
}

// The no-transpose triangular multiplies reduce each output row in the
// accumulator too, so the partial sum 1.62 of row [0.9, 0.9, -0.9] (upper) or
// [-0.9, 0.9, 0.9] (lower) with x = 0.9 must not saturate before the last term
// cancels it.
void test_triangular_no_transpose_cancellation() {
    using F = fixed<std::int16_t, 15, thefblas::saturate>; // Q1.15
    const int n = 3;
    const F z(0.0);
    const F p(0.9);
    const F m(-0.9);
    const F h(0.5);
    const double expected_upper[3] = {0.81, 0.45, 0.45};
    const double expected_lower[3] = {0.45, 0.45, 0.81};
    const auto check = [](const F *x, const double *expected) {
        for (int i = 0; i < 3; ++i) {
            CHECK_PRED(close, x[i].to_float<double>(), expected[i], 0.01);
        }
    };

    const F a_upper[9] = {p, z, z, p, h, z, m, z, h};
    F x[3] = {p, p, p};
    thefblas::trmv('U', 'N', 'N', n, a_upper, n, x, 1);
    check(x, expected_upper);

    const F a_lower[9] = {h, z, m, z, h, p, z, z, p};
    F x_lower[3] = {p, p, p};
    thefblas::trmv('L', 'N', 'N', n, a_lower, n, x_lower, 1);
    check(x_lower, expected_lower);

    const F ab_upper[9] = {z, z, p, z, p, h, m, z, h};
    F xb[3] = {p, p, p};
    thefblas::tbmv('U', 'N', 'N', n, 2, ab_upper, 3, xb, 1);
    check(xb, expected_upper);

    const F ab_lower[9] = {h, z, m, h, p, z, p, z, z};
    F xb_lower[3] = {p, p, p};
    thefblas::tbmv('L', 'N', 'N', n, 2, ab_lower, 3, xb_lower, 1);
    check(xb_lower, expected_lower);

    const F ap_upper[6] = {p, p, h, m, z, h};
    F xp[3] = {p, p, p};
    thefblas::tpmv('U', 'N', 'N', n, ap_upper, xp, 1);
    check(xp, expected_upper);

    const F ap_lower[6] = {h, z, m, h, p, p};
    F xp_lower[3] = {p, p, p};
    thefblas::tpmv('L', 'N', 'N', n, ap_lower, xp_lower, 1);
    check(xp_lower, expected_lower);
}

// The final rounding of a wrapped running sum must be well defined even at the
// signed limits of the accumulator, where negation or `value + half` would
// overflow.
void test_round_shift_limits() {
    using thefblas::detail::round_shift;
    constexpr std::int64_t lo = (std::numeric_limits<std::int64_t>::min)();
    constexpr std::int64_t hi = (std::numeric_limits<std::int64_t>::max)();
    static_assert(round_shift(lo, 1) == lo / 2, "min / 2 is exact");
    static_assert(round_shift(hi, 1) == hi / 2 + 1, "max / 2 rounds away from zero");
    static_assert(round_shift(lo, 31) == -(std::int64_t(1) << 32), "min / 2^31");
    static_assert(round_shift(hi, 31) == (std::int64_t(1) << 32), "max / 2^31 rounds up");
    static_assert(round_shift(std::int64_t(-3), 1) == -2, "ties away from zero");
    static_assert(round_shift(std::int64_t(3), 1) == 2, "ties away from zero");
    static_assert(round_shift(std::int64_t(-5), 2) == -1, "-1.25 rounds to -1");
    static_assert(round_shift(std::int64_t(6), 2) == 2, "1.5 rounds to 2");
}

// The sum of squares behind nrm2 and rotg is exact while it fits its unsigned
// accumulator and is reported as out of range once it does not, instead of
// wrapping back to a small in-range norm.
void test_norm_accumulator_overflow() {
    using S8 = fixed<std::int8_t, 4, thefblas::saturate>;
    const auto max8 = (std::numeric_limits<S8>::max)().raw();
    // 127^2 * 266289 just exceeds 2^32, the range of the uint32 sum; wrapped,
    // it would leave 7985, a norm of about 5.6. One element fewer still fits.
    const std::vector<S8> x(266289, S8::from_raw(max8));
    CHECK_EQ(thefblas::nrm2(static_cast<int>(x.size()), x.data(), 1).raw(), max8);
    CHECK_EQ(thefblas::nrm2(static_cast<int>(x.size()) - 1, x.data(), 1).raw(), max8);
#if defined(__SIZEOF_INT128__)
    using S64 = fixed<std::int64_t, 32, thefblas::saturate>;
    const auto max64 = (std::numeric_limits<S64>::max)().raw();
    // Five squares of 2^63 - 1 exceed 2^128; with signed wrapping, two did.
    const S64 y[5] = {S64::from_raw(max64), S64::from_raw(max64), S64::from_raw(max64),
                      S64::from_raw(max64), S64::from_raw(max64)};
    CHECK_EQ(thefblas::nrm2(5, y, 1).raw(), max64);
    CHECK_EQ(thefblas::nrm2(2, y, 1).raw(), max64);
    std::complex<S64> a(y[0], y[0]);
    S64 c{};
    std::complex<S64> s{};
    thefblas::rotg(&a, std::complex<S64>(y[0], y[0]), &c, &s);
    CHECK_EQ(a.real().raw(), max64);
    CHECK_EQ(a.imag().raw(), max64);
#endif
}

// Formats without integer bits (q15 covers [-1, 1)) cannot represent 1, so
// beta == 1 must be recognised without constructing 1, and the rank updates
// must not depend on it.
void test_formats_without_unity() {
    using thefblas::q15;
    using C = std::complex<q15>;
    const q15 half(0.5);
    const q15 a[1] = {half};
    const q15 x[1] = {half};
    for (const char trans : {'N', 'T', 'C'}) {
        q15 y[1] = {q15(0.25)};
        thefblas::gemv(trans, 1, 1, half, a, 1, x, 1, q15(0), y, 1);
        CHECK_EQ(y[0].raw(), q15(0.125).raw());
        thefblas::gemv(trans, 1, 1, half, a, 1, x, 1, half, y, 1);
        CHECK_EQ(y[0].raw(), q15(0.1875).raw());
        // alpha == 0 scales y by beta alone.
        thefblas::gemv(trans, 1, 1, q15(0), a, 1, x, 1, half, y, 1);
        CHECK_EQ(y[0].raw(), q15(0.09375).raw());
        y[0] = q15(0.25);
        thefblas::gbmv(trans, 1, 1, 0, 0, half, a, 1, x, 1, q15(0), y, 1);
        CHECK_EQ(y[0].raw(), q15(0.125).raw());
        thefblas::gbmv(trans, 1, 1, 0, 0, q15(0), a, 1, x, 1, half, y, 1);
        CHECK_EQ(y[0].raw(), q15(0.0625).raw());
    }
    q15 r[1] = {q15(0)};
    thefblas::ger(1, 1, half, x, 1, x, 1, r, 1);
    CHECK_EQ(r[0].raw(), q15(0.125).raw());
    r[0] = q15(0);
    thefblas::syr('U', 1, half, x, 1, r, 1);
    CHECK_EQ(r[0].raw(), q15(0.125).raw());
    r[0] = q15(0);
    thefblas::syr2('U', 1, half, x, 1, x, 1, r, 1);
    CHECK_EQ(r[0].raw(), q15(0.25).raw());
    const C cx[1] = {C(half, q15(0))};
    C h[1] = {C(q15(0), q15(0))};
    thefblas::her2('U', 1, C(half, q15(0)), cx, 1, cx, 1, h, 1);
    CHECK_EQ(h[0].real().raw(), q15(0.25).raw());
    CHECK_EQ(h[0].imag().raw(), 0);
}

// Empty reductions return zero for the 16-bit formats used above.
void test_empty_reductions() {
    const Q7_8 q[1] = {Q7_8(0.5)};
    CHECK_EQ(thefblas::dot(0, q, 1, q, 1).raw(), 0);
    using S15 = fixed<std::int16_t, 15, thefblas::saturate>;
    const S15 s15[1] = {S15(0.5)};
    CHECK_EQ(thefblas::dot(0, s15, 1, s15, 1).raw(), 0);
    using S8 = fixed<std::int16_t, 8, thefblas::saturate>;
    const S8 s8[1] = {S8(0.5)};
    CHECK_EQ(thefblas::dot(0, s8, 1, s8, 1).raw(), 0);
}

// her2/hpr2 scale their second term by conj(alpha). With alpha = -8i, the most
// negative imaginary part of Q3.4, conj(alpha) is not representable, but
// a(0,1) = conj(alpha) * y(0) * conj(x(1)) = 8i * 1.5 * 0.5 = 6i is.
template <typename F> void check_hermitian_rank2_min_alpha() {
    using C = std::complex<F>;
    const C alpha(F(0), F::from_raw((std::numeric_limits<typename F::rep>::min)()));
    const C x[2] = {C(F(0), F(0)), C(F(0.5), F(0))};
    const C y[2] = {C(F(1.5), F(0)), C(F(0), F(0))};
    C a[4] = {};
    thefblas::her2('U', 2, alpha, x, 1, y, 1, a, 2);
    CHECK_EQ(a[2].real().raw(), 0);
    CHECK_EQ(a[2].imag().raw(), F(6.0).raw());
    C ap[3] = {};
    thefblas::hpr2('U', 2, alpha, x, 1, y, 1, ap);
    CHECK_EQ(ap[1].real().raw(), 0);
    CHECK_EQ(ap[1].imag().raw(), F(6.0).raw());
}

void test_hermitian_rank2_min_alpha() {
    check_hermitian_rank2_min_alpha<fixed<std::int8_t, 4>>();
    check_hermitian_rank2_min_alpha<fixed<std::int8_t, 4, thefblas::wrap>>();
    check_hermitian_rank2_min_alpha<fixed<std::int8_t, 4, thefblas::saturate>>();
}

// A tall banded matrix has empty rows below its band. Their x offset must not
// be formed: with incx = INT_MAX, row 2 would compute 2 * INT_MAX (signed
// overflow, reported by the sanitizer build). beta still applies to them.
void test_gbmv_empty_band_rows() {
    using F = fixed<std::int32_t, 20>;
    const F a[1] = {F(2.0)};
    const F x[1] = {F(3.0)};
    F y[3] = {F(1.0), F(2.0), F(3.0)};
    thefblas::gbmv('N', 3, 1, 0, 0, F(1.0), a, 1, x, (std::numeric_limits<int>::max)(), F(0.5), y,
                   1);
    CHECK_EQ(y[0].raw(), F(6.5).raw());
    CHECK_EQ(y[1].raw(), F(1.0).raw());
    CHECK_EQ(y[2].raw(), F(1.5).raw());
}

#if defined(__SIZEOF_INT128__)
// 64-bit fixed-point reductions accumulate in the same-width Q format, since
// `__int128` has no wider type of its own; the sums are still formed exactly
// in 128 bits by the multiply-accumulate.
void test_int64_reductions() {
    using F = fixed<std::int64_t, 32>;
    static_assert(std::is_same<thefblas::detail::accumulator_t<F>, F>::value,
                  "64-bit fixed accumulates in the same width");
    static_assert(std::is_same<thefblas::detail::accumulator_t<fixed<std::int32_t, 16>>,
                               fixed<std::int64_t, 16>>::value,
                  "32-bit fixed accumulates in 64 bits");
    const F x[2] = {F(3.0), F(4.0)};
    CHECK_PRED(close, thefblas::dot(2, x, 1, x, 1).to_float<double>(), 25.0, 1e-6);
    CHECK_PRED(close, thefblas::nrm2(2, x, 1).to_float<double>(), 5.0, 1e-6);
    CHECK_PRED(close, thefblas::asum(2, x, 1).to_float<double>(), 7.0, 1e-6);

    F y[1] = {F(1.0)};
    thefblas::gemv('T', 2, 1, F(2.0), x, 2, x, 1, F(1.0), y, 1);
    CHECK_PRED(close, y[0].to_float<double>(), 51.0, 1e-6);
}
#endif

} // namespace

int main() {
    test_dot_wide_accumulator();
    test_dot_exactness();
    test_nrm2_wide_accumulator();
    test_asum_wide_accumulator();
    test_gemv_transpose_accumulator();
    test_float_bit_identical();
    test_complex_accumulator();
    test_asum_raw_min();
    test_from_integer_overflow();
    test_round_shift_limits();
    test_norm_accumulator_overflow();
    test_formats_without_unity();
    test_hermitian_rank2_min_alpha();
    test_gbmv_empty_band_rows();
    test_empty_reductions();
    test_symv_cancellation();
    test_symmetric_variants_cancellation();
    test_output_cancellation();
    test_beta_scaling_in_accumulator();
    test_float_symmetric_netlib_order();
    test_triangular_no_transpose_cancellation();
#if defined(__SIZEOF_INT128__)
    test_int64_reductions();
#endif
    return thefblas_test::finish();
}
