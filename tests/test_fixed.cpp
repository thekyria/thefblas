// NOLINTNEXTLINE(portability-avoid-pragma-once)
#include "test_support.hpp"
#include "thefblas/fixed.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

using thefblas::fixed;

namespace {

template <typename IntType, int FracBits>
bool almost_equal(fixed<IntType, FracBits> a, fixed<IntType, FracBits> b, double eps = 1e-3) {
    return std::fabs(a.template to_float<double>() - b.template to_float<double>()) < eps;
}

void test_construction_and_conversion() {
    using F = fixed<std::int32_t, 24>;
    F a(1.5);
    CHECK_PRED(almost_equal, a, F(1.5));
    F b(-2);
    CHECK_PRED(almost_equal, b, F(-2.0));
    F zero;
    CHECK(zero.raw() == 0);
}

void test_arithmetic() {
    using F = fixed<std::int32_t, 20>;
    F a(3.25);
    F b(1.5);
    CHECK_PRED(almost_equal, a + b, F(4.75));
    CHECK_PRED(almost_equal, a - b, F(1.75));
    CHECK_PRED(almost_equal, a * b, F(4.875));
    CHECK_PRED(almost_equal, a / b, F(3.25 / 1.5));
    F c = a;
    c += b;
    CHECK_PRED(almost_equal, c, F(4.75));
    c -= b;
    CHECK_PRED(almost_equal, c, a);
    c *= b;
    CHECK_PRED(almost_equal, c, F(3.25 * 1.5));
}

void test_comparisons() {
    using F = fixed<std::int16_t, 12>;
    F a(1.0);
    F b(2.0);
    CHECK(a < b);
    CHECK(b > a);
    CHECK(a <= a);
    CHECK(a >= a);
    CHECK(a == F(1.0));
    CHECK(a != b);
}

void test_negation_and_abs() {
    using F = fixed<std::int32_t, 16>;
    F a(4.5);
    F neg = -a;
    CHECK_PRED(almost_equal, neg, F(-4.5));
    CHECK_PRED(almost_equal, thefblas::abs(neg), a);
    CHECK_PRED(almost_equal, thefblas::abs(a), a);
}

void test_sqrt() {
    using F = fixed<std::int32_t, 20>;
    CHECK_PRED(almost_equal, thefblas::sqrt(F(16.0)), F(4.0));
    CHECK_PRED(almost_equal, thefblas::sqrt(F(2.0)), F(1.41421356), 1e-2);
    CHECK(thefblas::sqrt(F(0.0)) == F(0.0));

    // No fractional bits: 0.5 is not representable, so halving must not rely on it.
    using I = fixed<std::int32_t, 0>;
    CHECK(thefblas::sqrt(I(16.0)) == I(4.0));
    CHECK(thefblas::sqrt(I(1.0)) == I(1.0));
    CHECK(thefblas::sqrt(I(10000.0)) == I(100.0));
}

// sqrt(raw * 2^-F) on the grid is the integer nearest to sqrt(raw * 2^F): with
// root = result.raw(), 4 * raw * 2^F must lie in [(2 root - 1)^2, (2 root + 1)^2).
template <typename F> bool is_rounded_root(F value, F root) {
    const std::uint64_t radicand = static_cast<std::uint64_t>(value.raw()) << F::frac_bits;
    const auto r = static_cast<std::uint64_t>(root.raw());
    const std::uint64_t below = (2 * r - 1) * (2 * r - 1);
    const std::uint64_t above = (2 * r + 1) * (2 * r + 1);
    return r > 0 && 4 * radicand >= below && 4 * radicand < above;
}

// Formats without integer bits: the root of a value in (0, 1) is in (0, 1) and
// representable, but 1 (which a Newton step `f / x` with x = f reaches) is not.
// Every positive value of the 8- and 16-bit formats, under every policy.
template <typename Policy> void check_sqrt_without_integer_bits() {
    using Q7 = fixed<std::int8_t, 7, Policy>;
    for (int raw = 1; raw <= 127; ++raw) {
        const Q7 value = Q7::from_raw(static_cast<std::int8_t>(raw));
        CHECK_MSG(is_rounded_root(value, thefblas::sqrt(value)), "q7 raw ", raw);
    }
    using Q15 = fixed<std::int16_t, 15, Policy>;
    for (int raw = 1; raw <= 32767; ++raw) {
        const Q15 value = Q15::from_raw(static_cast<std::int16_t>(raw));
        CHECK_MSG(is_rounded_root(value, thefblas::sqrt(value)), "q15 raw ", raw);
    }
    using Q31 = fixed<std::int32_t, 31, Policy>;
    CHECK(thefblas::sqrt(Q31(0.25)) == Q31(0.5));
    const Q31 value(0.81);
    CHECK(std::fabs(thefblas::sqrt(value).to_float() - std::sqrt(value.to_float())) <=
          std::ldexp(1.0, -31));
    // The largest value: its root rounds to the largest value, not past it.
    const Q31 top = Q31::from_raw((std::numeric_limits<std::int32_t>::max)());
    CHECK(thefblas::sqrt(top) == top);
}

void test_sqrt_without_integer_bits() {
    check_sqrt_without_integer_bits<thefblas::checked>();
    check_sqrt_without_integer_bits<thefblas::wrap>();
    check_sqrt_without_integer_bits<thefblas::saturate>();
    static_assert(thefblas::sqrt(thefblas::q15::from_raw(8192)) == thefblas::q15::from_raw(16384),
                  "sqrt(0.25) == 0.5 in Q1.15, in a constant expression");
}

void test_different_widths() {
    fixed<std::int8_t, 4> small(2.0);
    CHECK_PRED(almost_equal, small + small, fixed<std::int8_t, 4>(4.0));

#if defined(__SIZEOF_INT128__)
    // fixed<int64_t> requires a 128-bit intermediate type; the configuration
    // is rejected at compile time on platforms without one.
    fixed<std::int64_t, 32> big(123.456);
    CHECK(std::fabs(big.to_float<double>() - 123.456) < 1e-6);
#endif
}

// ---------------------------------------------------------------------------
// Overflow policy matrix
// ---------------------------------------------------------------------------

template <typename F> constexpr F raw(typename F::rep v) {
    return F::from_raw(v);
}

void test_wrap_policy() {
    // Q1.7: raw range [-128, 127], one unit = 1/128.
    using F = fixed<std::int8_t, 7, thefblas::wrap>;
    constexpr std::int8_t lo = -128;
    constexpr std::int8_t hi = 127;

    // Addition past the top wraps around to the bottom.
    CHECK((raw<F>(hi) + raw<F>(1)).raw() == lo);
    CHECK((raw<F>(hi) + raw<F>(hi)).raw() == -2);
    // Subtraction past the bottom wraps around to the top.
    CHECK((raw<F>(lo) - raw<F>(1)).raw() == hi);
    // Negating the most negative value is its own negation.
    CHECK((-raw<F>(lo)).raw() == lo);
    CHECK(thefblas::abs(raw<F>(lo)).raw() == lo);
    // Division overflow wraps: 65/128 divided by 1/128 is 65, whose scaled raw
    // value 8320 reduces to -128 in eight bits.
    CHECK((raw<F>(65) / raw<F>(1)).raw() == lo);
    // Division by zero maps to the limit of the dividend's sign, without asserting.
    CHECK((raw<F>(3) / raw<F>(0)).raw() == hi);
    CHECK((raw<F>(-3) / raw<F>(0)).raw() == lo);
    // Out-of-range construction is clamped under every policy.
    CHECK(F(10.0).raw() == hi);
    CHECK(F(-10.0).raw() == lo);
}

void test_saturate_policy() {
    using F = fixed<std::int8_t, 7, thefblas::saturate>;
    constexpr std::int8_t lo = -128;
    constexpr std::int8_t hi = 127;

    CHECK((raw<F>(hi) + raw<F>(1)).raw() == hi);
    CHECK((raw<F>(hi) + raw<F>(hi)).raw() == hi);
    CHECK((raw<F>(lo) - raw<F>(1)).raw() == lo);
    CHECK((-raw<F>(lo)).raw() == hi);
    CHECK(thefblas::abs(raw<F>(lo)).raw() == hi);
    CHECK((raw<F>(64) / raw<F>(1)).raw() == hi);
    CHECK((raw<F>(-64) / raw<F>(1)).raw() == lo);
    CHECK((raw<F>(3) / raw<F>(0)).raw() == hi);
    CHECK((raw<F>(-3) / raw<F>(0)).raw() == lo);
    CHECK(F(10.0).raw() == hi);
    CHECK(F(-10.0).raw() == lo);

    // In-range results are unaffected by the policy.
    CHECK((raw<F>(40) + raw<F>(40)).raw() == 80);

    // Saturating arithmetic is not associative: clamping happens per operation.
    const F a = raw<F>(100);
    const F b = raw<F>(100);
    const F c = raw<F>(-100);
    CHECK(((a + b) + c).raw() == 27);  // clamped to 127 first
    CHECK((a + (b + c)).raw() == 100); // no intermediate overflow
}

void test_checked_policy_in_range() {
    // The default policy asserts on overflow, so only in-range behaviour can be
    // exercised portably here; the defined release fallback is wrap.
    using F = fixed<std::int8_t, 7>;
    static_assert(std::is_same<typename F::policy, thefblas::checked>::value,
                  "checked must be the default overflow policy");
    CHECK((raw<F>(60) + raw<F>(60)).raw() == 120);
    CHECK((raw<F>(-60) - raw<F>(60)).raw() == -120);
    CHECK(thefblas::abs(raw<F>(-120)).raw() == 120);
}

void test_nan_construction() {
    using F = fixed<std::int32_t, 16>;
    const double nan = std::nan("");
    CHECK(F(nan).raw() == 0);
}

void test_rounding() {
    // Round to nearest, ties away from zero, symmetric about zero.
    using F = fixed<std::int32_t, 2>;
    CHECK((F(0.75) * F(0.75)).raw() == 2);   // 0.5625 -> 0.5
    CHECK((F(-0.75) * F(0.75)).raw() == -2); // symmetric
    CHECK((F(0.25) * F(0.25)).raw() == 0);
    CHECK((F(-0.25) * F(-0.25)).raw() == 0);
}

void test_policy_aliases() {
    static_assert(std::is_same<thefblas::q15, fixed<std::int16_t, 15>>::value, "q15");
    static_assert(
        std::is_same<thefblas::q15_sat, fixed<std::int16_t, 15, thefblas::saturate>>::value,
        "q15_sat");
}

// Conversion from double rounds to nearest, ties away from zero, exactly once.
void test_float_conversion_rounding() {
    using I = fixed<std::int32_t, 0>;
    // The largest double below 0.5: adding 0.5 before truncating rounds it up.
    CHECK(I(0.49999999999999994).raw() == 0);
    CHECK(I(-0.49999999999999994).raw() == 0);
    CHECK(I(0.5).raw() == 1);
    CHECK(I(-0.5).raw() == -1);
    CHECK(I(2.5).raw() == 3);
    CHECK(I(-2.5).raw() == -3);
    CHECK(I(1.4999999999999998).raw() == 1);

    // Values that round onto the ends of the range are in range...
    using B = fixed<std::int8_t, 0>;
    CHECK(B(-128.3).raw() == -128);
    CHECK(B(127.4).raw() == 127);
    // ...and those that round past them are clamped under every policy.
    using BSat = fixed<std::int8_t, 0, thefblas::saturate>;
    using BWrap = fixed<std::int8_t, 0, thefblas::wrap>;
    CHECK(BSat(127.5).raw() == 127);
    CHECK(BSat(-128.5).raw() == -128);
    CHECK(BWrap(127.5).raw() == 127);
    CHECK(BWrap(-128.5).raw() == -128);
    CHECK(BWrap(1e300).raw() == 127);

    // Fractional formats: half a unit rounds away from zero.
    using Q = fixed<std::int16_t, 4>;
    CHECK(Q(0.03125).raw() == 1);
    CHECK(Q(-0.03125).raw() == -1);
    CHECK(Q(0.031249999999999997).raw() == 0);

#if defined(__SIZEOF_INT128__)
    // 64-bit limits: no out-of-range floating-point-to-integer cast.
    using L = fixed<std::int64_t, 0, thefblas::saturate>;
    CHECK(L(-9223372036854775808.0).raw() == (std::numeric_limits<std::int64_t>::min)());
    CHECK(L(9223372036854775808.0).raw() == (std::numeric_limits<std::int64_t>::max)());
    CHECK(L(-1e19).raw() == (std::numeric_limits<std::int64_t>::min)());
#endif
}

void test_q31_conversion_and_sqrt() {
    using F = thefblas::q31_sat;
    const F quarter(0.25);
    CHECK(std::fabs(quarter.to_float<double>() - 0.25) < 1e-9);
    CHECK(std::fabs(thefblas::sqrt(quarter).to_float<double>() - 0.5) < 1e-4);
}

// ---------------------------------------------------------------------------
// Conversions between Q formats and policies
// ---------------------------------------------------------------------------

void test_q_format_conversion() {
    using thefblas::fixed_cast;
    using thefblas::q15;
    using thefblas::q15_sat;
    using thefblas::q16_16;
    using thefblas::q31;

    static_assert(std::is_constructible<q31, q15>::value, "explicit conversion");
    static_assert(!std::is_convertible<q15, q31>::value, "no implicit conversion");
    static_assert(fixed_cast<q31>(q15::from_raw(1)).raw() == (1 << 16), "usable in constexpr");

    // Adding fractional bits, or only changing the policy, is exact.
    const q15 half(0.5);
    CHECK(q31(half).raw() == (1 << 30));
    CHECK(q16_16(half).raw() == (1 << 15));
    using Q16_15 = fixed<std::int32_t, 15>;
    CHECK(Q16_15(half).raw() == half.raw());
    CHECK(fixed_cast<q15_sat>(half).raw() == half.raw());
    CHECK(fixed_cast<q15>(q31(half)) == half);

    // Dropping fractional bits rounds to nearest, ties away from zero.
    using Q8 = fixed<std::int16_t, 8>;
    CHECK(Q8(raw<q16_16>(384)).raw() == 2);   // 1.5 units -> 2
    CHECK(Q8(raw<q16_16>(-384)).raw() == -2); // symmetric
    CHECK(Q8(raw<q16_16>(383)).raw() == 1);
    CHECK(Q8(raw<q16_16>(128)).raw() == 1); // exactly half a unit
    CHECK(Q8(raw<q16_16>(-128)).raw() == -1);
    CHECK(Q8(raw<q16_16>(127)).raw() == 0);
    CHECK(fixed_cast<q15>(raw<q31>(1 << 15)).raw() == 1);
    using Int = fixed<std::int32_t, 0>;
    CHECK(fixed_cast<Int>(q16_16(2.5)).raw() == 3);
    CHECK(fixed_cast<Int>(q16_16(-2.5)).raw() == -3);

    // Out of range for the target: resolved by the target's policy.
    const q16_16 big(200.0); // Q8.8 holds [-128, 128)
    using Q8_sat = fixed<std::int16_t, 8, thefblas::saturate>;
    using Q8_wrap = fixed<std::int16_t, 8, thefblas::wrap>;
    CHECK(fixed_cast<Q8_sat>(big).raw() == 32767);
    CHECK(fixed_cast<Q8_sat>(-big).raw() == -32768);
    CHECK(fixed_cast<Q8_wrap>(big).raw() == 51200 - 65536);
    // The largest Q1.31 value rounds up to 1, which Q1.15 cannot hold.
    CHECK(fixed_cast<q15_sat>((std::numeric_limits<q31>::max)()).raw() == 32767);
    CHECK(fixed_cast<q15_sat>(std::numeric_limits<q31>::lowest()).raw() == -32768);

#if defined(__SIZEOF_INT128__)
    using Q32_32 = fixed<std::int64_t, 32>;
    CHECK(Q32_32(half).raw() == (std::int64_t(1) << 31));
    CHECK(q15(Q32_32(0.25)).raw() == (1 << 13));
    CHECK(fixed_cast<Q32_32>((std::numeric_limits<q31>::max)()).raw() ==
          (std::int64_t(0x7fffffff) << 1));
#endif
}

// ---------------------------------------------------------------------------
// std::numeric_limits
// ---------------------------------------------------------------------------

void test_numeric_limits() {
    using thefblas::q15;
    using L = std::numeric_limits<q15>;
    static_assert(L::is_specialized, "specialised");
    static_assert(L::is_signed, "signed");
    static_assert(L::is_bounded, "bounded");
    static_assert(L::is_exact, "exact");
    static_assert(!L::is_integer, "not an integer type");
    static_assert(!L::is_iec559, "not IEC 559");
    // Assertions are enabled in this test, so checked overflow aborts.
    static_assert(L::traps, "checked traps while assertions are enabled");
    static_assert(!std::numeric_limits<thefblas::q15_sat>::traps, "saturate never traps");
    static_assert(!std::numeric_limits<fixed<std::int16_t, 15, thefblas::wrap>>::traps,
                  "wrap never traps");
    static_assert(!L::has_infinity, "no infinity");
    static_assert(!L::has_quiet_NaN, "no NaN");
    static_assert(!L::has_signaling_NaN, "no NaN");
    static_assert(L::digits == 15, "digits of the storage type");
    static_assert(L::radix == 2, "binary");
    static_assert(L::round_style == std::round_to_nearest, "round to nearest");
    static_assert(!L::is_modulo, "checked is not modular");
    static_assert(!std::numeric_limits<thefblas::q15_sat>::is_modulo, "saturate is not modular");
    static_assert(std::numeric_limits<fixed<std::int16_t, 15, thefblas::wrap>>::is_modulo,
                  "wrap is modular");
    static_assert((L::max)().raw() == 32767, "usable in constexpr");

    // Floating-point convention: min() is the smallest positive value.
    CHECK((L::min)().raw() == 1);
    CHECK(L::lowest().raw() == -32768);
    CHECK((L::max)().raw() == 32767);
    CHECK(L::epsilon().raw() == 1);
    CHECK(L::round_error() == q15(0.5));
    CHECK(L::denorm_min() == (L::min)());
    CHECK(L::infinity().raw() == 0);
    CHECK(L::quiet_NaN().raw() == 0);
    CHECK(L::signaling_NaN().raw() == 0);

    using Q16 = std::numeric_limits<thefblas::q16_16>;
    CHECK(Q16::lowest().to_float<double>() == -32768.0);
    CHECK(Q16::epsilon().to_float<double>() == 1.0 / 65536.0);
    CHECK(Q16::round_error().to_float<double>() == 0.5);

    // No fractional bits: 0.5 is not representable, so round_error() is one unit.
    using I = std::numeric_limits<fixed<std::int32_t, 0>>;
    CHECK(I::round_error().raw() == 1);
    CHECK((I::min)().raw() == 1);
}

} // namespace

int main() {
    test_construction_and_conversion();
    test_arithmetic();
    test_comparisons();
    test_negation_and_abs();
    test_sqrt();
    test_sqrt_without_integer_bits();
    test_different_widths();
    test_wrap_policy();
    test_saturate_policy();
    test_checked_policy_in_range();
    test_nan_construction();
    test_rounding();
    test_policy_aliases();
    test_float_conversion_rounding();
    test_q31_conversion_and_sqrt();
    test_q_format_conversion();
    test_numeric_limits();
    return thefblas_test::finish();
}
