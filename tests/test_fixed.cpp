// NOLINTNEXTLINE(portability-avoid-pragma-once)
#include "thefblas/fixed.hpp"

#include <cassert>
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
    assert(almost_equal(a, F(1.5)));
    F b(-2);
    assert(almost_equal(b, F(-2.0)));
    F zero;
    assert(zero.raw() == 0);
}

void test_arithmetic() {
    using F = fixed<std::int32_t, 20>;
    F a(3.25);
    F b(1.5);
    assert(almost_equal(a + b, F(4.75)));
    assert(almost_equal(a - b, F(1.75)));
    assert(almost_equal(a * b, F(4.875)));
    assert(almost_equal(a / b, F(3.25 / 1.5)));
    F c = a;
    c += b;
    assert(almost_equal(c, F(4.75)));
    c -= b;
    assert(almost_equal(c, a));
    c *= b;
    assert(almost_equal(c, F(3.25 * 1.5)));
}

void test_comparisons() {
    using F = fixed<std::int16_t, 12>;
    F a(1.0);
    F b(2.0);
    assert(a < b);
    assert(b > a);
    assert(a <= a);
    assert(a >= a);
    assert(a == F(1.0));
    assert(a != b);
}

void test_negation_and_abs() {
    using F = fixed<std::int32_t, 16>;
    F a(4.5);
    F neg = -a;
    assert(almost_equal(neg, F(-4.5)));
    assert(almost_equal(thefblas::abs(neg), a));
    assert(almost_equal(thefblas::abs(a), a));
}

void test_sqrt() {
    using F = fixed<std::int32_t, 20>;
    assert(almost_equal(thefblas::sqrt(F(16.0)), F(4.0)));
    assert(almost_equal(thefblas::sqrt(F(2.0)), F(1.41421356), 1e-2));
    assert(thefblas::sqrt(F(0.0)) == F(0.0));

    // No fractional bits: 0.5 is not representable, so halving must not rely on it.
    using I = fixed<std::int32_t, 0>;
    assert(thefblas::sqrt(I(16.0)) == I(4.0));
    assert(thefblas::sqrt(I(1.0)) == I(1.0));
    assert(thefblas::sqrt(I(10000.0)) == I(100.0));
}

void test_different_widths() {
    fixed<std::int8_t, 4> small(2.0);
    assert(almost_equal(small + small, fixed<std::int8_t, 4>(4.0)));

#if defined(__SIZEOF_INT128__)
    // fixed<int64_t> requires a 128-bit intermediate type; the configuration
    // is rejected at compile time on platforms without one.
    fixed<std::int64_t, 32> big(123.456);
    assert(std::fabs(big.to_float<double>() - 123.456) < 1e-6);
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
    assert((raw<F>(hi) + raw<F>(1)).raw() == lo);
    assert((raw<F>(hi) + raw<F>(hi)).raw() == -2);
    // Subtraction past the bottom wraps around to the top.
    assert((raw<F>(lo) - raw<F>(1)).raw() == hi);
    // Negating the most negative value is its own negation.
    assert((-raw<F>(lo)).raw() == lo);
    assert(thefblas::abs(raw<F>(lo)).raw() == lo);
    // Division overflow wraps: 65/128 divided by 1/128 is 65, whose scaled raw
    // value 8320 reduces to -128 in eight bits.
    assert((raw<F>(65) / raw<F>(1)).raw() == lo);
    // Division by zero maps to the limit of the dividend's sign, without asserting.
    assert((raw<F>(3) / raw<F>(0)).raw() == hi);
    assert((raw<F>(-3) / raw<F>(0)).raw() == lo);
    // Out-of-range construction is clamped under every policy.
    assert(F(10.0).raw() == hi);
    assert(F(-10.0).raw() == lo);
}

void test_saturate_policy() {
    using F = fixed<std::int8_t, 7, thefblas::saturate>;
    constexpr std::int8_t lo = -128;
    constexpr std::int8_t hi = 127;

    assert((raw<F>(hi) + raw<F>(1)).raw() == hi);
    assert((raw<F>(hi) + raw<F>(hi)).raw() == hi);
    assert((raw<F>(lo) - raw<F>(1)).raw() == lo);
    assert((-raw<F>(lo)).raw() == hi);
    assert(thefblas::abs(raw<F>(lo)).raw() == hi);
    assert((raw<F>(64) / raw<F>(1)).raw() == hi);
    assert((raw<F>(-64) / raw<F>(1)).raw() == lo);
    assert((raw<F>(3) / raw<F>(0)).raw() == hi);
    assert((raw<F>(-3) / raw<F>(0)).raw() == lo);
    assert(F(10.0).raw() == hi);
    assert(F(-10.0).raw() == lo);

    // In-range results are unaffected by the policy.
    assert((raw<F>(40) + raw<F>(40)).raw() == 80);

    // Saturating arithmetic is not associative: clamping happens per operation.
    const F a = raw<F>(100);
    const F b = raw<F>(100);
    const F c = raw<F>(-100);
    assert(((a + b) + c).raw() == 27);  // clamped to 127 first
    assert((a + (b + c)).raw() == 100); // no intermediate overflow
}

void test_checked_policy_in_range() {
    // The default policy asserts on overflow, so only in-range behaviour can be
    // exercised portably here; the defined release fallback is wrap.
    using F = fixed<std::int8_t, 7>;
    static_assert(std::is_same<typename F::policy, thefblas::checked>::value,
                  "checked must be the default overflow policy");
    assert((raw<F>(60) + raw<F>(60)).raw() == 120);
    assert((raw<F>(-60) - raw<F>(60)).raw() == -120);
    assert(thefblas::abs(raw<F>(-120)).raw() == 120);
}

void test_nan_construction() {
    using F = fixed<std::int32_t, 16>;
    const double nan = std::nan("");
    assert(F(nan).raw() == 0);
}

void test_rounding() {
    // Round to nearest, ties away from zero, symmetric about zero.
    using F = fixed<std::int32_t, 2>;
    assert((F(0.75) * F(0.75)).raw() == 2);   // 0.5625 -> 0.5
    assert((F(-0.75) * F(0.75)).raw() == -2); // symmetric
    assert((F(0.25) * F(0.25)).raw() == 0);
    assert((F(-0.25) * F(-0.25)).raw() == 0);
}

void test_policy_aliases() {
    static_assert(std::is_same<thefblas::q15, fixed<std::int16_t, 15>>::value, "q15");
    static_assert(
        std::is_same<thefblas::q15_sat, fixed<std::int16_t, 15, thefblas::saturate>>::value,
        "q15_sat");
}

void test_q31_conversion_and_sqrt() {
    using F = thefblas::q31_sat;
    const F quarter(0.25);
    assert(std::fabs(quarter.to_float<double>() - 0.25) < 1e-9);
    assert(std::fabs(thefblas::sqrt(quarter).to_float<double>() - 0.5) < 1e-4);
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
    assert(q31(half).raw() == (1 << 30));
    assert(q16_16(half).raw() == (1 << 15));
    using Q16_15 = fixed<std::int32_t, 15>;
    assert(Q16_15(half).raw() == half.raw());
    assert(fixed_cast<q15_sat>(half).raw() == half.raw());
    assert(fixed_cast<q15>(q31(half)) == half);

    // Dropping fractional bits rounds to nearest, ties away from zero.
    using Q8 = fixed<std::int16_t, 8>;
    assert(Q8(raw<q16_16>(384)).raw() == 2);   // 1.5 units -> 2
    assert(Q8(raw<q16_16>(-384)).raw() == -2); // symmetric
    assert(Q8(raw<q16_16>(383)).raw() == 1);
    assert(Q8(raw<q16_16>(128)).raw() == 1); // exactly half a unit
    assert(Q8(raw<q16_16>(-128)).raw() == -1);
    assert(Q8(raw<q16_16>(127)).raw() == 0);
    assert(fixed_cast<q15>(raw<q31>(1 << 15)).raw() == 1);
    using Int = fixed<std::int32_t, 0>;
    assert(fixed_cast<Int>(q16_16(2.5)).raw() == 3);
    assert(fixed_cast<Int>(q16_16(-2.5)).raw() == -3);

    // Out of range for the target: resolved by the target's policy.
    const q16_16 big(200.0); // Q8.8 holds [-128, 128)
    using Q8_sat = fixed<std::int16_t, 8, thefblas::saturate>;
    using Q8_wrap = fixed<std::int16_t, 8, thefblas::wrap>;
    assert(fixed_cast<Q8_sat>(big).raw() == 32767);
    assert(fixed_cast<Q8_sat>(-big).raw() == -32768);
    assert(fixed_cast<Q8_wrap>(big).raw() == 51200 - 65536);
    // The largest Q1.31 value rounds up to 1, which Q1.15 cannot hold.
    assert(fixed_cast<q15_sat>((std::numeric_limits<q31>::max)()).raw() == 32767);
    assert(fixed_cast<q15_sat>(std::numeric_limits<q31>::lowest()).raw() == -32768);

#if defined(__SIZEOF_INT128__)
    using Q32_32 = fixed<std::int64_t, 32>;
    assert(Q32_32(half).raw() == (std::int64_t(1) << 31));
    assert(q15(Q32_32(0.25)).raw() == (1 << 13));
    assert(fixed_cast<Q32_32>((std::numeric_limits<q31>::max)()).raw() ==
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
    assert((L::min)().raw() == 1);
    assert(L::lowest().raw() == -32768);
    assert((L::max)().raw() == 32767);
    assert(L::epsilon().raw() == 1);
    assert(L::round_error() == q15(0.5));
    assert(L::denorm_min() == (L::min)());
    assert(L::infinity().raw() == 0);
    assert(L::quiet_NaN().raw() == 0);
    assert(L::signaling_NaN().raw() == 0);

    using Q16 = std::numeric_limits<thefblas::q16_16>;
    assert(Q16::lowest().to_float<double>() == -32768.0);
    assert(Q16::epsilon().to_float<double>() == 1.0 / 65536.0);
    assert(Q16::round_error().to_float<double>() == 0.5);

    // No fractional bits: 0.5 is not representable, so round_error() is one unit.
    using I = std::numeric_limits<fixed<std::int32_t, 0>>;
    assert(I::round_error().raw() == 1);
    assert((I::min)().raw() == 1);
}

} // namespace

int main() {
    test_construction_and_conversion();
    test_arithmetic();
    test_comparisons();
    test_negation_and_abs();
    test_sqrt();
    test_different_widths();
    test_wrap_policy();
    test_saturate_policy();
    test_checked_policy_in_range();
    test_nan_construction();
    test_rounding();
    test_policy_aliases();
    test_q31_conversion_and_sqrt();
    test_q_format_conversion();
    test_numeric_limits();
    return 0;
}
