// Exercises the release-build fallback of the `checked` overflow policy: with
// assertions disabled it must behave like `wrap` for arithmetic and for
// conversions from another fixed type, and clamp out-of-range conversions from
// floating-point and integer values, all without undefined behaviour. NDEBUG
// is defined here, before any header, because the test CMakeLists keeps
// assertions enabled for every other target; results are therefore checked
// without assert().
#ifndef NDEBUG
#define NDEBUG
#endif

#include "thefblas/thefblas.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *what) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        ++failures;
    }
}

} // namespace

int main() {
    using F = thefblas::q15; // fixed<std::int16_t, 15, checked>
    static_assert(std::is_same<F::policy, thefblas::checked>::value, "q15 uses checked");

    const F max = F::from_raw(32767);
    const F min = F::from_raw(-32768);
    const F tiny = F::from_raw(1);

    // Arithmetic overflow wraps modulo 2^16.
    check((max + tiny).raw() == -32768, "max + 1 wraps to min");
    check((min - tiny).raw() == 32767, "min - 1 wraps to max");
    check((-min).raw() == -32768, "-min wraps to min");
    check((min * min).raw() == -32768, "(-1) * (-1) = 1.0 wraps to min");

    // Out-of-range conversions and division by zero clamp.
    check(F(2.0).raw() == 32767, "2.0 clamps to max");
    check(F(-2.0).raw() == -32768, "-2.0 clamps to min");
    check(F(1).raw() == 32767, "integer 1 clamps to max");
    check((F(0.5) / F::from_raw(0)).raw() == 32767, "positive / 0 clamps to max");
    check((F(-0.5) / F::from_raw(0)).raw() == -32768, "negative / 0 clamps to min");

    using F8 = thefblas::fixed<std::int8_t, 4, thefblas::checked>;
    using F32 = thefblas::fixed<std::int32_t, 20, thefblas::checked>;
    check(F8(100).raw() == 127, "int8 conversion clamps to max");
    check(F32(10000).raw() == (std::numeric_limits<std::int32_t>::max)(),
          "int32 conversion clamps to max");
#if defined(__SIZEOF_INT128__)
    // 64-bit fixed-point needs a 128-bit intermediate type (not available on MSVC).
    using F64 = thefblas::fixed<std::int64_t, 32, thefblas::checked>;
    check(F64(std::int64_t(1) << 40).raw() == (std::numeric_limits<std::int64_t>::max)(),
          "int64 conversion clamps to max");
#endif

    // Conversion from another fixed type goes through the policy's narrowing, so
    // it wraps: 1.5 in Q16.16 is raw 98304, i.e. 49152 in Q1.15, which wraps to
    // 49152 - 65536. With assertions disabled, nothing traps.
    check(F(thefblas::q16_16(1.5)).raw() == 49152 - 65536, "fixed conversion wraps");
    check(thefblas::fixed_cast<F>(thefblas::q16_16(-1.5)).raw() == 65536 - 49152,
          "fixed_cast wraps");
    static_assert(!std::numeric_limits<F>::traps, "checked does not trap under NDEBUG");

    // Range overloads: with assertions disabled, ranges of different sizes make
    // the call a no-op (or return zero) rather than read past the shorter one.
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0};
    check(thefblas::dot(a, b) == 0.0, "mismatched dot returns zero");
    thefblas::axpy(1.0, a, b);
    check(b[0] == 4.0 && b[1] == 5.0, "mismatched axpy is a no-op");

    // A BLAS reduction whose result leaves the format wraps once, at the final
    // narrowing: 0.9 * 0.9 + 0.9 * 0.9 = 1.62 -> raw 53083 -> 53083 - 65536.
    const F x[2] = {F(0.9), F(0.9)};
    check(thefblas::dot(2, x, 1, x, 1).raw() == 53083 - 65536, "dot wraps at final narrowing");

    return failures == 0 ? 0 : 1;
}
