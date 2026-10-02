// Exercises the release-build fallback of the `checked` overflow policy: with
// assertions disabled it must behave like `wrap` for arithmetic and clamp
// out-of-range conversions, all without undefined behaviour. NDEBUG is defined
// here, before any header, because the test CMakeLists keeps assertions enabled
// for every other target; results are therefore checked without assert().
#ifndef NDEBUG
#define NDEBUG
#endif

#include "thefblas/thefblas.h"

#include <cstdint>
#include <cstdio>
#include <type_traits>

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

    // A BLAS reduction whose result leaves the format wraps once, at the final
    // narrowing: 0.9 * 0.9 + 0.9 * 0.9 = 1.62 -> raw 53083 -> 53083 - 65536.
    const F x[2] = {F(0.9), F(0.9)};
    check(thefblas::dot(2, x, 1, x, 1).raw() == 53083 - 65536, "dot wraps at final narrowing");

    return failures == 0 ? 0 : 1;
}
