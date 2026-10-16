// Embedded-style use of thefblas: Q1.15 with saturation, no iostream, no heap
// and no floating point at run time (every constant is converted at compile
// time). The program checks its own results and reports through its exit
// status, which also makes it a code-size probe: CI prints its size for each
// bare-metal Cortex-M target.

#include "thefblas/level1.hpp"
#include "thefblas/level2.hpp"

#include <array>
#include <cstdlib>
#include <limits>

namespace {

using q15 = thefblas::q15_sat;

// Results may differ from the exact value by the rounding of the inputs and
// of the single narrowing at the end: allow two raw units (2^-14).
bool within_two_units(q15 got, q15 expected) {
    const int diff = got.raw() - expected.raw();
    return std::abs(diff) <= 2;
}

// One output sample of a 4-tap moving-average FIR filter.
bool fir_example() {
    constexpr q15 tap(0.25);
    constexpr std::array<q15, 4> taps = {tap, tap, tap, tap};
    constexpr std::array<q15, 4> window = {q15(0.5), q15(0.25), q15(-0.5), q15(0.75)};
    // 0.25 * (0.5 + 0.25 - 0.5 + 0.75) = 0.25
    constexpr q15 expected(0.25);
    return within_two_units(thefblas::dot(4, taps.data(), 1, window.data(), 1), expected);
}

// y = A x for a 2x3 column-major matrix.
bool gemv_example() {
    constexpr std::array<q15, 6> a = {q15(0.5),   q15(0.125), q15(-0.25),
                                      q15(0.375), q15(0.25),  q15(-0.5)};
    constexpr std::array<q15, 3> x = {q15(0.5), q15(0.25), q15(0.5)};
    // Q1.15 cannot hold 1, so alpha is the largest value, 1 - 2^-15.
    constexpr q15 alpha = (std::numeric_limits<q15>::max)();
    // y0 = 0.5 * 0.5 - 0.25 * 0.25 + 0.25 * 0.5 = 0.3125
    // y1 = 0.125 * 0.5 + 0.375 * 0.25 - 0.5 * 0.5 = -0.09375
    constexpr q15 y0(0.3125);
    constexpr q15 y1(-0.09375);
    std::array<q15, 2> y = {};
    thefblas::gemv('N', 2, 3, alpha, a.data(), 2, x.data(), 1, q15(), y.data(), 1);
    return within_two_units(y[0], y0) && within_two_units(y[1], y1);
}

// The Euclidean norm of (0.3, 0.4) is 0.5.
bool nrm2_example() {
    constexpr std::array<q15, 2> v = {q15(0.3), q15(0.4)};
    constexpr q15 expected(0.5);
    return within_two_units(thefblas::nrm2(2, v.data(), 1), expected);
}

} // namespace

int main() {
    return (fir_example() && gemv_example() && nrm2_example()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
