# thefblas

Minimal, value-type-generic (templated), header-only C++17 BLAS-style library
implementing Level 1 vector routines and dense Level 2 matrix-vector routines
for real and complex types, including a generic fixed-point number type.

thefblas is the fixed-point-capable counterpart of
[theblas](https://github.com/thekyria/theblas). Instead of separate
`s`/`d`/`c`/`z` precision-prefixed functions, every routine is a single C++
template parameterized on the element type `T`. Callers choose the precision
by instantiating with `float`, `double`, `thefblas::fixed<IntType, FracBits, Policy>`
(a generic Q-format fixed-point type usable at any integer width), or
`std::complex<T>` of any of the above.

## Design

- **Fixed-point type**: `thefblas::fixed<IntType, FracBits, Policy>`
  (`include/thefblas/fixed.hpp`) is a generic templated Q-format fixed-point
  number, usable at any width (e.g. `fixed<std::int16_t, 15>` ~ Q1.15,
  `fixed<std::int32_t, 31>` ~ Q1.31). Multiplication and division round to
  nearest, ties away from zero; overflow behaviour is selected by the `Policy`
  template parameter (see [Overflow policies](#overflow-policies)).
  Square root uses an integer-only Newton iteration; no external dependencies
  beyond the STL are required, keeping the library portable to bare-metal
  targets (e.g. ARM Cortex-M without an FPU).
- **Wide accumulators**: internal reductions are performed in a type twice as
  wide as the element type, so intermediate values do not overflow the
  caller's Q format (see [Wide accumulators](#wide-accumulators)).
- **Whole-library genericity**: the API is templated end-to-end, so
  `float`/`double`/`fixed<...>` and their `std::complex<...>` forms share the
  same function templates (`axpy`, `gemv`, `hemv`, etc.) instead of distinct
  precision-prefixed symbols.
- **Complex support**: `std::complex<fixed<...>>` is fully in scope, including
  the Hermitian/conjugated routine families (`dotc`, `hemv`, `her`, `her2`,
  `gerc`, conjugate-transpose in `gemv`/`trmv`/`trsv`).
- **Header-only**: no build step is required to consume the library; just add
  `include/` to your include path, or link against the `thefblas::thefblas`
  CMake interface target.

## Overflow policies

Fixed-point arithmetic has no exponent, so every operation can leave the
representable range of its Q format. `fixed` takes a third template parameter
that selects, at compile time, what happens when it does:

| Policy | Behaviour on overflow | Cost | Typical use |
| --- | --- | --- | --- |
| `thefblas::checked` (default) | `assert()` in debug builds; well-defined two's-complement wraparound in release | none in release | development, tests |
| `thefblas::wrap` | two's-complement wraparound, always well-defined | none | when the format is known to be sufficient, or when modular semantics are wanted |
| `thefblas::saturate` | clamps to the minimum/maximum representable value | one comparison per operation | signal processing, control loops, production firmware |

The policy governs every narrowing operation: `+`, `-`, unary negation, `*`,
`/`, the compound assignments, `abs`, and construction from a floating-point
or integral value. Note that `-x` and `abs(x)` can overflow, because the most
negative value of a two's-complement format has no positive counterpart.

Convenience aliases are provided for the common formats:
`q7`, `q15`, `q31`, `q16_16` use the default `checked` policy, and
`q7_sat`, `q15_sat`, `q31_sat`, `q16_16_sat` use `saturate`.

### Worked example

In Q1.7 (`fixed<std::int8_t, 7>`, one unit = 2^-7, range [-1, 1)):

| Expression | `wrap` | `saturate` | `checked` |
| --- | --- | --- | --- |
| `0.9921875 + 0.0078125` (127 + 1) | `-1.0` (raw -128) | `0.9921875` (raw 127) | asserts, else `-1.0` |
| `-1.0 - 0.0078125` (-128 - 1) | `0.9921875` (raw 127) | `-1.0` (raw -128) | asserts, else `0.9921875` |
| `-(-1.0)` | `-1.0` | `0.9921875` | asserts, else `-1.0` |
| `abs(-1.0)` | `-1.0` | `0.9921875` | asserts, else `0.9921875` |

Out-of-range construction from `float`/`double` clamps under every policy
(modular reduction of an arbitrary real number is not meaningful), and `NaN`
converts to zero.

> **Saturating arithmetic is not associative.** With `saturate` in Q1.7,
> `(0.78125 + 0.78125) + (-0.78125)` is `0.2109375` — the first addition
> clamps to `0.9921875` before the subtraction — whereas
> `0.78125 + (0.78125 + (-0.78125))` is `0.78125`. Expression order matters,
> and no compiler reassociation is permitted.

## Wide accumulators

Saturation fixes the *final* value but not the intermediate ones. Consider
`dot` of `[0.9, 0.9]` with itself in Q1.15: the mathematical result is 1.62,
outside the format, but each product is 0.81 and is perfectly representable.
A naive running sum in Q1.15 overflows on the second addition.

thefblas therefore accumulates every internal reduction in a type twice as
wide as the element type, keeping the same number of fractional bits (so the
widening is a sign extension, with no shift and no rounding), and narrows back
through the element type's overflow policy exactly once, at the end:

- `dot`, `dotu`, `dotc`, `nrm2` and `asum` accumulate the *unshifted* products
  at `2 * FracBits` fractional bits. This removes the per-product rounding
  error as well as the overflow: the reduction is exact up to the final
  rounding.
- `nrm2` takes its square root in the accumulator type, because a sum of
  squares routinely leaves the element range even when the norm does not.
- The Level 2 routines (`symv`/`sbmv`/`spmv`, `hemv`/`hbmv`/`hpmv`, the
  transpose and conjugate-transpose paths of `gemv`/`gbmv`, and
  `trmv`/`tbmv`/`tpmv`, `trsv`/`tbsv`/`tpsv`) use a widened `temp` accumulator.

Two deliberate limitations:

- The `'N'` (no-transpose) paths of `gemv` and `gbmv` accumulate directly into
  `y`, as Netlib BLAS does. Widening them would require an `O(m)` temporary
  buffer, which is unacceptable on the bare-metal targets this library is
  aimed at, so those paths are marginally less accurate than `'T'`/`'C'`.
- When `IntType` is 64 bits and the compiler provides no `__int128`, there is
  no wider type available and the accumulator silently falls back to the same
  width, restoring the naive behaviour.

`float` and `double` accumulate in themselves, so floating-point results are
bit-identical to the straightforward loops that Netlib BLAS specifies.

## Implemented routines

- **Level 1** (`include/thefblas/level1.hpp`): `swap`, `copy`, `axpy`, `scal`
  (including real-scalar scaling of complex vectors), `dot`, `dotu`, `dotc`,
  `nrm2`, `asum`, `rot`, `rotg`, `rotm`, `rotmg`, `iamax`.
- **Level 2, dense** (`include/thefblas/level2.hpp`): `gemv`, `ger`, `geru`,
  `gerc`, `symv`, `hemv`, `syr`, `her`, `syr2`, `her2`, `trmv`, `trsv`.
- **Level 2, banded** (`include/thefblas/level2_banded.hpp`): `gbmv`, `sbmv`,
  `hbmv`, `tbmv`, `tbsv`.
- **Level 2, packed** (`include/thefblas/level2_packed.hpp`): `spmv`, `hpmv`,
  `tpmv`, `tpsv`, `spr`, `hpr`, `spr2`, `hpr2`.

Matrices are column-major. Banded matrices use the classic BLAS band storage
(`ab[(ku + i - j) + j * ldab]` for general bands, `ab[(k + i - j) + j * ldab]`
/ `ab[(i - j) + j * ldab]` for upper/lower symmetric, Hermitian and triangular
bands), and packed matrices store the referenced triangle column by column
(`ap[i + j * (j + 1) / 2]` for upper, `ap[(i - j) + j * (2n - j + 1) / 2]` for
lower); see the header comments for details.

Level 3 routines, and integer division/sqrt algorithms optimized for a
specific embedded target (e.g. CORDIC or Cortex-M DSP intrinsics), are not yet
implemented; the current `fixed<>` division/sqrt are portable, dependency-free
integer algorithms suitable as a baseline for any target, including bare-metal
ARM.

## Building and testing

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The library itself is header-only (`INTERFACE` CMake target); the build only
compiles the test suite and the examples.

`examples/fixed_point_ops.cpp` shows Level 1, dense, banded and packed Level 2
calls on a `fixed<std::int32_t, 20>` (Q11.20) element type:

```sh
./build/examples/thefblas_example_fixed_point_ops
```

