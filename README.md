# thefblas

Minimal, value-type-generic (templated), header-only C++17 BLAS-style library
implementing the Level 1, Level 2 and Level 3 BLAS routines for real and
complex types, including a generic fixed-point number type.

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
  Square root uses an integer-only Newton iteration and, like the arithmetic,
  is `constexpr`; no external dependencies
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
`/`, the compound assignments, `abs`, conversion from another `fixed` type
(see [Conversions](#conversions-and-numeric_limits)), and construction from a
floating-point or integral value. Note that `-x` and `abs(x)` can overflow, because the most
negative value of a two's-complement format has no positive counterpart.
Division by zero is treated as out-of-range and maps to the signed limit chosen
by the active policy (with a debug assertion under `checked`).

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

## Conversions and `numeric_limits`

A `fixed` value converts to another Q format or policy explicitly, either with
the converting constructor or with `thefblas::fixed_cast`:

```cpp
thefblas::q31 x(0.3);
auto a = thefblas::fixed_cast<thefblas::q15_sat>(x); // drop 16 fractional bits
thefblas::q16_16 b(a);                               // exact: widens the grid
```

Dropped fractional bits round to nearest, ties away from zero, as the
arithmetic does. A result outside the target's range goes through the
target's policy, so it saturates under `saturate`, wraps under `wrap`, and
asserts under `checked`. The conversion is exact when the target has at least
as many integer bits and at least as many fractional bits as the source, which
includes changing only the policy.

`std::numeric_limits<fixed<...>>` is specialised. Its values follow the
floating-point convention rather than the integer one: `min()` is the smallest
positive value (one raw unit, `2^-FracBits`), `lowest()` is the most negative
value, and `epsilon()` is the grid step, even in formats such as Q1.15 where 1
itself is not representable. `is_exact` is true, `is_integer` is false, and
`is_modulo` is true only for `wrap`. `traps` is true only for `checked` while
assertions are enabled, since its overflow check then aborts. The remaining properties (`digits`,
`radix`, no infinity or NaN) are those of the storage type.

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
- `nrm2` takes the integer square root of the exact sum of squares in the
  accumulator, because that sum routinely leaves the element range even when
  the norm does not; the norm is rounded once. `rotg` computes its radius the
  same way.
- The Level 2 routines (`symv`/`sbmv`/`spmv`, `hemv`/`hbmv`/`hpmv`,
  `gemv`/`gbmv` in every `trans` form, and `trmv`/`tbmv`/`tpmv`,
  `trsv`/`tbsv`/`tpsv`) form each output element as a
  single widened multiply-accumulate over its whole row, again at
  `2 * FracBits`, and narrow through the policy exactly once per element.
  Where the routine computes `alpha * op(A) * x + beta * y`, the `alpha`
  scaling, the `beta` scaling of the `y` element and their sum also happen in
  the accumulator type, before that single narrowing. The
  symmetric/Hermitian routines gather the mirrored element from the stored
  triangle instead of updating `y` term by term, so partial sums that leave
  the element range cannot saturate or wrap before later terms cancel them.
  The non-transposed `gemv`/`gbmv` walk each row of `A` instead of Netlib's
  columns, so they need no `O(m)` temporary.
- The rank-1 and rank-2 updates (`ger`/`geru`/`gerc`, `syr`/`spr`,
  `her`/`hpr`, `syr2`/`spr2`, `her2`/`hpr2`) form each element's products
  exactly, scale their sum by `alpha` in the accumulator type, add the element
  and narrow once. Netlib's order rounds `alpha * y(j)` first, and that
  rounding would be multiplied by `x(i)`.
- The Level 3 routines do the same for every element type: each element of
  `C` (or of `B` in `trmm`, or of the solution `X` in `trsm`) is one widened
  multiply-accumulate over the inner dimension, with `alpha`, the
  `beta`-scaled old element (or, in `trsm`, `alpha * B` and the division by
  the diagonal) in the accumulator, and a single narrowing. `her2k` keeps its
  two products in separate accumulators, since they are scaled by `alpha` and
  `conj(alpha)`. No temporary buffers are needed.

Two deliberate limitations:

- When `IntType` is 64 bits and the compiler provides no `__int128`, there is
  no wider type to evaluate the arithmetic in, so `fixed<std::int64_t, ...>`
  is rejected at compile time rather than silently risking undefined signed
  overflow.
- The widest supported element type — `fixed<std::int64_t, ...>` with
  `__int128`, or `fixed<std::int32_t, ...>` without it — has no wider `fixed`
  storage, so its accumulator keeps the element width. The products are still
  summed in the wider integer, but the final `alpha` scaling and `y` update are
  narrowed through the policy in the element width. (In the rank-1 and rank-2
  updates this means `x(i) * y(j)` itself must fit the element range.)

`float` and `double` accumulate in themselves, so floating-point Level 1 and
Level 2 results are bit-identical to the straightforward loops that Netlib BLAS
specifies. The Level 3 routines use the per-element order above for every type,
which differs from Netlib's loop order, so their floating-point results agree
with Netlib to rounding rather than bit for bit. The other exception is `nrm2`, which for `float`/`double` (and their complex forms) uses
Blue's scaled algorithm, as Netlib reference BLAS 3.10+ does: elements in the
mid range are summed unscaled (so the result is bit-identical to
`sqrt(sum of squares)` there), while very large or very small elements are
scaled by exact powers of two so that the norm neither overflows nor
underflows whenever it is itself representable. The
row gathering above is used for fixed point only: for `float`, `double` and
their complex counterparts, `symv`/`sbmv`/`spmv` and `hemv`/`hbmv`/`hpmv`
keep the Netlib column-update order.

## Implemented routines

- **Level 1** (`include/thefblas/level1.hpp`): `swap`, `copy`, `axpy`, `scal`
  (including real-scalar scaling of complex vectors), `dot`, `dotu`, `dotc`,
  `sdsdot`, `dsdot`, `nrm2`, `asum`, `rot`, `rotg`, `rotm`, `rotmg`, `iamax`.
  `dsdot` returns `x^T y` summed in a wider type (`double` for `float`, the
  widened fixed-point type for `fixed`), and `sdsdot` sums `sb` with the
  products in that type before narrowing once. They accept `float`, and
  fixed-point types with 8- or 16-bit storage, or 32-bit storage where the
  compiler has a 128-bit integer.
- **Level 2, dense** (`include/thefblas/level2.hpp`): `gemv`, `ger`, `geru`,
  `gerc`, `symv`, `hemv`, `syr`, `her`, `syr2`, `her2`, `trmv`, `trsv`.
- **Level 2, banded** (`include/thefblas/level2_banded.hpp`): `gbmv`, `sbmv`,
  `hbmv`, `tbmv`, `tbsv`.
- **Level 2, packed** (`include/thefblas/level2_packed.hpp`): `spmv`, `hpmv`,
  `tpmv`, `tpsv`, `spr`, `hpr`, `spr2`, `hpr2`.
- **Level 3** (`include/thefblas/level3.hpp`): `gemm`, `symm`, `hemm`, `syrk`,
  `herk`, `syr2k`, `her2k`, `trmm`, `trsm`.

Matrices are column-major. As in Netlib BLAS, a dense routine whose leading
dimension is smaller than the number of rows (`lda < m` for `gemv`/`ger`/`geru`/`gerc`,
`lda < n` for the square routines) is rejected as invalid and returns without
touching any operand, like the existing banded `lda` checks. Banded matrices use the classic BLAS band storage
(`ab[(ku + i - j) + j * ldab]` for general bands, `ab[(k + i - j) + j * ldab]`
/ `ab[(i - j) + j * ldab]` for upper/lower symmetric, Hermitian and triangular
bands), and packed matrices store the referenced triangle column by column
(`ap[i + j * (j + 1) / 2]` for upper, `ap[(i - j) + j * (2n - j + 1) / 2]` for
lower); see the header comments for details.

Every public routine is `noexcept`. The supported element types never throw
and the routines never allocate, so an element type whose operations do throw
would terminate the program rather than propagate the exception.

### Range overloads

The Level 1 routines also accept whole contiguous ranges with unit stride:
`std::vector`, `std::array`, C arrays, or C++20 `std::span`. The element count
comes from `std::size`:

```cpp
std::vector<double> x = {1.0, 2.0, 3.0}, y = {4.0, 5.0, 6.0};
double d = thefblas::dot(x, y);  // 32
thefblas::axpy(2.0, x, y);       // y = {6, 9, 12}
int i = thefblas::iamax(y);      // 3: 1-based, as in the pointer API
```

Overloads exist for `copy`, `axpy`, `scal`, `dot`, `dotu`, `dotc`, `dsdot`,
`sdsdot`, `nrm2`, `asum`, `iamax`, `rot` and `rotm`. Two ranges of different sizes, or a size
above `INT_MAX`, assert in debug builds and otherwise act as `n == 0` (a
no-op, or the routine's empty result: zero, or `sb` for `sdsdot`). There is no range `swap`, so that it can't compete with
`std::swap` through argument-dependent lookup.

Integer division/sqrt algorithms optimized for a specific embedded target
(e.g. CORDIC or Cortex-M DSP intrinsics) are not yet implemented; the current `fixed<>` division/sqrt are portable, dependency-free
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

A failed check prints its location, the expression and the parameters it ran
with (and, for the value-aware checks, the values involved), and the test
carries on, so one run reports every failure (the first 50 in detail, the rest
only in the count); each test ends with a line such as
`5809 checks, 0 failures`.

`tests/test_widths.cpp` runs every routine on 8-bit and 64-bit fixed-point
storage and compares each result with the same routine in `double`, within
the rounding bound of that routine. Its inputs are random but reproducible;
set `THEFBLAS_TEST_SEED` to try others.

With `-DTHEFBLAS_ENABLE_REFERENCE_TESTS=ON`, the build also compiles
`tests/test_reference_blas.cpp`, which needs a system BLAS with the CBLAS
interface (for example `libblas-dev` or `libopenblas-dev` on Debian/Ubuntu). It
runs every `float`, `double` and complex routine on random sizes, strides
(including negative ones), leading dimensions and `uplo`/`trans`/`diag`
combinations, and compares the whole output buffers with the reference. The
seed is fixed for reproducibility; set `THEFBLAS_TEST_SEED` to try others.

### Cross-compiling

The toolchain files in `cmake/toolchains/` cover 32-bit ARM Linux and
bare-metal Cortex-M. CI builds every test and example with both (warnings as
errors):

```sh
# 32-bit ARM Linux (no __int128); runs the tests under qemu-user.
cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake \
      -DTHEFBLAS_BUILD_TESTS=ON "-DCMAKE_CROSSCOMPILING_EMULATOR=qemu-arm;-L;/usr/arm-linux-gnueabihf"
cmake --build build-arm && ctest --test-dir build-arm

# Bare-metal Cortex-M4F; links against newlib's stubs, so it checks that the
# code compiles and links but the binaries are not run.
cmake -S . -B build-m4 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake \
      -DTHEFBLAS_ARM_MCU=cortex-m4 -DTHEFBLAS_ARM_FLOAT_ABI=hard -DTHEFBLAS_ARM_FPU=fpv4-sp-d16 \
      -DTHEFBLAS_BUILD_TESTS=ON -DCMAKE_EXE_LINKER_FLAGS=--specs=nosys.specs
cmake --build build-m4
```

`THEFBLAS_ARM_MCU` (default `cortex-m4`), `THEFBLAS_ARM_FLOAT_ABI` (default
`soft`) and `THEFBLAS_ARM_FPU` (default none) select the Cortex-M variant.

`examples/fixed_point_ops.cpp` shows Level 1, dense, banded and packed Level 2
calls on a `fixed<std::int32_t, 20>` (Q11.20) element type:

```sh
./build/examples/thefblas_example_fixed_point_ops
```
