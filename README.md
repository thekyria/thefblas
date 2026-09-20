# thefblas

Minimal, value-type-generic (templated), header-only C++17 BLAS-style library
implementing Level 1 vector routines and dense Level 2 matrix-vector routines
for real and complex types, including a generic fixed-point number type.

thefblas is the fixed-point-capable counterpart of
[theblas](https://github.com/thekyria/theblas). Instead of separate
`s`/`d`/`c`/`z` precision-prefixed functions, every routine is a single C++
template parameterized on the element type `T`. Callers choose the precision
by instantiating with `float`, `double`, `thefblas::fixed<IntType, FracBits>`
(a generic Q-format fixed-point type usable at any integer width), or
`std::complex<T>` of any of the above.

## Design

- **Fixed-point type**: `thefblas::fixed<IntType, FracBits>` (`include/thefblas/fixed.hpp`)
  is a generic templated Q-format fixed-point number, usable at any width
  (e.g. `fixed<std::int16_t, 15>` ~ Q1.15, `fixed<std::int32_t, 31>` ~ Q1.31).
  Multiplication/division round to nearest; overflow is guarded by `assert()`
  (undefined behavior in release builds) rather than saturating or wrapping.
  Square root uses an integer-only Newton iteration; no external dependencies
  beyond the STL are required, keeping the library portable to bare-metal
  targets (e.g. ARM Cortex-M without an FPU).
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

## Implemented routines

- **Level 1** (`include/thefblas/level1.hpp`): `swap`, `copy`, `axpy`, `scal`
  (including real-scalar scaling of complex vectors), `dot`, `dotu`, `dotc`,
  `nrm2`, `asum`, `rot`, `rotg`, `rotm`, `rotmg`, `iamax`.
- **Level 2, dense** (`include/thefblas/level2.hpp`): `gemv`, `ger`, `geru`,
  `gerc`, `symv`, `hemv`, `syr`, `her`, `syr2`, `her2`, `trmv`, `trsv`.

Banded and packed Level 2 storage variants, and integer division/sqrt
algorithms optimized for a specific embedded target (e.g. CORDIC or
Cortex-M DSP intrinsics), are not yet implemented; the current `fixed<>`
division/sqrt are portable, dependency-free integer algorithms suitable as a
baseline for any target, including bare-metal ARM.

## Building and testing

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The library itself is header-only (`INTERFACE` CMake target); the build only
compiles the test suite.

