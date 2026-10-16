# Changelog

All notable changes to thefblas are documented here.

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** — incompatible API changes
- **MINOR** — new functionality, backward-compatible
- **PATCH** — backward-compatible bug fixes

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed

- `sqrt` for `fixed` failed in formats without integer bits (`q7`, `q15`,
  `q31`): its first Newton step formed `f / f = 1`, which these formats cannot
  represent, so `checked` asserted for every positive argument and `wrap`
  returned negative roots (`sqrt(0.25)` gave -0.5). It is now the integer
  square root of `raw * 2^FracBits`, rounded to nearest, which needs no
  intermediate value in range and is exact for every format.
- Real `rotg` constructed 1 on every call, so with the default `checked`
  policy it asserted for every nonzero input in `q7`, `q15` and `q31`. 1 is
  now only formed on the branches that return it.
- `rotm` and `rotmg` cannot work in formats without integer bits, which do not
  hold their flag values -2 and 1; they asserted (`checked`) or silently
  misread the flag. Such formats are now rejected at compile time.
- Conversion from `double` rounded twice: `v * 2^FracBits + 0.5` was rounded
  before the truncation, so e.g. 0.49999999999999994 converted to 1 in
  `fixed<int32_t, 0>`. The fractional part is now compared instead. Values
  that round onto the end of the range (e.g. -128.3 in `fixed<int8_t, 0>`) are
  now accepted rather than treated as out of range.

## [0.2.0] - 2026-10-16

### Added

- Compile-time overflow policies for `thefblas::fixed`: the new third template
  parameter selects `thefblas::checked` (default), `thefblas::wrap` or
  `thefblas::saturate`. The policy governs `+`, `-`, unary negation, `*`, `/`,
  the compound assignments, `abs`, and construction from floating-point or
  integral values (`include/thefblas/overflow.hpp`).
- Convenience aliases `q7`, `q15`, `q31`, `q16_16` and their `_sat` variants.
- Wide accumulators for all internal reductions. `dot`, `dotu`, `dotc`, `nrm2`
  and `asum` now accumulate unshifted products at twice the fractional width,
  and the Level 2 reductions use a doubly wide `temp` accumulator, so
  intermediate values no longer overflow the caller's Q format. Floating-point
  results are unchanged.
- `tests/test_accumulators.cpp` and an overflow-policy matrix in
  `tests/test_fixed.cpp`.
- Repository infrastructure ported from theblas: CMake presets and toolchain
  files, quality options, linter/formatter configuration, CI workflows, Conan
  and vcpkg packaging, and the community/documentation files.
- CI (`build.yml`): a GCC/Clang/AppleClang x Debug/Release test matrix built
  with warnings as errors, and an ASan/UBSan job (GCC and Clang) using
  `THEFBLAS_ENABLE_SANITIZERS`. Static analysis moved to its own `analysis` job.

- `tests/test_reference_blas.cpp` (opt-in, `THEFBLAS_ENABLE_REFERENCE_TESTS`):
  randomised comparison of every floating-point and complex routine against a
  system CBLAS, covering random sizes, positive and negative strides, `lda`
  padding and all `uplo`/`trans`/`diag` options. CI runs it against Netlib
  reference BLAS in the `analysis` and `sanitizers` jobs.
- CI cross-compiles every test and example with warnings as errors:
  `arm-linux-gnueabihf` (32-bit ARM, no `__int128`; Debug and Release, tests
  run under qemu-user) and `arm-none-eabi` for Cortex-M4, Cortex-M4F
  (hard float) and Cortex-M0 (compiled and linked against newlib stubs, not
  run).
- `THEFBLAS_ARM_FLOAT_ABI` and `THEFBLAS_ARM_FPU` options in the
  `arm-none-eabi` toolchain file, next to `THEFBLAS_ARM_MCU`.

- Level 3 routines `gemm`, `syrk`, `herk` and `trsm`
  (`include/thefblas/level3.hpp`, included by `thefblas.h`). Each output
  element is a single widened multiply-accumulate with `alpha`/`beta` (or, in
  `trsm`, `alpha * B` and the division by the diagonal) in the accumulator and
  one narrowing, for every element type and without temporary buffers;
  floating-point results match Netlib to rounding, not bit for bit. Argument
  checks and the `alpha == 0` / `beta == 0` rules follow Netlib.
  `tests/test_level3.cpp` covers every option combination for float, double,
  fixed point and their complex forms, and the reference-BLAS comparison now
  includes the four routines.

- Conversions between Q formats and overflow policies: an explicit converting
  constructor on `fixed` and the equivalent `thefblas::fixed_cast<To>(x)`.
  Dropped fractional bits round to nearest, ties away from zero; results out of
  range are resolved by the target type's policy.
- `std::numeric_limits` specialisation for `fixed`. `min()` is the smallest
  positive value (one raw unit), as for floating-point types; `lowest()` and
  `max()` are the ends of the range and `epsilon()` is the grid step
  `2^-FracBits`. `is_modulo` is true only under `wrap`, and `traps` only under
  `checked` with assertions enabled.
- Contiguous-range overloads of the Level 1 routines `copy`, `axpy`, `scal`,
  `dot`, `dotu`, `dotc`, `nrm2`, `asum`, `iamax`, `rot` and `rotm`, taking
  `std::vector`, `std::array`, C arrays or C++20 `std::span` with unit stride.
  Mismatched sizes, or sizes above `INT_MAX`, assert in debug builds and are a
  no-op otherwise.
- `tests/test_api.cpp`: checks that every public routine is `noexcept`, and
  covers the range overloads and constant evaluation of `fixed` `sqrt`.
- `sdsdot` and `dsdot`: dot products accumulated in a wider type (`double`
  for `float` input, the widened fixed-point type for `fixed`), with range
  overloads. `dsdot` returns the wide sum; `sdsdot` sums `sb` with the
  products and narrows once. Fixed-point types without a wider accumulator
  (64-bit storage, and 32-bit storage without `__int128`) are rejected. The `float` forms are compared against CBLAS in the
  reference test.
- The remaining Level 3 routines `symm`, `hemm`, `syr2k`, `her2k` and `trmm`,
  with the same single widened multiply-accumulate and single narrowing per
  output element as `gemm`, and the same argument checks and `alpha == 0` /
  `beta == 0` rules as Netlib. `tests/test_level3.cpp` covers every option for
  float, double, fixed point and complex, and the reference-BLAS comparison
  includes all five.

### Changed

- `llms.txt` describes the current library: header-only templates over
  `float`, `double`, `fixed` and `std::complex`, Levels 1 to 3, and the files,
  tests, options, presets and workflows that exist. It previously described
  `s`/`d`/`c`/`z` variants, Level 1 only, and source files and examples that
  are not in the repository.
- The tests use a small in-repo helper (`tests/test_support.hpp`) instead of
  `assert()`. A failed check prints its location, the expression and the active
  parameters (routine, options, element index), plus the values involved for
  the value-aware checks (`CHECK_EQ`, `CHECK_PRED`, `CHECK_VALUES`), and the
  test continues. Every failure is counted, but only the first 50 are printed;
  each test ends with a summary line and a non-zero exit code on failure.
  `tests/test_support_selftest.cpp` checks the helper itself.
- `tests/test_widths.cpp` runs every routine on 8-bit (`fixed<int8_t, 4>`) and
  64-bit (`fixed<int64_t, 32>`, where a 128-bit integer exists) storage, real
  and complex, and compares each result with the same routine in `double`
  within the rounding bound of the routine.
- Every public BLAS routine is now `noexcept`.
- `sqrt` for `fixed` and the compound assignments `+=`, `-=`, `*=`, `/=` are
  now `constexpr`.
- `scal` and `asum` now treat a non-positive `incx` as an empty vector (no-op,
  and a result of 0), as Netlib BLAS does and as `iamax` already did. They
  previously walked the vector backwards.
- Complex `rotg` with `a == 0` now follows Netlib reference BLAS 3.10+:
  `c = 0`, `s = conj(b) / |b|` and `a = |b|` (real), or `c = 1`, `s = 0` when
  `b == 0` too. It previously returned `c = 0`, `s = 1`, `a = b`.

- `thefblas::fixed` now takes three template parameters; the third is
  defaulted, so existing two-parameter uses continue to compile.
- Release builds no longer have undefined behaviour on fixed-point overflow:
  every policy, including `checked`, is well defined once `NDEBUG` is set.

### Fixed

- With the formats that have no integer bits (`q7`, `q15`, `q31` and other
  `FracBits == digits` types), routines taking a `beta` (`gemv`, `gbmv`,
  `symv` and the other Level 2 matrix-vector products, and the Level 3
  routines) compared it with a constructed 1, which these formats cannot
  represent, so the `checked` policy asserted for any `beta`. They now test
  for 1 without constructing it.

- Fixed-point `gemv` and `gbmv` with `trans = 'N'`, and the rank-1 and rank-2
  updates (`ger`, `geru`, `gerc`, `syr`, `spr`, `her`, `hpr`, `syr2`, `spr2`,
  `her2`, `hpr2`), rounded `alpha * x(j)` (or `alpha * y(j)`) before
  multiplying by the other operand, so their error grew with that operand's
  magnitude: up to about 390 ulps for `fixed<int64_t, 32>` inputs near 256.
  They now round once per element, like the other Level 2 and Level 3
  routines, and stay within a few ulps. `gemv`/`gbmv` walk the rows of `A` for
  this, so no temporary is needed. Floating-point results are unchanged
  (bit-identical).

- Fixed-point `nrm2` rounded the sum of squares to the element's fractional
  bits before taking the square root, so small norms lost up to half of their
  bits (`nrm2` of `[0.125]` in `fixed<int8_t, 4>` was 0). It now takes the
  integer square root of the exact sum and rounds once. A sum of squares too
  large for its accumulator (for example two 64-bit values near the limit of
  the format) is now detected and clamped, with `checked` asserting, instead
  of wrapping to a wrong norm or to 0.
- Fixed-point `rotg` (real and complex) scaled its inputs as floating point
  does, which multiplied the rounding error of the square root by the scale:
  about 500 ulps for `fixed<int64_t, 32>` inputs near 256. The radius and
  `|a|` are now rounded once from exact sums of squares, and the complex
  radius is formed as `a * norm / |a|` with one rounding.

- The Level 2 option characters (`trans`, `uplo`, `diag`) are upper-cased
  with a `constexpr` ASCII conversion instead of `std::toupper`, which depends
  on the locale. With newlib's out-of-line `std::toupper`, GCC could not tell
  which branch a constant option selects and reported `-Warray-bounds` on a
  branch that does not run (in `test_accumulators.cpp`, Cortex-M Release).
- Multiplication of negative fixed-point values rounded towards negative
  infinity instead of to nearest, ties away from zero, introducing a one-ULP
  bias. Rounding is now symmetric about zero.
- `FracBits` equal to the number of value bits of `IntType` (the canonical
  Q1.15 / Q1.31 formats) is now accepted.
- `nrm2` for `float`/`double` and their complex forms no longer overflows or
  underflows when the norm itself is representable (e.g. `[3e20F, 4e20F]`
  returned `inf`). It now uses Blue's scaled algorithm, as Netlib reference
  BLAS 3.10+ does; results for mid-range inputs are unchanged.
- Dense Level 2 routines (`gemv`, `ger`, `geru`, `gerc`, `symv`, `hemv`, `syr`,
  `her`, `syr2`, `her2`, `trmv`, `trsv`) now return early when `lda` is smaller
  than the number of rows, instead of accessing the matrix out of bounds.
- `rotmg` returned a matrix `H` that did not eliminate `b2` unless `d1 == d2`,
  and rejected any non-positive `d2`. It is now a port of Netlib's algorithm
  (flags -1/0/1/-2, Netlib's handling of negative `d1`/`d2`, and, for
  `float`/`double`, rescaling of the magnitudes of nonzero `d1`/`d2` into
  [2^-24, 2^24]; a valid negative `d2` keeps its sign). This also removes the
  hard-coded `1e-30` floor it used.
- The release workflow tagged a commit whose `CHANGELOG.md`, `Doxyfile` and
  `conanfile.py` were the previous versions, and pushed to `main` directly,
  which its rules reject. Releases are now prepared in a pull request with
  `scripts/prepare-release.py`; once merged, the workflow only tags that
  commit and creates the GitHub Release. The vcpkg port is updated afterwards
  with `scripts/update-vcpkg-port.py`, since it records the tag's tarball
  hash.
- The README now documents that offsets are computed in `int`, with the
  resulting size limits, and the Copilot instructions no longer describe the
  project as a scaffold.

## [0.1.0] - 2026-09-25

### Added

- Initial templated, header-only C++17 implementation of the BLAS-style API
  (`thefblas` namespace). Every routine is generic over its value type and
  works with `float`, `double`, `thefblas::fixed<IntType, FracBits>`, and
  `std::complex<>` element types.
- Fixed-point value type `thefblas::fixed<IntType, FracBits>` with
  saturating/overflow helpers, arithmetic operators, `sqrt`/`abs`, and
  stream output.
- Level-1 routines (unprefixed, templated):
  - `swap`, `copy`, `axpy`, `scal`, `dot` (and complex `dotu`/`dotc`),
    `nrm2`, `asum`, `iamax`
  - Plane rotations: `rot`, `rotg`, `rotm`, `rotmg`
- Level-2 routines over dense, banded, and packed storage:
  - General matrix-vector: `gemv`, `gbmv`
  - Hermitian matrix-vector: `hemv`, `hbmv`, `hpmv`
  - Symmetric matrix-vector: `symv`, `sbmv`, `spmv`
  - Triangular matrix-vector: `trmv`, `tbmv`, `tpmv`
  - Triangular solves: `trsv`, `tbsv`, `tpsv`
  - Rank-1/rank-2 updates: `ger`, `geru`, `gerc`, `her`, `hpr`, `her2`,
    `hpr2`, `syr`, `spr`, `syr2`, `spr2`
- Header-only CMake interface library with `find_package(thefblas CONFIG REQUIRED)`
  support and the `thefblas::thefblas` imported target.
- CMake presets (GCC, Clang, MSVC, ARM cross-compile) and quality options:
  strict warnings, warnings-as-errors, sanitizers, coverage, IPO/LTO, and
  release hardening applied to the test/example executables.
- CTest unit-test suite and runnable examples.
- Packaging: header-only Conan recipe and a vcpkg overlay port/registry.
- Static analysis (cppcheck, clang-tidy, scan-build), pre-commit hooks
  (clang-format, cmake-format, codespell, conventional commit), and CI via
  GitHub Actions.

[Unreleased]: https://github.com/thekyria/thefblas/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/thekyria/thefblas/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thekyria/thefblas/releases/tag/v0.1.0
