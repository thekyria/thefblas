# Changelog

All notable changes to thefblas are documented here.

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** — incompatible API changes
- **MINOR** — new functionality, backward-compatible
- **PATCH** — backward-compatible bug fixes

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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

### Changed

- `thefblas::fixed` now takes three template parameters; the third is
  defaulted, so existing two-parameter uses continue to compile.
- Release builds no longer have undefined behaviour on fixed-point overflow:
  every policy, including `checked`, is well defined once `NDEBUG` is set.

### Fixed

- Multiplication of negative fixed-point values rounded towards negative
  infinity instead of to nearest, ties away from zero, introducing a one-ULP
  bias. Rounding is now symmetric about zero.
- `FracBits` equal to the number of value bits of `IntType` (the canonical
  Q1.15 / Q1.31 formats) is now accepted.

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

[Unreleased]: https://github.com/thekyria/thefblas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thekyria/thefblas/releases/tag/v0.1.0
