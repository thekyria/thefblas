# Changelog

All notable changes to thefblas are documented here.

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** — incompatible API changes
- **MINOR** — new functionality, backward-compatible
- **PATCH** — backward-compatible bug fixes

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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
