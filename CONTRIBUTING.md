# Contributing to thefblas

Thanks for contributing to thefblas.

This guide explains how to set up your environment, make changes, and open high-quality pull requests.

## Project Basics

- Language: C++17
- Build system: CMake (minimum 3.21)
- Tests: CTest with tests in `tests/`

## Development Setup

### Option 1: Local environment

Requirements:

- CMake 3.21+
- A C++ compiler (GCC, Clang, or MSVC)

Build and test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

### Option 2: GitHub Codespaces / Dev Container

This repository includes a ready-to-use dev container in `.devcontainer/`.

After startup, run:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## What to Contribute

Good contributions include:

- Bug fixes
- New Level 1 BLAS routine implementations
- Cross-platform portability improvements
- Tests and documentation improvements

## Coding Guidelines

- Keep changes minimal and focused.
- Preserve public API stability unless an API change is intentional and documented.
- Keep code portable across GCC, Clang, and MSVC.
- Prefer clear and maintainable implementations over clever but hard-to-read code.
- Avoid unrelated refactors in the same pull request.

## Testing Expectations

When behavior changes, add or update tests in `tests/`.

Before opening a pull request, run:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If you add new functionality, include tests for:

- Expected behavior
- Edge cases
- Any regressions fixed by your change

Write checks with the helpers in `tests/test_support.hpp` rather than
`assert()`. Every failed check prints its location and expression;
`CHECK_EQ(a, b)`, `CHECK_PRED(pred, got, expected)` and
`CHECK_VALUES(cond, got, expected)` also print the values involved, while plain
`CHECK(cond)` does not, so prefer those when values help, or use
`CHECK_MSG(cond, parts...)` to attach a note. `TEST_CONTEXT(...)` adds the
current parameters (e.g. `"uplo=", uplo`) to any failure in its scope. A failed check does not stop the test, and `main` ends
with `return thefblas_test::finish();`, which prints a summary and sets the exit
code.

## Documentation Expectations

Update documentation when code changes affect behavior or usage.

Typical updates include:

- `README.md` for new features or changed usage
- API comments in public headers when public behavior changes

## Pull Request Guidelines

Please keep pull requests easy to review.

Checklist:

1. Build succeeds locally.
2. Tests pass locally.
3. Documentation is updated if needed.
4. Changes are scoped to one clear purpose.
5. Pull request description explains what changed and why.

Suggested PR description format:

- Summary
- Motivation
- Testing
- Notes on compatibility or API impact

## Releasing

1. In a pull request, run `python3 scripts/prepare-release.py <VERSION>`. It
   sets the version in `CMakeLists.txt`, `Doxyfile` and `conanfile.py`, and
   moves the `[Unreleased]` changelog entries into a `[<VERSION>]` section.
2. Merge it. The Release workflow tags the merge commit `v<VERSION>` and
   creates the GitHub Release, with that changelog section as the notes and
   SPDX and CycloneDX SBOMs attached.
3. In a follow-up pull request, run
   `python3 scripts/update-vcpkg-port.py <VERSION>` to point the vcpkg port
   and versions database at the new tag.

## Reporting Issues

When opening an issue, include:

- What you expected to happen
- What happened instead
- Steps to reproduce
- Compiler and platform details
- Build/test command output when relevant

## Code of Conduct

This project follows the Contributor Covenant.

See `CODE_OF_CONDUCT.md` for expected behavior.
