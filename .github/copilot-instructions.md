# thefblas Copilot Instructions

## Project Overview

This repository contains a minimal C++ library scaffold named thefblas.
Focus on correctness, portability, and maintainability while keeping changes small and testable.

## Tech Stack

- Language: C++17
- Build system: CMake (minimum 3.15)
- Test framework: CTest with tests in the tests directory

## Repository Structure

- include/thefblas/: public headers and exported API surface
- src/: library implementation
- tests/: unit/integration tests used by CTest
- build/: generated artifacts; do not edit generated files

## Development Workflow

- Prefer minimal diffs and preserve existing style
- Update or add tests in tests/ when behavior changes
- Keep public headers stable unless API changes are intentional and documented
- Keep code cross-platform for GCC, Clang, and MSVC

## Build and Test

Typical local workflow:

1. Configure: cmake -S . -B build
2. Build: cmake --build build
3. Test: ctest --test-dir build

## Instruction Sources

For detailed guidance, follow these instruction files:

- .github/instructions/agent-safety.instructions.md
- .github/instructions/agent-skills.instructions.md
- .github/instructions/agents.instructions.md
- .github/instructions/cmake-vcpkg.instructions.md
- .github/instructions/makefile.instructions.md
- .github/instructions/code-review-generic.instructions.md
- .github/instructions/containerization-docker-best-practices.instructions.md
- .github/instructions/cpp-language-service-tools.instructions.md
- .github/instructions/security-and-owasp.instructions.md
- .github/instructions/performance-optimization.instructions.md
- .github/instructions/update-docs-on-code-change.instructions.md
- .github/instructions/github-actions-ci-cd-best-practices.instructions.md
- .github/instructions/shell.instructions.md

Also consult AGENTS.md for available custom agents and skills.

## Documentation Expectations

When code changes affect usage, setup, APIs, or behavior, update README.md and related documentation in the same change.

## CI/CD Expectations

When creating or editing GitHub Actions workflows, follow the CI/CD instruction file above and keep workflow security settings explicit.
