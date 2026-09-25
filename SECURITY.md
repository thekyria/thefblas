# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

Only the latest release on `master` receives security fixes.

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately via
[GitHub Security Advisories](https://github.com/thekyria/thefblas/security/advisories/new).

Include the following in your report:

- A clear description of the vulnerability
- Affected version(s) and platform(s)
- Steps to reproduce or a minimal proof-of-concept
- Potential impact assessment (e.g., memory corruption, undefined behaviour)

## Response Timeline

This is a personal open-source project maintained on a best-effort basis.
We aim to send an **initial acknowledgement within 14 days** of receiving a
report. Patch timelines depend on severity and maintainer availability;
critical issues will be prioritised. There are no guaranteed fix SLAs.

## Scope

This library is a **pure-computation C++ library** with no network I/O, file
I/O, authentication, or privilege management. The primary security concern is
**memory safety** — buffer overruns, undefined behaviour from signed overflow,
and incorrect stride/pointer arithmetic.

Out of scope: vulnerabilities in the host application's use of the library
(e.g., passing attacker-controlled `n` or strides without validation at the
application boundary).

## Security Measures

Static analysis runs on every push and pull request:

- **cppcheck** — data-flow, style, and portability checks
- **clang-tidy** — AST-based bug-prone pattern and modernization checks
- **scan-build** — Clang Static Analyzer HTML reports
- **CodeQL** (`security-extended`) — weekly scan; results published to the
  Security tab

Sanitizer presets are available for local testing:

```bash
cmake --preset gcc-debug-sanitized
cmake --build --preset gcc-debug-sanitized
ctest --preset gcc-debug-sanitized
```
