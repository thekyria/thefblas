# API Reference

This page explains how the public API of **thefblas** is organised and how to
call it: the element types, the `fixed` type and its overflow policies, the
argument conventions shared by every routine, and one table per BLAS level
with each routine's signature and the Netlib routines it replaces. The formula
each routine evaluates, and the fixed-point arithmetic behind it, are in the
[Mathematical Reference](math-reference.md); the per-function Doxygen pages
are in the API documentation generated from the headers.

---

## Headers and namespace

Everything lives in namespace `thefblas` and is header-only.

| Header | Contents |
|---|---|
| `thefblas/thefblas.h` | Umbrella header: includes all of the below |
| `thefblas/fixed.hpp` | `fixed<IntType, FracBits, Policy>`, `abs`, `sqrt`, `fixed_cast`, the `q*` aliases, `std::numeric_limits<fixed<...>>` |
| `thefblas/overflow.hpp` | The overflow policies `checked`, `wrap`, `saturate` |
| `thefblas/level1.hpp` | Level 1 vector routines and their range overloads |
| `thefblas/level2.hpp` | Level 2, dense storage |
| `thefblas/level2_banded.hpp` | Level 2, band storage |
| `thefblas/level2_packed.hpp` | Level 2, packed storage |
| `thefblas/level3.hpp` | Level 3 matrix-matrix routines |

`thefblas/detail.hpp` and namespace `thefblas::detail` are internal.

---

## Element types

Each routine is one function template over the element type `T`. The
precision prefix of Netlib BLAS becomes the template argument, which the
compiler deduces from the pointer arguments:

| Netlib prefix | thefblas element type |
|---|---|
| `s` | `float` |
| `d` | `double` |
| `c` | `std::complex<float>` |
| `z` | `std::complex<double>` |
| — | `thefblas::fixed<IntType, FracBits, Policy>` |
| — | `std::complex<thefblas::fixed<IntType, FracBits, Policy>>` |

So `thefblas::gemm` on `double*` is `dgemm`, on `std::complex<float>*` it is
`cgemm`, and on `q15_sat*` it is a Q1.15 saturating `gemm` that has no Netlib
counterpart.

Where Netlib has separate real and complex routines, thefblas keeps them as
separate templates:

- **Real only:** `dot`, `rotm`, `rotmg`, `ger`, `symv`, `sbmv`, `spmv`, `syr`,
  `spr`, `syr2`, `spr2`, `sdsdot`, `dsdot`.
- **Complex only** (`T` is the real part type, the arrays are
  `std::complex<T>`): `dotu`, `dotc`, `geru`, `gerc`, `hemv`, `hbmv`, `hpmv`,
  `her`, `hpr`, `her2`, `hpr2`, `hemm`, `herk`, `her2k`.
- **Both:** all other routines. `symm`, `syrk` and `syr2k` accept complex
  types too (`csymm`, `zsyrk`, ...), as in Netlib.

A few routines mix a real scalar with complex data, exactly where Netlib does:
`scal` with a real `alpha` on a complex vector (`csscal`, `zdscal`), `rot`
with real `c` and `s` (`csrot`, `zdrot`), `rotg` with a real `c`, `her`/`hpr`
and `herk` with real `alpha` (and `beta` for `herk`), and `her2k` with a real
`beta`. `nrm2` and `asum` of a complex vector return the real part type, as
`scnrm2` and `dzasum` do.

---

## The `fixed` type

```cpp
template <typename IntType, int FracBits, typename Policy = checked>
class fixed;
```

A signed binary fixed-point number: the value is the integer `raw()` times
$2^{-\text{FracBits}}$. `IntType` is the storage (`std::int8_t`,
`std::int16_t`, `std::int32_t`, or `std::int64_t` where the compiler has
`__int128`); `FracBits` is between 0 and the number of value bits of
`IntType`; `Policy` selects what happens on overflow. See
[Fixed-point arithmetic](math-reference.md#fixed-point-arithmetic) in the
Mathematical Reference for the representation, rounding and error analysis.

### Aliases

| Alias | Type | Q format | Range | Step |
|---|---|---|---|---|
| `q7` / `q7_sat` | `fixed<std::int8_t, 7>` | Q1.7 | $[-1, 1)$ | $2^{-7}$ |
| `q15` / `q15_sat` | `fixed<std::int16_t, 15>` | Q1.15 | $[-1, 1)$ | $2^{-15}$ |
| `q31` / `q31_sat` | `fixed<std::int32_t, 31>` | Q1.31 | $[-1, 1)$ | $2^{-31}$ |
| `q16_16` / `q16_16_sat` | `fixed<std::int32_t, 16>` | Q16.16 | $[-2^{15}, 2^{15})$ | $2^{-16}$ |

The plain aliases use `checked`, the `_sat` ones `saturate`. Any other format
is a `fixed<...>` away, for example `fixed<std::int32_t, 20, saturate>` for
Q12.20.

### Members and free functions

| Expression | Meaning |
|---|---|
| `fixed()` | Zero |
| `fixed(v)` (explicit) | From an arithmetic value `v`: a floating-point `v` is rounded to the nearest grid point (ties away from zero), an integral `v` is exact. Out of range clamps under every policy (and `checked` asserts); NaN gives zero |
| `fixed(other)` (explicit) | From another `fixed` format or policy; same as `fixed_cast` |
| `fixed::from_raw(r)` | The value with raw integer `r`, no scaling |
| `x.raw()` | The raw integer |
| `x.to_float<F = double>()` | The value as `float` or `double` |
| `+ - * /`, unary `-`, `+= -= *= /=` | Arithmetic; `*` and `/` round to nearest, ties away from zero; every result goes through `Policy` |
| `== != < > <= >=` | Comparisons on the raw value (exact) |
| `os << x` | Writes `x.to_float<double>()` |
| `abs(x)` | Absolute value; `abs(lowest())` goes through `Policy` |
| `sqrt(x)` | Square root, rounded to nearest; `constexpr` and integer-only |
| `fixed_cast<To>(x)` | Converts to the `fixed` type `To`, rounding dropped fractional bits to nearest and applying `To`'s policy |
| `fixed::rep`, `fixed::policy`, `fixed::frac_bits` | `IntType`, `Policy`, `FracBits` |

All of these are `constexpr` and `noexcept` (except `operator<<`). There are
no implicit conversions to or from `fixed`, so mixing it with `double` in an
expression does not compile: write `q15(0.5)` or `x.to_float()`.

### Overflow policies

| Policy | On overflow | `numeric_limits::is_modulo` |
|---|---|---|
| `checked` (default) | `assert` in debug builds, two's-complement wraparound with `NDEBUG` | `false` |
| `wrap` | Two's-complement wraparound | `true` |
| `saturate` | Clamp to `lowest()` or `max()` | `false` |

The policy is part of the type. Values of different policies or formats do
not mix in arithmetic; convert with `fixed_cast` first. Division by zero gives
`max()` for a non-negative dividend and `lowest()` for a negative one under
every policy (and `checked` asserts).

### `std::numeric_limits`

`std::numeric_limits<fixed<...>>` follows the floating-point convention:
`min()` and `epsilon()` are one raw unit ($2^{-\text{FracBits}}$), `lowest()`
is the most negative value, `max()` the largest, and `round_error()` half a
unit. There are no infinities or NaNs. For `q15`: `epsilon()` is
$2^{-15} \approx 3.05 \times 10^{-5}$, `max()` is $1 - 2^{-15}$ and
`lowest()` is $-1$.

---

## Calling conventions

The pointer API keeps the Netlib BLAS argument order and semantics, so a
Netlib call translates by dropping the prefix and passing values instead of
Fortran references:

```cpp
// Fortran: CALL DGEMV('N', M, N, ALPHA, A, LDA, X, 1, BETA, Y, 1)
thefblas::gemv('N', m, n, alpha, a, lda, x, 1, beta, y, 1);
```

- **Sizes, strides, leading dimensions** are `int`. Every offset a call forms
  must fit in `int` (see the README for the limits).
- **Strides.** `incx`/`incy` may be negative: the vector then starts at the
  far end of the array, as in Netlib, so element $i$ (1-based) is at
  `x[(i - 1) * incx]` for positive and `x[(n - i) * |incx|]` for negative
  strides. `scal`, `asum` and `iamax` require `incx > 0`, as in Netlib, and
  return without effect (or zero) otherwise.
- **Matrices** are column-major: element $(i, j)$ (0-based) is at
  `a[i + j * lda]`, with `lda` at least the number of rows. Band and packed
  storage follow Netlib exactly; the formulas are in the README and in the
  header comments.
- **Option characters** (`trans`, `uplo`, `diag`, `side`) are case-insensitive
  `char`s: `'N'`/`'T'`/`'C'`, `'U'`/`'L'`, `'N'`/`'U'`, `'L'`/`'R'`. For real
  element types `'C'` means `'T'`.
- **Quick returns.** As in Netlib, `n <= 0` (or `m`, `k`) makes a call a
  no-op, `A` and `B` are not read when `alpha` is zero, and `y` or `C` is not
  read when `beta` is zero (so it may hold garbage).
- **Invalid arguments** (an unknown option character, a negative size, a zero
  stride, a leading dimension that is too small) make the call return without
  touching any operand. There is no `xerbla`-style handler.
- **No exceptions, no allocation.** Every routine is `noexcept` and uses no
  heap memory or temporary arrays, which is what bare-metal targets need.
- **Return values.** `iamax` returns a **1-based** index (0 for an empty or
  invalid call). The dot products, `nrm2` and `asum` return the result by
  value; the other routines return `void` and write through their pointer
  arguments.

---

## Level 1 — vector operations

| thefblas template | Netlib routines | Operation |
|---|---|---|
| `swap(n, x, incx, y, incy)` | `sswap` `dswap` `cswap` `zswap` | $x \leftrightarrow y$ |
| `copy(n, x, incx, y, incy)` | `scopy` `dcopy` `ccopy` `zcopy` | $y \leftarrow x$ |
| `axpy(n, alpha, x, incx, y, incy)` | `saxpy` `daxpy` `caxpy` `zaxpy` | $y \leftarrow \alpha x + y$ |
| `scal(n, alpha, x, incx)` | `sscal` `dscal` `cscal` `zscal` `csscal` `zdscal` | $x \leftarrow \alpha x$ |
| `dot(n, x, incx, y, incy)` | `sdot` `ddot` | $x^T y$ |
| `dotu(n, x, incx, y, incy)` | `cdotu` `zdotu` | $x^T y$ |
| `dotc(n, x, incx, y, incy)` | `cdotc` `zdotc` | $x^H y$ |
| `dsdot(n, x, incx, y, incy)` | `dsdot` | $x^T y$, returned in a wider type |
| `sdsdot(n, sb, x, incx, y, incy)` | `sdsdot` | $s_b + x^T y$, summed in a wider type |
| `nrm2(n, x, incx)` | `snrm2` `dnrm2` `scnrm2` `dznrm2` | $\lVert x \rVert_2$ |
| `asum(n, x, incx)` | `sasum` `dasum` `scasum` `dzasum` | $\sum \lvert \operatorname{Re} x_i \rvert + \lvert \operatorname{Im} x_i \rvert$ |
| `iamax(n, x, incx)` | `isamax` `idamax` `icamax` `izamax` | First index of the largest $\lvert x_i \rvert_1$ |
| `rot(n, x, incx, y, incy, c, s)` | `srot` `drot` `csrot` `zdrot` | Apply a plane rotation |
| `rotg(a, b, c, s)` | `srotg` `drotg` `crotg` `zrotg` | Construct a plane rotation |
| `rotm(n, x, incx, y, incy, param)` | `srotm` `drotm` | Apply a modified rotation |
| `rotmg(d1, d2, b1, b2, param)` | `srotmg` `drotmg` | Construct a modified rotation |

Notes:

- `dsdot` returns `double` for `float` input and, for `fixed<I, F, P>`, the
  type `fixed<wider I, F, P>`; `sdsdot` returns `T`. Both accept `float` and
  fixed-point types with 8- or 16-bit storage, or 32-bit storage where the
  compiler has a 128-bit integer.
- Real `rotg` takes all four arguments by pointer and overwrites `a` with $r$
  and `b` with Netlib's $z$. Complex `rotg` is
  `rotg(std::complex<T>* a, std::complex<T> b, T* c, std::complex<T>* s)`.
- `rotm` and `rotmg` store the flag values $-2$ and $1$ in `param[0]`, so they
  need a format with at least one integer bit; `q7`, `q15` and `q31` are
  rejected at compile time.

### Range overloads

`copy`, `axpy`, `scal`, `dot`, `dotu`, `dotc`, `dsdot`, `sdsdot`, `nrm2`,
`asum`, `iamax`, `rot` and `rotm` also accept whole contiguous ranges (C
arrays, `std::array`, `std::vector`, `std::span`, anything with `std::data`
and `std::size`) with unit stride:

```cpp
std::vector<double> x = {1.0, 2.0, 3.0}, y = {4.0, 5.0, 6.0};
double d = thefblas::dot(x, y);  // 32
thefblas::axpy(2.0, x, y);       // y = {6, 9, 12}
int i = thefblas::iamax(y);      // 3 (1-based)
```

Two ranges of different sizes, or a size above `INT_MAX`, assert in debug
builds and otherwise behave as `n == 0`.

---

## Level 2 — matrix-vector operations

| thefblas template | Netlib routines | Operation |
|---|---|---|
| `gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)` | `?gemv` (s d c z) | $y \leftarrow \alpha\,\operatorname{op}(A)\,x + \beta y$ |
| `gbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)` | `?gbmv` (s d c z) | same, $A$ banded |
| `symv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)` | `ssymv` `dsymv` | $y \leftarrow \alpha A x + \beta y$, $A = A^T$ |
| `sbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)` | `ssbmv` `dsbmv` | same, banded |
| `spmv(uplo, n, alpha, ap, x, incx, beta, y, incy)` | `sspmv` `dspmv` | same, packed |
| `hemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)` | `chemv` `zhemv` | $y \leftarrow \alpha A x + \beta y$, $A = A^H$ |
| `hbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)` | `chbmv` `zhbmv` | same, banded |
| `hpmv(uplo, n, alpha, ap, x, incx, beta, y, incy)` | `chpmv` `zhpmv` | same, packed |
| `trmv(uplo, trans, diag, n, a, lda, x, incx)` | `?trmv` (s d c z) | $x \leftarrow \operatorname{op}(A)\,x$ |
| `tbmv(uplo, trans, diag, n, k, a, lda, x, incx)` | `?tbmv` (s d c z) | same, banded |
| `tpmv(uplo, trans, diag, n, ap, x, incx)` | `?tpmv` (s d c z) | same, packed |
| `trsv(uplo, trans, diag, n, a, lda, x, incx)` | `?trsv` (s d c z) | solve $\operatorname{op}(A)\,x = b$ |
| `tbsv(uplo, trans, diag, n, k, a, lda, x, incx)` | `?tbsv` (s d c z) | same, banded |
| `tpsv(uplo, trans, diag, n, ap, x, incx)` | `?tpsv` (s d c z) | same, packed |
| `ger(m, n, alpha, x, incx, y, incy, a, lda)` | `sger` `dger` | $A \leftarrow \alpha x y^T + A$ |
| `geru(m, n, alpha, x, incx, y, incy, a, lda)` | `cgeru` `zgeru` | $A \leftarrow \alpha x y^T + A$ |
| `gerc(m, n, alpha, x, incx, y, incy, a, lda)` | `cgerc` `zgerc` | $A \leftarrow \alpha x y^H + A$ |
| `syr(uplo, n, alpha, x, incx, a, lda)` | `ssyr` `dsyr` | $A \leftarrow \alpha x x^T + A$ |
| `spr(uplo, n, alpha, x, incx, ap)` | `sspr` `dspr` | same, packed |
| `syr2(uplo, n, alpha, x, incx, y, incy, a, lda)` | `ssyr2` `dsyr2` | $A \leftarrow \alpha x y^T + \alpha y x^T + A$ |
| `spr2(uplo, n, alpha, x, incx, y, incy, ap)` | `sspr2` `dspr2` | same, packed |
| `her(uplo, n, alpha, x, incx, a, lda)` | `cher` `zher` | $A \leftarrow \alpha x x^H + A$, real $\alpha$ |
| `hpr(uplo, n, alpha, x, incx, ap)` | `chpr` `zhpr` | same, packed |
| `her2(uplo, n, alpha, x, incx, y, incy, a, lda)` | `cher2` `zher2` | $A \leftarrow \alpha x y^H + \bar\alpha y x^H + A$ |
| `hpr2(uplo, n, alpha, x, incx, y, incy, ap)` | `chpr2` `zhpr2` | same, packed |

---

## Level 3 — matrix-matrix operations

| thefblas template | Netlib routines | Operation |
|---|---|---|
| `gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)` | `?gemm` (s d c z) | $C \leftarrow \alpha\,\operatorname{op}(A)\operatorname{op}(B) + \beta C$ |
| `symm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)` | `?symm` (s d c z) | $C \leftarrow \alpha A B + \beta C$ or $\alpha B A + \beta C$, $A = A^T$ |
| `hemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)` | `chemm` `zhemm` | same, $A = A^H$ |
| `syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)` | `?syrk` (s d c z) | $C \leftarrow \alpha A A^T + \beta C$ or $\alpha A^T A + \beta C$ |
| `herk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)` | `cherk` `zherk` | $C \leftarrow \alpha A A^H + \beta C$ or $\alpha A^H A + \beta C$, real $\alpha, \beta$ |
| `syr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)` | `?syr2k` (s d c z) | $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$ (or transposed) |
| `her2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)` | `cher2k` `zher2k` | $C \leftarrow \alpha A B^H + \bar\alpha B A^H + \beta C$ (or transposed), real $\beta$ |
| `trmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)` | `?trmm` (s d c z) | $B \leftarrow \alpha\,\operatorname{op}(A) B$ or $\alpha B \operatorname{op}(A)$ |
| `trsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)` | `?trsm` (s d c z) | solve $\operatorname{op}(A) X = \alpha B$ or $X \operatorname{op}(A) = \alpha B$ |

A Level 3 call on fixed point looks exactly like the floating-point one:

```cpp
using q = thefblas::q15_sat;
// A = [[0.5, -0.25], [0.25, 0.5]], B = [[0.5, 0.5], [0.5, -0.5]], column-major.
const q a[] = {q(0.5), q(0.25), q(-0.25), q(0.5)};
const q b[] = {q(0.5), q(0.5), q(0.5), q(-0.5)};
q c[4];
// C = 0.5 * A * B; with beta = 0, C is not read.
thefblas::gemm('N', 'N', 2, 2, 2, q(0.5), a, 2, b, 2, q(0.0), c, 2);
// C = [[0.0625, 0.1875], [0.1875, -0.0625]], stored as {0.0625, 0.1875, 0.1875, -0.0625}.
```

---

## Choosing an element type

- **`float` / `double`**: results match Netlib BLAS (bit for bit in Level 1
  and Level 2, except `nrm2`; to rounding in Level 3).
- **`q15_sat`, `q31_sat`**: signals normalised to $[-1, 1)$, as in DSP code.
  The value 1 is not representable, so `alpha = 1` cannot be passed; scale the
  data instead, or use a format with an integer bit.
- **`fixed<std::int32_t, F, saturate>` with fewer fractional bits** (for
  example Q12.20 or Q16.16): general numerics where values above 1 occur.
  Pick the integer bits from the largest magnitude a *result* can reach; the
  accumulators take care of the intermediate sums (see
  [Choosing a Q format](math-reference.md#choosing-a-q-format)).
- **`checked`** while developing and testing, so that every overflow asserts;
  switch to `saturate` (or `wrap`, if modular results are wanted) for
  production.
- **`std::complex<fixed<...>>`** works with every complex routine. MSVC users
  define `_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING`.
