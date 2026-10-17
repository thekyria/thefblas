# Mathematical Reference

This page defines the mathematical operation performed by each Level-1,
Level-2 and Level-3 BLAS routine in **thefblas**. Notation is introduced first, followed by
one section per operation family, and a closing part on the fixed-point
arithmetic the routines use. How to call each routine is described in the
[Overview](overview.md).

Unlike reference BLAS, **thefblas** exposes a single, unprefixed function
*template* per operation (e.g. `axpy`, `gemv`) that is generic over its value
type. The classical `s`/`d`/`c`/`z`-prefixed routine names listed under each
section indicate which traditional BLAS routines the corresponding template
subsumes when instantiated with the matching element type.

---

## Notation

| Symbol | Meaning |
|---|---|
| $n \in \mathbb{Z}_{>0}$ | Number of logical vector elements |
| $\text{incx}, \text{incy} \in \mathbb{Z} \setminus \{0\}$ | Element stride (may be negative) |
| $x_i$ | The $i$-th logical element: $x[\,(i-1)\cdot\text{incx}\,]$ for $i = 1,\dots,n$ |
| $\alpha, \beta$ | Scalar values matching the vector precision |
| $\bar{z}$ | Complex conjugate of $z$ |
| $\lvert z \rvert$ | For real $z$: absolute value. For complex $z = a + bi$: $\sqrt{a^2+b^2}$ |
| $\lvert z \rvert_1$ | Manhattan modulus of complex $z = a + bi$: $\lvert a \rvert + \lvert b \rvert$ |

Precision variants follow standard BLAS prefixes; in **thefblas** they map onto
a single function template instantiated with the corresponding element type:

| Prefix | Element type (template argument) |
|---|---|
| `s` | `float` |
| `d` | `double` |
| `c` | `std::complex<float>` |
| `z` | `std::complex<double>` |

In addition, every routine may be instantiated with a fixed-point element type
`thefblas::fixed<IntType, FracBits, Policy>` (for example
`thefblas::fixed<std::int32_t, 20>`). For fixed-point instantiations the
formulas below describe the exact mathematical result; the computed result
differs from it by the quantization of the element format and, where the exact
result leaves the representable range, by the element type's overflow policy
(`checked`, `wrap` or `saturate`).

Internal reductions — the sums in `dot`, `dotu`, `dotc`, `nrm2`, `asum`, and
the row and column sums of the Level 2 and Level 3 routines — are accumulated
exactly in a type twice as wide as the element type, so partial sums neither
round nor overflow, and narrowed into the element format once, at the end. The
rank-1 and rank-2 updates (`ger*`, `syr`, `spr`, `her`, `hpr`, `syr2`, `spr2`,
`her2`, `hpr2`) form each element's products exactly as well, scale their sum
by $\alpha$ in the accumulator and round once per element. The exception is
the non-transposed vector solves (`trsv`, `tbsv`, `tpsv` with `trans` =
`'N'`), which keep Netlib's column order in the element type. The
**Fixed-Point Arithmetic** part at the end of this page covers the
representation, rounding, overflow and the error of each routine in detail.

---

## swap — Vector Swap

**Routines:** `sswap`, `dswap`, `cswap`, `zswap`

Exchanges two vectors element-wise:

$$x_i \leftrightarrow y_i, \quad i = 1, \dots, n$$

Both vectors are updated in place.

---

## copy — Vector Copy

**Routines:** `scopy`, `dcopy`, `ccopy`, `zcopy`

Copies one vector into another:

$$y_i \leftarrow x_i, \quad i = 1, \dots, n$$

The source vector $x$ is unchanged.

---

## axpy — Scalar-Vector Multiply-Add

**Routines:** `saxpy`, `daxpy`, `caxpy`, `zaxpy`

Computes a scaled vector addition:

$$y_i \leftarrow \alpha\, x_i + y_i, \quad i = 1, \dots, n$$

The scalar $\alpha$ and both vectors share the same type. When $\alpha = 0$,
the operation is a no-op (the reference implementation does not short-circuit
for $\alpha = 0$; the loop still executes).

---

## scal — Vector Scaling

**Routines:** `sscal`, `dscal`, `cscal`, `zscal`, `csscal`, `zdscal`

Multiplies every element of a vector by a scalar in place:

$$x_i \leftarrow \alpha\, x_i, \quad i = 1, \dots, n$$

The mixed-precision variants `csscal` and `zdscal` apply a **real** scalar to a
complex vector, scaling both real and imaginary components independently:

$$\operatorname{Re}(x_i) \leftarrow \alpha\operatorname{Re}(x_i), \quad
\operatorname{Im}(x_i) \leftarrow \alpha\operatorname{Im}(x_i)$$

This is distinct from `cscal`/`zscal`, where $\alpha \in \mathbb{C}$:

$$x_i \leftarrow (\alpha_r + i\,\alpha_i)(x_{i,r} + i\,x_{i,i})
= (\alpha_r x_{i,r} - \alpha_i x_{i,i}) + i\,(\alpha_r x_{i,i} + \alpha_i x_{i,r})$$

As in Netlib BLAS, a non-positive stride ($\text{incx} \le 0$) makes the call a
no-op.

---

## dot — Dot Product

**Routines:** `sdot`, `ddot`, `cdotu`, `cdotc`, `zdotu`, `zdotc`

### Real dot product

$$\text{dot}(x, y) = \sum_{i=1}^{n} x_i\, y_i$$

### Unconjugated complex dot product (`cdotu`, `zdotu`)

$$\text{dotu}(x, y) = \sum_{i=1}^{n} x_i\, y_i$$

Components are multiplied without conjugation:
$(a + bi)(c + di) = (ac - bd) + i(ad + bc)$.

### Conjugated complex dot product (`cdotc`, `zdotc`)

$$\text{dotc}(x, y) = \sum_{i=1}^{n} \bar{x}_i\, y_i$$

The **first** argument $x$ is conjugated. This computes the standard inner
product on $\mathbb{C}^n$, which satisfies $\langle x, x \rangle \geq 0$.

### Extended-precision dot products (`sdsdot`, `dsdot`)

$$\text{dsdot}(x, y) = \sum_{i=1}^{n} x_i\, y_i, \qquad
\text{sdsdot}(s_b, x, y) = s_b + \sum_{i=1}^{n} x_i\, y_i$$

The sum is formed in a wider type than the elements: `double` for `float`
input (as in Netlib BLAS), and for `fixed<I, F, P>` the fixed-point type with
twice the integer width and the same $F$. `dsdot` returns that wider value.
`sdsdot` sums $s_b$ together with the products and rounds the result to the
element type once; for fixed point, $s_b$ and the products are summed exactly
at $2F$ fractional bits, so $s_b + \sum x_i y_i$ is rounded exactly once. With
$n \le 0$, `dsdot` is $0$ and `sdsdot` returns $s_b$ unchanged. Fixed-point
types are accepted only where the running sum has an integer wider than the
products: 8- and 16-bit storage, and 32-bit storage with a 128-bit integer.

---

## nrm2 — Euclidean Norm

**Routines:** `snrm2`, `dnrm2`, `scnrm2`, `dznrm2`

Computes the $\ell_2$ norm of a vector:

$$\|x\|_2 = \sqrt{\sum_{i=1}^{n} \lvert x_i \rvert^2}$$

For complex vectors, $\lvert x_i \rvert^2 = \operatorname{Re}(x_i)^2 + \operatorname{Im}(x_i)^2$.

The result is always a non-negative real number, regardless of the input type
(hence the return type of `scnrm2` is `float` and `dznrm2` is `double`).

For `float`/`double` element types thefblas evaluates this with Blue's
algorithm (as Netlib reference BLAS 3.10+ does): mid-range $\lvert x_i \rvert$
are squared and summed directly, while very large and very small values are
first scaled by exact powers of two, so the computation neither overflows nor
underflows when $\|x\|_2$ is representable. Fixed-point types instead
accumulate the exact squares in a wide integer accumulator and take its
integer square root, so the norm is rounded once (to within half a unit in
the last place). A sum too large for the accumulator means a norm beyond the
range of the format; it is clamped to the largest value under every overflow
policy, and `checked` asserts.

---

## asum — Sum of Absolute Values

**Routines:** `sasum`, `dasum`, `scasum`, `dzasum`

### Real vectors

$$\text{asum}(x) = \sum_{i=1}^{n} \lvert x_i \rvert$$

### Complex vectors

$$\text{asum}(x) = \sum_{i=1}^{n} \lvert x_i \rvert_1
= \sum_{i=1}^{n} \bigl(\lvert \operatorname{Re}(x_i) \rvert + \lvert \operatorname{Im}(x_i) \rvert\bigr)$$

Note: the complex variant uses the **Manhattan modulus** $\lvert \cdot \rvert_1$,
not the Euclidean modulus. This matches the Netlib BLAS specification.

As in Netlib BLAS, the result is $0$ when $\text{incx} \le 0$.

---

## iamax — Index of Maximum Absolute Value

**Routines:** `isamax`, `idamax`, `icamax`, `izamax`

Returns the **1-based** index of the first element with the largest absolute value:

$$k = \operatorname*{argmax}_{i=1,\dots,n} \lvert x_i \rvert$$

with ties broken in favour of the first occurrence ($k$ is the smallest such index).

For complex vectors the same Manhattan modulus $\lvert \cdot \rvert_1$ used
by `asum` is applied:

$$\lvert x_i \rvert_1 = \lvert \operatorname{Re}(x_i) \rvert + \lvert \operatorname{Im}(x_i) \rvert$$

Returns $0$ when $n \leq 0$ or $\text{incx} \leq 0$.

> **Index convention.** The return value is 1-based to match Netlib BLAS
> semantics. Subtract 1 to convert to a C/C++ zero-based array index.

---

## rot — Givens Plane Rotation (Apply)

**Routines:** `srot`, `drot`, `csrot`, `zdrot`

Applies a Givens rotation matrix to paired elements of two vectors:

$$\begin{pmatrix} x_i \\ y_i \end{pmatrix}
\leftarrow
\begin{pmatrix} c & s \\ -s & c \end{pmatrix}
\begin{pmatrix} x_i \\ y_i \end{pmatrix}, \quad i = 1, \dots, n$$

expanded:

$$x_i \leftarrow c\, x_i + s\, y_i$$
$$y_i \leftarrow -s\, x_i + c\, y_i$$

For `csrot` and `zdrot` the rotation coefficients $c \in \mathbb{R}$ and
$s \in \mathbb{R}$ are real, but the vectors are complex. The rotation is
applied component-wise with real arithmetic on the complex elements, equivalent
to treating $c$ and $s$ as purely real scalars in the above formula.

---

## rotg — Givens Rotation Construction

**Routines:** `srotg`, `drotg`, `crotg`, `zrotg`

### Real case (`srotg`, `drotg`)

Given scalars $a, b$, computes $c$ and $s$ such that:

$$\begin{pmatrix} c & s \\ -s & c \end{pmatrix}
\begin{pmatrix} a \\ b \end{pmatrix}
=
\begin{pmatrix} r \\ 0 \end{pmatrix}$$

where $r = \sqrt{a^2 + b^2}$, $c = a/r$, $s = b/r$ (with $c = 1, s = 0$ when
$r = 0$). On output, $a$ is overwritten with $r$.

### Complex case (`crotg`, `zrotg`)

Given $a \in \mathbb{C}$ and $b \in \mathbb{C}$, computes real $c \in \mathbb{R}$
and $s \in \mathbb{C}$ such that:

$$\begin{pmatrix} c & s \\ -\bar{s} & c \end{pmatrix}
\begin{pmatrix} a \\ b \end{pmatrix}
=
\begin{pmatrix} r \\ 0 \end{pmatrix}$$

where $r = a / \lvert a \rvert \cdot \sqrt{\lvert a \rvert^2 + \lvert b \rvert^2}$
when $a \neq 0$. When $a = 0$, as in Netlib reference BLAS 3.10+, $c = 0$,
$s = \bar{b} / \lvert b \rvert$ and $r = \lvert b \rvert$ (real), or $c = 1$,
$s = 0$, $r = 0$ if $b = 0$ as well. The rotation matrix satisfies
$c^2 + \lvert s \rvert^2 = 1$.

For fixed-point types the sums of squares are formed exactly in the wide
accumulator and their square roots are rounded once, so $r$ (and in the
complex case $\lvert a \rvert$) needs none of the scaling that floating point
uses to avoid overflow. In the real case the returned $z$ is $1/c$ when
$\lvert a \rvert \le \lvert b \rvert$; for small $c$ it can exceed the range
of a format with few integer bits, and the overflow policy then applies.

---

## rotm — Modified Givens Rotation (Apply)

**Routines:** `srotm`, `drotm`

Applies a **modified Givens** transformation using a 5-element parameter array
`param`. The first element `param[0]` is a flag that selects the matrix form:

| `param[0]` | Matrix $H$ applied |
|---|---|
| $-1$ | $H = \begin{pmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{pmatrix}$ (general) |
| $0$ | $H = \begin{pmatrix} 1 & h_{12} \\ h_{21} & 1 \end{pmatrix}$ |
| $1$ | $H = \begin{pmatrix} h_{11} & 1 \\ -1 & h_{22} \end{pmatrix}$ |
| $-2$ | $H = I$ (identity, no-op) |

The transformation applied to each pair is:

$$\begin{pmatrix} x_i \\ y_i \end{pmatrix}
\leftarrow H
\begin{pmatrix} x_i \\ y_i \end{pmatrix}, \quad i = 1, \dots, n$$

---

## rotmg — Modified Givens Rotation Construction

**Routines:** `srotmg`, `drotmg`

Given diagonal scaling factors $d_1, d_2$ and vector components $b_1, b_2$,
constructs a modified Givens parameter array `param` such that the encoded
$H$ eliminates the second component, and updates $d_1$, $d_2$, $b_1$:

$$H \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}
= \begin{pmatrix} b_1' \\ 0 \end{pmatrix}, \qquad
d_1' b_1'^2 = d_1 b_1^2 + d_2 b_2^2$$

The algorithm is Netlib's: flag $0$ when $\lvert d_1 b_1^2 \rvert >
\lvert d_2 b_2^2 \rvert$, flag $1$ otherwise, flag $-2$ when $d_2 b_2 = 0$, and
flag $-1$ with $H$, $d_1$, $d_2$, $b_1$ all zero for invalid input ($d_1 < 0$,
or a negative $d_2$ that makes $d_2 b_2^2$ dominate). For `float` and `double`,
the magnitudes of nonzero $d_1$ and $d_2$ are then rescaled by powers of $4096$
into $[2^{-24}, 2^{24}]$ (a valid negative $d_2$ keeps its sign), switching $H$
to its full form (flag $-1$); fixed-point
types skip this step, since their range is bounded and the constants are not
representable in most Q formats.

> **Note.** When that rescaling runs, Netlib reference BLAS 3.12 resets
> $h_{21}$ and $h_{12}$ on every pass of its loop and can return an $H$ that no
> longer satisfies the equation above. thefblas switches to the full form only
> once, so its $H$ does; flag, $d_1$, $d_2$ and $b_1$ agree with Netlib.

---

# Level 2 BLAS — Matrix-Vector Operations

## Level 2 Notation

In addition to the Level 1 symbols, Level 2 routines use:

| Symbol | Meaning |
|---|---|
| $A$ | Matrix stored in **column-major** order |
| $\text{lda}$ | Leading dimension of $A$ (i.e., the declared first dimension of the 2-D array) |
| $\text{uplo}$ | `'U'` — upper triangle stored; `'L'` — lower triangle stored |
| $\text{trans}$ | `'N'` — no transpose; `'T'` — transpose; `'C'` — conjugate transpose |
| $\text{diag}$ | `'U'` — unit diagonal (diagonal elements assumed 1); `'N'` — non-unit diagonal |
| $\text{op}(A)$ | $A$ if trans=`'N'`; $A^{T}$ if trans=`'T'`; $A^{H}$ if trans=`'C'` |
| $k_l, k_u$ | Number of sub-diagonals and super-diagonals of a band matrix |

---

## gemv — General Matrix-Vector Multiply

**Routines:** `sgemv`, `dgemv`, `cgemv`, `zgemv`

$$y \leftarrow \alpha\,\text{op}(A)\,x + \beta\,y$$

---

## gbmv — General Band Matrix-Vector Multiply

**Routines:** `sgbmv`, `dgbmv`, `cgbmv`, `zgbmv`

$$y \leftarrow \alpha\,\text{op}(A)\,x + \beta\,y$$

where $A$ is a band matrix with $k_l$ sub-diagonals and $k_u$ super-diagonals.

---

## hemv — Hermitian Matrix-Vector Multiply

**Routines:** `chemv`, `zhemv`

$$y \leftarrow \alpha\,A\,x + \beta\,y$$

where $A$ is Hermitian ($A = A^H$).

---

## hbmv — Hermitian Band Matrix-Vector Multiply

**Routines:** `chbmv`, `zhbmv`

$$y \leftarrow \alpha\,A\,x + \beta\,y$$

where $A$ is Hermitian and banded.

---

## hpmv — Hermitian Packed Matrix-Vector Multiply

**Routines:** `chpmv`, `zhpmv`

$$y \leftarrow \alpha\,A\,x + \beta\,y$$

where $A$ is Hermitian, stored in packed format.

---

## symv — Symmetric Matrix-Vector Multiply

**Routines:** `ssymv`, `dsymv`

$$y \leftarrow \alpha\,A\,x + \beta\,y$$

where $A$ is symmetric ($A = A^T$).

---

## sbmv — Symmetric Band Matrix-Vector Multiply

**Routines:** `ssbmv`, `dsbmv`

$$y \leftarrow \alpha\,A\,x + \beta\,y$$

where $A$ is symmetric and banded.

---

## spmv — Symmetric Packed Matrix-Vector Multiply

**Routines:** `sspmv`, `dspmv`

$$y \leftarrow \alpha\,A\,x + \beta\,y$$

where $A$ is symmetric, stored in packed format.

---

## trmv — Triangular Matrix-Vector Multiply

**Routines:** `strmv`, `dtrmv`, `ctrmv`, `ztrmv`

$$x \leftarrow \text{op}(A)\,x$$

where $A$ is triangular.

---

## tbmv — Triangular Band Matrix-Vector Multiply

**Routines:** `stbmv`, `dtbmv`, `ctbmv`, `ztbmv`

$$x \leftarrow \text{op}(A)\,x$$

where $A$ is triangular and banded.

---

## tpmv — Triangular Packed Matrix-Vector Multiply

**Routines:** `stpmv`, `dtpmv`, `ctpmv`, `ztpmv`

$$x \leftarrow \text{op}(A)\,x$$

where $A$ is triangular, stored in packed format.

---

## trsv — Triangular Solve

**Routines:** `strsv`, `dtrsv`, `ctrsv`, `ztrsv`

Solves:

$$\text{op}(A)\,x = b$$

where $A$ is triangular. On entry $x$ holds $b$; on exit $x$ holds the solution.

---

## tbsv — Triangular Band Solve

**Routines:** `stbsv`, `dtbsv`, `ctbsv`, `ztbsv`

Solves:

$$\text{op}(A)\,x = b$$

where $A$ is triangular and banded.

---

## tpsv — Triangular Packed Solve

**Routines:** `stpsv`, `dtpsv`, `ctpsv`, `ztpsv`

Solves:

$$\text{op}(A)\,x = b$$

where $A$ is triangular, stored in packed format.

---

## ger — General Rank-1 Update (Real)

**Routines:** `sger`, `dger`

$$A \leftarrow \alpha\,x\,y^T + A$$

---

## geru — General Rank-1 Update (Complex, Unconjugated)

**Routines:** `cgeru`, `zgeru`

$$A \leftarrow \alpha\,x\,y^T + A$$

---

## gerc — General Rank-1 Update (Complex, Conjugated)

**Routines:** `cgerc`, `zgerc`

$$A \leftarrow \alpha\,x\,y^H + A$$

---

## her — Hermitian Rank-1 Update

**Routines:** `cher`, `zher`

$$A \leftarrow \alpha\,x\,x^H + A$$

where $A$ is Hermitian and $\alpha \in \mathbb{R}$.

---

## hpr — Hermitian Packed Rank-1 Update

**Routines:** `chpr`, `zhpr`

$$A \leftarrow \alpha\,x\,x^H + A$$

where $A$ is Hermitian, stored in packed format, and $\alpha \in \mathbb{R}$.

---

## her2 — Hermitian Rank-2 Update

**Routines:** `cher2`, `zher2`

$$A \leftarrow \alpha\,x\,y^H + \bar{\alpha}\,y\,x^H + A$$

where $A$ is Hermitian.

---

## hpr2 — Hermitian Packed Rank-2 Update

**Routines:** `chpr2`, `zhpr2`

$$A \leftarrow \alpha\,x\,y^H + \bar{\alpha}\,y\,x^H + A$$

where $A$ is Hermitian, stored in packed format.

---

## syr — Symmetric Rank-1 Update

**Routines:** `ssyr`, `dsyr`

$$A \leftarrow \alpha\,x\,x^T + A$$

where $A$ is symmetric.

---

## spr — Symmetric Packed Rank-1 Update

**Routines:** `sspr`, `dspr`

$$A \leftarrow \alpha\,x\,x^T + A$$

where $A$ is symmetric, stored in packed format.

---

## syr2 — Symmetric Rank-2 Update

**Routines:** `ssyr2`, `dsyr2`

$$A \leftarrow \alpha\,x\,y^T + \alpha\,y\,x^T + A$$

where $A$ is symmetric.

---

## spr2 — Symmetric Packed Rank-2 Update

**Routines:** `sspr2`, `dspr2`

$$A \leftarrow \alpha\,x\,y^T + \alpha\,y\,x^T + A$$

where $A$ is symmetric, stored in packed format.

---

# Level 3 BLAS — Matrix-Matrix Operations

Level 3 routines use the Level 2 notation, plus $B$ and $C$ (column-major, with
leading dimensions $\text{ldb}$ and $\text{ldc}$) and
$\operatorname{op}(X) \in \{X, X^T, X^H\}$ selected by a `trans` argument
`'N'`, `'T'` or `'C'` (for real types `'C'` means `'T'`). When $\alpha = 0$,
$A$ and $B$ are not read; when $\beta = 0$, $C$ is not read.

---

## gemm — General Matrix-Matrix Multiply

**Routines:** `sgemm`, `dgemm`, `cgemm`, `zgemm`

$$C \leftarrow \alpha\,\operatorname{op}(A)\,\operatorname{op}(B) + \beta\,C$$

with $\operatorname{op}(A)$ of size $m \times k$, $\operatorname{op}(B)$ of size
$k \times n$ and $C$ of size $m \times n$.

---

## symm — Symmetric Matrix-Matrix Multiply

**Routines:** `ssymm`, `dsymm`, `csymm`, `zsymm`

$$C \leftarrow \alpha\,A\,B + \beta\,C \quad (\text{side} = \texttt{'L'}), \qquad
C \leftarrow \alpha\,B\,A + \beta\,C \quad (\text{side} = \texttt{'R'})$$

where $B$ and $C$ are $m \times n$ and $A$ is symmetric ($m \times m$ for `'L'`,
$n \times n$ for `'R'`); only the `uplo` triangle of $A$ is read.

---

## hemm — Hermitian Matrix-Matrix Multiply

**Routines:** `chemm`, `zhemm`

As `symm`, for a Hermitian $A$: $a_{sr} = \overline{a_{rs}}$ for the triangle
that is not stored, and the imaginary parts of the diagonal are taken as zero
(not read).

---

## syrk — Symmetric Rank-k Update

**Routines:** `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`

$$C \leftarrow \alpha\,A\,A^T + \beta\,C \quad (\text{trans} = \texttt{'N'}), \qquad
C \leftarrow \alpha\,A^T A + \beta\,C \quad (\text{trans} = \texttt{'T'})$$

where $C$ is $n \times n$ symmetric and only its `uplo` triangle is updated. For
complex types there is no `'C'` form (that is `herk`).

---

## herk — Hermitian Rank-k Update

**Routines:** `cherk`, `zherk`

$$C \leftarrow \alpha\,A\,A^H + \beta\,C \quad (\text{trans} = \texttt{'N'}), \qquad
C \leftarrow \alpha\,A^H A + \beta\,C \quad (\text{trans} = \texttt{'C'})$$

with real $\alpha, \beta$. $C$ is $n \times n$ Hermitian; only its `uplo`
triangle is updated, and the imaginary parts of its diagonal are taken as zero
and set to zero. As in Netlib BLAS, the call returns without touching $C$ when
$\beta = 1$ and either $\alpha = 0$ or $k = 0$, so in that case the imaginary
parts of the diagonal are left as they are.

---

## syr2k — Symmetric Rank-2k Update

**Routines:** `ssyr2k`, `dsyr2k`, `csyr2k`, `zsyr2k`

$$C \leftarrow \alpha\,A\,B^T + \alpha\,B\,A^T + \beta\,C \quad (\text{trans} = \texttt{'N'}),$$
$$C \leftarrow \alpha\,A^T B + \alpha\,B^T A + \beta\,C \quad (\text{trans} = \texttt{'T'})$$

where $A$ and $B$ are $n \times k$ (`'N'`) or $k \times n$ (`'T'`), and only the
`uplo` triangle of the $n \times n$ symmetric $C$ is updated. As for `syrk`,
complex types have no `'C'` form (that is `her2k`).

---

## her2k — Hermitian Rank-2k Update

**Routines:** `cher2k`, `zher2k`

$$C \leftarrow \alpha\,A\,B^H + \bar\alpha\,B\,A^H + \beta\,C \quad (\text{trans} = \texttt{'N'}),$$
$$C \leftarrow \alpha\,A^H B + \bar\alpha\,B^H A + \beta\,C \quad (\text{trans} = \texttt{'C'})$$

with complex $\alpha$ and real $\beta$. As for `herk`, only the `uplo` triangle of
$C$ is updated, the imaginary parts of its diagonal are taken as zero and set
to zero, and the call returns without touching $C$ when $\beta = 1$ and either
$\alpha = 0$ or $k = 0$. The two products are accumulated separately, because
they are scaled by $\alpha$ and $\bar\alpha$, and combined before the single
narrowing.

---

## trmm — Triangular Matrix-Matrix Multiply

**Routines:** `strmm`, `dtrmm`, `ctrmm`, `ztrmm`

$$B \leftarrow \alpha\,\operatorname{op}(A)\,B \quad (\text{side} = \texttt{'L'}), \qquad
B \leftarrow \alpha\,B\,\operatorname{op}(A) \quad (\text{side} = \texttt{'R'})$$

where $B$ is $m \times n$ and $A$ is triangular ($m \times m$ for `'L'`,
$n \times n$ for `'R'`), with an implicit unit diagonal when `diag` is `'U'`.
$B$ is updated in place, each element computed before any element it depends
on is overwritten.

---

## trsm — Triangular Solve with Multiple Right-Hand Sides

**Routines:** `strsm`, `dtrsm`, `ctrsm`, `ztrsm`

Solves for $X$, which overwrites $B$ ($m \times n$):

$$\operatorname{op}(A)\,X = \alpha\,B \quad (\text{side} = \texttt{'L'}), \qquad
X\,\operatorname{op}(A) = \alpha\,B \quad (\text{side} = \texttt{'R'})$$

where $A$ is triangular ($m \times m$ for `'L'`, $n \times n$ for `'R'`), with an
implicit unit diagonal when `diag` is `'U'`. No test for singularity is made.

---

# Fixed-Point Arithmetic

The sections above give the exact result of every routine. This part explains
how `thefblas::fixed<IntType, FracBits, Policy>` approximates real numbers,
how its operations round and overflow, and how the routines arrange their
arithmetic so that each output element is rounded as few times as possible.
The [Overview](overview.md) lists the members of the type.

---

## Q-format representation

Let $b$ be the width of `IntType` in bits and $F$ = `FracBits`. A value of
type `fixed<IntType, F, Policy>` is an integer $r$ (its `raw()`) read as

$$x = r \cdot 2^{-F}, \qquad r \in [-2^{b-1},\; 2^{b-1} - 1] \cap \mathbb{Z}.$$

With $I = b - F$ bits for the sign and the integer part, the format is written
Q$I.F$, and

| Quantity | Value |
|---|---|
| Step (one unit in the last place) | $u = 2^{-F}$ |
| Smallest value, `lowest()` | $-2^{I-1}$ |
| Largest value, `max()` | $2^{I-1} - u$ |
| Values representable | $2^b$, evenly spaced |

| Format | Type | Range | $u$ |
|---|---|---|---|
| Q1.7 | `fixed<std::int8_t, 7>` | $[-1, 1)$ | $2^{-7} \approx 7.8 \times 10^{-3}$ |
| Q1.15 | `fixed<std::int16_t, 15>` | $[-1, 1)$ | $2^{-15} \approx 3.1 \times 10^{-5}$ |
| Q16.16 | `fixed<std::int32_t, 16>` | $[-32768, 32768)$ | $2^{-16} \approx 1.5 \times 10^{-5}$ |
| Q12.20 | `fixed<std::int32_t, 20>` | $[-2048, 2048)$ | $2^{-20} \approx 9.5 \times 10^{-7}$ |
| Q1.31 | `fixed<std::int32_t, 31>` | $[-1, 1)$ | $2^{-31} \approx 4.7 \times 10^{-10}$ |

Two properties set fixed point apart from floating point and drive the rest
of this part:

- **The grid is uniform.** Precision is *absolute*: every value carries an
  error of up to $u/2$ whatever its magnitude. Floating point keeps a
  constant *relative* precision instead. A small result therefore has a large
  relative error in fixed point, and a large one may not be representable at
  all.
- **The range is asymmetric and bounded.** $-2^{I-1}$ is representable but
  $+2^{I-1}$ is not, so negating or taking the absolute value of `lowest()`
  overflows. In the Q1.$F$ formats (`q7`, `q15`, `q31`) the value $1$ itself
  is not representable.

---

## Rounding

Every operation that drops fractional bits rounds to the nearest grid point,
and a tie (a value exactly halfway) goes away from zero:

$$\operatorname{rnd}(t) = \operatorname{sign}(t)\,u\,\Bigl\lfloor \frac{\lvert t \rvert}{u} + \tfrac12 \Bigr\rfloor,
\qquad \lvert \operatorname{rnd}(t) - t \rvert \le \tfrac{u}{2}.$$

On the integers this is a rounding right shift: an integer $v$ at $F + s$
fractional bits becomes $\operatorname{sign}(v)\,\lfloor (\lvert v \rvert + 2^{s-1}) / 2^s \rfloor$
at $F$ bits, computed on the magnitude so that it does not depend on how the
compiler shifts negative numbers.

Ties away from zero is symmetric, $\operatorname{rnd}(-t) = -\operatorname{rnd}(t)$,
but it is biased in magnitude: every tie grows $\lvert t \rvert$. In Q1.7,
$2^{-7} \cdot 0.5 = 2^{-8}$ is a tie and rounds to $2^{-7}$, and $-2^{-8}$
rounds to $-2^{-7}$. Summing four such products one by one gives $4u$, twice
the exact $2u$; `dot` rounds once, after the exact sum, and returns $2u$.
Avoiding repeated rounding is the main purpose of the accumulators described
below.

---

## Arithmetic operations

For $x = r_x u$ and $y = r_y u$ every operation is evaluated exactly in an
integer twice as wide as `IntType` (`__int128` for 64-bit storage), rounded
where stated, and then passed to the overflow policy $N_P$ (next section):

| Operation | Exact integer computation | Result | Rounding error |
|---|---|---|---|
| $x \pm y$ | $r_x \pm r_y$ | $N_P(r_x \pm r_y)\,u$ | none |
| $-x$, $\lvert x \rvert$ | $-r_x$, $\lvert r_x \rvert$ | through $N_P$ | none |
| $x \cdot y$ | $r_x r_y$ at $2F$ fractional bits, shifted right by $F$ | $N_P(\operatorname{rnd}(xy))$ | $\le u/2$ |
| $x / y$ | $r_x 2^F / r_y$, rounded to nearest | $N_P(\operatorname{rnd}(x/y))$ | $\le u/2$ |
| $\sqrt{x}$, $x \ge 0$ | $\sqrt{r_x 2^F}$, integer root rounded to nearest | $\operatorname{rnd}(\sqrt{x})$ | $\le u/2$ |

Notes:

- **Addition is exact**; only multiplication, division, square root and
  conversions round. A chain of additions can only overflow.
- **Products lose the low bits.** $r_x r_y$ has $2F$ fractional bits, and the
  shift keeps $F$ of them. A product smaller than $u/2$ in magnitude becomes
  zero: in Q1.15, $0.003 \times 0.003 \approx 9 \times 10^{-6}$ rounds to $0$.
- **Quotients overflow easily.** $\lvert x / y \rvert$ must be below
  $2^{I-1}$: in Q1.15, $0.25 / 0.5 = 0.5$ is fine, but $0.5 / 0.25 = 2$ is out
  of range ($0.99997$ under `saturate`). Division by zero is treated as an
  overflow towards the sign of the dividend.
- **Square root.** Since $\sqrt{x} = \sqrt{r_x 2^F}\,u$, a single integer
  square root of the exact radicand gives the result, with no Newton
  iteration and no intermediate value that has to be representable; ties
  cannot occur because $(k + \tfrac12)^2$ is never an integer. The result
  never exceeds the range. In Q1.15, $\sqrt{2^{-15}}$ has raw value
  $\operatorname{rnd}(\sqrt{2^{15}}) = 181$.
- **Conversions.** From `double`, $v$ becomes $\operatorname{rnd}(v)$, clamped
  to the range under every policy (NaN becomes $0$). Between Q formats, the
  raw value is shifted left (exact) or right with rounding, then passed to the
  target policy. The conversion is exact when the target has at least as many
  integer bits and at least as many fractional bits as the source.
- **Complex values** are pairs of `fixed` values, and
  $(a + bi)(c + di) = (ac - bd) + i(ad + bc)$ is evaluated with two rounded
  products per part, so each part of a complex product is within $u$.
  Complex division in the solves uses Smith's algorithm, which divides by the
  larger part of the denominator first so that intermediate values stay
  small.

---

## Overflow policies

Let $r_{\min} = -2^{b-1}$ and $r_{\max} = 2^{b-1} - 1$. The policy is the map
$N_P$ from the exact wide integer result $v$ back to the raw range:

$$N_{\text{wrap}}(v) = \bigl((v - r_{\min}) \bmod 2^b\bigr) + r_{\min}, \qquad
N_{\text{saturate}}(v) = \min\bigl(r_{\max}, \max(r_{\min}, v)\bigr),$$

and `checked` asserts $r_{\min} \le v \le r_{\max}$, then behaves like `wrap`
(with `NDEBUG`). None of them has undefined behaviour.

- **`wrap` is modular arithmetic.** Addition and subtraction become
  operations in $\mathbb{Z}/2^b\mathbb{Z}$, which are associative and exact
  modulo $2^b$. A sum or difference of any number of terms is therefore
  correct whenever the *final* value is in range, even if partial sums
  overflowed on the way. This does not extend to products, whose rounding
  shift does not commute with the reduction.
- **`saturate` is monotone and sign preserving**: a result too large stays at
  the largest value instead of changing sign, which is why it suits signal
  processing and control. It is **not associative**: in Q1.7,
  $(0.78125 + 0.78125) - 0.78125 = 0.2109375$ (the first sum clamps to
  $0.9921875$), whereas $0.78125 + (0.78125 - 0.78125) = 0.78125$. Neither
  policy is distributive. The result of a saturating computation depends on
  its evaluation order, which is why thefblas fixes that order and keeps the
  intermediate values out of the element format.
- Every overflow is a loss of information. The policy decides only which
  wrong answer is returned; choosing a format with enough integer bits is
  what makes the result right.

---

## Wide accumulators

A reduction such as $S = \sum_{i=1}^n x_i y_i$ fails in a naive fixed-point
loop in two ways: each product is rounded, so the errors add up to $n u / 2$
and, with ties away from zero, drift in one direction; and a partial sum can
leave the range even when $S$ does not. With `saturate` in Q1.15,
$x = (0.9, 0.9, -0.9)$ and $y = (0.9, 0.9, 0.9)$ give $S = 0.81$, but the
naive loop clamps $0.81 + 0.81$ to $0.99997$ and returns $0.19$.

thefblas forms every reduction in integers instead:

1. Each product $r_{x_i} r_{y_i}$ is kept **unshifted**, as an exact integer at
   $2F$ fractional bits.
2. The products are summed in an integer of width $b_{\text{acc}}$:

   | Element storage | Products | Running sum | Exact for at least $n$ terms |
   |---|---|---|---|
   | 8-bit | 16-bit | 32-bit | $2^{17}$ |
   | 16-bit | 32-bit | 64-bit | $2^{33}$ |
   | 32-bit, with `__int128` | 64-bit | 128-bit | $2^{65}$ |
   | 32-bit, without `__int128` | 64-bit | 64-bit | while $\lvert S \rvert < 2^{63 - 2F}$ |
   | 64-bit (needs `__int128`) | 128-bit | 128-bit | while $\lvert S \rvert < 2^{127 - 2F}$ |

   Each product is at most $2^{2(b-1)}$ in magnitude, so the running sum
   cannot overflow for $n \le 2^{b_{\text{acc}} - 2b + 1}$, more terms than an
   `int` can count for the first three rows. In the last two rows the running
   sum wraps modulo $2^{b_{\text{acc}}}$ (well defined, never undefined
   behaviour). Because the additions are modular, partial sums may wrap as
   long as the final sum fits; a final sum beyond $2^{b_{\text{acc}} - 1}$ raw
   units is not reliably detected.
3. The exact sum is rounded **once** to $F$ fractional bits and passed through
   the element policy into the accumulator format `fixed<wider IntType, F,
   Policy>`, which has $b$ more integer bits: Q1.15 is accumulated as Q17.15,
   so the range of the intermediate result grows from $[-1, 1)$ to
   $[-65536, 65536)$. Where no wider `fixed` storage exists (64-bit storage,
   or 32-bit without `__int128`), this step keeps the element width.
4. Any scaling by $\alpha$ and $\beta$ happens in the accumulator format, and
   the final value is narrowed into the element format by the policy, once.

The computed `dot` is therefore $N_P(\operatorname{rnd}(S))$: when $S$ is in
range, its error is at most $u/2$ **independently of $n$**, and partial sums
never overflow. In the example above it returns $0.81$. Compare floating
point, where the usual bound for a recursive dot product is
$\lvert \hat S - S \rvert \le n\,\varepsilon \sum \lvert x_i y_i \rvert$
to first order: relative to the terms and growing with $n$, but with no range
problem.

Sums of squares (`nrm2`, `rotg`, $\lvert z \rvert$ for complex $z$) use the
same exact accumulation in an unsigned integer, so overflow can be detected;
the integer square root of the exact sum at $2F$ fractional bits is the raw
result at $F$ bits, rounded once. A sum that overflows the accumulator means
a norm far beyond the range of the format; it is clamped to `max()` under
every policy, and `checked` asserts.

---

## Rounding error by routine

The bounds below are for real fixed-point data, assume that no value
overflows, and are in units of $u$; for complex data, each real part of a
complex product rounds separately, roughly doubling the product terms.
"Rounded sum" stands for $\operatorname{rnd}(S)$ of an exactly accumulated
reduction.

| Routines | How the result is formed | Error bound |
|---|---|---|
| `copy`, `swap` | Moves values | exact |
| `scal`, `axpy` | One rounded product per element ($\alpha x_i$), exact addition | $u/2$ |
| `rot`, `rotm` | Two rounded products per element ($c x_i$, $s y_i$), exact addition | $u$ |
| `dot`, `dotu`, `dotc`, `asum`, `dsdot`, `sdsdot` | Rounded sum | $u/2$ |
| `nrm2` | Rounded square root of the exact sum of squares | $u/2$ |
| `trmv`, `tbmv`, `tpmv` | Rounded sum per element | $u/2$ |
| `gemv`, `gbmv`, `symv`, `sbmv`, `spmv`, `hemv`, `hbmv`, `hpmv` | $N_P\bigl(\operatorname{rnd}(\beta y_i) + \operatorname{rnd}(\alpha \operatorname{rnd}(S_i))\bigr)$ | $\tfrac12(1 + \lvert\alpha\rvert) + \tfrac12$ |
| `ger`, `syr`, `syr2`, `spr`, `spr2`, `her`, `hpr`, `her2`, `hpr2` | $N_P\bigl(a_{ij} + \operatorname{rnd}(\alpha \operatorname{rnd}(S_{ij}))\bigr)$, $S_{ij}$ the exact product(s) | $\tfrac12(1 + \lvert\alpha\rvert)$ |
| `gemm`, `symm`, `hemm`, `syrk`, `herk` | As `gemv`, per element of the result | $\tfrac12(1 + \lvert\alpha\rvert) + \tfrac12$ |
| `syr2k`, `her2k` | Two rounded sums, each scaled by $\alpha$ (or $\bar\alpha$), plus $\operatorname{rnd}(\beta c_{ij})$ | $1 + \lvert\alpha\rvert + \tfrac12$ |
| `trmm` | $N_P(\operatorname{rnd}(\alpha \operatorname{rnd}(S_{ij})))$ | $\tfrac12(1 + \lvert\alpha\rvert)$ |
| `trsm`; `trsv`, `tbsv`, `tpsv` with `trans` = `'T'`/`'C'` | Per element: rounded sum (in `trsm` including the exact products $\alpha b_{ij}$), divided by the diagonal in the accumulator, narrowed once | two roundings per element, plus the error of the earlier elements |
| `trsv`, `tbsv`, `tpsv` with `trans` = `'N'` | Netlib's column order in the element type: $x_i \leftarrow x_i - x_j a_{ij}$, rounded and narrowed at every step | up to $n - 1$ roundings per element, plus propagated error |

Notes:

- The $\beta$ term vanishes when $\beta$ is $0$ or exactly $1$, and the
  $\alpha$ rounding vanishes when $\alpha$ is exactly $1$ (multiplying by $1$
  is exact), so `gemv` with $\alpha = 1$, $\beta = 0$ rounds once.
- In the solves the error of each computed $x_j$ is carried into the later
  elements through $a_{ij}$ and amplified by the division by $a_{ii}$, so the
  total error depends on the conditioning of the triangular matrix. For
  well-conditioned matrices (dominant diagonal) it stays within a few $u$.
  No test for singularity is made: a zero diagonal is a division by zero and
  goes through the policy.
- In the `trans` = `'N'` vector solves the partial values of $x_i$ live in the
  element format, so they must stay within its range; the transposed forms and
  `trsm` keep them in the accumulator.
- For `float` and `double` the floating-point paths are unchanged and match
  Netlib (bit for bit in Level 1 and Level 2, except `nrm2`).

---

## Formats without integer bits

`q7`, `q15` and `q31` (and every Q1.$F$ format) cover $[-1, 1)$ and cannot
represent $1$. This is the usual convention for normalised signals, and it has
some consequences for BLAS routines, whose definitions are full of ones:

- **$\alpha = 1$ cannot be passed.** The largest scalar is $1 - u$, so
  `gemv('N', m, n, max(), ...)` computes $(1 - u) A x$, which differs from
  $Ax$ by at most $u \lvert Ax \rvert$. Comparisons with $1$ inside the
  routines never construct $1$, so they work in these formats.
- **Unit diagonals** (`diag` = `'U'`) are implicit, so the triangular
  routines work, but a stored diagonal of ones is not possible.
- **Rotations.** `rotg` returns $c = 1$ for $b = 0$, and Netlib's $z = 1/c$
  when $\lvert a \rvert \le \lvert b \rvert$, which is at least $1$; both go
  through the policy. `rotm` and `rotmg` store the flags $-2$ and $1$, so they
  are rejected at compile time in these formats.
- **Norms of complex vectors.** Each part is in $[-1, 1)$, but
  $\lvert z \rvert$ can reach $\sqrt2$, and $\lVert x \rVert_2$ grows with
  $\sqrt n$; such results overflow even though every input is valid.

---

## Choosing a Q format

The accumulators protect intermediate sums, but every *result* must fit the
element format. With $b$ fixed by the storage type, the choice of $F$ trades
range for precision:

1. **Range.** Bound the largest result and give it enough integer bits:
   $2^{I-1} > \max \lvert \text{result} \rvert$. For example
   $\lvert x^T y \rvert \le n \max\lvert x_i\rvert \max\lvert y_i\rvert$, and an
   element of $\alpha A x + \beta y$ is at most
   $\lvert\alpha\rvert \max_i \sum_j \lvert a_{ij} \rvert \lvert x_j \rvert + \lvert\beta\rvert \lvert y_i \rvert$.
   A dot product of 64 elements of magnitude at most 1 can reach 64, which
   needs $I \ge 8$ (Q8.8 in 16 bits, Q8.24 in 32 bits).
2. **Precision.** The absolute error of each result is a small multiple of
   $u = 2^{-F}$ (table above), so $F$ must make $u$ small against the smallest
   result that matters.
3. **Scaling.** If both cannot be met, scale the data. Powers of two from $u$
   up to the range are exact, so $\alpha = 2^{-k}$ divides a result by $2^k$ losing only
   the $k$ bits shifted out (rounded once), and `fixed_cast` moves values
   between formats with the same rounding.
4. **Policy.** Develop with `checked`, which asserts on the first overflow and
   so shows whether the format is large enough; ship with `saturate`, which
   degrades gracefully when it is not.
