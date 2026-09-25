# Mathematical Reference

This page defines the mathematical operation performed by each Level-1 and
Level-2 BLAS routine in **thefblas**. Notation is introduced first, followed by
one section per operation family.

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
`thefblas::fixed<IntType, FracBits>` (for example `thefblas::fixed<std::int32_t, 20>`).

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

---

## nrm2 — Euclidean Norm

**Routines:** `snrm2`, `dnrm2`, `scnrm2`, `dznrm2`

Computes the $\ell_2$ norm of a vector:

$$\|x\|_2 = \sqrt{\sum_{i=1}^{n} \lvert x_i \rvert^2}$$

For complex vectors, $\lvert x_i \rvert^2 = \operatorname{Re}(x_i)^2 + \operatorname{Im}(x_i)^2$.

The result is always a non-negative real number, regardless of the input type
(hence the return type of `scnrm2` is `float` and `dznrm2` is `double`).

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

where $r = \lvert a \rvert / a \cdot \sqrt{\lvert a \rvert^2 + \lvert b \rvert^2}$
when $a \neq 0$, and $r = b$ when $a = 0$. The rotation matrix satisfies
$c^2 + \lvert s \rvert^2 = 1$.

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
constructs a modified Givens parameter array `param` such that applying the
encoded $H$ to the scaled pair eliminates the second component:

$$H \begin{pmatrix} \sqrt{d_1}\, b_1 \\ \sqrt{d_2}\, b_2 \end{pmatrix}
= \begin{pmatrix} r \\ 0 \end{pmatrix}$$

The scaling factors $d_1$ and $d_2$ are updated on output to absorb the
transformation, keeping elements of $H$ near unity for numerical stability.

> **Note.** The implementations of `srotmg` and `drotmg` in thefblas are
> intentionally simple reference implementations suitable for learning and
> testing. They do not include the rescaling guard present in production BLAS
> libraries.

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
