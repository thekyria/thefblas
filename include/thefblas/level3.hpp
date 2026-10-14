// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "thefblas/detail.hpp"
#include "thefblas/fixed.hpp"
#include "thefblas/level2.hpp"

#include <complex>
#include <type_traits>

namespace thefblas {

/**
 * @file level3.hpp
 * @brief Header-only generic Level-3 BLAS-style matrix-matrix routines.
 *
 * Implemented: `gemm`, `symm`, `hemm`, `syrk`, `herk`, `syr2k`, `her2k`, `trmm`
 * and `trsm`.
 *
 * Matrices use column-major (Fortran) layout: `a(i,j)` resides at offset
 * `i + j * lda`. `op(X)` is `X`, `X^T` or `X^H` for `trans` `'N'`, `'T'` or
 * `'C'`; for real element types `'C'` means `'T'`.
 *
 * Parameter conventions and argument checks follow Netlib BLAS: an invalid
 * character option, a negative size, or a leading dimension smaller than the
 * number of rows of the matrix as stored returns early without touching any
 * operand, as do `m == 0` / `n == 0`. When `alpha` is zero, `a` and `b` are not
 * read; when `beta` is zero, `c` is not read.
 *
 * Every output element is formed as a single widened multiply-accumulate over
 * its whole inner dimension, with the `alpha` scaling, the `beta` scaling of
 * the old element and their sum held in the accumulator type, and is narrowed
 * through the element type's overflow policy exactly once (see README,
 * "Accumulators"). No temporary buffers are used. For `float` and `double`
 * this sums each element in a fixed order that differs from Netlib's loop
 * order, so results agree with Netlib to rounding rather than bit for bit.
 */

namespace detail {

inline bool valid_side(char value) {
    const char upper = to_upper(value);
    return upper == 'L' || upper == 'R';
}

constexpr int max_one(int value) {
    return value > 1 ? value : 1;
}

/// `sum += op(a) * op(b)`, conjugating each operand whose flag is set. The
/// flags are only ever set for complex element types.
template <typename Mac, typename T>
inline void add_op_product(Mac &sum, const T &a, bool conj_a, const T &b, bool conj_b) {
    if constexpr (is_complex_v<T>) {
        if (conj_a && conj_b) {
            sum.add_conj_conj_product(a, b);
            return;
        }
        if (conj_b) {
            sum.add_conj_product(b, a);
            return;
        }
    }
    if (conj_a) {
        sum.add_conj_product(a, b);
    } else {
        sum.add_product(a, b);
    }
}

template <typename T>
inline void gemm_impl(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda,
                      const T *b, int ldb, T beta, T *c, int ldc) {
    const char ta = to_upper(transa);
    const char tb = to_upper(transb);
    const int nrowa = (ta == 'N') ? m : k;
    const int nrowb = (tb == 'N') ? k : n;
    if (!valid_trans(transa) || !valid_trans(transb) || m <= 0 || n <= 0 || k < 0 ||
        lda < max_one(nrowa) || ldb < max_one(nrowb) || ldc < m) {
        return;
    }
    const T zero = value_constants<T>::zero();
    if ((alpha == zero || k == 0) && beta == value_constants<T>::one()) {
        return;
    }
    if (alpha == zero) {
        for (int j = 0; j < n; ++j) {
            scale_vector(m, beta, c + j * ldc, 1);
        }
        return;
    }

    const bool conj_a = is_complex_v<T> && ta == 'C';
    const bool conj_b = is_complex_v<T> && tb == 'C';
    // op(A)(i, l) = a[i * a_row + l * a_inner], op(B)(l, j) = b[l * b_inner + j * b_col].
    const int a_row = (ta == 'N') ? 1 : lda;
    const int a_inner = (ta == 'N') ? lda : 1;
    const int b_inner = (tb == 'N') ? 1 : ldb;
    const int b_col = (tb == 'N') ? ldb : 1;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mac_t<T> sum;
            for (int l = 0; l < k; ++l) {
                add_op_product(sum, a[i * a_row + l * a_inner], conj_a, b[l * b_inner + j * b_col],
                               conj_b);
            }
            T &cij = c[i + j * ldc];
            cij = acc_scale_add_and_narrow(beta, cij, alpha, sum.wide_value());
        }
    }
}

/// The value of a diagonal element of a Hermitian matrix (its real part) when
/// `Hermitian` is set; the value itself otherwise.
template <bool Hermitian, typename T> inline T hermitian_diag(const T &value) {
    if constexpr (Hermitian) {
        return T(value.real(), typename T::value_type{});
    } else {
        return value;
    }
}

/// `C <- alpha * op(A) * op(A)^T + beta * C` (or, with `Hermitian`,
/// `op(A) * op(A)^H`) on the `uplo` triangle of the `n x n` matrix `C`. For
/// `Hermitian`, `alpha` and `beta` are real (imaginary part zero), the
/// imaginary part of the diagonal of `C` is not read and is set to zero.
template <bool Hermitian, typename T>
inline void rank_k_update(char uplo, char trans, int n, int k, T alpha, const T *a, int lda, T beta,
                          T *c, int ldc) {
    const bool upper = to_upper(uplo) == 'U';
    const bool no_trans = to_upper(trans) == 'N';
    const T zero = value_constants<T>::zero();
    if ((alpha == zero || k == 0) && beta == value_constants<T>::one()) {
        return;
    }

    // With alpha == 0 the sums are empty, a is not read and C <- beta * C.
    const int k_used = (alpha == zero) ? 0 : k;

    // u(i, l) = a[i * a_row + l * a_inner] is op(A)(i, l). The update is
    // sum_l u(i, l) * u(j, l); Hermitian conjugates u(j, l) for 'N' and
    // u(i, l) for 'C', both giving the (i, j) element of op(A) * op(A)^H.
    const int a_row = no_trans ? 1 : lda;
    const int a_inner = no_trans ? lda : 1;
    const bool conj_i = Hermitian && !no_trans;
    const bool conj_j = Hermitian && no_trans;
    for (int j = 0; j < n; ++j) {
        const int first = upper ? 0 : j;
        const int last = upper ? j + 1 : n;
        for (int i = first; i < last; ++i) {
            mac_t<T> sum;
            for (int l = 0; l < k_used; ++l) {
                add_op_product(sum, a[i * a_row + l * a_inner], conj_i, a[j * a_row + l * a_inner],
                               conj_j);
            }
            T &cij = c[i + j * ldc];
            if (i == j) {
                // c is not read when beta is zero.
                const T old_diag = (beta == zero) ? zero : hermitian_diag<Hermitian>(cij);
                cij = hermitian_diag<Hermitian>(
                    acc_scale_add_and_narrow(beta, old_diag, alpha, sum.wide_value()));
            } else {
                cij = acc_scale_add_and_narrow(beta, cij, alpha, sum.wide_value());
            }
        }
    }
}

template <typename T>
inline void syrk_impl(char uplo, char trans, int n, int k, T alpha, const T *a, int lda, T beta,
                      T *c, int ldc) {
    const char tr = to_upper(trans);
    // Complex syrk has no conjugate-transpose form; real syrk treats 'C' as 'T'.
    const bool trans_ok = tr == 'N' || tr == 'T' || (!is_complex_v<T> && tr == 'C');
    const int nrowa = (tr == 'N') ? n : k;
    if (!valid_uplo(uplo) || !trans_ok || n <= 0 || k < 0 || lda < max_one(nrowa) || ldc < n) {
        return;
    }
    rank_k_update<false>(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <typename T>
inline void herk_impl(char uplo, char trans, int n, int k, T alpha, const std::complex<T> *a,
                      int lda, T beta, std::complex<T> *c, int ldc) {
    using C = std::complex<T>;
    const char tr = to_upper(trans);
    const int nrowa = (tr == 'N') ? n : k;
    if (!valid_uplo(uplo) || (tr != 'N' && tr != 'C') || n <= 0 || k < 0 || lda < max_one(nrowa) ||
        ldc < n) {
        return;
    }
    rank_k_update<true>(uplo, trans, n, k, C(alpha, T{}), a, lda, C(beta, T{}), c, ldc);
}

/// `C <- alpha * A * B + beta * C` (`side == 'L'`, `A` is `m x m`) or
/// `C <- alpha * B * A + beta * C` (`'R'`, `A` is `n x n`) for a symmetric or,
/// with `Hermitian`, Hermitian `A` of which only the `uplo` triangle is read.
/// For `Hermitian` the imaginary parts of the diagonal of `A` are not read.
template <bool Hermitian, typename T>
inline void symm_like(char side, char uplo, int m, int n, T alpha, const T *a, int lda, const T *b,
                      int ldb, T beta, T *c, int ldc) {
    const bool left = to_upper(side) == 'L';
    const int nrowa = left ? m : n;
    if (!valid_side(side) || !valid_uplo(uplo) || m <= 0 || n <= 0 || lda < nrowa || ldb < m ||
        ldc < m) {
        return;
    }
    const T zero = value_constants<T>::zero();
    if (alpha == zero && beta == value_constants<T>::one()) {
        return;
    }
    if (alpha == zero) {
        for (int j = 0; j < n; ++j) {
            scale_vector(m, beta, c + j * ldc, 1);
        }
        return;
    }

    const bool upper = to_upper(uplo) == 'U';
    // sum += A(r, s) * x, reading A(r, s) from the stored triangle: as is, or
    // transposed (and, for Hermitian, conjugated) from (s, r).
    const auto add_a_times = [&](mac_t<T> &sum, int r, int s, const T &x) {
        if (r == s) {
            sum.add_product(hermitian_diag<Hermitian>(a[r + r * lda]), x);
        } else if ((r < s) == upper) {
            sum.add_product(a[r + s * lda], x);
        } else {
            add_op_product(sum, a[s + r * lda], Hermitian, x, false);
        }
    };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mac_t<T> sum;
            if (left) {
                for (int l = 0; l < m; ++l) {
                    add_a_times(sum, i, l, b[l + j * ldb]);
                }
            } else {
                for (int l = 0; l < n; ++l) {
                    add_a_times(sum, l, j, b[i + l * ldb]);
                }
            }
            T &cij = c[i + j * ldc];
            cij = acc_scale_add_and_narrow(beta, cij, alpha, sum.wide_value());
        }
    }
}

/// `C <- alpha * op(A) * op(B)^T + alpha * op(B) * op(A)^T + beta * C` (or,
/// with `Hermitian`, `alpha * op(A) * op(B)^H + conj(alpha) * op(B) * op(A)^H`)
/// on the `uplo` triangle of the `n x n` matrix `C`. For `Hermitian`, `beta`
/// is real (imaginary part zero), the imaginary part of the diagonal of `C` is
/// not read and is set to zero.
template <bool Hermitian, typename T>
inline void rank_2k_update(char uplo, char trans, int n, int k, T alpha, const T *a, int lda,
                           const T *b, int ldb, T beta, T *c, int ldc) {
    const bool upper = to_upper(uplo) == 'U';
    const bool no_trans = to_upper(trans) == 'N';
    const T zero = value_constants<T>::zero();
    if ((alpha == zero || k == 0) && beta == value_constants<T>::one()) {
        return;
    }
    // With alpha == 0 the sums are empty, a and b are not read and C <- beta * C.
    const int k_used = (alpha == zero) ? 0 : k;

    // u(i, l) = x[i * row + l * inner] is op(X)(i, l) for X = A or B. The two
    // terms are sum_l uA(i, l) * uB(j, l) and sum_l uB(i, l) * uA(j, l), with the
    // same conjugation as in rank_k_update. They are kept in separate
    // accumulators because Hermitian scales them by alpha and conj(alpha).
    const int a_row = no_trans ? 1 : lda;
    const int a_inner = no_trans ? lda : 1;
    const int b_row = no_trans ? 1 : ldb;
    const int b_inner = no_trans ? ldb : 1;
    const bool conj_i = Hermitian && !no_trans;
    const bool conj_j = Hermitian && no_trans;
    const T alpha2 = Hermitian ? conj_value(alpha) : alpha;
    for (int j = 0; j < n; ++j) {
        const int first = upper ? 0 : j;
        const int last = upper ? j + 1 : n;
        for (int i = first; i < last; ++i) {
            mac_t<T> ab;
            mac_t<T> ba;
            for (int l = 0; l < k_used; ++l) {
                add_op_product(ab, a[i * a_row + l * a_inner], conj_i, b[j * b_row + l * b_inner],
                               conj_j);
                add_op_product(ba, b[i * b_row + l * b_inner], conj_i, a[j * a_row + l * a_inner],
                               conj_j);
            }
            T &cij = c[i + j * ldc];
            T old = zero; // c is not read when beta is zero
            if (beta != zero) {
                old = (i == j) ? hermitian_diag<Hermitian>(cij) : cij;
            }
            const T result = acc_scale_add2_and_narrow(beta, old, alpha, ab.wide_value(), alpha2,
                                                       ba.wide_value());
            cij = (i == j) ? hermitian_diag<Hermitian>(result) : result;
        }
    }
}

template <typename T>
inline void syr2k_impl(char uplo, char trans, int n, int k, T alpha, const T *a, int lda,
                       const T *b, int ldb, T beta, T *c, int ldc) {
    const char tr = to_upper(trans);
    // As for syrk: complex syr2k has no conjugate-transpose form.
    const bool trans_ok = tr == 'N' || tr == 'T' || (!is_complex_v<T> && tr == 'C');
    const int nrowa = (tr == 'N') ? n : k;
    if (!valid_uplo(uplo) || !trans_ok || n <= 0 || k < 0 || lda < max_one(nrowa) ||
        ldb < max_one(nrowa) || ldc < n) {
        return;
    }
    rank_2k_update<false>(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename T>
inline void her2k_impl(char uplo, char trans, int n, int k, std::complex<T> alpha,
                       const std::complex<T> *a, int lda, const std::complex<T> *b, int ldb, T beta,
                       std::complex<T> *c, int ldc) {
    using C = std::complex<T>;
    const char tr = to_upper(trans);
    const int nrowa = (tr == 'N') ? n : k;
    if (!valid_uplo(uplo) || (tr != 'N' && tr != 'C') || n <= 0 || k < 0 || lda < max_one(nrowa) ||
        ldb < max_one(nrowa) || ldc < n) {
        return;
    }
    rank_2k_update<true>(uplo, trans, n, k, alpha, a, lda, b, ldb, C(beta, T{}), c, ldc);
}

template <typename T>
inline void trmm_impl(char side, char uplo, char transa, char diag, int m, int n, T alpha,
                      const T *a, int lda, T *b, int ldb) {
    const bool left = to_upper(side) == 'L';
    const char tr = to_upper(transa);
    const int nrowa = left ? m : n;
    if (!valid_side(side) || !valid_uplo(uplo) || !valid_trans(transa) || !valid_diag(diag) ||
        m <= 0 || n <= 0 || lda < nrowa || ldb < m) {
        return;
    }
    const T zero = value_constants<T>::zero();
    if (alpha == zero) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                b[i + j * ldb] = zero;
            }
        }
        return;
    }

    const bool unit = to_upper(diag) == 'U';
    const bool conj = is_complex_v<T> && tr == 'C';
    // As in trsm: op(A)(r, s) = a[r * a_row + s * a_col], upper triangular when
    // op_upper is set.
    const bool op_upper = (to_upper(uplo) == 'U') == (tr == 'N');
    const int a_row = (tr == 'N') ? 1 : lda;
    const int a_col = (tr == 'N') ? lda : 1;

    // sum += op(A)(d, d) * x; an implicit unit diagonal is not read.
    const auto add_diag = [&](mac_t<T> &sum, int d, const T &x) {
        if (unit) {
            sum.add(x);
        } else {
            add_op_product(sum, a[d + d * lda], conj, x, false);
        }
    };

    // B is overwritten in place, so each element is computed before any
    // element it depends on is replaced: an upper op(A) needs the elements at
    // and after it (rows going down on the left, columns going left on the
    // right), a lower one those at and before it.
    if (left) {
        // B(:, j) <- alpha * op(A) * B(:, j)
        for (int j = 0; j < n; ++j) {
            T *col = b + j * ldb;
            for (int step = 0; step < m; ++step) {
                const int i = op_upper ? step : m - 1 - step;
                mac_t<T> sum;
                add_diag(sum, i, col[i]);
                const int first = op_upper ? i + 1 : 0;
                const int last = op_upper ? m : i;
                for (int l = first; l < last; ++l) {
                    add_op_product(sum, a[i * a_row + l * a_col], conj, col[l], false);
                }
                col[i] = acc_scale_add_and_narrow(zero, zero, alpha, sum.wide_value());
            }
        }
    } else {
        // B(i, :) <- alpha * B(i, :) * op(A)
        for (int i = 0; i < m; ++i) {
            for (int step = 0; step < n; ++step) {
                const int j = op_upper ? n - 1 - step : step;
                mac_t<T> sum;
                add_diag(sum, j, b[i + j * ldb]);
                const int first = op_upper ? 0 : j + 1;
                const int last = op_upper ? j : n;
                for (int l = first; l < last; ++l) {
                    add_op_product(sum, a[l * a_row + j * a_col], conj, b[i + l * ldb], false);
                }
                b[i + j * ldb] = acc_scale_add_and_narrow(zero, zero, alpha, sum.wide_value());
            }
        }
    }
}

template <typename T>
inline void trsm_impl(char side, char uplo, char transa, char diag, int m, int n, T alpha,
                      const T *a, int lda, T *b, int ldb) {
    const bool left = to_upper(side) == 'L';
    const char tr = to_upper(transa);
    const int nrowa = left ? m : n;
    if (!valid_side(side) || !valid_uplo(uplo) || !valid_trans(transa) || !valid_diag(diag) ||
        m <= 0 || n <= 0 || lda < nrowa || ldb < m) {
        return;
    }
    const T zero = value_constants<T>::zero();
    if (alpha == zero) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                b[i + j * ldb] = zero;
            }
        }
        return;
    }

    const bool unit = to_upper(diag) == 'U';
    const bool conj = is_complex_v<T> && tr == 'C';
    // op(A) is upper triangular for an upper A that is not transposed, or a
    // lower A that is. op(A)(r, s) = a[r * a_row + s * a_col].
    const bool op_upper = (to_upper(uplo) == 'U') == (tr == 'N');
    const int a_row = (tr == 'N') ? 1 : lda;
    const int a_col = (tr == 'N') ? lda : 1;

    // Each element of X starts from alpha * B in the accumulator, takes off the
    // already solved terms, is divided by the diagonal of op(A) in the
    // accumulator, and is narrowed once, overwriting B.
    const auto solve = [&](mac_t<T> &sum, int d) {
        accumulator_t<T> value = sum.wide_value();
        if (!unit) {
            const T &pivot = a[d + d * lda];
            value = divide_value(value, to_accumulator(conj ? conj_value(pivot) : pivot));
        }
        return from_accumulator<T>(value);
    };
    const auto subtract = [&](mac_t<T> &sum, const T &op_a, const T &x) {
        if (conj) {
            sum.subtract_conj_product(op_a, x);
        } else {
            sum.subtract_product(op_a, x);
        }
    };

    if (left) {
        // op(A) * X = alpha * B, column by column.
        for (int j = 0; j < n; ++j) {
            T *col = b + j * ldb;
            for (int step = 0; step < m; ++step) {
                const int i = op_upper ? m - 1 - step : step;
                mac_t<T> sum;
                sum.add_product(alpha, col[i]);
                const int first = op_upper ? i + 1 : 0;
                const int last = op_upper ? m : i;
                for (int l = first; l < last; ++l) {
                    subtract(sum, a[i * a_row + l * a_col], col[l]);
                }
                col[i] = solve(sum, i);
            }
        }
    } else {
        // X * op(A) = alpha * B, row by row.
        for (int i = 0; i < m; ++i) {
            for (int step = 0; step < n; ++step) {
                const int j = op_upper ? step : n - 1 - step;
                mac_t<T> sum;
                sum.add_product(alpha, b[i + j * ldb]);
                const int first = op_upper ? 0 : j + 1;
                const int last = op_upper ? j : n;
                for (int l = first; l < last; ++l) {
                    subtract(sum, a[l * a_row + j * a_col], b[i + l * ldb]);
                }
                b[i + j * ldb] = solve(sum, j);
            }
        }
    }
}

} // namespace detail

/**
 * @brief General matrix-matrix multiply:
 *        `c <- alpha * op(a) * op(b) + beta * c`.
 *
 * `op(a)` is `m x k`, `op(b)` is `k x n` and `c` is `m x n`. `a` is stored with
 * `m` rows for `transa == 'N'` and `k` rows otherwise; likewise `b` has `k` or
 * `n` rows.
 */
template <typename T>
inline void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda,
                 const T *b, int ldb, T beta, T *c, int ldc) noexcept {
    detail::gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Symmetric rank-k update:
 *        `c <- alpha * a * a^T + beta * c` (`trans == 'N'`, `a` is `n x k`) or
 *        `c <- alpha * a^T * a + beta * c` (`trans == 'T'`, `a` is `k x n`).
 *
 * Only the `uplo` triangle of `c` is referenced and updated. For complex
 * element types `trans` must be `'N'` or `'T'`; for real ones `'C'` means
 * `'T'`.
 */
template <typename T>
inline void syrk(char uplo, char trans, int n, int k, T alpha, const T *a, int lda, T beta, T *c,
                 int ldc) noexcept {
    detail::syrk_impl(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

/**
 * @brief Hermitian rank-k update:
 *        `c <- alpha * a * a^H + beta * c` (`trans == 'N'`, `a` is `n x k`) or
 *        `c <- alpha * a^H * a + beta * c` (`trans == 'C'`, `a` is `k x n`).
 *
 * `alpha` and `beta` are real. Only the `uplo` triangle of `c` is referenced
 * and updated; the imaginary parts of its diagonal are not read and are set
 * to zero (unless the call returns early because `beta == 1` and `alpha == 0`
 * or `k == 0`, as in Netlib BLAS).
 */
template <typename T>
inline void herk(char uplo, char trans, int n, int k, T alpha, const std::complex<T> *a, int lda,
                 T beta, std::complex<T> *c, int ldc) noexcept {
    detail::herk_impl(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

/**
 * @brief Symmetric matrix-matrix multiply:
 *        `c <- alpha * a * b + beta * c` (`side == 'L'`) or
 *        `c <- alpha * b * a + beta * c` (`side == 'R'`).
 *
 * `b` and `c` are `m x n`; `a` is a symmetric `m x m` (`'L'`) or `n x n` (`'R'`)
 * matrix of which only the `uplo` triangle is referenced.
 */
template <typename T>
inline void symm(char side, char uplo, int m, int n, T alpha, const T *a, int lda, const T *b,
                 int ldb, T beta, T *c, int ldc) noexcept {
    detail::symm_like<false>(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Hermitian matrix-matrix multiply:
 *        `c <- alpha * a * b + beta * c` (`side == 'L'`) or
 *        `c <- alpha * b * a + beta * c` (`side == 'R'`).
 *
 * As `symm`, for a Hermitian `a`: only its `uplo` triangle is referenced, and
 * the imaginary parts of its diagonal are not read (taken as zero).
 */
template <typename T>
inline void hemm(char side, char uplo, int m, int n, std::complex<T> alpha,
                 const std::complex<T> *a, int lda, const std::complex<T> *b, int ldb,
                 std::complex<T> beta, std::complex<T> *c, int ldc) noexcept {
    detail::symm_like<true>(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Symmetric rank-2k update:
 *        `c <- alpha * a * b^T + alpha * b * a^T + beta * c` (`trans == 'N'`,
 *        `a`, `b` are `n x k`) or
 *        `c <- alpha * a^T * b + alpha * b^T * a + beta * c` (`trans == 'T'`,
 *        `a`, `b` are `k x n`).
 *
 * Only the `uplo` triangle of `c` is referenced and updated. As for `syrk`,
 * complex element types take `'N'` or `'T'`, and real ones treat `'C'` as
 * `'T'`.
 */
template <typename T>
inline void syr2k(char uplo, char trans, int n, int k, T alpha, const T *a, int lda, const T *b,
                  int ldb, T beta, T *c, int ldc) noexcept {
    detail::syr2k_impl(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Hermitian rank-2k update:
 *        `c <- alpha * a * b^H + conj(alpha) * b * a^H + beta * c`
 *        (`trans == 'N'`, `a`, `b` are `n x k`) or
 *        `c <- alpha * a^H * b + conj(alpha) * b^H * a + beta * c`
 *        (`trans == 'C'`, `a`, `b` are `k x n`).
 *
 * `alpha` is complex and `beta` real. Only the `uplo` triangle of `c` is
 * referenced and updated; the imaginary parts of its diagonal are not read and
 * are set to zero (unless the call returns early because `beta == 1` and
 * `alpha == 0` or `k == 0`, as in Netlib BLAS).
 */
template <typename T>
inline void her2k(char uplo, char trans, int n, int k, std::complex<T> alpha,
                  const std::complex<T> *a, int lda, const std::complex<T> *b, int ldb, T beta,
                  std::complex<T> *c, int ldc) noexcept {
    detail::her2k_impl(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Triangular matrix-matrix multiply:
 *        `b <- alpha * op(a) * b` (`side == 'L'`) or
 *        `b <- alpha * b * op(a)` (`side == 'R'`), in place.
 *
 * `b` is `m x n`; `a` is a triangular `m x m` (`'L'`) or `n x n` (`'R'`) matrix
 * whose `uplo` triangle is referenced, with an implicit unit diagonal for
 * `diag == 'U'`.
 */
template <typename T>
inline void trmm(char side, char uplo, char transa, char diag, int m, int n, T alpha, const T *a,
                 int lda, T *b, int ldb) noexcept {
    detail::trmm_impl(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

/**
 * @brief Triangular solve with multiple right-hand sides:
 *        `op(a) * x = alpha * b` (`side == 'L'`) or
 *        `x * op(a) = alpha * b` (`side == 'R'`); `x` overwrites `b`.
 *
 * `b` is `m x n`; `a` is a triangular `m x m` (`'L'`) or `n x n` (`'R'`)
 * matrix whose `uplo` triangle is referenced, with an implicit unit diagonal
 * for `diag == 'U'`. No test for singularity is made.
 */
template <typename T>
inline void trsm(char side, char uplo, char transa, char diag, int m, int n, T alpha, const T *a,
                 int lda, T *b, int ldb) noexcept {
    detail::trsm_impl(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

} // namespace thefblas
