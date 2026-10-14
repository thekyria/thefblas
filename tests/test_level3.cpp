#include "test_support.hpp"
#include "thefblas/fixed.hpp"
#include "thefblas/level3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

// Level 3 routines checked against a direct complex<double> evaluation of their
// defining formulas, for float, double, fixed point and their complex forms.
// Inputs are multiples of 1/16, which every element type represents exactly.

namespace {

using fixed32 = thefblas::fixed<std::int32_t, 20>;
using cd = std::complex<double>;

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

/// Offset of element (r, c) of a column-major matrix with leading dimension ld.
std::size_t at(int r, int c, int ld) {
    return static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
}

template <typename T> double real_to_double(const T &value) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(value);
    } else {
        return value.template to_float<double>();
    }
}

template <typename T> cd to_cd(const T &value) {
    if constexpr (is_complex<T>::value) {
        return {real_to_double(value.real()), real_to_double(value.imag())};
    } else {
        return {real_to_double(value), 0.0};
    }
}

template <typename R> R real_from_double(double value) {
    if constexpr (std::is_floating_point_v<R>) {
        return static_cast<R>(value);
    } else {
        return R(value);
    }
}

template <typename T> T from_cd(const cd &value) {
    if constexpr (is_complex<T>::value) {
        using R = typename T::value_type;
        return T(real_from_double<R>(value.real()), real_from_double<R>(value.imag()));
    } else {
        return real_from_double<T>(value.real());
    }
}

template <typename T> double tolerance() {
    if constexpr (is_complex<T>::value) {
        return tolerance<typename T::value_type>();
    } else if constexpr (std::is_same_v<T, double>) {
        return 1e-12;
    } else {
        return 1e-5; // float, and fixed32 with its 20 fractional bits
    }
}

template <typename T> bool near(const T &got, const cd &expected, double scale = 1.0) {
    return std::abs(to_cd(got) - expected) <= tolerance<T>() * scale * (1.0 + std::abs(expected));
}

/// Deterministic inputs: multiples of 1/16 in [-1, 1] (real and imaginary
/// parts for complex types).
class values {
  public:
    double next() {
        state_ = state_ * 1103515245U + 12345U;
        return static_cast<double>(static_cast<int>((state_ >> 16) % 33U) - 16) / 16.0;
    }
    template <typename T> T element() {
        const double re = next();
        if constexpr (is_complex<T>::value) {
            return from_cd<T>(cd(re, next()));
        } else {
            return from_cd<T>(cd(re, 0.0));
        }
    }
    template <typename T> std::vector<T> matrix(int ld, int cols) {
        std::vector<T> result(at(0, cols, ld));
        std::generate(result.begin(), result.end(), [this]() { return element<T>(); });
        return result;
    }

  private:
    std::uint32_t state_ = 12345U;
};

/// op(X)(r, s) of a column-major matrix in complex<double>.
template <typename T> cd op_at(const std::vector<T> &x, int ld, char trans, int r, int s) {
    if (trans == 'N') {
        return to_cd(x[at(r, s, ld)]);
    }
    const cd value = to_cd(x[at(s, r, ld)]);
    return trans == 'C' ? std::conj(value) : value;
}

template <typename T> bool same(const T &lhs, const T &rhs) {
    return to_cd(lhs) == to_cd(rhs);
}

template <typename T> bool unchanged(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!same(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// gemm
// ---------------------------------------------------------------------------

template <typename T> void check_gemm(char transa, char transb, T alpha, T beta, values &gen) {
    TEST_CONTEXT("gemm transa=", transa, " transb=", transb, " alpha=", alpha, " beta=", beta);
    const int m = 3;
    const int n = 2;
    const int k = 4;
    const int nrowa = transa == 'N' ? m : k;
    const int ncola = transa == 'N' ? k : m;
    const int nrowb = transb == 'N' ? k : n;
    const int ncolb = transb == 'N' ? n : k;
    const int lda = nrowa + 1;
    const int ldb = nrowb + 2;
    const int ldc = m + 1;
    const std::vector<T> a = gen.matrix<T>(lda, ncola);
    const std::vector<T> b = gen.matrix<T>(ldb, ncolb);
    const std::vector<T> c0 = gen.matrix<T>(ldc, n);
    std::vector<T> c = c0;
    thefblas::gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(),
                   ldc);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < ldc; ++i) {
            const auto idx = at(i, j, ldc);
            TEST_CONTEXT("i=", i, " j=", j);
            if (i >= m) {
                CHECK_VALUES(same(c[idx], c0[idx]), c[idx], c0[idx]); // padding row untouched
                continue;
            }
            cd sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += op_at(a, lda, transa, i, l) * op_at(b, ldb, transb, l, j);
            }
            const cd expected = to_cd(alpha) * sum + to_cd(beta) * to_cd(c0[idx]);
            CHECK_VALUES(near(c[idx], expected), c[idx], expected);
        }
    }
}

template <typename T> void test_gemm_all_transposes() {
    values gen;
    const T alpha = from_cd<T>(cd(0.75, is_complex<T>::value ? -0.5 : 0.0));
    const T beta = from_cd<T>(cd(-0.5, is_complex<T>::value ? 0.25 : 0.0));
    for (const char ta : {'N', 'T', 'C'}) {
        for (const char tb : {'N', 'T', 'C'}) {
            check_gemm(ta, tb, alpha, beta, gen);
            check_gemm(ta, tb, alpha, T{}, gen); // beta == 0
        }
    }
}

void test_gemm_special_cases() {
    const double a[4] = {1.0, 2.0, 3.0, 4.0};
    const double b[4] = {1.0, 0.0, 0.0, 1.0};
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    // beta == 0: c is not read, so NaN in c does not propagate.
    double c[4] = {nan, nan, nan, nan};
    thefblas::gemm('N', 'N', 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
    CHECK(c[0] == 1.0 && c[1] == 2.0 && c[2] == 3.0 && c[3] == 4.0);

    // alpha == 0: a and b are not read; c is scaled by beta (or zeroed).
    const double nan_ab[4] = {nan, nan, nan, nan};
    double c2[4] = {1.0, 2.0, 3.0, 4.0};
    thefblas::gemm('N', 'N', 2, 2, 2, 0.0, nan_ab, 2, nan_ab, 2, 2.0, c2, 2);
    CHECK(c2[0] == 2.0 && c2[3] == 8.0);
    thefblas::gemm('T', 'C', 2, 2, 2, 0.0, nan_ab, 2, nan_ab, 2, 0.0, c2, 2);
    CHECK(c2[0] == 0.0 && c2[3] == 0.0);

    // alpha == 0 or k == 0 with beta == 1: quick return.
    double c3[4] = {1.0, 2.0, 3.0, nan};
    thefblas::gemm('N', 'N', 2, 2, 2, 0.0, nan_ab, 2, nan_ab, 2, 1.0, c3, 2);
    thefblas::gemm('N', 'N', 2, 2, 0, 1.0, a, 1, b, 1, 1.0, c3, 2);
    CHECK(c3[0] == 1.0 && std::isnan(c3[3]));
    // k == 0 with beta != 1: c <- beta * c.
    thefblas::gemm('N', 'N', 2, 2, 0, 1.0, a, 2, b, 1, 3.0, c3, 2);
    CHECK(c3[0] == 3.0 && c3[2] == 9.0);

    // Invalid arguments leave c untouched.
    const std::array<double, 4> before = {5.0, 6.0, 7.0, 8.0};
    double c4[4] = {5.0, 6.0, 7.0, 8.0};
    const auto check_untouched = [&]() {
        for (int i = 0; i < 4; ++i) {
            CHECK(c4[i] == before[static_cast<std::size_t>(i)]);
        }
    };
    thefblas::gemm('X', 'N', 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c4, 2);
    thefblas::gemm('N', 'X', 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c4, 2);
    thefblas::gemm('N', 'N', 0, 2, 2, 1.0, a, 2, b, 2, 0.0, c4, 2);
    thefblas::gemm('N', 'N', 2, -1, 2, 1.0, a, 2, b, 2, 0.0, c4, 2);
    thefblas::gemm('N', 'N', 2, 2, -1, 1.0, a, 2, b, 2, 0.0, c4, 2);
    thefblas::gemm('N', 'N', 2, 2, 2, 1.0, a, 1, b, 2, 0.0, c4, 2); // lda < m
    thefblas::gemm('T', 'N', 2, 2, 2, 1.0, a, 1, b, 2, 0.0, c4, 2); // lda < k
    thefblas::gemm('N', 'N', 2, 2, 2, 1.0, a, 2, b, 1, 0.0, c4, 2); // ldb < k
    thefblas::gemm('N', 'T', 2, 2, 2, 1.0, a, 2, b, 1, 0.0, c4, 2); // ldb < n
    thefblas::gemm('N', 'N', 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c4, 1); // ldc < m
    check_untouched();
    // Lower-case options are accepted.
    thefblas::gemm('n', 't', 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c4, 2);
    CHECK(c4[0] == 1.0 && c4[3] == 4.0);
}

// The row sum 0.9 * 0.9 + 0.9 * 0.9 = 1.62 is outside Q1.15, but the output
// 1.62 - 0.9 = 0.72 is not: summing, scaling and adding beta * c in the
// accumulator before a single narrowing gives 0.72 rather than the ~0.1 a
// saturated intermediate would.
void test_gemm_fixed_cancellation() {
    using F = thefblas::fixed<std::int16_t, 15, thefblas::saturate>;
    const F a[2] = {F(0.9), F(0.9)};
    const F b[2] = {F(0.9), F(0.9)};
    const F almost_one = F::from_raw(32767);
    F c[1] = {F(-0.9)};
    thefblas::gemm('N', 'N', 1, 1, 2, almost_one, a, 1, b, 2, almost_one, c, 1);
    CHECK(std::fabs(c[0].to_float<double>() - 0.72) < 0.01);
}

// ---------------------------------------------------------------------------
// syrk / herk
// ---------------------------------------------------------------------------

/// Checks the result of syrk (Hermitian = false) or herk (true) on the uplo
/// triangle and that the other triangle and the padding are untouched.
template <bool Hermitian, typename T>
void check_rank_k(char uplo, char trans, const cd &alpha, const cd &beta, const std::vector<T> &a,
                  int lda, int n, int k, const std::vector<T> &c0, const std::vector<T> &c,
                  int ldc) {
    TEST_CONTEXT(Hermitian ? "herk" : "syrk", " uplo=", uplo, " trans=", trans, " alpha=", alpha,
                 " beta=", beta);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < ldc; ++i) {
            const auto idx = at(i, j, ldc);
            TEST_CONTEXT("i=", i, " j=", j);
            const bool in_triangle = i < n && (uplo == 'U' ? i <= j : i >= j);
            if (!in_triangle) {
                CHECK_VALUES(same(c[idx], c0[idx]), c[idx], c0[idx]);
                continue;
            }
            cd sum = 0.0;
            for (int l = 0; l < k; ++l) {
                const cd ui = op_at(a, lda, trans == 'N' ? 'N' : 'T', i, l);
                const cd uj = op_at(a, lda, trans == 'N' ? 'N' : 'T', j, l);
                if (Hermitian) {
                    sum += trans == 'N' ? ui * std::conj(uj) : std::conj(ui) * uj;
                } else {
                    sum += ui * uj;
                }
            }
            cd old = to_cd(c0[idx]);
            if (Hermitian && i == j) {
                old = old.real();
            }
            cd expected = alpha * sum + (beta == 0.0 ? cd(0.0) : beta * old);
            if (Hermitian && i == j) {
                CHECK(to_cd(c[idx]).imag() == 0.0);
                expected = expected.real();
            }
            CHECK_VALUES(near(c[idx], expected), c[idx], expected);
        }
    }
}

template <typename T> void test_syrk_all_options() {
    values gen;
    const int n = 3;
    const int k = 2;
    const T alpha = from_cd<T>(cd(0.5, is_complex<T>::value ? 0.25 : 0.0));
    const std::vector<char> transes =
        is_complex<T>::value ? std::vector<char>{'N', 'T'} : std::vector<char>{'N', 'T', 'C'};
    for (const char uplo : {'U', 'L'}) {
        for (const char trans : transes) {
            for (const T beta : {from_cd<T>(cd(-0.75, 0.0)), T{}}) {
                const int nrowa = trans == 'N' ? n : k;
                const int lda = nrowa + 1;
                const int ldc = n + 1;
                const std::vector<T> a = gen.matrix<T>(lda, trans == 'N' ? k : n);
                const std::vector<T> c0 = gen.matrix<T>(ldc, n);
                std::vector<T> c = c0;
                thefblas::syrk(uplo, trans, n, k, alpha, a.data(), lda, beta, c.data(), ldc);
                check_rank_k<false>(uplo, trans, to_cd(alpha), to_cd(beta), a, lda, n, k, c0, c,
                                    ldc);
            }
        }
    }
}

template <typename R> void test_herk_all_options() {
    using C = std::complex<R>;
    values gen;
    const int n = 3;
    const int k = 2;
    const R alpha = from_cd<R>(cd(0.5, 0.0));
    for (const char uplo : {'U', 'L'}) {
        for (const char trans : {'N', 'C'}) {
            for (const R beta : {from_cd<R>(cd(-0.75, 0.0)), R{}}) {
                const int nrowa = trans == 'N' ? n : k;
                const int lda = nrowa + 1;
                const int ldc = n + 1;
                const std::vector<C> a = gen.matrix<C>(lda, trans == 'N' ? k : n);
                const std::vector<C> c0 = gen.matrix<C>(ldc, n);
                std::vector<C> c = c0;
                thefblas::herk(uplo, trans, n, k, alpha, a.data(), lda, beta, c.data(), ldc);
                check_rank_k<true>(uplo, trans, to_cd(alpha), to_cd(beta), a, lda, n, k, c0, c,
                                   ldc);
            }
        }
    }
}

void test_rank_k_special_cases() {
    using C = std::complex<double>;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    const C a[4] = {{1.0, 1.0}, {2.0, 0.0}, {0.0, 1.0}, {1.0, -1.0}};
    const C nan_a[4] = {{nan, nan}, {nan, nan}, {nan, nan}, {nan, nan}};

    // herk, beta == 0: c is not read; the diagonal comes out real.
    C c[4] = {{nan, nan}, {nan, nan}, {nan, nan}, {nan, nan}};
    thefblas::herk('U', 'N', 2, 2, 1.0, a, 2, 0.0, c, 2);
    CHECK(c[0] == C(3.0, 0.0));     // |1+i|^2 + |i|^2
    CHECK(c[3] == C(6.0, 0.0));     // |2|^2 + |1-i|^2
    CHECK(std::isnan(c[1].real())); // lower triangle untouched

    // herk, alpha == 0: a is not read; c <- beta * c with a real diagonal.
    C c2[4] = {{1.0, 5.0}, {9.0, 9.0}, {2.0, 3.0}, {4.0, -1.0}};
    thefblas::herk('L', 'C', 2, 2, 0.0, nan_a, 2, 2.0, c2, 2);
    CHECK(c2[0] == C(2.0, 0.0) && c2[1] == C(18.0, 18.0) && c2[3] == C(8.0, 0.0));
    CHECK(c2[2] == C(2.0, 3.0)); // upper triangle untouched
    thefblas::herk('L', 'C', 2, 2, 0.0, nan_a, 2, 0.0, c2, 2);
    CHECK(c2[0] == C(0.0, 0.0) && c2[1] == C(0.0, 0.0) && c2[3] == C(0.0, 0.0));

    // herk, k == 0 and beta == 1: quick return, the diagonal keeps its
    // imaginary part (as in Netlib BLAS).
    C c3[4] = {{1.0, 5.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}};
    thefblas::herk('U', 'N', 2, 0, 1.0, a, 2, 1.0, c3, 2);
    CHECK(c3[0] == C(1.0, 5.0));
    // k == 0 with beta != 1: c <- beta * c, diagonal made real.
    thefblas::herk('U', 'N', 2, 0, 1.0, a, 2, 2.0, c3, 2);
    CHECK(c3[0] == C(2.0, 0.0) && c3[3] == C(2.0, 0.0));

    // Invalid arguments leave c untouched.
    C c4[4] = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
    const auto untouched = [&]() {
        return c4[0] == C(1.0, 1.0) && c4[1] == C(2.0, 2.0) && c4[2] == C(3.0, 3.0) &&
               c4[3] == C(4.0, 4.0);
    };
    thefblas::herk('X', 'N', 2, 2, 1.0, a, 2, 0.0, c4, 2);
    thefblas::herk('U', 'T', 2, 2, 1.0, a, 2, 0.0, c4, 2); // no 'T' for herk
    thefblas::herk('U', 'N', 0, 2, 1.0, a, 2, 0.0, c4, 2);
    thefblas::herk('U', 'N', 2, -1, 1.0, a, 2, 0.0, c4, 2);
    thefblas::herk('U', 'N', 2, 2, 1.0, a, 1, 0.0, c4, 2);       // lda < n
    thefblas::herk('U', 'C', 2, 2, 1.0, a, 1, 0.0, c4, 2);       // lda < k
    thefblas::herk('U', 'N', 2, 2, 1.0, a, 2, 0.0, c4, 1);       // ldc < n
    thefblas::syrk('U', 'C', 2, 2, C(1.0), a, 2, C(0.0), c4, 2); // no 'C' for complex syrk
    thefblas::syrk('X', 'N', 2, 2, C(1.0), a, 2, C(0.0), c4, 2);
    thefblas::syrk('U', 'N', 2, -1, C(1.0), a, 2, C(0.0), c4, 2);
    thefblas::syrk('U', 'N', 2, 2, C(1.0), a, 1, C(0.0), c4, 2);
    thefblas::syrk('U', 'N', 2, 2, C(1.0), a, 2, C(0.0), c4, 1);
    CHECK(untouched());

    // syrk, alpha == 0 (both beta cases) and the quick return.
    double r[4] = {1.0, 2.0, 3.0, 4.0};
    const double nan_r[4] = {nan, nan, nan, nan};
    thefblas::syrk('U', 'N', 2, 2, 0.0, nan_r, 2, 3.0, r, 2);
    CHECK(r[0] == 3.0 && r[1] == 2.0 && r[2] == 9.0 && r[3] == 12.0);
    thefblas::syrk('U', 'N', 2, 2, 0.0, nan_r, 2, 1.0, r, 2);
    CHECK(r[0] == 3.0);
    thefblas::syrk('L', 'T', 2, 2, 0.0, nan_r, 2, 0.0, r, 2);
    CHECK(r[0] == 0.0 && r[1] == 0.0 && r[2] == 9.0 && r[3] == 0.0);
}

// ---------------------------------------------------------------------------
// trsm
// ---------------------------------------------------------------------------

/// Triangular, well-conditioned A: diagonal entries of magnitude >= 2, small
/// off-diagonal entries in the uplo triangle and a large value (100) in the
/// other triangle and, for a unit diagonal, on the diagonal, so that reading
/// either would show up in the residual.
template <typename T>
std::vector<T> triangular(values &gen, char uplo, char diag, int dim, int lda) {
    std::vector<T> a = gen.matrix<T>(lda, dim);
    for (int s = 0; s < dim; ++s) {
        for (int r = 0; r < dim; ++r) {
            T &value = a[at(r, s, lda)];
            const bool stored = uplo == 'U' ? r <= s : r >= s;
            if (r == s) {
                value = diag == 'U' ? from_cd<T>(cd(100.0, 0.0))
                                    : from_cd<T>(to_cd(value) * 0.25 + cd(2.0, 0.0));
            } else if (stored) {
                value = from_cd<T>(to_cd(value) * 0.25);
            } else {
                value = from_cd<T>(cd(100.0, 0.0));
            }
        }
    }
    return a;
}

/// The triangular matrix trsm actually uses, as op(A)(r, s) in complex<double>.
template <typename T>
cd op_triangular(const std::vector<T> &a, int lda, char uplo, char trans, char diag, int r, int s) {
    // Element (p, q) of A as referenced: zero outside the uplo triangle, one on
    // a unit diagonal.
    const auto ref = [&](int p, int q) -> cd {
        if (p == q && diag == 'U') {
            return 1.0;
        }
        const bool stored = uplo == 'U' ? p <= q : p >= q;
        return stored ? to_cd(a[at(p, q, lda)]) : cd(0.0);
    };
    if (trans == 'N') {
        return ref(r, s);
    }
    return trans == 'C' ? std::conj(ref(s, r)) : ref(s, r);
}

template <typename T>
void check_trsm(char side, char uplo, char trans, char diag, const T &alpha, values &gen) {
    TEST_CONTEXT("trsm side=", side, " uplo=", uplo, " trans=", trans, " diag=", diag,
                 " alpha=", alpha);
    const int m = 3;
    const int n = 2;
    const int dim = side == 'L' ? m : n;
    const int lda = dim + 1;
    const int ldb = m + 1;
    const std::vector<T> a = triangular<T>(gen, uplo, diag, dim, lda);
    const std::vector<T> b0 = gen.matrix<T>(ldb, n);
    std::vector<T> b = b0;
    thefblas::trsm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b.data(), ldb);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < ldb; ++i) {
            const auto idx = at(i, j, ldb);
            TEST_CONTEXT("i=", i, " j=", j);
            if (i >= m) {
                CHECK_VALUES(same(b[idx], b0[idx]), b[idx], b0[idx]);
                continue;
            }
            // Residual: (op(A) * X)(i, j) or (X * op(A))(i, j) against alpha * B.
            cd lhs = 0.0;
            if (side == 'L') {
                for (int l = 0; l < m; ++l) {
                    lhs += op_triangular(a, lda, uplo, trans, diag, i, l) * to_cd(b[at(l, j, ldb)]);
                }
            } else {
                for (int l = 0; l < n; ++l) {
                    lhs += to_cd(b[at(i, l, ldb)]) * op_triangular(a, lda, uplo, trans, diag, l, j);
                }
            }
            const cd rhs = to_cd(alpha) * to_cd(b0[idx]);
            CHECK_VALUES(std::abs(lhs - rhs) <= 10.0 * tolerance<T>() * (1.0 + std::abs(rhs)), lhs,
                         rhs);
        }
    }
}

template <typename T> void test_trsm_all_options() {
    values gen;
    const T alpha = from_cd<T>(cd(0.75, is_complex<T>::value ? 0.5 : 0.0));
    for (const char side : {'L', 'R'}) {
        for (const char uplo : {'U', 'L'}) {
            for (const char trans : {'N', 'T', 'C'}) {
                for (const char diag : {'N', 'U'}) {
                    check_trsm(side, uplo, trans, diag, alpha, gen);
                }
            }
        }
    }
}

void test_trsm_special_cases() {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    const double nan_a[4] = {nan, nan, nan, nan};
    const double a[4] = {2.0, 0.0, 1.0, 4.0};

    // alpha == 0: a is not read and b is zeroed.
    double b[4] = {1.0, 2.0, 3.0, 4.0};
    thefblas::trsm('L', 'U', 'N', 'N', 2, 2, 0.0, nan_a, 2, b, 2);
    CHECK(b[0] == 0.0 && b[1] == 0.0 && b[2] == 0.0 && b[3] == 0.0);

    // Invalid arguments leave b untouched.
    double b2[4] = {1.0, 2.0, 3.0, 4.0};
    thefblas::trsm('X', 'U', 'N', 'N', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trsm('L', 'X', 'N', 'N', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trsm('L', 'U', 'X', 'N', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trsm('L', 'U', 'N', 'X', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trsm('L', 'U', 'N', 'N', 0, 2, 1.0, a, 2, b2, 2);
    thefblas::trsm('L', 'U', 'N', 'N', 2, -1, 1.0, a, 2, b2, 2);
    thefblas::trsm('L', 'U', 'N', 'N', 2, 2, 1.0, a, 1, b2, 2); // lda < m
    thefblas::trsm('R', 'U', 'N', 'N', 2, 2, 1.0, a, 1, b2, 2); // lda < n
    thefblas::trsm('L', 'U', 'N', 'N', 2, 2, 1.0, a, 2, b2, 1); // ldb < m
    CHECK(b2[0] == 1.0 && b2[1] == 2.0 && b2[2] == 3.0 && b2[3] == 4.0);
}

// alpha * B is formed in the accumulator: in Q7.8 (range [-128, 128))
// 16 * 10 = 160 is out of range, but the solutions computed from it
// (160 / 2 = 80 below, and x = [0, 80] in the left-side case) are not.
void test_trsm_fixed_accumulator() {
    using F = thefblas::fixed<std::int16_t, 8, thefblas::saturate>;
    const F a[4] = {F(1.0), F(0.0), F(10.0), F(2.0)}; // upper: [[1, 10], [0, 2]]
    F b[2] = {F(0.0), F(10.0)};
    // Solve X * A = 16 * B for the 1 x 2 row [0, 10]:
    // x0 = 0, x1 = (16 * 10 - x0 * 10) / 2 = 80.
    thefblas::trsm('R', 'U', 'N', 'N', 1, 2, F(16.0), a, 2, b, 1);
    CHECK(b[0] == F(0.0));
    CHECK(std::fabs(b[1].to_float<double>() - 80.0) < 0.01);
    // Left side: [[1, 10], [0, 2]] * x = 16 * [50, 10]. The intermediates
    // 16 * 10 = 160 and 16 * 50 = 800 are out of range, the solution
    // x = [0, 80] is not.
    F b2[2] = {F(50.0), F(10.0)};
    thefblas::trsm('L', 'U', 'N', 'N', 2, 1, F(16.0), a, 2, b2, 2);
    CHECK(std::fabs(b2[1].to_float<double>() - 80.0) < 0.01);
    CHECK(std::fabs(b2[0].to_float<double>()) < 0.01);
}

// ---------------------------------------------------------------------------
// symm / hemm
// ---------------------------------------------------------------------------

/// A symmetric (or Hermitian) matrix stored in its uplo triangle, with 100 in
/// the other triangle and, for Hermitian, a large imaginary part on the
/// diagonal, neither of which may be read.
template <bool Hermitian, typename T>
std::vector<T> symmetric(values &gen, char uplo, int dim, int lda) {
    std::vector<T> a = gen.matrix<T>(lda, dim);
    for (int s = 0; s < dim; ++s) {
        for (int r = 0; r < dim; ++r) {
            const bool stored = uplo == 'U' ? r <= s : r >= s;
            T &value = a[at(r, s, lda)];
            if (r == s && Hermitian) {
                value = from_cd<T>(cd(to_cd(value).real(), 7.0));
            } else if (!stored) {
                value = from_cd<T>(cd(100.0, 0.0));
            }
        }
    }
    return a;
}

/// Element (r, s) of the full symmetric (Hermitian) matrix described by the
/// uplo triangle of a.
template <bool Hermitian, typename T>
cd symmetric_at(const std::vector<T> &a, int lda, char uplo, int r, int s) {
    if (r == s) {
        const cd value = to_cd(a[at(r, r, lda)]);
        return Hermitian ? cd(value.real()) : value;
    }
    const bool stored = uplo == 'U' ? r < s : r > s;
    if (stored) {
        return to_cd(a[at(r, s, lda)]);
    }
    const cd value = to_cd(a[at(s, r, lda)]);
    return Hermitian ? std::conj(value) : value;
}

template <bool Hermitian, typename T>
void check_symm(char side, char uplo, const T &alpha, const T &beta, values &gen) {
    TEST_CONTEXT(Hermitian ? "hemm" : "symm", " side=", side, " uplo=", uplo, " alpha=", alpha,
                 " beta=", beta);
    const int m = 3;
    const int n = 2;
    const int dim = side == 'L' ? m : n;
    const int lda = dim + 1;
    const int ldb = m + 2;
    const int ldc = m + 1;
    const std::vector<T> a = symmetric<Hermitian, T>(gen, uplo, dim, lda);
    const std::vector<T> b = gen.matrix<T>(ldb, n);
    const std::vector<T> c0 = gen.matrix<T>(ldc, n);
    std::vector<T> c = c0;
    if constexpr (Hermitian) {
        thefblas::hemm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    } else {
        thefblas::symm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < ldc; ++i) {
            const auto idx = at(i, j, ldc);
            TEST_CONTEXT("i=", i, " j=", j);
            if (i >= m) {
                CHECK_VALUES(same(c[idx], c0[idx]), c[idx], c0[idx]);
                continue;
            }
            cd sum = 0.0;
            if (side == 'L') {
                for (int l = 0; l < m; ++l) {
                    sum += symmetric_at<Hermitian>(a, lda, uplo, i, l) * to_cd(b[at(l, j, ldb)]);
                }
            } else {
                for (int l = 0; l < n; ++l) {
                    sum += to_cd(b[at(i, l, ldb)]) * symmetric_at<Hermitian>(a, lda, uplo, l, j);
                }
            }
            const cd old = same(beta, T{}) ? cd(0.0) : to_cd(beta) * to_cd(c0[idx]);
            const cd expected = to_cd(alpha) * sum + old;
            CHECK_VALUES(near(c[idx], expected), c[idx], expected);
        }
    }
}

template <bool Hermitian, typename T> void test_symm_all_options() {
    values gen;
    const T alpha = from_cd<T>(cd(0.75, is_complex<T>::value ? -0.5 : 0.0));
    const T beta = from_cd<T>(cd(-0.5, is_complex<T>::value ? 0.25 : 0.0));
    for (const char side : {'L', 'R'}) {
        for (const char uplo : {'U', 'L'}) {
            check_symm<Hermitian>(side, uplo, alpha, beta, gen);
            check_symm<Hermitian>(side, uplo, alpha, T{}, gen);
        }
    }
}

void test_symm_special_cases() {
    using C = std::complex<double>;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    const double a[4] = {2.0, 1.0, 1.0, 3.0}; // symmetric [[2, 1], [1, 3]]
    const double b[4] = {1.0, 0.0, 0.0, 1.0};
    const double nan_ab[4] = {nan, nan, nan, nan};

    // beta == 0: c is not read.
    double c[4] = {nan, nan, nan, nan};
    thefblas::symm('L', 'U', 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
    CHECK(c[0] == 2.0 && c[1] == 1.0 && c[2] == 1.0 && c[3] == 3.0);

    // alpha == 0: a and b are not read; c <- beta * c, or zero.
    double c2[4] = {1.0, 2.0, 3.0, 4.0};
    thefblas::symm('R', 'L', 2, 2, 0.0, nan_ab, 2, nan_ab, 2, 2.0, c2, 2);
    CHECK(c2[0] == 2.0 && c2[3] == 8.0);
    thefblas::symm('R', 'L', 2, 2, 0.0, nan_ab, 2, nan_ab, 2, 1.0, c2, 2); // quick return
    CHECK(c2[0] == 2.0 && c2[3] == 8.0);
    thefblas::symm('R', 'L', 2, 2, 0.0, nan_ab, 2, nan_ab, 2, 0.0, c2, 2);
    CHECK(c2[0] == 0.0 && c2[3] == 0.0);

    // Invalid arguments leave c untouched; lower-case options are accepted.
    double c3[4] = {5.0, 6.0, 7.0, 8.0};
    thefblas::symm('X', 'U', 2, 2, 1.0, a, 2, b, 2, 0.0, c3, 2);
    thefblas::symm('L', 'X', 2, 2, 1.0, a, 2, b, 2, 0.0, c3, 2);
    thefblas::symm('L', 'U', 0, 2, 1.0, a, 2, b, 2, 0.0, c3, 2);
    thefblas::symm('L', 'U', 2, -1, 1.0, a, 2, b, 2, 0.0, c3, 2);
    thefblas::symm('L', 'U', 2, 2, 1.0, a, 1, b, 2, 0.0, c3, 2); // lda < m
    thefblas::symm('R', 'U', 2, 2, 1.0, a, 1, b, 2, 0.0, c3, 2); // lda < n
    thefblas::symm('L', 'U', 2, 2, 1.0, a, 2, b, 1, 0.0, c3, 2); // ldb < m
    thefblas::symm('L', 'U', 2, 2, 1.0, a, 2, b, 2, 0.0, c3, 1); // ldc < m
    CHECK(c3[0] == 5.0 && c3[1] == 6.0 && c3[2] == 7.0 && c3[3] == 8.0);
    thefblas::symm('l', 'u', 2, 2, 1.0, a, 2, b, 2, 0.0, c3, 2);
    CHECK(c3[0] == 2.0 && c3[3] == 3.0);

    // hemm reads only the real part of the diagonal of a.
    const C ha[4] = {
        {2.0, 9.0}, {nan, nan}, {1.0, 1.0}, {3.0, -9.0}}; // upper: [[2, 1+i], [1-i, 3]]
    const C hb[2] = {{1.0, 0.0}, {0.0, 0.0}};
    C hc[2] = {{nan, nan}, {nan, nan}};
    thefblas::hemm('L', 'U', 2, 1, C(1.0), ha, 2, hb, 2, C(0.0), hc, 2);
    CHECK(hc[0] == C(2.0, 0.0) && hc[1] == C(1.0, -1.0));
}

// ---------------------------------------------------------------------------
// syr2k / her2k
// ---------------------------------------------------------------------------

template <bool Hermitian, typename T>
void check_rank_2k(char uplo, char trans, const cd &alpha, const cd &beta, const std::vector<T> &a,
                   int lda, const std::vector<T> &b, int ldb, int n, int k,
                   const std::vector<T> &c0, const std::vector<T> &c, int ldc) {
    TEST_CONTEXT(Hermitian ? "her2k" : "syr2k", " uplo=", uplo, " trans=", trans, " alpha=", alpha,
                 " beta=", beta);
    const char t = trans == 'N' ? 'N' : 'T';
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < ldc; ++i) {
            const auto idx = at(i, j, ldc);
            TEST_CONTEXT("i=", i, " j=", j);
            const bool in_triangle = i < n && (uplo == 'U' ? i <= j : i >= j);
            if (!in_triangle) {
                CHECK_VALUES(same(c[idx], c0[idx]), c[idx], c0[idx]);
                continue;
            }
            cd ab = 0.0;
            cd ba = 0.0;
            for (int l = 0; l < k; ++l) {
                const cd ai = op_at(a, lda, t, i, l);
                const cd aj = op_at(a, lda, t, j, l);
                const cd bi = op_at(b, ldb, t, i, l);
                const cd bj = op_at(b, ldb, t, j, l);
                if (Hermitian) {
                    ab += trans == 'N' ? ai * std::conj(bj) : std::conj(ai) * bj;
                    ba += trans == 'N' ? bi * std::conj(aj) : std::conj(bi) * aj;
                } else {
                    ab += ai * bj;
                    ba += bi * aj;
                }
            }
            cd old = to_cd(c0[idx]);
            if (Hermitian && i == j) {
                old = old.real();
            }
            const cd alpha2 = Hermitian ? std::conj(alpha) : alpha;
            cd expected = alpha * ab + alpha2 * ba + (beta == 0.0 ? cd(0.0) : beta * old);
            if (Hermitian && i == j) {
                CHECK(to_cd(c[idx]).imag() == 0.0);
                expected = expected.real();
            }
            CHECK_VALUES(near(c[idx], expected), c[idx], expected);
        }
    }
}

template <typename T> void test_syr2k_all_options() {
    values gen;
    const int n = 3;
    const int k = 2;
    const T alpha = from_cd<T>(cd(0.5, is_complex<T>::value ? 0.25 : 0.0));
    const std::vector<char> transes =
        is_complex<T>::value ? std::vector<char>{'N', 'T'} : std::vector<char>{'N', 'T', 'C'};
    for (const char uplo : {'U', 'L'}) {
        for (const char trans : transes) {
            for (const T beta : {from_cd<T>(cd(-0.75, 0.0)), T{}, from_cd<T>(cd(1.0, 0.0))}) {
                const int nrowa = trans == 'N' ? n : k;
                const int lda = nrowa + 1;
                const int ldb = nrowa + 2;
                const int ldc = n + 1;
                const std::vector<T> a = gen.matrix<T>(lda, trans == 'N' ? k : n);
                const std::vector<T> b = gen.matrix<T>(ldb, trans == 'N' ? k : n);
                const std::vector<T> c0 = gen.matrix<T>(ldc, n);
                std::vector<T> c = c0;
                thefblas::syr2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), ldb, beta,
                                c.data(), ldc);
                check_rank_2k<false>(uplo, trans, to_cd(alpha), to_cd(beta), a, lda, b, ldb, n, k,
                                     c0, c, ldc);
            }
        }
    }
}

template <typename R> void test_her2k_all_options() {
    using C = std::complex<R>;
    values gen;
    const int n = 3;
    const int k = 2;
    const C alpha = from_cd<C>(cd(0.5, -0.25));
    for (const char uplo : {'U', 'L'}) {
        for (const char trans : {'N', 'C'}) {
            for (const R beta : {from_cd<R>(cd(-0.75, 0.0)), R{}, from_cd<R>(cd(1.0, 0.0))}) {
                const int nrowa = trans == 'N' ? n : k;
                const int lda = nrowa + 1;
                const int ldb = nrowa + 2;
                const int ldc = n + 1;
                const std::vector<C> a = gen.matrix<C>(lda, trans == 'N' ? k : n);
                const std::vector<C> b = gen.matrix<C>(ldb, trans == 'N' ? k : n);
                const std::vector<C> c0 = gen.matrix<C>(ldc, n);
                std::vector<C> c = c0;
                thefblas::her2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), ldb, beta,
                                c.data(), ldc);
                check_rank_2k<true>(uplo, trans, to_cd(alpha), to_cd(beta), a, lda, b, ldb, n, k,
                                    c0, c, ldc);
            }
        }
    }
}

void test_rank_2k_special_cases() {
    using C = std::complex<double>;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    const C a[4] = {{1.0, 1.0}, {2.0, 0.0}, {0.0, 1.0}, {1.0, -1.0}};
    const C nan_a[4] = {{nan, nan}, {nan, nan}, {nan, nan}, {nan, nan}};

    // her2k, alpha == 0: a and b are not read; c <- beta * c, real diagonal.
    C c[4] = {{1.0, 5.0}, {9.0, 9.0}, {2.0, 3.0}, {4.0, -1.0}};
    thefblas::her2k('L', 'C', 2, 2, C(0.0), nan_a, 2, nan_a, 2, 2.0, c, 2);
    CHECK(c[0] == C(2.0, 0.0) && c[1] == C(18.0, 18.0) && c[3] == C(8.0, 0.0));
    CHECK(c[2] == C(2.0, 3.0)); // upper triangle untouched
    // her2k, k == 0 and beta == 1: quick return, the diagonal keeps its
    // imaginary part (as in Netlib BLAS).
    C c2[4] = {{1.0, 5.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}};
    thefblas::her2k('U', 'N', 2, 0, C(1.0), a, 2, a, 2, 1.0, c2, 2);
    CHECK(c2[0] == C(1.0, 5.0));
    // k == 0 with beta != 1: c <- beta * c, diagonal made real.
    thefblas::her2k('U', 'N', 2, 0, C(1.0), a, 2, a, 2, 3.0, c2, 2);
    CHECK(c2[0] == C(3.0, 0.0) && c2[3] == C(3.0, 0.0));

    // Invalid arguments leave c untouched.
    C c3[4] = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
    const auto untouched = [&]() {
        return c3[0] == C(1.0, 1.0) && c3[1] == C(2.0, 2.0) && c3[2] == C(3.0, 3.0) &&
               c3[3] == C(4.0, 4.0);
    };
    thefblas::her2k('X', 'N', 2, 2, C(1.0), a, 2, a, 2, 0.0, c3, 2);
    thefblas::her2k('U', 'T', 2, 2, C(1.0), a, 2, a, 2, 0.0, c3, 2); // no 'T' for her2k
    thefblas::her2k('U', 'N', 0, 2, C(1.0), a, 2, a, 2, 0.0, c3, 2);
    thefblas::her2k('U', 'N', 2, -1, C(1.0), a, 2, a, 2, 0.0, c3, 2);
    thefblas::her2k('U', 'N', 2, 2, C(1.0), a, 1, a, 2, 0.0, c3, 2);    // lda < n
    thefblas::her2k('U', 'N', 2, 2, C(1.0), a, 2, a, 1, 0.0, c3, 2);    // ldb < n
    thefblas::her2k('U', 'C', 2, 2, C(1.0), a, 2, a, 1, 0.0, c3, 2);    // ldb < k
    thefblas::her2k('U', 'N', 2, 2, C(1.0), a, 2, a, 2, 0.0, c3, 1);    // ldc < n
    thefblas::syr2k('U', 'C', 2, 2, C(1.0), a, 2, a, 2, C(0.0), c3, 2); // no 'C' for complex syr2k
    thefblas::syr2k('X', 'N', 2, 2, C(1.0), a, 2, a, 2, C(0.0), c3, 2);
    thefblas::syr2k('U', 'N', 2, -1, C(1.0), a, 2, a, 2, C(0.0), c3, 2);
    thefblas::syr2k('U', 'N', 2, 2, C(1.0), a, 1, a, 2, C(0.0), c3, 2);
    thefblas::syr2k('U', 'N', 2, 2, C(1.0), a, 2, a, 1, C(0.0), c3, 2);
    thefblas::syr2k('U', 'N', 2, 2, C(1.0), a, 2, a, 2, C(0.0), c3, 1);
    CHECK(untouched());

    // syr2k, alpha == 0 (both beta cases) and the quick return.
    double r[4] = {1.0, 2.0, 3.0, 4.0};
    const double nan_r[4] = {nan, nan, nan, nan};
    thefblas::syr2k('U', 'N', 2, 2, 0.0, nan_r, 2, nan_r, 2, 3.0, r, 2);
    CHECK(r[0] == 3.0 && r[1] == 2.0 && r[2] == 9.0 && r[3] == 12.0);
    thefblas::syr2k('U', 'N', 2, 2, 0.0, nan_r, 2, nan_r, 2, 1.0, r, 2);
    CHECK(r[0] == 3.0);
    thefblas::syr2k('L', 'T', 2, 2, 0.0, nan_r, 2, nan_r, 2, 0.0, r, 2);
    CHECK(r[0] == 0.0 && r[1] == 0.0 && r[2] == 9.0 && r[3] == 0.0);
    // beta == 0: c is not read.
    const double ra[2] = {1.0, 2.0};
    double rc[4] = {nan, nan, nan, nan};
    thefblas::syr2k('U', 'N', 2, 1, 1.0, ra, 2, ra, 2, 0.0, rc, 2);
    CHECK(rc[0] == 2.0 && rc[2] == 4.0 && rc[3] == 8.0 && std::isnan(rc[1]));
}

// The two products 0.9 * 0.9 of syr2k sum to 1.62, outside Q1.15; with beta *
// c = -0.9 added in the accumulator before the single narrowing, the result is
// 0.72, as in test_gemm_fixed_cancellation.
void test_syr2k_fixed_cancellation() {
    using F = thefblas::fixed<std::int16_t, 15, thefblas::saturate>;
    const F a[1] = {F(0.9)};
    const F b[1] = {F(0.9)};
    const F almost_one = F::from_raw(32767);
    F c[1] = {F(-0.9)};
    thefblas::syr2k('U', 'N', 1, 1, almost_one, a, 1, b, 1, almost_one, c, 1);
    CHECK(std::fabs(c[0].to_float<double>() - 0.72) < 0.01);
}

// ---------------------------------------------------------------------------
// trmm
// ---------------------------------------------------------------------------

template <typename T>
void check_trmm(char side, char uplo, char trans, char diag, const T &alpha, values &gen) {
    TEST_CONTEXT("trmm side=", side, " uplo=", uplo, " trans=", trans, " diag=", diag,
                 " alpha=", alpha);
    const int m = 3;
    const int n = 2;
    const int dim = side == 'L' ? m : n;
    const int lda = dim + 1;
    const int ldb = m + 1;
    const std::vector<T> a = triangular<T>(gen, uplo, diag, dim, lda);
    const std::vector<T> b0 = gen.matrix<T>(ldb, n);
    std::vector<T> b = b0;
    thefblas::trmm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b.data(), ldb);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < ldb; ++i) {
            const auto idx = at(i, j, ldb);
            TEST_CONTEXT("i=", i, " j=", j);
            if (i >= m) {
                CHECK_VALUES(same(b[idx], b0[idx]), b[idx], b0[idx]);
                continue;
            }
            cd sum = 0.0;
            if (side == 'L') {
                for (int l = 0; l < m; ++l) {
                    sum +=
                        op_triangular(a, lda, uplo, trans, diag, i, l) * to_cd(b0[at(l, j, ldb)]);
                }
            } else {
                for (int l = 0; l < n; ++l) {
                    sum +=
                        to_cd(b0[at(i, l, ldb)]) * op_triangular(a, lda, uplo, trans, diag, l, j);
                }
            }
            const cd expected = to_cd(alpha) * sum;
            CHECK_VALUES(near(b[idx], expected), b[idx], expected);
        }
    }
}

template <typename T> void test_trmm_all_options() {
    values gen;
    const T alpha = from_cd<T>(cd(0.75, is_complex<T>::value ? 0.5 : 0.0));
    for (const char side : {'L', 'R'}) {
        for (const char uplo : {'U', 'L'}) {
            for (const char trans : {'N', 'T', 'C'}) {
                for (const char diag : {'N', 'U'}) {
                    check_trmm(side, uplo, trans, diag, alpha, gen);
                }
            }
        }
    }
}

void test_trmm_special_cases() {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    const double nan_a[4] = {nan, nan, nan, nan};
    const double a[4] = {2.0, 0.0, 1.0, 4.0};

    // alpha == 0: a is not read and b is zeroed.
    double b[4] = {1.0, 2.0, 3.0, 4.0};
    thefblas::trmm('R', 'L', 'T', 'U', 2, 2, 0.0, nan_a, 2, b, 2);
    CHECK(b[0] == 0.0 && b[1] == 0.0 && b[2] == 0.0 && b[3] == 0.0);

    // Invalid arguments leave b untouched; lower-case options are accepted.
    double b2[4] = {1.0, 2.0, 3.0, 4.0};
    thefblas::trmm('X', 'U', 'N', 'N', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trmm('L', 'X', 'N', 'N', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trmm('L', 'U', 'X', 'N', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trmm('L', 'U', 'N', 'X', 2, 2, 1.0, a, 2, b2, 2);
    thefblas::trmm('L', 'U', 'N', 'N', 0, 2, 1.0, a, 2, b2, 2);
    thefblas::trmm('L', 'U', 'N', 'N', 2, -1, 1.0, a, 2, b2, 2);
    thefblas::trmm('L', 'U', 'N', 'N', 2, 2, 1.0, a, 1, b2, 2); // lda < m
    thefblas::trmm('R', 'U', 'N', 'N', 2, 2, 1.0, a, 1, b2, 2); // lda < n
    thefblas::trmm('L', 'U', 'N', 'N', 2, 2, 1.0, a, 2, b2, 1); // ldb < m
    CHECK(b2[0] == 1.0 && b2[1] == 2.0 && b2[2] == 3.0 && b2[3] == 4.0);
    // [[2, 1], [0, 4]] * [[1, 3], [2, 4]] = [[4, 10], [8, 16]]
    thefblas::trmm('l', 'u', 'n', 'n', 2, 2, 1.0, a, 2, b2, 2);
    CHECK(b2[0] == 4.0 && b2[1] == 8.0 && b2[2] == 10.0 && b2[3] == 16.0);
}

// op(A) * B is summed in the accumulator before alpha is applied: in Q7.8
// (range [-128, 128)) the row sum 50 + 10 * 10 = 150 is out of range, but
// 0.25 * 150 = 37.5 is not.
void test_trmm_fixed_accumulator() {
    using F = thefblas::fixed<std::int16_t, 8, thefblas::saturate>;
    const F a[4] = {F(1.0), F(0.0), F(10.0), F(2.0)}; // upper: [[1, 10], [0, 2]]
    F b[2] = {F(50.0), F(10.0)};
    thefblas::trmm('L', 'U', 'N', 'N', 2, 1, F(0.25), a, 2, b, 2);
    CHECK(std::fabs(b[0].to_float<double>() - 37.5) < 0.01);
    CHECK(std::fabs(b[1].to_float<double>() - 5.0) < 0.01);
    // The same with a unit diagonal: 0.25 * (50 + 10 * 10) and 0.25 * 10.
    F b2[2] = {F(50.0), F(10.0)};
    thefblas::trmm('L', 'U', 'N', 'U', 2, 1, F(0.25), a, 2, b2, 2);
    CHECK(std::fabs(b2[0].to_float<double>() - 37.5) < 0.01);
    CHECK(std::fabs(b2[1].to_float<double>() - 2.5) < 0.01);
}

} // namespace

int main() {
    test_gemm_all_transposes<float>();
    test_gemm_all_transposes<double>();
    test_gemm_all_transposes<std::complex<float>>();
    test_gemm_all_transposes<std::complex<double>>();
    test_gemm_all_transposes<fixed32>();
    test_gemm_all_transposes<std::complex<fixed32>>();
    test_gemm_special_cases();
    test_gemm_fixed_cancellation();

    test_syrk_all_options<float>();
    test_syrk_all_options<double>();
    test_syrk_all_options<std::complex<double>>();
    test_syrk_all_options<fixed32>();
    test_syrk_all_options<std::complex<fixed32>>();
    test_herk_all_options<float>();
    test_herk_all_options<double>();
    test_herk_all_options<fixed32>();
    test_rank_k_special_cases();

    test_trsm_all_options<float>();
    test_trsm_all_options<double>();
    test_trsm_all_options<std::complex<double>>();
    test_trsm_all_options<fixed32>();
    test_trsm_all_options<std::complex<fixed32>>();
    test_trsm_special_cases();
    test_trsm_fixed_accumulator();

    test_symm_all_options<false, float>();
    test_symm_all_options<false, double>();
    test_symm_all_options<false, std::complex<double>>();
    test_symm_all_options<false, fixed32>();
    test_symm_all_options<false, std::complex<fixed32>>();
    test_symm_all_options<true, std::complex<float>>();
    test_symm_all_options<true, std::complex<double>>();
    test_symm_all_options<true, std::complex<fixed32>>();
    test_symm_special_cases();

    test_syr2k_all_options<float>();
    test_syr2k_all_options<double>();
    test_syr2k_all_options<std::complex<double>>();
    test_syr2k_all_options<fixed32>();
    test_syr2k_all_options<std::complex<fixed32>>();
    test_her2k_all_options<float>();
    test_her2k_all_options<double>();
    test_her2k_all_options<fixed32>();
    test_rank_2k_special_cases();
    test_syr2k_fixed_cancellation();

    test_trmm_all_options<float>();
    test_trmm_all_options<double>();
    test_trmm_all_options<std::complex<double>>();
    test_trmm_all_options<fixed32>();
    test_trmm_all_options<std::complex<fixed32>>();
    test_trmm_special_cases();
    test_trmm_fixed_accumulator();
    return thefblas_test::finish();
}
