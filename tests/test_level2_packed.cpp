#include "thefblas/fixed.hpp"
#include "thefblas/level2.hpp"
#include "thefblas/level2_packed.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t uz(int value) {
    return static_cast<std::size_t>(value);
}

using fixed32 = thefblas::fixed<std::int32_t, 20>;
using cfloat = std::complex<float>;
using cfixed = std::complex<fixed32>;

template <typename T> double to_double(T value) {
    return static_cast<double>(value);
}

template <typename IntType, int FracBits>
double to_double(thefblas::fixed<IntType, FracBits> value) {
    return value.template to_float<double>();
}

template <typename T> bool almost_equal_scalar(T lhs, T rhs, double eps = 1e-5) {
    const double diff = to_double(lhs - rhs);
    return diff <= eps && diff >= -eps;
}

template <typename T>
bool almost_equal_scalar(std::complex<T> lhs, std::complex<T> rhs, double eps = 1e-5) {
    return almost_equal_scalar(lhs.real(), rhs.real(), eps) &&
           almost_equal_scalar(lhs.imag(), rhs.imag(), eps);
}

int packed_index(char uplo, int i, int j, int n) {
    return (uplo == 'U') ? (i + (j * (j + 1)) / 2) : ((i - j) + (j * (2 * n - j + 1)) / 2);
}

// Expands packed triangular storage into a dense column-major n-by-n matrix,
// filling only the triangle selected by `uplo`.
template <typename T> std::vector<T> expand_packed(char uplo, int n, const std::vector<T> &ap) {
    std::vector<T> dense(static_cast<std::size_t>(n) * uz(n), T{});
    for (int j = 0; j < n; ++j) {
        const int first = (uplo == 'U') ? 0 : j;
        const int last = (uplo == 'U') ? j : (n - 1);
        for (int i = first; i <= last; ++i) {
            dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * uz(n)] =
                ap[static_cast<std::size_t>(packed_index(uplo, i, j, n))];
        }
    }
    return dense;
}

// Builds a packed triangle whose entries are a simple deterministic pattern.
template <typename T> std::vector<T> make_packed_real(char uplo, int n) {
    std::vector<T> ap(static_cast<std::size_t>(n) * (uz(n) + 1) / 2, T{});
    for (int j = 0; j < n; ++j) {
        const int first = (uplo == 'U') ? 0 : j;
        const int last = (uplo == 'U') ? j : (n - 1);
        for (int i = first; i <= last; ++i) {
            ap[static_cast<std::size_t>(packed_index(uplo, i, j, n))] =
                static_cast<T>(1 + ((i * 3 + j * 5) % 4)) / static_cast<T>(2);
        }
    }
    return ap;
}

void test_spmv_matches_symv() {
    const int n = 4;
    for (const char uplo : {'U', 'L'}) {
        const std::vector<float> ap = make_packed_real<float>(uplo, n);
        const std::vector<float> dense = expand_packed(uplo, n, ap);
        const std::array<float, 4> x = {1.0F, -2.0F, 3.0F, 0.5F};
        std::array<float, 4> y_packed = {1.0F, 0.0F, -1.0F, 2.0F};
        std::array<float, 4> y_dense = y_packed;
        thefblas::spmv(uplo, n, 1.5F, ap.data(), x.data(), 1, 0.25F, y_packed.data(), 1);
        thefblas::symv(uplo, n, 1.5F, dense.data(), n, x.data(), 1, 0.25F, y_dense.data(), 1);
        for (int i = 0; i < n; ++i) {
            assert(almost_equal_scalar(y_packed[uz(i)], y_dense[uz(i)]));
        }
    }
}

void test_spmv_fixed_and_invalid_args() {
    // Upper packed [[1, 2], [2, 3]].
    const int n = 2;
    std::vector<fixed32> ap = {fixed32(1), fixed32(2), fixed32(3)};
    const std::array<fixed32, 2> x = {fixed32(1), fixed32(1)};
    std::array<fixed32, 2> y = {fixed32(0), fixed32(0)};
    thefblas::spmv('U', n, fixed32(1), ap.data(), x.data(), 1, fixed32(0), y.data(), 1);
    assert(almost_equal_scalar(y[0], fixed32(3)));
    assert(almost_equal_scalar(y[1], fixed32(5)));

    thefblas::spmv('X', n, fixed32(1), ap.data(), x.data(), 1, fixed32(0), y.data(), 1);
    thefblas::spmv('U', n, fixed32(1), ap.data(), x.data(), 0, fixed32(0), y.data(), 1);
    assert(almost_equal_scalar(y[0], fixed32(3)));
    assert(almost_equal_scalar(y[1], fixed32(5)));
}

void test_hpmv_matches_hemv() {
    const int n = 3;
    for (const char uplo : {'U', 'L'}) {
        std::vector<cfloat> ap(static_cast<std::size_t>(n) * (uz(n) + 1) / 2, cfloat(0.0F, 0.0F));
        for (int j = 0; j < n; ++j) {
            const int first = (uplo == 'U') ? 0 : j;
            const int last = (uplo == 'U') ? j : (n - 1);
            for (int i = first; i <= last; ++i) {
                const float re = static_cast<float>(1 + i + j);
                const float im = (i == j) ? 0.0F : ((uplo == 'U') ? 1.0F : -1.0F);
                ap[static_cast<std::size_t>(packed_index(uplo, i, j, n))] = cfloat(re, im);
            }
        }
        const std::vector<cfloat> dense = expand_packed(uplo, n, ap);
        const std::array<cfloat, 3> x = {cfloat(1.0F, 1.0F), cfloat(2.0F, 0.0F),
                                         cfloat(0.0F, -1.0F)};
        std::array<cfloat, 3> y_packed = {cfloat(1.0F, 0.0F), cfloat(0.0F, 1.0F),
                                          cfloat(-1.0F, 0.0F)};
        std::array<cfloat, 3> y_dense = y_packed;
        thefblas::hpmv(uplo, n, cfloat(2.0F, 1.0F), ap.data(), x.data(), 1, cfloat(0.5F, 0.0F),
                       y_packed.data(), 1);
        thefblas::hemv(uplo, n, cfloat(2.0F, 1.0F), dense.data(), n, x.data(), 1,
                       cfloat(0.5F, 0.0F), y_dense.data(), 1);
        for (int i = 0; i < n; ++i) {
            assert(almost_equal_scalar(y_packed[uz(i)], y_dense[uz(i)]));
        }
    }
}

void test_hpmv_complex_fixed() {
    // Upper packed Hermitian [[1, i], [-i, 2]].
    std::vector<cfixed> ap = {cfixed(fixed32(1), fixed32(0)), cfixed(fixed32(0), fixed32(1)),
                              cfixed(fixed32(2), fixed32(0))};
    const std::array<cfixed, 2> x = {cfixed(fixed32(1), fixed32(0)),
                                     cfixed(fixed32(1), fixed32(0))};
    std::array<cfixed, 2> y = {cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(0), fixed32(0))};
    thefblas::hpmv('U', 2, cfixed(fixed32(1), fixed32(0)), ap.data(), x.data(), 1,
                   cfixed(fixed32(0), fixed32(0)), y.data(), 1);
    assert(almost_equal_scalar(y[0], cfixed(fixed32(1), fixed32(1))));
    assert(almost_equal_scalar(y[1], cfixed(fixed32(2), fixed32(-1))));
}

void test_tpmv_and_tpsv_match_dense() {
    const int n = 4;
    for (const char uplo : {'U', 'L'}) {
        std::vector<float> ap = make_packed_real<float>(uplo, n);
        // Make the diagonal dominant so the triangular solve is well conditioned.
        for (int j = 0; j < n; ++j) {
            ap[static_cast<std::size_t>(packed_index(uplo, j, j, n))] = 4.0F;
        }
        const std::vector<float> dense = expand_packed(uplo, n, ap);
        for (const char diag : {'N', 'U'}) {
            for (const char trans : {'N', 'T', 'C'}) {
                const std::array<float, 4> x0 = {1.0F, -2.0F, 3.0F, 0.5F};
                std::array<float, 4> x_packed = x0;
                std::array<float, 4> x_dense = x0;
                thefblas::tpmv(uplo, trans, diag, n, ap.data(), x_packed.data(), 1);
                thefblas::trmv(uplo, trans, diag, n, dense.data(), n, x_dense.data(), 1);
                for (int i = 0; i < n; ++i) {
                    assert(almost_equal_scalar(x_packed[uz(i)], x_dense[uz(i)]));
                }

                thefblas::tpsv(uplo, trans, diag, n, ap.data(), x_packed.data(), 1);
                for (int i = 0; i < n; ++i) {
                    assert(almost_equal_scalar(x_packed[uz(i)], x0[uz(i)], 1e-4));
                }
            }
        }
    }
}

void test_tpmv_negative_stride_and_complex() {
    const int n = 3;
    const std::vector<float> ap = make_packed_real<float>('L', n);
    const std::vector<float> dense = expand_packed('L', n, ap);
    const std::array<float, 3> x0 = {1.0F, 2.0F, 3.0F};
    std::array<float, 3> x_packed = x0;
    std::array<float, 3> x_dense = x0;
    thefblas::tpmv('L', 'N', 'N', n, ap.data(), x_packed.data(), -1);
    thefblas::trmv('L', 'N', 'N', n, dense.data(), n, x_dense.data(), -1);
    for (int i = 0; i < n; ++i) {
        assert(almost_equal_scalar(x_packed[uz(i)], x_dense[uz(i)]));
    }

    // Conjugate-transpose solve against the dense equivalent.
    std::vector<cfloat> ap_c = {cfloat(2.0F, 0.0F), cfloat(1.0F, 1.0F), cfloat(3.0F, 0.0F)};
    const std::vector<cfloat> dense_c = expand_packed('U', 2, ap_c);
    std::array<cfloat, 2> xc_packed = {cfloat(2.0F, 0.0F), cfloat(4.0F, -1.0F)};
    std::array<cfloat, 2> xc_dense = xc_packed;
    thefblas::tpsv('U', 'C', 'N', 2, ap_c.data(), xc_packed.data(), 1);
    thefblas::trsv('U', 'C', 'N', 2, dense_c.data(), 2, xc_dense.data(), 1);
    for (int i = 0; i < 2; ++i) {
        assert(almost_equal_scalar(xc_packed[uz(i)], xc_dense[uz(i)]));
    }

    // Invalid characters leave the vector untouched.
    const std::array<cfloat, 2> before = xc_packed;
    thefblas::tpmv('U', 'X', 'N', 2, ap_c.data(), xc_packed.data(), 1);
    thefblas::tpsv('Z', 'N', 'N', 2, ap_c.data(), xc_packed.data(), 1);
    for (int i = 0; i < 2; ++i) {
        assert(almost_equal_scalar(xc_packed[uz(i)], before[uz(i)]));
    }
}

void test_spr_and_spr2_match_dense() {
    const int n = 3;
    for (const char uplo : {'U', 'L'}) {
        std::vector<float> ap = make_packed_real<float>(uplo, n);
        std::vector<float> dense = expand_packed(uplo, n, ap);
        const std::array<float, 3> x = {1.0F, -2.0F, 0.5F};
        const std::array<float, 3> y = {2.0F, 1.0F, -1.0F};

        thefblas::spr(uplo, n, 1.5F, x.data(), 1, ap.data());
        thefblas::syr(uplo, n, 1.5F, x.data(), 1, dense.data(), n);
        thefblas::spr2(uplo, n, -0.5F, x.data(), 1, y.data(), 1, ap.data());
        thefblas::syr2(uplo, n, -0.5F, x.data(), 1, y.data(), 1, dense.data(), n);

        for (int j = 0; j < n; ++j) {
            const int first = (uplo == 'U') ? 0 : j;
            const int last = (uplo == 'U') ? j : (n - 1);
            for (int i = first; i <= last; ++i) {
                assert(almost_equal_scalar(
                    ap[static_cast<std::size_t>(packed_index(uplo, i, j, n))],
                    dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * uz(n)]));
            }
        }
    }
}

void test_spr_fixed_and_invalid_args() {
    // Lower packed 2x2 zero matrix updated with alpha * x * x^T, x = [1, 2].
    std::vector<fixed32> ap = {fixed32(0), fixed32(0), fixed32(0)};
    const std::array<fixed32, 2> x = {fixed32(1), fixed32(2)};
    thefblas::spr('L', 2, fixed32(2), x.data(), 1, ap.data());
    assert(almost_equal_scalar(ap[0], fixed32(2))); // a(0,0) = 2 * 1 * 1
    assert(almost_equal_scalar(ap[1], fixed32(4))); // a(1,0) = 2 * 2 * 1
    assert(almost_equal_scalar(ap[2], fixed32(8))); // a(1,1) = 2 * 2 * 2

    thefblas::spr('X', 2, fixed32(2), x.data(), 1, ap.data());
    thefblas::spr('L', 2, fixed32(0), x.data(), 1, ap.data());
    thefblas::spr2('L', 2, fixed32(2), x.data(), 0, x.data(), 1, ap.data());
    assert(almost_equal_scalar(ap[0], fixed32(2)));
    assert(almost_equal_scalar(ap[1], fixed32(4)));
    assert(almost_equal_scalar(ap[2], fixed32(8)));
}

void test_hpr_and_hpr2_match_dense() {
    const int n = 3;
    for (const char uplo : {'U', 'L'}) {
        std::vector<cfloat> ap(static_cast<std::size_t>(n) * (uz(n) + 1) / 2, cfloat(0.0F, 0.0F));
        for (int j = 0; j < n; ++j) {
            const int first = (uplo == 'U') ? 0 : j;
            const int last = (uplo == 'U') ? j : (n - 1);
            for (int i = first; i <= last; ++i) {
                const float re = static_cast<float>(1 + i + j);
                const float im = (i == j) ? 0.0F : ((uplo == 'U') ? 1.0F : -1.0F);
                ap[static_cast<std::size_t>(packed_index(uplo, i, j, n))] = cfloat(re, im);
            }
        }
        std::vector<cfloat> dense = expand_packed(uplo, n, ap);
        const std::array<cfloat, 3> x = {cfloat(1.0F, 1.0F), cfloat(2.0F, 0.0F),
                                         cfloat(0.0F, -1.0F)};
        const std::array<cfloat, 3> y = {cfloat(0.5F, 0.0F), cfloat(-1.0F, 2.0F),
                                         cfloat(1.0F, 1.0F)};

        thefblas::hpr(uplo, n, 1.5F, x.data(), 1, ap.data());
        thefblas::her(uplo, n, 1.5F, x.data(), 1, dense.data(), n);
        thefblas::hpr2(uplo, n, cfloat(0.5F, -1.0F), x.data(), 1, y.data(), 1, ap.data());
        thefblas::her2(uplo, n, cfloat(0.5F, -1.0F), x.data(), 1, y.data(), 1, dense.data(), n);

        for (int j = 0; j < n; ++j) {
            const int first = (uplo == 'U') ? 0 : j;
            const int last = (uplo == 'U') ? j : (n - 1);
            for (int i = first; i <= last; ++i) {
                assert(almost_equal_scalar(
                    ap[static_cast<std::size_t>(packed_index(uplo, i, j, n))],
                    dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * uz(n)]));
            }
        }
        // The Hermitian diagonal must stay real.
        for (int j = 0; j < n; ++j) {
            assert(almost_equal_scalar(
                ap[static_cast<std::size_t>(packed_index(uplo, j, j, n))].imag(), 0.0F));
        }
    }
}

void test_hpr_complex_fixed() {
    // Upper packed zero matrix, x = [1, i], alpha = 1: a <- x * x^H.
    std::vector<cfixed> ap = {cfixed(fixed32(0), fixed32(0)), cfixed(fixed32(0), fixed32(0)),
                              cfixed(fixed32(0), fixed32(0))};
    const std::array<cfixed, 2> x = {cfixed(fixed32(1), fixed32(0)),
                                     cfixed(fixed32(0), fixed32(1))};
    thefblas::hpr('U', 2, fixed32(1), x.data(), 1, ap.data());
    assert(almost_equal_scalar(ap[0], cfixed(fixed32(1), fixed32(0))));  // a(0,0) = 1
    assert(almost_equal_scalar(ap[1], cfixed(fixed32(0), fixed32(-1)))); // a(0,1) = -i
    assert(almost_equal_scalar(ap[2], cfixed(fixed32(1), fixed32(0))));  // a(1,1) = 1
}

} // namespace

int main() {
    test_spmv_matches_symv();
    test_spmv_fixed_and_invalid_args();
    test_hpmv_matches_hemv();
    test_hpmv_complex_fixed();
    test_tpmv_and_tpsv_match_dense();
    test_tpmv_negative_stride_and_complex();
    test_spr_and_spr2_match_dense();
    test_spr_fixed_and_invalid_args();
    test_hpr_and_hpr2_match_dense();
    test_hpr_complex_fixed();
}
