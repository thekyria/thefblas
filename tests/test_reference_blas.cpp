// Randomised comparison of thefblas against a reference CBLAS (Netlib, OpenBLAS,
// ...) for float, double, std::complex<float> and std::complex<double>.
//
// Every trial draws sizes, strides (including negative ones), leading-dimension
// padding, uplo/trans/diag and scalars at random, runs the same call through
// thefblas and through CBLAS on identical copies of the inputs, and compares the
// whole output buffers. Comparing whole buffers also catches writes to elements
// a routine must not touch (stride gaps, lda padding, the other triangle).
//
// The seed defaults to a fixed value so runs are reproducible; set
// THEFBLAS_TEST_SEED to explore other inputs. Failures print the routine, the
// element type, the parameters and the first mismatching element.

#include "thefblas/thefblas.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

template <typename T> struct real_of {
    using type = T;
};
template <typename T> struct real_of<std::complex<T>> {
    using type = T;
};
template <typename T> using real_t = typename real_of<T>::type;

template <typename T> constexpr bool is_complex_v = !std::is_same_v<T, real_t<T>>;

constexpr int trials_per_routine = 200;
constexpr int max_size = 12;
constexpr int max_reports = 20;

std::mt19937 rng; // NOLINT(cert-msc32-c,cert-msc51-cpp): seeded in main()
long checks = 0;
long failures = 0;

// ---------------------------------------------------------------------------
// Random inputs
// ---------------------------------------------------------------------------

int uniform_int(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
}

template <typename T> T random_value() {
    if constexpr (is_complex_v<T>) {
        return T(random_value<real_t<T>>(), random_value<real_t<T>>());
    } else {
        return std::uniform_real_distribution<T>(T(-1), T(1))(rng);
    }
}

/// Scalar argument: mostly random, but 0 and 1 often enough to reach the
/// special-cased paths (beta == 0 overwrites y, alpha == 0 is a quick return).
template <typename T> T random_scalar() {
    switch (uniform_int(0, 7)) {
    case 0:
        return T(0);
    case 1:
        return T(1);
    default:
        return random_value<T>();
    }
}

int random_inc() {
    static const int incs[] = {-3, -2, -1, 1, 2, 3};
    return incs[uniform_int(0, 5)];
}

char random_of(const char *choices) {
    const int count = static_cast<int>(std::char_traits<char>::length(choices));
    return choices[uniform_int(0, count - 1)];
}

template <typename T> std::vector<T> random_buffer(std::size_t size) {
    std::vector<T> buffer(std::max<std::size_t>(size, 1));
    std::generate(buffer.begin(), buffer.end(), random_value<T>);
    return buffer;
}

/// Storage for a strided vector of `n` logical elements.
template <typename T> std::vector<T> random_vector(int n, int inc) {
    const std::size_t len =
        n > 0 ? 1 + static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(std::abs(inc)) : 1;
    return random_buffer<T>(len);
}

// ---------------------------------------------------------------------------
// Comparison and reporting
// ---------------------------------------------------------------------------

template <typename T> const char *type_name() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, cfloat>) {
        return "complex<float>";
    } else {
        return "complex<double>";
    }
}

template <typename... Args> std::string params(const Args &...args) {
    std::ostringstream os;
    os.precision(9);
    ((os << args), ...);
    return os.str();
}

template <typename T> std::string label(const char *routine, const std::string &args) {
    return std::string(routine) + "<" + type_name<T>() + "> " + args;
}

/// Both sides evaluate the same algorithm in (nearly) the same order, so they
/// agree to a few ulps; the bound leaves room for a reference built with FMA
/// or a different summation order (OpenBLAS kernels).
template <typename T> bool near(T got, T ref) {
    if (got == ref) {
        return true; // also covers equal infinities, whose difference is NaN
    }
    using R = real_t<T>;
    const double tol = 256.0 * static_cast<double>(std::numeric_limits<R>::epsilon());
    const auto diff = static_cast<double>(std::abs(got - ref));
    return diff <= tol * std::max(1.0, static_cast<double>(std::abs(ref)));
}

void report(const std::string &what, const std::string &detail) {
    ++failures;
    if (failures <= max_reports) {
        std::fprintf(stderr, "MISMATCH %s: %s\n", what.c_str(), detail.c_str());
    }
}

template <typename T> void check_scalar(const std::string &what, T got, T ref) {
    ++checks;
    if (!near(got, ref)) {
        report(what, params("got ", got, ", reference ", ref));
    }
}

void check_index(const std::string &what, int got, int ref) {
    ++checks;
    if (got != ref) {
        report(what, params("got ", got, ", reference ", ref));
    }
}

template <typename T>
void check_buffer(const std::string &what, const std::vector<T> &got, const std::vector<T> &ref) {
    ++checks;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!near(got[i], ref[i])) {
            report(what, params("element ", i, ": got ", got[i], ", reference ", ref[i]));
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// CBLAS dispatch (column-major, Netlib argument order)
// ---------------------------------------------------------------------------

CBLAS_TRANSPOSE cb_trans(char trans) {
    if (trans == 'N') {
        return CblasNoTrans;
    }
    return trans == 'T' ? CblasTrans : CblasConjTrans;
}

CBLAS_UPLO cb_uplo(char uplo) {
    return uplo == 'U' ? CblasUpper : CblasLower;
}

CBLAS_DIAG cb_diag(char diag) {
    return diag == 'U' ? CblasUnit : CblasNonUnit;
}

namespace ref {

template <typename T> void swap(int n, T *x, int incx, T *y, int incy) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_sswap(n, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dswap(n, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cswap(n, x, incx, y, incy);
    } else {
        cblas_zswap(n, x, incx, y, incy);
    }
}

template <typename T> void copy(int n, const T *x, int incx, T *y, int incy) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_scopy(n, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dcopy(n, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ccopy(n, x, incx, y, incy);
    } else {
        cblas_zcopy(n, x, incx, y, incy);
    }
}

template <typename T> void axpy(int n, T alpha, const T *x, int incx, T *y, int incy) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(n, alpha, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(n, alpha, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_caxpy(n, &alpha, x, incx, y, incy);
    } else {
        cblas_zaxpy(n, &alpha, x, incx, y, incy);
    }
}

template <typename T> void scal(int n, T alpha, T *x, int incx) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(n, alpha, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(n, alpha, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cscal(n, &alpha, x, incx);
    } else {
        cblas_zscal(n, &alpha, x, incx);
    }
}

/// Complex vector scaled by a real scalar (csscal / zdscal).
template <typename R> void scal_real(int n, R alpha, std::complex<R> *x, int incx) {
    if constexpr (std::is_same_v<R, float>) {
        cblas_csscal(n, alpha, x, incx);
    } else {
        cblas_zdscal(n, alpha, x, incx);
    }
}

template <typename R> R dot(int n, const R *x, int incx, const R *y, int incy) {
    if constexpr (std::is_same_v<R, float>) {
        return cblas_sdot(n, x, incx, y, incy);
    } else {
        return cblas_ddot(n, x, incx, y, incy);
    }
}

float sdsdot(int n, float sb, const float *x, int incx, const float *y, int incy) {
    return cblas_sdsdot(n, sb, x, incx, y, incy);
}

double dsdot(int n, const float *x, int incx, const float *y, int incy) {
    return cblas_dsdot(n, x, incx, y, incy);
}

template <typename R>
std::complex<R> dotu(int n, const std::complex<R> *x, int incx, const std::complex<R> *y,
                     int incy) {
    std::complex<R> result;
    if constexpr (std::is_same_v<R, float>) {
        cblas_cdotu_sub(n, x, incx, y, incy, &result);
    } else {
        cblas_zdotu_sub(n, x, incx, y, incy, &result);
    }
    return result;
}

template <typename R>
std::complex<R> dotc(int n, const std::complex<R> *x, int incx, const std::complex<R> *y,
                     int incy) {
    std::complex<R> result;
    if constexpr (std::is_same_v<R, float>) {
        cblas_cdotc_sub(n, x, incx, y, incy, &result);
    } else {
        cblas_zdotc_sub(n, x, incx, y, incy, &result);
    }
    return result;
}

template <typename T> real_t<T> nrm2(int n, const T *x, int incx) {
    if constexpr (std::is_same_v<T, float>) {
        return cblas_snrm2(n, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_dnrm2(n, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        return cblas_scnrm2(n, x, incx);
    } else {
        return cblas_dznrm2(n, x, incx);
    }
}

template <typename T> real_t<T> asum(int n, const T *x, int incx) {
    if constexpr (std::is_same_v<T, float>) {
        return cblas_sasum(n, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_dasum(n, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        return cblas_scasum(n, x, incx);
    } else {
        return cblas_dzasum(n, x, incx);
    }
}

/// CBLAS returns a 0-based index (and 0 for an empty vector).
template <typename T> int iamax(int n, const T *x, int incx) {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<int>(cblas_isamax(n, x, incx));
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<int>(cblas_idamax(n, x, incx));
    } else if constexpr (std::is_same_v<T, cfloat>) {
        return static_cast<int>(cblas_icamax(n, x, incx));
    } else {
        return static_cast<int>(cblas_izamax(n, x, incx));
    }
}

/// Plane rotation with real c and s (srot/drot, csrot/zdrot).
template <typename T> void rot(int n, T *x, int incx, T *y, int incy, real_t<T> c, real_t<T> s) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_srot(n, x, incx, y, incy, c, s);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_drot(n, x, incx, y, incy, c, s);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_csrot(n, x, incx, y, incy, c, s);
    } else {
        cblas_zdrot(n, x, incx, y, incy, c, s);
    }
}

template <typename R> void rotg(R *a, R *b, R *c, R *s) {
    if constexpr (std::is_same_v<R, float>) {
        cblas_srotg(a, b, c, s);
    } else {
        cblas_drotg(a, b, c, s);
    }
}

template <typename R> void rotg(std::complex<R> *a, std::complex<R> b, R *c, std::complex<R> *s) {
    if constexpr (std::is_same_v<R, float>) {
        cblas_crotg(a, &b, c, s);
    } else {
        cblas_zrotg(a, &b, c, s);
    }
}

template <typename R> void rotm(int n, R *x, int incx, R *y, int incy, const R *param) {
    if constexpr (std::is_same_v<R, float>) {
        cblas_srotm(n, x, incx, y, incy, param);
    } else {
        cblas_drotm(n, x, incx, y, incy, param);
    }
}

template <typename R> void rotmg(R *d1, R *d2, R *b1, R b2, R *param) {
    if constexpr (std::is_same_v<R, float>) {
        cblas_srotmg(d1, d2, b1, b2, param);
    } else {
        cblas_drotmg(d1, d2, b1, b2, param);
    }
}

template <typename T>
void gemv(char trans, int m, int n, T alpha, const T *a, int lda, const T *x, int incx, T beta,
          T *y, int incy) {
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemv(CblasColMajor, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemv(CblasColMajor, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cgemv(CblasColMajor, t, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    } else {
        cblas_zgemv(CblasColMajor, t, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

template <typename T>
void gbmv(char trans, int m, int n, int kl, int ku, T alpha, const T *a, int lda, const T *x,
          int incx, T beta, T *y, int incy) {
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgbmv(CblasColMajor, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgbmv(CblasColMajor, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cgbmv(CblasColMajor, t, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    } else {
        cblas_zgbmv(CblasColMajor, t, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/// symv for real types, hemv for complex types.
template <typename T>
void symv(char uplo, int n, T alpha, const T *a, int lda, const T *x, int incx, T beta, T *y,
          int incy) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssymv(CblasColMajor, u, n, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsymv(CblasColMajor, u, n, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_chemv(CblasColMajor, u, n, &alpha, a, lda, x, incx, &beta, y, incy);
    } else {
        cblas_zhemv(CblasColMajor, u, n, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/// sbmv for real types, hbmv for complex types.
template <typename T>
void sbmv(char uplo, int n, int k, T alpha, const T *a, int lda, const T *x, int incx, T beta, T *y,
          int incy) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssbmv(CblasColMajor, u, n, k, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsbmv(CblasColMajor, u, n, k, alpha, a, lda, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_chbmv(CblasColMajor, u, n, k, &alpha, a, lda, x, incx, &beta, y, incy);
    } else {
        cblas_zhbmv(CblasColMajor, u, n, k, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/// spmv for real types, hpmv for complex types.
template <typename T>
void spmv(char uplo, int n, T alpha, const T *ap, const T *x, int incx, T beta, T *y, int incy) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sspmv(CblasColMajor, u, n, alpha, ap, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dspmv(CblasColMajor, u, n, alpha, ap, x, incx, beta, y, incy);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_chpmv(CblasColMajor, u, n, &alpha, ap, x, incx, &beta, y, incy);
    } else {
        cblas_zhpmv(CblasColMajor, u, n, &alpha, ap, x, incx, &beta, y, incy);
    }
}

template <typename T>
void trmv(char uplo, char trans, char diag, int n, const T *a, int lda, T *x, int incx) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_strmv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtrmv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctrmv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    } else {
        cblas_ztrmv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    }
}

template <typename T>
void trsv(char uplo, char trans, char diag, int n, const T *a, int lda, T *x, int incx) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_strsv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtrsv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctrsv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    } else {
        cblas_ztrsv(CblasColMajor, u, t, d, n, a, lda, x, incx);
    }
}

template <typename T>
void tbmv(char uplo, char trans, char diag, int n, int k, const T *a, int lda, T *x, int incx) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_stbmv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtbmv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctbmv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    } else {
        cblas_ztbmv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    }
}

template <typename T>
void tbsv(char uplo, char trans, char diag, int n, int k, const T *a, int lda, T *x, int incx) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_stbsv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtbsv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctbsv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    } else {
        cblas_ztbsv(CblasColMajor, u, t, d, n, k, a, lda, x, incx);
    }
}

template <typename T>
void tpmv(char uplo, char trans, char diag, int n, const T *ap, T *x, int incx) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_stpmv(CblasColMajor, u, t, d, n, ap, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtpmv(CblasColMajor, u, t, d, n, ap, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctpmv(CblasColMajor, u, t, d, n, ap, x, incx);
    } else {
        cblas_ztpmv(CblasColMajor, u, t, d, n, ap, x, incx);
    }
}

template <typename T>
void tpsv(char uplo, char trans, char diag, int n, const T *ap, T *x, int incx) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_stpsv(CblasColMajor, u, t, d, n, ap, x, incx);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtpsv(CblasColMajor, u, t, d, n, ap, x, incx);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctpsv(CblasColMajor, u, t, d, n, ap, x, incx);
    } else {
        cblas_ztpsv(CblasColMajor, u, t, d, n, ap, x, incx);
    }
}

/// ger for real types; for complex types `conjugate` selects gerc over geru.
template <typename T>
void ger(bool conjugate, int m, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a,
         int lda) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        if (conjugate) {
            cblas_cgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, a, lda);
        } else {
            cblas_cgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, a, lda);
        }
    } else {
        if (conjugate) {
            cblas_zgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, a, lda);
        } else {
            cblas_zgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, a, lda);
        }
    }
}

/// syr for real types, her for complex types (alpha is real in both).
template <typename T>
void syr(char uplo, int n, real_t<T> alpha, const T *x, int incx, T *a, int lda) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssyr(CblasColMajor, u, n, alpha, x, incx, a, lda);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsyr(CblasColMajor, u, n, alpha, x, incx, a, lda);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cher(CblasColMajor, u, n, alpha, x, incx, a, lda);
    } else {
        cblas_zher(CblasColMajor, u, n, alpha, x, incx, a, lda);
    }
}

/// spr for real types, hpr for complex types (alpha is real in both).
template <typename T> void spr(char uplo, int n, real_t<T> alpha, const T *x, int incx, T *ap) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sspr(CblasColMajor, u, n, alpha, x, incx, ap);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dspr(CblasColMajor, u, n, alpha, x, incx, ap);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_chpr(CblasColMajor, u, n, alpha, x, incx, ap);
    } else {
        cblas_zhpr(CblasColMajor, u, n, alpha, x, incx, ap);
    }
}

/// syr2 for real types, her2 for complex types.
template <typename T>
void syr2(char uplo, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a, int lda) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssyr2(CblasColMajor, u, n, alpha, x, incx, y, incy, a, lda);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsyr2(CblasColMajor, u, n, alpha, x, incx, y, incy, a, lda);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cher2(CblasColMajor, u, n, &alpha, x, incx, y, incy, a, lda);
    } else {
        cblas_zher2(CblasColMajor, u, n, &alpha, x, incx, y, incy, a, lda);
    }
}

/// spr2 for real types, hpr2 for complex types.
template <typename T>
void spr2(char uplo, int n, T alpha, const T *x, int incx, const T *y, int incy, T *ap) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sspr2(CblasColMajor, u, n, alpha, x, incx, y, incy, ap);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dspr2(CblasColMajor, u, n, alpha, x, incx, y, incy, ap);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_chpr2(CblasColMajor, u, n, &alpha, x, incx, y, incy, ap);
    } else {
        cblas_zhpr2(CblasColMajor, u, n, &alpha, x, incx, y, incy, ap);
    }
}

template <typename T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda, const T *b,
          int ldb, T beta, T *c, int ldc) {
    const CBLAS_TRANSPOSE ta = cb_trans(transa);
    const CBLAS_TRANSPOSE tb = cb_trans(transb);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasColMajor, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_cgemm(CblasColMajor, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    } else {
        cblas_zgemm(CblasColMajor, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

template <typename T>
void syrk(char uplo, char trans, int n, int k, T alpha, const T *a, int lda, T beta, T *c,
          int ldc) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssyrk(CblasColMajor, u, t, n, k, alpha, a, lda, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsyrk(CblasColMajor, u, t, n, k, alpha, a, lda, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_csyrk(CblasColMajor, u, t, n, k, &alpha, a, lda, &beta, c, ldc);
    } else {
        cblas_zsyrk(CblasColMajor, u, t, n, k, &alpha, a, lda, &beta, c, ldc);
    }
}

template <typename R>
void herk(char uplo, char trans, int n, int k, R alpha, const std::complex<R> *a, int lda, R beta,
          std::complex<R> *c, int ldc) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    if constexpr (std::is_same_v<R, float>) {
        cblas_cherk(CblasColMajor, u, t, n, k, alpha, a, lda, beta, c, ldc);
    } else {
        cblas_zherk(CblasColMajor, u, t, n, k, alpha, a, lda, beta, c, ldc);
    }
}

template <typename T>
void symm(char side, char uplo, int m, int n, T alpha, const T *a, int lda, const T *b, int ldb,
          T beta, T *c, int ldc) {
    const CBLAS_SIDE sd = side == 'L' ? CblasLeft : CblasRight;
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssymm(CblasColMajor, sd, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsymm(CblasColMajor, sd, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_csymm(CblasColMajor, sd, u, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    } else {
        cblas_zsymm(CblasColMajor, sd, u, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

template <typename R>
void hemm(char side, char uplo, int m, int n, std::complex<R> alpha, const std::complex<R> *a,
          int lda, const std::complex<R> *b, int ldb, std::complex<R> beta, std::complex<R> *c,
          int ldc) {
    const CBLAS_SIDE sd = side == 'L' ? CblasLeft : CblasRight;
    const CBLAS_UPLO u = cb_uplo(uplo);
    if constexpr (std::is_same_v<R, float>) {
        cblas_chemm(CblasColMajor, sd, u, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    } else {
        cblas_zhemm(CblasColMajor, sd, u, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

template <typename T>
void syr2k(char uplo, char trans, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb,
           T beta, T *c, int ldc) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    if constexpr (std::is_same_v<T, float>) {
        cblas_ssyr2k(CblasColMajor, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dsyr2k(CblasColMajor, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_csyr2k(CblasColMajor, u, t, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    } else {
        cblas_zsyr2k(CblasColMajor, u, t, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

template <typename R>
void her2k(char uplo, char trans, int n, int k, std::complex<R> alpha, const std::complex<R> *a,
           int lda, const std::complex<R> *b, int ldb, R beta, std::complex<R> *c, int ldc) {
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(trans);
    if constexpr (std::is_same_v<R, float>) {
        cblas_cher2k(CblasColMajor, u, t, n, k, &alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        cblas_zher2k(CblasColMajor, u, t, n, k, &alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

template <typename T>
void trmm(char side, char uplo, char transa, char diag, int m, int n, T alpha, const T *a, int lda,
          T *b, int ldb) {
    const CBLAS_SIDE sd = side == 'L' ? CblasLeft : CblasRight;
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(transa);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_strmm(CblasColMajor, sd, u, t, d, m, n, alpha, a, lda, b, ldb);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtrmm(CblasColMajor, sd, u, t, d, m, n, alpha, a, lda, b, ldb);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctrmm(CblasColMajor, sd, u, t, d, m, n, &alpha, a, lda, b, ldb);
    } else {
        cblas_ztrmm(CblasColMajor, sd, u, t, d, m, n, &alpha, a, lda, b, ldb);
    }
}

template <typename T>
void trsm(char side, char uplo, char transa, char diag, int m, int n, T alpha, const T *a, int lda,
          T *b, int ldb) {
    const CBLAS_SIDE sd = side == 'L' ? CblasLeft : CblasRight;
    const CBLAS_UPLO u = cb_uplo(uplo);
    const CBLAS_TRANSPOSE t = cb_trans(transa);
    const CBLAS_DIAG d = cb_diag(diag);
    if constexpr (std::is_same_v<T, float>) {
        cblas_strsm(CblasColMajor, sd, u, t, d, m, n, alpha, a, lda, b, ldb);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtrsm(CblasColMajor, sd, u, t, d, m, n, alpha, a, lda, b, ldb);
    } else if constexpr (std::is_same_v<T, cfloat>) {
        cblas_ctrsm(CblasColMajor, sd, u, t, d, m, n, &alpha, a, lda, b, ldb);
    } else {
        cblas_ztrsm(CblasColMajor, sd, u, t, d, m, n, &alpha, a, lda, b, ldb);
    }
}

} // namespace ref

// ---------------------------------------------------------------------------
// Level 1
// ---------------------------------------------------------------------------

template <typename T> void test_level1() {
    using R = real_t<T>;
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const int n = uniform_int(0, max_size);
        const int incx = random_inc();
        const int incy = random_inc();
        const std::vector<T> x = random_vector<T>(n, incx);
        const std::vector<T> y = random_vector<T>(n, incy);
        const T alpha = random_scalar<T>();
        const std::string p = params("n=", n, " incx=", incx, " incy=", incy);

        {
            std::vector<T> x1 = x;
            std::vector<T> y1 = y;
            std::vector<T> x2 = x;
            std::vector<T> y2 = y;
            thefblas::swap(n, x1.data(), incx, y1.data(), incy);
            ref::swap(n, x2.data(), incx, y2.data(), incy);
            check_buffer(label<T>("swap x", p), x1, x2);
            check_buffer(label<T>("swap y", p), y1, y2);
        }
        {
            std::vector<T> y1 = y;
            std::vector<T> y2 = y;
            thefblas::copy(n, x.data(), incx, y1.data(), incy);
            ref::copy(n, x.data(), incx, y2.data(), incy);
            check_buffer(label<T>("copy", p), y1, y2);
        }
        {
            std::vector<T> y1 = y;
            std::vector<T> y2 = y;
            thefblas::axpy(n, alpha, x.data(), incx, y1.data(), incy);
            ref::axpy(n, alpha, x.data(), incx, y2.data(), incy);
            check_buffer(label<T>("axpy", p + params(" alpha=", alpha)), y1, y2);
        }
        {
            std::vector<T> x1 = x;
            std::vector<T> x2 = x;
            thefblas::scal(n, alpha, x1.data(), incx);
            ref::scal(n, alpha, x2.data(), incx);
            check_buffer(label<T>("scal", p + params(" alpha=", alpha)), x1, x2);
        }
        if constexpr (is_complex_v<T>) {
            const R real_alpha = random_scalar<R>();
            std::vector<T> x1 = x;
            std::vector<T> x2 = x;
            thefblas::scal(n, real_alpha, x1.data(), incx);
            ref::scal_real(n, real_alpha, x2.data(), incx);
            check_buffer(label<T>("scal (real alpha)", p + params(" alpha=", real_alpha)), x1, x2);

            check_scalar(label<T>("dotu", p), thefblas::dotu(n, x.data(), incx, y.data(), incy),
                         ref::dotu(n, x.data(), incx, y.data(), incy));
            check_scalar(label<T>("dotc", p), thefblas::dotc(n, x.data(), incx, y.data(), incy),
                         ref::dotc(n, x.data(), incx, y.data(), incy));
        } else {
            check_scalar(label<T>("dot", p), thefblas::dot(n, x.data(), incx, y.data(), incy),
                         ref::dot(n, x.data(), incx, y.data(), incy));
            if constexpr (std::is_same_v<T, float>) {
                check_scalar(label<T>("dsdot", p),
                             thefblas::dsdot(n, x.data(), incx, y.data(), incy),
                             ref::dsdot(n, x.data(), incx, y.data(), incy));
                check_scalar(label<T>("sdsdot", p + params(" sb=", alpha)),
                             thefblas::sdsdot(n, alpha, x.data(), incx, y.data(), incy),
                             ref::sdsdot(n, alpha, x.data(), incx, y.data(), incy));
            }
        }

        check_scalar(label<T>("nrm2", p), thefblas::nrm2(n, x.data(), incx),
                     ref::nrm2(n, x.data(), incx));
        check_scalar(label<T>("asum", p), thefblas::asum(n, x.data(), incx),
                     ref::asum(n, x.data(), incx));
        // thefblas returns a 1-based index and 0 for "no element"; CBLAS returns
        // a 0-based index and 0 in both of those cases.
        const int ref_index = (n <= 0 || incx <= 0) ? 0 : ref::iamax(n, x.data(), incx) + 1;
        check_index(label<T>("iamax", p), thefblas::iamax(n, x.data(), incx), ref_index);

        {
            const R angle = random_value<R>() * R(3.14159265358979);
            const R c = std::cos(angle);
            const R s = std::sin(angle);
            std::vector<T> x1 = x;
            std::vector<T> y1 = y;
            std::vector<T> x2 = x;
            std::vector<T> y2 = y;
            thefblas::rot(n, x1.data(), incx, y1.data(), incy, c, s);
            ref::rot(n, x2.data(), incx, y2.data(), incy, c, s);
            check_buffer(label<T>("rot x", p), x1, x2);
            check_buffer(label<T>("rot y", p), y1, y2);
        }
    }
}

template <typename T> void test_rotg() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        // Include zeros (they select the special branches of rotg) and, half of
        // the time, magnitudes far from 1 that would underflow or overflow if
        // squared naively.
        using R = real_t<T>;
        // 30 for float, 300 for double: well inside the finite range.
        const auto max_exponent = static_cast<R>(std::numeric_limits<R>::max_exponent10 - 8);
        const auto magnitude = [&]() {
            if (uniform_int(0, 1) == 0) {
                return R(1);
            }
            return std::pow(R(10),
                            std::uniform_real_distribution<R>(-max_exponent, max_exponent)(rng));
        };
        const T a0 = uniform_int(0, 5) == 0 ? T(0) : random_value<T>() * magnitude();
        const T b0 = uniform_int(0, 5) == 0 ? T(0) : random_value<T>() * magnitude();
        const std::string p = params("a=", a0, " b=", b0);
        real_t<T> c1{};
        real_t<T> c2{};
        T a1 = a0;
        T a2 = a0;
        T s1{};
        T s2{};
        if constexpr (is_complex_v<T>) {
            thefblas::rotg(&a1, b0, &c1, &s1);
            ref::rotg(&a2, b0, &c2, &s2);
        } else {
            T b1 = b0;
            T b2 = b0;
            thefblas::rotg(&a1, &b1, &c1, &s1);
            ref::rotg(&a2, &b2, &c2, &s2);
            check_scalar(label<T>("rotg z", p), b1, b2);
        }
        check_scalar(label<T>("rotg r", p), a1, a2);
        check_scalar(label<T>("rotg c", p), c1, c2);
        check_scalar(label<T>("rotg s", p), s1, s2);
    }
}

template <typename R> void test_rotm() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const int n = uniform_int(0, max_size);
        const int incx = random_inc();
        const int incy = random_inc();
        const std::vector<R> x = random_vector<R>(n, incx);
        const std::vector<R> y = random_vector<R>(n, incy);
        R param[5] = {R(uniform_int(-2, 1)), random_value<R>(), random_value<R>(),
                      random_value<R>(), random_value<R>()};
        const std::string p = params("n=", n, " incx=", incx, " incy=", incy, " flag=", param[0]);
        std::vector<R> x1 = x;
        std::vector<R> y1 = y;
        std::vector<R> x2 = x;
        std::vector<R> y2 = y;
        thefblas::rotm(n, x1.data(), incx, y1.data(), incy, param);
        ref::rotm(n, x2.data(), incx, y2.data(), incy, param);
        check_buffer(label<R>("rotm x", p), x1, x2);
        check_buffer(label<R>("rotm y", p), y1, y2);
    }
}

/// rotmg's contract: H * (b1, b2)^T = (b1', 0)^T for the H encoded in param.
template <typename R>
void check_rotmg_property(const std::string &what, const R *param, R b1, R b2, R b1_out) {
    ++checks;
    R h11 = param[1];
    R h21 = param[2];
    R h12 = param[3];
    R h22 = param[4];
    if (param[0] == R(0)) {
        h11 = R(1);
        h22 = R(1);
    } else if (param[0] == R(1)) {
        h21 = R(-1);
        h12 = R(1);
    }
    const double tol = 256.0 * static_cast<double>(std::numeric_limits<R>::epsilon());
    const auto x = static_cast<double>(h11 * b1 + h12 * b2);
    const auto y = static_cast<double>(h21 * b1 + h22 * b2);
    const double x_scale =
        std::abs(static_cast<double>(h11 * b1)) + std::abs(static_cast<double>(h12 * b2));
    const double y_scale =
        std::abs(static_cast<double>(h21 * b1)) + std::abs(static_cast<double>(h22 * b2));
    if (std::abs(x - static_cast<double>(b1_out)) > tol * x_scale || std::abs(y) > tol * y_scale) {
        report(what, params("H*(b1,b2) = (", x, ", ", y, "), expected (", b1_out, ", 0)"));
    }
}

template <typename R> void test_rotmg() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        // Magnitudes over many orders cover the 2^-24 / 2^24 rescaling loops;
        // negative d1 and d2 cover the rejected inputs, zeros the quick paths.
        const auto wide = [](R scale) {
            const R exponent = std::uniform_real_distribution<R>(R(-10), R(10))(rng);
            return random_value<R>() * scale * std::pow(R(10), exponent);
        };
        const int kind = uniform_int(0, 9);
        R d10 = kind == 0 ? R(0) : std::abs(wide(R(1)));
        if (kind == 1) {
            d10 = -d10;
        }
        const R d20 = kind == 2 ? R(0) : wide(R(1));
        const R b10 = kind == 3 ? R(0) : wide(R(1));
        const R b20 = kind == 4 ? R(0) : wide(R(1));
        const std::string p = params("d1=", d10, " d2=", d20, " b1=", b10, " b2=", b20);
        R d11 = d10;
        R d21 = d20;
        R b11 = b10;
        R d12 = d10;
        R d22 = d20;
        R b12 = b10;
        R param1[5] = {};
        R param2[5] = {};
        thefblas::rotmg(&d11, &d21, &b11, b20, param1);
        ref::rotmg(&d12, &d22, &b12, b20, param2);
        check_scalar(label<R>("rotmg d1", p), d11, d12);
        check_scalar(label<R>("rotmg d2", p), d21, d22);
        check_scalar(label<R>("rotmg b1", p), b11, b12);
        check_scalar(label<R>("rotmg flag", p), param1[0], param2[0]);
        // Only the entries the flag says are meaningful are compared. Flag -1
        // with d1 != 0 means the 2^-24 / 2^24 rescaling ran; there Netlib 3.12
        // resets h21/h12 on every pass of its loop and returns an H that no
        // longer satisfies H * (b1, b2) = (b1', 0), so thefblas is checked
        // against that defining property instead.
        const int flag = static_cast<int>(param2[0]);
        if (flag == -1 && d12 != R(0)) {
            check_rotmg_property(label<R>("rotmg H*(b1,b2)", p), param1, b10, b20, b11);
        } else if (flag == -1) {
            for (int i = 1; i < 5; ++i) {
                check_scalar(label<R>("rotmg h", p), param1[i], param2[i]);
            }
        } else if (flag == 0) {
            check_scalar(label<R>("rotmg h21", p), param1[2], param2[2]);
            check_scalar(label<R>("rotmg h12", p), param1[3], param2[3]);
        } else if (flag == 1) {
            check_scalar(label<R>("rotmg h11", p), param1[1], param2[1]);
            check_scalar(label<R>("rotmg h22", p), param1[4], param2[4]);
        }
    }
}

// ---------------------------------------------------------------------------
// Level 2
// ---------------------------------------------------------------------------

int random_lda(int rows) {
    return std::max(1, rows) + uniform_int(0, 3);
}

template <typename T> std::vector<T> random_matrix(int lda, int cols) {
    return random_buffer<T>(static_cast<std::size_t>(lda) * static_cast<std::size_t>(cols));
}

/// Shrink every element and put a magnitude in [1.5, 2.5] on the diagonal so
/// triangular solves stay well conditioned. `diag_index(j)` locates a(j, j).
template <typename T, typename DiagIndex>
void make_well_conditioned(std::vector<T> &a, int n, DiagIndex diag_index) {
    const auto shrink = static_cast<real_t<T>>(n + 1);
    std::transform(a.begin(), a.end(), a.begin(), [shrink](T value) { return value / shrink; });
    for (int j = 0; j < n; ++j) {
        const real_t<T> sign = uniform_int(0, 1) == 0 ? real_t<T>(1) : real_t<T>(-1);
        a[static_cast<std::size_t>(diag_index(j))] =
            T(sign * (real_t<T>(2) + real_t<T>(0.5) * std::real(random_value<T>())));
    }
}

int packed_size(int n) {
    return n * (n + 1) / 2;
}

template <typename T> void test_gemv() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char trans = random_of("NTC");
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int lda = random_lda(m);
        const int incx = random_inc();
        const int incy = random_inc();
        const int lenx = trans == 'N' ? n : m;
        const int leny = trans == 'N' ? m : n;
        const std::vector<T> a = random_matrix<T>(lda, n);
        const std::vector<T> x = random_vector<T>(lenx, incx);
        const std::vector<T> y = random_vector<T>(leny, incy);
        const T alpha = random_scalar<T>();
        const T beta = random_scalar<T>();
        const std::string p =
            params("trans=", trans, " m=", m, " n=", n, " lda=", lda, " incx=", incx,
                   " incy=", incy, " alpha=", alpha, " beta=", beta);
        std::vector<T> y1 = y;
        std::vector<T> y2 = y;
        thefblas::gemv(trans, m, n, alpha, a.data(), lda, x.data(), incx, beta, y1.data(), incy);
        ref::gemv(trans, m, n, alpha, a.data(), lda, x.data(), incx, beta, y2.data(), incy);
        check_buffer(label<T>("gemv", p), y1, y2);
    }
}

template <typename T> void test_gbmv() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char trans = random_of("NTC");
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int kl = uniform_int(0, 3);
        const int ku = uniform_int(0, 3);
        const int lda = kl + ku + 1 + uniform_int(0, 2);
        const int incx = random_inc();
        const int incy = random_inc();
        const int lenx = trans == 'N' ? n : m;
        const int leny = trans == 'N' ? m : n;
        const std::vector<T> a = random_matrix<T>(lda, n);
        const std::vector<T> x = random_vector<T>(lenx, incx);
        const std::vector<T> y = random_vector<T>(leny, incy);
        const T alpha = random_scalar<T>();
        const T beta = random_scalar<T>();
        const std::string p =
            params("trans=", trans, " m=", m, " n=", n, " kl=", kl, " ku=", ku, " lda=", lda,
                   " incx=", incx, " incy=", incy, " alpha=", alpha, " beta=", beta);
        std::vector<T> y1 = y;
        std::vector<T> y2 = y;
        thefblas::gbmv(trans, m, n, kl, ku, alpha, a.data(), lda, x.data(), incx, beta, y1.data(),
                       incy);
        ref::gbmv(trans, m, n, kl, ku, alpha, a.data(), lda, x.data(), incx, beta, y2.data(), incy);
        check_buffer(label<T>("gbmv", p), y1, y2);
    }
}

/// symv/sbmv/spmv for real types, hemv/hbmv/hpmv for complex types.
template <typename T> void test_symmetric_mv() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char uplo = random_of("UL");
        const int n = uniform_int(0, max_size);
        const int k = uniform_int(0, 3);
        const int lda = random_lda(n);
        const int ldab = k + 1 + uniform_int(0, 2);
        const int incx = random_inc();
        const int incy = random_inc();
        // The diagonal of a Hermitian matrix keeps a random imaginary part: the
        // routines must ignore it, as Netlib does.
        const std::vector<T> a = random_matrix<T>(lda, n);
        const std::vector<T> ab = random_matrix<T>(ldab, n);
        const std::vector<T> ap = random_buffer<T>(static_cast<std::size_t>(packed_size(n)));
        const std::vector<T> x = random_vector<T>(n, incx);
        const std::vector<T> y = random_vector<T>(n, incy);
        const T alpha = random_scalar<T>();
        const T beta = random_scalar<T>();
        const std::string p = params("uplo=", uplo, " n=", n, " incx=", incx, " incy=", incy,
                                     " alpha=", alpha, " beta=", beta);
        const bool herm = is_complex_v<T>;
        {
            std::vector<T> y1 = y;
            std::vector<T> y2 = y;
            if constexpr (is_complex_v<T>) {
                thefblas::hemv(uplo, n, alpha, a.data(), lda, x.data(), incx, beta, y1.data(),
                               incy);
            } else {
                thefblas::symv(uplo, n, alpha, a.data(), lda, x.data(), incx, beta, y1.data(),
                               incy);
            }
            ref::symv(uplo, n, alpha, a.data(), lda, x.data(), incx, beta, y2.data(), incy);
            check_buffer(label<T>(herm ? "hemv" : "symv", p + params(" lda=", lda)), y1, y2);
        }
        {
            std::vector<T> y1 = y;
            std::vector<T> y2 = y;
            if constexpr (is_complex_v<T>) {
                thefblas::hbmv(uplo, n, k, alpha, ab.data(), ldab, x.data(), incx, beta, y1.data(),
                               incy);
            } else {
                thefblas::sbmv(uplo, n, k, alpha, ab.data(), ldab, x.data(), incx, beta, y1.data(),
                               incy);
            }
            ref::sbmv(uplo, n, k, alpha, ab.data(), ldab, x.data(), incx, beta, y2.data(), incy);
            check_buffer(label<T>(herm ? "hbmv" : "sbmv", p + params(" k=", k, " lda=", ldab)), y1,
                         y2);
        }
        {
            std::vector<T> y1 = y;
            std::vector<T> y2 = y;
            if constexpr (is_complex_v<T>) {
                thefblas::hpmv(uplo, n, alpha, ap.data(), x.data(), incx, beta, y1.data(), incy);
            } else {
                thefblas::spmv(uplo, n, alpha, ap.data(), x.data(), incx, beta, y1.data(), incy);
            }
            ref::spmv(uplo, n, alpha, ap.data(), x.data(), incx, beta, y2.data(), incy);
            check_buffer(label<T>(herm ? "hpmv" : "spmv", p), y1, y2);
        }
    }
}

/// trmv/trsv, tbmv/tbsv and tpmv/tpsv.
template <typename T> void test_triangular() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char uplo = random_of("UL");
        const char trans = random_of("NTC");
        const char diag = random_of("NU");
        const int n = uniform_int(0, max_size);
        const int k = uniform_int(0, 3);
        const int lda = random_lda(n);
        const int ldab = k + 1 + uniform_int(0, 2);
        const int incx = random_inc();
        const bool upper = uplo == 'U';

        std::vector<T> a = random_matrix<T>(lda, n);
        make_well_conditioned(a, n, [lda](int j) { return j + j * lda; });
        std::vector<T> ab = random_matrix<T>(ldab, n);
        make_well_conditioned(ab, n, [=](int j) { return (upper ? k : 0) + j * ldab; });
        std::vector<T> ap = random_buffer<T>(static_cast<std::size_t>(packed_size(n)));
        make_well_conditioned(
            ap, n, [=](int j) { return upper ? j + j * (j + 1) / 2 : j * (2 * n - j + 1) / 2; });
        const std::vector<T> x = random_vector<T>(n, incx);
        const std::string p =
            params("uplo=", uplo, " trans=", trans, " diag=", diag, " n=", n, " incx=", incx);
        const std::string pa = p + params(" lda=", lda);
        const std::string pb = p + params(" k=", k, " lda=", ldab);

        std::vector<T> x1 = x;
        std::vector<T> x2 = x;
        thefblas::trmv(uplo, trans, diag, n, a.data(), lda, x1.data(), incx);
        ref::trmv(uplo, trans, diag, n, a.data(), lda, x2.data(), incx);
        check_buffer(label<T>("trmv", pa), x1, x2);

        x1 = x;
        x2 = x;
        thefblas::trsv(uplo, trans, diag, n, a.data(), lda, x1.data(), incx);
        ref::trsv(uplo, trans, diag, n, a.data(), lda, x2.data(), incx);
        check_buffer(label<T>("trsv", pa), x1, x2);

        x1 = x;
        x2 = x;
        thefblas::tbmv(uplo, trans, diag, n, k, ab.data(), ldab, x1.data(), incx);
        ref::tbmv(uplo, trans, diag, n, k, ab.data(), ldab, x2.data(), incx);
        check_buffer(label<T>("tbmv", pb), x1, x2);

        x1 = x;
        x2 = x;
        thefblas::tbsv(uplo, trans, diag, n, k, ab.data(), ldab, x1.data(), incx);
        ref::tbsv(uplo, trans, diag, n, k, ab.data(), ldab, x2.data(), incx);
        check_buffer(label<T>("tbsv", pb), x1, x2);

        x1 = x;
        x2 = x;
        thefblas::tpmv(uplo, trans, diag, n, ap.data(), x1.data(), incx);
        ref::tpmv(uplo, trans, diag, n, ap.data(), x2.data(), incx);
        check_buffer(label<T>("tpmv", p), x1, x2);

        x1 = x;
        x2 = x;
        thefblas::tpsv(uplo, trans, diag, n, ap.data(), x1.data(), incx);
        ref::tpsv(uplo, trans, diag, n, ap.data(), x2.data(), incx);
        check_buffer(label<T>("tpsv", p), x1, x2);
    }
}

/// ger for real types, geru and gerc for complex types.
template <typename T> void test_ger() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int lda = random_lda(m);
        const int incx = random_inc();
        const int incy = random_inc();
        const std::vector<T> a = random_matrix<T>(lda, n);
        const std::vector<T> x = random_vector<T>(m, incx);
        const std::vector<T> y = random_vector<T>(n, incy);
        const T alpha = random_scalar<T>();
        const std::string p = params("m=", m, " n=", n, " lda=", lda, " incx=", incx,
                                     " incy=", incy, " alpha=", alpha);
        std::vector<T> a1 = a;
        std::vector<T> a2 = a;
        if constexpr (is_complex_v<T>) {
            thefblas::geru(m, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
            ref::ger(false, m, n, alpha, x.data(), incx, y.data(), incy, a2.data(), lda);
            check_buffer(label<T>("geru", p), a1, a2);
            a1 = a;
            a2 = a;
            thefblas::gerc(m, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
            ref::ger(true, m, n, alpha, x.data(), incx, y.data(), incy, a2.data(), lda);
            check_buffer(label<T>("gerc", p), a1, a2);
        } else {
            thefblas::ger(m, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
            ref::ger(false, m, n, alpha, x.data(), incx, y.data(), incy, a2.data(), lda);
            check_buffer(label<T>("ger", p), a1, a2);
        }
    }
}

/// syr/syr2/spr/spr2 for real types, her/her2/hpr/hpr2 for complex types.
template <typename T> void test_symmetric_update() {
    using R = real_t<T>;
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char uplo = random_of("UL");
        const int n = uniform_int(0, max_size);
        const int lda = random_lda(n);
        const int incx = random_inc();
        const int incy = random_inc();
        const std::vector<T> a = random_matrix<T>(lda, n);
        const std::vector<T> ap = random_buffer<T>(static_cast<std::size_t>(packed_size(n)));
        const std::vector<T> x = random_vector<T>(n, incx);
        const std::vector<T> y = random_vector<T>(n, incy);
        const R real_alpha = random_scalar<R>();
        const T alpha = random_scalar<T>();
        const std::string p = params("uplo=", uplo, " n=", n, " incx=", incx);
        const std::string p2 = p + params(" incy=", incy, " alpha=", alpha);
        const bool herm = is_complex_v<T>;

        std::vector<T> a1 = a;
        std::vector<T> a2 = a;
        if constexpr (is_complex_v<T>) {
            thefblas::her(uplo, n, real_alpha, x.data(), incx, a1.data(), lda);
        } else {
            thefblas::syr(uplo, n, real_alpha, x.data(), incx, a1.data(), lda);
        }
        ref::syr(uplo, n, real_alpha, x.data(), incx, a2.data(), lda);
        check_buffer(
            label<T>(herm ? "her" : "syr", p + params(" lda=", lda, " alpha=", real_alpha)), a1,
            a2);

        a1 = a;
        a2 = a;
        if constexpr (is_complex_v<T>) {
            thefblas::her2(uplo, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
        } else {
            thefblas::syr2(uplo, n, alpha, x.data(), incx, y.data(), incy, a1.data(), lda);
        }
        ref::syr2(uplo, n, alpha, x.data(), incx, y.data(), incy, a2.data(), lda);
        check_buffer(label<T>(herm ? "her2" : "syr2", p2 + params(" lda=", lda)), a1, a2);

        std::vector<T> ap1 = ap;
        std::vector<T> ap2 = ap;
        if constexpr (is_complex_v<T>) {
            thefblas::hpr(uplo, n, real_alpha, x.data(), incx, ap1.data());
        } else {
            thefblas::spr(uplo, n, real_alpha, x.data(), incx, ap1.data());
        }
        ref::spr(uplo, n, real_alpha, x.data(), incx, ap2.data());
        check_buffer(label<T>(herm ? "hpr" : "spr", p + params(" alpha=", real_alpha)), ap1, ap2);

        ap1 = ap;
        ap2 = ap;
        if constexpr (is_complex_v<T>) {
            thefblas::hpr2(uplo, n, alpha, x.data(), incx, y.data(), incy, ap1.data());
        } else {
            thefblas::spr2(uplo, n, alpha, x.data(), incx, y.data(), incy, ap1.data());
        }
        ref::spr2(uplo, n, alpha, x.data(), incx, y.data(), incy, ap2.data());
        check_buffer(label<T>(herm ? "hpr2" : "spr2", p2), ap1, ap2);
    }
}

// ---------------------------------------------------------------------------
// Level 3
// ---------------------------------------------------------------------------

template <typename T> void test_gemm() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char transa = random_of("NTC");
        const char transb = random_of("NTC");
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int k = uniform_int(0, max_size);
        const int lda = random_lda(transa == 'N' ? m : k);
        const int ldb = random_lda(transb == 'N' ? k : n);
        const int ldc = random_lda(m);
        const std::vector<T> a = random_matrix<T>(lda, transa == 'N' ? k : m);
        const std::vector<T> b = random_matrix<T>(ldb, transb == 'N' ? n : k);
        const std::vector<T> c = random_matrix<T>(ldc, n);
        const T alpha = random_scalar<T>();
        const T beta = random_scalar<T>();
        const std::string p =
            params("transa=", transa, " transb=", transb, " m=", m, " n=", n, " k=", k,
                   " lda=", lda, " ldb=", ldb, " ldc=", ldc, " alpha=", alpha, " beta=", beta);
        std::vector<T> c1 = c;
        std::vector<T> c2 = c;
        thefblas::gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta,
                       c1.data(), ldc);
        ref::gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c2.data(),
                  ldc);
        check_buffer(label<T>("gemm", p), c1, c2);
    }
}

/// syrk for every type, and herk for complex types.
template <typename T> void test_rank_k() {
    using R = real_t<T>;
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char uplo = random_of("UL");
        const int n = uniform_int(0, max_size);
        const int k = uniform_int(0, max_size);
        const int ldc = random_lda(n);
        const std::vector<T> c = random_matrix<T>(ldc, n);
        {
            // Complex syrk has no 'C' form; real syrk treats 'C' as 'T'.
            const char trans = is_complex_v<T> ? random_of("NT") : random_of("NTC");
            const int lda = random_lda(trans == 'N' ? n : k);
            const std::vector<T> a = random_matrix<T>(lda, trans == 'N' ? k : n);
            const T alpha = random_scalar<T>();
            const T beta = random_scalar<T>();
            const std::string p =
                params("uplo=", uplo, " trans=", trans, " n=", n, " k=", k, " lda=", lda,
                       " ldc=", ldc, " alpha=", alpha, " beta=", beta);
            std::vector<T> c1 = c;
            std::vector<T> c2 = c;
            thefblas::syrk(uplo, trans, n, k, alpha, a.data(), lda, beta, c1.data(), ldc);
            ref::syrk(uplo, trans, n, k, alpha, a.data(), lda, beta, c2.data(), ldc);
            check_buffer(label<T>("syrk", p), c1, c2);
        }
        if constexpr (is_complex_v<T>) {
            const char trans = random_of("NC");
            const int lda = random_lda(trans == 'N' ? n : k);
            const std::vector<T> a = random_matrix<T>(lda, trans == 'N' ? k : n);
            const R alpha = random_scalar<R>();
            const R beta = random_scalar<R>();
            const std::string p =
                params("uplo=", uplo, " trans=", trans, " n=", n, " k=", k, " lda=", lda,
                       " ldc=", ldc, " alpha=", alpha, " beta=", beta);
            std::vector<T> c1 = c;
            std::vector<T> c2 = c;
            thefblas::herk(uplo, trans, n, k, alpha, a.data(), lda, beta, c1.data(), ldc);
            ref::herk(uplo, trans, n, k, alpha, a.data(), lda, beta, c2.data(), ldc);
            check_buffer(label<T>("herk", p), c1, c2);
        }
    }
}

template <typename T> void test_trsm() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char side = random_of("LR");
        const char uplo = random_of("UL");
        const char trans = random_of("NTC");
        const char diag = random_of("NU");
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int dim = side == 'L' ? m : n;
        const int lda = random_lda(dim);
        const int ldb = random_lda(m);
        std::vector<T> a = random_matrix<T>(lda, dim);
        make_well_conditioned(a, dim, [lda](int j) { return j + j * lda; });
        const std::vector<T> b = random_matrix<T>(ldb, n);
        const T alpha = random_scalar<T>();
        const std::string p =
            params("side=", side, " uplo=", uplo, " trans=", trans, " diag=", diag, " m=", m,
                   " n=", n, " lda=", lda, " ldb=", ldb, " alpha=", alpha);
        std::vector<T> b1 = b;
        std::vector<T> b2 = b;
        thefblas::trsm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b1.data(), ldb);
        ref::trsm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b2.data(), ldb);
        check_buffer(label<T>("trsm", p), b1, b2);
    }
}

/// symm for every type, and hemm for complex types.
template <typename T> void test_symm() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char side = random_of("LR");
        const char uplo = random_of("UL");
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int dim = side == 'L' ? m : n;
        const int lda = random_lda(dim);
        const int ldb = random_lda(m);
        const int ldc = random_lda(m);
        const std::vector<T> a = random_matrix<T>(lda, dim);
        const std::vector<T> b = random_matrix<T>(ldb, n);
        const std::vector<T> c = random_matrix<T>(ldc, n);
        const T alpha = random_scalar<T>();
        const T beta = random_scalar<T>();
        const std::string p =
            params("side=", side, " uplo=", uplo, " m=", m, " n=", n, " lda=", lda, " ldb=", ldb,
                   " ldc=", ldc, " alpha=", alpha, " beta=", beta);
        std::vector<T> c1 = c;
        std::vector<T> c2 = c;
        thefblas::symm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c1.data(), ldc);
        ref::symm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c2.data(), ldc);
        check_buffer(label<T>("symm", p), c1, c2);
        if constexpr (is_complex_v<T>) {
            c1 = c;
            c2 = c;
            thefblas::hemm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c1.data(),
                           ldc);
            ref::hemm(side, uplo, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c2.data(), ldc);
            check_buffer(label<T>("hemm", p), c1, c2);
        }
    }
}

/// syr2k for every type, and her2k for complex types.
template <typename T> void test_rank_2k() {
    using R = real_t<T>;
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char uplo = random_of("UL");
        const int n = uniform_int(0, max_size);
        const int k = uniform_int(0, max_size);
        const int ldc = random_lda(n);
        const std::vector<T> c = random_matrix<T>(ldc, n);
        const T alpha = random_scalar<T>();
        {
            // As for syrk: complex syr2k has no 'C' form.
            const char trans = is_complex_v<T> ? random_of("NT") : random_of("NTC");
            const int lda = random_lda(trans == 'N' ? n : k);
            const int ldb = random_lda(trans == 'N' ? n : k);
            const std::vector<T> a = random_matrix<T>(lda, trans == 'N' ? k : n);
            const std::vector<T> b = random_matrix<T>(ldb, trans == 'N' ? k : n);
            const T beta = random_scalar<T>();
            const std::string p =
                params("uplo=", uplo, " trans=", trans, " n=", n, " k=", k, " lda=", lda,
                       " ldb=", ldb, " ldc=", ldc, " alpha=", alpha, " beta=", beta);
            std::vector<T> c1 = c;
            std::vector<T> c2 = c;
            thefblas::syr2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c1.data(),
                            ldc);
            ref::syr2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c2.data(),
                       ldc);
            check_buffer(label<T>("syr2k", p), c1, c2);
        }
        if constexpr (is_complex_v<T>) {
            const char trans = random_of("NC");
            const int lda = random_lda(trans == 'N' ? n : k);
            const int ldb = random_lda(trans == 'N' ? n : k);
            const std::vector<T> a = random_matrix<T>(lda, trans == 'N' ? k : n);
            const std::vector<T> b = random_matrix<T>(ldb, trans == 'N' ? k : n);
            const R beta = random_scalar<R>();
            const std::string p =
                params("uplo=", uplo, " trans=", trans, " n=", n, " k=", k, " lda=", lda,
                       " ldb=", ldb, " ldc=", ldc, " alpha=", alpha, " beta=", beta);
            std::vector<T> c1 = c;
            std::vector<T> c2 = c;
            thefblas::her2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c1.data(),
                            ldc);
            ref::her2k(uplo, trans, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c2.data(),
                       ldc);
            check_buffer(label<T>("her2k", p), c1, c2);
        }
    }
}

template <typename T> void test_trmm() {
    for (int trial = 0; trial < trials_per_routine; ++trial) {
        const char side = random_of("LR");
        const char uplo = random_of("UL");
        const char trans = random_of("NTC");
        const char diag = random_of("NU");
        const int m = uniform_int(0, max_size);
        const int n = uniform_int(0, max_size);
        const int dim = side == 'L' ? m : n;
        const int lda = random_lda(dim);
        const int ldb = random_lda(m);
        const std::vector<T> a = random_matrix<T>(lda, dim);
        const std::vector<T> b = random_matrix<T>(ldb, n);
        const T alpha = random_scalar<T>();
        const std::string p =
            params("side=", side, " uplo=", uplo, " trans=", trans, " diag=", diag, " m=", m,
                   " n=", n, " lda=", lda, " ldb=", ldb, " alpha=", alpha);
        std::vector<T> b1 = b;
        std::vector<T> b2 = b;
        thefblas::trmm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b1.data(), ldb);
        ref::trmm(side, uplo, trans, diag, m, n, alpha, a.data(), lda, b2.data(), ldb);
        check_buffer(label<T>("trmm", p), b1, b2);
    }
}

template <typename T> void test_type() {
    test_level1<T>();
    test_rotg<T>();
    if constexpr (!is_complex_v<T>) {
        test_rotm<T>();
        test_rotmg<T>();
    }
    test_gemv<T>();
    test_gbmv<T>();
    test_symmetric_mv<T>();
    test_triangular<T>();
    test_ger<T>();
    test_symmetric_update<T>();
    test_gemm<T>();
    test_symm<T>();
    test_rank_k<T>();
    test_rank_2k<T>();
    test_trmm<T>();
    test_trsm<T>();
}

} // namespace

int main() {
    std::uint32_t seed = 20261011U;
    if (const char *env = std::getenv("THEFBLAS_TEST_SEED")) {
        seed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
    }
    rng.seed(seed);

    test_type<float>();
    test_type<double>();
    test_type<cfloat>();
    test_type<cdouble>();

    std::printf("reference BLAS comparison: seed %u, %ld checks, %ld mismatches\n",
                static_cast<unsigned>(seed), checks, failures);
    if (failures > max_reports) {
        std::fprintf(stderr, "(only the first %d mismatches were printed)\n", max_reports);
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
