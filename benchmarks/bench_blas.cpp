// Micro-benchmarks: time per call of dot, axpy, gemv and gemm for float, double
// and fixed point, with plain std::chrono and no benchmark framework.
//
// Build with -DTHEFBLAS_BUILD_BENCHMARKS=ON in a Release configuration and run
// thefblas_bench. Each measurement repeats the call until it has run for at
// least 50 ms. `--quick` runs every call once instead, as a smoke test.

#include "thefblas/thefblas.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

bool quick = false;

// Keeps results observable so that the timed calls are not optimised away.
volatile double sink = 0.0;

template <typename T> double to_double(const T &value) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(value);
    } else {
        return value.to_float();
    }
}

// Deterministic inputs in [-0.375, 0.375], so that no format overflows.
template <typename T> std::vector<T> make_values(int count) {
    std::vector<T> values;
    values.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        values.push_back(T(static_cast<double>((i * 7) % 13 - 6) / 16.0));
    }
    return values;
}

// Average time of one call of `f`, in nanoseconds.
template <typename F> double time_per_call_ns(F &&f) {
    using clock = std::chrono::steady_clock;
    const double min_ns = quick ? 0.0 : 5e7;
    long reps = 1;
    for (;;) {
        const auto start = clock::now();
        for (long r = 0; r < reps; ++r) {
            f();
        }
        const double elapsed = std::chrono::duration<double, std::nano>(clock::now() - start).count();
        if (elapsed >= min_ns || reps >= (1L << 24)) {
            return elapsed / static_cast<double>(reps);
        }
        reps *= 2;
    }
}

void report(const char *op, const char *type, const char *size, double flops, double ns) {
    std::printf("%-5s %-10s %-12s %12.1f ns %10.1f MFLOP/s\n", op, type, size, ns,
                flops / ns * 1e3);
}

template <typename T> void bench_type(const char *type) {
    // alpha is below 1 so that it is representable in Q1.15 too.
    const T alpha(0.5);
    const T beta(0.25);

    {
        const int n = 1024;
        const auto x = make_values<T>(n);
        auto y = make_values<T>(n);
        const double ns = time_per_call_ns([&] {
            sink = to_double(thefblas::dot(n, x.data(), 1, y.data(), 1));
        });
        report("dot", type, "n=1024", 2.0 * n, ns);
        const double ns_axpy = time_per_call_ns([&] {
            thefblas::axpy(n, alpha, x.data(), 1, y.data(), 1);
            sink = to_double(y[0]);
        });
        report("axpy", type, "n=1024", 2.0 * n, ns_axpy);
    }
    {
        const int n = 128;
        const auto a = make_values<T>(n * n);
        const auto x = make_values<T>(n);
        auto y = make_values<T>(n);
        const double ns = time_per_call_ns([&] {
            thefblas::gemv('N', n, n, alpha, a.data(), n, x.data(), 1, beta, y.data(), 1);
            sink = to_double(y[0]);
        });
        report("gemv", type, "128x128", 2.0 * n * n, ns);
    }
    {
        const int n = 64;
        const auto a = make_values<T>(n * n);
        const auto b = make_values<T>(n * n);
        auto c = make_values<T>(n * n);
        const double ns = time_per_call_ns([&] {
            thefblas::gemm('N', 'N', n, n, n, alpha, a.data(), n, b.data(), n, beta, c.data(), n);
            sink = to_double(c[0]);
        });
        report("gemm", type, "64x64x64", 2.0 * n * n * n, ns);
    }
}

} // namespace

int main(int argc, char **argv) {
    quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    std::printf("%-5s %-10s %-12s %15s %18s\n", "op", "type", "size", "time/call", "throughput");
    bench_type<float>("float");
    bench_type<double>("double");
    bench_type<thefblas::q15_sat>("q15_sat");
    bench_type<thefblas::q16_16_sat>("q16_16_sat");
    return 0;
}
