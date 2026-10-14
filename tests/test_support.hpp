// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

// Minimal test helpers. Unlike assert(), a failed check records the failure,
// prints its location, the expression and the active context, and lets the
// test continue; finish() prints a summary and returns the exit code. Only
// <cstdio> is used for output, so the tests still build for bare-metal targets.
//
//   CHECK(cond)                        the condition must hold (prints no values)
//   CHECK_EQ(a, b)                     a == b, printing both values
//   CHECK_VALUES(cond, got, expected)  cond must hold; prints got and expected
//   CHECK_PRED(pred, got, expected...) pred(got, expected...) must hold;
//                                      prints got and expected
//   CHECK_MSG(cond, parts...)          cond must hold; prints the parts
//   TEST_CONTEXT(parts...)             adds the parts to every failure in scope
//
// Parts are printed one after another: strings as they are, numbers, fixed
// point values (with their raw representation) and complex values formatted.
// Every failure is counted, but only the first 50 are printed.

#include "thefblas/fixed.hpp"

#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <type_traits>

namespace thefblas_test {

namespace detail {

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

// Unsigned integers, including unsigned __int128, which the standard traits
// only recognise with GNU extensions enabled (see thefblas::detail::
// is_signed_integer for the signed side).
template <typename T>
struct is_unsigned_integer
    : std::integral_constant<bool, std::is_integral<T>::value && std::is_unsigned<T>::value> {};
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
template <> struct is_unsigned_integer<unsigned __int128> : std::true_type {};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

template <typename T, bool = is_unsigned_integer<T>::value> struct unsigned_of {
    using type = T;
};
template <typename T> struct unsigned_of<T, false> {
    using type = thefblas::detail::make_unsigned_integer_t<T>;
};

/// Writes the decimal digits of an integer of any width (including __int128,
/// which printf cannot format) to `out` (of `room` bytes), keeping it terminated.
template <typename T> void append_integer(char *out, std::size_t room, T value) {
    using U = typename unsigned_of<T>::type;
    bool negative = false;
    U magnitude = static_cast<U>(value);
    if constexpr (!is_unsigned_integer<T>::value) {
        negative = value < 0;
        if (negative) {
            magnitude = static_cast<U>(U{0} - magnitude);
        }
    }
    char digits[48] = {};
    std::size_t n = 0;
    do {
        const U digit = static_cast<U>(magnitude % U{10});
        digits[n++] = static_cast<char>('0' + static_cast<int>(digit));
        magnitude = static_cast<U>(magnitude / U{10});
    } while (magnitude != 0);
    if (negative) {
        digits[n++] = '-';
    }
    std::size_t i = 0;
    for (; i < n && i + 1 < room; ++i) {
        out[i] = digits[n - 1 - i];
    }
    out[i] = '\0';
}

/// Appends a formatted value to `buf` (of `size` bytes), keeping it terminated.
template <typename T> void append(char *buf, std::size_t size, const T &value) {
    const std::size_t used = std::strlen(buf);
    if (used + 1 >= size) {
        return;
    }
    char *out = buf + used;
    const std::size_t room = size - used;
    if constexpr (std::is_same_v<T, bool>) {
        std::snprintf(out, room, "%s", value ? "true" : "false");
    } else if constexpr (std::is_same_v<T, char>) {
        std::snprintf(out, room, "%c", value);
    } else if constexpr (thefblas::detail::is_signed_integer<T>::value ||
                         is_unsigned_integer<T>::value) {
        append_integer(out, room, value);
    } else if constexpr (std::is_floating_point_v<T>) {
        std::snprintf(out, room, "%.17g", static_cast<double>(value));
    } else if constexpr (thefblas::detail::is_fixed_v<T>) {
        std::snprintf(out, room, "%.17g (raw ", value.template to_float<double>());
        append(buf, size, value.raw());
        append(buf, size, ")");
    } else if constexpr (is_complex<T>::value) {
        append(buf, size, "(");
        append(buf, size, value.real());
        append(buf, size, ", ");
        append(buf, size, value.imag());
        append(buf, size, ")");
    } else if constexpr (std::is_convertible_v<const T &, const char *>) {
        std::snprintf(out, room, "%s", static_cast<const char *>(value));
    } else {
        std::snprintf(out, room, "%s", "<value>");
    }
}

template <typename... Parts> void append_all(char *buf, std::size_t size, const Parts &...parts) {
    (append(buf, size, parts), ...);
}

constexpr int max_contexts = 8;
constexpr std::size_t line_size = 256;
constexpr int max_reports = 50;

struct state {
    long checks = 0;
    long failures = 0;
    int depth = 0;
    char contexts[max_contexts][line_size] = {};
};

inline state &global() {
    static state s;
    return s;
}

/// Records one check; on failure prints the location, the expression and the
/// active contexts, and returns true so the caller can print details.
inline bool record(bool ok, const char *expr, const char *file, int line) {
    state &s = global();
    ++s.checks;
    if (ok) {
        return false;
    }
    ++s.failures;
    if (s.failures > max_reports) {
        return false;
    }
    std::fprintf(stderr, "FAILED %s:%d: %s\n", file, line, expr);
    const int shown = s.depth < max_contexts ? s.depth : max_contexts;
    for (int i = 0; i < shown; ++i) {
        std::fprintf(stderr, "  context:  %s\n", s.contexts[i]);
    }
    return true;
}

template <typename... Parts> void report_line(const char *label, const Parts &...parts) {
    char buf[line_size] = {};
    append_all(buf, sizeof(buf), parts...);
    std::fprintf(stderr, "  %-9s %s\n", label, buf);
}

} // namespace detail

/// Adds a description of the current parameters to every failure reported
/// while it is in scope.
class scoped_context {
  public:
    template <typename... Parts> explicit scoped_context(const Parts &...parts) {
        detail::state &s = detail::global();
        if (s.depth < detail::max_contexts) {
            s.contexts[s.depth][0] = '\0';
            detail::append_all(s.contexts[s.depth], detail::line_size, parts...);
        }
        ++s.depth;
    }
    ~scoped_context() { --detail::global().depth; }
    scoped_context(const scoped_context &) = delete;
    scoped_context &operator=(const scoped_context &) = delete;
    scoped_context(scoped_context &&) = delete;
    scoped_context &operator=(scoped_context &&) = delete;
};

inline void check(bool ok, const char *expr, const char *file, int line) {
    detail::record(ok, expr, file, line);
}

template <typename A, typename B>
void check_eq(const A &a, const B &b, const char *expr, const char *file, int line) {
    if (detail::record(a == b, expr, file, line)) {
        detail::report_line("left:", a);
        detail::report_line("right:", b);
    }
}

template <typename G, typename E>
void check_values(bool ok, const G &got, const E &expected, const char *expr, const char *file,
                  int line) {
    if (detail::record(ok, expr, file, line)) {
        detail::report_line("got:", got);
        detail::report_line("expected:", expected);
    }
}

template <typename Pred, typename G, typename E, typename... Rest>
void check_pred(Pred pred, const char *expr, const char *file, int line, const G &got,
                const E &expected, const Rest &...rest) {
    if (detail::record(pred(got, expected, rest...), expr, file, line)) {
        detail::report_line("got:", got);
        detail::report_line("expected:", expected);
    }
}

template <typename... Parts>
void check_msg(bool ok, const char *expr, const char *file, int line, const Parts &...parts) {
    if (detail::record(ok, expr, file, line)) {
        detail::report_line("note:", parts...);
    }
}

/// Prints a one-line summary and returns the process exit code.
inline int finish() {
    const detail::state &s = detail::global();
    if (s.failures > detail::max_reports) {
        std::fprintf(stderr, "(only the first %d failures were printed)\n", detail::max_reports);
    }
    std::printf("%ld checks, %ld failures\n", s.checks, s.failures);
    return s.failures == 0 ? 0 : 1;
}

} // namespace thefblas_test

#define THEFBLAS_TEST_CAT2(a, b) a##b
#define THEFBLAS_TEST_CAT(a, b) THEFBLAS_TEST_CAT2(a, b)

#define CHECK(...)                                                                                 \
    ::thefblas_test::check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)
#define CHECK_EQ(a, b) ::thefblas_test::check_eq((a), (b), #a " == " #b, __FILE__, __LINE__)
#define CHECK_VALUES(cond, got, expected)                                                          \
    ::thefblas_test::check_values(static_cast<bool>(cond), (got), (expected), #cond, __FILE__,     \
                                  __LINE__)
// pred(got, expected, extra...) must hold; each argument is evaluated once.
#define CHECK_PRED(pred, ...)                                                                      \
    ::thefblas_test::check_pred(                                                                   \
        [&](const auto &...thefblas_test_args) {                                                   \
            return static_cast<bool>(pred(thefblas_test_args...));                                 \
        },                                                                                         \
        #pred "(" #__VA_ARGS__ ")", __FILE__, __LINE__, __VA_ARGS__)
#define CHECK_MSG(cond, ...)                                                                       \
    ::thefblas_test::check_msg(static_cast<bool>(cond), #cond, __FILE__, __LINE__, __VA_ARGS__)
#define TEST_CONTEXT(...)                                                                          \
    const ::thefblas_test::scoped_context THEFBLAS_TEST_CAT(thefblas_test_context_,                \
                                                            __LINE__)(__VA_ARGS__)
