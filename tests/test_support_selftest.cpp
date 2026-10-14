// Checks the test helper itself: value formatting, failure counting, the
// report limit, context nesting and the exit code from finish(). The failures
// below are deliberate; they print reports to stderr and are then discounted,
// so the test passes when the helper behaves as documented.
#include "test_support.hpp"

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

namespace td = thefblas_test::detail;

struct opaque {};

template <typename T> bool formats_as(const T &value, const char *expected) {
    char buf[td::line_size] = {};
    td::append(buf, sizeof(buf), value);
    const bool ok = std::strcmp(buf, expected) == 0;
    if (!ok) {
        std::fprintf(stderr, "  formatted \"%s\", expected \"%s\"\n", buf, expected);
    }
    return ok;
}

void test_formatting() {
    CHECK(formats_as(true, "true"));
    CHECK(formats_as('x', "x"));
    CHECK(formats_as(std::int8_t{-128}, "-128"));
    CHECK(formats_as(std::numeric_limits<std::int64_t>::min(), "-9223372036854775808"));
    CHECK(formats_as(std::numeric_limits<std::uint64_t>::max(), "18446744073709551615"));
    CHECK(formats_as(0, "0"));
    CHECK(formats_as(0.5, "0.5"));
    CHECK(formats_as(thefblas::q16_16(-1.5), "-1.5 (raw -98304)"));
    CHECK(formats_as(std::complex<double>(1, -2), "(1, -2)"));
    CHECK(formats_as("text", "text"));
    CHECK(formats_as(opaque{}, "<value>"));

    // 128-bit integers are printed in full, also without GNU extensions
    // (the strict build of this test), where std::is_integral rejects them.
#if defined(__SIZEOF_INT128__)
    {
        using wide = thefblas::detail::widen_t<std::int64_t>;
        using uwide = thefblas::detail::make_unsigned_integer_t<wide>;
        const wide big = static_cast<wide>(std::numeric_limits<std::int64_t>::min()) * 4;
        CHECK(formats_as(big, "-36893488147419103232"));
        CHECK(formats_as(static_cast<uwide>(~uwide{0}), "340282366920938463463374607431768211455"));
    }
#endif

    // Output that does not fit is cut off, never overrun.
    char small[6] = {};
    td::append_all(small, sizeof(small), "abc", 12345);
    CHECK(std::strcmp(small, "abc12") == 0);
    td::append(small, sizeof(small), "more");
    CHECK(std::strcmp(small, "abc12") == 0);
}

void test_failures() {
    td::state &s = td::global();
    const td::state saved = s;
    s.failures = 0;
    s.checks = 0;

    std::fprintf(stderr, "--- deliberate failures follow ---\n");
    {
        TEST_CONTEXT("outer=", 1);
        {
            TEST_CONTEXT("inner=", 2);
            CHECK(1 + 1 == 3);
            CHECK_EQ(2, 3);
        }
        CHECK_MSG(false, "note ", 42);
    }
    CHECK(true);
    const long checks = s.checks;
    const long failures = s.failures;
    const int depth = s.depth;
    const int failed_exit = thefblas_test::finish();

    // Past the report limit, failures are still counted but not printed.
    s.failures = td::max_reports;
    const bool printed = td::record(false, "beyond the limit", __FILE__, __LINE__);
    const long counted = s.failures;
    std::fprintf(stderr, "--- end of deliberate failures ---\n");

    s = saved;
    CHECK_EQ(checks, 4L);
    CHECK_EQ(failures, 3L);
    CHECK_EQ(depth, 0);
    CHECK_EQ(failed_exit, 1);
    CHECK(!printed);
    CHECK_EQ(counted, static_cast<long>(td::max_reports) + 1);
}

} // namespace

int main() {
    test_formatting();
    test_failures();
    return thefblas_test::finish();
}
