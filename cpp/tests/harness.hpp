// A ~100-line test harness. The project has one external C++ dependency
// (pybind11, and only for the bindings target); adding Catch2 to get ASSERT
// macros is not worth the build-system weight.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace vsetest {

struct Registry {
    struct Case { std::string suite, name; std::function<void()> fn; };
    std::vector<Case> cases;
    static Registry& get() { static Registry r; return r; }
};

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        Registry::get().cases.push_back({suite, name, std::move(fn)});
    }
};

struct Failure : std::exception {
    std::string msg;
    explicit Failure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

inline int& check_count() { static int n = 0; return n; }

inline void fail(const char* file, int line, const std::string& detail) {
    char buf[2048];
    std::snprintf(buf, sizeof buf, "%s:%d\n      %s", file, line, detail.c_str());
    throw Failure(buf);
}

inline void check_true(bool ok, const char* expr, const char* file, int line) {
    ++check_count();
    if (!ok) fail(file, line, std::string("expected true: ") + expr);
}

inline void check_close(double got, double want, double tol, const char* expr,
                        const char* file, int line) {
    ++check_count();
    const double err = std::fabs(got - want);
    const double scale = std::fmax(1.0, std::fabs(want));
    if (!(err / scale <= tol) || std::isnan(got)) {
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "%s\n      got  %.17g\n      want %.17g\n      rel err %.3g > tol %.3g",
                      expr, got, want, err / scale, tol);
        fail(file, line, buf);
    }
}

inline void check_abs(double got, double want, double tol, const char* expr,
                      const char* file, int line) {
    ++check_count();
    const double err = std::fabs(got - want);
    if (!(err <= tol) || std::isnan(got)) {
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "%s\n      got  %.17g\n      want %.17g\n      abs err %.3g > tol %.3g",
                      expr, got, want, err, tol);
        fail(file, line, buf);
    }
}

int run_all(int argc, char** argv);

}  // namespace vsetest

#define VSE_CONCAT_(a, b) a##b
#define VSE_CONCAT(a, b) VSE_CONCAT_(a, b)

/// TEST(suite, name) { ... } — registers a property/unit test.
#define TEST(suite, name)                                                        \
    static void VSE_CONCAT(vse_test_fn_, __LINE__)();                            \
    static ::vsetest::Registrar VSE_CONCAT(vse_test_reg_, __LINE__)(             \
        #suite, #name, VSE_CONCAT(vse_test_fn_, __LINE__));                      \
    static void VSE_CONCAT(vse_test_fn_, __LINE__)()

#define CHECK(expr) ::vsetest::check_true((expr), #expr, __FILE__, __LINE__)
#define CHECK_CLOSE(got, want, tol) \
    ::vsetest::check_close((got), (want), (tol), #got " ~= " #want, __FILE__, __LINE__)
#define CHECK_ABS(got, want, tol) \
    ::vsetest::check_abs((got), (want), (tol), #got " ~= " #want, __FILE__, __LINE__)
#define CHECK_THROWS(expr)                                                       \
    do {                                                                         \
        bool threw = false;                                                      \
        try { (void)(expr); } catch (...) { threw = true; }                       \
        ::vsetest::check_true(threw, "throws: " #expr, __FILE__, __LINE__);      \
    } while (0)
