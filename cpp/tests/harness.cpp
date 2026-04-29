#include "harness.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace vsetest {

int run_all(int argc, char** argv) {
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--filter=", 9) == 0) filter = argv[i] + 9;
    }

    auto& cases = Registry::get().cases;
    std::stable_sort(cases.begin(), cases.end(),
                     [](const auto& a, const auto& b) { return a.suite < b.suite; });

    int passed = 0, failed = 0, skipped = 0;
    std::string current_suite;
    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& c : cases) {
        if (!filter.empty() &&
            (c.suite + "." + c.name).find(filter) == std::string::npos) {
            ++skipped;
            continue;
        }
        if (c.suite != current_suite) {
            current_suite = c.suite;
            std::printf("\n[%s]\n", current_suite.c_str());
        }
        const int before = check_count();
        try {
            c.fn();
            std::printf("  ok   %-46s (%d checks)\n", c.name.c_str(),
                        check_count() - before);
            ++passed;
        } catch (const Failure& f) {
            std::printf("  FAIL %-46s\n      %s\n", c.name.c_str(), f.what());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("  FAIL %-46s\n      unexpected exception: %s\n",
                        c.name.c_str(), e.what());
            ++failed;
        }
    }

    const auto dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%d properties passed, %d failed", passed, failed);
    if (skipped) std::printf(", %d filtered out", skipped);
    std::printf("  (%d assertions, %.2fs)\n", check_count(), dt);
    return failed == 0 ? 0 : 1;
}

}  // namespace vsetest

int main(int argc, char** argv) { return vsetest::run_all(argc, argv); }
