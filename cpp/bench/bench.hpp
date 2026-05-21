// Benchmark harness.
//
// Every number this emits ends up in benchmarks/RESULTS.md and, if it is any
// good, on a CV. So the harness is built to make the numbers hard to fake by
// accident:
//
//   * Results carry their units and a free-text note, and the note is where the
//     conditions live (single-threaded, working-set size, what the reference
//     was). A throughput figure without those is not a measurement.
//   * Every kernel returns a checksum that is fed to a sink the optimiser cannot
//     see through. Without that, -O2 deletes the loop and the number becomes
//     spectacular and meaningless.
//   * Timing runs a warmup pass, then repeats until it has accumulated at least
//     min_seconds, then reports the *best* of several batches. Best-of rather
//     than mean because the noise here is one-sided: scheduler preemption and
//     frequency dips can only make a run slower, never faster. The batch count
//     and duration are set high deliberately -- on a mobile part with turbo and
//     heterogeneous cores, best-of-5 over 0.4s still moved 6% run to run, which
//     is more than most of the differences worth measuring.
//   * The environment block (compiler, flags, CPU) is emitted with the results
//     rather than typed into the report by hand.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace vsebench {

struct Result {
    std::string id;      ///< stable key, e.g. "iv.throughput"
    std::string metric;  ///< human-readable name
    double      value = 0.0;
    std::string unit;
    std::string note;
};

inline std::vector<Result>& results() {
    static std::vector<Result> r;
    return r;
}

inline void report(std::string id, std::string metric, double value, std::string unit,
                   std::string note = "") {
    results().push_back({std::move(id), std::move(metric), value, std::move(unit),
                         std::move(note)});
}

/// Optimiser barrier. The checksum has to be observably consumed or the whole
/// measured loop is dead code.
inline void sink(double v) {
    static volatile double s = 0.0;
    s = s + v;
}

/// Best-of-batches timing. `f` performs `ops_per_call` operations and returns a
/// checksum. Returns nanoseconds per operation.
template <class F>
double time_ns_per_op(F&& f, long ops_per_call, double min_seconds = 2.0, int batches = 9) {
    using clock = std::chrono::steady_clock;

    sink(f());  // warm caches, branch predictors and any lazy page faults

    double best = 1e300;
    for (int b = 0; b < batches; ++b) {
        long calls = 0;
        double checksum = 0.0;
        const auto t0 = clock::now();
        double elapsed = 0.0;
        do {
            checksum += f();
            ++calls;
            elapsed = std::chrono::duration<double>(clock::now() - t0).count();
        } while (elapsed < min_seconds / batches);
        sink(checksum);
        const double per_op = elapsed * 1e9 / double(calls * ops_per_call);
        if (per_op < best) best = per_op;
    }
    return best;
}

struct Registry {
    struct Case { std::string name; std::function<void()> fn; };
    std::vector<Case> cases;
    static Registry& get() { static Registry r; return r; }
};

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        Registry::get().cases.push_back({name, std::move(fn)});
    }
};

std::string environment_json();
int run_all(int argc, char** argv);

}  // namespace vsebench

#define VSE_BENCH_CONCAT_(a, b) a##b
#define VSE_BENCH_CONCAT(a, b) VSE_BENCH_CONCAT_(a, b)

/// BENCH(name) { ... } — registers a benchmark that calls report(...).
#define BENCH(name)                                                              \
    static void VSE_BENCH_CONCAT(vse_bench_fn_, __LINE__)();                     \
    static ::vsebench::Registrar VSE_BENCH_CONCAT(vse_bench_reg_, __LINE__)(     \
        name, VSE_BENCH_CONCAT(vse_bench_fn_, __LINE__));                        \
    static void VSE_BENCH_CONCAT(vse_bench_fn_, __LINE__)()
