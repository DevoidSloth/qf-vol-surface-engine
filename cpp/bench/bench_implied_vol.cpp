// Throughput and accuracy of the Black layer and the implied-vol inversion.
#include "bench.hpp"
#include "vse/black.hpp"
#include "vse/implied_vol.hpp"

#include <array>
#include <map>
#include <vector>

using namespace vse;

namespace {

struct Point { Real x, s, beta; };

/// The same grid the round-trip property test uses: ten expiries x ten vols x
/// 101 strikes from -5 to +5 standard deviations, out-of-the-money side only.
/// 10,100 points at 24 bytes is 242 KB, so the working set lives in L2 on any
/// current core and the figure is a compute number, not a memory-bandwidth one.
std::vector<Point> grid() {
    std::vector<Point> g;
    for (Real T : {1.0 / 365, 3.0 / 365, 7.0 / 365, 1.0 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        for (Real sigma : {0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.70, 1.00, 1.50}) {
            const Real s = sigma * std::sqrt(T);
            for (int i = 0; i < 101; ++i) {
                const Real z = -5.0 + 10.0 * i / 100.0;
                const Real x = -std::fabs(z) * s;
                const Real beta = normalised_black(x, s);
                if (beta > 0.0) g.push_back({x, s, beta});
            }
        }
    }
    return g;
}

}  // namespace

BENCH("black.price") {
    const auto g = grid();
    const double ns = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (const auto& p : g) acc += normalised_black(p.x, p.s);
        return acc;
    }, long(g.size()));

    vsebench::report("black.price.latency", "Normalised Black price", ns, "ns/option",
                     "single-threaded, 242 KB working set");
    vsebench::report("black.price.throughput", "Normalised Black price", 1e3 / ns,
                     "M prices/sec", "single-threaded");
}

BENCH("iv.inversion") {
    const auto g = grid();

    // Accuracy first: a throughput number for a routine that gets the wrong
    // answer is not worth printing, so both come out of the same grid.
    double worst_sigma = 0.0, worst_price = 0.0;
    long total_iterations = 0, non_converged = 0;
    std::map<int, long> histogram;
    for (const auto& p : g) {
        const auto r = implied_total_volatility(p.beta, p.x);
        if (!r.converged) ++non_converged;
        total_iterations += r.iterations;
        ++histogram[r.iterations];
        worst_sigma = std::fmax(worst_sigma, std::fabs(r.sigma / p.s - 1.0));
        worst_price = std::fmax(worst_price,
                                std::fabs(normalised_black(p.x, r.sigma) / p.beta - 1.0));
    }

    const double ns = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (const auto& p : g) acc += implied_total_volatility(p.beta, p.x).sigma;
        return acc;
    }, long(g.size()));

    std::printf("  iterations:");
    for (const auto& [k, v] : histogram) std::printf(" %d:%ld", k, v);
    std::printf("\n");

    vsebench::report("iv.accuracy.sigma", "Implied vol round-trip error (sigma)",
                     worst_sigma, "max relative",
                     "10,100-point grid, +/-5 sd, 1 day to 10 years, 3 to 150 vol points");
    vsebench::report("iv.accuracy.price", "Implied vol round-trip error (price)",
                     worst_price, "max relative", "same grid");
    vsebench::report("iv.iterations.mean", "Householder steps per inversion",
                     double(total_iterations) / double(g.size()), "iterations",
                     "excludes the two bracket evaluations");
    vsebench::report("iv.non_converged", "Inversions that hit the iteration cap",
                     double(non_converged), "count", "");
    vsebench::report("iv.latency", "Implied vol inversion", ns, "ns/inversion",
                     "single-threaded, 242 KB working set");
    vsebench::report("iv.throughput", "Implied vol inversion", 1e3 / ns,
                     "M inversions/sec", "single-threaded");
}

BENCH("black.greeks") {
    std::vector<std::array<Real, 4>> pts;
    for (int i = 0; i < 4000; ++i) {
        const Real u = (i % 100) / 100.0;
        pts.push_back({100.0, 60.0 + 80.0 * u, 0.05 + 3.0 * ((i / 100) % 40) / 40.0,
                       0.08 + 0.9 * u});
    }
    const double ns = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (const auto& p : pts) {
            const Greeks gk = bs_greeks(p[0], p[1], p[2], 0.03, 0.01, p[3], OptionType::Call);
            acc += gk.price + gk.delta + gk.gamma + gk.vega + gk.theta + gk.rho +
                   gk.vanna + gk.volga + gk.dual_delta + gk.dual_gamma;
        }
        return acc;
    }, long(pts.size()));
    vsebench::report("greeks.analytic.latency", "Ten analytic Black-Scholes Greeks",
                     ns, "ns/option", "single-threaded, price plus 9 sensitivities");
}
