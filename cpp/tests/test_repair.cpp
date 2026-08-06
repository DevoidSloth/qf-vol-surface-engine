// Projecting an arbitrageable smile onto the nearest one that is a distribution.
#include "harness.hpp"
#include "vse/sabr.hpp"
#include "vse/smile_repair.hpp"
#include "vse/svi.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace vse;
using namespace vsetest;

namespace {

/// Long-dated rates parameters where Hagan's expansion is known to fail.
SABRParams broken_sabr(Real beta = 0.5) {
    const Real F = 0.03, T = 10.0;
    const Real atm = sabr_lognormal_vol(SABRParams{0.025, 0.5, -0.2, 0.45, 0.0}, F, F, T);
    return SABRParams{sabr_alpha_from_atm(atm, F, T, beta, -0.2, 0.45, 0.0), beta, -0.2, 0.45,
                      0.0};
}

}  // namespace

TEST(isotonic, is_the_l2_projection_onto_non_decreasing_sequences) {
    // Two properties define it, and both are checked against a brute-force
    // search rather than against another implementation of the same idea.
    std::mt19937 rng(20240701);
    std::uniform_real_distribution<Real> draw(-1.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<Real> y(8);
        for (Real& v : y) v = draw(rng);
        std::vector<Real> fitted = y;
        isotonic_increasing(fitted);

        // Non-decreasing.
        for (std::size_t i = 1; i < fitted.size(); ++i) CHECK(fitted[i] >= fitted[i - 1] - 1e-14);

        // The mean is preserved -- a projection onto a cone containing the
        // constants cannot move the centroid.
        const Real sum_in = std::accumulate(y.begin(), y.end(), 0.0);
        const Real sum_out = std::accumulate(fitted.begin(), fitted.end(), 0.0);
        CHECK_CLOSE(sum_out, sum_in, 1e-12);

        // And it is nearest: no non-decreasing perturbation of the answer is
        // closer to the input. Tested by perturbing along directions that keep
        // feasibility, which is enough to catch an implementation that merges
        // the wrong blocks.
        auto distance = [&](const std::vector<Real>& z) {
            Real d = 0.0;
            for (std::size_t i = 0; i < z.size(); ++i) d += sqr(z[i] - y[i]);
            return d;
        };
        const Real best = distance(fitted);
        for (std::size_t i = 0; i < fitted.size(); ++i) {
            for (Real step : {-1e-3, 1e-3}) {
                std::vector<Real> candidate = fitted;
                candidate[i] += step;
                if (!std::is_sorted(candidate.begin(), candidate.end())) continue;
                CHECK(distance(candidate) >= best - 1e-15);
            }
        }
    }
}

TEST(isotonic, leaves_an_already_sorted_sequence_alone) {
    std::vector<Real> y{-2.0, -0.5, 0.0, 0.0, 3.0, 7.5};
    const std::vector<Real> before = y;
    isotonic_increasing(y);
    for (std::size_t i = 0; i < y.size(); ++i) CHECK_CLOSE(y[i], before[i], 0.0);
}

TEST(repair, removes_every_butterfly_violation_from_a_broken_sabr_smile) {
    // The headline claim. Three betas, spanning a smile that is badly
    // arbitraged to one that is barely.
    std::printf("       %-6s %10s %10s %13s %13s %11s\n", "beta", "viol in", "viol out",
                "density in", "density out", "max d(vol)");
    for (Real beta : {0.0, 0.5, 1.0}) {
        const auto p = broken_sabr(beta);
        const auto r = repair_sabr(p, 0.03, 10.0);
        std::printf("       %-6.2f %10d %10d %13.3e %13.3e %10.2f%%\n", beta,
                    r.report.violations_before, r.report.violations_after,
                    r.report.min_density_before, r.report.min_density_after,
                    r.report.max_vol_change * 100.0);
        CHECK(r.report.violations_before > 0);      // the input really is broken
        CHECK(r.report.violations_after == 0);      // and the output really is not
        CHECK(r.report.min_density_after >= -r.report.density_tolerance);
        CHECK(r.report.repaired);
    }
}

TEST(repair, leaves_a_smile_that_is_already_a_distribution_alone) {
    // The other half of the claim, and the one that would catch a "repair" that
    // simply flattens everything. An SVI slice satisfying Durrleman's condition
    // must come back unchanged to the accuracy of the price round trip.
    const SVIRaw s{0.012, 0.09, -0.35, 0.02, 0.14};
    const Real F = 100.0, T = 1.5;
    const auto bf = check_butterfly(s, T, 0.0, 2001);
    CHECK(bf.free);

    const auto r = repair_smile([&](Real K) { return s.implied_vol(std::log(K / F), T); }, F, T,
                                0.25, 2.5, 1201);
    std::printf("       already arbitrage-free: %d violations, worst vol change %.2e\n",
                r.report.violations_before, r.report.max_vol_change);
    CHECK(r.report.violations_before == 0);
    CHECK(!r.report.repaired);
    // Round-tripping price to vol and back is what limits this, not the
    // projection, which is the identity on a feasible point.
    CHECK(r.report.max_vol_change < 1e-9);
}

TEST(repair, the_repaired_prices_are_convex_and_monotone_in_strike) {
    // What "arbitrage-free" means, stated directly on the prices rather than
    // via the density estimate that the report uses. Checking both is the point:
    // the density is a second difference and could hide a violation inside its
    // own rounding, but convexity of the reconstructed call curve is exact.
    const auto p = broken_sabr(0.5);
    const auto r = repair_sabr(p, 0.03, 10.0);
    const std::size_t n = r.strikes.size();

    auto call = [&](std::size_t i) {
        return r.strikes[i] < r.forward ? r.otm_price[i] + r.forward - r.strikes[i]
                                        : r.otm_price[i];
    };
    const Real h = r.strikes[1] - r.strikes[0];
    int slope_failures = 0, convexity_failures = 0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const Real slope = (call(i + 1) - call(i)) / h;
        // A call price is non-increasing in strike, and never falls faster than
        // one for one -- that bound IS the digital price being between 0 and 1.
        if (!(slope <= 1e-12 && slope >= -1.0 - 1e-12)) ++slope_failures;
    }
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const bool straddles = (r.strikes[i - 1] < r.forward) != (r.strikes[i + 1] < r.forward);
        if (straddles) continue;
        const Real d2 = call(i + 1) - 2.0 * call(i) + call(i - 1);
        if (d2 < -1e-14 * r.forward) ++convexity_failures;
    }
    std::printf("       %zu strikes: %d slope failures, %d convexity failures\n", n,
                slope_failures, convexity_failures);
    CHECK(slope_failures == 0);
    CHECK(convexity_failures == 0);
}

TEST(repair, moves_the_smile_only_where_it_was_broken) {
    // The cost has to have the right SHAPE, not just the right size. A
    // projection that spread its correction evenly would produce the same worst
    // case while corrupting strikes that were fine, and the max alone cannot
    // tell the two apart.
    const auto p = broken_sabr(0.5);
    const Real F = 0.03;
    const auto scan = sabr_density_scan(p, F, 10.0, 0.02, 3.0, 1500, false);
    const auto r = repair_sabr(p, F, 10.0);
    CHECK(scan.arbitrage_boundary > 0.0);

    Real worst_inside = 0.0, worst_outside = 0.0;
    for (std::size_t i = 0; i < r.strikes.size(); ++i) {
        const Real change = std::fabs(r.implied_vol[i] - r.input_vol[i]);
        // "Outside" means comfortably above the boundary the scan located.
        if (r.strikes[i] > 1.5 * scan.arbitrage_boundary) {
            worst_outside = std::fmax(worst_outside, change);
        } else {
            worst_inside = std::fmax(worst_inside, change);
        }
    }
    std::printf("       boundary K/F %.3f: worst change %.3f%% below, %.3f%% well above\n",
                scan.arbitrage_boundary / F, worst_inside * 100.0, worst_outside * 100.0);
    CHECK(worst_inside > 10.0 * worst_outside);
}

TEST(repair, is_idempotent) {
    // Projecting onto a convex set twice is projecting once. If a second pass
    // moved anything, the first did not land in the set.
    const auto p = broken_sabr(0.0);
    const auto first = repair_sabr(p, 0.03, 10.0);
    const auto second = repair_smile([&](Real K) { return first.vol_at(K); }, first.forward,
                                     first.expiry, 0.02, 3.0, 1501);
    std::printf("       second pass: %d violations, worst vol change %.2e\n",
                second.report.violations_before, second.report.max_vol_change);
    CHECK(second.report.violations_before == 0);
    // The interpolation in vol_at is piecewise linear, so the second pass sees
    // a slightly different function; the tolerance is that interpolation, not
    // the projection.
    CHECK(second.report.max_vol_change < 1e-4);
}

TEST(repair, restores_convexity_but_not_missing_mass) {
    // A limitation, measured. Convexity constrains the SHAPE of the call curve
    // and says nothing about its level, so the projection cannot invent
    // probability that the input smile never had.
    //
    // Hagan's expansion does not have it, and by a much wider margin than the
    // negative density suggests. As the strike goes to zero the expansion sends
    // the volatility to 117%, which leaves the put at K = 0.001F worth 15% of
    // its strike; measured end to end, the input smile carries 0.65 of a unit
    // of probability. It is not a near-miss distribution with a negative patch
    // in it -- it is not a distribution at all, and the negative density is the
    // smaller of its two problems.
    //
    // The projection recovers about two thirds of the shortfall, because
    // clamping the slopes to [-1, 0] and [0, 1] constrains level as well as
    // shape: those bounds are the digital prices, and forcing them back between
    // zero and one drags mass with them. It cannot close the rest.
    //
    // An SVI slice satisfies Lee's moment bounds by construction, so its wings
    // have the right asymptotics and the same measurement returns unit mass.
    const auto p = broken_sabr(0.5);
    const auto sabr = repair_sabr(p, 0.03, 10.0, 0.001, 8.0, 4001);

    const SVIRaw s{0.012, 0.09, -0.35, 0.02, 0.14};
    const auto svi = repair_smile([&](Real K) { return s.implied_vol(std::log(K / 100.0), 1.5); },
                                  100.0, 1.5, 0.02, 6.0, 4001);

    std::printf("       %-22s %12s %12s %12s\n", "smile", "mass before", "mass after",
                "vol at K_lo");
    std::printf("       %-22s %12.6f %12.6f %11.1f%%\n", "SABR (Hagan)",
                sabr.report.mass_before, sabr.report.mass_after,
                sabr.input_vol.front() * 100.0);
    std::printf("       %-22s %12.6f %12.6f %11.1f%%\n", "SVI (Lee-compliant)",
                svi.report.mass_before, svi.report.mass_after,
                svi.input_vol.front() * 100.0);

    // SVI: sound wings in, unit mass out. The tolerance is the one-sided
    // difference used for the endpoint slopes, which is O(h), not the method.
    CHECK_CLOSE(svi.report.mass_after, 1.0, 1e-6);

    // SABR, and this is not what I expected to be asserting. Hagan's smile is
    // short a THIRD of its probability mass before the repair -- it is not a
    // near-miss distribution with a negative patch, it is not a distribution at
    // all. The projection then recovers about two thirds of what is missing,
    // because clamping the slopes to [-1, 0] and [0, 1] is a statement about
    // level as well as shape: it forces the digital prices back between zero
    // and one, and mass follows.
    CHECK(sabr.report.mass_before < 0.75);
    CHECK(sabr.report.mass_after > sabr.report.mass_before + 0.15);

    // What it cannot do is close the gap. Convexity and the slope bounds still
    // leave the level free, and the input's low-strike blow-up is a hole the
    // projection has no way to fill.
    CHECK(sabr.report.mass_after < 0.95);
}
