// Properties of the implied-volatility inversion.
//
// The headline property is the round trip: take a volatility, price it, invert
// the price, recover the volatility. Anything the solver gets wrong shows up
// here, and the grid is deliberately wider than any real option chain.
//
// The grid always inverts the out-of-the-money member of the put/call pair,
// which is what a surface pipeline feeds it and what an exchange actually
// quotes with a meaningful spread. Inverting the in-the-money side is a
// different and strictly worse-conditioned problem; it gets its own test below,
// which measures the loss rather than pretending it is not there.
#include "harness.hpp"
#include "vse/black.hpp"
#include "vse/implied_vol.hpp"

#include <vector>

using namespace vse;
using namespace vsetest;

namespace {

struct GridPoint {
    Real K, T, sigma;
    OptionType type;
};

/// Strikes placed at fixed multiples of a standard deviation, which is the only
/// scale on which "how far out" means the same thing at one day and five years.
std::vector<GridPoint> round_trip_grid(Real F, Real z_max, int n_z) {
    std::vector<GridPoint> g;
    for (Real T : {1.0 / 365, 3.0 / 365, 7.0 / 365, 1.0 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        for (Real sigma : {0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.70, 1.00, 1.50}) {
            const Real s = sigma * std::sqrt(T);
            for (int i = 0; i < n_z; ++i) {
                const Real z = -z_max + 2.0 * z_max * i / Real(n_z - 1);
                g.push_back({F * std::exp(z * s), T, sigma,
                             z >= 0.0 ? OptionType::Call : OptionType::Put});
            }
        }
    }
    return g;
}

}  // namespace

TEST(implied_vol, round_trips_over_a_10k_point_grid) {
    const Real F = 100.0, df = 0.97;
    const auto grid = round_trip_grid(F, 5.0, 101);
    CHECK(grid.size() >= 10000);

    Real worst_sigma = 0.0, worst_price = 0.0;
    long total_iterations = 0, max_iterations = 0, inverted = 0;

    for (const auto& g : grid) {
        const Real price = black76(F, g.K, g.T, g.sigma, df, g.type);
        if (price <= 0.0) continue;  // below the smallest representable double
        const auto r = implied_volatility_ex(price, F, g.K, g.T, df, g.type);
        CHECK(r.converged);
        ++inverted;
        total_iterations += r.iterations;
        max_iterations = std::max<long>(max_iterations, r.iterations);

        worst_sigma = std::fmax(worst_sigma, std::fabs(r.sigma / g.sigma - 1.0));
        const Real price2 = black76(F, g.K, g.T, r.sigma, df, g.type);
        worst_price = std::fmax(worst_price, std::fabs(price2 / price - 1.0));
    }

    std::printf("       %ld points inverted, worst |dsigma/sigma| = %.2e, "
                "worst |dprice/price| = %.2e,\n"
                "       %.2f iterations mean, %ld max\n",
                inverted, worst_sigma, worst_price,
                double(total_iterations) / double(inverted), max_iterations);

    // Measured on this grid: 8e-15 in sigma. That is the conditioning limit, not
    // a solver limit -- perturbing the price by one ulp at the worst point
    // already moves sigma by 3e-15.
    CHECK(worst_sigma < 5e-14);
    CHECK(worst_price < 5e-14);
}

TEST(implied_vol, holds_up_at_eight_standard_deviations) {
    // Past five standard deviations no exchange quotes anything, but the solver
    // should degrade gracefully rather than diverge or return garbage. The
    // limiting factor out here is the price itself: once it drops below the
    // smallest representable double there is no information left to invert.
    const Real F = 100.0, df = 1.0;
    const auto grid = round_trip_grid(F, 8.0, 33);
    Real worst = 0.0;
    long n = 0;
    for (const auto& g : grid) {
        const Real price = black76(F, g.K, g.T, g.sigma, df, g.type);
        if (price <= 1e-290) continue;
        const auto r = implied_volatility_ex(price, F, g.K, g.T, df, g.type);
        CHECK(r.converged);
        worst = std::fmax(worst, std::fabs(r.sigma / g.sigma - 1.0));
        ++n;
    }
    std::printf("       %ld points beyond 5 sd, worst |dsigma/sigma| = %.2e\n", n, worst);
    CHECK(worst < 1e-12);
}

TEST(implied_vol, in_the_money_inversion_degrades_by_the_intrinsic_ratio) {
    // Recovering the time value of an in-the-money quote means subtracting an
    // intrinsic value that can be many orders of magnitude larger, and those
    // digits are gone before the solver is called. The loss is therefore
    // predictable: about eps * intrinsic / time_value in the price, amplified
    // into sigma by price/(vega * sigma).
    //
    // This is a property of the arithmetic, not of the implementation, and the
    // point of pinning it is that the number should track the prediction rather
    // than being merely "bad".
    const Real F = 100.0, T = 1.0, df = 1.0, sigma = 0.20;
    std::printf("       K/F   intrinsic/time  predicted    measured\n");
    for (Real moneyness : {1.05, 1.2, 1.5, 2.0, 3.0}) {
        const Real K = F / moneyness;               // in-the-money call
        const Real price = black76(F, K, T, sigma, df, OptionType::Call);
        const Real intrinsic = F - K;
        const Real time_value = price - intrinsic;
        const Real g = bs_greeks(F, K, T, 0.0, 0.0, sigma, OptionType::Call).vega;
        const Real predicted = DBL_EPS * (intrinsic / time_value) * (price / (g * sigma));
        const Real iv = implied_volatility(price, F, K, T, df, OptionType::Call);
        const Real measured = std::fabs(iv / sigma - 1.0);
        std::printf("       %.2f  %13.3e  %.2e     %.2e\n",
                    1.0 / moneyness, intrinsic / time_value, predicted, measured);
        CHECK(measured <= std::fmax(2000.0 * predicted, 1e-14));
    }
    // The out-of-the-money member of the same pair carries identical information
    // with none of the cancellation, and is what the caller should use.
    for (Real moneyness : {1.05, 1.2, 1.5, 2.0, 3.0}) {
        const Real K = F / moneyness;
        const Real put = black76(F, K, T, sigma, df, OptionType::Put);
        CHECK_CLOSE(implied_volatility(put, F, K, T, df, OptionType::Put), sigma, 1e-13);
    }
}

TEST(implied_vol, matches_the_closed_form_at_the_money) {
    // b(0,s) = 2 N(s/2) - 1, so 2 N^{-1}((1+beta)/2) is the exact inverse -- on
    // paper. In doubles it is not, because (1 + beta)/2 buries a small beta in
    // the mantissa of 1. The solver uses it as a seed only and refines on the
    // price, so the recovered vol beats the closed form it started from.
    for (Real s : {1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0}) {
        const Real beta = normalised_black(0.0, s);
        const auto r = implied_total_volatility(beta, 0.0);
        CHECK_CLOSE(r.sigma, s, 1e-14);
        CHECK(r.converged);
        const Real closed_form = 2.0 * norm_inv_cdf(0.5 * (1.0 + beta));
        CHECK(std::fabs(r.sigma / s - 1.0) <= std::fabs(closed_form / s - 1.0) + 1e-16);
    }
}

TEST(implied_vol, inverts_both_branches_of_the_inflection_split) {
    // s_c = sqrt(2|x|) separates the two seeds and the two objectives. Walk
    // across it to make sure nothing is discontinuous at the handover.
    for (Real x : {-2.0, -0.5, -0.05}) {
        const Real s_c = normalised_black_inflection(x);
        for (Real f : {0.5, 0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1, 2.0}) {
            const Real s = s_c * f;
            const Real beta = normalised_black(x, s);
            if (beta <= 0.0) continue;
            const auto r = implied_total_volatility(beta, x);
            CHECK(r.converged);
            CHECK_CLOSE(r.sigma, s, 1e-13);
        }
    }
}

TEST(implied_vol, recovers_volatility_from_quoted_premiums) {
    // The market-facing entry point on a realistic SPX-shaped chain, always on
    // the out-of-the-money side, with a non-unit discount factor.
    const Real F = 4275.0, T = 0.37, df = 0.982;
    for (Real K : {2500.0, 3400.0, 4000.0, 4275.0, 4500.0, 5200.0, 6800.0}) {
        const auto type = K >= F ? OptionType::Call : OptionType::Put;
        for (Real sigma : {0.09, 0.15, 0.24, 0.55}) {
            const Real p = black76(F, K, T, sigma, df, type);
            CHECK_CLOSE(implied_volatility(p, F, K, T, df, type), sigma, 1e-12);
        }
    }
}

TEST(implied_vol, is_monotone_in_price) {
    const Real x = -0.4;
    Real prev = -1.0;
    for (Real beta = 1e-6; beta < std::exp(0.5 * x) * 0.999; beta *= 1.2) {
        const auto r = implied_total_volatility(beta, x);
        CHECK(r.sigma > prev);
        prev = r.sigma;
    }
}

TEST(implied_vol, returns_zero_at_intrinsic_value) {
    for (Real x : {-1.0, -0.1, 0.0, 0.1, 1.0}) {
        const Real intrinsic = std::fmax(std::exp(0.5 * x) - std::exp(-0.5 * x), 0.0);
        const auto r = implied_total_volatility(intrinsic, x);
        CHECK_ABS(r.sigma, 0.0, 1e-14);
        CHECK(r.converged);
    }
}

TEST(implied_vol, rejects_prices_that_admit_arbitrage) {
    // Below intrinsic, and at or above the forward, are not hard inversions --
    // they are bad quotes, and the caller needs to be told the difference.
    CHECK_THROWS(implied_total_volatility(-0.1, -0.5));
    CHECK_THROWS(implied_total_volatility(std::exp(-0.25), -0.5));       // == e^{x/2}
    CHECK_THROWS(implied_total_volatility(2.0 * std::exp(-0.25), -0.5));
    CHECK_THROWS(implied_volatility(-1.0, 100.0, 100.0, 1.0, 1.0, OptionType::Call));
    CHECK_THROWS(implied_volatility(101.0, 100.0, 100.0, 1.0, 1.0, OptionType::Call));
    CHECK_THROWS(implied_volatility(5.0, 100.0, 100.0, 0.0, 1.0, OptionType::Call));
    CHECK_THROWS(implied_volatility(5.0, -100.0, 100.0, 1.0, 1.0, OptionType::Call));
}

TEST(implied_vol, survives_a_price_pressed_against_the_bound) {
    // Real chains contain quotes sitting right on the no-arbitrage boundary, and
    // a solver seeded purely by fitted rational functions has no reason to
    // behave there. The bracket does. Note what is and is not asserted: the vol
    // implied by a price within 1e-12 of the forward is enormous and its exact
    // value is not meaningful, so the property is that the solver returns a
    // finite, positive, price-consistent answer -- not a particular number.
    for (Real x : {-1.5, -0.3, -0.01}) {
        const Real b_max = std::exp(0.5 * x);
        for (Real eps : {1e-3, 1e-6, 1e-9, 1e-12}) {
            const Real beta = b_max * (1.0 - eps);
            const auto r = implied_total_volatility(beta, x);
            CHECK(std::isfinite(r.sigma));
            CHECK(r.sigma > 0.0);
            CHECK_CLOSE(normalised_black(x, r.sigma) / beta, 1.0, 1e-9);
        }
        for (Real eps : {1e-6, 1e-12, 1e-15}) {
            const auto r = implied_total_volatility(eps * b_max, x);
            CHECK(r.converged);
            CHECK(r.sigma >= 0.0);
            CHECK(std::isfinite(r.sigma));
        }
    }
}
