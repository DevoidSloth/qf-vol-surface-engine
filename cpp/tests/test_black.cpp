// Properties of the normal functions and the Black-Scholes layer.
#include "harness.hpp"
#include "reference_values.hpp"
#include "vse/black.hpp"
#include "vse/normal.hpp"

#include <vector>

using namespace vse;
using namespace vsetest;

namespace {

/// Grid used by several properties: strikes 5 sigma either side of the forward,
/// expiries from one day to five years, vols from 5 to 150 points.
struct GridPoint { Real F, K, T, sigma; };

std::vector<GridPoint> moneyness_grid() {
    std::vector<GridPoint> g;
    const Real F = 100.0;
    for (Real T : {1.0 / 365, 7.0 / 365, 1.0 / 12, 0.25, 0.5, 1.0, 2.0, 5.0}) {
        for (Real sigma : {0.05, 0.10, 0.20, 0.40, 0.80, 1.50}) {
            const Real sd = sigma * std::sqrt(T);
            for (Real z : {-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0}) {
                g.push_back({F, F * std::exp(z * sd), T, sigma});
            }
        }
    }
    return g;
}

}  // namespace

// ---------------------------------------------------------------------------
// Normal functions
// ---------------------------------------------------------------------------

TEST(normal, cdf_matches_50_digit_reference) {
    // Relative, not absolute: N(-20) is 2.75e-89 and an absolute tolerance would
    // pass on any implementation that simply returned zero.
    //
    // Measured profile of this implementation against mpmath (scripts/audit_normal.py):
    //   |x| <=  7   1.1e-16
    //   |x|  = 10   8.9e-15
    //   |x|  = 20   3.6e-14   (value 2.8e-89)
    //   |x|  = 36   2.4e-14   (value 4.2e-284)
    // The drift past |x| = 10 is the residual rounding of Cody's exp(-t^2) *
    // exp(-(y-t)(y+t)) argument split. For comparison, the Hart/West rational
    // form this replaced was already at 2.6e-9 by |x| = 7.
    for (const auto& c : kNormCdfReference) {
        CHECK_CLOSE(norm_cdf(c.x) / c.cdf, 1.0, 1e-13);
    }
}

TEST(normal, cdf_is_monotone_and_bounded) {
    Real prev = 0.0;
    for (int i = -4000; i <= 4000; ++i) {
        const Real x = i * 0.01;
        const Real c = norm_cdf(x);
        CHECK(c >= prev);
        CHECK(c >= 0.0 && c <= 1.0);
        prev = c;
    }
}

TEST(normal, cdf_reflection_identity) {
    for (int i = 0; i <= 800; ++i) {
        const Real x = i * 0.01;
        CHECK_ABS(norm_cdf(x) + norm_cdf(-x), 1.0, 1e-15);
    }
}

TEST(normal, pdf_is_derivative_of_cdf) {
    const Real h = 1e-5;
    for (Real x = -6.0; x <= 6.0; x += 0.25) {
        const Real fd = (norm_cdf(x + h) - norm_cdf(x - h)) / (2 * h);
        CHECK_CLOSE(fd, norm_pdf(x), 1e-9);
    }
}

TEST(normal, inverse_cdf_round_trips) {
    for (int i = 1; i < 1000; ++i) {
        const Real p = i / 1000.0;
        CHECK_ABS(norm_cdf(norm_inv_cdf(p)), p, 5e-16);
    }
    // Tails, where Acklam alone is only good to ~1e-9 and the Halley step earns
    // its keep.
    for (Real p : {1e-12, 1e-9, 1e-6, 1e-3, 0.999, 1 - 1e-9, 1 - 1e-12}) {
        CHECK_CLOSE(norm_cdf(norm_inv_cdf(p)) / p, 1.0, 1e-13);
    }
}

TEST(normal, inverse_cdf_rejects_out_of_range) {
    CHECK_THROWS(norm_inv_cdf(0.0));
    CHECK_THROWS(norm_inv_cdf(1.0));
    CHECK_THROWS(norm_inv_cdf(-0.5));
}

TEST(normal, erfcx_matches_its_definition_where_both_are_representable) {
    // erfcx(x) = e^{x^2} erfc(x). Below x = 5 the naive right-hand side is
    // computable and the two must agree; above it the right-hand side dies of
    // underflow and erfcx is the only one still carrying digits.
    for (Real x = -3.0; x <= 5.0; x += 0.05) {
        CHECK_CLOSE(erfcx(x), std::exp(x * x) * erfc_(x), 1e-13);
    }
    for (Real x : {30.0, 1e3, 1e7, 1e9, 1e30}) {
        // erfcx(x) -> 1/(x sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4)); the truncation
        // error of that three-term series is 15/(8 x^6), so it is only a fair
        // reference well past x = 20.
        const Real a = (1.0 / (x * std::sqrt(PI))) *
                       (1.0 - 0.5 / (x * x) + 0.75 / std::pow(x, 4));
        CHECK_CLOSE(erfcx(x), a, 1e-10);
        CHECK(erfcx(x) > 0.0);
    }
}

TEST(normal, erf_and_erfc_are_complementary) {
    for (Real x = -6.0; x <= 6.0; x += 0.05) {
        CHECK_ABS(erf_(x) + erfc_(x), 1.0, 1e-15);
    }
}

TEST(normal, mills_ratio_is_smooth_and_matches_its_asymptote) {
    // No branch to be discontinuous across any more: R(z) is one erfcx call.
    // Check it against the direct quotient where that is well conditioned, and
    // against the asymptotic series where it is not.
    for (Real z = -2.0; z <= 5.0; z += 0.01) {
        CHECK_CLOSE(mills_ratio(z), norm_cdf(-z) / norm_pdf(z), 1e-13);
    }
    for (Real z : {40.0, 100.0, 1e4, 1e8}) {
        const Real asym = 1.0 / z - 1.0 / (z * z * z) + 3.0 / std::pow(z, 5);
        CHECK_CLOSE(mills_ratio(z), asym, 1e-8);
    }
    // R satisfies R'(z) = z R(z) - 1 exactly.
    for (Real z = -2.0; z <= 30.0; z += 0.25) {
        const Real h = 1e-6;
        const Real fd = (mills_ratio(z + h) - mills_ratio(z - h)) / (2 * h);
        CHECK_CLOSE(fd, z * mills_ratio(z) - 1.0, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Normalised Black
// ---------------------------------------------------------------------------

TEST(normalised_black, matches_50_digit_reference) {
    // 1e-13 rather than machine epsilon. Near the money the Taylor branch of
    // mills_difference removes the cancellation entirely and these come back
    // exact; what is left is the deep wing, where R(u-t) and R(u+t) are far
    // enough apart that the series is not selected and the subtraction costs
    // about log10(|x|/s^2) digits. Measured worst over the reference set is
    // 6.9e-14, at x = -1, s = 0.05 -- twenty standard deviations out, where the
    // option is worth 6.8e-92 of sqrt(F K). It does not reach the inverted vol:
    // vega there is 8e13 times the price, so the same error is 2e-16 in sigma.
    for (const auto& c : kNormalisedBlackReference) {
        CHECK_CLOSE(normalised_black(c.x, c.s) / c.b, 1.0, 1e-13);
    }
}

TEST(normalised_black, satisfies_parity_in_normalised_units) {
    for (Real x = -6.0; x <= 6.0; x += 0.13) {
        for (Real s : {0.02, 0.1, 0.5, 1.0, 3.0}) {
            const Real lhs = normalised_black(x, s) - normalised_black(-x, s);
            const Real rhs = std::exp(0.5 * x) - std::exp(-0.5 * x);
            CHECK_ABS(lhs, rhs, 1e-12 * std::fmax(1.0, std::fabs(rhs)));
        }
    }
}

TEST(normalised_black, vega_is_the_exact_s_derivative) {
    // db/ds = nu(x,s) is an identity, not an approximation. If it ever fails,
    // the Householder step in the implied-vol solver is differentiating a
    // different function from the one it is inverting.
    for (Real x : {-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0}) {
        for (Real s : {0.05, 0.2, 0.7, 1.5}) {
            const Real h = 1e-6 * s;
            const Real fd = (normalised_black(x, s + h) - normalised_black(x, s - h)) / (2 * h);
            CHECK_CLOSE(fd, normalised_vega(x, s), 1e-7);
        }
    }
}

TEST(normalised_black, second_and_third_derivatives_agree_with_differences) {
    for (Real x : {-2.5, -0.7, 0.0, 0.7, 2.5}) {
        for (Real s : {0.15, 0.4, 1.1}) {
            const Real h = 1e-5 * s;
            const Real d2 = (normalised_vega(x, s + h) - normalised_vega(x, s - h)) / (2 * h);
            CHECK_CLOSE(d2, normalised_black_d2(x, s), 1e-6);
            const Real d3 = (normalised_black_d2(x, s + h) - normalised_black_d2(x, s - h)) / (2 * h);
            CHECK_CLOSE(d3, normalised_black_d3(x, s), 1e-5);
        }
    }
}

TEST(normalised_black, inflection_point_is_where_the_curvature_vanishes) {
    for (Real x : {-4.0, -2.0, -0.5, 0.5, 2.0, 4.0}) {
        const Real sc = normalised_black_inflection(x);
        CHECK_ABS(normalised_black_d2(x, sc), 0.0, 1e-14);
    }
}

TEST(normalised_black, is_increasing_in_s) {
    // Non-decreasing everywhere; strictly increasing wherever the value is
    // actually representable. Far enough into the wing b underflows to exactly
    // zero, which is the right answer, not a monotonicity violation.
    for (Real x : {-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0}) {
        const Real intrinsic = std::fmax(std::exp(0.5 * x) - std::exp(-0.5 * x), 0.0);
        Real prev = -1.0;
        for (Real s = 1e-3; s < 6.0; s *= 1.05) {
            const Real b = normalised_black(x, s);
            CHECK(b >= prev);
            // Strictness is only meaningful where the time value is above the
            // resolution of a double: an in-the-money option at s = 1e-3 sits on
            // its intrinsic value and the next increment is below one ulp.
            const bool resolvable = prev > 1e-290 &&
                                    (prev - intrinsic) > 1e-12 * std::fmax(1.0, intrinsic);
            if (resolvable) CHECK(b > prev);
            prev = b;
        }
    }
}

TEST(normalised_black, respects_its_limits) {
    for (Real x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        const Real intrinsic = std::fmax(std::exp(0.5 * x) - std::exp(-0.5 * x), 0.0);
        CHECK_ABS(normalised_black(x, 1e-12), intrinsic, 1e-11);
        CHECK_ABS(normalised_black(x, 60.0), std::exp(0.5 * x), 1e-10);
        for (Real s : {0.1, 0.5, 2.0}) {
            const Real b = normalised_black(x, s);
            CHECK(b >= intrinsic - 1e-15);
            CHECK(b <= std::exp(0.5 * x) + 1e-15);
        }
    }
}

TEST(normalised_black, keeps_relative_accuracy_in_the_deep_wing) {
    // The whole point of the Mills form. Naive e^{x/2}N(d1) - e^{-x/2}N(d2)
    // returns exactly 0.0 for several of these.
    struct Case { Real x, s; };
    for (Case c : {Case{-6.0, 0.3}, Case{-8.0, 0.5}, Case{-10.0, 0.8},
                   Case{-4.0, 0.15}, Case{-15.0, 1.0}}) {
        const Real b = normalised_black(c.x, c.s);
        CHECK(b > 0.0);
        CHECK(std::isfinite(b));
        // Leading asymptotic: b ~ nu * s^3 / x^2.
        const Real lead = normalised_vega(c.x, c.s) * std::pow(c.s, 3) / (c.x * c.x);
        CHECK_CLOSE(b / lead, 1.0, 0.35);
    }
    // Past the point where nu underflows the true value is below the smallest
    // representable double, and zero is the correct answer -- not NaN, and not
    // a negative number from a cancelled difference.
    for (Real s : {0.05, 0.1}) {
        const Real b = normalised_black(-2.0, s);   // 40 standard deviations
        CHECK(b >= 0.0);
        CHECK(!std::isnan(b));
    }
}

// ---------------------------------------------------------------------------
// Black-76 / Black-Scholes
// ---------------------------------------------------------------------------

TEST(black, matches_50_digit_reference_prices) {
    for (const auto& c : kBSReference) {
        CHECK_CLOSE(bs_price(c.S, c.K, c.T, c.r, c.q, c.sigma, OptionType::Call) / c.call,
                    1.0, 5e-14);
        CHECK_CLOSE(bs_price(c.S, c.K, c.T, c.r, c.q, c.sigma, OptionType::Put) / c.put,
                    1.0, 5e-14);
    }
}

TEST(black, put_call_parity_holds_on_the_grid) {
    const Real df = 0.97;
    for (const auto& g : moneyness_grid()) {
        const Real c = black76(g.F, g.K, g.T, g.sigma, df, OptionType::Call);
        const Real p = black76(g.F, g.K, g.T, g.sigma, df, OptionType::Put);
        CHECK_ABS(c - p, df * (g.F - g.K), 1e-11 * g.F);
    }
}

TEST(black, prices_stay_inside_no_arbitrage_bounds) {
    for (const auto& g : moneyness_grid()) {
        for (auto type : {OptionType::Call, OptionType::Put}) {
            Real lo, hi;
            black_price_bounds(g.F, g.K, type, lo, hi);
            const Real v = black76_undiscounted(g.F, g.K, g.T, g.sigma, type);
            CHECK(v >= lo - 1e-12);
            CHECK(v <= hi + 1e-12);
        }
    }
}

TEST(black, price_is_monotone_in_volatility) {
    for (const auto& g : moneyness_grid()) {
        const Real a = black76_undiscounted(g.F, g.K, g.T, g.sigma, OptionType::Call);
        const Real b = black76_undiscounted(g.F, g.K, g.T, g.sigma * 1.01, OptionType::Call);
        CHECK(b >= a);
    }
}

TEST(black, call_is_convex_and_decreasing_in_strike) {
    const Real F = 100.0, T = 0.75, sigma = 0.25;
    for (Real K = 40.0; K <= 200.0; K += 2.5) {
        const Real dK = 0.5;
        const Real cm = black76_undiscounted(F, K - dK, T, sigma, OptionType::Call);
        const Real c0 = black76_undiscounted(F, K, T, sigma, OptionType::Call);
        const Real cp = black76_undiscounted(F, K + dK, T, sigma, OptionType::Call);
        CHECK(c0 <= cm);                       // decreasing
        CHECK(cm - 2 * c0 + cp >= -1e-12);     // butterfly / density non-negative
    }
}

TEST(black, greeks_match_central_differences) {
    const Real S = 100.0, K = 105.0, T = 0.7, r = 0.03, q = 0.012, sig = 0.28;
    for (auto type : {OptionType::Call, OptionType::Put}) {
        const Greeks g = bs_greeks(S, K, T, r, q, sig, type);
        auto P = [&](Real s, Real k, Real t, Real rr, Real qq, Real v) {
            return bs_price(s, k, t, rr, qq, v, type);
        };
        const Real hS = 1e-4 * S, hv = 1e-5, hr = 1e-6, hT = 1e-6, hK = 1e-4 * K;

        CHECK_CLOSE((P(S + hS, K, T, r, q, sig) - P(S - hS, K, T, r, q, sig)) / (2 * hS),
                    g.delta, 1e-6);
        CHECK_CLOSE((P(S + hS, K, T, r, q, sig) - 2 * P(S, K, T, r, q, sig)
                     + P(S - hS, K, T, r, q, sig)) / (hS * hS), g.gamma, 1e-5);
        CHECK_CLOSE((P(S, K, T, r, q, sig + hv) - P(S, K, T, r, q, sig - hv)) / (2 * hv),
                    g.vega, 1e-7);
        CHECK_CLOSE((P(S, K, T, r + hr, q, sig) - P(S, K, T, r - hr, q, sig)) / (2 * hr),
                    g.rho, 1e-6);
        // theta is d/dt, and t runs the opposite way to T.
        CHECK_CLOSE(-(P(S, K, T + hT, r, q, sig) - P(S, K, T - hT, r, q, sig)) / (2 * hT),
                    g.theta, 1e-6);
        CHECK_CLOSE((P(S, K + hK, T, r, q, sig) - P(S, K - hK, T, r, q, sig)) / (2 * hK),
                    g.dual_delta, 1e-6);
        CHECK_CLOSE((P(S, K + hK, T, r, q, sig) - 2 * P(S, K, T, r, q, sig)
                     + P(S, K - hK, T, r, q, sig)) / (hK * hK), g.dual_gamma, 1e-5);
    }
}

TEST(black, cross_greeks_match_mixed_differences) {
    const Real S = 100.0, K = 95.0, T = 1.3, r = 0.02, q = 0.0, sig = 0.22;
    const Greeks g = bs_greeks(S, K, T, r, q, sig, OptionType::Call);
    const Real hS = 1e-3 * S, hv = 1e-4;
    auto P = [&](Real s, Real v) { return bs_price(s, K, T, r, q, v, OptionType::Call); };
    const Real vanna_fd = (P(S + hS, sig + hv) - P(S + hS, sig - hv)
                           - P(S - hS, sig + hv) + P(S - hS, sig - hv)) / (4 * hS * hv);
    CHECK_CLOSE(vanna_fd, g.vanna, 1e-5);
    const Real volga_fd = (P(S, sig + hv) - 2 * P(S, sig) + P(S, sig - hv)) / (hv * hv);
    CHECK_CLOSE(volga_fd, g.volga, 1e-5);
}

TEST(black, call_and_put_deltas_differ_by_the_carry_factor) {
    const Real S = 100.0, T = 1.4, r = 0.035, q = 0.02, sig = 0.3;
    for (Real K = 50.0; K <= 180.0; K += 5.0) {
        const Real dc = bs_greeks(S, K, T, r, q, sig, OptionType::Call).delta;
        const Real dp = bs_greeks(S, K, T, r, q, sig, OptionType::Put).delta;
        CHECK_ABS(dc - dp, std::exp(-q * T), 1e-13);
    }
}

TEST(black, rejects_invalid_inputs) {
    CHECK_THROWS(bs_price(-1.0, 100.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call));
    CHECK_THROWS(black76(100.0, -1.0, 1.0, 0.2, 1.0, OptionType::Call));
    CHECK_THROWS(black76(100.0, 100.0, -1.0, 0.2, 1.0, OptionType::Call));
    CHECK_THROWS(black76(100.0, 100.0, 1.0, -0.2, 1.0, OptionType::Call));
    CHECK_THROWS(bs_greeks(100.0, 100.0, 0.0, 0.0, 0.0, 0.2, OptionType::Call));
}
