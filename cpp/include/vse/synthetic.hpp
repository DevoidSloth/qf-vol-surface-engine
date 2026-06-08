// vse/synthetic.hpp — a deterministic, realistic option chain with a known
// answer.
//
// Read this before believing any number produced from it.
//
// Every quote here is MANUFACTURED. It is not market data and no result derived
// from it is a statement about any real market. What it is good for is the thing
// real data is bad for: it has a known ground truth. When the calibrator returns
// an SSVI surface, the residual against the parameters that generated the chain
// separates "the fitter works" from "the fitter found something plausible", and
// no amount of real data can make that separation.
//
// The generator is built to be hard in the ways real chains are hard, because a
// pipeline validated on clean data is validated on nothing:
//
//   * Strike ladders are irregular -- 5 point near the money, 10, then 25 in the
//     wings, as SPX actually lists -- so the fitter never sees an even grid.
//   * Prices are rounded to the exchange tick before being quoted. This is not
//     cosmetic: rounding a 0.05-wide option to 0.05 makes its implied vol jump by
//     whole points, and it is the dominant error source in the wings of a real
//     chain. A pipeline that only works on unrounded prices does not work.
//   * Bid-ask spreads widen with moneyness and shorten with expiry, and the mid
//     is what gets fitted -- so the weighting scheme is exercised rather than
//     bypassed.
//   * Volume and open interest decay away from the money, so liquidity filters
//     have something to filter.
//   * The forward carries a borrow spread over the quoted rate, so anything that
//     assumes F = S e^{rT} gets the wrong answer and the put-call parity
//     regression has something real to recover.
//
// Deterministic given a seed: the same seed produces the same chain on every
// platform, because the generator is xoshiro256++ and the normals come from
// inversion rather than from a standard-library distribution.
#pragma once

#include "vse/black.hpp"
#include "vse/chain.hpp"
#include "vse/common.hpp"
#include "vse/rng.hpp"
#include "vse/svi.hpp"

#include <algorithm>
#include <vector>

namespace vse {

struct SyntheticChainConfig {
    Real spot = 4275.0;
    Real quoted_rate = 0.045;      ///< what a screen would show
    Real borrow_spread = 0.0075;   ///< what the option market actually prices
    Real dividend_yield = 0.0155;
    Real tick = 0.05;              ///< SPX quotes in nickels
    Real atm_spread_fraction = 0.004;   ///< half-spread as a fraction of price, ATM
    Real wing_spread_multiplier = 12.0; ///< how much wider five sd out
    std::uint64_t seed = 20260615;
    bool round_to_tick = true;
    bool add_quote_noise = true;
    Real quote_noise_vol_points = 0.0015;  ///< idiosyncratic per-quote vol error
};

/// The ground truth a synthetic chain was generated from.
struct SyntheticTruth {
    ESSVISurface surface;
    std::vector<Real> expiries;
    std::vector<Real> forwards;
    std::vector<Real> discounts;
};

struct SyntheticExpiry {
    Real expiry = 0.0;
    Real true_forward = 0.0;
    Real true_discount = 1.0;
    std::vector<RawQuote> quotes;   ///< calls and puts at every strike
};

struct SyntheticChain {
    SyntheticTruth truth;
    std::vector<SyntheticExpiry> expiries;
};

namespace detail {

/// SPX-style strike ladder: dense near the money, sparse in the wings.
inline std::vector<Real> spx_strike_ladder(Real forward, Real sd) {
    std::vector<Real> strikes;
    const Real lo = forward * std::exp(-5.0 * sd);
    const Real hi = forward * std::exp(3.5 * sd);   // call wing is shorter, as listed

    auto add_range = [&](Real from, Real to, Real step) {
        const Real start = std::ceil(from / step) * step;
        for (Real k = start; k <= to; k += step) strikes.push_back(k);
    };

    const Real near_lo = forward * std::exp(-1.0 * sd);
    const Real near_hi = forward * std::exp(1.0 * sd);
    const Real mid_lo  = forward * std::exp(-2.5 * sd);
    const Real mid_hi  = forward * std::exp(2.0 * sd);

    add_range(lo, mid_lo, 25.0);
    add_range(mid_lo, near_lo, 10.0);
    add_range(near_lo, near_hi, 5.0);
    add_range(near_hi, mid_hi, 10.0);
    add_range(mid_hi, hi, 25.0);

    std::sort(strikes.begin(), strikes.end());
    strikes.erase(std::unique(strikes.begin(), strikes.end()), strikes.end());
    return strikes;
}

}  // namespace detail

/// Generate a chain from an eSSVI ground truth.
///
/// The default surface is shaped like an equity index: ATM term structure in
/// contango, skew steep at the front and flattening with maturity, curvature
/// decaying. Those are the features that make a surface hard to fit, and a
/// generator that omits them tests nothing.
inline SyntheticChain generate_synthetic_chain(
    const std::vector<Real>& expiries = {7.0 / 365, 30.0 / 365, 60.0 / 365, 91.0 / 365,
                                         182.0 / 365, 365.0 / 365, 730.0 / 365},
    const SyntheticChainConfig& cfg = {}) {
    require(!expiries.empty(), "generate_synthetic_chain: no expiries");

    SyntheticChain chain;
    Xoshiro256pp rng(cfg.seed);

    // Ground-truth eSSVI: ATM variance in contango, skew flattening with T.
    ESSVISurface truth;
    truth.expiries = expiries;
    for (Real T : expiries) {
        const Real atm_vol = 0.132 + 0.045 * std::sqrt(T) - 0.010 * T;
        const Real theta = atm_vol * atm_vol * T;
        truth.theta.push_back(theta);
        // rho steepens toward the front, psi grows with maturity as required by
        // the calendar conditions.
        truth.rho.push_back(-0.82 + 0.28 * (1.0 - std::exp(-1.4 * T)));
        truth.psi.push_back(0.55 * std::pow(theta, 0.42));
    }
    // Enforce the eSSVI calendar conditions on the truth itself, so that a
    // failure downstream is a fitter failure and never a bad target.
    for (std::size_t i = 1; i < truth.size(); ++i) {
        truth.psi[i] = std::fmax(truth.psi[i], truth.psi[i - 1] * (1.0 + 1e-6));
        const Real headroom = truth.psi[i] - truth.psi[i - 1];
        const Real prev = truth.rho[i - 1] * truth.psi[i - 1];
        truth.rho[i] = clampv(truth.rho[i], (prev - headroom) / truth.psi[i],
                              (prev + headroom) / truth.psi[i]);
    }
    chain.truth.surface = truth;
    chain.truth.expiries = expiries;

    for (std::size_t ei = 0; ei < expiries.size(); ++ei) {
        const Real T = expiries[ei];
        const Real discount = std::exp(-(cfg.quoted_rate) * T);
        // The forward the option market prices: spot, less dividends, plus the
        // borrow spread. Anything that reconstructs it from the quoted rate
        // alone will be wrong by borrow_spread * T, which at 75 bp is tens of
        // basis points of forward and a visibly tilted smile.
        const Real forward = cfg.spot * std::exp(
            (cfg.quoted_rate + cfg.borrow_spread - cfg.dividend_yield) * T);

        const SSVISlice slice = truth.slice_at_index(ei);
        const Real atm_vol = slice.implied_vol(0.0, T);
        const Real sd = atm_vol * std::sqrt(T);

        SyntheticExpiry se;
        se.expiry = T;
        se.true_forward = forward;
        se.true_discount = discount;

        for (Real K : detail::spx_strike_ladder(forward, sd)) {
            const Real k = std::log(K / forward);
            Real vol = slice.implied_vol(k, T);
            if (cfg.add_quote_noise) {
                vol += cfg.quote_noise_vol_points * rng.normal();
            }
            if (!(vol > 0.005)) continue;

            const Real z = std::fabs(k) / std::fmax(sd, 1e-8);
            const Real spread_scale =
                1.0 + (cfg.wing_spread_multiplier - 1.0) * std::fmin(z / 5.0, 1.0);

            for (auto type : {OptionType::Call, OptionType::Put}) {
                const Real mid = black76(forward, K, T, vol, discount, type);
                if (mid < cfg.tick) continue;   // not listed once it is worth less than a tick

                Real half_spread = std::fmax(cfg.atm_spread_fraction * mid * spread_scale,
                                             cfg.tick);
                // Short-dated options quote wider in relative terms.
                half_spread *= 1.0 + 0.35 / (1.0 + 12.0 * T);

                Real bid = mid - half_spread;
                Real ask = mid + half_spread;
                if (cfg.round_to_tick) {
                    bid = std::floor(bid / cfg.tick) * cfg.tick;
                    ask = std::ceil(ask / cfg.tick) * cfg.tick;
                }
                if (!(bid > 0.0)) bid = 0.0;

                RawQuote q;
                q.strike = K;
                q.bid = bid;
                q.ask = ask;
                q.type = type;
                // Liquidity: heaviest at the money, and puts more traded than
                // calls in the downside wing, as in a real index chain.
                const Real base = 4000.0 * std::exp(-0.5 * z * z);
                const Real skewed = (type == OptionType::Put && k < 0.0) ? 1.8 : 1.0;
                q.volume = std::floor(base * skewed * rng.uniform(0.4, 1.6));
                q.open_interest = std::floor(12.0 * base * skewed * rng.uniform(0.5, 2.0));
                se.quotes.push_back(q);
            }
        }

        chain.truth.forwards.push_back(forward);
        chain.truth.discounts.push_back(discount);
        chain.expiries.push_back(std::move(se));
    }
    return chain;
}

/// Split an expiry's quotes into calls and puts, which is the shape the
/// put-call parity regression wants.
inline void split_by_type(const std::vector<RawQuote>& quotes,
                          std::vector<RawQuote>& calls, std::vector<RawQuote>& puts) {
    calls.clear();
    puts.clear();
    for (const auto& q : quotes) {
        (q.type == OptionType::Call ? calls : puts).push_back(q);
    }
}

}  // namespace vse
