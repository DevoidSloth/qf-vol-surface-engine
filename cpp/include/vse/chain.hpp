// vse/chain.hpp — turning a raw option chain into something a surface can be
// fitted to.
//
// The single most consequential step in surface work happens here, and it is not
// the fitting. It is deciding what F and P(0,T) are.
//
// A listed equity-index chain does not tell you the forward. The spot is
// observable and the risk-free rate is quotable, but the *implied* forward --
// the one the option market is actually pricing against, net of borrow, funding
// spread, and whatever the dividend forecast is this week -- is not either of
// those. Using the spot compounded at a screen rate produces a forward that is
// wrong by tens of basis points, and a forward that is wrong by tens of basis
// points produces a smile with a visible spurious tilt. Everything downstream
// then fits that tilt: the SVI rho absorbs it, the calibrated Heston rho
// absorbs it, and the surface is quietly wrong in a way no fit statistic shows.
//
// Put-call parity gives it up for free. For every strike with a two-sided
// market on both wings,
//
//     C(K) - P(K) = P(0,T) (F - K),
//
// which is affine in K with slope -P(0,T) and intercept P(0,T) F. One regression
// per expiry recovers both, and the residuals of that regression are themselves
// a data-quality signal: a strike whose parity residual is several ticks wide
// has a stale quote on one leg.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/implied_vol.hpp"
#include "vse/linalg.hpp"

#include <algorithm>
#include <vector>

namespace vse {

/// One side of one strike, as quoted.
struct RawQuote {
    Real strike = 0.0;
    Real bid = 0.0;
    Real ask = 0.0;
    Real volume = 0.0;
    Real open_interest = 0.0;
    OptionType type = OptionType::Call;

    Real mid() const { return 0.5 * (bid + ask); }
    Real spread() const { return ask - bid; }
    bool two_sided() const { return bid > 0.0 && ask > bid; }
};

struct ForwardFit {
    Real forward = 0.0;
    Real discount = 1.0;
    Real implied_rate = 0.0;     ///< -ln(P(0,T))/T
    int  pairs_used = 0;
    Real residual_rms = 0.0;     ///< in price units
    Real worst_residual = 0.0;
    bool ok = false;
};

/// Recover F and P(0,T) from put-call parity by regressing C - K on K.
///
/// Restricted to strikes near the money by `moneyness_window`, expressed as a
/// fraction of the anchor. Two reasons, and the second is the one that matters:
///
///   * Far from the money one leg is deep in the money, so its quoted spread is
///     wide in absolute terms and dominated by the intrinsic value it carries.
///     The parity difference there is a large number minus a large number.
///   * Deep wings are where stale quotes live. A single strike that last traded
///     an hour ago will drag the regression, and because the relationship is
///     affine, a dragged slope is a wrong *discount factor*, which then shifts
///     every implied vol on the slice in the same direction -- a bias, not noise.
///
/// The fit is weighted by the inverse of the combined spread, so a tight
/// two-sided market counts for more than a wide one.
inline ForwardFit implied_forward_from_parity(const std::vector<RawQuote>& calls,
                                              const std::vector<RawQuote>& puts,
                                              Real expiry, Real anchor,
                                              Real moneyness_window = 0.10) {
    require(expiry > 0.0, "implied_forward_from_parity: expiry must be positive");
    require(anchor > 0.0, "implied_forward_from_parity: need a positive anchor "
                          "(spot or a previous forward) to select strikes");

    struct Pair { Real k, diff, weight; };
    std::vector<Pair> pairs;

    for (const auto& c : calls) {
        if (!c.two_sided()) continue;
        const auto p = std::find_if(puts.begin(), puts.end(), [&](const RawQuote& q) {
            return q.strike == c.strike && q.two_sided();
        });
        if (p == puts.end()) continue;
        if (std::fabs(c.strike / anchor - 1.0) > moneyness_window) continue;

        const Real combined_spread = c.spread() + p->spread();
        if (!(combined_spread > 0.0)) continue;
        pairs.push_back({c.strike, c.mid() - p->mid(), 1.0 / combined_spread});
    }

    ForwardFit fit;
    fit.pairs_used = int(pairs.size());
    if (pairs.size() < 3) return fit;   // ok stays false; caller decides

    // Weighted least squares of diff = alpha + beta (K - Kbar), with
    // beta = -P(0,T) and alpha = P(0,T) (F - Kbar).
    //
    // The centring is not optional. Strikes on an index are ~4000 with a spread
    // of a few hundred, so an uncentred design has a column of ones beside a
    // column with mean 4000 and the normal-equations matrix has a condition
    // number around 1e9. Centring makes the two columns orthogonal and drops it
    // to O(1). Without it, this regression returned a discount factor wrong in
    // the third decimal on input that satisfied parity to 3e-13.
    Real kbar = 0.0, wsum = 0.0;
    for (const auto& pr : pairs) { kbar += pr.weight * pr.k; wsum += pr.weight; }
    kbar /= wsum;

    Matrix design(pairs.size(), 2);
    std::vector<Real> y(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const Real sw = std::sqrt(pairs[i].weight);
        design(i, 0) = sw;
        design(i, 1) = sw * (pairs[i].k - kbar);
        y[i] = sw * pairs[i].diff;
    }
    std::vector<Real> beta;
    if (!least_squares(design, y, beta)) return fit;

    const Real discount = -beta[1];
    if (!(discount > 0.0 && discount <= 1.5)) return fit;   // nonsensical slope

    fit.discount = discount;
    fit.forward = beta[0] / discount + kbar;
    fit.implied_rate = -std::log(discount) / expiry;

    Real ss = 0.0;
    for (const auto& pr : pairs) {
        const Real resid = pr.diff - (beta[0] + beta[1] * (pr.k - kbar));
        ss += resid * resid;
        fit.worst_residual = std::fmax(fit.worst_residual, std::fabs(resid));
    }
    fit.residual_rms = std::sqrt(ss / Real(pairs.size()));
    fit.ok = fit.forward > 0.0;
    return fit;
}

/// A quote that survived filtering, reduced to what a fitter needs.
struct SurfacePoint {
    Real log_moneyness = 0.0;   ///< k = ln(K/F)
    Real strike = 0.0;
    Real implied_vol = 0.0;
    Real total_variance = 0.0;
    Real vega = 0.0;            ///< undiscounted, per unit of vol
    Real spread_vol = 0.0;      ///< half the bid-ask expressed in vol points
    Real weight = 1.0;
    OptionType type = OptionType::Call;
};

struct FilterConfig {
    Real min_price = 0.05;             ///< drop quotes worth less than this
    Real max_relative_spread = 1.0;    ///< drop when (ask-bid)/mid exceeds this
    Real max_abs_log_moneyness = 1.5;  ///< keep |ln(K/F)| within this
    Real min_volume = 0.0;
    Real min_open_interest = 0.0;
    bool require_two_sided = true;
};

struct SliceBuildReport {
    int input_quotes = 0;
    int dropped_in_the_money = 0;   ///< the other half of each strike pair
    int dropped_one_sided = 0;
    int dropped_cheap = 0;
    int dropped_wide = 0;
    int dropped_moneyness = 0;
    int dropped_liquidity = 0;
    int dropped_arbitrage = 0;   ///< price outside the no-arbitrage bounds
    int kept = 0;

    /// Every input quote must land in exactly one bucket.
    ///
    /// This is not decoration. Without it the in-the-money side -- roughly half
    /// of every board -- disappeared from the report entirely, so a slice that
    /// kept 56 of 123 quotes accounted for one of the 67 it lost. Anyone using
    /// the report to work out why a board came out thin would have concluded
    /// the filters were eating quotes silently, which is exactly the failure it
    /// exists to rule out.
    int dropped_total() const {
        return dropped_in_the_money + dropped_one_sided + dropped_cheap + dropped_wide
             + dropped_moneyness + dropped_liquidity + dropped_arbitrage;
    }
    bool balances() const { return kept + dropped_total() == input_quotes; }
};

/// Build the fittable points of one expiry.
///
/// Always takes the out-of-the-money side of each strike: calls above the
/// forward, puts below. Both members of a pair carry the same information, but
/// the in-the-money one carries it behind an intrinsic value that can be orders
/// of magnitude larger than the time value, and inverting it throws away the
/// digits that distinguish one volatility from another.
inline std::vector<SurfacePoint> build_slice(const std::vector<RawQuote>& quotes,
                                             Real forward, Real expiry, Real discount,
                                             const FilterConfig& cfg,
                                             SliceBuildReport* report = nullptr) {
    require(forward > 0.0 && expiry > 0.0 && discount > 0.0,
            "build_slice: forward, expiry and discount must be positive");

    SliceBuildReport rep;
    std::vector<SurfacePoint> out;

    for (const auto& q : quotes) {
        ++rep.input_quotes;
        // Only the out-of-the-money side.
        const bool otm = (q.type == OptionType::Call) ? (q.strike >= forward)
                                                      : (q.strike < forward);
        if (!otm) { ++rep.dropped_in_the_money; continue; }

        if (cfg.require_two_sided && !q.two_sided()) { ++rep.dropped_one_sided; continue; }
        const Real mid = q.mid();
        if (mid < cfg.min_price) { ++rep.dropped_cheap; continue; }
        if (q.spread() / mid > cfg.max_relative_spread) { ++rep.dropped_wide; continue; }

        const Real k = std::log(q.strike / forward);
        if (std::fabs(k) > cfg.max_abs_log_moneyness) { ++rep.dropped_moneyness; continue; }
        if (q.volume < cfg.min_volume || q.open_interest < cfg.min_open_interest) {
            ++rep.dropped_liquidity;
            continue;
        }

        // No-arbitrage bounds before inversion, so that a bad quote is reported
        // as a bad quote rather than as a solver failure.
        Real lo, hi;
        black_price_bounds(forward, q.strike, q.type, lo, hi);
        const Real undiscounted = mid / discount;
        if (!(undiscounted > lo && undiscounted < hi)) { ++rep.dropped_arbitrage; continue; }

        const auto iv = implied_volatility_ex(mid, forward, q.strike, expiry, discount, q.type);
        if (!iv.converged || !(iv.sigma > 0.0)) { ++rep.dropped_arbitrage; continue; }

        SurfacePoint p;
        p.log_moneyness = k;
        p.strike = q.strike;
        p.implied_vol = iv.sigma;
        p.total_variance = iv.sigma * iv.sigma * expiry;
        p.type = q.type;

        // Undiscounted vega, from the normalised form: dV/dsigma = sqrt(FK) nu sqrt(T).
        const Real s = iv.sigma * std::sqrt(expiry);
        p.vega = std::sqrt(forward * q.strike) * normalised_vega(k, s) * std::sqrt(expiry);

        // Half-spread converted to vol points. This is the quantity that says how
        // much a quote is actually worth as an observation, and it is why deep
        // wings must not be fitted unweighted: out there a one-tick price move is
        // several vol points, so an unweighted fit spends its degrees of freedom
        // chasing quotes that carry almost no information.
        p.spread_vol = (p.vega > 0.0) ? (0.5 * q.spread() / p.vega) : DBL_HUGE;
        p.weight = (p.spread_vol > 0.0 && std::isfinite(p.spread_vol))
                       ? 1.0 / p.spread_vol
                       : 0.0;
        if (!(p.weight > 0.0)) { ++rep.dropped_wide; continue; }

        out.push_back(p);
        ++rep.kept;
    }

    std::sort(out.begin(), out.end(), [](const SurfacePoint& a, const SurfacePoint& b) {
        return a.log_moneyness < b.log_moneyness;
    });
    if (report) *report = rep;
    return out;
}

/// ATM total variance by interpolation of the fitted points at k = 0.
///
/// Linear in k between the two straddling observations. The ATM point anchors
/// theta for SSVI, so it is worth doing on total variance rather than on vol.
inline Real atm_total_variance(const std::vector<SurfacePoint>& slice) {
    require(slice.size() >= 2, "atm_total_variance: need at least two points");
    for (std::size_t i = 1; i < slice.size(); ++i) {
        if (slice[i].log_moneyness >= 0.0 && slice[i - 1].log_moneyness <= 0.0) {
            const Real k0 = slice[i - 1].log_moneyness, k1 = slice[i].log_moneyness;
            if (k1 == k0) return slice[i].total_variance;
            const Real u = (0.0 - k0) / (k1 - k0);
            return slice[i - 1].total_variance + u * (slice[i].total_variance - slice[i - 1].total_variance);
        }
    }
    // Entirely on one side of the money: take the nearest point rather than
    // extrapolating a quantity the caller will treat as observed.
    const auto nearest = std::min_element(
        slice.begin(), slice.end(), [](const SurfacePoint& a, const SurfacePoint& b) {
            return std::fabs(a.log_moneyness) < std::fabs(b.log_moneyness);
        });
    return nearest->total_variance;
}

}  // namespace vse
