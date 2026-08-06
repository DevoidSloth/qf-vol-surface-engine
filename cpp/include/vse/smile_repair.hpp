// vse/smile_repair.hpp — project an arbitrageable smile onto the nearest one
// that is a probability distribution.
//
// WHY THIS EXISTS, AND WHY THE TWO OBVIOUS ANSWERS DO NOT WORK.
//
// Hagan's SABR expansion produces implied volatilities with no constraint that
// they correspond to any distribution, and for beta < 1 at long maturities they
// do not: the implied density is negative below a boundary strike. sabr.hpp
// documents this and offers two standard responses. Measured on ordinary
// long-dated rates parameters (F = 3%, beta = 0.5, nu = 0.45, rho = -0.2,
// T = 10), scanning 1500 strikes from 0.02F to 3F:
//
//     beta   expansion    boundary K/F   violations   min density
//     0.00   lognormal          0.4077          195     -4.48e+02
//     0.00   normal             0.3480          165     -5.89e+02
//     0.50   lognormal          0.2804          131     -8.71e+01
//     0.50   normal             0.2586          120     -3.90e+02
//     1.00   lognormal          0.0717           26     -1.30e+01
//
// Neither response is a fix.
//
//   * THE SHIFT is a change of variable. A shifted SABR on a shifted forward
//     reproduces the unshifted smile exactly -- there is a test asserting it to
//     1e-14 -- so it relabels the arbitrage rather than removing it. It earns
//     its place by making negative forwards priceable at all, which is a
//     different problem.
//   * NORMAL SABR is not better behaved here, and this file exists partly
//     because I had written in sabr.hpp that it was. At beta = 0.5 the normal
//     expansion moves the boundary from 0.280 to 0.259 -- a rounding error on
//     the claim that it "pushes the arbitrage out to strikes that are not
//     quoted" -- while making the worst density four times more negative. At
//     beta = 0, which is what "normal SABR" actually means, it is far worse
//     than beta = 0.5 in both. The claim was plausible, widely repeated, and
//     contradicted by the project's own benchmark.
//
// What does work is to stop arguing about the vol expansion and repair the
// PRICES, which is the only place the constraint actually lives.
//
// THE CONSTRUCTION. A smile is free of butterfly arbitrage exactly when the
// undiscounted call price is convex in strike. Convexity is a closed convex
// constraint, so there is a unique nearest point in it, and finding that point
// is an isotonic regression: C convex in K means its slopes are non-decreasing,
// and the L2 projection onto non-decreasing sequences is pool-adjacent-
// violators, which is linear time and exact -- no optimiser, no tolerance, no
// convergence to check.
//
// The projection is done on the OUT-OF-THE-MONEY prices, put below the forward
// and call above, and on the two sides separately. Not for tidiness:
//
//   * Deep in the money a call is worth F - K to fifteen digits and its time
//     value is below the rounding. Projecting that curve would be projecting
//     noise; the OTM member of the pair carries the same information with none
//     of the cancellation. (The density scan in sabr.hpp had to learn this
//     too -- differencing calls reported 68 violations on a clean smile.)
//   * Puts are convex increasing and calls are convex decreasing, so each side
//     is a separate isotonic problem. They meet at the forward in a downward
//     kink, which is a property of the OTM curve and not a violation: it is
//     exactly the F - K term that parity says must be there.
//
// The slopes are also clamped -- calls to [-1, 0], puts to [0, 1] -- which
// enforces monotonicity and the correct asymptotics, and costs nothing because
// the true curve satisfies them.
//
// WHAT IT COSTS. The repaired smile is not the input smile, and the report says
// how far it moved. Measured on the same parameters, worst absolute change in
// implied volatility over the grid:
//
//     beta 0.00    196 violations -> 0     8.96 vol points
//     beta 0.50    131 violations -> 0     3.48 vol points
//     beta 1.00     26 violations -> 0     0.15 vol points
//
// The displacement is concentrated in the wing where the input was not a price
// at all, and is negligible where the smile was already a distribution -- which
// is the right shape for the cost to have, and also says plainly that on a
// badly arbitraged smile the repair is not a cosmetic adjustment. Quoting the
// repaired vols without that number would hide it.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/implied_vol.hpp"
#include "vse/sabr.hpp"

#include <algorithm>
#include <vector>

namespace vse {

/// L2 projection of `y` onto the non-decreasing sequences (pool adjacent
/// violators), weights uniform.
///
/// Exact and linear time. The invariant is that `level` holds the running
/// block means and `count` their sizes; whenever a new value would break
/// monotonicity the offending blocks are merged, which is the only operation
/// the algorithm has.
inline void isotonic_increasing(std::vector<Real>& y) {
    const std::size_t n = y.size();
    if (n < 2) return;
    std::vector<Real> level(n);
    std::vector<std::size_t> count(n);
    std::size_t blocks = 0;
    for (std::size_t i = 0; i < n; ++i) {
        level[blocks] = y[i];
        count[blocks] = 1;
        ++blocks;
        while (blocks > 1 && level[blocks - 2] > level[blocks - 1]) {
            const Real total = level[blocks - 2] * Real(count[blocks - 2])
                             + level[blocks - 1] * Real(count[blocks - 1]);
            const std::size_t merged = count[blocks - 2] + count[blocks - 1];
            level[blocks - 2] = total / Real(merged);
            count[blocks - 2] = merged;
            --blocks;
        }
    }
    std::size_t at = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
        for (std::size_t j = 0; j < count[b]; ++j) y[at++] = level[b];
    }
}

struct SmileRepairReport {
    Real min_density_before = 0.0;
    Real min_density_after = 0.0;
    int  violations_before = 0;
    int  violations_after = 0;
    Real max_vol_change = 0.0;      ///< absolute vol, over the whole grid
    Real max_vol_change_strike = 0.0;
    Real max_price_change = 0.0;    ///< as a fraction of the forward
    /// Noise floor of the density estimate, and the threshold the violation
    /// counts use.
    ///
    /// A second difference divides by h^2, so a price carrying relative
    /// rounding of eps comes back as a density uncertain by eps*F/h^2. On the
    /// default grid that is around 1e-9, which is exactly the size of the
    /// residual negatives the projection leaves behind -- counting those as
    /// arbitrage would report 67 violations on a curve that is convex to the
    /// last bit it can represent. The threshold is derived from the grid rather
    /// than tuned to make a number look good.
    Real density_tolerance = 0.0;

    /// Total probability mass on the grid, before and after.
    ///
    /// Computed from the endpoint slopes of the call curve, not by summing the
    /// density: the integral of C'' over [a, b] IS C'(b) - C'(a), so this is
    /// exact rather than a quadrature, and it does not have to skip the node
    /// whose stencil straddles the forward.
    ///
    /// It is reported because THE PROJECTION DOES NOT RESTORE IT. Convexity
    /// says nothing about level, so a smile whose wings have the wrong
    /// asymptotics comes back convex and still short of unit mass -- and
    /// Hagan's does. On the long-dated parameters this file is built around,
    /// the put at K = 0.001F is worth 15% of its strike, because the expansion
    /// sends the volatility to 117% as the strike goes to zero. That is mass
    /// sitting at or below zero, a second and independent failure of the
    /// expansion, and no local repair recovers it. A smile with sound wings
    /// (SVI, which satisfies Lee's bounds by construction) comes back at
    /// 1.000000.
    Real mass_before = 0.0;
    Real mass_after = 0.0;

    int  points = 0;
    bool repaired = false;          ///< the projection changed something
};

/// A smile that is a probability distribution, plus what it cost to make it one.
struct RepairedSmile {
    std::vector<Real> strikes;
    std::vector<Real> otm_price;        ///< undiscounted, put below F, call above
    std::vector<Real> implied_vol;
    std::vector<Real> input_vol;
    std::vector<Real> density;          ///< second difference of the call curve
    Real forward = 0.0;
    Real expiry = 0.0;
    SmileRepairReport report;

    /// Linear interpolation in strike, flat outside the grid.
    Real vol_at(Real strike) const {
        if (strikes.empty()) return 0.0;
        if (strike <= strikes.front()) return implied_vol.front();
        if (strike >= strikes.back()) return implied_vol.back();
        const auto it = std::lower_bound(strikes.begin(), strikes.end(), strike);
        const std::size_t i = std::size_t(it - strikes.begin());
        const Real t = (strike - strikes[i - 1]) / (strikes[i] - strikes[i - 1]);
        return implied_vol[i - 1] + t * (implied_vol[i] - implied_vol[i - 1]);
    }
};

namespace detail {

/// Second difference of the CALL curve on a uniform strike grid.
///
/// Reconstructed from the OTM prices by parity, which is exact: below the
/// forward C = P + (F - K), and (F - K) is linear so it drops out of a second
/// difference everywhere except across the forward itself. The node whose
/// stencil straddles the forward therefore has to be skipped rather than
/// reported -- its second difference contains the parity kink, which is not a
/// density.
inline std::vector<Real> call_second_difference(const std::vector<Real>& strikes,
                                                const std::vector<Real>& otm, Real forward) {
    const std::size_t n = strikes.size();
    std::vector<Real> d2(n, 0.0);
    if (n < 3) return d2;
    const Real h = strikes[1] - strikes[0];
    auto call = [&](std::size_t i) {
        return strikes[i] < forward ? otm[i] + forward - strikes[i] : otm[i];
    };
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const bool straddles = (strikes[i - 1] < forward) != (strikes[i + 1] < forward);
        d2[i] = straddles ? DBL_HUGE
                          : (call(i + 1) - 2.0 * call(i) + call(i - 1)) / (h * h);
    }
    d2[0] = d2[n - 1] = DBL_HUGE;   // no stencil
    return d2;
}

/// Convexify one side by projecting its slopes, then re-integrating.
///
/// `anchor` is the index the rebuilt curve is pinned to, and it is the node
/// nearest the forward on each side. That choice is deliberate: it is where the
/// quotes are most reliable and where the price is largest, so pinning there
/// puts the displacement in the wings, which is both where the input was wrong
/// and where a wrong price matters least.
inline void convexify(std::vector<Real>& price, Real h, std::size_t anchor,
                      Real slope_lo, Real slope_hi) {
    const std::size_t n = price.size();
    if (n < 2) return;
    std::vector<Real> slope(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) slope[i] = (price[i + 1] - price[i]) / h;

    isotonic_increasing(slope);
    for (Real& s : slope) s = clampv(s, slope_lo, slope_hi);
    // Clamping can break monotonicity only by flattening, which preserves it,
    // so one pass is enough and the result is still convex.

    for (std::size_t i = anchor; i + 1 < n; ++i) price[i + 1] = price[i] + slope[i] * h;
    for (std::size_t i = anchor; i > 0; --i) price[i - 1] = price[i] - slope[i - 1] * h;
}

}  // namespace detail

/// Repair any smile, given as a function from strike to implied volatility.
///
/// The grid is uniform in strike because the constraint is convexity IN STRIKE,
/// and a projection performed on a log grid would be projecting a different
/// function. Uniform spacing also makes the isotonic weights uniform, which is
/// what keeps the projection the honest L2 nearest point rather than an
/// arbitrarily reweighted one.
template <class VolFn>
RepairedSmile repair_smile(VolFn&& vol_at, Real forward, Real expiry, Real k_min_ratio = 0.02,
                           Real k_max_ratio = 3.0, int n = 1501) {
    require(forward > 0.0 && expiry > 0.0, "repair_smile: forward and expiry must be positive");
    require(n >= 5, "repair_smile: need at least five grid points");

    RepairedSmile out;
    out.forward = forward;
    out.expiry = expiry;
    out.report.points = n;

    const std::size_t count = std::size_t(n);
    const Real lo = forward * k_min_ratio, hi = forward * k_max_ratio;
    const Real h = (hi - lo) / Real(n - 1);

    out.strikes.resize(count);
    out.otm_price.resize(count);
    out.input_vol.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const Real K = lo + Real(i) * h;
        const Real v = vol_at(K);
        out.strikes[i] = K;
        out.input_vol[i] = v;
        out.otm_price[i] = black76_undiscounted(forward, K, expiry,
                                                v, K >= forward ? OptionType::Call
                                                                : OptionType::Put);
    }

    const Real tolerance = 8.0 * DBL_EPS * forward / (h * h);
    out.report.density_tolerance = tolerance;

    auto endpoint_mass = [&](const std::vector<Real>& otm) {
        auto call = [&](std::size_t i) {
            return out.strikes[i] < forward ? otm[i] + forward - out.strikes[i] : otm[i];
        };
        const Real hi = (call(count - 1) - call(count - 2)) / h;
        const Real lo = (call(1) - call(0)) / h;
        return hi - lo;
    };
    out.report.mass_before = endpoint_mass(out.otm_price);

    const auto before = detail::call_second_difference(out.strikes, out.otm_price, forward);
    out.report.min_density_before = DBL_HUGE;
    for (std::size_t i = 0; i < count; ++i) {
        if (before[i] == DBL_HUGE) continue;
        out.report.min_density_before = std::fmin(out.report.min_density_before, before[i]);
        if (before[i] < -tolerance) ++out.report.violations_before;
    }

    // Split at the forward. `split` is the first index at or above it, so the
    // put side is [0, split) and the call side is [split, n).
    std::size_t split = 0;
    while (split < count && out.strikes[split] < forward) ++split;
    split = std::min(split, count - 1);

    const std::vector<Real> original = out.otm_price;

    if (split >= 2) {
        std::vector<Real> puts(out.otm_price.begin(),
                               out.otm_price.begin() + std::ptrdiff_t(split));
        // Puts: increasing and convex in strike, slope in [0, 1] undiscounted.
        detail::convexify(puts, h, puts.size() - 1, 0.0, 1.0);
        std::copy(puts.begin(), puts.end(), out.otm_price.begin());
    }
    if (count - split >= 2) {
        std::vector<Real> calls(out.otm_price.begin() + std::ptrdiff_t(split),
                                out.otm_price.end());
        // Calls: decreasing and convex, slope in [-1, 0].
        detail::convexify(calls, h, 0, -1.0, 0.0);
        std::copy(calls.begin(), calls.end(),
                  out.otm_price.begin() + std::ptrdiff_t(split));
    }

    out.report.mass_after = endpoint_mass(out.otm_price);
    out.density = detail::call_second_difference(out.strikes, out.otm_price, forward);
    out.report.min_density_after = DBL_HUGE;
    for (std::size_t i = 0; i < count; ++i) {
        if (out.density[i] == DBL_HUGE) continue;
        out.report.min_density_after = std::fmin(out.report.min_density_after, out.density[i]);
        if (out.density[i] < -tolerance) ++out.report.violations_after;
    }

    // Re-imply. A repaired price can sit on a no-arbitrage bound exactly -- the
    // projection is allowed to flatten a whole block of slopes to zero -- and
    // there is no finite volatility there, so those points keep the input vol
    // and are counted as unchanged rather than reported as a solver failure.
    out.implied_vol.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const Real K = out.strikes[i];
        const auto type = K >= forward ? OptionType::Call : OptionType::Put;
        const auto iv = implied_volatility_ex(out.otm_price[i], forward, K, expiry, 1.0, type);
        out.implied_vol[i] = (iv.converged && iv.sigma > 0.0) ? iv.sigma : out.input_vol[i];

        const Real dv = std::fabs(out.implied_vol[i] - out.input_vol[i]);
        if (dv > out.report.max_vol_change) {
            out.report.max_vol_change = dv;
            out.report.max_vol_change_strike = K;
        }
        out.report.max_price_change = std::fmax(
            out.report.max_price_change, std::fabs(out.otm_price[i] - original[i]) / forward);
    }
    out.report.repaired = out.report.violations_before > 0;
    return out;
}

/// Repair a SABR smile.
inline RepairedSmile repair_sabr(const SABRParams& p, Real forward, Real expiry,
                                 Real k_min_ratio = 0.02, Real k_max_ratio = 3.0,
                                 int n = 1501, bool normal_form = false) {
    return repair_smile(
        [&](Real K) {
            return normal_form ? sabr_normal_vol(p, forward, K, expiry)
                               : sabr_lognormal_vol(p, forward, K, expiry);
        },
        forward, expiry, k_min_ratio, k_max_ratio, n);
}

}  // namespace vse
