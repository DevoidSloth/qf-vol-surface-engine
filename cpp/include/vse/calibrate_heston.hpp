// vse/calibrate_heston.hpp — Heston calibration by Levenberg-Marquardt with an
// exact Jacobian.
//
// The residual is in implied volatility, not price, for the reason set out in
// calibrate_svi.hpp: a chain spans five orders of magnitude in price and a
// price-space objective simply ignores the wings.
//
// That choice makes the Jacobian look harder than it is. What the AD gives is
// d(price)/d(parameter); what the optimiser needs is d(vol)/d(parameter). The
// implicit function theorem supplies the link in one line -- the model vol is
// defined by Black(sigma) = HestonPrice(params), so differentiating both sides
//
//     vega * d(sigma)/d(p) = d(HestonPrice)/d(p),
//
// and vega is closed form. No extra pricing, no finite differences, and the
// division by vega is exactly the vega weighting the objective wanted anyway:
// residuals end up in units of price divided by vega, weighted by 1/spread.
//
// Cost, for a five-parameter model and a board of N quotes: one pass giving
// price and all five derivatives, against eleven passes for a central-difference
// Jacobian -- and those eleven are each accurate to about 1e-8, which is enough
// to stall the optimiser two digits above the noise floor of the data.
#pragma once

#include "vse/black.hpp"
#include "vse/chain.hpp"
#include "vse/common.hpp"
#include "vse/heston.hpp"
#include "vse/implied_vol.hpp"
#include "vse/lm.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace vse {

/// One quote as the Heston calibrator sees it.
struct CalibrationQuote {
    Real forward = 0.0;
    Real strike = 0.0;
    Real expiry = 0.0;
    Real implied_vol = 0.0;
    Real weight = 1.0;
};

struct HestonFitResult {
    HestonParams params;
    Real rmse_vol_points = 0.0;      ///< unweighted, in absolute vol (0.01 = 1 point)
    Real max_error_vol_points = 0.0;
    Real weighted_rms = 0.0;
    int  quotes = 0;
    int  iterations = 0;
    int  slice_builds = 0;      ///< quadrature passes; one per expiry per LM step
    bool converged = false;
    std::vector<bool> at_bound;
    std::string message;
};

/// Flatten a set of cleaned slices into calibration quotes.
inline std::vector<CalibrationQuote> quotes_from_slices(
    const std::vector<std::vector<SurfacePoint>>& slices,
    const std::vector<Real>& expiries, const std::vector<Real>& forwards) {
    require(slices.size() == expiries.size() && slices.size() == forwards.size(),
            "quotes_from_slices: mismatched lengths");
    std::vector<CalibrationQuote> out;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        for (const auto& p : slices[i]) {
            out.push_back({forwards[i], p.strike, expiries[i], p.implied_vol, p.weight});
        }
    }
    return out;
}

namespace detail {

/// Residuals and Jacobian for a Heston calibration.
///
/// Quotes are bucketed by expiry once, at construction, and each residual or
/// Jacobian pass builds one HestonSliceEngine per bucket rather than one per
/// quote. That is the whole performance story: the characteristic function does
/// not depend on the strike, so a board of 40 strikes on 7 expiries needs 7
/// passes over the quadrature, not 280. Measured 21x on the Jacobian pass.
struct HestonObjective {
    const std::vector<CalibrationQuote>* quotes;
    int order = 32;
    int panels = 16;
    mutable int slice_builds = 0;

    std::vector<Real> expiries;                        ///< unique, ascending
    std::vector<std::vector<std::size_t>> by_expiry;   ///< indices into *quotes

    explicit HestonObjective(const std::vector<CalibrationQuote>& q, int ord = 32,
                             int pan = 16)
        : quotes(&q), order(ord), panels(pan) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            const auto it = std::find_if(expiries.begin(), expiries.end(),
                                         [&](Real e) { return e == q[i].expiry; });
            if (it == expiries.end()) {
                expiries.push_back(q[i].expiry);
                by_expiry.push_back({i});
            } else {
                by_expiry[std::size_t(it - expiries.begin())].push_back(i);
            }
        }
    }

    static HestonParams unpack(const std::vector<Real>& x) {
        HestonParams p;
        p.v0 = x[0]; p.kappa = x[1]; p.theta = x[2]; p.sigma = x[3]; p.rho = x[4];
        return p;
    }

    /// Implied vol of a Heston price, or the market vol if the price falls
    /// outside the no-arbitrage bounds.
    ///
    /// The fallback matters. Deep in a wing at extreme trial parameters the
    /// quadrature can return a price a hair outside [intrinsic, forward], and
    /// inverting that throws. A calibration that dies because one trial vector
    /// produced one unusable price is not usable. Returning the market vol makes
    /// that residual zero, which is neutral rather than attractive, so the
    /// optimiser is not drawn towards the bad region -- an important distinction
    /// from returning a large penalty, which would push it away and bias the fit.
    static Real vol_from_call(Real call, const CalibrationQuote& q) {
        const bool otm_call = q.strike >= q.forward;
        const Real price = otm_call ? call : call - (q.forward - q.strike);
        const auto type = otm_call ? OptionType::Call : OptionType::Put;
        Real lo, hi;
        black_price_bounds(q.forward, q.strike, type, lo, hi);
        if (!(price > lo && price < hi)) return q.implied_vol;
        const auto iv = implied_volatility_ex(price, q.forward, q.strike, q.expiry, 1.0, type);
        return (iv.converged && iv.sigma > 0.0) ? iv.sigma : q.implied_vol;
    }

    void residual(const std::vector<Real>& x, std::vector<Real>& r) const {
        const HestonParams p = unpack(x);
        r.assign(quotes->size(), 0.0);
        for (std::size_t e = 0; e < expiries.size(); ++e) {
            const HestonSliceEngine engine(p, expiries[e], false, order, panels);
            ++slice_builds;
            for (std::size_t i : by_expiry[e]) {
                const auto& q = (*quotes)[i];
                r[i] = q.weight * (vol_from_call(engine.call(q.forward, q.strike), q) -
                                   q.implied_vol);
            }
        }
    }

    void jacobian(const std::vector<Real>& x, Matrix& j) const {
        const HestonParams p = unpack(x);
        for (std::size_t e = 0; e < expiries.size(); ++e) {
            const HestonSliceEngine engine(p, expiries[e], true, order, panels);
            ++slice_builds;
            for (std::size_t i : by_expiry[e]) {
                const auto& q = (*quotes)[i];
                const auto pg = engine.call_and_gradient(q.forward, q.strike);
                const Real vol = vol_from_call(pg.price, q);

                // d(vol)/d(param) = d(price)/d(param) / vega, by the implicit
                // function theorem applied to Black(vol) = HestonPrice(params).
                // Call and put share both, since they differ by F - K, which is
                // parameter-free.
                const Real s = vol * std::sqrt(q.expiry);
                const Real k = std::log(q.forward / q.strike);
                const Real vega = std::sqrt(q.forward * q.strike) * normalised_vega(k, s) *
                                  std::sqrt(q.expiry);
                const Real inv_vega = (vega > 1e-300) ? 1.0 / vega : 0.0;
                for (int c = 0; c < 5; ++c) {
                    j(i, std::size_t(c)) = q.weight * pg.gradient[std::size_t(c)] * inv_vega;
                }
            }
        }
    }

    /// Model vols for every quote, one slice engine per expiry.
    std::vector<Real> model_vols(const HestonParams& p) const {
        std::vector<Real> out(quotes->size(), 0.0);
        for (std::size_t e = 0; e < expiries.size(); ++e) {
            const HestonSliceEngine engine(p, expiries[e], false, order, panels);
            for (std::size_t i : by_expiry[e]) {
                const auto& q = (*quotes)[i];
                out[i] = vol_from_call(engine.call(q.forward, q.strike), q);
            }
        }
        return out;
    }
};

}  // namespace detail

/// Calibrate Heston to a board of implied vols.
inline HestonFitResult calibrate_heston(const std::vector<CalibrationQuote>& quotes,
                                        const HestonParams& start = {},
                                        const LMOptions& opt = {},
                                        int order = 32, int panels = 16) {
    require(quotes.size() >= 5, "calibrate_heston: need at least five quotes");

    detail::HestonObjective obj(quotes, order, panels);
    ResidualFn res = [&obj](const std::vector<Real>& x, std::vector<Real>& r) { obj.residual(x, r); };
    JacobianFn jac = [&obj](const std::vector<Real>& x, Matrix& j) { obj.jacobian(x, j); };

    Box box;
    // Wide but not unbounded. rho is kept off +/-1 because the characteristic
    // function is singular there, and kappa off zero because a vanishing
    // mean-reversion makes theta unidentifiable rather than merely large.
    box.lower = {1e-6, 1e-3, 1e-6, 1e-3, -0.9995};
    box.upper = {4.0,  30.0, 4.0,  10.0,  0.9995};

    const std::vector<Real> x0 = {start.v0, start.kappa, start.theta, start.sigma, start.rho};
    const LMResult lm = levenberg_marquardt(res, jac, x0, box, opt);

    HestonFitResult out;
    out.params = detail::HestonObjective::unpack(lm.x);
    out.iterations = lm.iterations;
    out.converged = lm.converged();
    out.at_bound = lm.at_bound;
    out.weighted_rms = lm.rms;
    out.message = lm.status_text();
    out.quotes = int(quotes.size());
    out.slice_builds = obj.slice_builds;

    const std::vector<Real> vols = obj.model_vols(out.params);
    Real ss = 0.0;
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        const Real e = vols[i] - quotes[i].implied_vol;
        ss += e * e;
        out.max_error_vol_points = std::fmax(out.max_error_vol_points, std::fabs(e));
    }
    out.rmse_vol_points = std::sqrt(ss / Real(quotes.size()));
    return out;
}

/// Calibrate from several starting points and keep the best.
///
/// The Heston objective is not convex and the (kappa, theta) pair is weakly
/// identified from a single board -- a fast mean reversion to a low long-run
/// variance and a slow one to a higher variance produce nearly the same
/// six-month smile. Reporting the spread across starting points is more
/// informative than reporting the winner alone, so the caller gets both.
struct HestonMultiStartResult {
    HestonFitResult best;
    std::vector<HestonParams> all_optima;
    std::vector<Real> all_rmse;

    /// Coefficient of variation of each parameter across the starts that landed
    /// within `tolerance` of the best RMSE. Large values are a genuine finding
    /// about identifiability, not a defect in the optimiser.
    std::array<Real, 5> dispersion(Real tolerance = 1.05) const {
        std::array<Real, 5> out{};
        Real best_rmse = DBL_HUGE;
        for (Real r : all_rmse) best_rmse = std::fmin(best_rmse, r);
        std::array<Real, 5> mean{}, m2{};
        int n = 0;
        for (std::size_t i = 0; i < all_optima.size(); ++i) {
            if (all_rmse[i] > best_rmse * tolerance) continue;
            const auto& p = all_optima[i];
            const std::array<Real, 5> v = {p.v0, p.kappa, p.theta, p.sigma, p.rho};
            ++n;
            for (int c = 0; c < 5; ++c) {
                const Real delta = v[std::size_t(c)] - mean[std::size_t(c)];
                mean[std::size_t(c)] += delta / Real(n);
                m2[std::size_t(c)] += delta * (v[std::size_t(c)] - mean[std::size_t(c)]);
            }
        }
        if (n < 2) return out;
        for (int c = 0; c < 5; ++c) {
            const Real sd = std::sqrt(m2[std::size_t(c)] / Real(n - 1));
            out[std::size_t(c)] = sd / std::fmax(std::fabs(mean[std::size_t(c)]), 1e-12);
        }
        return out;
    }
};

inline HestonMultiStartResult calibrate_heston_multistart(
    const std::vector<CalibrationQuote>& quotes, const LMOptions& opt = {},
    int order = 32, int panels = 16) {
    HestonMultiStartResult out;
    out.best.rmse_vol_points = DBL_HUGE;

    for (Real v0 : {0.01, 0.04, 0.09}) {
        for (Real kappa : {0.5, 2.0, 6.0}) {
            for (Real rho : {-0.9, -0.6}) {
                HestonParams start{v0, kappa, v0, 0.6, rho};
                const auto fit = calibrate_heston(quotes, start, opt, order, panels);
                out.all_optima.push_back(fit.params);
                out.all_rmse.push_back(fit.rmse_vol_points);
                if (fit.rmse_vol_points < out.best.rmse_vol_points) out.best = fit;
            }
        }
    }
    return out;
}

}  // namespace vse
