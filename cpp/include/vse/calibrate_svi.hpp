// vse/calibrate_svi.hpp — fitting SVI, SSVI and eSSVI to a cleaned chain.
//
// What is being minimised, and why it is not the obvious thing:
//
// The obvious objective is the sum of squared differences between model and
// market *prices*. It is wrong here, because a chain spans five orders of
// magnitude in price and the deep wings would contribute essentially nothing.
// The next obvious objective is the sum of squared differences in *implied vol*,
// unweighted. That is wrong in the opposite direction: out in the wings a single
// tick of price is several vol points, so unweighted vol least squares spends
// most of its degrees of freedom fitting the noisiest observations on the slice.
//
// What is minimised instead is the vol error divided by the half-spread
// expressed in vol points -- that is, each residual is measured in units of how
// precisely that quote is known. Equivalently, since dPrice = vega dVol, it is
// the price error in units of the bid-ask. Tight at-the-money quotes dominate,
// wide wing quotes contribute in proportion to their information content, and a
// quote whose spread is wider than the smile it is supposed to constrain is
// almost ignored without having to be deleted.
//
// RMSE is then reported in implied vol points, unweighted, because that is the
// number that is comparable across strikes and across papers. Reporting a
// calibration RMSE in price is meaningless -- 0.01 is superb at the money and
// catastrophic five standard deviations out.
#pragma once

#include "vse/chain.hpp"
#include "vse/common.hpp"
#include "vse/lm.hpp"
#include "vse/svi.hpp"

#include <string>
#include <vector>

namespace vse {

struct SVIFitResult {
    SVIRaw params;
    Real rmse_vol = 0.0;      ///< absolute vol: 0.0162 is 1.62 vol points
    Real max_error_vol = 0.0;
    Real weighted_rms = 0.0;         ///< in units of the half-spread
    int  iterations = 0;
    bool converged = false;
    ButterflyReport butterfly;
    std::string message;
};

namespace detail {

/// Residuals and Jacobian for one SVI slice.
///
/// Parameters are (a, b, rho, m, sigma). The Jacobian is analytic:
///     dw/da = 1
///     dw/db = rho (k-m) + sqrt((k-m)^2 + sigma^2)
///     dw/drho = b (k-m)
///     dw/dm = -dw/dk
///     dw/dsigma = b sigma / sqrt((k-m)^2 + sigma^2)
/// and the chain rule to implied vol is dsigma/dp = (dw/dp) / (2 T sigma).
struct SVIObjective {
    const std::vector<SurfacePoint>* points;
    Real expiry;

    static SVIRaw unpack(const std::vector<Real>& x) {
        SVIRaw s;
        s.a = x[0]; s.b = x[1]; s.rho = x[2]; s.m = x[3]; s.sigma = x[4];
        return s;
    }

    void residual(const std::vector<Real>& x, std::vector<Real>& r) const {
        const SVIRaw s = unpack(x);
        r.resize(points->size());
        for (std::size_t i = 0; i < points->size(); ++i) {
            const auto& p = (*points)[i];
            const Real w = s.total_variance(p.log_moneyness);
            const Real model_vol = std::sqrt(std::fmax(w, 1e-12) / expiry);
            r[i] = p.weight * (model_vol - p.implied_vol);
        }
    }

    void jacobian(const std::vector<Real>& x, Matrix& j) const {
        const SVIRaw s = unpack(x);
        for (std::size_t i = 0; i < points->size(); ++i) {
            const auto& p = (*points)[i];
            const Real k = p.log_moneyness;
            const Real y = k - s.m;
            const Real root = std::sqrt(y * y + s.sigma * s.sigma);
            const Real w = std::fmax(s.total_variance(k), 1e-12);
            const Real model_vol = std::sqrt(w / expiry);
            const Real chain = p.weight / (2.0 * expiry * model_vol);

            j(i, 0) = chain * 1.0;
            j(i, 1) = chain * (s.rho * y + root);
            j(i, 2) = chain * (s.b * y);
            j(i, 3) = chain * (-s.b * (s.rho + y / root));
            j(i, 4) = chain * (s.b * s.sigma / root);
        }
    }
};

}  // namespace detail

/// Fit one raw-SVI slice.
///
/// Multi-start over (m, sigma, rho) rather than a single seed. The SVI objective
/// is genuinely multi-modal in m: a slice with a pronounced smile admits a good
/// local minimum with the vertex on either side of the money, and which one a
/// single-start optimiser finds depends on the seed. The extra cost is a few
/// milliseconds and the alternative is a surface that changes shape day to day
/// for no reason in the data, which then shows up as "unstable calibration" and
/// gets blamed on the market.
inline SVIFitResult fit_svi_slice(const std::vector<SurfacePoint>& points, Real expiry,
                                  const LMOptions& opt = {}) {
    require(points.size() >= 5, "fit_svi_slice: need at least five quotes to fit five parameters");
    require(expiry > 0.0, "fit_svi_slice: expiry must be positive");

    detail::SVIObjective obj{&points, expiry};
    ResidualFn res = [&obj](const std::vector<Real>& x, std::vector<Real>& r) { obj.residual(x, r); };
    JacobianFn jac = [&obj](const std::vector<Real>& x, Matrix& j) { obj.jacobian(x, j); };

    Real w_min = DBL_HUGE, w_max = 0.0;
    for (const auto& p : points) {
        w_min = std::fmin(w_min, p.total_variance);
        w_max = std::fmax(w_max, p.total_variance);
    }

    Box box;
    // b is capped by Lee's moment formula: the wing slopes b(1 +/- rho) may not
    // exceed 2, so b <= 2 is necessary and the tighter coupled bound is checked
    // afterwards rather than imposed as a box.
    box.lower = {-2.0 * w_max, 0.0,   -0.9999, -3.0, 1e-4};
    box.upper = { 2.0 * w_max, 2.0,    0.9999,  3.0, 5.0};

    SVIFitResult best;
    best.rmse_vol = DBL_HUGE;

    for (Real m0 : {-0.15, 0.0, 0.15}) {
        for (Real sig0 : {0.05, 0.2, 0.5}) {
            for (Real rho0 : {-0.7, -0.3, 0.0}) {
                std::vector<Real> x0 = {std::fmax(w_min * 0.5, 1e-6), 0.1, rho0, m0, sig0};
                const LMResult lm = levenberg_marquardt(res, jac, x0, box, opt);

                const SVIRaw s = detail::SVIObjective::unpack(lm.x);
                Real ss = 0.0, worst = 0.0;
                for (const auto& p : points) {
                    const Real e = s.implied_vol(p.log_moneyness, expiry) - p.implied_vol;
                    ss += e * e;
                    worst = std::fmax(worst, std::fabs(e));
                }
                const Real rmse = std::sqrt(ss / Real(points.size()));
                if (rmse < best.rmse_vol) {
                    best.params = s;
                    best.rmse_vol = rmse;
                    best.max_error_vol = worst;
                    best.weighted_rms = lm.rms;
                    best.iterations = lm.iterations;
                    best.converged = lm.converged();
                    best.message = lm.status_text();
                }
            }
        }
    }

    best.butterfly = check_butterfly(best.params, expiry);
    if (!best.butterfly.free) {
        best.message += "; butterfly arbitrage present (min g = " +
                        std::to_string(best.butterfly.min_g) + ")";
    }
    return best;
}

// ---------------------------------------------------------------------------
// SSVI
// ---------------------------------------------------------------------------

struct SSVIFitResult {
    SSVISurface surface;
    Real rmse_vol = 0.0;
    Real max_error_vol = 0.0;
    int  quotes = 0;
    int  iterations = 0;
    bool converged = false;
    std::vector<SSVIConditionReport> conditions;   ///< one per expiry
    CalendarReport calendar;
    std::vector<ButterflyReport> butterfly;        ///< one per expiry
    std::string message;
};

namespace detail {

/// Residuals and Jacobian for a whole SSVI surface in (eta, gamma, rho).
///
/// theta is not a free parameter: it is the observed at-the-money total variance
/// of each expiry. That is the design of SSVI -- the term structure comes from
/// the market and the three parameters describe the shape. Fitting theta as well
/// would let the surface drift away from the one point on each slice that is
/// quoted most tightly.
struct SSVIObjective {
    const std::vector<std::vector<SurfacePoint>>* slices;
    const std::vector<Real>* expiries;
    const std::vector<Real>* theta;

    std::size_t total_points() const {
        std::size_t n = 0;
        for (const auto& s : *slices) n += s.size();
        return n;
    }

    void residual(const std::vector<Real>& x, std::vector<Real>& r) const {
        PowerLawPhi phi{x[0], x[1]};
        const Real rho = x[2];
        r.resize(total_points());
        std::size_t idx = 0;
        for (std::size_t si = 0; si < slices->size(); ++si) {
            const Real th = (*theta)[si], T = (*expiries)[si];
            SSVISlice slice{th, rho, phi(th)};
            for (const auto& p : (*slices)[si]) {
                const Real w = std::fmax(slice.total_variance(p.log_moneyness), 1e-12);
                r[idx++] = p.weight * (std::sqrt(w / T) - p.implied_vol);
            }
        }
    }

    void jacobian(const std::vector<Real>& x, Matrix& j) const {
        const Real eta = x[0], gamma = x[1], rho = x[2];
        PowerLawPhi phi{eta, gamma};
        std::size_t idx = 0;
        for (std::size_t si = 0; si < slices->size(); ++si) {
            const Real th = (*theta)[si], T = (*expiries)[si];
            const Real ph = phi(th);
            // dphi/deta = phi/eta;  dphi/dgamma = phi * ln((1+theta)/theta)
            const Real dphi_deta   = ph / eta;
            const Real dphi_dgamma = ph * std::log((1.0 + th) / th);

            for (const auto& p : (*slices)[si]) {
                const Real k = p.log_moneyness;
                const Real z = ph * k + rho;
                const Real D = std::sqrt(z * z + 1.0 - rho * rho);
                const Real w = std::fmax(0.5 * th * (1.0 + rho * ph * k + D), 1e-12);
                const Real model_vol = std::sqrt(w / T);
                const Real chain = p.weight / (2.0 * T * model_vol);

                const Real dw_dphi = 0.5 * th * (rho * k + z * k / D);
                const Real dw_drho = 0.5 * th * (ph * k + (z - rho) / D);

                j(idx, 0) = chain * dw_dphi * dphi_deta;
                j(idx, 1) = chain * dw_dphi * dphi_dgamma;
                j(idx, 2) = chain * dw_drho;
                ++idx;
            }
        }
    }
};

}  // namespace detail

/// Fit an SSVI surface to every slice at once.
inline SSVIFitResult fit_ssvi(const std::vector<std::vector<SurfacePoint>>& slices,
                              const std::vector<Real>& expiries,
                              const std::vector<Real>& theta,
                              const LMOptions& opt = {}) {
    require(slices.size() == expiries.size() && slices.size() == theta.size(),
            "fit_ssvi: slices, expiries and theta must have the same length");
    require(!slices.empty(), "fit_ssvi: no slices");

    detail::SSVIObjective obj{&slices, &expiries, &theta};
    ResidualFn res = [&obj](const std::vector<Real>& x, std::vector<Real>& r) { obj.residual(x, r); };
    JacobianFn jac = [&obj](const std::vector<Real>& x, Matrix& j) { obj.jacobian(x, j); };

    Box box;
    box.lower = {1e-3, 0.0,    -0.9999};
    box.upper = {10.0, 1.0,     0.9999};

    SSVIFitResult best;
    best.rmse_vol = DBL_HUGE;

    for (Real eta0 : {0.3, 1.0, 2.5}) {
        for (Real gamma0 : {0.25, 0.5}) {
            for (Real rho0 : {-0.8, -0.5, -0.2}) {
                const LMResult lm = levenberg_marquardt(res, jac, {eta0, gamma0, rho0}, box, opt);

                SSVISurface surf;
                surf.phi = PowerLawPhi{lm.x[0], lm.x[1]};
                surf.rho = lm.x[2];
                surf.expiries = expiries;
                surf.theta = theta;

                Real ss = 0.0, worst = 0.0;
                int n = 0;
                for (std::size_t si = 0; si < slices.size(); ++si) {
                    const SSVISlice sl = surf.slice_at_index(si);
                    for (const auto& p : slices[si]) {
                        const Real e = sl.implied_vol(p.log_moneyness, expiries[si]) - p.implied_vol;
                        ss += e * e;
                        worst = std::fmax(worst, std::fabs(e));
                        ++n;
                    }
                }
                const Real rmse = std::sqrt(ss / Real(n));
                if (rmse < best.rmse_vol) {
                    best.surface = surf;
                    best.rmse_vol = rmse;
                    best.max_error_vol = worst;
                    best.quotes = n;
                    best.iterations = lm.iterations;
                    best.converged = lm.converged();
                    best.message = lm.status_text();
                }
            }
        }
    }

    // Report the conditions rather than assume them.
    best.conditions.clear();
    best.butterfly.clear();
    for (std::size_t si = 0; si < expiries.size(); ++si) {
        const Real th = theta[si];
        const Real ph = best.surface.phi(th);
        auto rep = check_ssvi_conditions(th, best.surface.rho, ph,
                                         best.surface.phi.d_theta_phi(th));
        rep.theta_increasing = si == 0 || theta[si] >= theta[si - 1];
        best.conditions.push_back(rep);
        best.butterfly.push_back(check_butterfly(best.surface.slice_at_index(si), expiries[si]));
    }
    std::vector<Real> dense_times;
    for (int i = 0; i <= 200; ++i) {
        dense_times.push_back(expiries.front() +
                              (expiries.back() - expiries.front()) * i / 200.0);
    }
    best.calendar = check_calendar(best.surface, dense_times);
    return best;
}

// ---------------------------------------------------------------------------
// eSSVI
// ---------------------------------------------------------------------------

struct ESSVIFitResult {
    ESSVISurface surface;
    Real rmse_vol = 0.0;
    Real max_error_vol = 0.0;
    int  quotes = 0;
    bool converged = true;
    bool calendar_conditions_hold = false;
    CalendarReport calendar;
    std::vector<ButterflyReport> butterfly;
    std::string message;
};

/// Fit eSSVI slice by slice, forward in maturity, with the calendar conditions
/// carried across as constraints.
///
/// psi = theta * phi and rho are free per slice; theta is the observed ATM total
/// variance as in SSVI. Both calendar conditions are enforced by construction
/// rather than by penalty or projection -- see the reparameterisation inside.
inline ESSVIFitResult fit_essvi(const std::vector<std::vector<SurfacePoint>>& slices,
                                const std::vector<Real>& expiries,
                                const std::vector<Real>& theta,
                                const LMOptions& opt = {}) {
    require(slices.size() == expiries.size() && slices.size() == theta.size(),
            "fit_essvi: slices, expiries and theta must have the same length");

    ESSVIFitResult out;
    out.surface.expiries = expiries;
    out.surface.theta = theta;
    out.surface.rho.assign(slices.size(), -0.5);
    out.surface.psi.assign(slices.size(), 0.5);

    Real ss = 0.0, worst = 0.0;
    int n_total = 0;

    for (std::size_t si = 0; si < slices.size(); ++si) {
        const auto& pts = slices[si];
        const Real th = theta[si], T = expiries[si];
        require(pts.size() >= 3, "fit_essvi: need at least three quotes per slice");

        const bool has_prev = si > 0;
        const Real prev_psi = has_prev ? out.surface.psi[si - 1] : 0.0;
        const Real prev_rho_psi = has_prev ? out.surface.rho[si - 1] * prev_psi : 0.0;

        // Reparameterise so the calendar condition cannot be violated.
        //
        // The constraint on consecutive slices is
        //     psi_2 >= psi_1  and  |rho_2 psi_2 - rho_1 psi_1| <= psi_2 - psi_1,
        // which couples rho and psi and so cannot be written as a box. Writing
        // the skew product as
        //     rho_2 psi_2 = rho_1 psi_1 + t (psi_2 - psi_1),   t in [-1, 1],
        // makes it an identity: the second condition holds for every admissible
        // (t, psi_2), and it is then a box.
        //
        // It also delivers |rho_2| < 1 for free, since
        //     |rho_1 psi_1 + t (psi_2 - psi_1)| <= |rho_1| psi_1 + psi_2 - psi_1 < psi_2.
        //
        // This replaced a penalty term. The penalty worked on a seven-expiry
        // board and failed on a fourteen-expiry one, for the reason penalties
        // usually fail: with expiries close together psi_2 - psi_1 is small, so
        // the constraint is tight exactly where the penalty is weakest relative
        // to data residuals that are themselves weighted by 1/spread. A
        // constraint that holds by construction has no weight to tune.
        auto unpack = [&](const std::vector<Real>& x, Real& rho_out, Real& psi_out) {
            if (!has_prev) {
                rho_out = x[0];
                psi_out = x[1];
            } else {
                psi_out = x[1];
                rho_out = (prev_rho_psi + x[0] * (psi_out - prev_psi)) / psi_out;
            }
        };

        ResidualFn res = [&](const std::vector<Real>& x, std::vector<Real>& r) {
            Real rho_i, psi_i;
            unpack(x, rho_i, psi_i);
            const SSVISlice sl{th, rho_i, psi_i / th};
            r.resize(pts.size());
            for (std::size_t i = 0; i < pts.size(); ++i) {
                const Real w = std::fmax(sl.total_variance(pts[i].log_moneyness), 1e-12);
                r[i] = pts[i].weight * (std::sqrt(w / T) - pts[i].implied_vol);
            }
        };

        Box box;
        // First coordinate is rho on the front slice and the interpolation
        // weight t on every later one; both live in [-1, 1].
        box.lower = {-0.9999, has_prev ? prev_psi : 1e-4};
        box.upper = { 0.9999, 50.0};

        LMResult best_lm;
        Real best_rmse = DBL_HUGE;
        bool any = false;
        for (Real first0 : {-0.9, -0.5, 0.0, 0.5}) {
            for (Real psi_mult : {1.0, 1.2, 2.0, 5.0}) {
                const Real psi0 = std::fmax(has_prev ? prev_psi * psi_mult : 0.5 * psi_mult, 1e-3);
                const LMResult lm = levenberg_marquardt(res, numerical_jacobian(res),
                                                        {first0, psi0}, box, opt);
                Real rho_i, psi_i;
                unpack(lm.x, rho_i, psi_i);
                const SSVISlice sl{th, rho_i, psi_i / th};
                Real s2 = 0.0;
                for (const auto& p : pts) s2 += sqr(sl.implied_vol(p.log_moneyness, T) - p.implied_vol);
                const Real rmse = std::sqrt(s2 / Real(pts.size()));
                if (rmse < best_rmse) { best_rmse = rmse; best_lm = lm; any = true; }
            }
        }
        require(any, "fit_essvi: no start produced a usable fit");

        unpack(best_lm.x, out.surface.rho[si], out.surface.psi[si]);
        out.converged = out.converged && best_lm.converged();

        const SSVISlice sl = out.surface.slice_at_index(si);
        for (const auto& p : pts) {
            const Real e = sl.implied_vol(p.log_moneyness, T) - p.implied_vol;
            ss += e * e;
            worst = std::fmax(worst, std::fabs(e));
            ++n_total;
        }
        out.butterfly.push_back(check_butterfly(sl, T));
    }

    out.rmse_vol = std::sqrt(ss / Real(n_total));
    out.max_error_vol = worst;
    out.quotes = n_total;
    out.calendar_conditions_hold = out.surface.calendar_conditions_hold();
    out.calendar = check_calendar(out.surface, expiries);
    out.message = out.calendar_conditions_hold ? "calendar conditions hold"
                                               : "calendar conditions violated";
    return out;
}

}  // namespace vse
