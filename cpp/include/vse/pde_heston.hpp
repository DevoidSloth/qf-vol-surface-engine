// vse/pde_heston.hpp — the two-dimensional Heston PDE by ADI.
//
//     dV/dtau = 1/2 v S^2 V_SS + rho sigma v S V_Sv + 1/2 sigma^2 v V_vv
//               + (r - q) S V_S + kappa (theta - v) V_v - r V
//
// This exists to be a third opinion. The Lewis integral and the Carr-Madan
// transform share a characteristic function; if that function is wrong they are
// wrong together and agree beautifully. A finite-difference solution of the PDE
// shares nothing with either -- no complex arithmetic, no contour, no Fourier
// transform -- so agreement between them is evidence rather than coincidence.
//
// THE MIXED DERIVATIVE
//
// A straight implicit scheme in two dimensions means solving a banded system
// with bandwidth equal to a whole grid row, which is why ADI exists: split the
// operator by direction and solve two tridiagonal systems instead. The
// difficulty is the rho sigma v S V_Sv cross term, which belongs to no single
// direction and cannot be made implicit by any directional split.
//
// Douglas's answer is to treat the mixed term explicitly in a predictor and
// correct the directional terms implicitly:
//
//     Y0 = V + dt (A0 + A1 + A2) V
//     (I - t dt A1) Y1 = Y0 - t dt A1 V
//     (I - t dt A2) Y2 = Y1 - t dt A2 V
//
// with t = 1/2. That is second-order accurate in time when the mixed term is
// absent and only FIRST-order when it is present, and the difference is not
// academic: measured on a one-year 25% out-of-the-money call, halving the time
// step halved the error, so reaching 1% took 400 steps.
//
// Craig-Sneyd repairs it with one extra explicit correction of the mixed term,
// evaluated at the Douglas answer rather than at the old time level:
//
//     Y0hat = Y0 + 1/2 dt (A0 Y2 - A0 V)
//     (I - t dt A1) Y1hat = Y0hat - t dt A1 V
//     (I - t dt A2) Y2hat = Y1hat - t dt A2 V
//
// This restores second order at the cost of one more application of A0 and one
// more pair of tridiagonal sweeps -- roughly double the work per step, in
// exchange for needing far fewer of them. Both schemes are available; Craig-Sneyd
// is the default and the test suite measures the observed convergence order of
// each, rather than asserting what the papers say it should be.
//
// THE v = 0 BOUNDARY
//
// At v = 0 the diffusion vanishes in both directions and the equation degenerates
// to a first-order transport equation, dV/dtau = (r-q) S V_S + kappa theta V_v -
// r V. The characteristic there points INTO the domain (kappa theta > 0 pushes
// variance up), so no boundary condition may be imposed: the correct treatment
// is to apply the degenerate equation itself with a one-sided difference in v.
// Imposing a Dirichlet value instead -- the Black-Scholes price at zero vol, say
// -- over-determines the problem and the error propagates inward. This is the
// part of a Heston PDE that is most often got wrong.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/heston.hpp"
#include "vse/linalg.hpp"
#include "vse/pde.hpp"

#include <vector>

namespace vse {

struct HestonPDEConfig {
    int  spot_steps = 160;
    int  var_steps  = 80;
    int  time_steps = 100;
    Real spot_width_sd = 5.0;   ///< half-width in log-spot, in standard deviations
    Real var_width_sd = 10.0;   ///< v_max as standard deviations of v_T above its mean
    Real var_max_floor_mult = 6.0;  ///< and never less than this multiple of max(v0, theta)
    Real spot_concentration = 0.2;
    Real var_concentration = 0.5;
    Real theta = 0.5;           ///< ADI implicitness parameter
    int  rannacher_steps = 2;
    bool craig_sneyd = true;    ///< false selects plain Douglas
};

struct HestonPDEResult {
    Real price = 0.0;
    Real delta = 0.0;
    Real gamma = 0.0;
    std::size_t spot_nodes = 0;
    std::size_t var_nodes = 0;
};

namespace detail {

/// Non-uniform three-point weights for d/dz and d2/dz2 at an interior node.
struct Stencil {
    Real d1_lo, d1_mid, d1_hi;
    Real d2_lo, d2_mid, d2_hi;
};

inline Stencil central_stencil(Real hm, Real hp) {
    const Real hs = hm + hp;
    Stencil s;
    s.d1_lo = -hp / (hm * hs);
    s.d1_mid = (hp - hm) / (hm * hp);
    s.d1_hi = hm / (hp * hs);
    s.d2_lo = 2.0 / (hm * hs);
    s.d2_mid = -2.0 / (hm * hp);
    s.d2_hi = 2.0 / (hp * hs);
    return s;
}

}  // namespace detail

/// Price a European vanilla under Heston by ADI.
inline HestonPDEResult heston_pde(const HestonParams& p, Real spot, Real strike, Real expiry,
                                  Real rate, Real dividend, OptionType type,
                                  const HestonPDEConfig& cfg = {}) {
    require(p.is_well_formed(), "heston_pde: parameters out of range");
    require(spot > 0.0 && strike > 0.0 && expiry > 0.0,
            "heston_pde: spot, strike and expiry must be positive");

    // ---- grids ----------------------------------------------------------
    const Real ref_var = std::fmax(std::fmax(p.v0, p.theta), 1e-6);
    const Real sd = std::sqrt(ref_var * expiry);
    const Real x_strike = std::log(strike);
    const Real centre = std::log(spot) + (rate - dividend) * expiry;
    const Real width = cfg.spot_width_sd * sd + std::fabs(centre - x_strike);
    const std::vector<Real> x = detail::sinh_grid(
        std::fmin(centre, x_strike) - width, std::fmax(centre, x_strike) + width, x_strike,
        std::fmax(cfg.spot_concentration * width, 1e-6), cfg.spot_steps);

    // v_max from the actual distribution of v_T, not from a multiple of v0.
    //
    // A fixed multiple is the obvious choice and it is badly wrong for the
    // parameters an equity index calibrates to. The CIR variance has
    //     mean  m = theta + (v0 - theta) e^{-kappa T}
    //     var   s^2 = v0 sigma^2 e^{-kT}(1 - e^{-kT})/kappa
    //                 + theta sigma^2 (1 - e^{-kT})^2 / (2 kappa)
    // and with sigma = 0.92 that standard deviation is 0.103 against a mean of
    // 0.043 -- so the distribution reaches far past any small multiple of v0.
    // Truncating at 8 v0 and imposing dV/dv = 0 there cost 1.45% on a
    // one-year at-the-money call, and the error did NOT shrink with grid
    // refinement, which is the signature of a domain that is too small rather
    // than a grid that is too coarse.
    const Real kt = p.kappa * expiry;
    const Real decay = std::exp(-kt);
    const Real var_mean = p.theta + (p.v0 - p.theta) * decay;
    const Real var_var = p.v0 * p.sigma * p.sigma * decay * (1.0 - decay) / p.kappa +
                         p.theta * p.sigma * p.sigma * sqr(1.0 - decay) / (2.0 * p.kappa);
    const Real v_max = std::fmax(var_mean + cfg.var_width_sd * std::sqrt(std::fmax(var_var, 0.0)),
                                 cfg.var_max_floor_mult * ref_var);
    std::vector<Real> v(std::size_t(cfg.var_steps) + 1);
    {
        // Concentrated near zero, where the variance density lives and where the
        // boundary treatment is delicate.
        const Real alpha = std::fmax(cfg.var_concentration * ref_var, 1e-8);
        const Real c2 = std::asinh(v_max / alpha);
        for (int j = 0; j <= cfg.var_steps; ++j) {
            v[std::size_t(j)] = alpha * std::sinh(c2 * Real(j) / Real(cfg.var_steps));
        }
        v.front() = 0.0;
    }

    const std::size_t nx = x.size(), nv = v.size();
    std::vector<Real> spots(nx);
    for (std::size_t i = 0; i < nx; ++i) spots[i] = std::exp(x[i]);

    auto at = [nv](std::size_t i, std::size_t j) { return i * nv + j; };
    std::vector<Real> u(nx * nv), y0(nx * nv), y1(nx * nv), rhs_dir(nx * nv);

    const Real w = omega(type);
    for (std::size_t i = 0; i < nx; ++i) {
        const Real payoff = std::fmax(w * (spots[i] - strike), 0.0);
        for (std::size_t j = 0; j < nv; ++j) u[at(i, j)] = payoff;
    }

    // ---- operator application -------------------------------------------
    // A1 acts along x (log-spot), A2 along v, A0 is the mixed term. The -rV
    // source is split evenly between A1 and A2 so that each directional solve
    // carries half of it, which keeps both sub-problems diagonally dominant.
    auto apply_a1 = [&](const std::vector<Real>& in, std::vector<Real>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            const auto s = detail::central_stencil(x[i] - x[i - 1], x[i + 1] - x[i]);
            for (std::size_t j = 0; j < nv; ++j) {
                const Real half_v = 0.5 * v[j];
                const Real drift = rate - dividend - half_v;
                out[at(i, j)] = half_v * (s.d2_lo * in[at(i - 1, j)] + s.d2_mid * in[at(i, j)] +
                                          s.d2_hi * in[at(i + 1, j)]) +
                                drift * (s.d1_lo * in[at(i - 1, j)] + s.d1_mid * in[at(i, j)] +
                                         s.d1_hi * in[at(i + 1, j)]) -
                                0.5 * rate * in[at(i, j)];
            }
        }
    };

    auto apply_a2 = [&](const std::vector<Real>& in, std::vector<Real>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (std::size_t i = 0; i < nx; ++i) {
            // v = 0: the degenerate transport equation, one-sided in v.
            {
                const Real h = v[1] - v[0];
                out[at(i, 0)] = p.kappa * p.theta * (in[at(i, 1)] - in[at(i, 0)]) / h -
                                0.5 * rate * in[at(i, 0)];
            }
            for (std::size_t j = 1; j + 1 < nv; ++j) {
                const auto s = detail::central_stencil(v[j] - v[j - 1], v[j + 1] - v[j]);
                const Real diff = 0.5 * p.sigma * p.sigma * v[j];
                const Real drift = p.kappa * (p.theta - v[j]);
                out[at(i, j)] = diff * (s.d2_lo * in[at(i, j - 1)] + s.d2_mid * in[at(i, j)] +
                                        s.d2_hi * in[at(i, j + 1)]) +
                                drift * (s.d1_lo * in[at(i, j - 1)] + s.d1_mid * in[at(i, j)] +
                                         s.d1_hi * in[at(i, j + 1)]) -
                                0.5 * rate * in[at(i, j)];
            }
            // v = v_max: dV/dv = 0.
            out[at(i, nv - 1)] = -0.5 * rate * in[at(i, nv - 1)];
        }
    };

    auto apply_a0 = [&](const std::vector<Real>& in, std::vector<Real>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            const Real hxm = x[i] - x[i - 1], hxp = x[i + 1] - x[i];
            for (std::size_t j = 1; j + 1 < nv; ++j) {
                const Real hvm = v[j] - v[j - 1], hvp = v[j + 1] - v[j];
                const auto sx = detail::central_stencil(hxm, hxp);
                const auto sv = detail::central_stencil(hvm, hvp);
                // Tensor product of the two first-derivative stencils.
                Real mixed = 0.0;
                const Real cx[3] = {sx.d1_lo, sx.d1_mid, sx.d1_hi};
                const Real cv[3] = {sv.d1_lo, sv.d1_mid, sv.d1_hi};
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        mixed += cx[a] * cv[b] * in[at(i + std::size_t(a) - 1, j + std::size_t(b) - 1)];
                    }
                }
                out[at(i, j)] = p.rho * p.sigma * v[j] * mixed;
            }
        }
    };

    // ---- boundaries ------------------------------------------------------
    auto set_boundaries = [&](std::vector<Real>& field, Real tau) {
        const Real dfr = std::exp(-rate * tau), dfq = std::exp(-dividend * tau);
        for (std::size_t j = 0; j < nv; ++j) {
            if (type == OptionType::Call) {
                field[at(0, j)] = 0.0;
                field[at(nx - 1, j)] = spots[nx - 1] * dfq - strike * dfr;
            } else {
                field[at(0, j)] = strike * dfr - spots[0] * dfq;
                field[at(nx - 1, j)] = 0.0;
            }
        }
    };

    // ---- time stepping ---------------------------------------------------
    const Real dt_full = expiry / Real(cfg.time_steps);
    struct Step { Real dt; Real theta; };
    std::vector<Step> schedule;
    for (int i = 0; i < cfg.rannacher_steps; ++i) {
        schedule.push_back({0.5 * dt_full, 1.0});
        schedule.push_back({0.5 * dt_full, 1.0});
    }
    for (int i = cfg.rannacher_steps; i < cfg.time_steps; ++i) {
        schedule.push_back({dt_full, cfg.theta});
    }

    std::vector<Real> a0(nx * nv), a1(nx * nv), a2(nx * nv);
    std::vector<Real> a0_corr(nx * nv), y0_hat(nx * nv), y2(nx * nv);
    std::vector<Real> sub, dia, sup, b, sol, work;
    Real tau = 0.0;

    // One implicit sweep along x. `source` is the right-hand side before the
    // -theta dt A1 V correction, which is supplied separately because both the
    // Douglas and the Craig-Sneyd stages subtract the same term.
    auto solve_x = [&](const std::vector<Real>& source, const std::vector<Real>& a1_old,
                       Real theta_dt, Real tau_new, std::vector<Real>& out) {
        const std::size_t mx = nx - 2;
        sub.assign(mx, 0.0); dia.assign(mx, 0.0); sup.assign(mx, 0.0); b.assign(mx, 0.0);
        const Real dfr = std::exp(-rate * tau_new), dfq = std::exp(-dividend * tau_new);
        const Real lo_val = (type == OptionType::Call) ? 0.0 : strike * dfr - spots[0] * dfq;
        const Real hi_val = (type == OptionType::Call) ? spots[nx - 1] * dfq - strike * dfr : 0.0;

        for (std::size_t j = 0; j < nv; ++j) {
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const auto s = detail::central_stencil(x[i] - x[i - 1], x[i + 1] - x[i]);
                const Real half_v = 0.5 * v[j];
                const Real drift = rate - dividend - half_v;
                const std::size_t r = i - 1;
                sub[r] = -theta_dt * (half_v * s.d2_lo + drift * s.d1_lo);
                dia[r] = 1.0 - theta_dt * (half_v * s.d2_mid + drift * s.d1_mid - 0.5 * rate);
                sup[r] = -theta_dt * (half_v * s.d2_hi + drift * s.d1_hi);
                b[r] = source[at(i, j)] - theta_dt * a1_old[at(i, j)];
            }
            b[0] -= sub[0] * lo_val;
            b[mx - 1] -= sup[mx - 1] * hi_val;
            thomas_solve(sub, dia, sup, b, sol, work);
            out[at(0, j)] = lo_val;
            out[at(nx - 1, j)] = hi_val;
            for (std::size_t r = 0; r < mx; ++r) out[at(r + 1, j)] = sol[r];
        }
    };

    // One implicit sweep along v. The v = 0 row and the v = v_max row are rows
    // of the system, not boundary values: the first because the characteristic
    // points into the domain, the second because dV/dv = 0 is a Neumann
    // condition.
    auto solve_v = [&](const std::vector<Real>& source, const std::vector<Real>& a2_old,
                       Real theta_dt, std::vector<Real>& out) {
        sub.assign(nv, 0.0); dia.assign(nv, 0.0); sup.assign(nv, 0.0); b.assign(nv, 0.0);
        for (std::size_t i = 0; i < nx; ++i) {
            {
                const Real h = v[1] - v[0];
                const Real coef = p.kappa * p.theta / h;
                sub[0] = 0.0;
                dia[0] = 1.0 - theta_dt * (-coef - 0.5 * rate);
                sup[0] = -theta_dt * coef;
                b[0] = source[at(i, 0)] - theta_dt * a2_old[at(i, 0)];
            }
            for (std::size_t j = 1; j + 1 < nv; ++j) {
                const auto s = detail::central_stencil(v[j] - v[j - 1], v[j + 1] - v[j]);
                const Real diff = 0.5 * p.sigma * p.sigma * v[j];
                const Real drift = p.kappa * (p.theta - v[j]);
                sub[j] = -theta_dt * (diff * s.d2_lo + drift * s.d1_lo);
                dia[j] = 1.0 - theta_dt * (diff * s.d2_mid + drift * s.d1_mid - 0.5 * rate);
                sup[j] = -theta_dt * (diff * s.d2_hi + drift * s.d1_hi);
                b[j] = source[at(i, j)] - theta_dt * a2_old[at(i, j)];
            }
            sub[nv - 1] = -1.0;
            dia[nv - 1] = 1.0;
            sup[nv - 1] = 0.0;
            b[nv - 1] = 0.0;

            thomas_solve(sub, dia, sup, b, sol, work);
            for (std::size_t j = 0; j < nv; ++j) out[at(i, j)] = sol[j];
        }
    };

    for (const Step& step : schedule) {
        tau += step.dt;
        set_boundaries(u, tau - step.dt);

        apply_a0(u, a0);
        apply_a1(u, a1);
        apply_a2(u, a2);
        for (std::size_t k = 0; k < u.size(); ++k) {
            y0[k] = u[k] + step.dt * (a0[k] + a1[k] + a2[k]);
        }

        const Real theta_dt = step.theta * step.dt;
        solve_x(y0, a1, theta_dt, tau, y1);
        solve_v(y1, a2, theta_dt, y2);

        if (cfg.craig_sneyd && step.theta < 1.0) {
            // Correct the mixed term using the Douglas answer, then repeat the
            // two directional solves. Skipped during the Rannacher phase, where
            // the whole point is first-order damping.
            apply_a0(y2, a0_corr);
            for (std::size_t k = 0; k < u.size(); ++k) {
                y0_hat[k] = y0[k] + 0.5 * step.dt * (a0_corr[k] - a0[k]);
            }
            solve_x(y0_hat, a1, theta_dt, tau, y1);
            solve_v(y1, a2, theta_dt, y2);
        }

        u.swap(y2);
        set_boundaries(u, tau);
    }

    // ---- read off the answer at (spot, v0) --------------------------------
    // Quadratic in both directions. Linear interpolation in v looks harmless --
    // the grid is fine near v0 -- but the value is convex in variance, so a
    // linear read-off carries a one-sided O(h^2 V_vv) bias that does not average
    // out and does not shrink when the other two dimensions are refined.
    std::size_t jv = 1;
    while (jv + 2 < nv && v[jv + 1] < p.v0) ++jv;
    jv = std::min(jv, nv - 3);
    const Real u0 = v[jv], u1 = v[jv + 1], u2 = v[jv + 2];
    const Real e01 = u0 - u1, e02 = u0 - u2, e12 = u1 - u2;
    const Real wv0 = (p.v0 - u1) * (p.v0 - u2) / (e01 * e02);
    const Real wv1 = (p.v0 - u0) * (p.v0 - u2) / (-e01 * e12);
    const Real wv2 = (p.v0 - u0) * (p.v0 - u1) / (e02 * e12);

    std::size_t i0 = 1;
    while (i0 + 2 < nx && spots[i0 + 1] < spot) ++i0;
    i0 = std::min(i0, nx - 3);

    auto value_at = [&](std::size_t i) {
        return wv0 * u[at(i, jv)] + wv1 * u[at(i, jv + 1)] + wv2 * u[at(i, jv + 2)];
    };
    const Real s0 = spots[i0], s1 = spots[i0 + 1], s2 = spots[i0 + 2];
    const Real v0v = value_at(i0), v1v = value_at(i0 + 1), v2v = value_at(i0 + 2);
    const Real d01 = s0 - s1, d02 = s0 - s2, d12 = s1 - s2;

    HestonPDEResult out;
    out.price = v0v * (spot - s1) * (spot - s2) / (d01 * d02) +
                v1v * (spot - s0) * (spot - s2) / (-d01 * d12) +
                v2v * (spot - s0) * (spot - s1) / (d02 * d12);
    out.delta = v0v * (2.0 * spot - s1 - s2) / (d01 * d02) +
                v1v * (2.0 * spot - s0 - s2) / (-d01 * d12) +
                v2v * (2.0 * spot - s0 - s1) / (d02 * d12);
    out.gamma = 2.0 * (v0v / (d01 * d02) - v1v / (d01 * d12) + v2v / (d02 * d12));
    out.spot_nodes = nx;
    out.var_nodes = nv;
    return out;
}

}  // namespace vse
