// vse/pde.hpp — finite-difference pricing on a non-uniform grid.
//
// THE OSCILLATION
//
// Crank-Nicolson is second-order accurate in time and unconditionally stable,
// which is why it is the default choice, and it is the wrong choice for an
// option payoff on its own. Stability is not the same as damping. The CN
// amplification factor is (1 - z/2)/(1 + z/2), whose modulus is below one for
// every z > 0 but which tends to MINUS one for large z. A payoff has a kink (a
// call) or a jump in its derivative (a digital), so its high-frequency Fourier
// content is large, and those are exactly the modes with large z. They do not
// grow, but they alternate in sign and decay slowly, and the result is a
// solution that ripples near the strike for many timesteps.
//
// The value itself absorbs this. The Greeks do not: gamma is a second difference
// of the solution, which amplifies precisely the oscillating modes, so a
// CN gamma near expiry is visibly wrong even when the price looks fine. That is
// the failure this file exists to demonstrate and then fix.
//
// Rannacher's remedy is to start with a few fully implicit steps. Implicit
// Euler's amplification factor is 1/(1+z), which tends to zero for large z, so
// two half-steps of it annihilate the high-frequency content before CN ever
// sees it. Second-order accuracy is preserved because the damping is applied
// over an O(dt) portion of the time axis. Two implicit half-steps is the
// standard prescription and is what this uses; the test suite measures the
// gamma oscillation with and without it.
//
// THE GRID
//
// Uniform grids waste points. The payoff kink is at the strike and that is where
// resolution is needed, so the grid is a Tavella-Randall sinh transform that
// concentrates points there and stretches them into the wings. It also places a
// grid point exactly ON the strike: with a kink sitting between two nodes, the
// second-order convergence of the whole scheme degrades to first order, which is
// a well-known and easily missed halving of the convergence rate.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/linalg.hpp"

#include <algorithm>
#include <vector>

namespace vse {

enum class Exercise { European, American };

struct PDEConfig {
    int  space_steps = 400;
    int  time_steps  = 200;
    Real grid_width_sd = 6.0;    ///< half-width of the grid in standard deviations
    Real concentration = 0.15;   ///< Tavella-Randall alpha, as a fraction of the width
    int  rannacher_steps = 2;    ///< fully implicit half-steps before Crank-Nicolson
    Real theta = 0.5;            ///< 0.5 is Crank-Nicolson, 1.0 fully implicit
    Real psor_omega = 1.35;      ///< over-relaxation factor for the American solve
    int  psor_max_iterations = 5000;
    Real psor_tolerance = 1e-11;
    bool use_brennan_schwartz = false;  ///< cheaper American solve; see below
};

struct PDEResult {
    Real price = 0.0;
    Real delta = 0.0;
    Real gamma = 0.0;
    Real theta_per_year = 0.0;
    int  psor_iterations = 0;    ///< total across all timesteps
    std::vector<Real> spots;     ///< the grid, for plotting or diagnostics
    std::vector<Real> values;
};

namespace detail {

/// Tavella-Randall grid in log-spot, concentrated at `focus` and containing it
/// exactly.
///
/// The exactness is the point. A payoff kink between two nodes drops the scheme
/// from second-order to first-order convergence in space, and the symptom --
/// twice as many points needed for the same accuracy -- looks like the scheme
/// simply being expensive rather than like a bug.
inline std::vector<Real> sinh_grid(Real x_min, Real x_max, Real focus, Real alpha, int n) {
    require(n >= 3, "sinh_grid: need at least three points");
    require(x_min < focus && focus < x_max, "sinh_grid: focus must lie inside the grid");

    const Real c1 = std::asinh((x_min - focus) / alpha);
    const Real c2 = std::asinh((x_max - focus) / alpha);

    std::vector<Real> x(std::size_t(n) + 1);
    for (int i = 0; i <= n; ++i) {
        const Real u = Real(i) / Real(n);
        x[std::size_t(i)] = focus + alpha * std::sinh(c1 + u * (c2 - c1));
    }
    x.front() = x_min;
    x.back() = x_max;

    // Snap the nearest node onto the focus. The shift is at most half a local
    // spacing, so the grid stays monotone and smoothly graded.
    std::size_t nearest = 0;
    Real best = DBL_HUGE;
    for (std::size_t i = 1; i + 1 < x.size(); ++i) {
        const Real d = std::fabs(x[i] - focus);
        if (d < best) { best = d; nearest = i; }
    }
    const Real shift = focus - x[nearest];
    for (std::size_t i = 1; i + 1 < x.size(); ++i) {
        // Taper the shift to zero at the boundaries so the ends stay put.
        const Real w = (i < nearest) ? Real(i) / Real(nearest)
                                     : Real(x.size() - 1 - i) / Real(x.size() - 1 - nearest);
        x[i] += shift * w;
    }
    x[nearest] = focus;
    return x;
}

/// Tridiagonal coefficients of the Black-Scholes operator in log-spot on a
/// non-uniform grid.
///
///     L V = 1/2 sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V
///
/// Second-order accurate central differences with unequal spacings.
struct Tridiagonal {
    std::vector<Real> lower, diag, upper;
};

inline Tridiagonal black_scholes_operator(const std::vector<Real>& x, Real rate,
                                          Real dividend, Real sigma) {
    const std::size_t n = x.size();
    Tridiagonal op{std::vector<Real>(n, 0.0), std::vector<Real>(n, 0.0),
                   std::vector<Real>(n, 0.0)};
    const Real half_var = 0.5 * sigma * sigma;
    const Real drift = rate - dividend - half_var;

    for (std::size_t i = 1; i + 1 < n; ++i) {
        const Real hm = x[i] - x[i - 1];
        const Real hp = x[i + 1] - x[i];
        const Real hs = hm + hp;

        const Real d2_lo = 2.0 / (hm * hs), d2_mid = -2.0 / (hm * hp), d2_hi = 2.0 / (hp * hs);
        const Real d1_lo = -hp / (hm * hs), d1_mid = (hp - hm) / (hm * hp), d1_hi = hm / (hp * hs);

        op.lower[i] = half_var * d2_lo + drift * d1_lo;
        op.diag[i]  = half_var * d2_mid + drift * d1_mid - rate;
        op.upper[i] = half_var * d2_hi + drift * d1_hi;
    }
    return op;
}

/// Projected SOR for the linear complementarity problem of an American option.
///
/// Solves min(A V - b, V - payoff) = 0 by relaxing towards the linear solution
/// and projecting onto V >= payoff after each update. Slow compared with a
/// direct solve, and correct for any shape of exercise region.
inline int psor_solve(const Tridiagonal& a, const std::vector<Real>& b,
                      const std::vector<Real>& payoff, std::vector<Real>& v, Real omega,
                      int max_iterations, Real tolerance) {
    const std::size_t n = v.size();
    for (int it = 1; it <= max_iterations; ++it) {
        Real change = 0.0;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const Real residual = b[i] - a.lower[i] * v[i - 1] - a.upper[i] * v[i + 1];
            const Real gauss_seidel = residual / a.diag[i];
            const Real relaxed = v[i] + omega * (gauss_seidel - v[i]);
            const Real projected = std::fmax(relaxed, payoff[i]);
            change = std::fmax(change, std::fabs(projected - v[i]));
            v[i] = projected;
        }
        if (change < tolerance) return it;
    }
    return max_iterations;
}

/// Brennan-Schwartz: one Thomas sweep with the projection folded into the back
/// substitution.
///
/// Direct, so a fixed cost per timestep instead of PSOR's iteration, and it is
/// exact -- but ONLY when the continuation region is connected and lies on one
/// known side of the exercise region. For a vanilla American put that holds: the
/// option is exercised for all spots below a single boundary. It does not hold
/// for a general payoff, and using it where it does not apply produces a wrong
/// answer silently, which is why it is not the default here.
inline void brennan_schwartz(const Tridiagonal& a, const std::vector<Real>& b,
                             const std::vector<Real>& payoff, std::vector<Real>& v,
                             bool exercise_at_low_spot) {
    const std::size_t n = v.size();
    std::vector<Real> c(n, 0.0), d(n, 0.0);

    if (exercise_at_low_spot) {
        // Sweep from high spot down, so the projection is applied last where the
        // exercise region is.
        c[n - 2] = a.lower[n - 2] / a.diag[n - 2];
        d[n - 2] = (b[n - 2] - a.upper[n - 2] * v[n - 1]) / a.diag[n - 2];
        for (std::size_t k = n - 3; k >= 1; --k) {
            const Real denom = a.diag[k] - a.upper[k] * c[k + 1];
            c[k] = a.lower[k] / denom;
            d[k] = (b[k] - a.upper[k] * d[k + 1]) / denom;
            if (k == 1) break;
        }
        v[1] = std::fmax(d[1], payoff[1]);
        for (std::size_t k = 2; k + 1 < n; ++k) {
            v[k] = std::fmax(d[k] - c[k] * v[k - 1], payoff[k]);
        }
    } else {
        c[1] = a.upper[1] / a.diag[1];
        d[1] = (b[1] - a.lower[1] * v[0]) / a.diag[1];
        for (std::size_t k = 2; k + 1 < n; ++k) {
            const Real denom = a.diag[k] - a.lower[k] * c[k - 1];
            c[k] = a.upper[k] / denom;
            d[k] = (b[k] - a.lower[k] * d[k - 1]) / denom;
        }
        v[n - 2] = std::fmax(d[n - 2], payoff[n - 2]);
        for (std::size_t k = n - 3; k >= 1; --k) {
            v[k] = std::fmax(d[k] - c[k] * v[k + 1], payoff[k]);
            if (k == 1) break;
        }
    }
}

}  // namespace detail

/// Price a vanilla option by finite differences.
inline PDEResult pde_vanilla(Real spot, Real strike, Real expiry, Real rate, Real dividend,
                             Real sigma, OptionType type, Exercise exercise = Exercise::European,
                             const PDEConfig& cfg = {}) {
    require(spot > 0.0 && strike > 0.0, "pde_vanilla: spot and strike must be positive");
    require(expiry > 0.0, "pde_vanilla: expiry must be positive");
    require(sigma > 0.0, "pde_vanilla: volatility must be positive");
    require(cfg.space_steps >= 8 && cfg.time_steps >= 1, "pde_vanilla: grid too coarse");

    const Real sd = sigma * std::sqrt(expiry);
    const Real x_strike = std::log(strike);
    const Real centre = std::log(spot) + (rate - dividend - 0.5 * sigma * sigma) * expiry;
    const Real width = cfg.grid_width_sd * sd + std::fabs(centre - x_strike);
    const Real x_min = std::fmin(centre, x_strike) - width;
    const Real x_max = std::fmax(centre, x_strike) + width;

    const std::vector<Real> x =
        detail::sinh_grid(x_min, x_max, x_strike, std::fmax(cfg.concentration * width, 1e-6),
                          cfg.space_steps);
    const std::size_t n = x.size();

    std::vector<Real> spots(n), payoff(n), v(n);
    for (std::size_t i = 0; i < n; ++i) {
        spots[i] = std::exp(x[i]);
        payoff[i] = std::fmax(omega(type) * (spots[i] - strike), 0.0);
        v[i] = payoff[i];
    }

    const auto op = detail::black_scholes_operator(x, rate, dividend, sigma);
    const Real dt_full = expiry / Real(cfg.time_steps);

    std::vector<Real> rhs(n, 0.0), work, solution;
    detail::Tridiagonal lhs{std::vector<Real>(n, 0.0), std::vector<Real>(n, 0.0),
                            std::vector<Real>(n, 0.0)};

    int psor_total = 0;
    Real tau = 0.0;

    // Rannacher: `rannacher_steps` fully implicit half-steps, then Crank-Nicolson.
    struct Step { Real dt; Real theta; };
    std::vector<Step> schedule;
    for (int i = 0; i < cfg.rannacher_steps; ++i) {
        schedule.push_back({0.5 * dt_full, 1.0});
        schedule.push_back({0.5 * dt_full, 1.0});
    }
    for (int i = cfg.rannacher_steps; i < cfg.time_steps; ++i) {
        schedule.push_back({dt_full, cfg.theta});
    }

    for (const Step& step : schedule) {
        const Real next_tau = tau + step.dt;

        // Boundary values at the new time level, from the exact European
        // boundary behaviour: worthless at one end, forward-like at the other.
        auto boundary = [&](Real s) {
            if (type == OptionType::Call) {
                const Real intrinsic = s * std::exp(-dividend * next_tau) -
                                       strike * std::exp(-rate * next_tau);
                return (exercise == Exercise::American) ? std::fmax(intrinsic, s - strike)
                                                        : std::fmax(intrinsic, 0.0);
            }
            const Real intrinsic = strike * std::exp(-rate * next_tau) -
                                   s * std::exp(-dividend * next_tau);
            return (exercise == Exercise::American) ? std::fmax(intrinsic, strike - s)
                                                    : std::fmax(intrinsic, 0.0);
        };
        const Real v_lo = (type == OptionType::Call) ? 0.0 : boundary(spots.front());
        const Real v_hi = (type == OptionType::Call) ? boundary(spots.back()) : 0.0;

        // rhs = (I + (1 - theta) dt L) v
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const Real explicit_part = op.lower[i] * v[i - 1] + op.diag[i] * v[i] +
                                       op.upper[i] * v[i + 1];
            rhs[i] = v[i] + (1.0 - step.theta) * step.dt * explicit_part;
        }
        // lhs = I - theta dt L
        for (std::size_t i = 1; i + 1 < n; ++i) {
            lhs.lower[i] = -step.theta * step.dt * op.lower[i];
            lhs.diag[i]  = 1.0 - step.theta * step.dt * op.diag[i];
            lhs.upper[i] = -step.theta * step.dt * op.upper[i];
        }
        // Dirichlet ends folded into the right-hand side.
        rhs[1] -= lhs.lower[1] * v_lo;
        rhs[n - 2] -= lhs.upper[n - 2] * v_hi;

        std::vector<Real> payoff_now = payoff;
        v[0] = v_lo;
        v[n - 1] = v_hi;

        if (exercise == Exercise::American) {
            if (cfg.use_brennan_schwartz) {
                detail::brennan_schwartz(lhs, rhs, payoff_now, v, type == OptionType::Put);
            } else {
                psor_total += detail::psor_solve(lhs, rhs, payoff_now, v, cfg.psor_omega,
                                                 cfg.psor_max_iterations, cfg.psor_tolerance);
            }
        } else {
            // Interior tridiagonal solve.
            const std::size_t m = n - 2;
            std::vector<Real> sub(m), dia(m), sup(m), b(m);
            for (std::size_t i = 0; i < m; ++i) {
                sub[i] = lhs.lower[i + 1];
                dia[i] = lhs.diag[i + 1];
                sup[i] = lhs.upper[i + 1];
                b[i] = rhs[i + 1];
            }
            thomas_solve(sub, dia, sup, b, solution, work);
            for (std::size_t i = 0; i < m; ++i) v[i + 1] = solution[i];
        }
        tau = next_tau;
    }

    // Interpolate the answer and its derivatives at the spot. Quadratic through
    // the three surrounding nodes: linear would give a delta that is piecewise
    // constant and a gamma that is zero.
    std::size_t i = 1;
    while (i + 2 < n && spots[i + 1] < spot) ++i;
    i = std::min(i, n - 3);

    const Real s0 = spots[i], s1 = spots[i + 1], s2 = spots[i + 2];
    const Real v0 = v[i], v1 = v[i + 1], v2 = v[i + 2];
    const Real d01 = s0 - s1, d02 = s0 - s2, d12 = s1 - s2;
    const Real l0 = (spot - s1) * (spot - s2) / (d01 * d02);
    const Real l1 = (spot - s0) * (spot - s2) / (-d01 * d12);
    const Real l2 = (spot - s0) * (spot - s1) / (d02 * d12);

    PDEResult out;
    out.price = l0 * v0 + l1 * v1 + l2 * v2;
    out.delta = v0 * (2.0 * spot - s1 - s2) / (d01 * d02) +
                v1 * (2.0 * spot - s0 - s2) / (-d01 * d12) +
                v2 * (2.0 * spot - s0 - s1) / (d02 * d12);
    out.gamma = 2.0 * (v0 / (d01 * d02) - v1 / (d01 * d12) + v2 / (d02 * d12));
    out.psor_iterations = psor_total;
    out.spots = std::move(spots);
    out.values = std::move(v);
    return out;
}

}  // namespace vse
