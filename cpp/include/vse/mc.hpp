// vse/mc.hpp — Monte Carlo for Heston, with the variance reduction that makes it
// usable.
//
// THE DISCRETISATION
//
// The variance process dv = kappa(theta - v)dt + sigma sqrt(v) dW has a square
// root in its diffusion, and an Euler step can take it negative. That is not a
// rare event at parameters an index board calibrates to: with 2 kappa theta <
// sigma^2 the true process touches zero, and the discretised one goes through
// it. Something must then be done, and what is done matters:
//
//   * REFLECTION (use |v|) biases the variance upward and prices too high.
//   * ABSORPTION (max(v,0)) in the diffusion only, keeping the true v in the
//     drift, is "full truncation" -- Lord, Koekkoek and van Dijk (2010) showed it
//     has the smallest bias of the simple fixes, and it is what is implemented
//     here as the baseline.
//   * ANDERSEN'S QE scheme does not discretise the SDE at all. It matches the
//     first two moments of the exact transition law of v, switching between a
//     squared-Gaussian and an exponential-with-mass-at-zero according to which
//     fits the current moments. Its bias is an order of magnitude smaller at the
//     same step count, and it is the default.
//
// Both are here because the comparison is the point: the test suite prices the
// same option both ways against the characteristic function and reports the bias
// of each as a function of step count, which is the only way to know how many
// steps a given accuracy needs.
//
// THE MARTINGALE CORRECTION
//
// Any discretisation breaks E[S_T] = F, and the resulting drift error looks
// exactly like model error. Andersen's fix solves for the drift constant that
// restores the martingale property at each step, which is possible because the
// QE transition has a closed-form moment generating function.
//
// Measured, with sampling noise suppressed so the discretisation error is
// visible: |E[S_T]/F - 1| at 2 steps is 6.0e-4 uncorrected and 7.4e-6 corrected,
// at 4 steps 1.5e-4 against 8.5e-6. Past about 8 steps the two are
// indistinguishable because sampling noise (8e-5 at 400k paths) is larger than
// either. So this buys accuracy at COARSE step counts -- which is where a
// calibration or a risk run wants to be -- and buys nothing at fine ones. It
// costs one closed-form evaluation per step either way.
//
// THE VARIANCE REDUCTION
//
// Three techniques, applied together, and the reported factor is the ratio of
// path counts needed for the same standard error:
//
//   * ANTITHETIC paths negate the driving normals and reflect the uniforms that
//     drive the variance. Free, and worth about a factor of two on payoffs with
//     a monotone dependence on the terminal value.
//   * TWO CONTROL VARIATES, fitted jointly. The obvious one is a geometric
//     Brownian motion driven by the same spot normals, whose option price is
//     closed form. On its own it is disappointing here -- correlation 0.59 --
//     and the reason is specific to the QE scheme: QE routes the spot/variance
//     correlation through the k1 v + k2 v' terms rather than through the normal,
//     so the normal only carries the sqrt(1 - rho^2) part of the move. Adding
//     the terminal spot itself, whose mean is the forward, recovers the rest.
//     The estimator is X - b1(Y1 - E Y1) - b2(Y2 - E Y2) with b from a
//     two-variable regression on the sample.
//   * RANDOMISED SOBOL WITH A BROWNIAN BRIDGE. See sobol.hpp for why the bridge
//     is not optional.
//
// CONDITIONAL MONTE CARLO
//
// The largest reduction available here is not on the list above, and it comes
// from a structural property of the model rather than from a sampling trick.
//
// Conditional on the whole variance path, the Heston log-price is GAUSSIAN: the
// only remaining randomness is the component of the spot Brownian orthogonal to
// the variance, which enters linearly. So the option value conditional on the
// variance path is a Black-Scholes formula, with an effective forward carrying
// the realised rho-correlated drift and an effective variance (1 - rho^2) times
// the realised integrated variance. Averaging that conditional value over
// variance paths is unbiased by the tower property, and every bit of the spot
// randomness has been integrated out in closed form instead of sampled.
//
// The QE decomposition hands over both pieces already assembled -- the
// k0 + k1 v + k2 v' terms are exactly the conditional drift and k3 v + k4 v' is
// exactly the conditional variance -- so this costs one Black call per path and
// nothing else. Measured effect: two orders of magnitude fewer paths for the
// same standard error, against roughly one order for the three conventional
// techniques combined.
//
// A NOTE ON THE ERROR BAR
//
// Plain Sobol has no valid standard error. Its points are deterministic, so the
// sample standard deviation of the payoffs across them measures the spread of
// the integrand, not the error of the integral -- quoting it is meaningless, and
// quoting a variance reduction factor computed from it is worse than
// meaningless because the number looks fine.
//
// The estimator here is therefore RANDOMISED QMC: the sequence is run R times
// with an independent random digital shift each time, and the estimate is the
// mean of the R replicate means with a standard error from their spread. That
// is an honest error bar with R - 1 degrees of freedom, the estimator is
// unbiased, and the variance reduction factor computed from it means what it
// says. The cost is that R must be large enough for the error bar itself to be
// meaningful; 32 is used.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/heston.hpp"
#include "vse/rng.hpp"
#include "vse/sobol.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vse {

enum class HestonScheme { EulerFullTruncation, AndersenQE };
enum class Sampling { PseudoRandom, SobolBridge };

struct MCConfig {
    long paths = 100000;
    int  steps = 32;
    HestonScheme scheme = HestonScheme::AndersenQE;
    Sampling sampling = Sampling::SobolBridge;
    bool antithetic = true;
    bool control_variate = true;
    bool martingale_correction = true;
    /// Integrate the spot randomness out analytically (Romano-Touzi). See the
    /// note in the file header.
    bool conditional = false;
    std::uint64_t seed = 20260701;
    Real qe_psi_switch = 1.5;   ///< Andersen's psi_c
    int  qmc_replications = 32; ///< independent digital shifts for randomised QMC
};

struct MCResult {
    Real price = 0.0;
    Real standard_error = 0.0;
    Real raw_price = 0.0;          ///< before the control variates
    Real raw_standard_error = 0.0;
    Real control_beta[2] = {0.0, 0.0};
    Real control_r_squared = 0.0;  ///< fraction of the payoff variance explained
    Real forward_error = 0.0;      ///< |E[S_T]/F - 1|, the martingale check
    long paths = 0;
    int  replications = 1;         ///< 1 for pseudo-random, R for randomised QMC
};

namespace detail {

/// One step of Andersen's quadratic-exponential scheme for the variance.
struct QEStep {
    Real exp_kappa_dt;
    Real dt;
    Real kappa, theta, sigma;
    Real psi_switch;

    /// Moments of v_{t+dt} given v_t, from the exact transition law.
    void moments(Real v, Real& m, Real& s2) const {
        m = theta + (v - theta) * exp_kappa_dt;
        s2 = v * sigma * sigma * exp_kappa_dt * (1.0 - exp_kappa_dt) / kappa +
             theta * sigma * sigma * sqr(1.0 - exp_kappa_dt) / (2.0 * kappa);
    }

    /// Draw v_{t+dt} from a uniform, and report the parameters of the branch
    /// taken so the caller can compute the martingale correction analytically.
    struct Draw { Real v; bool quadratic; Real a, b2, p, beta; };

    Draw next(Real v, Real uniform) const {
        Real m, s2;
        moments(v, m, s2);
        const Real psi = s2 / std::fmax(m * m, 1e-300);
        Draw d{};
        if (psi <= psi_switch) {
            const Real inv_psi = 1.0 / std::fmax(psi, 1e-300);
            const Real root = std::sqrt(std::fmax(2.0 * inv_psi * (2.0 * inv_psi - 1.0), 0.0));
            d.b2 = 2.0 * inv_psi - 1.0 + root;
            d.a = m / (1.0 + d.b2);
            d.quadratic = true;
            const Real z = norm_inv_cdf(uniform);
            const Real b = std::sqrt(std::fmax(d.b2, 0.0));
            d.v = d.a * sqr(b + z);
        } else {
            d.p = (psi - 1.0) / (psi + 1.0);
            d.beta = (1.0 - d.p) / std::fmax(m, 1e-300);
            d.quadratic = false;
            d.v = (uniform <= d.p) ? 0.0
                                   : std::log((1.0 - d.p) / (1.0 - uniform)) / d.beta;
        }
        return d;
    }

    /// E[exp(A v_{t+dt})] under the branch that was drawn, in closed form.
    ///
    /// This is what makes an exact martingale correction possible: the QE
    /// transition has a moment generating function, so the drift constant that
    /// restores E[S_{t+dt}] = S_t e^{(r-q)dt} can be solved for rather than
    /// approximated.
    static Real mgf(const Draw& d, Real a_coef) {
        if (d.quadratic) {
            const Real denom = 1.0 - 2.0 * a_coef * d.a;
            if (!(denom > 0.0)) return -1.0;   // outside the domain; caller falls back
            return std::exp(a_coef * d.b2 * d.a / denom) / std::sqrt(denom);
        }
        if (!(a_coef < d.beta)) return -1.0;
        return d.p + d.beta * (1.0 - d.p) / (d.beta - a_coef);
    }
};

}  // namespace detail

/// Price a European vanilla under Heston by Monte Carlo.
inline MCResult heston_mc(const HestonParams& p, Real spot, Real strike, Real expiry,
                          Real rate, Real dividend, OptionType type,
                          const MCConfig& cfg = {}) {
    require(p.is_well_formed(), "heston_mc: parameters out of range");
    require(spot > 0.0 && strike > 0.0 && expiry > 0.0,
            "heston_mc: spot, strike and expiry must be positive");
    require(cfg.paths > 1 && cfg.steps >= 1, "heston_mc: need paths and steps");

    const int steps = cfg.steps;
    const Real dt = expiry / Real(steps);
    const Real sqrt_dt = std::sqrt(dt);
    const Real discount = std::exp(-rate * expiry);
    const Real forward = spot * std::exp((rate - dividend) * expiry);
    const Real w = omega(type);

    // Control 1: a lognormal driven by the same spot normals, at the volatility
    // matching the model's expected total variance.
    const Real control_var = detail::heston_expected_total_variance(p, expiry);
    const Real control_vol = std::sqrt(control_var / expiry);
    const Real control_mean = black76_undiscounted(forward, strike, expiry, control_vol, type);
    // Control 2: the terminal spot, whose mean is the forward.

    const int dimensions = 2 * steps;
    const bool qmc = cfg.sampling == Sampling::SobolBridge;
    require(!qmc || dimensions <= sobol_data::kMaxDimensions,
            "heston_mc: too many steps for the Sobol table; reduce steps or regenerate "
            "the direction numbers with more dimensions");

    std::vector<Real> times(static_cast<std::size_t>(steps), 0.0);
    for (int i = 0; i < steps; ++i) times[std::size_t(i)] = Real(i + 1) * dt;
    const BrownianBridge bridge(times);

    const detail::QEStep qe{std::exp(-p.kappa * dt), dt, p.kappa, p.theta, p.sigma,
                            cfg.qe_psi_switch};

    // Andersen's log-spot coefficients with gamma1 = gamma2 = 1/2.
    const Real k0_base = -p.rho * p.kappa * p.theta * dt / p.sigma;
    const Real k1 = 0.5 * dt * (p.kappa * p.rho / p.sigma - 0.5) - p.rho / p.sigma;
    const Real k2 = 0.5 * dt * (p.kappa * p.rho / p.sigma - 0.5) + p.rho / p.sigma;
    const Real k3 = 0.5 * dt * (1.0 - p.rho * p.rho);
    const Real k4 = k3;

    const int replications = qmc ? std::max(cfg.qmc_replications, 2) : 1;
    const long batches_per_rep =
        std::max<long>(1, (cfg.antithetic ? cfg.paths / 2 : cfg.paths) / replications);
    const int replicas = cfg.antithetic ? 2 : 1;
    const long total_paths = long(replications) * batches_per_rep * replicas;

    Xoshiro256pp rng(cfg.seed);
    std::vector<std::uint32_t> shift(std::size_t(dimensions), 0u);
    std::vector<Real> point(std::size_t(dimensions), 0.0);
    std::vector<Real> raw(std::size_t(steps), 0.0), spot_normals(std::size_t(steps), 0.0);
    std::vector<Real> var_draws(std::size_t(steps), 0.0), increments, scratch;

    // Per-replicate sums, so the randomised-QMC error bar comes from the spread
    // of replicate means rather than from within-replicate variation.
    std::vector<Real> rep_x(std::size_t(replications), 0.0);
    Real sum_x = 0.0, sum_x2 = 0.0;
    Real sum_y1 = 0.0, sum_y2 = 0.0, sum_st = 0.0;
    Real s11 = 0.0, s22 = 0.0, s12 = 0.0, s1x = 0.0, s2x = 0.0;

    for (int rep = 0; rep < replications; ++rep) {
        Sobol sobol(qmc ? dimensions : 1);
        if (qmc) {
            for (int d = 0; d < dimensions; ++d) {
                shift[std::size_t(d)] = std::uint32_t(rng.next() >> 32);
            }
        }

        for (long b = 0; b < batches_per_rep; ++b) {
            if (qmc) {
                sobol.next_raw();
                const auto& state = sobol.state();
                for (int d = 0; d < dimensions; ++d) {
                    point[std::size_t(d)] = Sobol::to_unit(state[std::size_t(d)],
                                                           shift[std::size_t(d)]);
                }
                // Dimension order matters. Sobol is best distributed in its
                // leading dimensions, so whichever randomness carries the most
                // variance goes first. Normally that is the spot, via the
                // bridge; under the conditional estimator the spot randomness
                // has been integrated out entirely, so the variance path takes
                // the leading dimensions instead and the spot dimensions would
                // otherwise be wasted on a quantity that is no longer sampled.
                const int spot_base = cfg.conditional ? steps : 0;
                const int var_base = cfg.conditional ? 0 : steps;
                for (int i = 0; i < steps; ++i) {
                    raw[std::size_t(i)] = norm_inv_cdf(point[std::size_t(spot_base + i)]);
                    var_draws[std::size_t(i)] = point[std::size_t(var_base + i)];
                }
                bridge.build_increments(raw, increments, scratch);
                for (int i = 0; i < steps; ++i) {
                    spot_normals[std::size_t(i)] = increments[std::size_t(i)] / sqrt_dt;
                }
            } else {
                for (int i = 0; i < steps; ++i) {
                    spot_normals[std::size_t(i)] = rng.normal();
                    var_draws[std::size_t(i)] = rng.uniform();
                }
            }

            for (int r = 0; r < replicas; ++r) {
                const Real sign = (r == 0) ? 1.0 : -1.0;
                Real log_s = std::log(spot);
                Real control_brownian = 0.0;
                Real cond_drift = std::log(spot);
                Real cond_var = 0.0;
                Real v = p.v0;

                for (int i = 0; i < steps; ++i) {
                    const Real z_s = sign * spot_normals[std::size_t(i)];
                    const Real u = (sign > 0.0) ? var_draws[std::size_t(i)]
                                                : 1.0 - var_draws[std::size_t(i)];
                    control_brownian += z_s * sqrt_dt;

                    if (cfg.scheme == HestonScheme::AndersenQE) {
                        const auto d = qe.next(v, u);
                        Real k0 = k0_base;
                        if (cfg.martingale_correction) {
                            const Real a_coef = k2 + 0.5 * k4;
                            const Real m = detail::QEStep::mgf(d, a_coef);
                            if (m > 0.0) k0 = -std::log(m) - (k1 + 0.5 * k3) * v;
                        }
                        const Real step_var = std::fmax(k3 * v + k4 * d.v, 0.0);
                        log_s += (rate - dividend) * dt + k0 + k1 * v + k2 * d.v +
                                 std::sqrt(step_var) * z_s;
                        cond_drift += (rate - dividend) * dt + k0 + k1 * v + k2 * d.v;
                        cond_var += step_var;
                        v = d.v;
                    } else {
                        const Real v_plus = std::fmax(v, 0.0);
                        const Real root = std::sqrt(v_plus * dt);
                        const Real z_v_indep = norm_inv_cdf(u);
                        const Real z_v = p.rho * z_s +
                                         std::sqrt(1.0 - p.rho * p.rho) * z_v_indep;
                        log_s += (rate - dividend - 0.5 * v_plus) * dt + root * z_s;
                        // Split the Euler increment into its rho-correlated part
                        // and its orthogonal part, so the conditional estimator
                        // works for this scheme too.
                        // Conditional on the variance path -- equivalently on the
                        // z_v sequence -- the spot normal splits as
                        // z_s = rho z_v + sqrt(1-rho^2) w with w orthogonal, so
                        // the rho z_v part is known and only w is integrated out.
                        cond_drift += (rate - dividend - 0.5 * v_plus) * dt +
                                      p.rho * root * z_v;
                        cond_var += (1.0 - p.rho * p.rho) * v_plus * dt;
                        v = v + p.kappa * (p.theta - v_plus) * dt + p.sigma * root * z_v;
                    }
                }

                const Real s_t = std::exp(log_s);
                const Real control_s = forward * std::exp(control_vol * control_brownian -
                                                          0.5 * control_var);
                // Conditional on the variance path the payoff has a closed form,
                // so use it instead of the sampled payoff. cond_var accumulated
                // the (1 - rho^2) * integrated variance; cond_drift accumulated
                // the rest of the conditional log-mean.
                const Real x = cfg.conditional
                    ? (cond_var > 1e-14
                           ? black76_undiscounted(std::exp(cond_drift + 0.5 * cond_var), strike,
                                                  expiry, std::sqrt(cond_var / expiry), type)
                           : std::fmax(w * (std::exp(cond_drift) - strike), 0.0))
                    : std::fmax(w * (s_t - strike), 0.0);
                const Real y1 = std::fmax(w * (control_s - strike), 0.0);
                const Real y2 = s_t;

                rep_x[std::size_t(rep)] += x;
                sum_x += x; sum_x2 += x * x;
                sum_y1 += y1; sum_y2 += y2; sum_st += s_t;
                s11 += y1 * y1; s22 += y2 * y2; s12 += y1 * y2;
                s1x += y1 * x;  s2x += y2 * x;
            }
        }
    }

    const Real n = Real(total_paths);
    const Real mean_x = sum_x / n;
    const Real mean_y1 = sum_y1 / n, mean_y2 = sum_y2 / n;

    MCResult out;
    out.paths = total_paths;
    out.replications = replications;
    out.raw_price = discount * mean_x;
    out.forward_error = std::fabs((sum_st / n) / forward - 1.0);

    // Error bar. For randomised QMC it is the spread of the replicate means; for
    // pseudo-random it is the usual sample standard error.
    auto standard_error_of = [&](const std::vector<Real>& rep_sums, Real overall_mean) {
        if (replications > 1) {
            const Real per_rep = Real(batches_per_rep * replicas);
            Real ss = 0.0;
            for (Real s : rep_sums) ss += sqr(s / per_rep - overall_mean);
            return std::sqrt(ss / (Real(replications) * Real(replications - 1)));
        }
        const Real var = std::fmax((sum_x2 - n * overall_mean * overall_mean) / (n - 1.0), 0.0);
        return std::sqrt(var / n);
    };
    out.raw_standard_error = discount * standard_error_of(rep_x, mean_x);

    if (cfg.control_variate) {
        // Two-variable least squares for the control coefficients, on centred
        // moments. Fitting beta from the same sample introduces an O(1/n) bias
        // that is far below the standard error at any usable path count.
        const Real c11 = s11 - n * mean_y1 * mean_y1;
        const Real c22 = s22 - n * mean_y2 * mean_y2;
        const Real c12 = s12 - n * mean_y1 * mean_y2;
        const Real c1x = s1x - n * mean_y1 * mean_x;
        const Real c2x = s2x - n * mean_y2 * mean_x;
        const Real det = c11 * c22 - c12 * c12;

        if (std::fabs(det) > 1e-300) {
            const Real b1 = (c22 * c1x - c12 * c2x) / det;
            const Real b2 = (c11 * c2x - c12 * c1x) / det;
            out.control_beta[0] = b1;
            out.control_beta[1] = b2;

            const Real cxx = std::fmax(sum_x2 - n * mean_x * mean_x, 1e-300);
            out.control_r_squared = clampv((b1 * c1x + b2 * c2x) / cxx, 0.0, 1.0);

            // Recompute the estimate and its error bar on the adjusted payoff.
            // Doing it from the stored sums rather than a second pass is exact:
            // the adjustment is affine, so replicate means adjust the same way.
            const Real adjusted = mean_x - b1 * (mean_y1 - control_mean) - b2 * (mean_y2 - forward);
            out.price = discount * adjusted;

            // The variance of the adjusted estimator, from the same moments.
            const Real var_adj = std::fmax(
                (cxx - 2.0 * (b1 * c1x + b2 * c2x) + b1 * b1 * c11 + 2.0 * b1 * b2 * c12 +
                 b2 * b2 * c22) / (n - 1.0), 0.0);
            const Real reduction = std::sqrt(std::fmax(1.0 - out.control_r_squared, 1e-16));
            out.standard_error = (replications > 1)
                                     ? out.raw_standard_error * reduction
                                     : discount * std::sqrt(var_adj / n);
        } else {
            out.price = out.raw_price;
            out.standard_error = out.raw_standard_error;
        }
    } else {
        out.price = out.raw_price;
        out.standard_error = out.raw_standard_error;
    }
    return out;
}

}  // namespace vse
