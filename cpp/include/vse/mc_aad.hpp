// vse/mc_aad.hpp — Heston Monte Carlo with pathwise adjoint Greeks.
//
// The whole risk ladder in the cost of a few pricings, regardless of how long
// the ladder is. That is the claim reverse-mode AD exists to support, and this
// is where it is measured.
//
// The comparison is against bump-and-revalue, which for n parameters needs
// 2n + 1 full Monte Carlo runs -- and not merely 2n + 1 times the work. Bumping
// has two further problems that the cost comparison understates:
//
//   * The bumped runs must share random numbers with the base run, or the Monte
//     Carlo noise swamps the difference. With common random numbers a delta
//     needs perhaps 1e4 paths; without them it needs 1e8.
//   * Even with common random numbers the step size is a compromise between
//     truncation error, which wants it small, and cancellation, which wants it
//     large. The best achievable accuracy is around 1e-8 relative, and there is
//     no way to know from the result which side of the compromise you are on.
//
// Adjoint differentiation has no step size and no noise beyond the Monte Carlo
// noise already in the price. Its derivatives are exact for the discretised
// model, which is the model actually being priced.
//
// TEN PARAMETERS
//
// spot, v0, kappa, theta, sigma, rho, rate, dividend, strike, expiry. The last
// two are unusual to differentiate and are included deliberately: dual delta and
// theta come out of the same sweep at no extra cost, which is exactly the point
// being demonstrated.
//
// WHICH SCHEME CAN BE DIFFERENTIATED
//
// Not the one that prices best. Pathwise differentiation needs the path value to
// be almost surely differentiable in the parameters, and Andersen QE is not:
//
//   * It switches between a squared-Gaussian and an exponential branch at
//     psi = 1.5, and the two do not agree at the switch, so the path value jumps
//     as a parameter moves the switching surface.
//   * The exponential branch places an ATOM at zero with probability p, and p
//     depends on the parameters. Moving a parameter moves probability mass
//     between {0} and the continuous part, and a pathwise derivative cannot see
//     that -- it differentiates within each branch and misses the flow between
//     them.
//
// Both effects are invisible in the price and appear in the variance Greeks:
// measured against a common-random-number bump, QE pathwise d/dv0 came out 36%
// low. Euler full truncation is a smooth function of every parameter -- max(v,0)
// is a kink, which is fine -- so it is what the adjoint pricer differentiates,
// and the tests check both schemes so the discrepancy is documented rather than
// discovered later.
//
// This is the usual trade in adjoint work: the discretisation is chosen for
// differentiability, not only for accuracy, and the choice has to be deliberate.
#pragma once

#include "vse/aad.hpp"
#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/heston.hpp"
#include "vse/mc.hpp"
#include "vse/rng.hpp"

#include <array>
#include <string>
#include <vector>

namespace vse {

inline constexpr int kHestonRiskFactors = 10;

/// Names in the order the gradient is returned.
inline const std::array<const char*, kHestonRiskFactors>& heston_risk_factor_names() {
    static const std::array<const char*, kHestonRiskFactors> names = {
        "spot", "v0", "kappa", "theta", "sigma", "rho", "rate", "dividend", "strike", "expiry"};
    return names;
}

struct MCGreeksResult {
    Real price = 0.0;
    Real standard_error = 0.0;
    std::array<Real, kHestonRiskFactors> gradient{};
    std::array<Real, kHestonRiskFactors> gradient_se{};
    long paths = 0;
    std::size_t tape_nodes_per_path = 0;
};

namespace detail {

/// One path of the QE scheme, written generically so it runs on Real for a plain
/// price and on ADouble for the adjoint sweep.
///
/// The random draws are inputs, not parameters: z and u are fixed numbers for a
/// given path, so the mapping from parameters to payoff is deterministic and
/// differentiable, which is what makes pathwise differentiation legitimate.
template <class T>
inline T heston_qe_path(const T& spot, const T& v0, const T& kappa, const T& theta,
                        const T& sigma, const T& rho, const T& rate, const T& dividend,
                        const T& strike, const T& expiry, OptionType type,
                        const std::vector<Real>& z, const std::vector<Real>& u,
                        int steps, Real psi_switch) {
    using std::exp;
    using std::log;
    using std::sqrt;

    const T dt = expiry / T(Real(steps));
    const T half = T(0.5);
    const T one = T(1.0);

    // Andersen's log-spot coefficients, recomputed inside so that their
    // dependence on the parameters is on the tape.
    const T k1 = half * dt * (kappa * rho / sigma - half) - rho / sigma;
    const T k2 = half * dt * (kappa * rho / sigma - half) + rho / sigma;
    const T k3 = half * dt * (one - rho * rho);
    const T k4 = k3;

    T log_s = log(spot);
    T v = v0;
    const T decay = exp(-(kappa * dt));

    for (int i = 0; i < steps; ++i) {
        const T m = theta + (v - theta) * decay;
        const T s2 = v * sigma * sigma * decay * (one - decay) / kappa +
                     theta * sigma * sigma * (one - decay) * (one - decay) / (T(2.0) * kappa);
        const T psi = s2 / (m * m);

        T v_next;
        // The branch is chosen by the VALUE, so the tape records whichever leg
        // was taken. The switching surface is measure-zero in parameter space,
        // which is the same argument that licenses the payoff kink.
        if (value_of(psi) <= psi_switch) {
            const T inv_psi = one / psi;
            const T b2 = T(2.0) * inv_psi - one +
                         sqrt(T(2.0) * inv_psi * (T(2.0) * inv_psi - one));
            const T a = m / (one + b2);
            const T b = sqrt(b2);
            const Real zv = norm_inv_cdf(u[std::size_t(i)]);
            v_next = a * (b + zv) * (b + zv);
        } else {
            const T p = (psi - one) / (psi + one);
            const T beta = (one - p) / m;
            if (u[std::size_t(i)] <= value_of(p)) {
                v_next = T(0.0);
            } else {
                v_next = log((one - p) / T(1.0 - u[std::size_t(i)])) / beta;
            }
        }

        const T k0 = -(rho * kappa * theta * dt / sigma);
        log_s = log_s + (rate - dividend) * dt + k0 + k1 * v + k2 * v_next +
                sqrt(k3 * v + k4 * v_next) * T(z[std::size_t(i)]);
        v = v_next;
    }

    const T terminal = exp(log_s);
    const T payoff = (type == OptionType::Call) ? (terminal - strike) : (strike - terminal);
    if constexpr (std::is_same_v<T, ADouble>) {
        return exp(-(rate * expiry)) * max_zero(payoff);
    } else {
        return std::exp(-(value_of(rate) * value_of(expiry))) * std::fmax(payoff, T(0.0));
    }
}


/// One path of Euler full truncation, written generically.
///
/// Every operation is a smooth function of the parameters except max(v, 0),
/// whose kink is hit with probability zero, so the pathwise derivative is
/// unbiased. That is why this and not QE is what gets differentiated.
template <class T>
inline T heston_euler_path(const T& spot, const T& v0, const T& kappa, const T& theta,
                           const T& sigma, const T& rho, const T& rate, const T& dividend,
                           const T& strike, const T& expiry, OptionType type,
                           const std::vector<Real>& z_spot,
                           const std::vector<Real>& z_var, int steps) {
    using std::exp;
    using std::log;
    using std::sqrt;

    const T dt = expiry / Real(steps);
    T log_s = log(spot);
    T v = v0;
    const Real root_corr = 0.0;   // placeholder; correlation applied below
    (void)root_corr;

    for (int i = 0; i < steps; ++i) {
        // max(v, 0) in the diffusion, v itself in the drift: full truncation.
        T v_plus = v;
        if (value_of(v) < 0.0) {
            if constexpr (std::is_same_v<T, ADouble>) {
                v_plus = max_zero(v);
            } else {
                v_plus = T(0.0);
            }
        }
        const T root = sqrt(v_plus * dt);
        const Real zs = z_spot[std::size_t(i)];
        const Real zi = z_var[std::size_t(i)];

        log_s = log_s + (rate - dividend) * dt - (v_plus * dt) * 0.5 + root * zs;
        // The variance normal, correlated with the spot normal.
        const T z_v = rho * zs + sqrt(1.0 - rho * rho) * zi;
        v = v + kappa * (theta - v_plus) * dt + sigma * root * z_v;
    }

    const T terminal = exp(log_s);
    const T payoff = (type == OptionType::Call) ? (terminal - strike) : (strike - terminal);
    if constexpr (std::is_same_v<T, ADouble>) {
        return exp(-(rate * expiry)) * max_zero(payoff);
    } else {
        return std::exp(-(value_of(rate) * value_of(expiry))) * std::fmax(payoff, 0.0);
    }
}

}  // namespace detail

/// Price and all ten first-order Greeks in one pass.
///
/// Uses Euler full truncation, for the differentiability reason set out in the
/// file header. `scheme` is honoured so the QE bias can be measured rather than
/// asserted, but Euler is the default and is what the reported numbers use.
inline MCGreeksResult heston_mc_greeks_aad(const HestonParams& p, Real spot, Real strike,
                                           Real expiry, Real rate, Real dividend,
                                           OptionType type, const MCConfig& cfg = {}) {
    require(p.is_well_formed(), "heston_mc_greeks_aad: parameters out of range");
    require(spot > 0.0 && strike > 0.0 && expiry > 0.0,
            "heston_mc_greeks_aad: spot, strike and expiry must be positive");
    require(cfg.paths > 1 && cfg.steps >= 1, "heston_mc_greeks_aad: need paths and steps");

    const int steps = cfg.steps;
    Xoshiro256pp rng(cfg.seed);
    std::vector<Real> z(std::size_t(steps), 0.0), u(std::size_t(steps), 0.0);

    MCGreeksResult out;
    std::array<Real, kHestonRiskFactors> sum_grad{}, sum_grad2{};
    Real sum = 0.0, sum2 = 0.0;

    Tape& tape = Tape::active();
    tape.clear();
    tape.reserve(std::size_t(steps) * 64 + 256);

    for (long path = 0; path < cfg.paths; ++path) {
        for (int i = 0; i < steps; ++i) {
            z[std::size_t(i)] = rng.normal();
            u[std::size_t(i)] = (cfg.scheme == HestonScheme::AndersenQE) ? rng.uniform()
                                                                        : rng.normal();
        }

        tape.clear();
        const ADouble a_spot(spot), a_v0(p.v0), a_kappa(p.kappa), a_theta(p.theta),
            a_sigma(p.sigma), a_rho(p.rho), a_rate(rate), a_dividend(dividend),
            a_strike(strike), a_expiry(expiry);

        const ADouble value =
            (cfg.scheme == HestonScheme::AndersenQE)
                ? detail::heston_qe_path<ADouble>(a_spot, a_v0, a_kappa, a_theta, a_sigma,
                                                  a_rho, a_rate, a_dividend, a_strike,
                                                  a_expiry, type, z, u, steps,
                                                  cfg.qe_psi_switch)
                : detail::heston_euler_path<ADouble>(a_spot, a_v0, a_kappa, a_theta, a_sigma,
                                                     a_rho, a_rate, a_dividend, a_strike,
                                                     a_expiry, type, z, u, steps);

        tape.backpropagate(value.index());
        if (path == 0) out.tape_nodes_per_path = tape.size();

        const std::array<Real, kHestonRiskFactors> g = {
            a_spot.adjoint(),   a_v0.adjoint(),       a_kappa.adjoint(), a_theta.adjoint(),
            a_sigma.adjoint(),  a_rho.adjoint(),      a_rate.adjoint(),  a_dividend.adjoint(),
            a_strike.adjoint(), a_expiry.adjoint()};

        sum += value.value();
        sum2 += value.value() * value.value();
        for (int k = 0; k < kHestonRiskFactors; ++k) {
            sum_grad[std::size_t(k)] += g[std::size_t(k)];
            sum_grad2[std::size_t(k)] += g[std::size_t(k)] * g[std::size_t(k)];
        }
    }

    const Real n = Real(cfg.paths);
    out.paths = cfg.paths;
    out.price = sum / n;
    out.standard_error =
        std::sqrt(std::fmax((sum2 - n * out.price * out.price) / (n - 1.0), 0.0) / n);
    for (int k = 0; k < kHestonRiskFactors; ++k) {
        const Real mean = sum_grad[std::size_t(k)] / n;
        out.gradient[std::size_t(k)] = mean;
        const Real var = std::fmax((sum_grad2[std::size_t(k)] - n * mean * mean) / (n - 1.0), 0.0);
        out.gradient_se[std::size_t(k)] = std::sqrt(var / n);
    }
    return out;
}

/// The same Greeks by bump-and-revalue, for the comparison.
///
/// Uses common random numbers -- the same seed for every bumped run -- because
/// without them the difference of two independent Monte Carlo estimates is
/// dominated by their noise and the exercise is meaningless. This is the fair
/// version of the baseline, not a straw man.
inline MCGreeksResult heston_mc_greeks_bump(const HestonParams& p, Real spot, Real strike,
                                            Real expiry, Real rate, Real dividend,
                                            OptionType type, const MCConfig& cfg = {},
                                            Real relative_step = 1e-4) {
    const bool qe = cfg.scheme == HestonScheme::AndersenQE;

    auto price_with = [&](const std::array<Real, kHestonRiskFactors>& x) {
        Xoshiro256pp rng(cfg.seed);
        std::vector<Real> z(std::size_t(cfg.steps), 0.0), uu(std::size_t(cfg.steps), 0.0);
        Real total = 0.0;
        for (long path = 0; path < cfg.paths; ++path) {
            for (int i = 0; i < cfg.steps; ++i) {
                z[std::size_t(i)] = rng.normal();
                uu[std::size_t(i)] = qe ? rng.uniform() : rng.normal();
            }
            total += qe ? detail::heston_qe_path<Real>(x[0], x[1], x[2], x[3], x[4], x[5],
                                                       x[6], x[7], x[8], x[9], type, z, uu,
                                                       cfg.steps, cfg.qe_psi_switch)
                        : detail::heston_euler_path<Real>(x[0], x[1], x[2], x[3], x[4], x[5],
                                                          x[6], x[7], x[8], x[9], type, z, uu,
                                                          cfg.steps);
        }
        return total / Real(cfg.paths);
    };

    const std::array<Real, kHestonRiskFactors> base = {
        spot, p.v0, p.kappa, p.theta, p.sigma, p.rho, rate, dividend, strike, expiry};

    MCGreeksResult out;
    out.paths = cfg.paths;
    out.price = price_with(base);
    for (int k = 0; k < kHestonRiskFactors; ++k) {
        const Real h = relative_step * std::fmax(std::fabs(base[std::size_t(k)]), 0.01);
        auto up = base, down = base;
        up[std::size_t(k)] += h;
        down[std::size_t(k)] -= h;
        out.gradient[std::size_t(k)] = (price_with(up) - price_with(down)) / (2.0 * h);
    }
    return out;
}

/// Price alone, on the same code path, for a like-for-like cost comparison.
inline Real heston_mc_price_only(const HestonParams& p, Real spot, Real strike, Real expiry,
                                 Real rate, Real dividend, OptionType type,
                                 const MCConfig& cfg = {}) {
    const bool qe = cfg.scheme == HestonScheme::AndersenQE;
    Xoshiro256pp rng(cfg.seed);
    std::vector<Real> z(std::size_t(cfg.steps), 0.0), uu(std::size_t(cfg.steps), 0.0);
    Real total = 0.0;
    for (long path = 0; path < cfg.paths; ++path) {
        for (int i = 0; i < cfg.steps; ++i) {
            z[std::size_t(i)] = rng.normal();
            uu[std::size_t(i)] = qe ? rng.uniform() : rng.normal();
        }
        total += qe ? detail::heston_qe_path<Real>(spot, p.v0, p.kappa, p.theta, p.sigma,
                                                   p.rho, rate, dividend, strike, expiry,
                                                   type, z, uu, cfg.steps, cfg.qe_psi_switch)
                    : detail::heston_euler_path<Real>(spot, p.v0, p.kappa, p.theta, p.sigma,
                                                      p.rho, rate, dividend, strike, expiry,
                                                      type, z, uu, cfg.steps);
    }
    return total / Real(cfg.paths);
}

}  // namespace vse
