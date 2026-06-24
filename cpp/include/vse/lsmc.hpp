// vse/lsmc.hpp — Longstaff-Schwartz, and the dual bound that makes it honest.
//
// WHY A SINGLE NUMBER IS NOT AN ANSWER
//
// Longstaff-Schwartz prices an American option by regressing the continuation
// value on a basis of functions of the state and exercising when the immediate
// payoff exceeds the regressed continuation value. The result is a LOWER bound,
// and necessarily so: any exercise policy, however bad, is admissible, so the
// value it produces cannot exceed the value of the optimal policy. A regression
// that is too coarse gives a worse policy and a lower number, and there is
// nothing in the estimate itself that says how much was left on the table.
//
// That makes a bare Longstaff-Schwartz price uncheckable. It is biased low by an
// unknown amount, and the two obvious diagnostics -- adding paths, adding basis
// functions -- move it in the same direction, so a number that has stopped
// moving may still be wrong.
//
// TWO BIASES, ONE INTERVAL
//
// Rogers, and Haugh and Kogan, showed that the American value also has a DUAL
// representation as a minimisation over martingales:
//
//     V_0 = inf_M E[ max_k ( h_k - M_k ) ]
//
// so any martingale gives an UPPER bound. Andersen and Broadie construct one
// from the exercise policy itself: the martingale part of the value process
// under that policy, estimated by nested simulation. A good policy gives a tight
// bound, and the gap between the two is a computable statement about how much
// the policy is losing.
//
// The pair is what gets reported. "8.5645 to 8.5698" is a result; "8.5645" on
// its own is a hypothesis.
//
// COST
//
// The dual is expensive -- an inner simulation at every exercise date on every
// outer path -- which is why it is not run at production path counts. It is a
// validation instrument: run it once at a modest path count to establish that
// the policy is good, then run the cheap lower bound in anger.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/linalg.hpp"
#include "vse/rng.hpp"

#include <vector>

namespace vse {

struct LSMCConfig {
    long paths = 100000;         ///< paths for the pricing pass
    long training_paths = 50000; ///< separate paths for fitting the regression
    int  exercise_dates = 50;
    int  basis_degree = 3;       ///< polynomial degree in the (scaled) spot
    bool run_dual = false;
    long dual_outer_paths = 2000;
    long dual_inner_paths = 200;
    std::uint64_t seed = 20260714;
};

struct LSMCResult {
    Real lower = 0.0;            ///< Longstaff-Schwartz, biased low
    Real lower_se = 0.0;
    Real upper = 0.0;            ///< Andersen-Broadie dual, biased high
    Real upper_se = 0.0;
    Real duality_gap = 0.0;      ///< upper - lower
    Real point_estimate = 0.0;   ///< midpoint
    bool dual_run = false;
    long paths = 0;
    Real control_correlation = 0.0;   ///< between the American and European payoffs
};

namespace detail {

/// Regression basis: powers of the moneyness, which is scaled so that the
/// design matrix stays conditioned.
///
/// Scaling matters more than it looks. Raw powers of a spot near 4000 give a
/// design matrix whose columns differ by 10^11, and the normal equations then
/// have a condition number past 1e22 -- the regression returns coefficients that
/// are numerically meaningless while reporting no error at all. Dividing by the
/// strike costs one multiply and removes the problem entirely.
inline void polynomial_basis(Real spot, Real strike, int degree, std::vector<Real>& out) {
    out.resize(std::size_t(degree) + 1);
    const Real x = spot / strike;
    Real power = 1.0;
    for (int i = 0; i <= degree; ++i) {
        out[std::size_t(i)] = power;
        power *= x;
    }
}

/// Simulate GBM paths at the exercise dates.
inline void simulate_gbm(Real spot, Real rate, Real dividend, Real sigma, Real expiry,
                         int dates, long paths, std::uint64_t seed,
                         std::vector<Real>& out) {
    Xoshiro256pp rng(seed);
    const Real dt = expiry / Real(dates);
    const Real drift = (rate - dividend - 0.5 * sigma * sigma) * dt;
    const Real vol = sigma * std::sqrt(dt);
    out.assign(std::size_t(paths) * std::size_t(dates), 0.0);
    for (long p = 0; p < paths; ++p) {
        Real log_s = std::log(spot);
        for (int k = 0; k < dates; ++k) {
            log_s += drift + vol * rng.normal();
            out[std::size_t(p) * std::size_t(dates) + std::size_t(k)] = std::exp(log_s);
        }
    }
}

}  // namespace detail

/// Price an American option on a lognormal underlying by Longstaff-Schwartz.
///
/// Bermudan in truth, with `exercise_dates` opportunities; the American limit is
/// approached as that count rises, and the test suite measures the gap against a
/// continuously-exercisable lattice.
inline LSMCResult lsmc_american(Real spot, Real strike, Real expiry, Real rate,
                                Real dividend, Real sigma, OptionType type,
                                const LSMCConfig& cfg = {}) {
    require(spot > 0.0 && strike > 0.0 && expiry > 0.0 && sigma > 0.0,
            "lsmc_american: spot, strike, expiry and vol must be positive");
    require(cfg.exercise_dates >= 1 && cfg.paths > 1 && cfg.training_paths > 1,
            "lsmc_american: need dates and paths");

    const int dates = cfg.exercise_dates;
    const Real dt = expiry / Real(dates);
    const Real step_discount = std::exp(-rate * dt);
    const Real w = omega(type);
    const int basis_size = cfg.basis_degree + 1;

    auto payoff = [&](Real s) { return std::fmax(w * (s - strike), 0.0); };

    // ---- pass one: fit the exercise policy on training paths ---------------
    //
    // Fitted on paths that are then thrown away. Using the same paths to fit the
    // policy and to price it is the standard shortcut and it destroys the one
    // property that makes the estimator interpretable: the policy is then chosen
    // with knowledge of the realisations it is evaluated on, so the estimate is
    // biased UPWARD by an unknown amount and is no longer a lower bound at all.
    std::vector<Real> train;
    detail::simulate_gbm(spot, rate, dividend, sigma, expiry, dates, cfg.training_paths,
                         cfg.seed, train);

    // coefficients[k] holds the regression for exercise date k (k < dates - 1).
    std::vector<std::vector<Real>> coefficients(static_cast<std::size_t>(dates));
    {
        std::vector<Real> cashflow(static_cast<std::size_t>(cfg.training_paths), 0.0);
        for (long p = 0; p < cfg.training_paths; ++p) {
            cashflow[std::size_t(p)] =
                payoff(train[std::size_t(p) * std::size_t(dates) + std::size_t(dates - 1)]);
        }

        std::vector<Real> basis;
        for (int k = dates - 2; k >= 0; --k) {
            // Discount one step back.
            for (auto& c : cashflow) c *= step_discount;

            // Regress on in-the-money paths only. Out-of-the-money paths carry no
            // exercise decision and including them makes the fit chase the region
            // where the answer does not matter, which is the single most common
            // way to get a visibly wrong exercise boundary.
            std::vector<long> itm;
            for (long p = 0; p < cfg.training_paths; ++p) {
                const Real s = train[std::size_t(p) * std::size_t(dates) + std::size_t(k)];
                if (payoff(s) > 0.0) itm.push_back(p);
            }
            if (itm.size() < std::size_t(basis_size) * 4) {
                coefficients[std::size_t(k)].assign(std::size_t(basis_size), 0.0);
                continue;
            }

            Matrix design(itm.size(), std::size_t(basis_size));
            std::vector<Real> y(itm.size());
            for (std::size_t i = 0; i < itm.size(); ++i) {
                const long p = itm[i];
                detail::polynomial_basis(
                    train[std::size_t(p) * std::size_t(dates) + std::size_t(k)], strike,
                    cfg.basis_degree, basis);
                for (int j = 0; j < basis_size; ++j) design(i, std::size_t(j)) = basis[std::size_t(j)];
                y[i] = cashflow[std::size_t(p)];
            }
            std::vector<Real> beta;
            if (!least_squares(design, y, beta)) {
                beta.assign(std::size_t(basis_size), 0.0);
            }
            coefficients[std::size_t(k)] = beta;

            // Update the training cashflows under the fitted policy.
            for (long p : itm) {
                const Real s = train[std::size_t(p) * std::size_t(dates) + std::size_t(k)];
                detail::polynomial_basis(s, strike, cfg.basis_degree, basis);
                Real continuation = 0.0;
                for (int j = 0; j < basis_size; ++j) {
                    continuation += beta[std::size_t(j)] * basis[std::size_t(j)];
                }
                const Real exercise = payoff(s);
                if (exercise > continuation) cashflow[std::size_t(p)] = exercise;
            }
        }
    }

    // The policy, as a function of (date, spot).
    auto exercise_now = [&](int k, Real s) {
        const Real exercise = payoff(s);
        if (exercise <= 0.0) return false;
        if (k == dates - 1) return true;
        const auto& beta = coefficients[std::size_t(k)];
        if (beta.empty()) return false;
        std::vector<Real> basis;
        detail::polynomial_basis(s, strike, cfg.basis_degree, basis);
        Real continuation = 0.0;
        for (int j = 0; j < basis_size; ++j) continuation += beta[std::size_t(j)] * basis[std::size_t(j)];
        return exercise > continuation;
    };

    // ---- pass two: price the policy on fresh paths -------------------------
    std::vector<Real> pricing;
    detail::simulate_gbm(spot, rate, dividend, sigma, expiry, dates, cfg.paths,
                         cfg.seed ^ 0x9E3779B97F4A7C15ULL, pricing);

    // The European option on the same paths is a control variate with a known
    // mean and a correlation above 0.9 with the American payoff -- the two
    // differ only by the early-exercise premium. It costs one extra max() per
    // path and cuts the standard error by a factor of three or four, which
    // matters here because the width of the duality interval is otherwise
    // dominated by the noise in its lower end rather than by the quality of the
    // exercise policy, and then the interval says nothing about the policy.
    const Real european = bs_price(spot, strike, expiry, rate, dividend, sigma, type);
    Real sum = 0.0, sum2 = 0.0, sum_c = 0.0, sum_c2 = 0.0, sum_xc = 0.0;
    const Real terminal_discount = std::exp(-rate * expiry);

    for (long p = 0; p < cfg.paths; ++p) {
        Real value = 0.0;
        for (int k = 0; k < dates; ++k) {
            const Real s = pricing[std::size_t(p) * std::size_t(dates) + std::size_t(k)];
            if (exercise_now(k, s)) {
                value = std::exp(-rate * dt * Real(k + 1)) * payoff(s);
                break;
            }
        }
        const Real control =
            terminal_discount *
            payoff(pricing[std::size_t(p) * std::size_t(dates) + std::size_t(dates - 1)]);
        sum += value;
        sum2 += value * value;
        sum_c += control;
        sum_c2 += control * control;
        sum_xc += value * control;
    }

    LSMCResult out;
    out.paths = cfg.paths;
    const Real n = Real(cfg.paths);
    const Real mean_x = sum / n, mean_c = sum_c / n;
    const Real var_x = std::fmax((sum2 - n * mean_x * mean_x) / (n - 1.0), 0.0);
    const Real var_c = std::fmax((sum_c2 - n * mean_c * mean_c) / (n - 1.0), 0.0);
    const Real cov = (sum_xc - n * mean_x * mean_c) / (n - 1.0);

    if (var_c > 0.0) {
        const Real beta = cov / var_c;
        out.lower = mean_x - beta * (mean_c - european);
        const Real var_adj = std::fmax(var_x - 2.0 * beta * cov + beta * beta * var_c, 0.0);
        out.lower_se = std::sqrt(var_adj / n);
        out.control_correlation = cov / std::sqrt(std::fmax(var_x * var_c, 1e-300));
    } else {
        out.lower = mean_x;
        out.lower_se = std::sqrt(var_x / n);
    }
    out.point_estimate = out.lower;

    if (!cfg.run_dual) return out;

    // ---- the dual bound ----------------------------------------------------
    //
    // Andersen-Broadie. The martingale is the martingale part of the value
    // process under the fitted policy,
    //
    //     M_k - M_{k-1} = Vtilde_k - E_{k-1}[Vtilde_k],
    //
    // and the upper bound is E[ max_k ( D_k h_k - M_k ) ].
    //
    // The implementation detail that decides whether this works at all: at any
    // date where the policy does NOT exercise, the policy value satisfies
    // Vtilde_{k-1} = E_{k-1}[Vtilde_k] by construction, so the increment is
    // simply Vtilde_k - Vtilde_{k-1} and NO inner simulation is needed. Nested
    // simulation is required only at dates where the policy does exercise.
    //
    // That is not merely an optimisation. Estimating every conditional
    // expectation independently makes the martingale a sum of fifty independent
    // estimation errors, and since the estimator then takes a maximum over those
    // fifty dates, the noise biases the bound upward -- badly. Measured with 150
    // inner paths at 50 dates: an upper bound of 10.11 against a true value of
    // 8.57, an 18% "bound" that is really a measurement of its own noise. With
    // the telescoping the pre-exercise increments cancel and only the genuine
    // exercise-date estimates remain.
    Xoshiro256pp rng(cfg.seed ^ 0xD1B54A32D192ED03ULL);
    const Real drift = (rate - dividend - 0.5 * sigma * sigma) * dt;
    const Real vol = sigma * std::sqrt(dt);

    /// Value AT date k of following the fitted policy from date k onwards,
    /// starting from spot s. Discounted back to date k, not to time zero.
    auto policy_value = [&](int k, Real s, Xoshiro256pp& gen) {
        Real current = s;
        for (int j = k; j < dates; ++j) {
            if (j > k) current *= std::exp(drift + vol * gen.normal());
            if (exercise_now(j, current)) {
                return std::exp(-rate * dt * Real(j - k)) * payoff(current);
            }
        }
        return 0.0;
    };

    auto estimate_value = [&](int k, Real s, Xoshiro256pp& gen) {
        if (exercise_now(k, s)) return payoff(s);
        Real acc = 0.0;
        for (long q = 0; q < cfg.dual_inner_paths; ++q) acc += policy_value(k, s, gen);
        return acc / Real(cfg.dual_inner_paths);
    };

    Real dual_sum = 0.0, dual_sum2 = 0.0;
    std::vector<Real> path(static_cast<std::size_t>(dates), 0.0);
    for (long p = 0; p < cfg.dual_outer_paths; ++p) {
        Real s = spot;
        for (int k = 0; k < dates; ++k) {
            s *= std::exp(drift + vol * rng.normal());
            path[std::size_t(k)] = s;
        }

        Xoshiro256pp inner(rng.next());
        Real martingale = 0.0;
        Real best = payoff(spot);                  // exercising at once; M_0 = 0
        Real previous_value = out.lower;           // Vtilde_0, the policy value
        Real previous_discount = 1.0;
        Real previous_spot = spot;
        bool previous_exercised = false;

        for (int k = 0; k < dates; ++k) {
            const Real s_k = path[std::size_t(k)];
            const Real discount_k = std::exp(-rate * dt * Real(k + 1));
            const Real v_hat = estimate_value(k, s_k, inner);

            if (previous_exercised) {
                // The policy stopped here, so the value process jumps and the
                // conditional expectation has to be simulated.
                Real conditional = 0.0;
                for (long q = 0; q < cfg.dual_inner_paths; ++q) {
                    const Real s_next = previous_spot * std::exp(drift + vol * inner.normal());
                    conditional += policy_value(k, s_next, inner);
                }
                conditional /= Real(cfg.dual_inner_paths);
                martingale += discount_k * (v_hat - conditional);
            } else {
                // Vtilde_{k-1} = E_{k-1}[Vtilde_k]: the increment telescopes.
                martingale += discount_k * v_hat - previous_discount * previous_value;
            }

            best = std::fmax(best, discount_k * payoff(s_k) - martingale);
            previous_value = v_hat;
            previous_discount = discount_k;
            previous_spot = s_k;
            previous_exercised = exercise_now(k, s_k);
        }
        dual_sum += best;
        dual_sum2 += best * best;
    }

    const Real m = Real(cfg.dual_outer_paths);
    out.upper = dual_sum / m;
    out.upper_se = std::sqrt(std::fmax((dual_sum2 - m * out.upper * out.upper) / (m - 1.0), 0.0) / m);
    out.duality_gap = out.upper - out.lower;
    out.point_estimate = 0.5 * (out.lower + out.upper);
    out.dual_run = true;
    return out;
}

}  // namespace vse
