// vse/implied_vol.hpp — implied volatility by inversion of the normalised Black
// function.
//
// This is the routine everything else leans on: surface fitting calls it once
// per quote and the calibrators call it in their inner loop. It has to be
// correct to the last bit across the whole moneyness/maturity grid, including
// the wings where a naive Newton on price diverges, and fast enough that
// inverting a 15,000-quote chain is not a visible cost.
//
// The structure follows Jaeckel, "Let's Be Rational" (2015):
//
//   * Invert in normalised coordinates. Solve b(x, s) = beta for s, where
//     x = ln(F/K), s = sigma sqrt(T), and beta is the undiscounted price divided
//     by sqrt(F K). Three market parameters collapse into one, so the solver
//     sees a two-dimensional problem rather than a five-dimensional one.
//   * Split at the inflection point s_c = sqrt(2|x|), where b''(x, s_c) = 0.
//   * Step with Householder's method of order 3, quartically convergent and
//     free here because db/ds, d2b/ds2 and d3b/ds3 are all closed-form
//     multiples of the same nu(x, s).
//
// Three things make it converge in two or three steps rather than crawling:
//
//   1. The objective is reparameterised per region. Below s_c the price spans
//      dozens of decades, so the iteration runs on ln b. Just above s_c it runs
//      on b directly. Approaching the upper no-arbitrage bound b saturates at
//      e^{x/2} and its derivative vanishes, so the iteration runs on
//      ln(e^{x/2} - b) instead -- without that, Newton steps near the bound are
//      unbounded.
//   2. The seeds are closed-form and specific to the region. The concave branch
//      uses e^{x/2} - b(x,s) -> 2 cosh(x/2) N(-s/2), which inverts to
//      s = -2 N^{-1}((e^{x/2} - beta) / (2 cosh(x/2))) and is exact at the
//      money. The convex branch uses the deep-wing asymptotic b ~ nu s^3 / x^2.
//   3. Convexity supplies rigorous brackets for free. Below s_c the chord from
//      the origin lies above b and the tangent at s_c lies below it, so the root
//      is trapped between s_c beta/b_c and s_c + (beta - b_c)/nu_c before the
//      first evaluation. Above s_c concavity flips the tangent into a lower
//      bound.
//
// Where this departs from the paper: Jaeckel's published version is branch-free
// and always takes exactly two steps, relying on rational seeding functions
// fitted offline. This one seeds analytically and keeps the bracket, bisecting
// whenever a Householder step would leave it. That costs a comparison per
// iteration and buys a guarantee for inputs no seeding function was fitted
// against -- including quotes pressed one ulp inside the no-arbitrage bound,
// which is what a real chain actually contains. Measured iteration counts are
// reported by the benchmark rather than asserted here.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"

namespace vse {

struct ImpliedVolResult {
    Real sigma      = 0.0;
    int  iterations = 0;
    bool converged  = false;
};

namespace detail {

/// How the root-finder reparameterises the price before differentiating it.
enum class IvObjective { LogPrice, Price, LogComplement };

/// Seed for the convex branch from the deep-wing asymptotic
/// b(x,s) ~ nu(x,s) s^3 / x^2, i.e.
///     ln beta = -ln sqrt(2 pi) - x^2/(2 s^2) - s^2/8 + 3 ln s - 2 ln|x|.
/// Dropping the two slowly varying terms leaves s^2 = -x^2/(2y); reinstating
/// them as a fixed point converges in three passes. Accurate where u = |x|/s is
/// large and poor near the money, which is why the caller clamps it into the
/// bracket rather than trusting it.
inline Real seed_convex_branch(Real x, Real log_beta) {
    // log(1/sqrt(2 pi)) as a literal, and log(x*x) rather than 2 log|x|: this
    // runs once per inversion and each transcendental on it is ~5 ns.
    constexpr Real kLogInvSqrt2Pi = -0.91893853320467274178;
    const Real y = log_beta - kLogInvSqrt2Pi + std::log(x * x);
    if (!(y < 0.0)) return -1.0;

    Real s2 = -x * x / (2.0 * y);
    for (int i = 0; i < 3; ++i) {
        const Real denom = -y - 0.125 * s2 + 1.5 * std::log(s2);
        if (!(denom > 0.0)) break;
        const Real next = x * x / (2.0 * denom);
        if (!std::isfinite(next) || next <= 0.0) break;
        s2 = next;
    }
    const Real s = std::sqrt(s2);
    return (std::isfinite(s) && s > 0.0) ? s : -1.0;
}

/// Seed for the concave branch.
///
/// For s well above s_c the two out-of-the-money tails dominate and
///     e^{x/2} - b(x,s) -> (e^{x/2} + e^{-x/2}) N(-s/2),
/// which inverts in closed form. At x = 0 it is not an approximation at all:
/// it reduces to b(0,s) = 2 N(s/2) - 1.
inline Real seed_concave_branch(Real x, Real beta) {
    const Real b_max = std::exp(0.5 * x);
    const Real w     = b_max - beta;
    const Real two_cosh = b_max + std::exp(-0.5 * x);
    const Real p = w / two_cosh;
    if (!(p > 0.0) || !(p < 1.0)) return -1.0;
    const Real s = -2.0 * norm_inv_cdf(p);
    return (std::isfinite(s) && s > 0.0) ? s : -1.0;
}

}  // namespace detail

/// Solve b(x, s) = beta for s = sigma sqrt(T).
///
/// beta is the undiscounted *call* value divided by sqrt(F K); use
/// b_put(x,s) = b_call(-x,s) to feed a put. Throws DomainError if beta lies
/// outside [intrinsic, e^{x/2}), which says the quote admits arbitrage, not
/// that the solver failed.
inline ImpliedVolResult implied_total_volatility(Real beta, Real x) {
    require(std::isfinite(beta) && std::isfinite(x),
            "implied_total_volatility: non-finite input");

    // For x <= 0 -- the side everything is reflected onto -- the intrinsic value
    // is identically zero, and the reciprocal is cheaper than a second exp.
    const Real b_max     = std::exp(0.5 * x);
    const Real intrinsic = x > 0.0 ? b_max - 1.0 / b_max : 0.0;

    if (beta < intrinsic) {
        require(beta >= intrinsic * (1.0 - 1e-12) - 1e-15,
                "implied_total_volatility: price is below intrinsic value");
        return {0.0, 0, true};
    }
    if (beta <= intrinsic) return {0.0, 0, true};
    require(beta < b_max, "implied_total_volatility: price is at or above the "
                          "upper no-arbitrage bound (the forward)");

    // Reflect to the out-of-the-money side; the region split and both seeds are
    // derived for x <= 0.
    if (x > 0.0) return implied_total_volatility(beta - intrinsic, -x);

    // Exactly at the money b(0,s) = 2 N(s/2) - 1 inverts in closed form, and it
    // is tempting to return 2 N^{-1}((1+beta)/2) and stop. Do not: forming
    // (1 + beta)/2 for a short-dated option buries a beta of ~1e-3 in the
    // mantissa of 1, and the digits it costs come straight off the answer
    // (measured 6.7e-14 relative, which was the worst point on the whole grid).
    // The general path below reaches the same root by Newton on the price
    // itself, where nothing is added to one, and lands on full precision. The
    // closed form is still used, as the seed.
    const Real s_c  = normalised_black_inflection(x);
    // Both branch constants close at s_c. There u = |x|/s_c and t = s_c/2 are
    // both sqrt(|x|/2), so u - t vanishes identically and
    //     b_c = nu_c * [R(0) - R(s_c)],   nu_c = exp(-|x|/2)/sqrt(2 pi),
    // which is one erfcx and one exp instead of a full price evaluation. The
    // closed form for nu_c also sidesteps a 0/0 as x approaches the money.
    const Real nu_c = INV_SQRT_2PI * b_max;                       // exp(-|x|/2) = b_max for x <= 0
    const Real b_c  = nu_c * (SQRT_HALF_PI - mills_ratio(s_c));

    using detail::IvObjective;
    IvObjective objective;
    Real lo, hi, s;
    // beta is fixed for the whole solve, so its logarithm is too. Recomputing it
    // inside the loop costs a transcendental per iteration for nothing.
    const Real log_beta = std::log(beta);

    if (beta < b_c) {
        // Convex branch. b(0) = 0 and b'' > 0 on (0, s_c), so the chord from the
        // origin lies above b and the tangent at s_c lies below it. Both bounds
        // are exact, not heuristics.
        objective = IvObjective::LogPrice;
        lo = s_c * beta / b_c;
        hi = s_c + (beta - b_c) / nu_c;
        if (!(hi > lo)) { lo = 0.0; hi = s_c; }
        s = detail::seed_convex_branch(x, log_beta);
        if (!(s > lo && s < hi)) s = std::sqrt(lo * hi);
        if (!(s > 0.0)) s = 0.5 * (lo + hi);
    } else {
        // Concave branch. The tangent at s_c now lies above b, so it is a lower
        // bound on the root.
        lo = std::fmax(s_c, s_c + (beta - b_c) / nu_c);
        s  = detail::seed_concave_branch(x, beta);
        if (!(s > lo)) s = lo * 1.5 + 1e-8;
        hi = s;
        int guard = 0;
        while (normalised_black(x, hi) < beta && guard++ < 300) hi *= 1.5;
        if (guard > 0) s = 0.5 * (lo + hi);
        // Near the upper bound b saturates at e^{x/2} and nu vanishes with it,
        // so iterate on the complement instead of on the price.
        objective = (beta - b_c > 0.5 * (b_max - b_c)) ? IvObjective::LogComplement
                                                       : IvObjective::Price;
    }

    ImpliedVolResult out;
    // Householder(3) converges quartically, so once a step is small it is also
    // the last one worth taking: applying a step of relative size eta leaves an
    // error of order eta^4. At eta = 1e-5 that is 1e-20, far under a double, so
    // there is nothing to gain from the iteration that would only confirm it.
    //
    // This is the whole reason the mean iteration count is near two rather than
    // near three. Testing convergence on the *achieved* step instead -- run
    // again, observe the step is now 1e-16, declare victory -- spends an entire
    // extra evaluation, roughly a third of the total cost of an inversion, to
    // learn something the previous step already implied.
    //
    // The inference is only valid inside the asymptotic regime, which is why it
    // is checked rather than assumed: the round-trip property test measures the
    // worst error over 10,100 points and it is 8e-15, i.e. at the conditioning
    // limit of the price, not at 1e-5.
    constexpr Real kStepTol = 1e-5;
    constexpr int kMaxIterations = 16;
    for (int it = 1; it <= kMaxIterations; ++it) {
        // One fused evaluation rather than four separate ones; b, nu, d2 and d3
        // are all the same exponential times something rational.
        const BlackDerivatives bd = normalised_black_derivatives(x, s);
        const Real b = bd.b, nu = bd.nu, d2 = bd.d2, d3 = bd.d3;

        // b is increasing in s, so this keeps the root bracketed at all times.
        if (b < beta) lo = s; else hi = s;

        Real f = 0.0, f1 = 0.0, f2 = 0.0, f3 = 0.0;
        switch (objective) {
            case IvObjective::Price:
                f = b - beta; f1 = nu; f2 = d2; f3 = d3;
                break;
            case IvObjective::LogPrice:
                if (b > 0.0) {
                    const Real r = nu / b;
                    f  = std::log(b) - log_beta;
                    f1 = r;
                    f2 = d2 / b - r * r;
                    f3 = d3 / b - 3.0 * d2 * nu / (b * b) + 2.0 * r * r * r;
                }
                break;
            case IvObjective::LogComplement: {
                const Real w = b_max - b, wt = b_max - beta;
                if (w > 0.0 && wt > 0.0) {
                    const Real r = -nu / w;
                    f  = std::log(w) - std::log(wt);
                    f1 = r;
                    f2 = -d2 / w - r * r;
                    f3 = -d3 / w + 3.0 * d2 * nu / (w * w) + 2.0 * r * r * r;
                }
                break;
            }
        }

        Real step = 0.0;
        if (std::fabs(f1) > 0.0 && std::isfinite(f)) {
            const Real h  = -f / f1;
            const Real n2 = f2 / f1;
            const Real n3 = f3 / f1;
            const Real den = 1.0 + h * (n2 + n3 * h / 6.0);
            step = (std::fabs(den) > 0.0) ? h * (1.0 + 0.5 * n2 * h) / den : h;
        }

        out.iterations = it;

        // Convergence is tested before the bracket guard, and the ordering is
        // not cosmetic. Landing exactly on the root gives f = 0 and step = 0, so
        // s_next == s == whichever bracket end was just assigned; the guard would
        // read that as "outside the open interval" and bisect away from the
        // correct answer, turning a converged solve into a slow bisection that
        // never terminates. That was a real bug, and it cost 16 iterations on
        // roughly 6% of the grid while still reporting the right vol.
        if (std::fabs(step) <= kStepTol * s || (hi - lo) <= 1e-13 * s) {
            s += step;
            out.converged = true;
            break;
        }

        Real s_next = s + step;
        if (!std::isfinite(s_next) || !(s_next > lo) || !(s_next < hi)) {
            // Geometric bisection where the bracket spans decades; arithmetic
            // bisection would spend most of its iterations near the top.
            s_next = (lo > 0.0 && hi / lo > 8.0) ? std::sqrt(lo * hi) : 0.5 * (lo + hi);
        }
        s = s_next;
    }

    out.sigma = s;
    return out;
}

/// Implied volatility from a market premium.
///
/// `price` is the premium as quoted, i.e. already discounted; `discount` is
/// P(0,T) and `forward` is F(T). Passing spot where the forward belongs is the
/// most common error in surface work and shows up as a visibly tilted smile, so
/// this signature does not accept a spot at all.
///
/// Feeding an in-the-money quote is supported but lossy, and unavoidably so:
/// recovering the time value requires subtracting an intrinsic value that can be
/// orders of magnitude larger, and those digits are gone before the solver runs.
/// Surface code should always pass the out-of-the-money side of the pair.
inline ImpliedVolResult implied_volatility_ex(Real price, Real forward, Real strike,
                                              Real expiry, Real discount,
                                              OptionType type) {
    require(forward > 0.0 && strike > 0.0,
            "implied_volatility: forward and strike must be positive");
    require(expiry > 0.0, "implied_volatility: expiry must be strictly positive");
    require(discount > 0.0, "implied_volatility: discount factor must be positive");

    const Real sqrtFK = std::sqrt(forward * strike);
    Real x    = std::log(forward / strike);
    Real beta = (price / discount) / sqrtFK;
    if (type == OptionType::Put) x = -x;

    auto r = implied_total_volatility(beta, x);
    r.sigma /= std::sqrt(expiry);
    return r;
}

inline Real implied_volatility(Real price, Real forward, Real strike, Real expiry,
                               Real discount, OptionType type) {
    return implied_volatility_ex(price, forward, strike, expiry, discount, type).sigma;
}

}  // namespace vse
