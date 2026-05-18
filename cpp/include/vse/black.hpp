// vse/black.hpp — Black-Scholes / Black-76 prices and analytic Greeks.
//
// The price is computed through the *normalised* Black function
//
//     b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2),
//     x = ln(F/K),  s = sigma * sqrt(T),
//
// so that an undiscounted call is sqrt(F*K) * b(x, s). Two things fall out of
// this parameterisation and both matter downstream:
//
//   1. db/ds = nu(x,s) = exp(-x^2/(2 s^2) - s^2/8) / sqrt(2 pi) exactly, and the
//      second and third derivatives are equally clean. The implied-vol solver in
//      implied_vol.hpp uses all three for a Householder(3) step.
//   2. Evaluated naively, b is a difference of two nearly equal numbers in the
//      wings and loses every significant digit. Written through the Mills ratio
//      R(z) = N(-z)/phi(z) it becomes
//
//          b(x,s) = nu(x,s) * [ R(u - t) - R(u + t) ],  u = -x/s, t = s/2
//
//      for x <= 0, which is a difference of two O(1) quantities scaled by a
//      factor carrying all of the exponential smallness. That is what lets the
//      round trip hold ~1e-13 at five standard deviations out.
#pragma once

#include "vse/common.hpp"
#include "vse/normal.hpp"

namespace vse {

/// Mills ratio R(z) = N(-z) / phi(z), valid for every real z.
///
/// R(z) = sqrt(pi/2) * erfcx(z / sqrt(2)) exactly. Written this way the two
/// exponentials that would otherwise appear in numerator and denominator never
/// exist, so there is nothing to underflow and nothing to cancel: R(40) comes
/// back as 0.02498 rather than as 0/0. It is also the fast path, because erfcx
/// evaluates as a single rational function for arguments above 0.46875.
inline Real mills_ratio(Real z) {
    return SQRT_HALF_PI * erfcx(z * INV_SQRT_2);
}

/// R(u - t) - R(u + t), which is the bracket in the normalised Black formula.
///
/// The Mills form removes the catastrophic cancellation of the raw
/// difference-of-CDFs, but a second, milder one survives inside it: the two
/// Mills ratios differ by roughly 2t|R'(u)|, so when t is small against the
/// scale on which R varies the subtraction still costs about
/// -log10(2t|R'(u)|/R(u)) digits. Near the money at short expiry that is three
/// or four digits, and it was the binding constraint on the implied-vol round
/// trip -- 2.8e-13, an order of magnitude worse than everything around it.
///
/// The fix is to stop subtracting. R is entire, so the difference has an exact
/// odd Taylor series
///
///     R(u-t) - R(u+t) = -2 sum_k t^{2k+1} R^{(2k+1)}(u) / (2k+1)!,
///
/// and the derivatives come from R'(z) = z R(z) - 1 with
/// R^{(n+1)} = z R^{(n)} + n R^{(n-1)}. Every term is added, nothing cancels,
/// and where t is small the series is also faster than two erfcx calls.
///
/// The forward recurrence is mildly unstable for large u, but only the first
/// two or three terms ever matter in the regime where the series is selected,
/// and by then the higher terms are 1e-5 of the leading one -- their lost digits
/// do not reach the result.
inline Real mills_difference(Real u, Real t) {
    const Real r0 = mills_ratio(u);
    const Real r1 = u * r0 - 1.0;

    // Estimated relative size of the difference against its operands. Above a
    // few percent the direct subtraction keeps essentially all its digits and
    // is the cheaper branch.
    if (!(2.0 * t * std::fabs(r1) < 0.1 * std::fabs(r0))) {
        return mills_ratio(u - t) - mills_ratio(u + t);
    }

    Real d_prev = r0, d_cur = r1;   // R^{(n-1)}, R^{(n)} at n = 1
    Real t_pow  = t;                // t^{2k+1}
    Real fact   = 1.0;              // (2k+1)!
    Real sum    = t_pow * d_cur / fact;
    const Real t2 = t * t;

    for (int k = 1; k <= 10; ++k) {
        const int n = 2 * k - 1;
        for (int j = 0; j < 2; ++j) {
            const Real d_next = u * d_cur + Real(n + j) * d_prev;
            d_prev = d_cur;
            d_cur  = d_next;
        }
        t_pow *= t2;
        fact  *= Real(2 * k) * Real(2 * k + 1);
        const Real term = t_pow * d_cur / fact;
        sum += term;
        if (std::fabs(term) <= 1e-18 * std::fabs(sum)) break;
    }
    return -2.0 * sum;
}

/// Normalised Black vega: db/ds. Never cancels, never overflows.
inline Real normalised_vega(Real x, Real s) noexcept {
    if (s <= 0.0) return 0.0;
    return INV_SQRT_2PI * std::exp(-0.5 * sqr(x / s) - 0.125 * s * s);
}

/// Undiscounted call value in units of sqrt(F*K).
inline Real normalised_black(Real x, Real s) {
    if (s <= 0.0) return std::fmax(std::exp(0.5 * x) - std::exp(-0.5 * x), 0.0);
    if (!std::isfinite(s)) return std::exp(0.5 * x);

    // Work on the out-of-the-money side. Parity in normalised units reads
    // b_call(x,s) - b_call(-x,s) = e^{x/2} - e^{-x/2}.
    if (x > 0.0) {
        return (std::exp(0.5 * x) - std::exp(-0.5 * x)) + normalised_black(-x, s);
    }

    const Real u = -x / s, t = 0.5 * s;
    if (s <= 20.0) {
        // b = nu * [R(u - t) - R(u + t)] is an identity for all x <= 0, and with
        // R routed through erfcx it holds full relative precision from the money
        // out to the point where nu itself underflows -- which is the correct
        // place to stop, because that is where the true price leaves the range
        // of a double. The s cap only guards erfcx(-y) = 2e^{y^2} - erfcx(y)
        // against overflow, and s = 20 is sigma * sqrt(T) = 2000 vol points at
        // one year.
        return normalised_vega(x, s) * mills_difference(u, t);
    }
    return std::exp(0.5 * x) * norm_cdf(x / s + t) -
           std::exp(-0.5 * x) * norm_cdf(x / s - t);
}

/// d^2 b / ds^2.
inline Real normalised_black_d2(Real x, Real s) noexcept {
    return normalised_vega(x, s) * (x * x / (s * s * s) - 0.25 * s);
}

/// d^3 b / ds^3.
inline Real normalised_black_d3(Real x, Real s) noexcept {
    const Real g  = x * x / (s * s * s) - 0.25 * s;
    const Real gp = -3.0 * x * x / (s * s * s * s) - 0.25;
    return normalised_vega(x, s) * (g * g + gp);
}

/// Inflection point of b in s: b''(x, s_c) = 0 at s_c = sqrt(2|x|).
/// Below it b is convex in s, above it concave. The implied-vol solver seeds
/// each branch differently.
inline Real normalised_black_inflection(Real x) noexcept {
    return std::sqrt(2.0 * std::fabs(x));
}

// ---------------------------------------------------------------------------
// Black-76: price on the forward.
// ---------------------------------------------------------------------------

/// Undiscounted Black-76 value. Multiply by the discount factor for a premium.
inline Real black76_undiscounted(Real F, Real K, Real T, Real sigma, OptionType type) {
    require(F > 0.0 && K > 0.0, "black76: forward and strike must be positive");
    require(T >= 0.0, "black76: expiry must be non-negative");
    require(sigma >= 0.0, "black76: volatility must be non-negative");

    const Real x      = std::log(F / K);
    const Real s      = sigma * std::sqrt(T);
    const Real sqrtFK = std::sqrt(F * K);
    // b_put(x,s) = b_call(-x,s).
    return sqrtFK * normalised_black(type == OptionType::Call ? x : -x, s);
}

inline Real black76(Real F, Real K, Real T, Real sigma, Real discount, OptionType type) {
    return discount * black76_undiscounted(F, K, T, sigma, type);
}

// ---------------------------------------------------------------------------
// Spot parameterisation with carry, plus the full Greek set.
// ---------------------------------------------------------------------------

inline Real bs_price(Real S, Real K, Real T, Real r, Real q, Real sigma, OptionType type) {
    require(S > 0.0, "bs_price: spot must be positive");
    const Real F = S * std::exp((r - q) * T);
    return black76(F, K, T, sigma, std::exp(-r * T), type);
}

/// Analytic Greeks. Conventions, stated once so nothing downstream has to guess:
///   vega   d(price)/d(sigma), sigma in absolute units (1.0 == 100 vol points)
///   theta  d(price)/dt per calendar year, i.e. value lost per year
///   rho    d(price)/dr in absolute units
///   vanna  d^2(price)/dS dsigma
///   volga  d^2(price)/dsigma^2
///   dual_* strike sensitivities, which is what Breeden-Litzenberger needs
struct Greeks {
    Real price      = 0.0;
    Real delta      = 0.0;
    Real gamma      = 0.0;
    Real vega       = 0.0;
    Real theta      = 0.0;
    Real rho        = 0.0;
    Real vanna      = 0.0;
    Real volga      = 0.0;
    Real dual_delta = 0.0;
    Real dual_gamma = 0.0;
};

inline Greeks bs_greeks(Real S, Real K, Real T, Real r, Real q, Real sigma,
                        OptionType type) {
    require(S > 0.0 && K > 0.0, "bs_greeks: spot and strike must be positive");
    require(T > 0.0, "bs_greeks: expiry must be strictly positive");
    require(sigma > 0.0, "bs_greeks: volatility must be strictly positive");

    const Real sqrtT = std::sqrt(T);
    const Real srt   = sigma * sqrtT;
    const Real dfq   = std::exp(-q * T);
    const Real dfr   = std::exp(-r * T);
    const Real d1    = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt;
    const Real d2    = d1 - srt;
    const Real pdf1  = norm_pdf(d1);
    const Real w     = omega(type);

    Greeks g;
    g.price = bs_price(S, K, T, r, q, sigma, type);
    g.delta = w * dfq * norm_cdf(w * d1);
    g.gamma = dfq * pdf1 / (S * srt);
    g.vega  = S * dfq * pdf1 * sqrtT;
    g.theta = -S * dfq * pdf1 * sigma / (2.0 * sqrtT)
              - w * r * K * dfr * norm_cdf(w * d2)
              + w * q * S * dfq * norm_cdf(w * d1);
    g.rho   = w * K * T * dfr * norm_cdf(w * d2);
    g.vanna = -dfq * pdf1 * d2 / sigma;
    g.volga = g.vega * d1 * d2 / sigma;
    g.dual_delta = -w * dfr * norm_cdf(w * d2);
    g.dual_gamma = dfr * norm_pdf(d2) / (K * srt);
    return g;
}

/// Undiscounted no-arbitrage bounds on a European price. Used as assertions
/// through the test suite and as brackets in the implied-vol solver.
inline void black_price_bounds(Real F, Real K, OptionType type, Real& lo, Real& hi) {
    if (type == OptionType::Call) {
        lo = std::fmax(F - K, 0.0);
        hi = F;
    } else {
        lo = std::fmax(K - F, 0.0);
        hi = K;
    }
}

}  // namespace vse
