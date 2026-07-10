// vse/sabr.hpp — SABR: Hagan's expansion, its documented failure, and two fixes.
//
// SABR is here for a specific reason. It is the market standard for interest
// rate smiles, its closed form makes it the cheapest smile parameterisation
// there is, and it is WRONG in a way that is worth being able to demonstrate
// rather than merely cite.
//
// Hagan, Kumar, Lesniewski and Woodward (2002) give the implied volatility of
// the SABR model as a singular perturbation expansion in the total variance. The
// expansion is asymptotic in alpha^2 T, so its error grows with maturity and
// with vol-of-vol -- and crucially it is an expansion of the *implied vol*, with
// no constraint that the resulting smile correspond to any probability
// distribution. For beta < 1 and low strikes it does not: the implied density
// goes negative, and the price of a butterfly spread that the formula produces
// is less than zero.
//
// This is not a subtle effect at exotic parameters. With a 30% vol-of-vol at ten
// years -- an ordinary long-dated rates calibration -- the density is negative
// over a visible range of strikes, and anything that integrates against it (a
// CMS convexity adjustment, a digital, a range accrual) is wrong. The test suite
// locates the violation and reports where it starts.
//
// Two standard responses are implemented:
//
//   * SHIFTED SABR. Replace F by F + s and K by K + s. The lognormal process
//     then lives on (-s, infinity), which is what makes the model usable at all
//     when rates are negative, and it moves the arbitrage down with the shift
//     rather than removing it.
//   * NORMAL SABR. At beta = 0 the SABR process is a Bachelier model with
//     stochastic vol, and Hagan's normal-vol expansion for that case is far
//     better behaved -- the arbitrage is pushed out to strikes that are not
//     quoted. Converting between normal and lognormal quoting conventions is
//     then an implied-vol inversion, which this library already does exactly.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/implied_vol.hpp"

#include <vector>

namespace vse {

struct SABRParams {
    Real alpha = 0.2;    ///< initial volatility level
    Real beta  = 0.5;    ///< CEV exponent, in [0, 1]
    Real rho   = -0.3;   ///< correlation between the forward and its volatility
    Real nu    = 0.4;    ///< volatility of volatility
    Real shift = 0.0;    ///< displacement, for negative forwards

    bool is_well_formed() const {
        return alpha > 0.0 && beta >= 0.0 && beta <= 1.0 &&
               std::fabs(rho) < 1.0 && nu >= 0.0;
    }
};

namespace detail {

/// z / x(z) where x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho)).
///
/// The ratio tends to 1 as z tends to 0, and computing it as written loses every
/// significant digit near the money -- which is where the formula is used most.
/// Expanded, z/x(z) = 1 + rho z/2 + (2 - 3 rho^2) z^2 / 12 + O(z^3), so the
/// series covers the neighbourhood of zero and the closed form covers the rest.
inline Real z_over_chi(Real z, Real rho) {
    if (std::fabs(z) < 1e-6) {
        return 1.0 + 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0;
    }
    const Real root = std::sqrt(1.0 - 2.0 * rho * z + z * z);
    const Real chi = std::log((root + z - rho) / (1.0 - rho));
    return z / chi;
}

}  // namespace detail

/// Hagan's lognormal (Black) implied volatility.
inline Real sabr_lognormal_vol(const SABRParams& p, Real forward, Real strike, Real expiry) {
    require(p.is_well_formed(), "sabr_lognormal_vol: parameters out of range");
    const Real f = forward + p.shift;
    const Real k = strike + p.shift;
    require(f > 0.0 && k > 0.0,
            "sabr_lognormal_vol: shifted forward and strike must be positive "
            "(increase the shift for negative rates)");
    require(expiry > 0.0, "sabr_lognormal_vol: expiry must be positive");

    const Real one_minus_beta = 1.0 - p.beta;
    const Real log_fk = std::log(f / k);
    const Real fk_pow = std::pow(f * k, 0.5 * one_minus_beta);

    // The T-proportional correction is common to both branches.
    const Real term1 = sqr(one_minus_beta) * p.alpha * p.alpha /
                       (24.0 * std::pow(f * k, one_minus_beta));
    const Real term2 = 0.25 * p.rho * p.beta * p.nu * p.alpha / fk_pow;
    const Real term3 = (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu / 24.0;
    const Real correction = 1.0 + (term1 + term2 + term3) * expiry;

    const Real denom = fk_pow * (1.0 + sqr(one_minus_beta) * sqr(log_fk) / 24.0 +
                                 std::pow(one_minus_beta, 4) * std::pow(log_fk, 4) / 1920.0);

    const Real z = (p.nu / p.alpha) * fk_pow * log_fk;
    return (p.alpha / denom) * detail::z_over_chi(z, p.rho) * correction;
}

/// Hagan's normal (Bachelier) implied volatility.
///
/// Better behaved than the lognormal expansion at low strikes, which is the
/// point: for the same parameters the density stays non-negative over a much
/// wider range. It is also the natural convention when the forward can be
/// negative.
inline Real sabr_normal_vol(const SABRParams& p, Real forward, Real strike, Real expiry) {
    require(p.is_well_formed(), "sabr_normal_vol: parameters out of range");
    const Real f = forward + p.shift;
    const Real k = strike + p.shift;
    require(f > 0.0 && k > 0.0, "sabr_normal_vol: shifted forward and strike must be positive");
    require(expiry > 0.0, "sabr_normal_vol: expiry must be positive");

    const Real one_minus_beta = 1.0 - p.beta;
    const Real log_fk = std::log(f / k);
    const Real fk_pow = std::pow(f * k, 0.5 * one_minus_beta);

    const Real numer = 1.0 + sqr(log_fk) / 24.0 + std::pow(log_fk, 4) / 1920.0;
    const Real denom = 1.0 + sqr(one_minus_beta) * sqr(log_fk) / 24.0 +
                       std::pow(one_minus_beta, 4) * std::pow(log_fk, 4) / 1920.0;

    const Real term1 = -p.beta * (2.0 - p.beta) * p.alpha * p.alpha /
                       (24.0 * std::pow(f * k, one_minus_beta));
    const Real term2 = 0.25 * p.rho * p.alpha * p.nu * p.beta / fk_pow;
    const Real term3 = (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu / 24.0;
    const Real correction = 1.0 + (term1 + term2 + term3) * expiry;

    const Real z = (p.nu / p.alpha) * fk_pow * log_fk;
    return p.alpha * std::pow(f * k, 0.5 * p.beta) * (numer / denom) *
           detail::z_over_chi(z, p.rho) * correction;
}

/// Flattened signatures, so the Python layer can broadcast over strikes without
/// building a parameter object per point.
inline Real sabr_lognormal_vol_scalar(Real alpha, Real beta, Real rho, Real nu, Real shift,
                                      Real forward, Real strike, Real expiry) {
    return sabr_lognormal_vol(SABRParams{alpha, beta, rho, nu, shift}, forward, strike, expiry);
}

inline Real sabr_normal_vol_scalar(Real alpha, Real beta, Real rho, Real nu, Real shift,
                                   Real forward, Real strike, Real expiry) {
    return sabr_normal_vol(SABRParams{alpha, beta, rho, nu, shift}, forward, strike, expiry);
}

/// At-the-money lognormal volatility, the F = K limit taken analytically.
inline Real sabr_atm_vol(const SABRParams& p, Real forward, Real expiry) {
    return sabr_lognormal_vol(p, forward, forward, expiry);
}

/// Solve for the alpha that reproduces a quoted at-the-money volatility.
///
/// Standard market practice: alpha is not quoted, the ATM vol is, and the other
/// three parameters carry the shape. The relation is a cubic in alpha, but
/// solving it by bisection on a monotone function is both shorter and safer than
/// selecting the right root of the cubic -- the cubic has up to three positive
/// roots and picking the wrong one produces a smile that is smooth, plausible
/// and wrong.
inline Real sabr_alpha_from_atm(Real atm_vol, Real forward, Real expiry, Real beta,
                                Real rho, Real nu, Real shift = 0.0) {
    require(atm_vol > 0.0, "sabr_alpha_from_atm: ATM vol must be positive");
    auto vol_of = [&](Real alpha) {
        SABRParams q{alpha, beta, rho, nu, shift};
        return sabr_atm_vol(q, forward, expiry);
    };
    Real lo = 1e-8, hi = std::fmax(1.0, atm_vol * std::pow(forward + shift, 1.0 - beta) * 4.0);
    int guard = 0;
    while (vol_of(hi) < atm_vol && guard++ < 200) hi *= 2.0;
    for (int i = 0; i < 200; ++i) {
        const Real mid = 0.5 * (lo + hi);
        if (vol_of(mid) < atm_vol) lo = mid; else hi = mid;
        if (hi - lo < 1e-14 * std::fmax(hi, 1.0)) break;
    }
    return 0.5 * (lo + hi);
}

/// Where, if anywhere, Hagan's expansion stops being a probability distribution.
///
/// Prices the smile the formula produces, takes the second derivative in strike
/// (Breeden-Litzenberger), and reports the lowest strike at which that is still
/// non-negative. Nothing about SABR is used except its implied vols, so the same
/// scan applies to any smile.
struct SABRArbitrageReport {
    bool free = true;
    /// Highest strike at which the implied density is still negative. Above it
    /// the smile is a distribution; below it, it is not. Zero when no violation
    /// was found. This, not "the lowest arbitrage-free strike", is the number
    /// that matters -- the violated region is an interval running down from this
    /// boundary to zero, so quoting its lower end says nothing.
    Real arbitrage_boundary = 0.0;
    Real min_density = 0.0;
    Real strike_at_min = 0.0;
    int  violations = 0;
    int  points = 0;
};

/// Where, if anywhere, an expansion stops being a probability distribution.
///
/// Prices the smile the formula produces, takes the second derivative in strike
/// (Breeden-Litzenberger), and reports the boundary above which the density is
/// non-negative. Nothing about SABR is used except its implied vols, so the same
/// scan applies to any smile -- which is why the volatility function is a
/// parameter.
template <class VolFn>
inline SABRArbitrageReport smile_density_scan(VolFn&& vol_at, Real forward, Real expiry,
                                              Real k_min_ratio = 0.02,
                                              Real k_max_ratio = 3.0, int n = 1500,
                                              bool normal_measure = false) {
    SABRArbitrageReport rep;
    rep.min_density = DBL_HUGE;
    rep.points = n;

    // The second difference is taken on the OUT-OF-THE-MONEY option, not on the
    // call. Both have the same second derivative in strike -- they differ by
    // F - K, which is linear -- but for K well below the forward the call is
    // worth nearly F - K and differencing three such numbers at a spacing of
    // 1e-5 leaves nothing but rounding. Measured: scanning calls reported 68
    // spurious violations on a smile that has none.
    auto otm = [&](Real K) {
        const Real v = vol_at(K);
        const auto type = (K >= forward) ? OptionType::Call : OptionType::Put;
        if (normal_measure) {
            const Real s = v * std::sqrt(expiry);
            const Real d = (forward - K) / s;
            return (type == OptionType::Call)
                       ? (forward - K) * norm_cdf(d) + s * norm_pdf(d)
                       : (K - forward) * norm_cdf(-d) + s * norm_pdf(d);
        }
        return black76_undiscounted(forward, K, expiry, v, type);
    };

    const Real lo = forward * k_min_ratio, hi = forward * k_max_ratio;
    const Real step = (hi - lo) / Real(n - 1);
    for (int i = 1; i < n - 1; ++i) {
        const Real K = lo + Real(i) * step;
        const Real h = 0.35 * step;
        // Straddling the forward would mix a call and a put in one difference,
        // which reintroduces the F - K term; nudge the stencil to stay on one
        // side when that would happen.
        Real centre = K;
        if ((K - h < forward) != (K + h < forward)) {
            centre = (K < forward) ? forward - 2.0 * h : forward + 2.0 * h;
        }
        const Real density =
            (otm(centre + h) - 2.0 * otm(centre) + otm(centre - h)) / (h * h);
        if (density < rep.min_density) { rep.min_density = density; rep.strike_at_min = K; }
        if (density < 0.0) {
            ++rep.violations;
            rep.arbitrage_boundary = K;   // keeps the highest such strike
        }
    }
    rep.free = rep.violations == 0;
    return rep;
}

inline SABRArbitrageReport sabr_density_scan(const SABRParams& p, Real forward, Real expiry,
                                             Real k_min_ratio = 0.02, Real k_max_ratio = 3.0,
                                             int n = 1500, bool normal_form = false) {
    return smile_density_scan(
        [&](Real K) {
            return normal_form ? sabr_normal_vol(p, forward, K, expiry)
                               : sabr_lognormal_vol(p, forward, K, expiry);
        },
        forward, expiry, k_min_ratio, k_max_ratio, n, normal_form);
}

}  // namespace vse
