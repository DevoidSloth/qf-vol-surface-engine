// vse/bates.hpp — Bates (1996): Heston stochastic volatility with Merton jumps.
//
// Why bother, given Heston already fits reasonably: Heston cannot produce a
// short-dated smile. A diffusion needs time to generate skew -- over a week the
// stochastic-vol contribution to the smile is O(sqrt(T)) and vanishes, so a
// Heston fit to a full board either matches the front and misses the back or
// matches the back and prices the front's wings at a fraction of their value.
// Anyone who has calibrated Heston to an index board has seen the one-week slice
// come out visibly too flat.
//
// Jumps do not vanish with maturity. A one-week option is priced off the
// probability of a jump in that week, which is O(lambda T) but the *payoff*
// given a jump is O(1), so the wing value stays finite as T falls. That is
// exactly the missing piece, and it is why the front end of an index surface is
// where jump models earn their keep.
//
// The characteristic function is a product, because the jump component is
// independent of the diffusion: phi_Bates = phi_Heston * phi_jump. Everything
// downstream -- the Lewis integral, the Carr-Madan transform, the parameter
// gradient -- works unchanged.
#pragma once

#include "vse/common.hpp"
#include "vse/dual.hpp"
#include "vse/heston.hpp"

#include <complex>

namespace vse {

struct BatesParams {
    HestonParams heston;
    Real lambda   = 0.3;    ///< jump intensity, jumps per year
    Real jump_mean = -0.08; ///< mean of the log jump size
    Real jump_vol  = 0.15;  ///< standard deviation of the log jump size

    bool is_well_formed() const {
        return heston.is_well_formed() && lambda >= 0.0 && jump_vol > 0.0;
    }

    /// Expected proportional jump size, E[e^Y] - 1. This is what the drift has
    /// to be compensated by so that the discounted forward stays a martingale.
    Real expected_jump() const {
        return std::exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0;
    }
};

/// Characteristic function of the jump component of ln(S_T/F), compensated.
///
/// exp( lambda T (e^{i u m - delta^2 u^2 / 2} - 1) - i u lambda T (e^{m + delta^2/2} - 1) )
///
/// The second term is the compensator. Without it the model is not risk-neutral:
/// adding jumps with a negative mean would lower the forward, and every price
/// would be quietly biased. Its presence is checkable -- phi(-i) must be exactly
/// 1 -- and the test suite checks it rather than trusting the algebra.
template <class C>
inline C merton_jump_cf_generic(const C& lambda, const C& jump_mean, const C& jump_vol,
                                Real expiry, std::complex<Real> u) {
    using Z = std::complex<Real>;
    const Z i(0.0, 1.0);
    const C one = C(Z(1.0, 0.0));
    const C iu = C(i * u);
    const C u2 = C(Z(u.real() * u.real() - u.imag() * u.imag(),
                     2.0 * u.real() * u.imag()));

    const C half = C(Z(0.5, 0.0));
    const C jump_var = jump_vol * jump_vol;
    const C char_jump = exp(iu * jump_mean - half * jump_var * u2);
    const C compensator = exp(jump_mean + half * jump_var) - one;

    return exp(C(Z(expiry, 0.0)) * lambda * ((char_jump - one) - iu * compensator));
}

template <class C>
inline C bates_cf_generic(const C& v0, const C& kappa, const C& theta, const C& sigma,
                          const C& rho, const C& lambda, const C& jump_mean,
                          const C& jump_vol, Real expiry, std::complex<Real> u) {
    return heston_cf_generic<C>(v0, kappa, theta, sigma, rho, expiry, u) *
           merton_jump_cf_generic<C>(lambda, jump_mean, jump_vol, expiry, u);
}

inline std::complex<Real> bates_cf(const BatesParams& p, Real expiry, std::complex<Real> u) {
    using Z = std::complex<Real>;
    return bates_cf_generic<Z>(Z(p.heston.v0), Z(p.heston.kappa), Z(p.heston.theta),
                               Z(p.heston.sigma), Z(p.heston.rho), Z(p.lambda),
                               Z(p.jump_mean), Z(p.jump_vol), expiry, u);
}

/// Undiscounted European call by the Lewis integral, as for Heston.
inline Real bates_call_lewis(const BatesParams& p, Real forward, Real strike, Real expiry,
                             int order = 32, int panels = 16) {
    require(p.is_well_formed(), "bates_call_lewis: parameters out of range");
    require(forward > 0.0 && strike > 0.0 && expiry > 0.0,
            "bates_call_lewis: forward, strike and expiry must be positive");

    const Real k = std::log(forward / strike);
    // The same scale as Heston, and deliberately so. It is tempting to widen it
    // for the jump variance, but the jump factor does not keep decaying: its
    // modulus is exp(lambda T (e^{-delta^2 u^2/2} cos(u m) - 1)), which falls
    // from 1 to exp(-lambda T) over u ~ 1/delta and is then flat. So the tail of
    // the Bates integrand is governed by the diffusion exactly as in Heston, and
    // scaling for the jumps would put the nodes in the wrong place -- which
    // showed up as Bates and Heston disagreeing by 5e-7 at lambda = 0, where
    // they are the same model.
    const Real scale = detail::heston_integrand_scale(p.heston, expiry);

    const Real integral = integrate_semi_infinite(
        [&](Real u) {
            const std::complex<Real> z(u, -0.5);
            const std::complex<Real> num =
                std::exp(std::complex<Real>(0.0, u * k)) * bates_cf(p, expiry, z);
            return num.real() / (u * u + 0.25);
        },
        scale, order, panels);

    // Clamped for the same reason as the Heston engine: the price is a
    // difference of two numbers close to F, so the accuracy is absolute.
    return clampv(forward - std::sqrt(forward * strike) / PI * integral,
                  std::fmax(forward - strike, 0.0), forward);
}

inline Real bates_price(const BatesParams& p, Real forward, Real strike, Real expiry,
                        Real discount, OptionType type, int order = 32, int panels = 16) {
    const Real call = bates_call_lewis(p, forward, strike, expiry, order, panels);
    const Real value = (type == OptionType::Call) ? call : call - (forward - strike);
    return discount * value;
}

}  // namespace vse
