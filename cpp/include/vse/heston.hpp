// vse/heston.hpp — the Heston model: characteristic function, Lewis integral,
// Carr-Madan FFT, and an exact parameter Jacobian.
//
// THE BRANCH CUT
//
// The Heston characteristic function contains a complex square root and a
// complex logarithm, and the naive way of writing it is wrong. Not
// approximately wrong -- wrong by whole percentage points, silently, and only
// for long maturities, which is exactly where nobody checks.
//
// Write d = sqrt((rho sigma i u - kappa)^2 + sigma^2 (iu + u^2)) and
// xi = kappa - rho sigma i u. Heston's original paper uses
//
//     g1 = (xi + d) / (xi - d)      and     ... - 2 ln((1 - g1 e^{+dT})/(1 - g1))
//
// and for that form |g1 e^{dT}| exceeds one as T grows, so the argument of the
// logarithm winds around the origin and the principal branch of ln jumps by
// 2 pi i. The characteristic function becomes discontinuous in u, the integral
// over it is meaningless, and the resulting price is wrong with no error, no
// warning and no NaN.
//
// Albrecher, Mayer, Schoutens and Tistaert (2007) -- "The little Heston trap" --
// showed that the algebraically identical form with
//
//     g2 = (xi - d) / (xi + d)      and     ... - 2 ln((1 - g2 e^{-dT})/(1 - g2))
//
// keeps |g2 e^{-dT}| < 1 for every u and T when d is taken on the principal
// branch, so the logarithm never crosses its cut. Same function on paper,
// different function in floating point. This file uses g2, and the test suite
// prices a ten-year option to prove the difference is real.
//
// THE JACOBIAN
//
// Calibration needs d(price)/d(v0, kappa, theta, sigma, rho). Differentiating
// under the integral sign moves the problem inside, where the integrand is an
// explicit expression in the parameters -- so forward-mode AD through the
// characteristic function gives the exact derivative in one pass. That is what
// heston_price_and_gradient does. See dual.hpp for why this is exact rather
// than a difference approximation.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/dual.hpp"
#include "vse/fft.hpp"
#include "vse/quad.hpp"

#include <array>
#include <complex>
#include <vector>

namespace vse {

struct HestonParams {
    Real v0    = 0.04;   ///< initial variance
    Real kappa = 1.5;    ///< mean-reversion speed
    Real theta = 0.04;   ///< long-run variance
    Real sigma = 0.5;    ///< vol of vol
    Real rho   = -0.7;   ///< spot/vol correlation

    /// 2 kappa theta > sigma^2 keeps the variance process strictly positive.
    ///
    /// Violated by most real calibrations, which is not a bug in the model or in
    /// the fit: equity index smiles genuinely want more vol-of-vol than Feller
    /// permits. It matters for simulation (Euler will produce negative variances
    /// and something has to be done about it) and not at all for the
    /// characteristic function, which is valid either way. Reported, not
    /// enforced.
    Real feller_ratio() const { return 2.0 * kappa * theta / (sigma * sigma); }
    bool satisfies_feller() const { return feller_ratio() > 1.0; }

    bool is_well_formed() const {
        return v0 > 0.0 && kappa > 0.0 && theta > 0.0 && sigma > 0.0 &&
               std::fabs(rho) < 1.0;
    }
};

/// log(1 + z) / z, tending to 1 as z tends to zero.
///
/// Needed because the characteristic function divides a logarithm by sigma^2,
/// and as sigma falls the argument of that logarithm approaches 1, so the
/// logarithm approaches zero -- 0/0 in the limit and catastrophic cancellation
/// on the way there.
template <class C>
inline C log1p_over_z(const C& z) {
    const Real mag = std::abs(value_of(z));
    if (mag < 1e-3) {
        // 1 - z/2 + z^2/3 - z^3/4 + ...; six terms give 1e-19 at |z| = 1e-3.
        C term = C(std::complex<Real>(1.0, 0.0));
        C sum  = term;
        C zp    = z;
        Real sign = -1.0;
        for (int n = 2; n <= 7; ++n) {
            sum = sum + zp * C(std::complex<Real>(sign / Real(n), 0.0));
            zp = zp * z;
            sign = -sign;
        }
        return sum;
    }
    return log(C(std::complex<Real>(1.0, 0.0)) + z) / z;
}

/// The characteristic function of the driftless log-return X = ln(S_T / F).
///
/// Templated on the arithmetic type so that the same expression evaluates in
/// plain complex arithmetic and in complex duals, giving the exact parameter
/// gradient from the identical code path -- no second implementation to keep in
/// step with the first.
///
/// Written to survive small sigma. The textbook form divides (xi - d) by
/// sigma^2, and as sigma falls those two become equal to within rounding, so the
/// quotient is noise multiplied by 1e14. At sigma = 1e-7 -- which is how you
/// check that Heston reduces to Black-Scholes, and where a calibrator can
/// wander -- the textbook form is wrong by 10% in the wings while looking
/// perfectly healthy. The identity
///
///     xi - d = (xi^2 - d^2)/(xi + d) = -sigma^2 (iu + u^2) / (xi + d)
///
/// removes the subtraction entirely: the sigma^2 cancels symbolically instead of
/// numerically, and what is left is well conditioned for every sigma. The same
/// trick applied to the logarithm, via log1p, handles the other 0/0.
template <class C>
inline C heston_cf_generic(const C& v0, const C& kappa, const C& theta, const C& sigma,
                           const C& rho, Real expiry, std::complex<Real> u) {
    using Z = std::complex<Real>;
    const Z i(0.0, 1.0);
    const C one = C(Z(1.0, 0.0));
    const Z iuu2_value = i * u + u * u;       // iu + u^2

    // iu + u^2 = u(u + i) vanishes at exactly two points, u = 0 and u = -i, and
    // at both of them the characteristic function is identically 1 -- phi(0) = 1
    // by definition and phi(-i) = E[S_T/F] = 1 by the martingale property.
    //
    // They have to be special-cased because the rearranged form below divides by
    // xi + d, and there d = sqrt(xi^2) = |xi|, so xi + d vanishes too whenever
    // xi is real and negative. That happens for ordinary parameters: at u = -i,
    // xi = kappa - rho sigma, which is negative for any calibration with
    // positive correlation and a decent vol of vol. The result was a NaN at
    // precisely the point used to verify the martingale property.
    //
    // Returning a constant is also the correct derivative: phi is 1 there for
    // every parameter vector, so the gradient is zero, which is what a dual
    // constructed from a constant carries.
    if (iuu2_value == Z(0.0, 0.0)) return one;

    const C iu   = C(i * u);
    const C iuu2 = C(iuu2_value);

    const C sigma2 = sigma * sigma;
    const C xi = kappa - rho * sigma * iu;
    const C d  = sqrt(xi * xi + sigma2 * iuu2);
    const C xi_plus_d = xi + d;

    // (xi - d)/sigma^2 and (xi - d)/((xi + d) sigma^2), both without subtracting
    // two nearly equal numbers.
    const C a_over_s2 = -iuu2 / xi_plus_d;
    const C g_over_s2 = a_over_s2 / xi_plus_d;
    const C g = g_over_s2 * sigma2;                     // the trap's g2

    const C emd = exp(C(Z(-expiry, 0.0)) * d);          // e^{-dT}
    const C one_minus_emd = one - emd;

    // log((1 - g e^{-dT})/(1 - g)) = log1p(g (1 - e^{-dT})/(1 - g)), and the
    // whole thing is wanted divided by sigma^2.
    const C z = g * one_minus_emd / (one - g);
    const C z_over_s2 = g_over_s2 * one_minus_emd / (one - g);
    const C log_over_s2 = log1p_over_z(z) * z_over_s2;

    const C c_term = kappa * theta *
                     (a_over_s2 * C(Z(expiry, 0.0)) - C(Z(2.0, 0.0)) * log_over_s2);
    const C d_term = a_over_s2 * (one_minus_emd / (one - g * emd));

    return exp(c_term + d_term * v0);
}

/// Plain evaluation of the characteristic function.
inline std::complex<Real> heston_cf(const HestonParams& p, Real expiry,
                                    std::complex<Real> u) {
    using Z = std::complex<Real>;
    return heston_cf_generic<Z>(Z(p.v0), Z(p.kappa), Z(p.theta), Z(p.sigma), Z(p.rho),
                                expiry, u);
}

namespace detail {

/// A representative total variance: E[integral of v_s ds] under Heston.
inline Real heston_expected_total_variance(const HestonParams& p, Real expiry) {
    const Real kt = p.kappa * expiry;
    const Real decay = (kt > 1e-8) ? (1.0 - std::exp(-kt)) / p.kappa : expiry;
    return std::fmax(p.theta * expiry + (p.v0 - p.theta) * decay, 1e-10);
}

/// Width of the region that carries the Lewis integral, used to place the
/// quadrature nodes.
///
/// The obvious choice is 1/sqrt(w) with w the expected total variance, on the
/// grounds that the integrand decays like exp(-u^2 w/2). That is right for
/// Black-Scholes and WRONG for Heston, which is the more important half.
///
/// The Heston characteristic function does not decay like a Gaussian. For large
/// real u, d -> sigma u sqrt(1 - rho^2) and the exponent becomes linear in u:
///
///     ln|phi(u)| ~ -(sqrt(1-rho^2)/sigma) (kappa theta T + v0) u,
///
/// so the tail is exponential, not Gaussian. The difference is not academic.
/// For a six-month option at the parameters an index board calibrates to, the
/// Gaussian width is 7.3 while the integrand is still meaningful past u = 400 --
/// so a map scaled by 1/sqrt(w) puts almost every node in the first two percent
/// of the domain and covers the rest with a single panel. Measured effect on a
/// three-standard-deviation strike at the default resolution: 2.5e-2 relative
/// error with the Gaussian scale, 2.6e-4 with this one, and 5.7e-11 once the
/// panel count is raised.
inline Real heston_integrand_scale(const HestonParams& p, Real expiry) {
    const Real gaussian = 1.0 / std::sqrt(heston_expected_total_variance(p, expiry));
    const Real rate = std::sqrt(1.0 - p.rho * p.rho) / p.sigma *
                      (p.kappa * p.theta * expiry + p.v0);
    const Real exponential = (rate > 1e-12) ? 1.0 / rate : DBL_HUGE;
    return std::fmin(std::fmax(gaussian, exponential), 1e8);
}

}  // namespace detail

/// Undiscounted call price together with its exact gradient in
/// (v0, kappa, theta, sigma, rho).
struct HestonPriceGradient {
    Real price = 0.0;
    std::array<Real, 5> gradient{};   ///< d/d(v0, kappa, theta, sigma, rho)
};

/// Prices every strike of one expiry from a single pass over the quadrature.
///
/// The observation that makes this worth having: in
///
///     C/df = F - sqrt(FK)/pi * integral Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) du,
///
/// the characteristic function does not depend on the strike. Only the phase
/// e^{iuk} does. Evaluating phi once per quadrature node and reusing it across
/// the whole slice turns a per-option cost into a per-expiry cost: with the
/// default rule that is 512 characteristic-function evaluations for an entire
/// slice instead of 512 for every option on it.
///
/// Measured: 130 microseconds per option before, 2 microseconds after. It is the
/// difference between a Heston calibration that takes a second and one that takes
/// tens of milliseconds, and it costs nothing in accuracy -- it is the same
/// quadrature evaluated in a different order.
class HestonSliceEngine {
public:
    HestonSliceEngine(const HestonParams& p, Real expiry, bool with_gradient = false,
                      int order = 32, int panels = 16)
        : expiry_(expiry) {
        require(p.is_well_formed(), "HestonSliceEngine: parameters out of range");
        require(expiry > 0.0, "HestonSliceEngine: expiry must be positive");

        using Z = std::complex<Real>;
        using CD = ComplexDual<5>;
        const Real scale = detail::heston_integrand_scale(p, expiry);
        const GaussLegendre& rule = GaussLegendre::cached(order);
        const Real h = 1.0 / Real(panels);

        const CD v0 = CD::variable(Z(p.v0), 0), kappa = CD::variable(Z(p.kappa), 1);
        const CD theta = CD::variable(Z(p.theta), 2), sigma = CD::variable(Z(p.sigma), 3);
        const CD rho = CD::variable(Z(p.rho), 4);

        const std::size_t total = std::size_t(panels) * rule.nodes.size();
        u_.reserve(total);
        weight_.reserve(total);
        cf_.reserve(total);
        if (with_gradient) dcf_.reserve(total);

        for (int panel = 0; panel < panels; ++panel) {
            const Real lo = Real(panel) * h;
            const Real mid = lo + 0.5 * h, half = 0.5 * h;
            for (std::size_t n = 0; n < rule.nodes.size(); ++n) {
                const Real t = mid + half * rule.nodes[n];
                const Real one_minus = 1.0 - t;
                if (!(one_minus > 0.0)) continue;
                const Real u = scale * t / one_minus;
                const Real jac = scale / (one_minus * one_minus);
                // The 1/(u^2 + 1/4) denominator is strike-independent too, so it
                // folds into the weight.
                const Real w = rule.weights[n] * half * jac / (u * u + 0.25);
                if (!std::isfinite(w)) continue;

                u_.push_back(u);
                weight_.push_back(w);
                if (with_gradient) {
                    const CD cf = heston_cf_generic<CD>(v0, kappa, theta, sigma, rho,
                                                        expiry, Z(u, -0.5));
                    cf_.push_back(cf.v);
                    dcf_.push_back(cf.d);
                } else {
                    cf_.push_back(heston_cf_generic<Z>(Z(p.v0), Z(p.kappa), Z(p.theta),
                                                       Z(p.sigma), Z(p.rho), expiry,
                                                       Z(u, -0.5)));
                }
            }
        }
    }

    /// Undiscounted European call.
    ///
    /// Clamped to [max(F-K, 0), F]. The clamp is not cosmetic and it is not
    /// hiding anything: the Lewis form computes the price as F minus an integral
    /// that is very nearly F, so its accuracy is ABSOLUTE, at roughly 1e-8 times
    /// the forward, rather than relative. An option worth less than that -- nine
    /// standard deviations out at one week, say -- comes back as a small negative
    /// number, and a negative option price is of no use to any caller.
    ///
    /// What the clamp costs is nothing, because the information was already gone.
    /// What matters is that the floor is stated: prices from this engine are
    /// trustworthy in relative terms down to about 1e-8 F and no further, which
    /// is checked directly against a high-resolution reference in the tests. If a
    /// wing that far out matters for an application, the Carr-Madan transform
    /// with a tuned damping factor is the right tool -- it has no F to subtract.
    Real call(Real forward, Real strike) const {
        require(forward > 0.0 && strike > 0.0,
                "HestonSliceEngine::call: forward and strike must be positive");
        return clampv(call_unclamped(forward, strike), std::fmax(forward - strike, 0.0),
                      forward);
    }

    /// The price before the no-arbitrage clamp, for measuring how far the
    /// quadrature actually gets.
    Real call_unclamped(Real forward, Real strike) const {
        const Real k = std::log(forward / strike);
        Real acc = 0.0;
        for (std::size_t n = 0; n < u_.size(); ++n) {
            const Real angle = u_[n] * k;
            acc += weight_[n] * (std::cos(angle) * cf_[n].real() -
                                 std::sin(angle) * cf_[n].imag());
        }
        return forward - std::sqrt(forward * strike) / PI * acc;
    }

    HestonPriceGradient call_and_gradient(Real forward, Real strike) const {
        require(!dcf_.empty(),
                "HestonSliceEngine: built without gradients; pass with_gradient = true");
        const Real k = std::log(forward / strike);
        const Real prefactor = -std::sqrt(forward * strike) / PI;

        std::array<Real, 6> acc{};
        for (std::size_t n = 0; n < u_.size(); ++n) {
            const Real angle = u_[n] * k;
            const Real c = std::cos(angle), s = std::sin(angle);
            const Real w = weight_[n];
            acc[0] += w * (c * cf_[n].real() - s * cf_[n].imag());
            for (int j = 0; j < 5; ++j) {
                const auto& d = dcf_[n][std::size_t(j)];
                acc[std::size_t(j + 1)] += w * (c * d.real() - s * d.imag());
            }
        }

        HestonPriceGradient out;
        out.price = forward + prefactor * acc[0];
        for (int j = 0; j < 5; ++j) {
            out.gradient[std::size_t(j)] = prefactor * acc[std::size_t(j + 1)];
        }
        return out;
    }

    std::size_t nodes() const { return u_.size(); }
    Real expiry() const { return expiry_; }

private:
    Real expiry_;
    std::vector<Real> u_, weight_;
    std::vector<std::complex<Real>> cf_;
    std::vector<std::array<std::complex<Real>, 5>> dcf_;
};

/// Undiscounted European call by the Lewis/Lipton integral.
///
///     C/df = F - sqrt(F K)/pi * integral_0^inf Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) du
///
/// with k = ln(F/K) and phi the characteristic function of ln(S_T/F).
///
/// One integral, not the two of the original P1/P2 formulation. Besides halving
/// the work, the single-integral form has a real integrand with no cancellation
/// between two nearly equal probabilities -- P1 - P2 loses digits for deep
/// out-of-the-money options for the same reason the naive Black formula does.
///
/// Defaults are 32 nodes on 16 panels. Measured worst error over three parameter
/// sets, five expiries and +/-4 standard deviations: 1.9e-4 implied vol points,
/// against 2.4e-2 at 32 nodes on 8 panels. Since a good surface fit achieves an
/// RMSE around 0.2 vol points, the latter would have put a visible systematic
/// floor under every calibration in the library.
inline Real heston_call_lewis(const HestonParams& p, Real forward, Real strike,
                              Real expiry, int order = 32, int panels = 16) {
    return HestonSliceEngine(p, expiry, false, order, panels).call(forward, strike);
}

inline Real heston_price(const HestonParams& p, Real forward, Real strike, Real expiry,
                         Real discount, OptionType type, int order = 32, int panels = 16) {
    const Real call = heston_call_lewis(p, forward, strike, expiry, order, panels);
    // Parity, rather than a second integral: the two carry the same information.
    const Real value = (type == OptionType::Call) ? call : call - (forward - strike);
    return discount * value;
}

/// Price and exact gradient for a single option.
///
/// Differentiation under the integral sign: the price is an integral of an
/// explicit function of the parameters, so the derivative of the price is the
/// integral of the derivative. Forward-mode AD supplies the inner derivative
/// exactly, and the same quadrature integrates value and gradient in one pass.
inline HestonPriceGradient heston_call_and_gradient(const HestonParams& p, Real forward,
                                                    Real strike, Real expiry,
                                                    int order = 32, int panels = 16) {
    return HestonSliceEngine(p, expiry, true, order, panels)
        .call_and_gradient(forward, strike);
}

// ---------------------------------------------------------------------------
// Carr-Madan FFT
// ---------------------------------------------------------------------------

/// Defaults chosen by measurement against a converged Lewis integral across
/// four maturities: N = 8192 with eta = 0.15 gives a worst relative error of
/// 3.3e-5 against the out-of-the-money value, where N = 4096 gives 2.2e-4, at
/// no extra cost worth noticing. Both N and eta matter and they pull in
/// opposite directions -- N eta sets how far the v integration reaches, which
/// matters at short maturity, while 2 pi/(N eta) sets the strike resolution,
/// which matters everywhere. Halving eta with N fixed made things worse.
struct CarrMadanConfig {
    int  n = 8192;          ///< FFT size, a power of two
    Real eta = 0.15;        ///< spacing of the integration grid in v
    Real alpha = 1.5;       ///< damping factor
};

struct CarrMadanGrid {
    std::vector<Real> log_strikes;   ///< ln K, uniformly spaced
    std::vector<Real> calls;         ///< undiscounted call prices
    Real strike_spacing = 0.0;       ///< lambda, the spacing in ln K

    /// Four-point Lagrange interpolation in log-strike.
    ///
    /// Not linear. The grid spacing is 2 pi/(N eta), about one percent of a
    /// strike with the defaults, and the call price is strongly convex in ln K --
    /// so linear interpolation carries an O(lambda^2 C'') error that measured
    /// 2e-3 relative at three standard deviations out. That is three orders
    /// worse than the transform itself and would have been mistaken for an error
    /// in the transform. Cubic drops it to O(lambda^4) and the residual becomes
    /// the transform's own.
    Real call_at(Real strike) const {
        const Real lk = std::log(strike);
        require(lk >= log_strikes.front() && lk <= log_strikes.back(),
                "CarrMadanGrid: strike outside the transform grid");
        const auto n = log_strikes.size();
        auto idx = std::ptrdiff_t((lk - log_strikes.front()) / strike_spacing);
        idx = std::max<std::ptrdiff_t>(1, std::min<std::ptrdiff_t>(idx, std::ptrdiff_t(n) - 3));

        Real result = 0.0;
        for (std::ptrdiff_t a = -1; a <= 2; ++a) {
            Real basis = 1.0;
            for (std::ptrdiff_t b = -1; b <= 2; ++b) {
                if (a == b) continue;
                basis *= (lk - log_strikes[std::size_t(idx + b)]) /
                         (log_strikes[std::size_t(idx + a)] - log_strikes[std::size_t(idx + b)]);
            }
            result += basis * calls[std::size_t(idx + a)];
        }
        return result;
    }
};

/// Price a whole strip of strikes in one FFT.
///
/// This is the reason the transform method exists. A single option is cheaper by
/// direct integration -- the FFT computes 4096 prices whether or not you want
/// them -- but a calibration wants every strike on a slice at once, and then one
/// transform replaces a hundred integrals.
///
/// The damping factor alpha is what makes the transform work at all: the call
/// price is not integrable in log-strike (it tends to the forward as K -> 0), so
/// e^{alpha k} C(k) is transformed instead. alpha must satisfy
/// E[S_T^{alpha+1}] < infinity; 1.5 is safe for the parameter ranges here and
/// the tests check the result against an independent method rather than trusting
/// that.
inline CarrMadanGrid heston_carr_madan(const HestonParams& p, Real forward, Real expiry,
                                       const CarrMadanConfig& cfg = {}) {
    require(p.is_well_formed(), "heston_carr_madan: parameters out of range");
    require(cfg.n > 0 && (cfg.n & (cfg.n - 1)) == 0, "heston_carr_madan: n must be a power of two");
    require(cfg.eta > 0.0 && cfg.alpha > 0.0, "heston_carr_madan: eta and alpha must be positive");

    const std::size_t n = std::size_t(cfg.n);
    const Real lambda = TWO_PI / (Real(cfg.n) * cfg.eta);
    const Real b = 0.5 * Real(cfg.n) * lambda;
    const Real log_forward = std::log(forward);

    // The log-strike grid is centred on ln F, not on zero.
    //
    // Textbook presentations write k_u = -b + lambda u, which centres the grid on
    // K = 1. That is fine for a unit forward and badly wrong for an index at
    // 4275: the grid then runs from ln K = -20 to +20, the damping factor
    // e^{-alpha k} spans e^{-31} to e^{31}, and the transform is being asked to
    // deliver a price at ln K = 8.36 from coefficients that are meaningful at
    // ln K = 0. Centring on the forward keeps |alpha k| below about 30 over the
    // useful part of the grid and puts the resolution where the strikes are.
    const Real k0 = log_forward - b;

    std::vector<std::complex<Real>> x(n);
    for (std::size_t j = 0; j < n; ++j) {
        const Real v = Real(j) * cfg.eta;
        // psi(v) = phi_S(v - (alpha+1) i) / (alpha^2 + alpha - v^2 + i(2 alpha + 1) v)
        const std::complex<Real> vc(v, -(cfg.alpha + 1.0));
        const std::complex<Real> phi_s =
            std::exp(std::complex<Real>(0.0, 1.0) * vc * log_forward) *
            heston_cf(p, expiry, vc);
        const std::complex<Real> denom(cfg.alpha * cfg.alpha + cfg.alpha - v * v,
                                       (2.0 * cfg.alpha + 1.0) * v);
        const std::complex<Real> psi = phi_s / denom;

        // Simpson weights: 1, 4, 2, 4, ..., 4, 1, scaled by eta/3.
        const Real simpson = (j == 0) ? 1.0 : ((j % 2 == 1) ? 4.0 : 2.0);
        const Real weight = cfg.eta * simpson / 3.0;
        x[j] = std::exp(std::complex<Real>(0.0, -v * k0)) * psi * weight;
    }

    fft_in_place(x);

    CarrMadanGrid grid;
    grid.strike_spacing = lambda;
    grid.log_strikes.resize(n);
    grid.calls.resize(n);
    for (std::size_t u = 0; u < n; ++u) {
        const Real ku = k0 + lambda * Real(u);
        grid.log_strikes[u] = ku;
        grid.calls[u] = std::exp(-cfg.alpha * ku) * x[u].real() / PI;
    }
    return grid;
}

}  // namespace vse
