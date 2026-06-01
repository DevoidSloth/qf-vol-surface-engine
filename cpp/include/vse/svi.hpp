// vse/svi.hpp — SVI, SSVI and eSSVI parameterisations of total implied variance,
// with the no-arbitrage conditions written down and checkable.
//
// Everything here works in total variance w(k) = sigma^2(k) T against
// log-moneyness k = ln(K/F). Not sigma, not k = ln(K/S). Both choices matter:
//
//   * Total variance is the quantity calendar-spread arbitrage is a statement
//     about (w must be non-decreasing in T at fixed k), and the quantity the
//     SVI functional form is affine in at the wings.
//   * The forward, not the spot. A surface fitted against ln(K/S) absorbs the
//     carry into the smile and comes out visibly tilted -- it is the single most
//     common error in amateur surface work, and it is invisible unless you
//     compare against a surface that got it right.
//
// Two distinct notions of arbitrage are checked, and the distinction is worth
// keeping straight because they fail in different places:
//
//   BUTTERFLY (static, within one slice). C(K) must be convex in K, equivalently
//   the risk-neutral density must be non-negative. Gatheral's function
//       g(k) = (1 - k w'/(2w))^2 - (w'/4)^2 (4/w + 1) + w''/2
//   is proportional to that density, so g >= 0 everywhere is the condition. A
//   slice can fit with a beautiful RMSE and still have g < 0 in the wings, and
//   then it is not a probability distribution and every exotic priced off it is
//   wrong. This is why the density plot matters more than the fit statistic.
//
//   CALENDAR (across slices). w(k, T) must be non-decreasing in T at every k.
//   Violations here are not subtle -- they let you buy a calendar spread for a
//   negative price -- but they are easy to introduce by fitting each expiry
//   independently, which is exactly what per-slice SVI does.
//
// SSVI exists to make the second one structural: a single function of the ATM
// total variance generates every slice, and Gatheral-Jacquier give conditions on
// that function under which no calendar arbitrage is possible at all.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"

#include <algorithm>
#include <vector>

namespace vse {

// ---------------------------------------------------------------------------
// Raw SVI, one slice
// ---------------------------------------------------------------------------

/// w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]
struct SVIRaw {
    Real a     = 0.0;    ///< vertical level of total variance
    Real b     = 0.0;    ///< wing slope scale, b >= 0
    Real rho   = 0.0;    ///< skew, |rho| < 1
    Real m     = 0.0;    ///< horizontal shift
    Real sigma = 0.1;    ///< smile curvature at the vertex, sigma > 0

    Real total_variance(Real k) const {
        const Real y = k - m;
        return a + b * (rho * y + std::sqrt(y * y + sigma * sigma));
    }

    Real dw(Real k) const {
        const Real y = k - m;
        return b * (rho + y / std::sqrt(y * y + sigma * sigma));
    }

    Real d2w(Real k) const {
        const Real y = k - m;
        const Real r = std::sqrt(y * y + sigma * sigma);
        return b * sigma * sigma / (r * r * r);
    }

    Real implied_vol(Real k, Real expiry) const {
        return std::sqrt(std::fmax(total_variance(k), 0.0) / expiry);
    }

    /// Minimum of w over k, attained at k = m - rho sigma / sqrt(1 - rho^2).
    Real min_variance() const { return a + b * sigma * std::sqrt(1.0 - rho * rho); }

    /// Asymptotic slopes of w: b(1 - rho) as k -> -infinity, b(1 + rho) as
    /// k -> +infinity. Lee's moment formula caps both at 2.
    Real left_slope() const { return b * (1.0 - rho); }
    Real right_slope() const { return b * (1.0 + rho); }

    /// Structural validity: the parameters describe a positive total variance
    /// with finite moments. This is necessary for no arbitrage, not sufficient.
    bool is_well_formed() const {
        return b >= 0.0 && std::fabs(rho) < 1.0 && sigma > 0.0 &&
               min_variance() >= 0.0 && left_slope() <= 2.0 && right_slope() <= 2.0;
    }
};

// ---------------------------------------------------------------------------
// Butterfly arbitrage: Gatheral's g, and the density it is proportional to
// ---------------------------------------------------------------------------

/// g(k) = (1 - k w'/(2w))^2 - (w'/4)^2 (4/w + 1) + w''/2.
///
/// Non-negative everywhere iff the slice implies a non-negative density.
template <class Slice>
inline Real durrleman_g(const Slice& s, Real k) {
    const Real w  = s.total_variance(k);
    if (!(w > 0.0)) return -DBL_HUGE;
    const Real w1 = s.dw(k), w2 = s.d2w(k);
    const Real t1 = 1.0 - k * w1 / (2.0 * w);
    return t1 * t1 - 0.25 * sqr(0.5 * w1) * (4.0 / w + 1.0) + 0.5 * w2;
}

/// Risk-neutral density of log-moneyness k = ln(K/F), implied by the slice.
///
/// p(k) = g(k) / sqrt(2 pi w(k)) * exp(-d2(k)^2 / 2), with
/// d2(k) = -k/sqrt(w) - sqrt(w)/2. Integrates to one over k when the slice is
/// arbitrage-free, which the tests check by quadrature -- a density that is
/// non-negative but does not integrate to one means the formula is wrong, and
/// the two failures are easy to confuse.
template <class Slice>
inline Real risk_neutral_density(const Slice& s, Real k) {
    const Real w = s.total_variance(k);
    if (!(w > 0.0)) return 0.0;
    const Real sqrt_w = std::sqrt(w);
    const Real d2 = -k / sqrt_w - 0.5 * sqrt_w;
    return durrleman_g(s, k) / (SQRT_TWO_PI * sqrt_w) * std::exp(-0.5 * d2 * d2);
}

struct ButterflyReport {
    bool free = true;
    Real min_g = 0.0;           ///< most negative value of g found
    Real k_at_min = 0.0;
    Real min_density = 0.0;
    Real density_integral = 0.0; ///< should be 1
    int  violations = 0;         ///< grid points with g < 0
};

/// Scan a slice for butterfly arbitrage.
///
/// A scan, not a proof. g is smooth and its violations are broad regions rather
/// than isolated spikes, so a grid this fine does not miss them in practice --
/// but the honest description is "no violation found on a 2001-point grid
/// spanning six wing standard deviations", and that is what the report means.
/// check_ssvi_conditions() gives the structural statement that holds everywhere
/// by construction; this gives the empirical one that holds for the object
/// actually being shipped.
///
/// `expiry` is accepted but unused: the scan window is set from total variance,
/// which already carries the maturity. It stays in the signature because every
/// call site has it and dropping it would make the calls read as if a slice were
/// maturity-free, which is exactly the confusion total variance exists to avoid.
template <class Slice>
inline ButterflyReport check_butterfly(const Slice& s, [[maybe_unused]] Real expiry,
                                       Real k_half_width = 0.0, int n = 2001) {
    if (k_half_width <= 0.0) {
        // Six standard deviations, but measured with the *wing* volatility, not
        // the at-the-money one. On a steep short-dated smile the wings can carry
        // two or three times the ATM vol, so a window set from sqrt(w(0)) covers
        // barely two effective standard deviations and leaves over a percent of
        // the mass outside -- which then shows up as a density that does not
        // integrate to one and looks like a bug in the formula. Two passes are
        // enough because w grows at most linearly in |k| (Lee), so sqrt(w) grows
        // like sqrt(|k|) and the iteration contracts.
        k_half_width = 6.0 * std::sqrt(std::fmax(s.total_variance(0.0), 1e-10));
        for (int pass = 0; pass < 3; ++pass) {
            const Real w_wing = std::fmax(s.total_variance(k_half_width),
                                          s.total_variance(-k_half_width));
            k_half_width = 6.0 * std::sqrt(std::fmax(w_wing, 1e-10));
        }
    }
    if (n % 2 == 0) ++n;   // Simpson needs an odd number of points

    ButterflyReport rep;
    rep.min_g = DBL_HUGE;
    rep.min_density = DBL_HUGE;

    const Real dk = 2.0 * k_half_width / Real(n - 1);
    Real integral = 0.0;
    for (int i = 0; i < n; ++i) {
        const Real k = -k_half_width + i * dk;
        const Real g = durrleman_g(s, k);
        const Real p = risk_neutral_density(s, k);
        if (g < rep.min_g) { rep.min_g = g; rep.k_at_min = k; }
        rep.min_density = std::fmin(rep.min_density, p);
        if (g < 0.0) ++rep.violations;
        // Composite Simpson: the density is a sharply peaked smooth function and
        // the trapezium rule leaves a visible O(h^2) deficit at the peak, which
        // is indistinguishable from missing tail mass in the reported number.
        const Real weight = (i == 0 || i == n - 1) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        integral += weight * p;
    }
    rep.density_integral = integral * dk / 3.0;
    rep.free = rep.violations == 0;
    return rep;
}

// ---------------------------------------------------------------------------
// SSVI
// ---------------------------------------------------------------------------

/// Power-law curvature function phi(theta) = eta / (theta^gamma (1+theta)^{1-gamma}).
///
/// The (1+theta)^{1-gamma} factor is Gatheral-Jacquier's, and it is not
/// cosmetic: the bare eta / theta^gamma form violates the calendar condition for
/// large theta, i.e. exactly at the long end where a surface is most often
/// extrapolated.
struct PowerLawPhi {
    Real eta   = 1.0;
    Real gamma = 0.5;

    Real operator()(Real theta) const {
        return eta / (std::pow(theta, gamma) * std::pow(1.0 + theta, 1.0 - gamma));
    }

    /// d(theta phi(theta))/d(theta), which is what the calendar condition bounds.
    Real d_theta_phi(Real theta) const {
        const Real p = (*this)(theta);
        // theta phi = eta theta^{1-gamma} (1+theta)^{gamma-1}
        // d/dtheta = theta phi * [ (1-gamma)/theta + (gamma-1)/(1+theta) ]
        return theta * p * ((1.0 - gamma) / theta + (gamma - 1.0) / (1.0 + theta));
    }
};

/// One SSVI slice at a given ATM total variance theta.
///
/// w(k) = theta/2 * [ 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) ]
struct SSVISlice {
    Real theta = 0.04;
    Real rho   = -0.5;
    Real phi   = 1.0;

    Real total_variance(Real k) const {
        const Real z = phi * k + rho;
        return 0.5 * theta * (1.0 + rho * phi * k + std::sqrt(z * z + 1.0 - rho * rho));
    }

    Real dw(Real k) const {
        const Real z = phi * k + rho;
        return 0.5 * theta * phi * (rho + z / std::sqrt(z * z + 1.0 - rho * rho));
    }

    Real d2w(Real k) const {
        const Real z = phi * k + rho;
        const Real r = std::sqrt(z * z + 1.0 - rho * rho);
        return 0.5 * theta * phi * phi * (1.0 - rho * rho) / (r * r * r);
    }

    Real implied_vol(Real k, Real expiry) const {
        return std::sqrt(std::fmax(total_variance(k), 0.0) / expiry);
    }

    /// The equivalent raw-SVI parameters, which is how an SSVI slice is handed
    /// to code that only speaks SVI.
    SVIRaw to_raw() const {
        SVIRaw s;
        s.b     = 0.5 * theta * phi;
        s.rho   = rho;
        s.m     = -rho / phi;
        s.sigma = std::sqrt(1.0 - rho * rho) / phi;
        s.a     = 0.5 * theta * (1.0 - rho * rho);
        return s;
    }
};

struct SSVIConditionReport {
    bool butterfly_free = true;
    bool calendar_free  = true;
    // Butterfly, Gatheral-Jacquier Theorem 4.2 (sufficient):
    Real bf_condition_1 = 0.0;   ///< theta phi (1 + |rho|), must be < 4
    Real bf_condition_2 = 0.0;   ///< theta phi^2 (1 + |rho|), must be <= 4
    // Calendar, Gatheral-Jacquier Theorem 4.2:
    Real cal_lower = 0.0;        ///< d(theta phi)/dtheta, must be >= 0
    Real cal_upper_bound = 0.0;  ///< (1/rho^2)(1 + sqrt(1-rho^2)) phi
    bool theta_increasing = true;
};

/// The closed-form SSVI conditions, evaluated rather than assumed.
///
/// These are the conditions of Gatheral & Jacquier, "Arbitrage-free SVI
/// volatility surfaces" (2014), Theorem 4.2. The butterfly pair is sufficient,
/// not necessary -- a surface can fail them and still have a non-negative
/// density -- which is why check_butterfly() also exists and why both numbers
/// end up in the report.
inline SSVIConditionReport check_ssvi_conditions(Real theta, Real rho, Real phi,
                                                 Real d_theta_phi) {
    SSVIConditionReport rep;
    const Real abs_rho = std::fabs(rho);

    rep.bf_condition_1 = theta * phi * (1.0 + abs_rho);
    rep.bf_condition_2 = theta * phi * phi * (1.0 + abs_rho);
    rep.butterfly_free = rep.bf_condition_1 < 4.0 && rep.bf_condition_2 <= 4.0;

    rep.cal_lower = d_theta_phi;
    rep.cal_upper_bound = (abs_rho > 0.0)
        ? (1.0 / (rho * rho)) * (1.0 + std::sqrt(1.0 - rho * rho)) * phi
        : DBL_HUGE;   // the bound is vacuous at rho = 0
    rep.calendar_free = d_theta_phi >= 0.0 && d_theta_phi <= rep.cal_upper_bound;

    return rep;
}

/// A whole SSVI surface: one curvature function, one skew, and the ATM total
/// variance term structure.
struct SSVISurface {
    PowerLawPhi phi;
    Real rho = -0.5;
    std::vector<Real> expiries;   ///< strictly increasing
    std::vector<Real> theta;      ///< ATM total variance at each expiry

    SSVISlice slice_at_index(std::size_t i) const {
        SSVISlice s;
        s.theta = theta[i];
        s.rho   = rho;
        s.phi   = phi(theta[i]);
        return s;
    }

    /// theta at an arbitrary maturity, linear in T between the fitted expiries.
    ///
    /// Linear interpolation of *total* variance, not of vol: that is what keeps
    /// the interpolated slices calendar-arbitrage-free, since a linear
    /// interpolant of a non-decreasing sequence is non-decreasing. Interpolating
    /// implied vol instead introduces calendar arbitrage between the pillars
    /// even when every pillar is fine.
    Real theta_at(Real t) const {
        require(!expiries.empty(), "SSVISurface: no expiries");
        if (t <= expiries.front()) return theta.front() * t / expiries.front();
        if (t >= expiries.back()) {
            // Flat total-variance extrapolation in vol terms: theta grows
            // linearly in t, which preserves monotonicity.
            return theta.back() * t / expiries.back();
        }
        const auto it = std::lower_bound(expiries.begin(), expiries.end(), t);
        const std::size_t hi = std::size_t(it - expiries.begin());
        const std::size_t lo = hi - 1;
        const Real u = (t - expiries[lo]) / (expiries[hi] - expiries[lo]);
        return theta[lo] + u * (theta[hi] - theta[lo]);
    }

    SSVISlice slice_at(Real t) const {
        SSVISlice s;
        s.theta = theta_at(t);
        s.rho   = rho;
        s.phi   = phi(s.theta);
        return s;
    }

    Real total_variance(Real k, Real t) const { return slice_at(t).total_variance(k); }

    Real implied_vol(Real k, Real t) const {
        return std::sqrt(std::fmax(total_variance(k, t), 0.0) / t);
    }
};

struct CalendarReport {
    bool free = true;
    Real worst_decrease = 0.0;   ///< most negative w(k,T2) - w(k,T1)
    Real k_at_worst = 0.0;
    Real t_at_worst = 0.0;
    int  violations = 0;
};

/// Direct numerical certificate of calendar-spread freedom: total variance is
/// non-decreasing in T at every log-moneyness on the grid.
///
/// This is checked *in addition to* the parametric condition, because they can
/// disagree in both directions -- the parametric one covers all k but only holds
/// for exact SSVI, while this one covers the actual object being shipped,
/// interpolation and all.
template <class Surface>
inline CalendarReport check_calendar(const Surface& surface, const std::vector<Real>& times,
                                     Real k_half_width = 1.5, int n_k = 401) {
    CalendarReport rep;
    const Real dk = 2.0 * k_half_width / Real(n_k - 1);
    for (std::size_t j = 1; j < times.size(); ++j) {
        for (int i = 0; i < n_k; ++i) {
            const Real k = -k_half_width + i * dk;
            const Real d = surface.total_variance(k, times[j]) -
                           surface.total_variance(k, times[j - 1]);
            if (d < rep.worst_decrease) {
                rep.worst_decrease = d;
                rep.k_at_worst = k;
                rep.t_at_worst = times[j];
            }
            if (d < 0.0) ++rep.violations;
        }
    }
    rep.free = rep.violations == 0;
    return rep;
}

// ---------------------------------------------------------------------------
// eSSVI
// ---------------------------------------------------------------------------

/// Extended SSVI: rho is allowed to vary with theta.
///
/// SSVI forces one skew across the whole surface, and equity index surfaces do
/// not have one -- short-dated skew is markedly steeper than long-dated. eSSVI
/// (Hendriks & Martini 2019; Corbetta, Cohort, Laachir & Martini 2019) restores
/// the degree of freedom. The price is that the clean Theorem 4.2 conditions no
/// longer apply as stated, so freedom from calendar arbitrage is enforced
/// slice-pair by slice-pair on the quantities that actually control it.
struct ESSVISurface {
    std::vector<Real> expiries;
    std::vector<Real> theta;     ///< ATM total variance
    std::vector<Real> rho;       ///< per-slice skew
    std::vector<Real> psi;       ///< per-slice theta * phi

    std::size_t size() const { return expiries.size(); }

    SSVISlice slice_at_index(std::size_t i) const {
        SSVISlice s;
        s.theta = theta[i];
        s.rho   = rho[i];
        s.phi   = psi[i] / theta[i];
        return s;
    }

    Real total_variance(Real k, Real t) const {
        // Nearest-pillar in expiry; interpolation between eSSVI pillars is a
        // separate problem (the interpolated slice is not itself eSSVI) and is
        // deliberately not pretended away here.
        std::size_t best = 0;
        Real best_d = DBL_HUGE;
        for (std::size_t i = 0; i < expiries.size(); ++i) {
            const Real d = std::fabs(expiries[i] - t);
            if (d < best_d) { best_d = d; best = i; }
        }
        return slice_at_index(best).total_variance(k);
    }

    /// Pairwise calendar condition for consecutive eSSVI slices.
    ///
    /// Corbetta et al. give: theta must be non-decreasing, psi = theta phi must
    /// be non-decreasing, and the skew products must satisfy
    ///     |rho_2 psi_2 - rho_1 psi_1| <= psi_2 - psi_1.
    /// The last is the one that bites: it says the skew may steepen with
    /// maturity only as fast as the curvature allows.
    bool calendar_conditions_hold(Real tol = 1e-12) const {
        for (std::size_t i = 1; i < size(); ++i) {
            if (theta[i] < theta[i - 1] - tol) return false;
            if (psi[i] < psi[i - 1] - tol) return false;
            const Real lhs = std::fabs(rho[i] * psi[i] - rho[i - 1] * psi[i - 1]);
            if (lhs > psi[i] - psi[i - 1] + tol) return false;
        }
        return true;
    }
};

}  // namespace vse
