// vse/normal.hpp — error functions, normal density, CDF and inverse CDF.
//
// Why this file is not three lines wrapping std::erfc:
//
// The headline accuracy claim of this library is a 1e-12 relative implied-vol
// round trip *including the deep wings*. Out there an option is worth 1e-20 of
// the forward, so every routine on the path has to be accurate in the relative
// sense; an absolute 1e-15 is worthless. Two consequences:
//
//   * Hart/West's rational CDF, which is what most option code uses, is built
//     for ~1e-15 absolute. Measured against a 60-digit mpmath evaluation its
//     *relative* error degrades steadily past |x| = 3 and reaches 2.6e-9 at
//     |x| = 7. That silently caps wing accuracy at eight digits.
//   * The quantity the pricer actually wants is not erfc but erfcx(z) =
//     e^{z^2} erfc(z), the scaled complementary error function. Written in terms
//     of erfcx, both the Mills ratio and the normalised Black function lose the
//     cancellation entirely -- and erfcx needs no exponential at all for
//     z > 0.46875, which is also why it is the fast path.
//
// So the primitive here is Cody's rational Chebyshev approximation (W. J. Cody,
// "Rational Chebyshev approximation for the error function", Math. Comp. 23
// (1969); the CALERF driver of ACM Algorithm 715), which delivers full double
// relative precision in all three modes. scripts/audit_normal.py checks this
// implementation against mpmath and prints the measured worst case.
#pragma once

#include "vse/common.hpp"

namespace vse {

namespace detail {

// |x| <= 0.46875, approximating erf.
inline constexpr Real kA[5] = {3.16112374387056560e+00, 1.13864154151050156e+02,
                               3.77485237685302021e+02, 3.20937758913846947e+03,
                               1.85777706184603153e-01};
inline constexpr Real kB[4] = {2.36012909523441209e+01, 2.44024637934444173e+02,
                               1.28261652607737228e+03, 2.84423683343917062e+03};

// 0.46875 <= |x| <= 4, approximating erfc.
inline constexpr Real kC[9] = {5.64188496988670089e-01, 8.88314979438837594e+00,
                               6.61191906371416295e+01, 2.98635138197400131e+02,
                               8.81952221241769090e+02, 1.71204761263407058e+03,
                               2.05107837782607147e+03, 1.23033935479799725e+03,
                               2.15311535474403846e-08};
inline constexpr Real kD[8] = {1.57449261107098347e+01, 1.17693950891312499e+02,
                               5.37181101862009858e+02, 1.62138957456669019e+03,
                               3.29079923573345963e+03, 4.36261909014324716e+03,
                               3.43936767414372164e+03, 1.23033935480374942e+03};

// |x| > 4, approximating erfc.
inline constexpr Real kP[6] = {3.05326634961232344e-01, 3.60344899949804439e-01,
                               1.25781726111229246e-01, 1.60837851487422766e-02,
                               6.58749161529837803e-04, 1.63153871373020978e-02};
inline constexpr Real kQ[5] = {2.56852019228982242e+00, 1.87295284992346047e+00,
                               5.27905102951428412e-01, 6.05183413124413191e-02,
                               2.33520497626869185e-03};

inline constexpr Real kInvSqrtPi = 5.6418958354775628695e-01;  // 1/sqrt(pi)
inline constexpr Real kThresh    = 0.46875;
inline constexpr Real kXSmall    = 1.11e-16;
inline constexpr Real kXBig      = 26.543;
inline constexpr Real kXHuge     = 6.71e+07;
inline constexpr Real kXNeg      = -26.628;

enum class ErfMode { Erf, Erfc, Erfcx };

/// Cody's CALERF. Splitting exp(-y^2) as exp(-t^2) * exp(-(y-t)(y+t)) with
/// t = trunc(16y)/16 keeps the argument reduction exact, which is what buys the
/// last two digits in the tail.
inline Real calerf(Real x, ErfMode mode) {
    const Real y = std::fabs(x);
    Real result;

    if (y <= kThresh) {
        Real ysq = (y > kXSmall) ? y * y : 0.0;
        Real xnum = kA[4] * ysq, xden = ysq;
        for (int i = 0; i < 3; ++i) {
            xnum = (xnum + kA[i]) * ysq;
            xden = (xden + kB[i]) * ysq;
        }
        result = x * (xnum + kA[3]) / (xden + kB[3]);  // erf(x)
        if (mode != ErfMode::Erf) result = 1.0 - result;
        if (mode == ErfMode::Erfcx) result *= std::exp(ysq);
        return result;
    }

    if (y <= 4.0) {
        Real xnum = kC[8] * y, xden = y;
        for (int i = 0; i < 7; ++i) {
            xnum = (xnum + kC[i]) * y;
            xden = (xden + kD[i]) * y;
        }
        result = (xnum + kC[7]) / (xden + kD[7]);
        if (mode != ErfMode::Erfcx) {
            const Real t = std::trunc(y * 16.0) / 16.0;
            const Real del = (y - t) * (y + t);
            result *= std::exp(-t * t) * std::exp(-del);
        }
    } else {
        result = 0.0;
        if (y >= kXBig && (mode != ErfMode::Erfcx || y >= DBL_HUGE)) {
            // erfc has underflowed; erfcx has not.
        } else if (mode == ErfMode::Erfcx && y >= kXHuge) {
            result = kInvSqrtPi / y;
        } else {
            const Real ysq = 1.0 / (y * y);
            Real xnum = kP[5] * ysq, xden = ysq;
            for (int i = 0; i < 4; ++i) {
                xnum = (xnum + kP[i]) * ysq;
                xden = (xden + kQ[i]) * ysq;
            }
            result = ysq * (xnum + kP[4]) / (xden + kQ[4]);
            result = (kInvSqrtPi - result) / y;
            if (mode != ErfMode::Erfcx) {
                const Real t = std::trunc(y * 16.0) / 16.0;
                const Real del = (y - t) * (y + t);
                result *= std::exp(-t * t) * std::exp(-del);
            }
        }
    }

    switch (mode) {
        case ErfMode::Erf:
            result = (0.5 - result) + 0.5;
            if (x < 0.0) result = -result;
            break;
        case ErfMode::Erfc:
            if (x < 0.0) result = 2.0 - result;
            break;
        case ErfMode::Erfcx:
            if (x < 0.0) {
                if (x < kXNeg) {
                    result = DBL_HUGE;
                } else {
                    const Real t = std::trunc(x * 16.0) / 16.0;
                    const Real del = (x - t) * (x + t);
                    const Real e = std::exp(t * t) * std::exp(del);
                    result = (e + e) - result;
                }
            }
            break;
    }
    return result;
}

}  // namespace detail

inline Real erf_(Real x)   { return detail::calerf(x, detail::ErfMode::Erf); }
inline Real erfc_(Real x)  { return detail::calerf(x, detail::ErfMode::Erfc); }

/// Scaled complementary error function, erfcx(x) = e^{x^2} erfc(x).
///
/// This is the workhorse of the pricing layer. For x > 0 it decays like
/// 1/(x sqrt(pi)) instead of underflowing, so every wing quantity built from it
/// keeps full relative precision, and for x > 0.46875 it costs one rational
/// evaluation with no call to exp.
inline Real erfcx(Real x) { return detail::calerf(x, detail::ErfMode::Erfcx); }

inline Real norm_pdf(Real x) noexcept {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

/// Standard normal CDF, relative-accurate in the lower tail down to underflow.
inline Real norm_cdf(Real x) {
    return 0.5 * erfc_(-x / SQRT_2);
}

/// Inverse standard normal CDF.
///
/// Acklam's rational approximation (~1.15e-9 relative) followed by one Halley
/// step against this file's own norm_cdf. Refining against the same CDF the
/// library uses elsewhere is deliberate: it makes the round trip exact to
/// rounding rather than exact-to-whatever-libm-thinks.
inline Real norm_inv_cdf(Real p) {
    require(p > 0.0 && p < 1.0, "norm_inv_cdf: p must lie strictly in (0,1)");

    static constexpr Real a[6] = {-3.969683028665376e+01,  2.209460984245205e+02,
                                  -2.759285104469687e+02,  1.383577518672690e+02,
                                  -3.066479806614716e+01,  2.506628277459239e+00};
    static constexpr Real b[5] = {-5.447609879822406e+01,  1.615858368580409e+02,
                                  -1.556989798598866e+02,  6.680131188771972e+01,
                                  -1.328068155288572e+01};
    static constexpr Real c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                  -2.400758277161838e+00, -2.549732539343734e+00,
                                   4.374664141464968e+00,  2.938163982698783e+00};
    static constexpr Real d[4] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                   2.445134137142996e+00,  3.754408661907416e+00};

    constexpr Real p_low = 0.02425, p_high = 1.0 - p_low;
    Real x;
    if (p < p_low) {
        const Real q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else if (p <= p_high) {
        const Real q = p - 0.5, r = q * q;
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    } else {
        const Real q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }

    // Halley. Formulated as e/pdf rather than e*sqrt(2pi)*exp(x^2/2) so that
    // p near the underflow boundary does not overflow the correction.
    const Real pdf = norm_pdf(x);
    if (pdf > 0.0) {
        const Real e = norm_cdf(x) - p;
        const Real u = e / pdf;
        x -= u / (1.0 + 0.5 * x * u);
    }
    return x;
}

}  // namespace vse
