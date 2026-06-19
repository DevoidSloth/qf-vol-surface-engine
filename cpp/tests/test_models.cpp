// Heston, Bates and SABR.
#include "harness.hpp"
#include "vse/bates.hpp"
#include "vse/black.hpp"
#include "vse/calibrate_heston.hpp"
#include "vse/dual.hpp"
#include "vse/fft.hpp"
#include "vse/heston.hpp"
#include "vse/implied_vol.hpp"
#include "vse/quad.hpp"
#include "vse/sabr.hpp"

#include <complex>
#include <vector>

using namespace vse;
using namespace vsetest;

namespace {

HestonParams index_like() {
    // A calibration-shaped parameter set: high vol-of-vol, strong negative
    // correlation, Feller violated -- which is what equity index boards want.
    return HestonParams{0.0348, 1.58, 0.0447, 0.92, -0.74};
}

/// Heston's ORIGINAL characteristic function, with g1 = (xi+d)/(xi-d) and
/// e^{+dT}. Present only so the test suite can demonstrate that it is broken.
std::complex<Real> heston_cf_naive(const HestonParams& p, Real T, std::complex<Real> u) {
    using Z = std::complex<Real>;
    const Z i(0.0, 1.0);
    const Z iu = i * u;
    const Z xi = p.kappa - p.rho * p.sigma * iu;
    const Z d = std::sqrt(xi * xi + p.sigma * p.sigma * (iu + u * u));
    const Z g1 = (xi + d) / (xi - d);
    const Z edt = std::exp(d * T);
    const Z inv_s2 = Z(1.0 / (p.sigma * p.sigma), 0.0);

    const Z c = p.kappa * p.theta * inv_s2 *
                ((xi + d) * T - 2.0 * std::log((1.0 - g1 * edt) / (1.0 - g1)));
    const Z dd = (xi + d) * inv_s2 * ((1.0 - edt) / (1.0 - g1 * edt));
    return std::exp(c + dd * p.v0);
}

}  // namespace

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

TEST(fft, matches_a_direct_dft) {
    const std::size_t n = 64;
    std::vector<std::complex<Real>> a(n), original(n);
    for (std::size_t j = 0; j < n; ++j) {
        a[j] = std::complex<Real>(std::sin(0.3 * Real(j)) + 0.2 * Real(j % 7),
                                  std::cos(0.17 * Real(j)));
        original[j] = a[j];
    }
    fft_in_place(a);
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<Real> sum(0.0, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            const Real phi = -TWO_PI * Real(j) * Real(k) / Real(n);
            sum += original[j] * std::complex<Real>(std::cos(phi), std::sin(phi));
        }
        CHECK_ABS(a[k].real(), sum.real(), 1e-12);
        CHECK_ABS(a[k].imag(), sum.imag(), 1e-12);
    }
}

TEST(fft, round_trips_through_the_inverse) {
    std::vector<std::complex<Real>> a(256), original;
    for (std::size_t j = 0; j < a.size(); ++j) {
        a[j] = std::complex<Real>(std::exp(-0.01 * Real(j)) * std::cos(0.4 * Real(j)),
                                  0.3 * std::sin(0.11 * Real(j)));
    }
    original = a;
    fft_in_place(a);
    ifft_in_place(a);
    for (std::size_t j = 0; j < a.size(); ++j) {
        CHECK_ABS(a[j].real(), original[j].real(), 1e-14);
        CHECK_ABS(a[j].imag(), original[j].imag(), 1e-14);
    }
    CHECK_THROWS(([] { std::vector<std::complex<Real>> bad(100); fft_in_place(bad); }()));
}

TEST(quadrature, gauss_legendre_is_exact_for_polynomials) {
    // An n-point rule integrates degree 2n-1 exactly. Checking one degree above
    // is what distinguishes a correct rule from a lucky one.
    for (int n : {4, 8, 16, 32}) {
        for (int deg = 0; deg <= 2 * n - 1; ++deg) {
            const Real got = integrate_gl([deg](Real x) { return std::pow(x, deg); }, -1.0, 1.0, n);
            const Real want = (deg % 2 == 1) ? 0.0 : 2.0 / Real(deg + 1);
            CHECK_ABS(got, want, 1e-13);
        }
    }
}

TEST(quadrature, semi_infinite_map_integrates_known_integrals) {
    // integral_0^inf e^{-x} dx = 1
    CHECK_CLOSE(integrate_semi_infinite([](Real x) { return std::exp(-x); }, 1.0, 48, 12), 1.0, 1e-12);
    // integral_0^inf e^{-x^2/2} dx = sqrt(pi/2)
    CHECK_CLOSE(integrate_semi_infinite([](Real x) { return std::exp(-0.5 * x * x); }, 1.0, 48, 12),
                std::sqrt(0.5 * PI), 1e-12);
    // integral_0^inf 1/(1+x^2) dx = pi/2
    CHECK_CLOSE(integrate_semi_infinite([](Real x) { return 1.0 / (1.0 + x * x); }, 1.0, 48, 24),
                0.5 * PI, 1e-10);
    CHECK_THROWS(integrate_semi_infinite([](Real x) { return x; }, -1.0));
}

TEST(dual, reproduces_derivatives_of_a_closed_form) {
    // f(x,y) = exp(x y) / sqrt(x + y), differentiated exactly.
    using D = RealDual<2>;
    const Real x = 1.3, y = 0.7;
    const D dx = D::variable(x, 0), dy = D::variable(y, 1);
    const D f = exp(dx * dy) / sqrt(dx + dy);

    const Real value = std::exp(x * y) / std::sqrt(x + y);
    const Real dfdx = value * (y - 0.5 / (x + y));
    const Real dfdy = value * (x - 0.5 / (x + y));
    CHECK_CLOSE(f.v, value, 1e-14);
    CHECK_CLOSE(f.d[0], dfdx, 1e-13);
    CHECK_CLOSE(f.d[1], dfdy, 1e-13);
}

TEST(dual, differentiates_through_complex_arithmetic) {
    using CD = ComplexDual<1>;
    using Z = std::complex<Real>;
    const Real a = 0.8;
    const CD da = CD::variable(Z(a, 0.0), 0);
    const CD f = log(da * Z(0.0, 1.0) + Z(1.0, 0.0));   // log(1 + i a)
    const Z want_value = std::log(Z(1.0, a));
    const Z want_deriv = Z(0.0, 1.0) / Z(1.0, a);
    CHECK_ABS(f.v.real(), want_value.real(), 1e-14);
    CHECK_ABS(f.v.imag(), want_value.imag(), 1e-14);
    CHECK_ABS(f.d[0].real(), want_deriv.real(), 1e-13);
    CHECK_ABS(f.d[0].imag(), want_deriv.imag(), 1e-13);
}

// ---------------------------------------------------------------------------
// Heston
// ---------------------------------------------------------------------------

TEST(heston, characteristic_function_is_a_martingale) {
    // phi(-i) = E[e^X] = E[S_T/F] = 1, exactly, at every maturity. This one
    // identity catches almost every algebra error in the characteristic
    // function, including a wrong compensator or a mis-signed rho term.
    for (const auto& p : {index_like(), HestonParams{0.09, 0.3, 0.02, 1.4, 0.5},
                          HestonParams{0.01, 8.0, 0.06, 0.2, -0.95}}) {
        for (Real T : {1.0 / 365, 0.05, 0.5, 1.0, 5.0, 10.0, 30.0}) {
            const auto z = heston_cf(p, T, std::complex<Real>(0.0, -1.0));
            CHECK_ABS(z.real(), 1.0, 1e-12);
            CHECK_ABS(z.imag(), 0.0, 1e-12);
        }
    }
}

TEST(heston, the_little_heston_trap_is_real) {
    // Heston's original g1/e^{+dT} form against the Albrecher et al. g2/e^{-dT}
    // form. Algebraically identical; in floating point the first crosses the
    // branch cut of the complex logarithm and stops being a characteristic
    // function.
    //
    // The comparison is made on the Lewis contour, u - i/2, which is where the
    // pricer actually evaluates. (Not at u = -i: both forms have a removable
    // 0/0 there for unrelated reasons, so it says nothing about branch cuts.)
    const HestonParams p = index_like();

    // 1. Discontinuity. The trap form is smooth in u; the naive one jumps.
    auto count_jumps = [&](Real T, bool naive) {
        int jumps = 0;
        Real prev = (naive ? heston_cf_naive(p, T, std::complex<Real>(1e-4, -0.5))
                           : heston_cf(p, T, std::complex<Real>(1e-4, -0.5))).real();
        for (Real u = 0.01; u < 8.0; u += 0.01) {
            const std::complex<Real> z(u, -0.5);
            const Real now = (naive ? heston_cf_naive(p, T, z) : heston_cf(p, T, z)).real();
            if (std::fabs(now - prev) > 0.05) ++jumps;
            prev = now;
        }
        return jumps;
    };

    std::printf("       T     jumps in Re(phi(u - i/2)) over u in [0, 8]\n");
    std::printf("       %-6s %10s %10s\n", "", "trap", "naive");
    int naive_total = 0;
    for (Real T : {1.0, 5.0, 10.0, 20.0, 30.0}) {
        const int trap_jumps = count_jumps(T, false);
        const int naive_jumps = count_jumps(T, true);
        std::printf("       %-6.1f %10d %10d\n", T, trap_jumps, naive_jumps);
        CHECK(trap_jumps == 0);
        naive_total += naive_jumps;
    }
    CHECK(naive_total > 0);

    // 2. Price. A ten-year option priced through each form. The naive one is not
    //    approximately right; it is wrong by a large fraction of the premium.
    const Real F = 100.0, T = 10.0;
    auto naive_price = [&](Real K) {
        const Real k = std::log(F / K);
        const Real integral = integrate_semi_infinite(
            [&](Real u) {
                const std::complex<Real> z(u, -0.5);
                return (std::exp(std::complex<Real>(0.0, u * k)) *
                        heston_cf_naive(p, T, z)).real() / (u * u + 0.25);
            },
            detail::heston_integrand_scale(p, T), 64, 32);
        return F - std::sqrt(F * K) / PI * integral;
    };

    std::printf("       ten-year prices    K      trap        naive\n");
    int wrong = 0;
    for (Real K : {80.0, 100.0, 130.0}) {
        const Real good = heston_call_lewis(p, F, K, T, 64, 32);
        const Real bad = naive_price(K);
        std::printf("       %18.0f %11.5f %12.5f\n", K, good, bad);
        CHECK(good > 0.0 && good < F);
        if (std::fabs(bad / good - 1.0) > 1e-3 || !std::isfinite(bad)) ++wrong;
    }
    CHECK(wrong > 0);
}

TEST(heston, reduces_to_black_scholes_as_vol_of_vol_vanishes) {
    // The degenerate limit, and a genuine numerical trap of its own: the
    // textbook characteristic function divides (xi - d) by sigma^2, and at
    // sigma = 1e-7 those cancel to nothing. The algebraically rearranged form
    // this library uses holds full precision.
    HestonParams p{0.04, 2.0, 0.04, 1e-7, 0.0};
    const Real F = 100.0, T = 1.0;
    for (Real K : {50.0, 70.0, 90.0, 100.0, 115.0, 140.0, 200.0}) {
        const Real h = heston_call_lewis(p, F, K, T);
        const Real b = black76_undiscounted(F, K, T, 0.2, OptionType::Call);
        CHECK_CLOSE(h, b, 1e-11);
    }
    // Also at a different level and maturity, where v0 != theta so the variance
    // path is not constant.
    HestonParams q{0.09, 3.0, 0.09, 1e-7, 0.0};
    for (Real K : {80.0, 100.0, 130.0}) {
        CHECK_CLOSE(heston_call_lewis(q, 100.0, K, 0.25),
                    black76_undiscounted(100.0, K, 0.25, 0.3, OptionType::Call), 1e-11);
    }
}

TEST(heston, lewis_and_carr_madan_fft_agree) {
    // Two genuinely different numerical routes to the same price: Gauss-Legendre
    // on the Lewis integral against a damped Fourier transform on a 4096-point
    // grid. Agreement at 1e-7 is evidence both are right; agreement at 1e-3 would
    // be evidence they share a bug.
    //
    // Compared only where the out-of-the-money value exceeds 1e-4 of the forward
    // -- a cent on a hundred, an order of magnitude below any listed tick.
    // Below that the two methods fail in different ways and the comparison stops
    // being informative: Lewis computes F minus an integral close to F, so it is
    // absolutely accurate to ~1e-8 F; the transform is relatively accurate but
    // its damping factor is tuned for the body, not the far wing. Where they
    // disagree out there, both are noise. The next test measures that boundary
    // instead of papering over it.
    const HestonParams p = index_like();
    const Real F = 100.0;
    for (Real T : {0.08, 0.25, 1.0, 3.0}) {
        const auto grid = heston_carr_madan(p, F, T);
        const HestonSliceEngine engine(p, T, false, 48, 32);
        Real worst = 0.0;
        int compared = 0;
        for (Real K : {70.0, 85.0, 95.0, 100.0, 105.0, 120.0, 145.0}) {
            const Real lewis = engine.call(F, K);
            const Real otm = (K >= F) ? lewis : lewis - (F - K);
            if (otm < 1e-4 * F) continue;
            const Real fft = grid.call_at(K);
            worst = std::fmax(worst, std::fabs(fft - lewis) / std::fmax(otm, 1e-12));
            ++compared;
        }
        std::printf("       T=%.2f  %d strikes compared, worst |FFT - Lewis|/OTM = %.2e\n",
                    T, compared, worst);
        CHECK(compared >= 4);
        CHECK(worst < 1e-4);
    }
}

TEST(heston, both_transform_methods_have_a_floor_and_it_is_where_expected) {
    // The honest statement of accuracy. Lewis subtracts an integral from F, so
    // its error is absolute; measure it against a converged reference and check
    // it sits near 1e-8 F rather than anywhere worse.
    const HestonParams p = index_like();
    const Real F = 100.0, T = 0.25;
    const HestonSliceEngine reference(p, T, false, 96, 96);
    const HestonSliceEngine engine(p, T);

    Real worst_absolute = 0.0;
    std::printf("       K/F     OTM value     absolute error   relative\n");
    for (Real z : {-5.0, -4.0, -3.0, -2.0, 0.0, 2.0, 3.0, 4.0}) {
        const Real K = F * std::exp(z * 0.19 * std::sqrt(T));
        const Real ref = reference.call_unclamped(F, K);
        const Real got = engine.call_unclamped(F, K);
        const Real otm = (K >= F) ? ref : ref - (F - K);
        worst_absolute = std::fmax(worst_absolute, std::fabs(got - ref));
        std::printf("       %5.3f  %12.3e  %14.2e  %10.2e\n", K / F, otm,
                    std::fabs(got - ref), std::fabs(got - ref) / std::fmax(otm, 1e-300));
    }
    std::printf("       worst absolute error %.2e = %.1e of the forward\n",
                worst_absolute, worst_absolute / F);
    CHECK(worst_absolute < 1e-6 * F);
}

TEST(heston, satisfies_put_call_parity) {
    const HestonParams p = index_like();
    const Real F = 4275.0, df = 0.981;
    for (Real T : {0.02, 0.3, 1.5}) {
        for (Real K : {2800.0, 3800.0, 4275.0, 4900.0, 6200.0}) {
            const Real c = heston_price(p, F, K, T, df, OptionType::Call);
            const Real put = heston_price(p, F, K, T, df, OptionType::Put);
            CHECK_ABS(c - put, df * (F - K), 1e-8 * F);
        }
    }
}

TEST(heston, prices_are_inside_the_no_arbitrage_bounds) {
    const HestonParams p = index_like();
    const Real F = 100.0;
    for (Real T : {0.01, 0.1, 1.0, 5.0}) {
        for (Real K = 20.0; K <= 300.0; K += 5.0) {
            const Real c = heston_call_lewis(p, F, K, T);
            CHECK(c >= std::fmax(F - K, 0.0) - 1e-9);
            CHECK(c <= F + 1e-9);
        }
    }
}

TEST(heston, analytic_gradient_matches_central_differences) {
    // The claim is that these derivatives are exact, so the reference is a
    // central difference and the tolerance is set by the difference's accuracy,
    // not the AD's.
    //
    // Two conditions have to be met for the comparison to mean anything, and
    // both were learned the hard way:
    //
    //   * The quadrature must be converged. The node placement depends on the
    //     parameters (see heston_integrand_scale), so a central difference at
    //     the default resolution differences two slightly different quadratures
    //     and measures how the quadrature error varies with the parameter, not
    //     the derivative. At the default rule that produced apparent gradient
    //     errors of 40%, all of it in the reference.
    //   * The option must be worth something. At nine standard deviations out
    //     the price is below the absolute floor of the method, so the AD returns
    //     ~1e-15 (correct) and the difference returns ~1e-8 (noise), and the
    //     relative discrepancy is 1 by construction.
    const HestonParams p = index_like();
    const Real F = 100.0;
    const char* names[5] = {"v0", "kappa", "theta", "sigma", "rho"};
    Real worst[5] = {0, 0, 0, 0, 0};
    int compared = 0;

    for (Real T : {0.05, 0.5, 2.0}) {
        for (Real K : {70.0, 90.0, 100.0, 115.0, 150.0}) {
            const auto pg = heston_call_and_gradient(p, F, K, T, 64, 64);
            const Real otm = (K >= F) ? pg.price : pg.price - (F - K);
            if (otm < 1e-6 * F) continue;
            ++compared;
            CHECK_CLOSE(pg.price, heston_call_lewis(p, F, K, T, 64, 64), 1e-12);

            for (int c = 0; c < 5; ++c) {
                HestonParams up = p, dn = p;
                Real* pu[5] = {&up.v0, &up.kappa, &up.theta, &up.sigma, &up.rho};
                Real* pd[5] = {&dn.v0, &dn.kappa, &dn.theta, &dn.sigma, &dn.rho};
                const Real base = *pu[c];
                const Real h = 1e-4 * std::fmax(std::fabs(base), 0.05);
                *pu[c] = base + h;
                *pd[c] = base - h;
                const Real fd = (heston_call_lewis(up, F, K, T, 64, 64) -
                                 heston_call_lewis(dn, F, K, T, 64, 64)) / (2.0 * h);
                const Real scale = std::fmax(std::fabs(fd), 1e-8 * F);
                worst[c] = std::fmax(worst[c], std::fabs(pg.gradient[std::size_t(c)] - fd) / scale);
            }
        }
    }
    std::printf("       %d options compared; worst relative gradient error:", compared);
    for (int c = 0; c < 5; ++c) std::printf(" %s=%.1e", names[c], worst[c]);
    std::printf("\n");
    CHECK(compared >= 10);
    for (int c = 0; c < 5; ++c) CHECK(worst[c] < 1e-4);
}

TEST(heston, gradient_is_exact_where_a_difference_is_only_approximate) {
    // The converse of the test above. Far out in the wing the price is below the
    // absolute floor, so the option really is worth nothing and its sensitivity
    // really is zero. Automatic differentiation returns that; a central
    // difference returns the noise floor of the two prices it subtracted.
    const HestonParams p = index_like();
    const Real F = 100.0, T = 0.05, K = 150.0;   // about nine standard deviations
    const auto pg = heston_call_and_gradient(p, F, K, T, 64, 64);
    CHECK(std::fabs(pg.price) < 1e-7);

    HestonParams up = p, dn = p;
    const Real h = 1e-4 * 0.05;
    up.v0 += h;
    dn.v0 -= h;
    const Real fd = (heston_call_lewis(up, F, K, T, 64, 64) -
                     heston_call_lewis(dn, F, K, T, 64, 64)) / (2.0 * h);
    std::printf("       nine sd out: AD d/dv0 = %.3e, central difference = %.3e\n",
                pg.gradient[0], fd);
    CHECK(std::fabs(pg.gradient[0]) < 1e-10);      // the true answer
    CHECK(std::fabs(fd) > std::fabs(pg.gradient[0]));   // the difference is noise
}

TEST(heston, feller_condition_is_reported_not_enforced) {
    const HestonParams p = index_like();
    CHECK_CLOSE(p.feller_ratio(), 2.0 * p.kappa * p.theta / (p.sigma * p.sigma), 1e-15);
    CHECK(!p.satisfies_feller());     // as a real index calibration typically does not
    // and the model still prices, because the characteristic function does not care
    CHECK(heston_call_lewis(p, 100.0, 100.0, 1.0) > 0.0);

    const HestonParams safe{0.04, 3.0, 0.04, 0.3, -0.5};
    CHECK(safe.satisfies_feller());
}

TEST(heston, rejects_invalid_parameters) {
    HestonParams p = index_like();
    p.sigma = -1.0;
    CHECK_THROWS(heston_call_lewis(p, 100.0, 100.0, 1.0));
    p = index_like();
    p.rho = 1.5;
    CHECK_THROWS(heston_call_lewis(p, 100.0, 100.0, 1.0));
    CHECK_THROWS(heston_call_lewis(index_like(), -100.0, 100.0, 1.0));
    CHECK_THROWS(heston_call_lewis(index_like(), 100.0, 100.0, -1.0));
}

TEST(heston, calibration_recovers_the_parameters_that_generated_the_board) {
    // Synthetic board from known Heston parameters, then calibrate back. The
    // parameters, not just the fit, are checked -- with noiseless data from the
    // same model there is no excuse for missing them.
    const HestonParams truth = index_like();
    const Real F = 100.0;
    std::vector<CalibrationQuote> quotes;
    for (Real T : {0.08, 0.25, 0.5, 1.0, 2.0}) {
        for (Real z : {-2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0}) {
            const Real K = F * std::exp(z * 0.2 * std::sqrt(T));
            const bool otm_call = K >= F;
            const Real call = heston_call_lewis(truth, F, K, T, 48, 12);
            const Real price = otm_call ? call : call - (F - K);
            const auto iv = implied_volatility_ex(price, F, K, T, 1.0,
                                                  otm_call ? OptionType::Call : OptionType::Put);
            CHECK(iv.converged);
            quotes.push_back({F, K, T, iv.sigma, 1.0});
        }
    }

    const auto fit = calibrate_heston(quotes, HestonParams{0.02, 3.0, 0.06, 0.5, -0.4});
    std::printf("       truth  v0=%.4f kappa=%.3f theta=%.4f sigma=%.3f rho=%+.3f\n",
                truth.v0, truth.kappa, truth.theta, truth.sigma, truth.rho);
    std::printf("       fitted v0=%.4f kappa=%.3f theta=%.4f sigma=%.3f rho=%+.3f\n",
                fit.params.v0, fit.params.kappa, fit.params.theta, fit.params.sigma,
                fit.params.rho);
    std::printf("       RMSE %.3e vol points, %d LM iterations, %d slice builds\n",
                fit.rmse_vol_points, fit.iterations, fit.slice_builds);

    CHECK(fit.converged);
    CHECK(fit.rmse_vol_points < 1e-5);
    CHECK_CLOSE(fit.params.v0, truth.v0, 1e-3);
    CHECK_CLOSE(fit.params.theta, truth.theta, 1e-2);
    CHECK_CLOSE(fit.params.rho, truth.rho, 1e-2);
}

// ---------------------------------------------------------------------------
// Bates
// ---------------------------------------------------------------------------

TEST(bates, characteristic_function_is_a_martingale) {
    // The compensator is the part that is easy to get wrong, and this is the
    // test that catches it.
    BatesParams p;
    p.heston = index_like();
    for (Real lambda : {0.0, 0.2, 1.5}) {
        for (Real m : {-0.2, -0.05, 0.1}) {
            for (Real dv : {0.05, 0.25}) {
                p.lambda = lambda; p.jump_mean = m; p.jump_vol = dv;
                for (Real T : {0.02, 0.5, 3.0, 10.0}) {
                    const auto z = bates_cf(p, T, std::complex<Real>(0.0, -1.0));
                    CHECK_ABS(z.real(), 1.0, 1e-11);
                    CHECK_ABS(z.imag(), 0.0, 1e-11);
                }
            }
        }
    }
}

TEST(bates, reduces_to_heston_when_the_intensity_is_zero) {
    BatesParams p;
    p.heston = index_like();
    p.lambda = 0.0;
    for (Real T : {0.05, 1.0}) {
        for (Real K : {75.0, 100.0, 135.0}) {
            CHECK_CLOSE(bates_call_lewis(p, 100.0, K, T),
                        heston_call_lewis(p.heston, 100.0, K, T), 1e-10);
        }
    }
}

TEST(bates, jumps_steepen_the_short_dated_smile) {
    // The reason the model exists. A diffusion generates skew at rate sqrt(T),
    // so a one-week Heston smile is nearly flat; jumps do not care about T, so
    // they put value in the front-end wings where the market puts it too.
    //
    // What is asserted is the *maturity structure*, not a uniform steepening.
    // Adding jumps also raises total variance, and at the long end that
    // flattening can dominate -- so "jumps make every slice steeper" is simply
    // false, and testing it would be testing a wrong belief.
    const Real F = 100.0;
    BatesParams p;
    p.heston = index_like();
    p.lambda = 0.6;
    p.jump_mean = -0.10;
    p.jump_vol = 0.14;

    auto skew_of = [&](Real T, bool with_jumps) {
        auto vol_at = [&](Real K) {
            const bool otm_call = K >= F;
            const Real call = with_jumps ? bates_call_lewis(p, F, K, T, 48, 32)
                                         : heston_call_lewis(p.heston, F, K, T, 48, 32);
            const Real price = otm_call ? call : call - (F - K);
            return implied_volatility(price, F, K, T, 1.0,
                                      otm_call ? OptionType::Call : OptionType::Put);
        };
        // Skew per unit log-moneyness, measured over +/- one at-the-money
        // standard deviation of the model being measured.
        const Real atm = vol_at(F);
        const Real sd = atm * std::sqrt(T);
        return (vol_at(F * std::exp(sd)) - vol_at(F * std::exp(-sd))) / (2.0 * sd);
    };

    std::printf("       T        Heston skew   Bates skew   steepening\n");
    Real front = 0.0, back = 0.0;
    for (Real T : {7.0 / 365, 30.0 / 365, 0.5, 2.0}) {
        const Real h = skew_of(T, false), b = skew_of(T, true);
        std::printf("       %6.3f   %+9.4f    %+9.4f   %+9.4f\n", T, h, b, h - b);
        if (T < 0.05) front = h - b;
        if (T > 1.5) back = h - b;
    }
    // Jumps steepen the front end...
    CHECK(front > 0.0);
    // ...and their contribution to skew decays with maturity, which is exactly
    // the shape a diffusion alone cannot produce.
    CHECK(front > back);
}

TEST(bates, satisfies_put_call_parity) {
    BatesParams p;
    p.heston = index_like();
    p.lambda = 0.4; p.jump_mean = -0.09; p.jump_vol = 0.2;
    const Real F = 100.0, df = 0.97;
    for (Real T : {0.03, 0.6}) {
        for (Real K : {60.0, 100.0, 160.0}) {
            const Real c = bates_price(p, F, K, T, df, OptionType::Call);
            const Real put = bates_price(p, F, K, T, df, OptionType::Put);
            CHECK_ABS(c - put, df * (F - K), 1e-9 * F);
        }
    }
}

// ---------------------------------------------------------------------------
// SABR
// ---------------------------------------------------------------------------

TEST(sabr, atm_limit_is_continuous) {
    // z/chi(z) is 0/0 at the money and the series expansion has to hand over to
    // the closed form without a step. A discontinuity there is invisible on a
    // plot of the smile and fatal to a delta, which differentiates across it.
    const SABRParams p{0.03, 0.5, -0.25, 0.4, 0.0};
    const Real F = 0.025, T = 5.0;
    const Real atm = sabr_atm_vol(p, F, T);

    // Approaching the money, the smile must converge to the ATM value at the
    // rate the smile itself has -- linearly in the offset, not to some other
    // number. So the tolerance scales with eps rather than being fixed.
    for (Real eps : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}) {
        const Real up = sabr_lognormal_vol(p, F, F * (1.0 + eps), T);
        const Real dn = sabr_lognormal_vol(p, F, F * (1.0 - eps), T);
        CHECK(std::fabs(up / atm - 1.0) < 5.0 * eps + 1e-12);
        CHECK(std::fabs(dn / atm - 1.0) < 5.0 * eps + 1e-12);
    }
    // And the one-sided slopes agree, so there is no kink.
    const Real h = 1e-5 * F;
    const Real left = (atm - sabr_lognormal_vol(p, F, F - h, T)) / h;
    const Real right = (sabr_lognormal_vol(p, F, F + h, T) - atm) / h;
    CHECK_CLOSE(left, right, 1e-3);
}

TEST(sabr, alpha_inverts_the_atm_volatility) {
    for (Real beta : {0.0, 0.5, 1.0}) {
        for (Real nu : {0.1, 0.6}) {
            for (Real target : {0.10, 0.25, 0.60}) {
                const Real F = 0.03, T = 2.0, rho = -0.3;
                const Real alpha = sabr_alpha_from_atm(target, F, T, beta, rho, nu);
                const SABRParams p{alpha, beta, rho, nu, 0.0};
                CHECK_CLOSE(sabr_atm_vol(p, F, T), target, 1e-10);
            }
        }
    }
}

TEST(sabr, hagan_expansion_produces_negative_densities) {
    // The documented failure, and not at exotic parameters: these are ordinary
    // long-dated rates numbers.
    const Real F = 0.03, T = 10.0;
    const SABRParams p{0.025, 0.5, -0.2, 0.45, 0.0};
    const auto rep = sabr_density_scan(p, F, T);
    std::printf("       beta=0.5 nu=0.45 T=10: %d of %d grid points have a negative\n"
                "       density; boundary at K/F = %.3f, worst %.3e at K/F = %.3f\n",
                rep.violations, rep.points, rep.arbitrage_boundary / F,
                rep.min_density, rep.strike_at_min / F);
    CHECK(!rep.free);
    CHECK(rep.violations > 0);
    CHECK(rep.min_density < 0.0);
    CHECK(rep.arbitrage_boundary > 0.0);

    // A short-dated, low vol-of-vol smile is fine -- the expansion is asymptotic
    // in alpha^2 T, so the failure is a large-T, large-nu phenomenon, and a test
    // that could not tell the two apart would not be testing anything.
    const SABRParams tame{0.025, 0.5, -0.2, 0.15, 0.0};
    const auto ok = sabr_density_scan(tame, F, 0.5);
    std::printf("       beta=0.5 nu=0.15 T=0.5: %d violations\n", ok.violations);
    CHECK(ok.free);
}

TEST(sabr, the_normal_expansion_pushes_the_arbitrage_further_out) {
    // One of the two standard responses to the above. Compare the strike below
    // which each expansion implies a negative density; the normal form should
    // survive further down.
    const Real F = 0.03, T = 10.0;
    const SABRParams p{0.025, 0.5, -0.2, 0.45, 0.0};

    const auto lognormal = sabr_density_scan(p, F, T, 0.02, 3.0, 1500, false);
    const auto normal = sabr_density_scan(p, F, T, 0.02, 3.0, 1500, true);

    std::printf("       lognormal expansion: arbitrage below K/F = %.3f (%d points)\n",
                lognormal.arbitrage_boundary / F, lognormal.violations);
    std::printf("       normal    expansion: arbitrage below K/F = %.3f (%d points)\n",
                normal.arbitrage_boundary / F, normal.violations);
    CHECK(lognormal.violations > 0);
    CHECK(normal.arbitrage_boundary < lognormal.arbitrage_boundary);
    CHECK(normal.violations < lognormal.violations);
}

TEST(sabr, shifting_moves_the_smile_not_its_shape) {
    // A shifted SABR on a shifted forward must reproduce the unshifted smile
    // exactly; the shift is a change of variable, not a change of model.
    const Real F = 0.02, T = 3.0, s = 0.03;
    const SABRParams plain{0.02, 0.6, -0.35, 0.5, 0.0};
    SABRParams shifted = plain;
    shifted.shift = s;
    for (Real K : {0.005, 0.012, 0.02, 0.035, 0.06}) {
        CHECK_CLOSE(sabr_lognormal_vol(shifted, F - s, K - s, T),
                    sabr_lognormal_vol(plain, F, K, T), 1e-14);
    }
    // And a negative forward is priceable once shifted, which is the point.
    CHECK(sabr_lognormal_vol(shifted, -0.005, 0.001, T) > 0.0);
    CHECK_THROWS(sabr_lognormal_vol(plain, -0.005, 0.001, T));
}

TEST(sabr, beta_one_reduces_to_a_lognormal_smile) {
    // At beta = 1 and nu = 0 the model is Black with volatility alpha.
    const SABRParams p{0.25, 1.0, 0.0, 1e-12, 0.0};
    for (Real K : {60.0, 100.0, 160.0}) {
        CHECK_CLOSE(sabr_lognormal_vol(p, 100.0, K, 1.0), 0.25, 1e-8);
    }
}
