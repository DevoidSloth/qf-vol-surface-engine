// Properties of the surface layer: SVI, SSVI, eSSVI, the arbitrage conditions,
// and the chain processing that feeds them.
#include "harness.hpp"
#include "vse/calibrate_svi.hpp"
#include "vse/chain.hpp"
#include "vse/interp.hpp"
#include "vse/linalg.hpp"
#include "vse/lm.hpp"
#include "vse/svi.hpp"
#include "vse/synthetic.hpp"

#include <numeric>

using namespace vse;
using namespace vsetest;

namespace {

SVIRaw reference_slice() {
    // A well-behaved equity-index slice: level 4%, moderate wings, negative skew.
    return SVIRaw{0.012, 0.10, -0.65, -0.02, 0.15};
}

}  // namespace

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

TEST(linalg, cholesky_solves_and_reports_indefiniteness) {
    Matrix a(3, 3);
    a(0, 0) = 4; a(0, 1) = 1; a(0, 2) = 2;
    a(1, 0) = 1; a(1, 1) = 3; a(1, 2) = 0;
    a(2, 0) = 2; a(2, 1) = 0; a(2, 2) = 5;
    std::vector<Real> b = {1.0, 2.0, 3.0}, x;
    CHECK(spd_solve(a, b, x));
    // Residual of the original system.
    for (std::size_t i = 0; i < 3; ++i) {
        Real s = 0.0;
        for (std::size_t j = 0; j < 3; ++j) s += a(i, j) * x[j];
        CHECK_ABS(s, b[i], 1e-13);
    }

    Matrix bad(2, 2);
    bad(0, 0) = 1; bad(0, 1) = 2;
    bad(1, 0) = 2; bad(1, 1) = 1;   // eigenvalues 3 and -1
    std::vector<Real> rhs = {1.0, 1.0}, out;
    CHECK(!spd_solve(bad, rhs, out));
}

TEST(linalg, thomas_solves_a_tridiagonal_system) {
    const std::size_t n = 64;
    std::vector<Real> sub(n, -1.0), diag(n, 4.0), sup(n, -1.0), rhs(n), x, work;
    for (std::size_t i = 0; i < n; ++i) rhs[i] = std::sin(0.1 * Real(i)) + 1.0;
    thomas_solve(sub, diag, sup, rhs, x, work);
    for (std::size_t i = 0; i < n; ++i) {
        Real s = diag[i] * x[i];
        if (i > 0) s += sub[i] * x[i - 1];
        if (i + 1 < n) s += sup[i] * x[i + 1];
        CHECK_ABS(s, rhs[i], 1e-12);
    }
}

TEST(lm, recovers_the_parameters_of_an_exactly_solvable_problem) {
    // y = A exp(-B t), fitted from noiseless data. Both the analytic and the
    // numerical Jacobian must find it.
    const std::vector<Real> t = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const Real A = 2.7, B = 0.83;
    std::vector<Real> y;
    for (Real ti : t) y.push_back(A * std::exp(-B * ti));

    ResidualFn res = [&](const std::vector<Real>& p, std::vector<Real>& r) {
        r.resize(t.size());
        for (std::size_t i = 0; i < t.size(); ++i) r[i] = p[0] * std::exp(-p[1] * t[i]) - y[i];
    };
    JacobianFn jac = [&](const std::vector<Real>& p, Matrix& j) {
        for (std::size_t i = 0; i < t.size(); ++i) {
            j(i, 0) = std::exp(-p[1] * t[i]);
            j(i, 1) = -p[0] * t[i] * std::exp(-p[1] * t[i]);
        }
    };

    const LMResult analytic = levenberg_marquardt(res, jac, {1.0, 0.1});
    CHECK(analytic.converged());
    CHECK_CLOSE(analytic.x[0], A, 1e-9);
    CHECK_CLOSE(analytic.x[1], B, 1e-9);

    const LMResult numeric = levenberg_marquardt(res, numerical_jacobian(res), {1.0, 0.1});
    CHECK(numeric.converged());
    CHECK_CLOSE(numeric.x[0], A, 1e-6);
    CHECK_CLOSE(numeric.x[1], B, 1e-6);
}

TEST(lm, respects_box_constraints) {
    // The unconstrained optimum is at 5; the box forbids anything above 2.
    ResidualFn res = [](const std::vector<Real>& p, std::vector<Real>& r) {
        r = {p[0] - 5.0, 0.0};
    };
    Box box;
    box.lower = {-10.0};
    box.upper = {2.0};
    const LMResult lm = levenberg_marquardt(res, numerical_jacobian(res), {0.0}, box);
    CHECK_ABS(lm.x[0], 2.0, 1e-9);
    CHECK(lm.at_bound[0]);
}

// ---------------------------------------------------------------------------
// SVI and the density
// ---------------------------------------------------------------------------

TEST(svi, derivatives_match_finite_differences) {
    const SVIRaw s = reference_slice();
    for (Real k = -1.2; k <= 1.2; k += 0.05) {
        const Real h = 1e-6;
        const Real d1 = (s.total_variance(k + h) - s.total_variance(k - h)) / (2 * h);
        CHECK_CLOSE(d1, s.dw(k), 1e-7);
        const Real d2 = (s.dw(k + h) - s.dw(k - h)) / (2 * h);
        CHECK_CLOSE(d2, s.d2w(k), 1e-6);
    }
}

TEST(svi, density_matches_breeden_litzenberger) {
    // The whole point of g(k): it is proportional to the second derivative of
    // the call price in strike. Check it against that second derivative computed
    // directly from Black prices off the same slice, which is an independent
    // route to the same number.
    const SVIRaw s = reference_slice();
    const Real T = 0.75, F = 1.0;

    auto call = [&](Real K) {
        const Real k = std::log(K / F);
        const Real vol = s.implied_vol(k, T);
        return black76_undiscounted(F, K, T, vol, OptionType::Call);
    };

    for (Real k = -0.8; k <= 0.8; k += 0.05) {
        const Real K = F * std::exp(k);
        const Real h = 1e-4 * K;
        // d2C/dK2 is the density in strike; multiply by K to get the density in
        // log-strike, since p(k) dk = p(K) dK and dK = K dk.
        const Real d2C = (call(K + h) - 2.0 * call(K) + call(K - h)) / (h * h);
        CHECK_CLOSE(d2C * K, risk_neutral_density(s, k), 1e-4);
    }
}

TEST(svi, density_integrates_to_one) {
    // A density that is non-negative but does not integrate to one means the
    // formula is wrong, and the two failures look identical on a plot.
    const SVIRaw s = reference_slice();
    const auto rep = check_butterfly(s, 0.75, 8.0, 40001);
    CHECK_CLOSE(rep.density_integral, 1.0, 1e-10);
}

TEST(svi, butterfly_check_finds_a_slice_that_violates_it) {
    // A test that only ever sees arbitrage-free slices proves nothing about the
    // detector. This one has too much curvature for its level: g goes negative.
    const SVIRaw bad{0.001, 0.55, -0.95, 0.05, 0.02};
    const auto rep = check_butterfly(bad, 1.0, 1.0, 4001);
    CHECK(!rep.free);
    CHECK(rep.min_g < 0.0);
    CHECK(rep.violations > 0);

    const auto good = check_butterfly(reference_slice(), 0.75);
    CHECK(good.free);
    CHECK(good.min_g > 0.0);
    CHECK(good.min_density >= 0.0);
}

TEST(svi, lee_wing_slopes_are_reported) {
    const SVIRaw s = reference_slice();
    CHECK_CLOSE(s.left_slope(), s.b * (1.0 - s.rho), 1e-15);
    CHECK_CLOSE(s.right_slope(), s.b * (1.0 + s.rho), 1e-15);
    CHECK(s.is_well_formed());

    // Lee's moment formula caps the wing slope of total variance at 2.
    SVIRaw steep = s;
    steep.b = 1.5;                       // right slope 1.5 * 0.35 fine, left 1.5*1.65 = 2.48
    CHECK(!steep.is_well_formed());
}

TEST(svi, minimum_variance_is_where_the_derivative_vanishes) {
    const SVIRaw s = reference_slice();
    const Real k_min = s.m - s.rho * s.sigma / std::sqrt(1.0 - s.rho * s.rho);
    CHECK_ABS(s.dw(k_min), 0.0, 1e-14);
    CHECK_CLOSE(s.total_variance(k_min), s.min_variance(), 1e-14);
}

// ---------------------------------------------------------------------------
// SSVI
// ---------------------------------------------------------------------------

TEST(ssvi, slice_converts_to_equivalent_raw_svi) {
    // An SSVI slice is a raw SVI slice with constrained parameters. If the
    // conversion is wrong, every downstream consumer that speaks raw SVI gets a
    // different surface from the one that was fitted.
    for (Real theta : {0.005, 0.04, 0.25}) {
        for (Real rho : {-0.9, -0.4, 0.0, 0.5}) {
            for (Real phi : {0.4, 1.5, 6.0}) {
                const SSVISlice s{theta, rho, phi};
                const SVIRaw r = s.to_raw();
                for (Real k = -1.0; k <= 1.0; k += 0.1) {
                    CHECK_CLOSE(r.total_variance(k), s.total_variance(k), 1e-12);
                    CHECK_CLOSE(r.dw(k), s.dw(k), 1e-10);
                    CHECK_CLOSE(r.d2w(k), s.d2w(k), 1e-9);
                }
            }
        }
    }
}

TEST(ssvi, atm_total_variance_is_theta) {
    // The defining property: w(0) = theta, for any rho and phi.
    for (Real theta : {0.004, 0.05, 0.3}) {
        for (Real rho : {-0.85, -0.2, 0.6}) {
            for (Real phi : {0.3, 2.0, 8.0}) {
                const SSVISlice s{theta, rho, phi};
                CHECK_CLOSE(s.total_variance(0.0), theta, 1e-14);
            }
        }
    }
}

TEST(ssvi, conditions_agree_with_the_direct_density_scan) {
    // Theorem 4.2's butterfly conditions are sufficient, not necessary. So the
    // property is one-directional: whenever they hold, the density scan must
    // find no violation. The converse is allowed to fail and is not asserted.
    PowerLawPhi phi{1.2, 0.45};
    int satisfied = 0, scanned_clean = 0;
    for (Real theta : {0.002, 0.01, 0.04, 0.12, 0.4}) {
        for (Real rho : {-0.9, -0.6, -0.3, 0.0, 0.4}) {
            const Real ph = phi(theta);
            const auto cond = check_ssvi_conditions(theta, rho, ph, phi.d_theta_phi(theta));
            const auto scan = check_butterfly(SSVISlice{theta, rho, ph}, 1.0, 6.0, 8001);
            if (cond.butterfly_free) {
                ++satisfied;
                CHECK(scan.free);
                if (scan.free) ++scanned_clean;
            }
        }
    }
    CHECK(satisfied > 0);            // the test would be vacuous otherwise
    CHECK(satisfied == scanned_clean);
}

TEST(ssvi, power_law_phi_derivative_is_correct) {
    PowerLawPhi phi{1.7, 0.35};
    for (Real theta = 0.002; theta < 1.0; theta *= 1.5) {
        const Real h = 1e-7 * theta;
        const Real fd = ((theta + h) * phi(theta + h) - (theta - h) * phi(theta - h)) / (2 * h);
        CHECK_CLOSE(fd, phi.d_theta_phi(theta), 1e-6);
    }
}

TEST(ssvi, surface_with_increasing_theta_is_calendar_arbitrage_free) {
    SSVISurface surf;
    surf.phi = PowerLawPhi{1.0, 0.45};
    surf.rho = -0.6;
    surf.expiries = {0.08, 0.25, 0.5, 1.0, 2.0};
    for (Real T : surf.expiries) surf.theta.push_back(0.0324 * T);   // flat 18 vol

    std::vector<Real> times;
    for (int i = 1; i <= 200; ++i) times.push_back(0.08 + (2.0 - 0.08) * i / 200.0);
    const auto rep = check_calendar(surf, times, 1.5, 401);
    CHECK(rep.free);
    CHECK(rep.worst_decrease >= 0.0);
}

TEST(ssvi, calendar_check_catches_a_decreasing_term_structure) {
    // Deliberately broken: theta falls between the second and third expiry.
    SSVISurface surf;
    surf.phi = PowerLawPhi{1.0, 0.45};
    surf.rho = -0.6;
    surf.expiries = {0.25, 0.5, 1.0};
    surf.theta = {0.010, 0.030, 0.020};
    const auto rep = check_calendar(surf, surf.expiries, 1.0, 201);
    CHECK(!rep.free);
    CHECK(rep.worst_decrease < 0.0);
}

// ---------------------------------------------------------------------------
// Chain processing
// ---------------------------------------------------------------------------

TEST(chain, parity_regression_recovers_the_forward_and_discount) {
    // The forward the market prices includes a borrow spread. Anything that
    // reconstructs it as spot * exp(rT) is wrong by borrow_spread * T, which is
    // exactly what this regression exists to avoid.
    SyntheticChainConfig cfg;
    cfg.add_quote_noise = false;
    cfg.round_to_tick = false;
    const auto chain = generate_synthetic_chain({0.25, 1.0}, cfg);

    for (std::size_t i = 0; i < chain.expiries.size(); ++i) {
        const auto& e = chain.expiries[i];
        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        const auto fit = implied_forward_from_parity(calls, puts, e.expiry, cfg.spot);
        CHECK(fit.ok);
        CHECK(fit.pairs_used >= 5);
        CHECK_CLOSE(fit.forward, e.true_forward, 1e-9);
        CHECK_CLOSE(fit.discount, e.true_discount, 1e-9);

        // And the naive alternative is wrong by a visible amount.
        const Real naive = cfg.spot * std::exp((cfg.quoted_rate - cfg.dividend_yield) * e.expiry);
        const Real error_bp = 1e4 * std::fabs(naive / e.true_forward - 1.0);
        std::printf("       T=%.2f  parity F=%.4f (true %.4f), naive F off by %.1f bp\n",
                    e.expiry, fit.forward, e.true_forward, error_bp);
        CHECK(error_bp > 10.0);
    }
}

TEST(chain, parity_regression_survives_tick_rounding_and_noise) {
    SyntheticChainConfig cfg;    // defaults: tick rounding and quote noise on
    const auto chain = generate_synthetic_chain({0.25, 1.0}, cfg);
    for (const auto& e : chain.expiries) {
        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        const auto fit = implied_forward_from_parity(calls, puts, e.expiry, cfg.spot);
        CHECK(fit.ok);
        // A tenth of a nickel of forward on a 4275 index is 1.2 bp.
        CHECK_CLOSE(fit.forward, e.true_forward, 5e-4);
        CHECK_CLOSE(fit.discount, e.true_discount, 5e-4);
    }
}

TEST(chain, parity_regression_reports_failure_rather_than_guessing) {
    std::vector<RawQuote> calls, puts;
    const auto fit = implied_forward_from_parity(calls, puts, 1.0, 100.0);
    CHECK(!fit.ok);
    CHECK(fit.pairs_used == 0);
    CHECK_THROWS(implied_forward_from_parity(calls, puts, -1.0, 100.0));
}

TEST(chain, slice_builder_keeps_the_out_of_the_money_side_and_reports_what_it_dropped) {
    SyntheticChainConfig cfg;
    const auto chain = generate_synthetic_chain({0.25}, cfg);
    const auto& e = chain.expiries[0];

    SliceBuildReport rep;
    FilterConfig fc;
    const auto slice = build_slice(e.quotes, e.true_forward, e.expiry, e.true_discount, fc, &rep);

    CHECK(!slice.empty());
    CHECK(rep.kept == int(slice.size()));
    std::printf("       %d quotes in, %d kept (%d one-sided, %d cheap, %d wide, "
                "%d moneyness, %d arb)\n",
                rep.input_quotes, rep.kept, rep.dropped_one_sided, rep.dropped_cheap,
                rep.dropped_wide, rep.dropped_moneyness, rep.dropped_arbitrage);

    for (const auto& p : slice) {
        if (p.type == OptionType::Call) CHECK(p.strike >= e.true_forward);
        else                            CHECK(p.strike < e.true_forward);
        CHECK(p.implied_vol > 0.0);
        CHECK(p.vega > 0.0);
        CHECK(p.weight > 0.0);
    }
    // Sorted by log-moneyness, which downstream fitting assumes.
    for (std::size_t i = 1; i < slice.size(); ++i) {
        CHECK(slice[i].log_moneyness >= slice[i - 1].log_moneyness);
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

TEST(calibrate, svi_recovers_the_slice_that_generated_the_data) {
    // Noiseless data from a known SVI slice: the fit should reproduce the curve,
    // not merely fit it well. Parameters themselves are not asserted -- SVI is
    // near-degenerate in (a, b, sigma) for a shallow smile, and two visibly
    // different parameter sets can give the same curve to 1e-9. The curve is the
    // observable, so the curve is what is checked.
    const SVIRaw truth = reference_slice();
    const Real T = 0.5;
    std::vector<SurfacePoint> pts;
    for (Real k = -0.6; k <= 0.45; k += 0.03) {
        SurfacePoint p;
        p.log_moneyness = k;
        p.implied_vol = truth.implied_vol(k, T);
        p.total_variance = truth.total_variance(k);
        p.weight = 1.0;
        pts.push_back(p);
    }

    const auto fit = fit_svi_slice(pts, T);
    CHECK(fit.converged);
    std::printf("       RMSE %.3e vol points, max %.3e, %d LM iterations\n",
                fit.rmse_vol, fit.max_error_vol, fit.iterations);
    CHECK(fit.rmse_vol < 1e-7);
    for (Real k = -0.6; k <= 0.45; k += 0.01) {
        CHECK_ABS(fit.params.total_variance(k), truth.total_variance(k), 1e-8);
    }
    CHECK(fit.butterfly.free);
}

TEST(calibrate, ssvi_fits_a_synthetic_chain_arbitrage_free) {
    const auto chain = generate_synthetic_chain();

    std::vector<std::vector<SurfacePoint>> slices;
    std::vector<Real> expiries, theta;
    int total = 0;
    for (const auto& e : chain.expiries) {
        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        const auto fwd = implied_forward_from_parity(calls, puts, e.expiry, 4275.0);
        CHECK(fwd.ok);
        FilterConfig fc;
        auto s = build_slice(e.quotes, fwd.forward, e.expiry, fwd.discount, fc);
        if (s.size() < 6) continue;
        total += int(s.size());
        theta.push_back(atm_total_variance(s));
        expiries.push_back(e.expiry);
        slices.push_back(std::move(s));
    }
    CHECK(slices.size() >= 5);

    const auto fit = fit_ssvi(slices, expiries, theta);
    std::printf("       %d quotes, %zu expiries: RMSE %.4f vol points (%.2f), max %.4f\n",
                total, slices.size(), fit.rmse_vol,
                fit.rmse_vol * 100.0, fit.max_error_vol);
    std::printf("       eta=%.4f gamma=%.4f rho=%.4f\n",
                fit.surface.phi.eta, fit.surface.phi.gamma, fit.surface.rho);

    CHECK(fit.converged);
    CHECK(fit.rmse_vol < 0.01);      // under one vol point

    // Arbitrage: zero violations, by both the parametric conditions and a direct
    // scan of the density.
    CHECK(fit.calendar.free);
    CHECK(fit.calendar.worst_decrease >= 0.0);
    int cond_ok = 0;
    for (std::size_t i = 0; i < fit.conditions.size(); ++i) {
        const auto& c = fit.conditions[i];
        std::printf("       T=%.3f  butterfly %s (%.3f < 4, %.3f <= 4)  calendar %s  "
                    "density min g = %+.3e\n",
                    expiries[i], c.butterfly_free ? "ok " : "FAIL",
                    c.bf_condition_1, c.bf_condition_2,
                    c.calendar_free ? "ok " : "FAIL", fit.butterfly[i].min_g);
        if (c.butterfly_free && c.calendar_free) ++cond_ok;
        CHECK(fit.butterfly[i].free);
        CHECK(fit.butterfly[i].min_g >= 0.0);
        CHECK_CLOSE(fit.butterfly[i].density_integral, 1.0, 1e-4);
    }
    CHECK(cond_ok == int(fit.conditions.size()));
}

TEST(calibrate, essvi_fits_better_than_ssvi_and_stays_arbitrage_free) {
    // The reason eSSVI exists: one rho cannot describe a front-end skew and a
    // long-end skew at the same time. If freeing rho per slice does not improve
    // the fit on a surface built with a term structure of skew, something is
    // wrong with the implementation.
    const auto chain = generate_synthetic_chain();

    std::vector<std::vector<SurfacePoint>> slices;
    std::vector<Real> expiries, theta;
    for (const auto& e : chain.expiries) {
        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        const auto fwd = implied_forward_from_parity(calls, puts, e.expiry, 4275.0);
        FilterConfig fc;
        auto s = build_slice(e.quotes, fwd.forward, e.expiry, fwd.discount, fc);
        if (s.size() < 6) continue;
        theta.push_back(atm_total_variance(s));
        expiries.push_back(e.expiry);
        slices.push_back(std::move(s));
    }

    const auto ssvi = fit_ssvi(slices, expiries, theta);
    const auto essvi = fit_essvi(slices, expiries, theta);
    std::printf("       SSVI  RMSE %.5f vol points\n       eSSVI RMSE %.5f vol points\n",
                ssvi.rmse_vol, essvi.rmse_vol);
    for (std::size_t i = 0; i < essvi.surface.size(); ++i) {
        std::printf("       T=%.3f  rho=%+.4f  psi=%.4f\n",
                    expiries[i], essvi.surface.rho[i], essvi.surface.psi[i]);
    }

    CHECK(essvi.rmse_vol < ssvi.rmse_vol);
    CHECK(essvi.calendar_conditions_hold);
    for (const auto& b : essvi.butterfly) CHECK(b.free);
}

TEST(calibrate, spline_in_delta_fits_better_and_is_not_a_probability_distribution) {
    // The control experiment. A cubic spline through the smile interpolates, so
    // its in-sample error is zero -- strictly better than any parametric fit by
    // that measure. It is also not arbitrage-free, and the density it implies
    // goes negative. This is what the SVI constraints are buying.
    const auto chain = generate_synthetic_chain({0.25});
    const auto& e = chain.expiries[0];
    std::vector<RawQuote> calls, puts;
    split_by_type(e.quotes, calls, puts);
    const auto fwd = implied_forward_from_parity(calls, puts, e.expiry, 4275.0);
    FilterConfig fc;
    const auto slice = build_slice(e.quotes, fwd.forward, e.expiry, fwd.discount, fc);
    CHECK(slice.size() > 10);

    const auto svi = fit_svi_slice(slice, e.expiry);

    // Spline of total variance against log-moneyness through every quote.
    std::vector<Real> ks, ws;
    for (const auto& p : slice) {
        if (!ks.empty() && p.log_moneyness <= ks.back()) continue;   // strictly increasing
        ks.push_back(p.log_moneyness);
        ws.push_back(p.total_variance);
    }
    const CubicSpline spline(ks, ws);

    struct SplineSlice {
        const CubicSpline* s;
        Real total_variance(Real k) const { return (*s)(k); }
        Real dw(Real k) const { return s->derivative(k); }
        Real d2w(Real k) const { return s->second_derivative(k); }
    } ss{&spline};

    Real spline_ss = 0.0;
    for (const auto& p : slice) {
        const Real vol = std::sqrt(std::fmax(spline(p.log_moneyness), 0.0) / e.expiry);
        spline_ss += sqr(vol - p.implied_vol);
    }
    const Real spline_rmse = std::sqrt(spline_ss / Real(slice.size()));

    const auto spline_bf = check_butterfly(ss, e.expiry, 0.9 * ks.back(), 4001);
    const auto svi_bf = check_butterfly(svi.params, e.expiry, 0.9 * ks.back(), 4001);

    std::printf("       spline: RMSE %.6f vol points, min g = %+.4e, %d density violations\n",
                spline_rmse, spline_bf.min_g, spline_bf.violations);
    std::printf("       SVI   : RMSE %.6f vol points, min g = %+.4e, %d density violations\n",
                svi.rmse_vol, svi_bf.min_g, svi_bf.violations);

    CHECK(spline_rmse < svi.rmse_vol);   // it interpolates, so of course
    CHECK(spline_bf.violations > 0);            // and it is not a distribution
    CHECK(svi_bf.violations == 0);
}

TEST(calibrate, essvi_calendar_conditions_hold_on_a_dense_board) {
    // Regression. With seven well-separated expiries a penalty term was enough
    // to keep the coupled skew condition; with fourteen, including weeklies at
    // the front, psi_2 - psi_1 between neighbours becomes small and the penalty
    // lost to data residuals weighted by 1/spread. The conditions are now
    // structural, so this is a test that the reparameterisation is correct
    // rather than that a weight is large enough.
    std::vector<Real> expiries;
    for (int d : {7, 14, 21, 30, 45, 60, 91, 121, 152, 182, 273, 365, 547, 730}) {
        expiries.push_back(d / 365.0);
    }
    const auto chain = generate_synthetic_chain(expiries);

    std::vector<std::vector<SurfacePoint>> slices;
    std::vector<Real> ts, theta;
    for (const auto& e : chain.expiries) {
        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        const auto fwd = implied_forward_from_parity(calls, puts, e.expiry, 4275.0);
        FilterConfig fc;
        auto s = build_slice(e.quotes, fwd.forward, e.expiry, fwd.discount, fc);
        if (s.size() < 6) continue;
        theta.push_back(atm_total_variance(s));
        ts.push_back(e.expiry);
        slices.push_back(std::move(s));
    }
    CHECK(slices.size() >= 12);

    const auto fit = fit_essvi(slices, ts, theta);
    std::printf("       %zu expiries, %d quotes: RMSE %.4f vol points, calendar %s\n",
                slices.size(), fit.quotes, fit.rmse_vol,
                fit.calendar_conditions_hold ? "ok" : "VIOLATED");
    CHECK(fit.calendar_conditions_hold);
    CHECK(fit.rmse_vol < 0.005);
    for (const auto& b : fit.butterfly) CHECK(b.free);

    // And the conditions hold term by term, which is the actual claim.
    for (std::size_t i = 1; i < fit.surface.size(); ++i) {
        CHECK(fit.surface.theta[i] >= fit.surface.theta[i - 1]);
        CHECK(fit.surface.psi[i] >= fit.surface.psi[i - 1]);
        const Real lhs = std::fabs(fit.surface.rho[i] * fit.surface.psi[i] -
                                   fit.surface.rho[i - 1] * fit.surface.psi[i - 1]);
        CHECK(lhs <= fit.surface.psi[i] - fit.surface.psi[i - 1] + 1e-12);
        CHECK(std::fabs(fit.surface.rho[i]) < 1.0);
    }
}
