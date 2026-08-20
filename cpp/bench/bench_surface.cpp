// Surface calibration: fit quality, arbitrage violations, and wall-clock.
#include "bench.hpp"
#include "vse/calibrate_svi.hpp"
#include "vse/chain.hpp"
#include "vse/interp.hpp"
#include "vse/synthetic.hpp"

#include <vector>

using namespace vse;

namespace {

struct PreparedChain {
    std::vector<std::vector<SurfacePoint>> slices;
    std::vector<Real> expiries, theta;
    int quotes = 0;
    int raw_quotes = 0;
};

/// Build a large chain and run it through the full pipeline: parity regression
/// for the forward, filters, inversion, ATM anchor.
PreparedChain prepare(int strike_density = 1) {
    SyntheticChainConfig cfg;
    cfg.seed = 20260615;
    std::vector<Real> expiries;
    // A full listed board: weeklies out to two years.
    for (int d : {7, 14, 21, 30, 45, 60, 91, 121, 152, 182, 273, 365, 547, 730}) {
        expiries.push_back(d / 365.0);
    }
    if (strike_density > 1) cfg.tick = 0.01;   // finer ladder for the large case

    const auto chain = generate_synthetic_chain(expiries, cfg);
    PreparedChain out;
    for (const auto& e : chain.expiries) {
        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        out.raw_quotes += int(e.quotes.size());
        const auto fwd = implied_forward_from_parity(calls, puts, e.expiry, cfg.spot);
        if (!fwd.ok) continue;
        FilterConfig fc;
        auto s = build_slice(e.quotes, fwd.forward, e.expiry, fwd.discount, fc);
        if (s.size() < 6) continue;
        out.quotes += int(s.size());
        out.theta.push_back(atm_total_variance(s));
        out.expiries.push_back(e.expiry);
        out.slices.push_back(std::move(s));
    }
    return out;
}

}  // namespace

BENCH("surface.pipeline") {
    const auto prepared = prepare();
    std::printf("  %d raw quotes -> %d fittable across %zu expiries\n",
                prepared.raw_quotes, prepared.quotes, prepared.slices.size());

    const double ns = vsebench::time_ns_per_op([&] {
        const auto p = prepare();
        return double(p.quotes);
    }, 1, 1.0, 5);
    vsebench::report("surface.pipeline.latency",
                     "Parity regression, filters and inversion for a full board",
                     ns / 1e6, "ms",
                     "14 expiries, " + std::to_string(prepared.raw_quotes) +
                         " raw quotes, single-threaded");
}

BENCH("surface.ssvi") {
    const auto p = prepare();
    const auto fit = fit_ssvi(p.slices, p.expiries, p.theta);

    int butterfly_violations = 0;
    Real worst_g = DBL_HUGE, worst_integral_error = 0.0;
    for (const auto& b : fit.butterfly) {
        butterfly_violations += b.violations;
        worst_g = std::fmin(worst_g, b.min_g);
        worst_integral_error = std::fmax(worst_integral_error,
                                         std::fabs(b.density_integral - 1.0));
    }
    int condition_failures = 0;
    for (const auto& c : fit.conditions) {
        if (!c.butterfly_free || !c.calendar_free) ++condition_failures;
    }

    std::printf("  eta=%.4f gamma=%.4f rho=%.4f\n",
                fit.surface.phi.eta, fit.surface.phi.gamma, fit.surface.rho);

    vsebench::report("surface.ssvi.rmse", "SSVI fit to a full board",
                     fit.rmse_vol * 100.0, "implied vol points",
                     "vega/spread-weighted fit, RMSE reported unweighted over " +
                         std::to_string(p.quotes) + " quotes");
    vsebench::report("surface.ssvi.max_error", "SSVI worst single-quote error",
                     fit.max_error_vol * 100.0, "implied vol points", "");
    vsebench::report("surface.ssvi.butterfly_violations",
                     "Butterfly violations, SSVI", double(butterfly_violations), "count",
                     "grid points with g(k) < 0 across all slices");
    vsebench::report("surface.ssvi.calendar_violations",
                     "Calendar violations, SSVI", double(fit.calendar.violations), "count",
                     "grid points where total variance decreases in T");
    vsebench::report("surface.ssvi.condition_failures",
                     "Gatheral-Jacquier conditions failed",
                     double(condition_failures), "slices",
                     "closed-form Theorem 4.2 conditions, evaluated per expiry");
    vsebench::report("surface.ssvi.min_g", "Smallest value of Gatheral's g",
                     worst_g, "dimensionless", "negative would mean a negative density");
    vsebench::report("surface.ssvi.density_integral_error",
                     "Worst |integral of density - 1|", worst_integral_error,
                     "absolute", "Simpson over six wing standard deviations");

    const double ns = vsebench::time_ns_per_op([&] {
        const auto f = fit_ssvi(p.slices, p.expiries, p.theta);
        return f.rmse_vol;
    }, 1, 2.0, 5);
    vsebench::report("surface.ssvi.calibration_time",
                     "SSVI calibration, 18 starts, analytic Jacobian",
                     ns / 1e6, "ms", "single-threaded, includes all arbitrage checks");
}

BENCH("surface.essvi") {
    const auto p = prepare();
    const auto fit = fit_essvi(p.slices, p.expiries, p.theta);
    int violations = 0;
    for (const auto& b : fit.butterfly) violations += b.violations;

    vsebench::report("surface.essvi.rmse", "eSSVI fit to a full board",
                     fit.rmse_vol * 100.0, "implied vol points",
                     "rho free per slice, calendar conditions enforced as constraints");
    vsebench::report("surface.essvi.max_error", "eSSVI worst single-quote error",
                     fit.max_error_vol * 100.0, "implied vol points", "");
    vsebench::report("surface.essvi.butterfly_violations",
                     "Butterfly violations, eSSVI", double(violations), "count", "");
    vsebench::report("surface.essvi.calendar_conditions",
                     "eSSVI calendar conditions hold",
                     fit.calendar_conditions_hold ? 1.0 : 0.0, "boolean",
                     "theta and psi non-decreasing, |d(rho psi)| <= d(psi)");
}

BENCH("surface.spline_control") {
    // The control: an unconstrained cubic spline through the same quotes.
    // It interpolates, so its in-sample error is zero. It is also not a
    // probability distribution. Both numbers go in the report together, because
    // either one alone is misleading.
    const auto p = prepare();
    const std::size_t si = p.slices.size() / 2;
    const auto& slice = p.slices[si];
    const Real T = p.expiries[si];

    std::vector<Real> ks, ws;
    for (const auto& q : slice) {
        if (!ks.empty() && q.log_moneyness <= ks.back()) continue;
        ks.push_back(q.log_moneyness);
        ws.push_back(q.total_variance);
    }
    const CubicSpline spline(ks, ws);
    struct SplineSlice {
        const CubicSpline* s;
        Real total_variance(Real k) const { return (*s)(k); }
        Real dw(Real k) const { return s->derivative(k); }
        Real d2w(Real k) const { return s->second_derivative(k); }
    } ss{&spline};

    Real ss_err = 0.0;
    for (const auto& q : slice) {
        ss_err += sqr(std::sqrt(std::fmax(spline(q.log_moneyness), 0.0) / T) - q.implied_vol);
    }
    const Real spline_rmse = std::sqrt(ss_err / Real(slice.size()));
    const auto bf = check_butterfly(ss, T, 0.9 * ks.back(), 4001);

    const auto svi = fit_svi_slice(slice, T);
    const auto svi_bf = check_butterfly(svi.params, T, 0.9 * ks.back(), 4001);

    vsebench::report("surface.spline.rmse", "Cubic spline through the same slice",
                     spline_rmse * 100.0, "implied vol points",
                     "interpolates every quote, so in-sample error is ~0 by construction");
    vsebench::report("surface.spline.butterfly_violations",
                     "Butterfly violations, cubic spline", double(bf.violations), "count",
                     "of 4001 grid points; the spline is not a probability distribution");
    vsebench::report("surface.spline.min_g", "Smallest g(k), cubic spline",
                     bf.min_g, "dimensionless", "");
    vsebench::report("surface.svi_slice.rmse", "Single-slice SVI on the same quotes",
                     svi.rmse_vol * 100.0, "implied vol points", "");
    vsebench::report("surface.svi_slice.butterfly_violations",
                     "Butterfly violations, single-slice SVI",
                     double(svi_bf.violations), "count", "same grid");
}

BENCH("surface.spline_noise") {
    // WHY the spline fails, which is not the reason usually given.
    //
    // The usual story is that a cubic spline is the wrong basis. It is not: a
    // spline through EXACT quotes is arbitrage-free. What breaks it is being an
    // interpolant -- as many degrees of freedom as quotes, obliged to honour
    // every one, including the half-tick that is a rounding rather than a price.
    // A second derivative then turns half a tick into a density of several
    // thousand.
    //
    // Four boards, identical but for what is done to the quotes, so the
    // comparison isolates the cause instead of demonstrating the symptom.
    struct Case { const char* id; bool noise; bool tick; };
    const Case cases[] = {
        {"exact", false, false},
        {"tick_only", false, true},
        {"noise_only", true, false},
        {"noise_and_tick", true, true},
    };

    for (const Case& c : cases) {
        SyntheticChainConfig cfg;
        cfg.seed = 20260615;
        cfg.add_quote_noise = c.noise;
        cfg.round_to_tick = c.tick;
        const auto chain = generate_synthetic_chain({0.25}, cfg);
        const auto& e = chain.expiries[0];

        std::vector<RawQuote> calls, puts;
        split_by_type(e.quotes, calls, puts);
        const auto fwd = implied_forward_from_parity(calls, puts, e.expiry, cfg.spot);
        FilterConfig fc;
        const auto slice = build_slice(e.quotes, fwd.forward, e.expiry, fwd.discount, fc);

        std::vector<Real> ks, ws;
        for (const auto& q : slice) {
            if (!ks.empty() && q.log_moneyness <= ks.back()) continue;
            ks.push_back(q.log_moneyness);
            ws.push_back(q.total_variance);
        }
        const CubicSpline spline(ks, ws);
        struct SplineSlice {
            const CubicSpline* s;
            Real total_variance(Real k) const { return (*s)(k); }
            Real dw(Real k) const { return s->derivative(k); }
            Real d2w(Real k) const { return s->second_derivative(k); }
        } ss{&spline};
        const auto bf = check_butterfly(ss, e.expiry, 0.9 * ks.back(), 4001);

        vsebench::report(std::string("surface.spline_noise.") + c.id + ".violations",
                         std::string("Butterfly violations, spline through ") + c.id + " quotes",
                         double(bf.violations), "of 4001",
                         c.noise || c.tick ? ""
                                           : "an interpolant through exact quotes is fine");
        vsebench::report(std::string("surface.spline_noise.") + c.id + ".min_g",
                         std::string("Smallest g(k), spline through ") + c.id + " quotes",
                         bf.min_g, "dimensionless", "");
    }
}
