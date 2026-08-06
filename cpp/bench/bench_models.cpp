// Heston: pricing throughput, cross-method agreement, calibration time.
#include "bench.hpp"
#include "vse/bates.hpp"
#include "vse/calibrate_heston.hpp"
#include "vse/heston.hpp"
#include "vse/implied_vol.hpp"
#include "vse/rng.hpp"
#include "vse/sabr.hpp"
#include "vse/smile_repair.hpp"

#include <array>
#include <vector>

using namespace vse;

namespace {

HestonParams index_like() { return HestonParams{0.0348, 1.58, 0.0447, 0.92, -0.74}; }

/// A board shaped like a listed index chain: 14 expiries, 31 strikes each.
struct Board {
    std::vector<Real> expiries;
    std::vector<std::vector<Real>> strikes;
    Real forward = 100.0;
    int quotes = 0;
};

Board make_board() {
    Board b;
    for (int d : {7, 14, 21, 30, 45, 60, 91, 121, 152, 182, 273, 365, 547, 730}) {
        const Real T = d / 365.0;
        b.expiries.push_back(T);
        std::vector<Real> ks;
        const Real sd = 0.19 * std::sqrt(T);
        for (int i = 0; i < 31; ++i) {
            ks.push_back(b.forward * std::exp((-3.0 + 6.0 * i / 30.0) * sd));
        }
        b.quotes += int(ks.size());
        b.strikes.push_back(std::move(ks));
    }
    return b;
}

}  // namespace

BENCH("heston.pricing") {
    const auto p = index_like();
    const Board b = make_board();

    // Per-option, rebuilding the characteristic function for every strike.
    const double per_option = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (std::size_t e = 0; e < b.expiries.size(); ++e) {
            for (Real K : b.strikes[e]) acc += heston_call_lewis(p, b.forward, K, b.expiries[e]);
        }
        return acc;
    }, b.quotes, 1.0, 5);

    // Per-slice, sharing the characteristic function across the strikes.
    const double per_slice = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (std::size_t e = 0; e < b.expiries.size(); ++e) {
            const HestonSliceEngine engine(p, b.expiries[e]);
            for (Real K : b.strikes[e]) acc += engine.call(b.forward, K);
        }
        return acc;
    }, b.quotes, 1.0, 5);

    const double per_slice_grad = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (std::size_t e = 0; e < b.expiries.size(); ++e) {
            const HestonSliceEngine engine(p, b.expiries[e], true);
            for (Real K : b.strikes[e]) acc += engine.call_and_gradient(b.forward, K).price;
        }
        return acc;
    }, b.quotes, 1.0, 5);

    vsebench::report("heston.price.per_option", "Heston price, one option at a time",
                     per_option / 1000.0, "us/option",
                     "32-node Gauss-Legendre on 16 panels of the Lewis integral");
    vsebench::report("heston.price.per_slice", "Heston price, sharing the CF across a slice",
                     per_slice / 1000.0, "us/option",
                     "same quadrature, 14 expiries x 31 strikes");
    vsebench::report("heston.price.slice_speedup", "Speedup from caching the CF per expiry",
                     per_option / per_slice, "x", "");
    vsebench::report("heston.gradient.per_slice",
                     "Heston price plus 5 exact derivatives, per slice",
                     per_slice_grad / 1000.0, "us/option", "forward-mode AD through the CF");
    vsebench::report("heston.gradient.overhead", "Cost of 5 derivatives over the price alone",
                     per_slice_grad / per_slice, "x",
                     "central differences would need 11 repricings, at ~1e-8 accuracy");
}

BENCH("heston.cross_method") {
    // Every model against every method: the validation matrix.
    //
    // Two numbers, because one is not enough. Both methods have an ABSOLUTE
    // accuracy floor rather than a relative one -- Lewis computes F minus an
    // integral close to F, and the Carr-Madan transform reconstructs a damped
    // price whose reconstruction error does not shrink with the price -- so a
    // single relative figure is dominated by whichever wing option happens to be
    // worth least on the grid. Reported here: the relative agreement over
    // options worth more than 1e-4 of the forward (a cent on a hundred, well
    // below any listed tick), and separately the absolute floor itself.
    const auto p = index_like();
    const Real F = 100.0;

    Real worst_fft_rel = 0.0, worst_fft_abs = 0.0;
    Real worst_quad_rel = 0.0, worst_quad_abs = 0.0;
    int compared = 0;
    for (Real T : {7.0 / 365, 30.0 / 365, 0.25, 1.0, 2.0}) {
        const auto grid = heston_carr_madan(p, F, T);
        const HestonSliceEngine engine(p, T);
        const HestonSliceEngine reference(p, T, false, 96, 96);
        const Real sd = 0.19 * std::sqrt(T);
        for (int i = 0; i <= 12; ++i) {
            const Real K = F * std::exp((-3.0 + 6.0 * i / 12.0) * sd);
            const Real ref = reference.call(F, K);
            const Real otm = (K >= F) ? ref : ref - (F - K);

            worst_fft_abs = std::fmax(worst_fft_abs, std::fabs(grid.call_at(K) - ref));
            worst_quad_abs = std::fmax(worst_quad_abs, std::fabs(engine.call(F, K) - ref));
            if (otm < 1e-4 * F) continue;
            ++compared;
            worst_fft_rel = std::fmax(worst_fft_rel, std::fabs(grid.call_at(K) - ref) / otm);
            worst_quad_rel = std::fmax(worst_quad_rel, std::fabs(engine.call(F, K) - ref) / otm);
        }
    }

    vsebench::report("heston.fft_vs_quadrature", "Carr-Madan FFT against the Lewis integral",
                     worst_fft_rel, "max relative",
                     "options worth over 1e-4 F; " + std::to_string(compared) +
                         " of them across 5 expiries");
    vsebench::report("heston.fft.absolute_floor", "Carr-Madan absolute accuracy",
                     worst_fft_abs / F, "fraction of the forward",
                     "the floor below which a relative comparison is meaningless");
    vsebench::report("heston.quadrature_convergence",
                     "Default Lewis rule against a converged one", worst_quad_rel,
                     "max relative", "32 nodes x 16 panels against 96 x 96");
    vsebench::report("heston.quadrature.absolute_floor", "Lewis absolute accuracy",
                     worst_quad_abs / F, "fraction of the forward",
                     "set by the cancellation in C = F - integral");

    // Black-Scholes limit: an independent closed form.
    HestonParams degenerate{0.04, 2.0, 0.04, 1e-7, 0.0};
    Real worst_bs = 0.0;
    for (Real K : {50.0, 70.0, 90.0, 100.0, 115.0, 140.0, 200.0}) {
        const Real h = heston_call_lewis(degenerate, F, K, 1.0);
        const Real bs = black76_undiscounted(F, K, 1.0, 0.2, OptionType::Call);
        worst_bs = std::fmax(worst_bs, std::fabs(h / bs - 1.0));
    }
    vsebench::report("heston.black_scholes_limit",
                     "Heston at vol-of-vol 1e-7 against Black-Scholes", worst_bs,
                     "max relative", "the degenerate limit, where the textbook CF loses 10%");

    // Martingale identity, the sharpest single check on the CF.
    Real worst_martingale = 0.0;
    for (Real T : {0.02, 1.0, 10.0, 30.0}) {
        const auto z = heston_cf(p, T, std::complex<Real>(0.0, -1.0));
        worst_martingale = std::fmax(worst_martingale, std::abs(z - std::complex<Real>(1.0, 0.0)));
    }
    vsebench::report("heston.martingale_error", "|phi(-i) - 1|", worst_martingale,
                     "max absolute", "out to 30 years; must be zero for E[S_T/F] = 1");
}

BENCH("heston.calibration") {
    const Board b = make_board();
    const auto truth = index_like();

    std::vector<CalibrationQuote> quotes;
    for (std::size_t e = 0; e < b.expiries.size(); ++e) {
        const Real T = b.expiries[e];
        const HestonSliceEngine engine(truth, T);
        for (Real K : b.strikes[e]) {
            const bool otm_call = K >= b.forward;
            const Real call = engine.call(b.forward, K);
            const Real price = otm_call ? call : call - (b.forward - K);
            const auto iv = implied_volatility_ex(price, b.forward, K, T, 1.0,
                                                  otm_call ? OptionType::Call : OptionType::Put);
            if (iv.converged && iv.sigma > 0.0) {
                quotes.push_back({b.forward, K, T, iv.sigma, 1.0});
            }
        }
    }

    const HestonParams start{0.02, 3.0, 0.06, 0.5, -0.4};
    const auto fit = calibrate_heston(quotes, start);
    std::printf("  %d quotes, %d expiries: %d LM iterations, %d slice builds\n",
                fit.quotes, int(b.expiries.size()), fit.iterations, fit.slice_builds);
    std::printf("  recovered v0=%.5f kappa=%.4f theta=%.5f sigma=%.4f rho=%+.4f\n",
                fit.params.v0, fit.params.kappa, fit.params.theta, fit.params.sigma,
                fit.params.rho);

    const double ns = vsebench::time_ns_per_op([&] {
        const auto f = calibrate_heston(quotes, start);
        return f.rmse_vol;
    }, 1, 2.0, 5);

    vsebench::report("heston.calibration.time",
                     "Heston calibration, LM with an exact Jacobian", ns / 1e6, "ms",
                     std::to_string(fit.quotes) + " quotes over 14 expiries, 5 free "
                     "parameters, single-threaded");
    vsebench::report("heston.calibration.iterations", "Levenberg-Marquardt iterations",
                     double(fit.iterations), "count", "");
    vsebench::report("heston.calibration.rmse", "Heston fit to a Heston board",
                     fit.rmse_vol * 100.0, "implied vol points",
                     "noiseless data from known parameters: this measures the machinery, "
                     "not the model");
    vsebench::report("heston.calibration.v0_error", "Recovered v0 against the truth",
                     std::fabs(fit.params.v0 / truth.v0 - 1.0), "relative", "");
    vsebench::report("heston.calibration.rho_error", "Recovered rho against the truth",
                     std::fabs(fit.params.rho / truth.rho - 1.0), "relative", "");
}

BENCH("heston.stability") {
    // Not a speed number, and the more interesting result.
    //
    // Calibrating noiseless data generated by the model itself recovers the
    // parameters to twelve digits from every starting point, which proves the
    // machinery works and says nothing about whether the parameters mean
    // anything. The question that matters on a desk is: if tomorrow's quotes
    // differ from today's by the width of a bid-ask, how far do the calibrated
    // parameters move?
    //
    // So this perturbs each quote by an independent 0.2 implied vol points --
    // roughly a tick on a liquid index option -- and calibrates twenty such
    // "days" from the same starting point. The dispersion of the answers is the
    // stability of the calibration. Instability here is a property of the model,
    // not a defect in the optimiser: kappa and theta are weakly identified from
    // a single board because a fast reversion to a low long-run variance and a
    // slow one to a higher variance produce nearly the same six-month smile.
    const Board b = make_board();
    const auto truth = index_like();

    std::vector<CalibrationQuote> base;
    for (std::size_t e = 0; e < b.expiries.size(); ++e) {
        const Real T = b.expiries[e];
        const HestonSliceEngine engine(truth, T);
        for (Real K : b.strikes[e]) {
            const bool otm_call = K >= b.forward;
            const Real call = engine.call(b.forward, K);
            const Real price = otm_call ? call : call - (b.forward - K);
            const auto iv = implied_volatility_ex(price, b.forward, K, T, 1.0,
                                                  otm_call ? OptionType::Call : OptionType::Put);
            if (iv.converged && iv.sigma > 0.0) base.push_back({b.forward, K, T, iv.sigma, 1.0});
        }
    }

    const int days = 20;
    const Real quote_noise = 0.002;   // 0.2 implied vol points
    Xoshiro256pp rng(20260810);
    std::vector<std::array<Real, 5>> fits;
    Real worst_rmse = 0.0;

    for (int d = 0; d < days; ++d) {
        std::vector<CalibrationQuote> day = base;
        for (auto& q : day) q.implied_vol += quote_noise * rng.normal();
        const auto fit = calibrate_heston(day, HestonParams{0.02, 3.0, 0.06, 0.5, -0.4});
        fits.push_back({fit.params.v0, fit.params.kappa, fit.params.theta,
                        fit.params.sigma, fit.params.rho});
        worst_rmse = std::fmax(worst_rmse, fit.rmse_vol);
    }

    const char* names[5] = {"v0", "kappa", "theta", "sigma", "rho"};
    std::array<Real, 5> mean{}, sd{};
    for (int c = 0; c < 5; ++c) {
        for (const auto& f : fits) mean[std::size_t(c)] += f[std::size_t(c)] / Real(days);
        for (const auto& f : fits) {
            sd[std::size_t(c)] += sqr(f[std::size_t(c)] - mean[std::size_t(c)]);
        }
        sd[std::size_t(c)] = std::sqrt(sd[std::size_t(c)] / Real(days - 1));
    }

    const std::array<Real, 5> truth_v = {truth.v0, truth.kappa, truth.theta, truth.sigma,
                                         truth.rho};
    std::printf("  %d days, quotes perturbed by %.1f vol points:\n", days, quote_noise * 100.0);
    std::printf("  %-8s %10s %10s %10s %12s\n", "param", "truth", "mean", "sd", "cv");
    for (int c = 0; c < 5; ++c) {
        const Real cv = sd[std::size_t(c)] / std::fmax(std::fabs(mean[std::size_t(c)]), 1e-12);
        std::printf("  %-8s %10.4f %10.4f %10.4f %11.1f%%\n", names[c],
                    truth_v[std::size_t(c)], mean[std::size_t(c)], sd[std::size_t(c)],
                    100.0 * cv);
        vsebench::report(std::string("heston.stability.") + names[c],
                         std::string("Day-to-day spread in ") + names[c], 100.0 * cv,
                         "% coefficient of variation",
                         c == 0 ? "20 boards, each quote perturbed by 0.2 vol points"
                                : "");
    }
    vsebench::report("heston.stability.worst_rmse", "Worst fit RMSE across the 20 days",
                     worst_rmse * 100.0, "implied vol points", "");
}

BENCH("sabr.arbitrage") {
    // Where Hagan's expansion stops being a probability distribution, for the
    // lognormal and normal forms of the same parameters.
    const Real F = 0.03, T = 10.0;
    const SABRParams p{0.025, 0.5, -0.2, 0.45, 0.0};
    const auto ln = sabr_density_scan(p, F, T, 0.02, 3.0, 1500, false);
    const auto nm = sabr_density_scan(p, F, T, 0.02, 3.0, 1500, true);

    vsebench::report("sabr.lognormal.arbitrage_boundary",
                     "Hagan lognormal: density negative below this strike",
                     ln.arbitrage_boundary / F, "K/F",
                     "beta=0.5, nu=0.45, 10 years -- ordinary long-dated rates parameters");
    vsebench::report("sabr.normal.arbitrage_boundary",
                     "Hagan normal: density negative below this strike",
                     nm.arbitrage_boundary / F, "K/F", "same parameters");
    vsebench::report("sabr.lognormal.violations", "Grid points with a negative density",
                     double(ln.violations), "of 1500", "");
    vsebench::report("sabr.normal.violations", "Grid points with a negative density",
                     double(nm.violations), "of 1500", "");

    const double ns = vsebench::time_ns_per_op([&] {
        double acc = 0.0;
        for (int i = 0; i < 1000; ++i) {
            acc += sabr_lognormal_vol(p, F, F * (0.5 + 0.001 * i), T);
        }
        return acc;
    }, 1000);
    vsebench::report("sabr.vol.latency", "Hagan lognormal volatility", ns, "ns/strike", "");

    // The repair. Neither the shift nor the normal expansion removes the
    // arbitrage (see smile_repair.hpp for the measurements that say so);
    // projecting the prices onto the nearest convex curve does.
    const auto fixed = repair_sabr(p, F, T, 0.02, 3.0, 1501, false);
    vsebench::report("sabr.repair.violations_before",
                     "Butterfly violations, Hagan smile as given",
                     double(fixed.report.violations_before), "of 1501", "");
    vsebench::report("sabr.repair.violations_after",
                     "Butterfly violations after convexity projection",
                     double(fixed.report.violations_after), "of 1501",
                     "isotonic regression on the call slopes; exact, not iterative");
    vsebench::report("sabr.repair.min_density_before", "Worst density, Hagan smile",
                     fixed.report.min_density_before, "density", "");
    vsebench::report("sabr.repair.min_density_after", "Worst density after repair",
                     fixed.report.min_density_after, "density",
                     "at the noise floor of a second difference on this grid");
    vsebench::report("sabr.repair.max_vol_change", "Cost of the repair",
                     fixed.report.max_vol_change * 100.0, "implied vol points",
                     "worst over the grid, concentrated in the wing that was not a price");
    vsebench::report("sabr.repair.mass_before", "Probability mass, Hagan smile",
                     fixed.report.mass_before, "dimensionless",
                     "a distribution integrates to one; this is the larger of the two failures");
    vsebench::report("sabr.repair.mass_after", "Probability mass after repair",
                     fixed.report.mass_after, "dimensionless",
                     "the slope bounds recover part of it; convexity cannot recover the rest");

    const double repair_ns = vsebench::time_ns_per_op([&] {
        const auto r = repair_sabr(p, F, T, 0.02, 3.0, 1501, false);
        return r.report.min_density_after;
    }, 1501);
    vsebench::report("sabr.repair.latency", "Convexity projection", repair_ns, "ns/strike",
                     "including pricing, the projection and re-implying every strike");
}
