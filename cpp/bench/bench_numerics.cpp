// PDE, Monte Carlo, Longstaff-Schwartz and adjoint Greeks.
#include "bench.hpp"
#include "vse/binomial.hpp"
#include "vse/black.hpp"
#include "vse/heston.hpp"
#include "vse/lsmc.hpp"
#include "vse/mc.hpp"
#include "vse/mc_aad.hpp"
#include "vse/pde.hpp"
#include "vse/pde_heston.hpp"

#include <string>
#include <vector>

using namespace vse;

namespace {

HestonParams index_like() { return HestonParams{0.0348, 1.58, 0.0447, 0.92, -0.74}; }

}  // namespace

BENCH("pde.european") {
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Greeks g = bs_greeks(S, K, T, r, q, sig, OptionType::Call);

    PDEConfig cfg;
    cfg.space_steps = 800;
    cfg.time_steps = 400;
    const PDEResult res = pde_vanilla(S, K, T, r, q, sig, OptionType::Call,
                                      Exercise::European, cfg);

    vsebench::report("pde.european.price_error", "Crank-Nicolson against Black-Scholes",
                     std::fabs(res.price / g.price - 1.0), "relative",
                     "800 space steps x 400 time steps, Rannacher startup");
    vsebench::report("pde.european.delta_error", "Delta from the same grid",
                     std::fabs(res.delta - g.delta), "absolute", "");
    vsebench::report("pde.european.gamma_error", "Gamma from the same grid",
                     std::fabs(res.gamma - g.gamma), "absolute", "");

    // The Rannacher demonstration, as a number.
    Real worst_without = 0.0, worst_with = 0.0;
    for (Real T2 : {0.02, 0.10, 0.25}) {
        const Greeks g2 = bs_greeks(K, K, T2, r, 0.0, 0.2, OptionType::Call);
        PDEConfig c2;
        c2.space_steps = 800;
        c2.time_steps = 100;
        c2.rannacher_steps = 0;
        worst_without = std::fmax(worst_without,
                                  std::fabs(pde_vanilla(K, K, T2, r, 0.0, 0.2, OptionType::Call,
                                                        Exercise::European, c2).gamma /
                                                g2.gamma - 1.0));
        c2.rannacher_steps = 2;
        worst_with = std::fmax(worst_with,
                               std::fabs(pde_vanilla(K, K, T2, r, 0.0, 0.2, OptionType::Call,
                                                     Exercise::European, c2).gamma /
                                             g2.gamma - 1.0));
    }
    vsebench::report("pde.gamma_without_rannacher", "Gamma error, plain Crank-Nicolson",
                     worst_without, "relative",
                     "at the strike; CN is stable but not damping and a payoff kink is "
                     "all high-frequency");
    vsebench::report("pde.gamma_with_rannacher", "Gamma error, two implicit half-steps first",
                     worst_with, "relative", "same grid, same everything else");

    const double ns = vsebench::time_ns_per_op([&] {
        return pde_vanilla(S, K, T, r, q, sig, OptionType::Call, Exercise::European, cfg).price;
    }, 1, 1.0, 5);
    vsebench::report("pde.european.time", "One European PDE solve", ns / 1e6, "ms",
                     "800 x 400, single-threaded");
}

BENCH("pde.american") {
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Real reference = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, true, 4001);

    PDEConfig cfg;
    cfg.space_steps = 800;
    cfg.time_steps = 400;
    const PDEResult psor = pde_vanilla(S, K, T, r, q, sig, OptionType::Put,
                                       Exercise::American, cfg);
    cfg.use_brennan_schwartz = true;
    const PDEResult bs = pde_vanilla(S, K, T, r, q, sig, OptionType::Put,
                                     Exercise::American, cfg);

    vsebench::report("pde.american.error_bp", "American put against a 4001-step lattice",
                     1e4 * std::fabs(psor.price - reference) / reference, "basis points",
                     "PSOR on an 800 x 400 grid");
    vsebench::report("pde.american.solver_agreement", "PSOR against Brennan-Schwartz",
                     std::fabs(psor.price - bs.price), "absolute",
                     "same discretisation, so this measures the solvers and not the grid");
    vsebench::report("pde.american.psor_iterations", "PSOR sweeps, summed over all steps",
                     Real(psor.psor_iterations), "count", "");
}

BENCH("pde.heston_adi") {
    const HestonParams p = index_like();
    const Real S = 100, r = 0.03, q = 0.01, T = 1.0;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);

    HestonPDEConfig cfg;
    cfg.spot_steps = 240;
    cfg.var_steps = 120;
    cfg.time_steps = 200;

    Real worst = 0.0;
    for (Real K : {80.0, 100.0, 125.0}) {
        const Real lewis = df * heston_call_lewis(p, F, K, T, 64, 64);
        const Real adi = heston_pde(p, S, K, T, r, q, OptionType::Call, cfg).price;
        worst = std::fmax(worst, std::fabs(adi / lewis - 1.0));
    }
    vsebench::report("pde.heston.vs_characteristic_function",
                     "Heston ADI against the Lewis integral", worst, "max relative",
                     "240 x 120 x 200, Craig-Sneyd; three strikes from 0.8 to 1.25 of spot");

    // Douglas against Craig-Sneyd at equal cost per step.
    HestonPDEConfig d = cfg;
    d.craig_sneyd = false;
    d.time_steps = 100;
    HestonPDEConfig c = cfg;
    c.time_steps = 100;
    const Real ref = df * heston_call_lewis(p, F, 125.0, T, 64, 64);
    vsebench::report("pde.heston.douglas_error", "Douglas ADI at 100 time steps",
                     std::fabs(heston_pde(p, S, 125.0, T, r, q, OptionType::Call, d).price / ref - 1.0),
                     "relative", "first-order in time because the mixed term is explicit");
    vsebench::report("pde.heston.craig_sneyd_error", "Craig-Sneyd at 100 time steps",
                     std::fabs(heston_pde(p, S, 125.0, T, r, q, OptionType::Call, c).price / ref - 1.0),
                     "relative", "one extra explicit correction restores second order");

    const double ns = vsebench::time_ns_per_op([&] {
        return heston_pde(p, S, 100.0, T, r, q, OptionType::Call, cfg).price;
    }, 1, 1.0, 3);
    vsebench::report("pde.heston.time", "One Heston ADI solve", ns / 1e6, "ms",
                     "240 x 120 x 200, single-threaded");
}

BENCH("mc.variance_reduction") {
    const HestonParams p = index_like();
    const Real S = 100, K = 100, T = 1.0, r = 0.03, q = 0.01;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);
    const Real exact = df * heston_call_lewis(p, F, K, T, 64, 64);

    auto run = [&](bool anti, bool control, bool qmc, bool conditional) {
        MCConfig cfg;
        cfg.paths = 200000;
        cfg.steps = 16;
        cfg.antithetic = anti;
        cfg.control_variate = control;
        cfg.conditional = conditional;
        cfg.sampling = qmc ? Sampling::SobolBridge : Sampling::PseudoRandom;
        return heston_mc(p, S, K, T, r, q, OptionType::Call, cfg);
    };

    const MCResult plain = run(false, false, false, false);
    struct Case { const char* id; const char* name; bool a, c, q, cond; };
    std::printf("  %-34s %11s %10s\n", "technique", "std error", "paths saved");
    std::printf("  %-34s %11.3e %10s\n", "none", plain.standard_error, "1x");

    for (Case k : {Case{"antithetic", "Antithetic paths", true, false, false, false},
                   Case{"control", "plus control variates", true, true, false, false},
                   Case{"sobol", "plus randomised Sobol and bridge", true, true, true, false},
                   Case{"conditional", "plus conditional Monte Carlo", true, true, true, true}}) {
        const MCResult r2 = run(k.a, k.c, k.q, k.cond);
        const Real factor = sqr(plain.standard_error / r2.standard_error);
        std::printf("  %-34s %11.3e %9.0fx\n", k.name, r2.standard_error, factor);
        vsebench::report(std::string("mc.variance_reduction.") + k.id, k.name, factor,
                         "paths saved",
                         "ratio of variances against a plain pseudo-random estimator, "
                         "200k paths, 16 steps");
        vsebench::report(std::string("mc.accuracy.") + k.id,
                         std::string("Error in standard errors, ") + k.name,
                         (r2.price - exact) / r2.standard_error, "std errors",
                         "against the characteristic function");
    }

    // Discretisation bias, which is a different thing from sampling error.
    for (int steps : {8, 32}) {
        MCConfig cfg;
        cfg.paths = 200000;
        cfg.steps = steps;
        cfg.sampling = Sampling::PseudoRandom;
        cfg.conditional = true;
        cfg.scheme = HestonScheme::AndersenQE;
        const Real qe = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg).price - exact;
        cfg.scheme = HestonScheme::EulerFullTruncation;
        const Real euler = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg).price - exact;
        vsebench::report("mc.bias.qe_" + std::to_string(steps),
                         "Andersen QE bias at " + std::to_string(steps) + " steps", qe,
                         "absolute", "");
        vsebench::report("mc.bias.euler_" + std::to_string(steps),
                         "Euler full-truncation bias at " + std::to_string(steps) + " steps",
                         euler, "absolute", "same paths, same everything else");
    }

    // The martingale correction, measured where it is visible.
    MCConfig cfg;
    cfg.paths = 400000;
    cfg.steps = 2;
    cfg.sampling = Sampling::SobolBridge;
    cfg.antithetic = true;
    cfg.control_variate = false;
    cfg.martingale_correction = true;
    const Real with = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg).forward_error;
    cfg.martingale_correction = false;
    const Real without = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg).forward_error;
    vsebench::report("mc.martingale.corrected", "|E[S_T]/F - 1| with the correction", with,
                     "absolute", "2 steps, where the discretisation drift is visible");
    vsebench::report("mc.martingale.uncorrected", "|E[S_T]/F - 1| without it", without,
                     "absolute", "");
}

BENCH("lsmc.duality") {
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Real american = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, true, 4001);
    vsebench::report("lsmc.reference", "American put, 4001-step Leisen-Reimer", american,
                     "price", "the value both Monte Carlo bounds are checked against");

    // The lower bound alone, at a production path count.
    LSMCConfig cheap;
    cheap.exercise_dates = 50;
    cheap.paths = 400000;
    cheap.training_paths = 200000;
    const LSMCResult low = lsmc_american(S, K, T, r, q, sig, OptionType::Put, cheap);
    vsebench::report("lsmc.lower_bound_error_bp",
                     "Longstaff-Schwartz lower bound against the lattice",
                     1e4 * (american - low.lower) / american, "basis points",
                     "50 exercise dates, 400k pricing paths on a policy fitted from 200k "
                     "separate ones");
    vsebench::report("lsmc.lower_bound_se_bp", "Its standard error",
                     1e4 * low.lower_se / american, "basis points",
                     "with a European control variate");
    vsebench::report("lsmc.control_correlation",
                     "Correlation of the American and European payoffs",
                     low.control_correlation, "dimensionless", "");

    // The dual bound, and how its bias converges in the inner path count.
    std::printf("  %8s %10s %10s %10s\n", "inner", "lower", "upper", "gap (bp)");
    for (long inner : {100L, 200L, 400L, 800L}) {
        LSMCConfig cfg;
        cfg.exercise_dates = 12;
        cfg.paths = 200000;
        cfg.training_paths = 100000;
        cfg.run_dual = true;
        cfg.dual_outer_paths = 500;
        cfg.dual_inner_paths = inner;
        const LSMCResult r2 = lsmc_american(S, K, T, r, q, sig, OptionType::Put, cfg);
        const Real gap_bp = 1e4 * r2.duality_gap / r2.lower;
        std::printf("  %8ld %10.5f %10.5f %10.1f\n", inner, r2.lower, r2.upper, gap_bp);
        vsebench::report("lsmc.duality_gap_bp_" + std::to_string(inner),
                         "Duality gap with " + std::to_string(inner) + " inner paths", gap_bp,
                         "basis points",
                         inner == 100 ? "12 exercise dates; the upper bound carries an "
                                        "inner-simulation bias that falls with the path count"
                                      : "");
    }
}

BENCH("aad.greeks") {
    // The headline for the adjoint layer.
    const HestonParams p{0.04, 3.0, 0.04, 0.36, -0.74};   // Feller satisfied; see below
    MCConfig cfg;
    cfg.paths = 100000;
    cfg.steps = 16;
    cfg.seed = 4242;
    cfg.scheme = HestonScheme::EulerFullTruncation;

    const double price_ns = vsebench::time_ns_per_op([&] {
        return heston_mc_price_only(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
    }, 1, 1.0, 5);
    const double aad_ns = vsebench::time_ns_per_op([&] {
        return heston_mc_greeks_aad(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg).price;
    }, 1, 1.0, 5);
    const double bump_ns = vsebench::time_ns_per_op([&] {
        return heston_mc_greeks_bump(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg).price;
    }, 1, 2.0, 3);

    const MCGreeksResult aad =
        heston_mc_greeks_aad(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
    const MCGreeksResult bump =
        heston_mc_greeks_bump(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);

    Real worst = 0.0;
    for (int k = 0; k < kHestonRiskFactors; ++k) {
        const Real a = aad.gradient[std::size_t(k)], b = bump.gradient[std::size_t(k)];
        worst = std::fmax(worst, std::fabs(a - b) /
                                     std::fmax(std::fabs(b), 4.0 * aad.gradient_se[std::size_t(k)]));
    }

    std::printf("  %zu tape nodes per path, %d risk factors\n", aad.tape_nodes_per_path,
                kHestonRiskFactors);

    vsebench::report("aad.overhead", "Adjoint pass against an undifferentiated price",
                     aad_ns / price_ns, "x",
                     "same code path, 10 risk factors, 100k paths x 16 steps");
    vsebench::report("aad.speedup", "Adjoint against bump-and-revalue",
                     bump_ns / aad_ns, "x",
                     "bumping needs 21 repricings with common random numbers");
    vsebench::report("aad.factors", "Risk factors from one backward sweep",
                     Real(kHestonRiskFactors), "count",
                     "spot, v0, kappa, theta, sigma, rho, rate, dividend, strike, expiry");
    vsebench::report("aad.tape_nodes", "Tape nodes per path",
                     Real(aad.tape_nodes_per_path), "count", "16 steps");
    vsebench::report("aad.vs_bump_agreement", "Adjoint against bumped Greeks", worst,
                     "worst, scaled by the larger of the value and its Monte Carlo error",
                     "");

    // Black-Scholes, where the answer is exact.
    Tape& tape = Tape::active();
    tape.clear();
    const ADouble a_s(100.0), a_k(105.0), a_t(0.7), a_r(0.03), a_q(0.012), a_sig(0.28);
    const ADouble v = bs_price_generic<ADouble>(a_s, a_k, a_t, a_r, a_q, a_sig, OptionType::Call);
    tape.backpropagate(v.index());
    const Greeks g = bs_greeks(100.0, 105.0, 0.7, 0.03, 0.012, 0.28, OptionType::Call);
    Real bs_worst = 0.0;
    bs_worst = std::fmax(bs_worst, std::fabs(a_s.adjoint() / g.delta - 1.0));
    bs_worst = std::fmax(bs_worst, std::fabs(a_sig.adjoint() / g.vega - 1.0));
    bs_worst = std::fmax(bs_worst, std::fabs(a_r.adjoint() / g.rho - 1.0));
    bs_worst = std::fmax(bs_worst, std::fabs(a_k.adjoint() / g.dual_delta - 1.0));
    bs_worst = std::fmax(bs_worst, std::fabs(-a_t.adjoint() / g.theta - 1.0));
    vsebench::report("aad.black_scholes_exactness",
                     "Adjoint Black-Scholes Greeks against the closed forms", bs_worst,
                     "max relative", "five sensitivities from one sweep of a 34-node tape");

    // The limitation, quantified.
    const HestonParams violated = index_like();
    const MCGreeksResult bad =
        heston_mc_greeks_aad(violated, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
    vsebench::report("aad.variance_greek_noise_feller_violated",
                     "Noise in d(price)/dv0 when Feller fails",
                     bad.gradient_se[1] / std::fabs(bad.gradient[1]),
                     "standard error over value",
                     "pathwise derivatives pass through 1/(2 sqrt(v)), unbounded as v "
                     "approaches zero");
    vsebench::report("aad.variance_greek_noise_feller_satisfied",
                     "The same quantity when Feller holds",
                     aad.gradient_se[1] / std::fabs(aad.gradient[1]),
                     "standard error over value", "");
    vsebench::report("aad.delta_noise_feller_violated",
                     "Noise in delta when Feller fails",
                     bad.gradient_se[0] / std::fabs(bad.gradient[0]),
                     "standard error over value",
                     "spot-like factors never touch the square root and are unaffected");
}
