// Finite differences, lattices, Sobol, Monte Carlo and Longstaff-Schwartz.
//
// Kept fast enough to run on every build: the convergence studies that need
// large grids and path counts live in cpp/bench, and what is asserted here is
// correctness and the qualitative behaviour that distinguishes a working scheme
// from a broken one.
#include "harness.hpp"
#include "vse/binomial.hpp"
#include "vse/black.hpp"
#include "vse/heston.hpp"
#include "vse/lsmc.hpp"
#include "vse/mc.hpp"
#include "vse/pde.hpp"
#include "vse/pde_heston.hpp"
#include "vse/rng.hpp"
#include "vse/sobol.hpp"

#include <vector>

using namespace vse;
using namespace vsetest;

namespace {

HestonParams index_like() { return HestonParams{0.0348, 1.58, 0.0447, 0.92, -0.74}; }

}  // namespace

// ---------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------

TEST(rng, xoshiro_is_reproducible_and_uniform) {
    Xoshiro256pp a(12345), b(12345);
    for (int i = 0; i < 100; ++i) CHECK(a.next() == b.next());

    Xoshiro256pp rng(7);
    const long n = 200000;
    Real sum = 0.0, sum2 = 0.0;
    int in_range = 0;
    for (long i = 0; i < n; ++i) {
        const Real u = rng.uniform();
        if (u > 0.0 && u < 1.0) ++in_range;
        sum += u;
        sum2 += u * u;
    }
    CHECK(in_range == int(n));                       // open interval, always
    CHECK_ABS(sum / Real(n), 0.5, 5e-3);
    CHECK_ABS(sum2 / Real(n) - sqr(sum / Real(n)), 1.0 / 12.0, 5e-3);
}

TEST(rng, normals_have_the_right_moments) {
    Xoshiro256pp rng(99);
    const long n = 400000;
    Real m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    for (long i = 0; i < n; ++i) {
        const Real z = rng.normal();
        m1 += z; m2 += z * z; m3 += z * z * z; m4 += z * z * z * z;
    }
    m1 /= Real(n); m2 /= Real(n); m3 /= Real(n); m4 /= Real(n);
    CHECK_ABS(m1, 0.0, 1e-2);
    CHECK_ABS(m2, 1.0, 1e-2);
    CHECK_ABS(m3, 0.0, 2e-2);
    CHECK_ABS(m4, 3.0, 5e-2);      // kurtosis, which Box-Muller also gets right
}

TEST(rng, jump_produces_a_disjoint_stream) {
    Xoshiro256pp a(2024), b(2024);
    b.jump();
    // The two streams must not coincide anywhere in a reasonable window.
    std::vector<std::uint64_t> from_a;
    for (int i = 0; i < 500; ++i) from_a.push_back(a.next());
    int collisions = 0;
    for (int i = 0; i < 500; ++i) {
        const std::uint64_t v = b.next();
        for (std::uint64_t x : from_a) if (x == v) ++collisions;
    }
    CHECK(collisions == 0);
}

// ---------------------------------------------------------------------------
// Sobol and the Brownian bridge
// ---------------------------------------------------------------------------

TEST(sobol, every_dimension_is_uniform) {
    // The failure this catches: misreading the polynomial encoding gives a
    // sequence whose first dimension is perfect and whose others have a mean of
    // 0.286. Checking dimension one only would have passed.
    for (int d : {1, 2, 8, 64, 256}) {
        Sobol s(d);
        std::vector<Real> point, sum(std::size_t(d), 0.0);
        const long n = 1 << 14;
        for (long i = 0; i < n; ++i) {
            s.next(point);
            for (int k = 0; k < d; ++k) sum[std::size_t(k)] += point[std::size_t(k)];
        }
        Real worst = 0.0;
        for (int k = 0; k < d; ++k) worst = std::fmax(worst, std::fabs(sum[std::size_t(k)] / Real(n) - 0.5));
        CHECK(worst < 1e-4);
    }
    CHECK_THROWS(Sobol(0));
    CHECK_THROWS(Sobol(sobol_data::kMaxDimensions + 1));
}

TEST(sobol, beats_pseudo_random_on_a_smooth_integrand) {
    // integral over [0,1]^8 of prod exp(x_i) = (e-1)^8.
    const Real exact = std::pow(std::exp(1.0) - 1.0, 8);
    const long n = 1 << 14;

    Sobol s(8);
    std::vector<Real> point;
    Real qmc = 0.0;
    for (long i = 0; i < n; ++i) {
        s.next(point);
        Real product = 1.0;
        for (Real u : point) product *= std::exp(u);
        qmc += product;
    }
    qmc /= Real(n);

    Xoshiro256pp rng(11);
    Real mc = 0.0;
    for (long i = 0; i < n; ++i) {
        Real product = 1.0;
        for (int k = 0; k < 8; ++k) product *= std::exp(rng.uniform());
        mc += product;
    }
    mc /= Real(n);

    const Real qmc_err = std::fabs(qmc / exact - 1.0);
    const Real mc_err = std::fabs(mc / exact - 1.0);
    std::printf("       %ld points, d=8: QMC error %.2e, MC error %.2e (%.0fx)\n",
                n, qmc_err, mc_err, mc_err / qmc_err);
    CHECK(qmc_err < mc_err);
}

TEST(brownian_bridge, reproduces_the_covariance_exactly) {
    // The bridge is a linear map A from normals to path values, so A A^T must
    // equal the Brownian covariance min(s,t). Recovering A by feeding unit
    // vectors makes this a deterministic identity rather than a sampling test --
    // no Monte Carlo noise to hide behind.
    for (int n : {1, 2, 4, 8, 16, 17, 33}) {
        std::vector<Real> times;
        for (int i = 1; i <= n; ++i) times.push_back(Real(i) / Real(n));
        const BrownianBridge bb(times);

        std::vector<std::vector<Real>> a(std::size_t(n), std::vector<Real>(std::size_t(n), 0.0));
        std::vector<Real> z(std::size_t(n), 0.0), path;
        for (int k = 0; k < n; ++k) {
            std::fill(z.begin(), z.end(), 0.0);
            z[std::size_t(k)] = 1.0;
            bb.build(z, path);
            for (int i = 0; i < n; ++i) a[std::size_t(i)][std::size_t(k)] = path[std::size_t(i)];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Real c = 0.0;
                for (int k = 0; k < n; ++k) c += a[std::size_t(i)][std::size_t(k)] * a[std::size_t(j)][std::size_t(k)];
                CHECK_CLOSE(c, std::fmin(times[std::size_t(i)], times[std::size_t(j)]), 1e-13);
            }
        }
    }
}

TEST(brownian_bridge, puts_the_terminal_value_in_dimension_zero) {
    // The whole reason the bridge exists: dimension zero must carry the terminal
    // value of the Brownian motion, so that Sobol's best-distributed direction
    // carries the payoff's dominant risk factor.
    const int n = 32;
    std::vector<Real> times;
    for (int i = 1; i <= n; ++i) times.push_back(Real(i) / Real(n));
    const BrownianBridge bb(times);

    std::vector<Real> z(std::size_t(n), 0.0), path;
    z[0] = 1.0;
    bb.build(z, path);
    // A unit in dimension zero moves the terminal point by sqrt(T) and nothing
    // else is needed to determine it.
    CHECK_CLOSE(path[std::size_t(n - 1)], std::sqrt(times.back()), 1e-14);
    CHECK_THROWS(BrownianBridge(std::vector<Real>{}));
}

// ---------------------------------------------------------------------------
// Lattices
// ---------------------------------------------------------------------------

TEST(binomial, converges_to_black_scholes_for_european_options) {
    const Real S = 100, K = 105, T = 0.75, r = 0.04, q = 0.015, sig = 0.28;
    const Real exact = bs_price(S, K, T, r, q, sig, OptionType::Call);
    Real previous = DBL_HUGE;
    for (int n : {101, 401, 1601}) {
        const Real lr = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Call, false, n);
        const Real err = std::fabs(lr / exact - 1.0);
        CHECK(err < previous);
        previous = err;
    }
    CHECK_CLOSE(binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Call, false, 1601),
                exact, 1e-7);
    CHECK_CLOSE(binomial_crr_averaged(S, K, T, r, q, sig, OptionType::Call, false, 2000),
                exact, 1e-5);
}

TEST(binomial, leisen_reimer_beats_crr_at_a_tenth_of_the_steps) {
    // The claim that justifies having two lattices.
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Real exact = bs_price(S, K, T, r, q, sig, OptionType::Put);
    const Real lr = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, false, 101);
    const Real crr = binomial_crr(S, K, T, r, q, sig, OptionType::Put, false, 1000);
    std::printf("       LR at 101 steps: %.2e    CRR at 1000 steps: %.2e\n",
                std::fabs(lr / exact - 1.0), std::fabs(crr / exact - 1.0));
    CHECK(std::fabs(lr / exact - 1.0) < std::fabs(crr / exact - 1.0));
}

TEST(binomial, american_put_exceeds_european_and_rejects_even_step_counts) {
    const Real S = 100, K = 110, T = 1.0, r = 0.06, q = 0.0, sig = 0.3;
    const Real american = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, true, 501);
    const Real european = bs_price(S, K, T, r, q, sig, OptionType::Put);
    CHECK(american > european);
    CHECK(american >= K - S);          // never below intrinsic

    // An American call on a non-dividend-paying underlying is never exercised
    // early, so it must equal the European.
    const Real call_a = binomial_leisen_reimer(S, K, T, r, 0.0, sig, OptionType::Call, true, 501);
    const Real call_e = bs_price(S, K, T, r, 0.0, sig, OptionType::Call);
    CHECK_CLOSE(call_a, call_e, 1e-6);

    CHECK_THROWS(binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, true, 500));
    CHECK_THROWS(binomial_crr(S, K, T, r, q, sig, OptionType::Put, true, 0));
}

// ---------------------------------------------------------------------------
// Finite differences
// ---------------------------------------------------------------------------

TEST(pde, converges_at_second_order_to_black_scholes) {
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Greeks g = bs_greeks(S, K, T, r, q, sig, OptionType::Call);

    Real previous = DBL_HUGE;
    std::printf("       grid          price error   delta error   gamma error\n");
    for (auto grid : {std::pair{200, 100}, {400, 200}, {800, 400}}) {
        PDEConfig cfg;
        cfg.space_steps = grid.first;
        cfg.time_steps = grid.second;
        const PDEResult res = pde_vanilla(S, K, T, r, q, sig, OptionType::Call,
                                          Exercise::European, cfg);
        const Real err = std::fabs(res.price / g.price - 1.0);
        std::printf("       %4d x %4d    %.2e      %.2e      %.2e\n", grid.first, grid.second,
                    err, std::fabs(res.delta - g.delta), std::fabs(res.gamma - g.gamma));
        // Halving the mesh must cut the error by roughly four.
        if (previous < DBL_HUGE) CHECK(err < previous / 3.0);
        previous = err;
        CHECK_ABS(res.delta, g.delta, 2e-5);
        CHECK_ABS(res.gamma, g.gamma, 2e-6);
    }
    CHECK(previous < 1e-5);
}

TEST(pde, crank_nicolson_without_rannacher_destroys_gamma) {
    // The demonstration. Crank-Nicolson is stable but not damping: its
    // amplification factor tends to -1 for high-frequency modes, and a payoff
    // kink is full of them. The price survives it; gamma, which is a second
    // difference of the solution, does not.
    const Real K = 100, r = 0.05, q = 0.0, sig = 0.2;
    std::printf("       T      rannacher   gamma          exact          error\n");
    for (Real T : {0.02, 0.25}) {
        const Greeks g = bs_greeks(K, K, T, r, q, sig, OptionType::Call);
        Real without = 0.0, with = 0.0;
        for (int rann : {0, 2}) {
            PDEConfig cfg;
            cfg.space_steps = 800;
            cfg.time_steps = 100;
            cfg.rannacher_steps = rann;
            const PDEResult res = pde_vanilla(K, K, T, r, q, sig, OptionType::Call,
                                              Exercise::European, cfg);
            std::printf("       %.3f  %9d   %-14.6f %-14.6f %.2e\n", T, rann, res.gamma,
                        g.gamma, std::fabs(res.gamma / g.gamma - 1.0));
            (rann == 0 ? without : with) = res.gamma;
        }
        // Without Rannacher gamma is wrong by two orders of magnitude; with two
        // implicit half-steps it is correct to five digits.
        CHECK(std::fabs(without / g.gamma - 1.0) > 10.0);
        CHECK(std::fabs(with / g.gamma - 1.0) < 1e-3);
    }
}

TEST(pde, american_put_matches_the_lattice_and_both_solvers_agree) {
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

    std::printf("       Leisen-Reimer 4001 = %.8f\n"
                "       PSOR               = %.8f  (%.3f bp)\n"
                "       Brennan-Schwartz   = %.8f  (%.3f bp)\n",
                reference, psor.price, 1e4 * std::fabs(psor.price - reference) / reference,
                bs.price, 1e4 * std::fabs(bs.price - reference) / reference);

    // Within a basis point of an independent lattice.
    CHECK(1e4 * std::fabs(psor.price - reference) / reference < 1.0);
    // And the two American solvers agree far more closely with each other than
    // either does with the lattice, since they share a discretisation.
    CHECK_ABS(psor.price, bs.price, 1e-7);
    CHECK(psor.price > bs_price(S, K, T, r, q, sig, OptionType::Put));
}

TEST(pde, rejects_degenerate_inputs) {
    CHECK_THROWS(pde_vanilla(-1.0, 100.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call));
    CHECK_THROWS(pde_vanilla(100.0, 100.0, 0.0, 0.0, 0.0, 0.2, OptionType::Call));
    CHECK_THROWS(pde_vanilla(100.0, 100.0, 1.0, 0.0, 0.0, 0.0, OptionType::Call));
    PDEConfig cfg;
    cfg.space_steps = 4;
    CHECK_THROWS(pde_vanilla(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call,
                             Exercise::European, cfg));
}

TEST(pde_heston, agrees_with_the_characteristic_function) {
    // The third opinion. The Lewis integral and the Carr-Madan transform share a
    // characteristic function; a finite-difference solution shares nothing with
    // either, so agreement between them is evidence rather than coincidence.
    const HestonParams p = index_like();
    const Real S = 100, r = 0.03, q = 0.01, T = 1.0;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);

    HestonPDEConfig cfg;
    cfg.spot_steps = 160;
    cfg.var_steps = 80;
    cfg.time_steps = 100;

    std::printf("       K     characteristic fn        ADI         relative\n");
    for (Real K : {80.0, 100.0, 125.0}) {
        const Real lewis = df * heston_call_lewis(p, F, K, T, 64, 64);
        const HestonPDEResult res = heston_pde(p, S, K, T, r, q, OptionType::Call, cfg);
        std::printf("       %5.0f  %.10f     %.10f   %.2e\n", K, lewis, res.price,
                    std::fabs(res.price / lewis - 1.0));
        CHECK(std::fabs(res.price / lewis - 1.0) < 5e-3);
    }
}

TEST(pde_heston, craig_sneyd_converges_faster_in_time_than_douglas) {
    // Douglas treats the mixed derivative explicitly and is only first-order in
    // time because of it; Craig-Sneyd corrects it and restores second order.
    // Measured on the same problem so the comparison means something.
    const HestonParams p = index_like();
    const Real S = 100, K = 125, r = 0.03, q = 0.01, T = 1.0;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);
    const Real reference = df * heston_call_lewis(p, F, K, T, 64, 64);

    std::printf("       time steps   Douglas error   Craig-Sneyd error\n");
    Real douglas_coarse = 0, douglas_fine = 0, cs_coarse = 0, cs_fine = 0;
    for (int steps : {50, 200}) {
        HestonPDEConfig cfg;
        cfg.spot_steps = 160;
        cfg.var_steps = 80;
        cfg.time_steps = steps;
        cfg.craig_sneyd = false;
        const Real d = std::fabs(heston_pde(p, S, K, T, r, q, OptionType::Call, cfg).price /
                                 reference - 1.0);
        cfg.craig_sneyd = true;
        const Real c = std::fabs(heston_pde(p, S, K, T, r, q, OptionType::Call, cfg).price /
                                 reference - 1.0);
        std::printf("       %10d   %.3e       %.3e\n", steps, d, c);
        if (steps == 50) { douglas_coarse = d; cs_coarse = c; }
        else { douglas_fine = d; cs_fine = c; }
    }
    // Quadrupling the step count should improve Douglas by roughly four and
    // Craig-Sneyd by much more.
    CHECK(cs_fine < douglas_fine);
    CHECK(cs_coarse < douglas_coarse);
    CHECK(douglas_coarse / douglas_fine > 2.0);
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

TEST(monte_carlo, andersen_qe_has_far_less_discretisation_bias_than_euler) {
    const HestonParams p = index_like();
    const Real S = 100, K = 100, T = 1.0, r = 0.03, q = 0.01;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);
    const Real exact = df * heston_call_lewis(p, F, K, T, 64, 64);

    std::printf("       steps      QE bias      Euler bias\n");
    for (int steps : {8, 32}) {
        MCConfig cfg;
        cfg.paths = 100000;
        cfg.steps = steps;
        cfg.sampling = Sampling::PseudoRandom;
        cfg.conditional = true;

        cfg.scheme = HestonScheme::AndersenQE;
        const MCResult qe = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg);
        cfg.scheme = HestonScheme::EulerFullTruncation;
        const MCResult euler = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg);

        std::printf("       %5d   %+.3e      %+.3e\n", steps, qe.price - exact,
                    euler.price - exact);
        CHECK(std::fabs(qe.price - exact) < std::fabs(euler.price - exact));
    }
}

TEST(monte_carlo, the_variance_reduction_stacks) {
    // Each technique is measured against the same baseline on the same problem,
    // and the factor quoted is in PATHS -- the ratio of variances -- since that
    // is what a user has to pay.
    const HestonParams p = index_like();
    const Real S = 100, K = 100, T = 1.0, r = 0.03, q = 0.01;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);
    const Real exact = df * heston_call_lewis(p, F, K, T, 64, 64);

    auto run = [&](bool anti, bool control, bool qmc, bool conditional) {
        MCConfig cfg;
        cfg.paths = 100000;
        cfg.steps = 16;
        cfg.antithetic = anti;
        cfg.control_variate = control;
        cfg.conditional = conditional;
        cfg.sampling = qmc ? Sampling::SobolBridge : Sampling::PseudoRandom;
        return heston_mc(p, S, K, T, r, q, OptionType::Call, cfg);
    };

    const MCResult plain = run(false, false, false, false);
    std::printf("       %-40s %10s %12s %10s\n", "technique", "std error", "paths saved",
                "error/se");
    std::printf("       %-40s %10.2e %12s %10.2f\n", "none", plain.standard_error, "1x",
                (plain.price - exact) / plain.standard_error);

    struct Case { const char* name; bool a, c, q, cond; };
    Real best_factor = 1.0;
    for (Case k : {Case{"antithetic", true, false, false, false},
                   Case{"+ control variates", true, true, false, false},
                   Case{"+ randomised Sobol with bridge", true, true, true, false},
                   Case{"+ conditional Monte Carlo", true, true, true, true}}) {
        const MCResult r2 = run(k.a, k.c, k.q, k.cond);
        const Real factor = sqr(plain.standard_error / r2.standard_error);
        std::printf("       %-40s %10.2e %11.0fx %10.2f\n", k.name, r2.standard_error, factor,
                    (r2.price - exact) / r2.standard_error);
        // Every estimator must still agree with the characteristic function.
        CHECK(std::fabs(r2.price - exact) < 5.0 * r2.standard_error);
        best_factor = std::fmax(best_factor, factor);
    }
    CHECK(best_factor > 20.0);
}

TEST(monte_carlo, the_martingale_correction_holds_the_forward) {
    // Any discretisation breaks E[S_T] = F, and the drift error looks exactly
    // like model error. The QE moment generating function makes the correction
    // exact, so this is a sharp test rather than a loose one.
    const HestonParams p = index_like();
    const Real S = 100, T = 1.0, r = 0.03, q = 0.01;

    // Two conditions are needed for this to measure anything. The step count
    // must be coarse, because a DISCRETISATION error is what is being corrected
    // and a fine grid leaves nothing to correct. And the sampling noise must be
    // suppressed -- at 400k pseudo-random paths the noise in E[S_T] is 8e-5,
    // larger than the effect at any step count worth using -- so the estimator
    // is antithetic randomised QMC.
    MCConfig cfg;
    cfg.paths = 400000;
    cfg.sampling = Sampling::SobolBridge;
    cfg.antithetic = true;
    cfg.control_variate = false;

    std::printf("       steps    corrected   uncorrected\n");
    for (int steps : {2, 4}) {
        cfg.steps = steps;
        cfg.martingale_correction = true;
        const Real with = heston_mc(p, S, 100.0, T, r, q, OptionType::Call, cfg).forward_error;
        cfg.martingale_correction = false;
        const Real without = heston_mc(p, S, 100.0, T, r, q, OptionType::Call, cfg).forward_error;
        std::printf("       %5d    %.2e     %.2e\n", steps, with, without);
        CHECK(with < 0.2 * without);      // an order of magnitude, not a whisker
        CHECK(with < 1e-4);
    }
}

TEST(monte_carlo, randomised_qmc_reports_an_honest_error_bar) {
    // A plain Sobol estimator has no valid standard error -- its points are
    // deterministic. The estimator here uses independent digital shifts and takes
    // the error bar from their spread, so the interval means something. The
    // property: over several independent runs the true value falls inside the
    // reported band about as often as it should.
    const HestonParams p = index_like();
    const Real S = 100, K = 110, T = 0.5, r = 0.03, q = 0.01;
    const Real F = S * std::exp((r - q) * T), df = std::exp(-r * T);
    const Real exact = df * heston_call_lewis(p, F, K, T, 64, 64);

    int inside = 0;
    const int trials = 12;
    for (int i = 0; i < trials; ++i) {
        MCConfig cfg;
        cfg.paths = 40000;
        cfg.steps = 16;
        cfg.seed = 1000 + std::uint64_t(i) * 7919;
        const MCResult r2 = heston_mc(p, S, K, T, r, q, OptionType::Call, cfg);
        CHECK(r2.replications > 1);
        if (std::fabs(r2.price - exact) < 3.0 * r2.standard_error) ++inside;
    }
    std::printf("       %d of %d randomised-QMC runs within 3 standard errors\n", inside, trials);
    CHECK(inside >= trials - 1);
}

TEST(monte_carlo, rejects_degenerate_inputs) {
    const HestonParams p = index_like();
    MCConfig cfg;
    cfg.paths = 1000;
    CHECK_THROWS(heston_mc(p, -1.0, 100.0, 1.0, 0.0, 0.0, OptionType::Call, cfg));
    CHECK_THROWS(heston_mc(p, 100.0, 100.0, -1.0, 0.0, 0.0, OptionType::Call, cfg));
    cfg.paths = 1;
    CHECK_THROWS(heston_mc(p, 100.0, 100.0, 1.0, 0.0, 0.0, OptionType::Call, cfg));
    // More Sobol dimensions than the table holds must be refused, not silently
    // wrapped around.
    cfg.paths = 1000;
    cfg.steps = sobol_data::kMaxDimensions;
    CHECK_THROWS(heston_mc(p, 100.0, 100.0, 1.0, 0.0, 0.0, OptionType::Call, cfg));
}

// ---------------------------------------------------------------------------
// Longstaff-Schwartz
// ---------------------------------------------------------------------------

TEST(lsmc, lower_bound_approaches_the_lattice_from_below) {
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Real american = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, true, 2001);

    std::printf("       dates    lower       se        vs lattice\n");
    for (int dates : {10, 25}) {
        LSMCConfig cfg;
        cfg.exercise_dates = dates;
        cfg.paths = 60000;
        cfg.training_paths = 30000;
        const LSMCResult res = lsmc_american(S, K, T, r, q, sig, OptionType::Put, cfg);
        std::printf("       %5d    %.5f   %.2e   %+.1f bp\n", dates, res.lower, res.lower_se,
                    1e4 * (res.lower - american) / american);
        // A Bermudan is worth less than the American, and the policy is
        // suboptimal, so the estimate must sit below -- allowing for noise.
        CHECK(res.lower < american + 4.0 * res.lower_se);
        CHECK(res.lower > 0.9 * american);
        CHECK(res.control_correlation > 0.8);
    }
}

TEST(lsmc, an_american_call_on_a_non_dividend_payer_is_never_exercised_early) {
    // A structural property with a known answer: with no dividends the early
    // exercise premium of a call is exactly zero, so Longstaff-Schwartz must
    // reproduce the European price. Any policy that exercises early shows up
    // immediately as a value below Black-Scholes.
    const Real S = 100, K = 100, T = 1.0, r = 0.05, sig = 0.3;
    LSMCConfig cfg;
    cfg.exercise_dates = 20;
    cfg.paths = 60000;
    cfg.training_paths = 30000;
    const LSMCResult res = lsmc_american(S, K, T, r, 0.0, sig, OptionType::Call, cfg);
    const Real european = bs_price(S, K, T, r, 0.0, sig, OptionType::Call);
    std::printf("       LSMC %.5f against Black-Scholes %.5f (%.1f bp)\n", res.lower, european,
                1e4 * std::fabs(res.lower - european) / european);
    CHECK_CLOSE(res.lower, european, 3e-3);
}

TEST(lsmc, the_dual_bound_brackets_the_true_value) {
    // The pair is the result. Run at a modest inner-path count so the test stays
    // fast; the bias of the upper bound and its convergence in the inner count
    // are measured in the benchmark rather than here.
    const Real S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, sig = 0.25;
    const Real american = binomial_leisen_reimer(S, K, T, r, q, sig, OptionType::Put, true, 2001);

    LSMCConfig cfg;
    cfg.exercise_dates = 12;
    cfg.paths = 60000;
    cfg.training_paths = 30000;
    cfg.run_dual = true;
    cfg.dual_outer_paths = 300;
    cfg.dual_inner_paths = 400;
    const LSMCResult res = lsmc_american(S, K, T, r, q, sig, OptionType::Put, cfg);

    std::printf("       lower %.5f  upper %.5f  gap %.1f bp   lattice %.5f\n", res.lower,
                res.upper, 1e4 * res.duality_gap / res.lower, american);
    CHECK(res.dual_run);
    CHECK(res.upper > res.lower);
    // The true value lies inside, allowing for the noise in each end.
    CHECK(american > res.lower - 4.0 * res.lower_se);
    CHECK(american < res.upper + 4.0 * res.upper_se);
}
