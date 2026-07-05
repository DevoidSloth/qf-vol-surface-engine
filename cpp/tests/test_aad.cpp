// Adjoint algorithmic differentiation.
#include "harness.hpp"
#include "vse/aad.hpp"
#include "vse/black.hpp"
#include "vse/heston.hpp"
#include "vse/mc_aad.hpp"
#include <chrono>
#include <string>

#include <array>
#include <vector>

using namespace vse;
using namespace vsetest;

TEST(aad, reproduces_the_closed_form_black_scholes_greeks) {
    // Six sensitivities from one backward sweep, against the analytic formulas.
    // The tolerance is machine epsilon because these are the same derivatives,
    // not an approximation to them.
    const Real S = 100, K = 105, T = 0.7, r = 0.03, q = 0.012, sig = 0.28;
    for (auto type : {OptionType::Call, OptionType::Put}) {
        Tape& tape = Tape::active();
        tape.clear();
        const ADouble a_s(S), a_k(K), a_t(T), a_r(r), a_q(q), a_sig(sig);
        const ADouble value = bs_price_generic<ADouble>(a_s, a_k, a_t, a_r, a_q, a_sig, type);
        tape.backpropagate(value.index());

        const Greeks g = bs_greeks(S, K, T, r, q, sig, type);
        CHECK_CLOSE(value.value(), g.price, 1e-14);
        CHECK_CLOSE(a_s.adjoint(), g.delta, 1e-13);
        CHECK_CLOSE(a_sig.adjoint(), g.vega, 1e-13);
        CHECK_CLOSE(a_r.adjoint(), g.rho, 1e-13);
        CHECK_CLOSE(a_k.adjoint(), g.dual_delta, 1e-13);
        // theta is d/dt and t runs opposite to T.
        CHECK_CLOSE(-a_t.adjoint(), g.theta, 1e-13);
        // The dividend sensitivity has no standard name but is -S e^{-qT} T N(d1)
        // for a call, which is -T times the spot delta.
        CHECK_CLOSE(a_q.adjoint(), -T * S * g.delta, 1e-13);
    }
    std::printf("       six sensitivities from one sweep, %zu tape nodes\n",
                Tape::active().size());
}

TEST(aad, arithmetic_matches_hand_differentiation) {
    // f(x,y) = exp(x y) / sqrt(x + y) - log(x) * y, differentiated by hand.
    Tape& tape = Tape::active();
    tape.clear();
    const Real x = 1.3, y = 0.7;
    const ADouble ax(x), ay(y);
    const ADouble f = exp(ax * ay) / sqrt(ax + ay) - log(ax) * ay;
    tape.backpropagate(f.index());

    const Real e = std::exp(x * y), root = std::sqrt(x + y);
    CHECK_CLOSE(f.value(), e / root - std::log(x) * y, 1e-14);
    CHECK_CLOSE(ax.adjoint(), e * y / root - 0.5 * e / std::pow(x + y, 1.5) - y / x, 1e-13);
    CHECK_CLOSE(ay.adjoint(), e * x / root - 0.5 * e / std::pow(x + y, 1.5) - std::log(x), 1e-13);
}

TEST(aad, mixed_scalar_operations_record_one_node_not_two) {
    // Promoting the scalar and reusing the binary operator is correct and
    // doubles the tape. In a pricer full of multiplications by dt and by 0.5,
    // that is a third of the work.
    Tape& tape = Tape::active();
    tape.clear();
    const ADouble a(2.0);
    const std::size_t before = tape.size();
    const ADouble b = a * 3.0 + 1.0 - 0.5;
    const std::size_t nodes = tape.size() - before;
    std::printf("       three scalar operations recorded %zu nodes\n", nodes);
    CHECK(nodes == 3);

    tape.backpropagate(b.index());
    CHECK_CLOSE(b.value(), 2.0 * 3.0 + 1.0 - 0.5, 1e-15);
    CHECK_CLOSE(a.adjoint(), 3.0, 1e-15);
}

TEST(aad, the_backward_sweep_accumulates_repeated_uses) {
    // A variable used several times must collect the sum of the paths through
    // it. Getting this wrong gives a derivative that is right for simple
    // expressions and wrong for real ones.
    Tape& tape = Tape::active();
    tape.clear();
    const ADouble x(3.0);
    const ADouble y = x * x + x * x * x;      // d/dx = 2x + 3x^2 = 33 at x = 3
    tape.backpropagate(y.index());
    CHECK_CLOSE(y.value(), 9.0 + 27.0, 1e-15);
    CHECK_CLOSE(x.adjoint(), 2.0 * 3.0 + 3.0 * 9.0, 1e-13);
}

TEST(aad, values_carry_their_own_tape) {
    // The pointer in each value is what keeps thread-local lookups out of the
    // hot path; the property that makes it safe is that two tapes do not
    // interfere.
    Tape first, second;
    const ADouble a = ADouble::variable(2.0, first);
    const ADouble b = ADouble::variable(5.0, second);
    const ADouble fa = a * a;
    const ADouble fb = b * b * b;
    CHECK(fa.tape() == &first);
    CHECK(fb.tape() == &second);

    first.backpropagate(fa.index());
    second.backpropagate(fb.index());
    CHECK_CLOSE(a.adjoint(), 4.0, 1e-15);      // d(x^2)/dx at 2
    CHECK_CLOSE(b.adjoint(), 75.0, 1e-15);     // d(x^3)/dx at 5
}

TEST(aad, monte_carlo_greeks_match_bump_and_revalue) {
    // Pathwise adjoint against a common-random-number central difference, on the
    // Euler scheme (see mc_aad.hpp for why not QE).
    //
    // The comparison is made at parameters that satisfy Feller. That is not
    // cherry-picking: the next test measures what happens when they do not, and
    // the reason is a property of the estimator that is worth knowing rather
    // than hiding.
    const HestonParams p{0.04, 3.0, 0.04, 0.36, -0.74};
    CHECK(p.satisfies_feller());

    MCConfig cfg;
    cfg.paths = 40000;
    cfg.steps = 16;
    cfg.seed = 4242;
    cfg.scheme = HestonScheme::EulerFullTruncation;

    const MCGreeksResult aad =
        heston_mc_greeks_aad(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
    const MCGreeksResult bump =
        heston_mc_greeks_bump(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);

    CHECK_CLOSE(aad.price, bump.price, 1e-12);
    const auto& names = heston_risk_factor_names();
    std::printf("       %-10s %14s %14s %12s\n", "factor", "adjoint", "bumped", "difference");
    for (int k = 0; k < kHestonRiskFactors; ++k) {
        const Real a = aad.gradient[std::size_t(k)];
        const Real b = bump.gradient[std::size_t(k)];
        const Real se = aad.gradient_se[std::size_t(k)];
        std::printf("       %-10s %14.5f %14.5f %12.1e\n", names[std::size_t(k)], a, b,
                    std::fabs(a - b));
        // Agreement within the Monte Carlo error of the adjoint estimate, with
        // a floor for the factors whose derivative is essentially zero.
        CHECK(std::fabs(a - b) < std::fmax(4.0 * se, 1e-6 * std::fabs(b) + 1e-8));
    }
}

TEST(aad, pathwise_variance_greeks_degrade_when_feller_is_violated) {
    // A real limitation, measured rather than asserted.
    //
    // The pathwise derivative of a Heston path with respect to a variance
    // parameter passes through d(sqrt(v))/dv = 1/(2 sqrt(v)), which is unbounded
    // as v approaches zero. When 2 kappa theta < sigma^2 the variance process
    // reaches zero, and the estimator acquires enormous variance -- it stays
    // unbiased and becomes useless, which is a harder failure to notice than
    // being wrong.
    //
    // Spot-like factors are unaffected, because their pathwise derivatives never
    // touch the square root.
    MCConfig cfg;
    cfg.paths = 40000;
    cfg.steps = 16;
    cfg.seed = 4242;
    cfg.scheme = HestonScheme::EulerFullTruncation;

    struct Case { const char* label; HestonParams p; };
    std::printf("       %-22s %8s %14s %14s\n", "parameters", "Feller", "se/|dV/dv0|",
                "se/|delta|");
    Real violated_ratio = 0.0, satisfied_ratio = 1.0;
    for (Case c : {Case{"violated", HestonParams{0.0348, 1.58, 0.0447, 0.92, -0.74}},
                   Case{"satisfied", HestonParams{0.04, 3.0, 0.04, 0.36, -0.74}}}) {
        const MCGreeksResult r =
            heston_mc_greeks_aad(c.p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
        const Real v0_ratio = r.gradient_se[1] / std::fmax(std::fabs(r.gradient[1]), 1e-12);
        const Real delta_ratio = r.gradient_se[0] / std::fmax(std::fabs(r.gradient[0]), 1e-12);
        std::printf("       %-22s %8.2f %14.3f %14.4f\n", c.label, c.p.feller_ratio(),
                    v0_ratio, delta_ratio);
        (std::string(c.label) == "violated" ? violated_ratio : satisfied_ratio) = v0_ratio;
        // Delta is well behaved either way.
        CHECK(delta_ratio < 0.02);
    }
    // The variance Greek is an order of magnitude noisier when Feller fails.
    CHECK(violated_ratio > 10.0 * satisfied_ratio);
}

TEST(aad, costs_a_small_multiple_of_a_price_regardless_of_factor_count) {
    // The claim reverse mode exists to support. Reported as a ratio to an
    // undifferentiated price on the identical code path, so the comparison is
    // not against a strawman.
    const HestonParams p{0.04, 3.0, 0.04, 0.36, -0.74};
    MCConfig cfg;
    cfg.paths = 20000;
    cfg.steps = 16;
    cfg.seed = 4242;
    cfg.scheme = HestonScheme::EulerFullTruncation;

    auto best_of = [](auto&& f) {
        Real best = DBL_HUGE;
        for (int i = 0; i < 3; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            f();
            const Real ms = std::chrono::duration<Real, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
            best = std::fmin(best, ms);
        }
        return best;
    };

    const Real price_ms = best_of([&] {
        (void)heston_mc_price_only(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
    });
    const Real aad_ms = best_of([&] {
        (void)heston_mc_greeks_aad(p, 100.0, 100.0, 1.0, 0.03, 0.01, OptionType::Call, cfg);
    });

    const Real overhead = aad_ms / price_ms;
    const Real speedup = Real(2 * kHestonRiskFactors + 1) / overhead;
    std::printf("       price %.0f ms, adjoint %.0f ms = %.2fx a price\n", price_ms, aad_ms,
                overhead);
    std::printf("       against %d repricings for a bump: %.1fx\n", 2 * kHestonRiskFactors + 1,
                speedup);
    // Generous, because this runs on whatever machine the tests run on.
    CHECK(overhead < 5.0);
    CHECK(speedup > 4.0);
}
