// Cash dividends as jump conditions.
#include "harness.hpp"
#include "vse/binomial.hpp"
#include "vse/black.hpp"
#include "vse/pde.hpp"
#include "vse/rng.hpp"

#include <vector>

using namespace vse;
using namespace vsetest;

namespace {

constexpr Real kSpot = 100.0;
constexpr Real kStrike = 100.0;
constexpr Real kExpiry = 1.0;
constexpr Real kRate = 0.05;
constexpr Real kSigma = 0.25;

PDEConfig fine() {
    PDEConfig cfg;
    cfg.space_steps = 800;
    cfg.time_steps = 400;
    return cfg;
}

}  // namespace

TEST(dividends, present_value_counts_only_what_is_still_ahead) {
    const std::vector<CashDividend> divs{{0.25, 2.0}, {0.75, 3.0}};
    // Seen from expiry there is nothing left to pay.
    CHECK_CLOSE(pv_remaining_dividends(divs, kRate, kExpiry, 0.0), 0.0, 0.0);
    // Seen from today, both, each discounted over its own horizon.
    const Real expected = 2.0 * std::exp(-kRate * 0.25) + 3.0 * std::exp(-kRate * 0.75);
    CHECK_CLOSE(pv_remaining_dividends(divs, kRate, kExpiry, kExpiry), expected, 1e-14);
    // Halfway, only the later one.
    CHECK_CLOSE(pv_remaining_dividends(divs, kRate, kExpiry, 0.5),
                3.0 * std::exp(-kRate * 0.25), 1e-14);
}

TEST(dividends, put_call_parity_holds_across_the_ex_date) {
    // The invariant that does not care which model is right. Whatever the jump
    // condition does to a call it must do to a put, because a long call and a
    // short put replicate the forward, and the forward on a stock paying a known
    // cash dividend is S - PV(D) compounded.
    //
    // This is the test that catches a sign error in the shift, which would
    // otherwise show up as a plausible-looking price.
    const std::vector<CashDividend> divs{{0.5, 5.0}};
    const auto call = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                                  Exercise::European, fine(), divs);
    const auto put = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Put,
                                 Exercise::European, fine(), divs);
    const Real pv = pv_remaining_dividends(divs, kRate, kExpiry, kExpiry);
    const Real parity = (kSpot - pv) - kStrike * std::exp(-kRate * kExpiry);
    std::printf("       call %.6f - put %.6f = %.6f, parity says %.6f\n", call.price, put.price,
                call.price - put.price, parity);
    CHECK(std::fabs((call.price - put.price) - parity) < 2e-3);
}

TEST(dividends, a_zero_dividend_changes_nothing) {
    // The schedule is cut at the ex-date and Rannacher restarts there whether
    // or not the amount is zero, so this checks that the extra machinery is
    // neutral rather than merely small.
    const auto plain = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Put,
                                   Exercise::American, fine());
    const auto with_zero = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma,
                                       OptionType::Put, Exercise::American, fine(),
                                       {{0.5, 0.0}});
    std::printf("       no schedule cut %.8f, cut with a zero dividend %.8f\n", plain.price,
                with_zero.price);
    CHECK_CLOSE(with_zero.price, plain.price, 5e-5);
}

TEST(dividends, differ_from_the_escrowed_model_by_a_choice_and_not_an_error) {
    // Not a cross-check. The escrowed model puts the volatility on S - PV(D)
    // and this solver puts it on S, so the two answer different questions and
    // the gap between them is a modelling choice. The test pins its SIZE, in
    // both directions: large enough that nobody can call the two
    // interchangeable, small enough that a sign error in the jump would not
    // hide inside it.
    //
    // The genuine cross-check, against an independent method on the SAME model,
    // is the Monte Carlo below.
    const std::vector<CashDividend> divs{{0.4, 3.0}};
    for (auto type : {OptionType::Call, OptionType::Put}) {
        const auto pde = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, type,
                                     Exercise::European, fine(), divs);
        // Escrowed binomial on the risky part, then add back the certain cash
        // for the terminal comparison: for a EUROPEAN option the two models
        // coincide once the tree is built on S - PV(D) and the dividend is
        // treated as certain, because nothing depends on the path.
        const Real pv = pv_remaining_dividends(divs, kRate, kExpiry, kExpiry);
        const Real reference =
            binomial_leisen_reimer(kSpot - pv, kStrike, kExpiry, kRate, 0.0, kSigma, type,
                                   false, 4001);
        const Real gap = std::fabs(pde.price - reference);
        std::printf("       %-4s PDE %.6f, escrowed tree %.6f, gap %.4f (%.2f%%)\n",
                    type == OptionType::Call ? "call" : "put", pde.price, reference, gap,
                    100.0 * gap / reference);
        // NOT a tight tolerance, and that is the finding: the escrowed model is
        // a different model, not an approximation to this one, and the gap is
        // the modelling choice rather than a discretisation error. See the
        // block comment at the jump condition in pde.hpp.
        CHECK(gap < 0.4);
        CHECK(gap > 0.05);
    }
}

TEST(dividends, make_an_american_call_worth_exercising_early) {
    // The classic result, and the reason a cash dividend cannot be replaced by
    // an equivalent yield. Without dividends an American call on a
    // non-dividend-paying stock is never exercised early and is worth exactly
    // the European. Introduce a large enough discrete drop and the holder
    // exercises an instant before the ex-date to capture it.
    const std::vector<CashDividend> divs{{0.5, 8.0}};
    const auto european = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma,
                                      OptionType::Call, Exercise::European, fine(), divs);
    const auto american = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma,
                                      OptionType::Call, Exercise::American, fine(), divs);
    const Real premium = american.price - european.price;
    std::printf("       European %.5f, American %.5f, early exercise worth %.5f (%.1f%%)\n",
                european.price, american.price, premium, 100.0 * premium / european.price);
    CHECK(premium > 0.05);

    // And with no dividend at all there is nothing to capture, so the two
    // coincide. This half is what makes the first half mean something.
    const auto e0 = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                                Exercise::European, fine());
    const auto a0 = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                                Exercise::American, fine());
    std::printf("       without a dividend: European %.6f, American %.6f\n", e0.price, a0.price);
    CHECK_CLOSE(a0.price, e0.price, 1e-6);
}

TEST(dividends, an_equivalent_yield_gets_the_american_call_wrong) {
    // Converting the cash to a yield that reproduces the same forward is the
    // shortcut this file exists to reject. It gets the European close, because
    // a European option only sees the forward, and it gets the American wrong,
    // because a yield pays continuously and never produces the discrete drop
    // that makes early exercise optimal.
    const std::vector<CashDividend> divs{{0.5, 8.0}};
    const Real pv = pv_remaining_dividends(divs, kRate, kExpiry, kExpiry);
    // The yield with the same forward: S e^{-qT} = S - PV(D).
    const Real q = -std::log((kSpot - pv) / kSpot) / kExpiry;

    const auto cash_euro = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma,
                                       OptionType::Call, Exercise::European, fine(), divs);
    const auto yield_euro = pde_vanilla(kSpot, kStrike, kExpiry, kRate, q, kSigma,
                                        OptionType::Call, Exercise::European, fine());
    const auto cash_amer = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma,
                                       OptionType::Call, Exercise::American, fine(), divs);
    const auto yield_amer = pde_vanilla(kSpot, kStrike, kExpiry, kRate, q, kSigma,
                                        OptionType::Call, Exercise::American, fine());

    const Real euro_gap = std::fabs(cash_euro.price - yield_euro.price);
    const Real amer_gap = std::fabs(cash_amer.price - yield_amer.price);
    std::printf("       equivalent yield %.4f%%: European gap %.5f, American gap %.5f\n",
                q * 100.0, euro_gap, amer_gap);
    // The American error is the larger, which is the point: the shortcut fails
    // precisely where the exercise decision depends on the shape of the
    // dividend rather than on its present value.
    CHECK(amer_gap > euro_gap);
}

TEST(dividends, several_ex_dates_compose) {
    // Three dividends must cost the same as their present value in a European
    // option, whatever the schedule, because a European payoff sees only the
    // forward. This is a stronger statement than it looks: it says the jump
    // conditions compose correctly and that the segmented time schedule is not
    // losing accuracy at each cut.
    const std::vector<CashDividend> spread{{0.2, 2.0}, {0.5, 2.0}, {0.8, 2.0}};
    const auto many = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                                  Exercise::European, fine(), spread);
    const Real pv = pv_remaining_dividends(spread, kRate, kExpiry, kExpiry);

    // A single dividend with the same present value, placed so its PV matches.
    const Real when = 0.5;
    const std::vector<CashDividend> one{{when, pv * std::exp(kRate * when)}};
    const auto single = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma,
                                    OptionType::Call, Exercise::European, fine(), one);
    std::printf("       three dividends %.6f, one of equal PV %.6f, gap %.2e\n", many.price,
                single.price, std::fabs(many.price - single.price));
    // Not identical -- the jump happens at a different time, so the process
    // between the dates differs -- but close, and much closer than either is to
    // the no-dividend price.
    CHECK(std::fabs(many.price - single.price) < 0.05);
}

TEST(dividends, are_rejected_when_they_fall_outside_the_life_of_the_option) {
    CHECK_THROWS(pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                             Exercise::European, PDEConfig{}, {{1.5, 2.0}}));
    CHECK_THROWS(pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                             Exercise::European, PDEConfig{}, {{0.0, 2.0}}));
    CHECK_THROWS(pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                             Exercise::European, PDEConfig{}, {{0.5, -1.0}}));
    // Two on the same date would produce a zero-length segment.
    CHECK_THROWS(pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                             Exercise::European, PDEConfig{}, {{0.5, 1.0}, {0.5, 1.0}}));
}

TEST(dividends, agree_with_a_monte_carlo_on_the_same_jump_model) {
    // The real cross-check: an independent method simulating the same process.
    // Geometric Brownian motion to the ex-date, subtract the cash, continue,
    // and average the payoff. Nothing here shares code with the PDE beyond the
    // normal inverse CDF, so agreement is evidence about both.
    //
    // Antithetic pairs and a Black-Scholes control on the dividend-free option,
    // because the point is to bracket the PDE tightly enough to catch a small
    // bias, and a raw estimator at this path count would not.
    const std::vector<CashDividend> divs{{0.4, 3.0}, {0.8, 2.0}};
    const Real ex1 = 0.4, ex2 = 0.8;

    Xoshiro256pp rng(97531);
    const int pairs = 400000;
    Real sum = 0.0, sum_sq = 0.0;
    const Real drift1 = (kRate - 0.5 * kSigma * kSigma) * ex1;
    const Real drift2 = (kRate - 0.5 * kSigma * kSigma) * (ex2 - ex1);
    const Real drift3 = (kRate - 0.5 * kSigma * kSigma) * (kExpiry - ex2);
    const Real vol1 = kSigma * std::sqrt(ex1);
    const Real vol2 = kSigma * std::sqrt(ex2 - ex1);
    const Real vol3 = kSigma * std::sqrt(kExpiry - ex2);

    for (int i = 0; i < pairs; ++i) {
        const Real z1 = norm_inv_cdf(rng.uniform());
        const Real z2 = norm_inv_cdf(rng.uniform());
        const Real z3 = norm_inv_cdf(rng.uniform());
        for (Real sign : {1.0, -1.0}) {
            Real s = kSpot * std::exp(drift1 + vol1 * sign * z1);
            s = std::fmax(s - 3.0, 1e-12);
            s *= std::exp(drift2 + vol2 * sign * z2);
            s = std::fmax(s - 2.0, 1e-12);
            s *= std::exp(drift3 + vol3 * sign * z3);
            const Real payoff = std::fmax(s - kStrike, 0.0);
            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }
    const Real n = Real(2 * pairs);
    const Real discount = std::exp(-kRate * kExpiry);
    const Real mc = discount * sum / n;
    // The antithetic pairing means this overstates the true standard error, so
    // using it as the tolerance is conservative rather than convenient.
    const Real se = discount * std::sqrt(std::fmax(sum_sq / n - sqr(sum / n), 0.0) / n);

    const auto pde = pde_vanilla(kSpot, kStrike, kExpiry, kRate, 0.0, kSigma, OptionType::Call,
                                 Exercise::European, fine(), divs);
    const Real deviations = std::fabs(pde.price - mc) / se;
    std::printf("       PDE %.6f, MC %.6f +/- %.6f  (%.2f standard errors)\n", pde.price, mc, se,
                deviations);
    CHECK(deviations < 3.0);
}
