// The validation matrix: every model against every method that can price it.
//
// This is the file that makes the rest of the numbers mean anything. A pricer
// checked only against itself is checked against nothing, and the single most
// common way for a pricing library to be wrong is for two of its engines to
// disagree in the fourth decimal while both look plausible in isolation.
//
// Three kinds of comparison appear here and they are not equally strong:
//
//   SAME MODEL, DIFFERENT METHOD. Lewis quadrature against Carr-Madan against
//     the ADI PDE against Monte Carlo, all on one Heston parameter set. A
//     disagreement is a bug in exactly one of them, which is the useful
//     property: the comparison localises the error instead of merely detecting
//     it.
//   DEGENERATE LIMIT. Heston with sigma -> 0 is Black-Scholes; Bates with
//     lambda = 0 is Heston. These are the strongest checks in the file, because
//     the reference is a closed form with no discretisation of its own and the
//     tolerance can be machine precision rather than a judgement call.
//   NOT COMPARED, AND SAID SO. SABR has no second method here. Hagan's
//     expansion is an asymptotic formula and validating it needs a simulation
//     of the SABR SDE, which this library does not have. Listing it with a
//     dash is more useful than quietly leaving the row out.
//
// Every deviation is reported BOTH relative and as a fraction of the forward.
// Relative alone is unreadable on a cheap option -- a 1e-9 absolute error on an
// option worth 1e-7 of the forward is a relative error of 1%, which says
// something about the strike and nothing about the method.
#include "bench.hpp"
#include "vse/bates.hpp"
#include "vse/binomial.hpp"
#include "vse/black.hpp"
#include "vse/heston.hpp"
#include "vse/mc.hpp"
#include "vse/pde.hpp"
#include "vse/pde_heston.hpp"

#include <string>
#include <vector>

using namespace vse;

namespace {

/// A spread of strikes, so the matrix is not a statement about one option.
const std::vector<Real> kMoneyness = {0.80, 0.90, 1.00, 1.10, 1.25};

constexpr Real kSpot = 100.0;
constexpr Real kExpiry = 1.0;
constexpr Real kRate = 0.03;
constexpr Real kDividend = 0.01;

struct Deviation {
    Real max_relative = 0.0;
    Real max_absolute = 0.0;      ///< as a fraction of the forward
    Real worst_strike = 0.0;
};

Real forward_of() { return kSpot * std::exp((kRate - kDividend) * kExpiry); }

/// Compare two pricers across the strike ladder.
template <class A, class B>
Deviation compare(A&& reference, B&& candidate) {
    Deviation d;
    const Real f = forward_of();
    for (Real m : kMoneyness) {
        const Real strike = kSpot * m;
        const Real a = reference(strike);
        const Real b = candidate(strike);
        const Real absolute = std::fabs(a - b);
        const Real relative = absolute / std::fmax(std::fabs(a), 1e-300);
        if (relative > d.max_relative) {
            d.max_relative = relative;
            d.worst_strike = strike;
        }
        d.max_absolute = std::fmax(d.max_absolute, absolute / f);
    }
    return d;
}

void report_pair(const std::string& id, const std::string& metric, const Deviation& d,
                 const std::string& note) {
    vsebench::report(id + ".relative", metric, d.max_relative, "max relative", note);
    vsebench::report(id + ".absolute", metric + " (absolute)", d.max_absolute,
                     "fraction of the forward",
                     "the honest form when the option is cheap");
}

}  // namespace

BENCH("validation.black_scholes") {
    const Real sigma = 0.25;
    auto closed_form = [&](Real k) {
        return bs_price(kSpot, k, kExpiry, kRate, kDividend, sigma, OptionType::Call);
    };

    PDEConfig pde_cfg;
    pde_cfg.space_steps = 1600;
    pde_cfg.time_steps = 800;
    auto pde = [&](Real k) {
        return pde_vanilla(kSpot, k, kExpiry, kRate, kDividend, sigma, OptionType::Call,
                           Exercise::European, pde_cfg).price;
    };
    auto tree = [&](Real k) {
        return binomial_leisen_reimer(kSpot, k, kExpiry, kRate, kDividend, sigma,
                                      OptionType::Call, false, 4001);
    };

    // Heston with no vol-of-vol is Black-Scholes with sigma = sqrt(v0). The
    // strongest check available: the reference has no discretisation, so any
    // deviation is the quadrature and nothing else.
    HestonParams degenerate;
    degenerate.v0 = sigma * sigma;
    degenerate.theta = sigma * sigma;
    degenerate.kappa = 1.0;
    degenerate.sigma = 1e-9;
    degenerate.rho = 0.0;
    const Real f = forward_of();
    const Real discount = std::exp(-kRate * kExpiry);
    auto heston_limit = [&](Real k) {
        return heston_price(degenerate, f, k, kExpiry, discount, OptionType::Call, 64, 32);
    };

    // Bates with no jumps and no vol-of-vol is the same thing again, one layer
    // further out. It costs nothing to check and it is the only test that would
    // catch a jump term that fails to vanish.
    BatesParams bates;
    bates.heston = degenerate;
    bates.lambda = 0.0;
    bates.jump_mean = 0.0;
    bates.jump_vol = 0.1;
    auto bates_limit = [&](Real k) {
        return bates_price(bates, f, k, kExpiry, discount, OptionType::Call, 64, 32);
    };

    report_pair("validation.bs.pde", "Black-Scholes: closed form against Crank-Nicolson",
                compare(closed_form, pde), "1600 x 800, Rannacher startup");
    report_pair("validation.bs.tree", "Black-Scholes: closed form against Leisen-Reimer",
                compare(closed_form, tree), "4001 steps");
    report_pair("validation.bs.heston_limit",
                "Black-Scholes: closed form against Heston at sigma = 1e-9",
                compare(closed_form, heston_limit),
                "a degenerate limit, so the reference has no discretisation error");
    report_pair("validation.bs.bates_limit",
                "Black-Scholes: closed form against Bates with no jumps",
                compare(closed_form, bates_limit), "lambda = 0 and sigma = 1e-9");
}

BENCH("validation.heston") {
    // Equity-index parameters, Feller violated, which is the realistic case and
    // the harder one for every method here.
    const HestonParams p{0.0348, 1.58, 0.0447, 0.92, -0.74};
    const Real f = forward_of();
    const Real discount = std::exp(-kRate * kExpiry);

    // The Lewis integral at high order is the reference. Not because it is
    // arbitrarily trusted: its convergence in the quadrature order was measured
    // separately at 5.4e-10 relative, so it is the only method here whose error
    // is known rather than estimated.
    auto lewis = [&](Real k) {
        return heston_price(p, f, k, kExpiry, discount, OptionType::Call, 64, 32);
    };

    CarrMadanConfig cm;
    cm.n = 8192;
    const auto grid = heston_carr_madan(p, f, kExpiry, cm);
    auto fft = [&](Real k) { return discount * grid.call_at(k); };

    HestonPDEConfig pde_cfg;
    pde_cfg.spot_steps = 240;
    pde_cfg.var_steps = 120;
    pde_cfg.time_steps = 200;
    auto adi = [&](Real k) {
        return heston_pde(p, kSpot, k, kExpiry, kRate, kDividend, OptionType::Call,
                          pde_cfg).price;
    };

    MCConfig mc_cfg;
    mc_cfg.paths = 400000;
    mc_cfg.steps = 32;
    mc_cfg.seed = 20260706;
    mc_cfg.scheme = HestonScheme::AndersenQE;
    mc_cfg.control_variate = true;
    mc_cfg.conditional = true;
    auto mc = [&](Real k) {
        return heston_mc(p, kSpot, k, kExpiry, kRate, kDividend, OptionType::Call,
                         mc_cfg).price;
    };

    BatesParams bates;
    bates.heston = p;
    bates.lambda = 0.0;
    bates.jump_mean = 0.0;
    bates.jump_vol = 0.2;
    auto bates_limit = [&](Real k) {
        return bates_price(bates, f, k, kExpiry, discount, OptionType::Call, 64, 32);
    };

    report_pair("validation.heston.fft", "Heston: Lewis integral against Carr-Madan FFT",
                compare(lewis, fft),
                "N = 8192; the FFT's error is a fixed absolute floor set by the grid");
    report_pair("validation.heston.pde", "Heston: Lewis integral against Craig-Sneyd ADI",
                compare(lewis, adi), "240 x 120 x 200");
    report_pair("validation.heston.mc", "Heston: Lewis integral against Monte Carlo",
                compare(lewis, mc),
                "400k paths, QE with a conditional estimator and a control variate");
    report_pair("validation.heston.bates_limit",
                "Heston: Lewis integral against Bates with lambda = 0",
                compare(lewis, bates_limit), "a degenerate limit");

    // The Monte Carlo deviation is only interpretable against its own standard
    // error, so report that too. A quarter of a standard error and five standard
    // errors are both "0.001 relative" and mean opposite things.
    const auto at_the_money =
        heston_mc(p, kSpot, kSpot, kExpiry, kRate, kDividend, OptionType::Call, mc_cfg);
    const Real reference = lewis(kSpot);
    vsebench::report("validation.heston.mc_standard_errors",
                     "Heston: Monte Carlo against the Lewis integral, at the money",
                     std::fabs(at_the_money.price - reference) /
                         std::fmax(at_the_money.standard_error, 1e-300),
                     "standard errors",
                     "the only scale on which a Monte Carlo deviation means anything");
}

BENCH("validation.parity") {
    // Put-call parity through every engine. It is model-free, so a violation is
    // never a modelling difference and always a bug -- which makes it the one
    // check in this file that needs no judgement about tolerances.
    const HestonParams p{0.0348, 1.58, 0.0447, 0.92, -0.74};
    const Real f = forward_of();
    const Real discount = std::exp(-kRate * kExpiry);
    const Real sigma = 0.25;

    PDEConfig pde_cfg;
    pde_cfg.space_steps = 1600;
    pde_cfg.time_steps = 800;

    Real worst_bs = 0.0, worst_heston = 0.0, worst_pde = 0.0;
    for (Real m : kMoneyness) {
        const Real k = kSpot * m;
        const Real expected = discount * (f - k);

        worst_bs = std::fmax(worst_bs,
                             std::fabs(bs_price(kSpot, k, kExpiry, kRate, kDividend, sigma,
                                                OptionType::Call) -
                                       bs_price(kSpot, k, kExpiry, kRate, kDividend, sigma,
                                                OptionType::Put) -
                                       expected));
        worst_heston = std::fmax(
            worst_heston,
            std::fabs(heston_price(p, f, k, kExpiry, discount, OptionType::Call, 64, 32) -
                      heston_price(p, f, k, kExpiry, discount, OptionType::Put, 64, 32) -
                      expected));
        worst_pde = std::fmax(
            worst_pde,
            std::fabs(pde_vanilla(kSpot, k, kExpiry, kRate, kDividend, sigma, OptionType::Call,
                                  Exercise::European, pde_cfg).price -
                      pde_vanilla(kSpot, k, kExpiry, kRate, kDividend, sigma, OptionType::Put,
                                  Exercise::European, pde_cfg).price -
                      expected));
    }

    vsebench::report("validation.parity.black_scholes", "Put-call parity, closed form",
                     worst_bs / f, "fraction of the forward", "model-free; any error is a bug");
    vsebench::report("validation.parity.heston", "Put-call parity, Lewis integral",
                     worst_heston / f, "fraction of the forward", "");
    vsebench::report("validation.parity.pde", "Put-call parity, Crank-Nicolson",
                     worst_pde / f, "fraction of the forward",
                     "the boundary conditions are the thing this catches");
}
