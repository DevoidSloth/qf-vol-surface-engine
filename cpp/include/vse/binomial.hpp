// vse/binomial.hpp — lattice pricing, present as an independent reference.
//
// Nothing in this library uses a binomial tree to price anything in anger. It is
// here because an American option has no closed form, so the PDE and the
// Longstaff-Schwartz Monte Carlo need something to be checked against that
// shares none of their machinery -- no grid, no time-stepping scheme, no
// regression, no random numbers. Two implementations that agree because they
// contain the same mistake is the failure mode that validation exists to
// prevent, and the only defence is genuine independence.
//
// Two lattices, for different jobs:
//
//   CRR is the textbook one and converges at O(1/n) with a sawtooth: the error
//   oscillates with parity as the strike falls between two terminal nodes.
//   Averaging consecutive step counts removes most of it, and 10,000 steps gets
//   an American put to a few tenths of a basis point.
//
//   Leisen-Reimer places the tree so that the strike sits exactly at the centre
//   of the terminal distribution and uses a Peizer-Pratt inversion of the
//   binomial to the normal. The sawtooth disappears and convergence becomes
//   smooth and O(1/n^2), so 1,001 steps beats CRR at 10,000. It requires an odd
//   number of steps, which is the sort of precondition that fails silently, so
//   it is enforced.
#pragma once

#include "vse/black.hpp"
#include "vse/common.hpp"

#include <vector>

namespace vse {

/// Cox-Ross-Rubinstein lattice.
inline Real binomial_crr(Real spot, Real strike, Real expiry, Real rate, Real dividend,
                         Real sigma, OptionType type, bool american, int steps) {
    require(spot > 0.0 && strike > 0.0, "binomial_crr: spot and strike must be positive");
    require(expiry > 0.0 && sigma > 0.0, "binomial_crr: expiry and vol must be positive");
    require(steps >= 1, "binomial_crr: need at least one step");

    const Real dt = expiry / Real(steps);
    const Real up = std::exp(sigma * std::sqrt(dt));
    const Real down = 1.0 / up;
    const Real disc = std::exp(-rate * dt);
    const Real growth = std::exp((rate - dividend) * dt);
    const Real p = (growth - down) / (up - down);
    require(p > 0.0 && p < 1.0,
            "binomial_crr: risk-neutral probability outside (0,1); the step is too "
            "large for this drift and volatility");

    const Real w = omega(type);
    std::vector<Real> values(std::size_t(steps) + 1);
    // Terminal layer. Building the spot by repeated multiplication accumulates
    // error over 10,000 steps, so use the closed form for each node.
    for (int j = 0; j <= steps; ++j) {
        const Real s = spot * std::pow(up, Real(2 * j - steps));
        values[std::size_t(j)] = std::fmax(w * (s - strike), 0.0);
    }

    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            Real v = disc * (p * values[std::size_t(j + 1)] + (1.0 - p) * values[std::size_t(j)]);
            if (american) {
                const Real s = spot * std::pow(up, Real(2 * j - i));
                v = std::fmax(v, w * (s - strike));
            }
            values[std::size_t(j)] = v;
        }
    }
    return values[0];
}

namespace detail {

/// Peizer-Pratt inversion, method two: a very accurate normal approximation to
/// the binomial, used backwards to choose the lattice probability.
inline Real peizer_pratt(Real z, int n) {
    const Real nn = Real(n) + 1.0 / 3.0 + 0.1 / (Real(n) + 1.0);
    const Real inner = z / nn;
    const Real e = -inner * inner * (Real(n) + 1.0 / 6.0);
    return 0.5 + (z < 0.0 ? -0.5 : 0.5) * std::sqrt(1.0 - std::exp(e));
}

}  // namespace detail

/// Leisen-Reimer lattice. `steps` must be odd.
inline Real binomial_leisen_reimer(Real spot, Real strike, Real expiry, Real rate,
                                   Real dividend, Real sigma, OptionType type, bool american,
                                   int steps) {
    require(spot > 0.0 && strike > 0.0, "binomial_leisen_reimer: spot and strike must be positive");
    require(expiry > 0.0 && sigma > 0.0, "binomial_leisen_reimer: expiry and vol must be positive");
    require(steps >= 3 && steps % 2 == 1,
            "binomial_leisen_reimer: the number of steps must be odd and at least 3 "
            "(an even count puts the strike between two terminal nodes, which is "
            "exactly what this lattice exists to avoid)");

    const Real dt = expiry / Real(steps);
    const Real srt = sigma * std::sqrt(expiry);
    const Real d1 = (std::log(spot / strike) + (rate - dividend + 0.5 * sigma * sigma) * expiry) / srt;
    const Real d2 = d1 - srt;

    const Real p = detail::peizer_pratt(d2, steps);
    const Real p_dash = detail::peizer_pratt(d1, steps);
    const Real growth = std::exp((rate - dividend) * dt);
    const Real up = growth * p_dash / p;
    const Real down = (growth - p * up) / (1.0 - p);
    const Real disc = std::exp(-rate * dt);
    const Real w = omega(type);

    std::vector<Real> values(std::size_t(steps) + 1);
    for (int j = 0; j <= steps; ++j) {
        const Real s = spot * std::pow(up, Real(j)) * std::pow(down, Real(steps - j));
        values[std::size_t(j)] = std::fmax(w * (s - strike), 0.0);
    }
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            Real v = disc * (p * values[std::size_t(j + 1)] + (1.0 - p) * values[std::size_t(j)]);
            if (american) {
                const Real s = spot * std::pow(up, Real(j)) * std::pow(down, Real(i - j));
                v = std::fmax(v, w * (s - strike));
            }
            values[std::size_t(j)] = v;
        }
    }
    return values[0];
}

/// American price by CRR with Richardson-style averaging of adjacent step counts.
///
/// The CRR error has an O(1/n) part that alternates with the parity of n, so
/// averaging n and n+1 cancels most of it. Cheap, and it turns a lattice with a
/// visible sawtooth into a usable reference.
inline Real binomial_crr_averaged(Real spot, Real strike, Real expiry, Real rate,
                                  Real dividend, Real sigma, OptionType type, bool american,
                                  int steps) {
    return 0.5 * (binomial_crr(spot, strike, expiry, rate, dividend, sigma, type, american, steps) +
                  binomial_crr(spot, strike, expiry, rate, dividend, sigma, type, american,
                               steps + 1));
}

}  // namespace vse
