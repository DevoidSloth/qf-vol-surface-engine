// vse/quad.hpp — Gauss-Legendre quadrature, including on a semi-infinite range.
//
// The characteristic-function pricers all reduce to an integral over
// [0, infinity) of a smooth, oscillatory, exponentially decaying integrand.
// Three things decide whether that integral is worth 1e-12 or 1e-6:
//
//   1. The rule. The integrands are analytic, so Gauss-Legendre converges
//      geometrically and Simpson does not. Composite Gauss-Legendre on a
//      handful of panels beats an adaptive Simpson with ten times the points.
//   2. The truncation. Cutting at a fixed u_max is the standard mistake: the
//      decay rate of the Heston integrand depends on v0, theta and T, so a
//      cutoff tuned at one year is far too short at one week (where the
//      integrand is still O(1) well past u = 200) and wastefully long at ten
//      years. The cutoff here is found from the integrand itself.
//   3. Not integrating to infinity at all. Mapping u = c t/(1-t) turns the
//      half-line into (0,1) and lets one fixed rule cover the whole tail. That
//      is what integrate_semi_infinite does, and it is the default.
#pragma once

#include "vse/common.hpp"

#include <functional>
#include <vector>

namespace vse {

/// Gauss-Legendre nodes and weights on [-1, 1].
///
/// Nodes come from Newton on the Legendre polynomial with the standard
/// Tricomi/Chebyshev starting guess, which converges in three or four steps for
/// every node at every order used here. Cached per order, because a calibration
/// asks for the same rule tens of thousands of times.
struct GaussLegendre {
    std::vector<Real> nodes, weights;

    explicit GaussLegendre(int n) : nodes(std::size_t(n)), weights(std::size_t(n)) {
        require(n > 0, "GaussLegendre: order must be positive");
        const int m = (n + 1) / 2;
        for (int i = 0; i < m; ++i) {
            // Chebyshev-like initial guess, then Newton on P_n.
            Real x = std::cos(PI * (Real(i) + 0.75) / (Real(n) + 0.5));
            Real dp = 0.0;
            for (int it = 0; it < 100; ++it) {
                Real p0 = 1.0, p1 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const Real p2 = p1;
                    p1 = p0;
                    p0 = ((2.0 * Real(j) + 1.0) * x * p1 - Real(j) * p2) / (Real(j) + 1.0);
                }
                dp = Real(n) * (x * p0 - p1) / (x * x - 1.0);
                const Real dx = -p0 / dp;
                x += dx;
                if (std::fabs(dx) <= 1e-15 * std::fmax(std::fabs(x), 1.0)) break;
            }
            const Real w = 2.0 / ((1.0 - x * x) * dp * dp);
            nodes[std::size_t(i)] = -x;
            nodes[std::size_t(n - 1 - i)] = x;
            weights[std::size_t(i)] = w;
            weights[std::size_t(n - 1 - i)] = w;
        }
    }

    static const GaussLegendre& cached(int n) {
        // One instance per order, built on first use. Orders used by this
        // library are few and small, so a flat array beats a map.
        static std::vector<std::pair<int, GaussLegendre>> cache;
        for (auto& [order, rule] : cache) {
            if (order == n) return rule;
        }
        cache.emplace_back(n, GaussLegendre(n));
        return cache.back().second;
    }
};

/// Composite Gauss-Legendre on [a, b].
template <class F>
inline Real integrate_gl(F&& f, Real a, Real b, int order = 24, int panels = 1) {
    const GaussLegendre& rule = GaussLegendre::cached(order);
    const Real h = (b - a) / Real(panels);
    Real total = 0.0;
    for (int p = 0; p < panels; ++p) {
        const Real lo = a + Real(p) * h, hi = lo + h;
        const Real mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
        Real sum = 0.0;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            sum += rule.weights[i] * f(mid + half * rule.nodes[i]);
        }
        total += half * sum;
    }
    return total;
}

/// Integral over [0, infinity) by the rational map u = c t / (1 - t).
///
/// du = c/(1-t)^2 dt, so the integrand is evaluated at points that crowd towards
/// the origin and stretch out to infinity, with no truncation error at all --
/// the far tail is inside the last panel rather than discarded.
///
/// `scale` sets where the map puts its resolution and should be near the width
/// of the integrand. For the Lewis integral that is roughly 1/sqrt(total
/// variance), which is what the Heston pricer passes.
template <class F>
inline Real integrate_semi_infinite(F&& f, Real scale, int order = 32, int panels = 6) {
    require(scale > 0.0, "integrate_semi_infinite: scale must be positive");
    return integrate_gl(
        [&](Real t) {
            const Real one_minus = 1.0 - t;
            if (one_minus <= 0.0) return 0.0;
            const Real u = scale * t / one_minus;
            const Real jac = scale / (one_minus * one_minus);
            const Real value = f(u);
            return std::isfinite(value) ? value * jac : 0.0;
        },
        0.0, 1.0, order, panels);
}

/// Where an integrand has decayed below `tolerance` relative to its value at the
/// origin, found by doubling then bisecting.
///
/// Used to size a truncated rule when one is wanted (the FFT pricer needs a
/// finite grid), and to report honestly how far the integrand actually reaches
/// rather than asserting a magic number.
template <class F>
inline Real integrand_cutoff(F&& f, Real start = 1.0, Real tolerance = 1e-16,
                             Real max_u = 1e7) {
    const Real scale = std::fmax(std::fabs(f(1e-8)), 1e-300);
    Real u = start;
    while (u < max_u && std::fabs(f(u)) > tolerance * scale) u *= 2.0;
    Real lo = u * 0.5, hi = u;
    for (int i = 0; i < 40; ++i) {
        const Real mid = 0.5 * (lo + hi);
        if (std::fabs(f(mid)) > tolerance * scale) lo = mid; else hi = mid;
    }
    return hi;
}

}  // namespace vse
