// vse/lm.hpp — Levenberg-Marquardt with an analytic Jacobian.
//
// Used by every calibration in the library: SVI and SSVI slices, Heston, SABR,
// Bates. The interesting design points:
//
//   * The Jacobian is a caller-supplied callback, so a model that can
//     differentiate itself does, and one that cannot falls back to central
//     differences through numerical_jacobian(). For Heston the analytic Jacobian
//     is the difference between a calibration that converges in a tenth of a
//     second and one that does not converge at all, because a five-parameter
//     central-difference Jacobian costs ten full surface repricings per step and
//     each of them is only accurate to ~1e-8.
//   * Marquardt's scaling (lambda times diag(J^T J), not lambda times I). Heston
//     parameters differ by three orders of magnitude in natural units --
//     v0 ~ 0.04 against kappa ~ 3 -- and an unscaled damping term effectively
//     freezes the small ones.
//   * Box constraints by projection rather than by parameter transform. Transforms
//     (tanh for rho, exp for kappa) are tidier to write but they flatten the
//     gradient as a parameter approaches its bound, so a calibration that wants
//     rho = -0.999 crawls. Projection keeps the geometry and reports which
//     parameters ended up pinned, which is diagnostic information a transform
//     throws away.
#pragma once

#include "vse/common.hpp"
#include "vse/linalg.hpp"

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace vse {

struct LMOptions {
    int  max_iterations   = 200;
    Real cost_tolerance   = 1e-14;   ///< relative decrease in 0.5*||r||^2
    Real gradient_tolerance = 1e-12; ///< max |J^T r|
    Real step_tolerance   = 1e-14;   ///< relative step size
    Real lambda_init      = 1e-3;
    Real lambda_up        = 10.0;
    Real lambda_down      = 0.1;
    Real lambda_max       = 1e12;
};

enum class LMStatus { CostTolerance, GradientTolerance, StepTolerance, MaxIterations, Singular };

struct LMResult {
    std::vector<Real> x;
    Real cost = 0.0;                 ///< 0.5 * sum of squared residuals
    Real rms  = 0.0;                 ///< sqrt(mean squared residual)
    int  iterations = 0;
    int  residual_evaluations = 0;
    int  jacobian_evaluations = 0;
    LMStatus status = LMStatus::MaxIterations;
    std::vector<bool> at_bound;      ///< which parameters finished pinned

    bool converged() const {
        return status == LMStatus::CostTolerance ||
               status == LMStatus::GradientTolerance ||
               status == LMStatus::StepTolerance;
    }
    const char* status_text() const {
        switch (status) {
            case LMStatus::CostTolerance:     return "cost tolerance reached";
            case LMStatus::GradientTolerance: return "gradient tolerance reached";
            case LMStatus::StepTolerance:     return "step tolerance reached";
            case LMStatus::MaxIterations:     return "iteration limit reached";
            case LMStatus::Singular:          return "normal equations singular at maximum damping";
        }
        return "unknown";
    }
};

/// r(x) -> residual vector.
using ResidualFn = std::function<void(const std::vector<Real>& x, std::vector<Real>& r)>;
/// J(x) -> m x n row-major Jacobian, dr_i/dx_j.
using JacobianFn = std::function<void(const std::vector<Real>& x, Matrix& j)>;

struct Box {
    std::vector<Real> lower, upper;
    bool empty() const { return lower.empty() && upper.empty(); }
};

/// Central-difference Jacobian, for models without an analytic one.
///
/// The step is relative to the parameter with an absolute floor, and central
/// rather than forward: forward differences give ~1e-8 relative accuracy at
/// best, which is enough to stall a calibration well above the noise floor of
/// the data.
inline JacobianFn numerical_jacobian(ResidualFn residual, Real rel_step = 1e-6) {
    return [residual, rel_step](const std::vector<Real>& x, Matrix& j) {
        const std::size_t n = x.size();
        std::vector<Real> xp = x, rp, rm;
        for (std::size_t k = 0; k < n; ++k) {
            const Real h = rel_step * std::fmax(std::fabs(x[k]), 1e-4);
            xp[k] = x[k] + h; residual(xp, rp);
            xp[k] = x[k] - h; residual(xp, rm);
            xp[k] = x[k];
            for (std::size_t i = 0; i < rp.size(); ++i) j(i, k) = (rp[i] - rm[i]) / (2.0 * h);
        }
    };
}

inline LMResult levenberg_marquardt(const ResidualFn& residual, const JacobianFn& jacobian,
                                    const std::vector<Real>& x0, const Box& box = {},
                                    const LMOptions& opt = {}) {
    const std::size_t n = x0.size();
    require(n > 0, "levenberg_marquardt: no parameters");

    auto clamp_to_box = [&](std::vector<Real>& x) {
        if (box.empty()) return;
        for (std::size_t k = 0; k < n; ++k) {
            if (!box.lower.empty()) x[k] = std::fmax(x[k], box.lower[k]);
            if (!box.upper.empty()) x[k] = std::fmin(x[k], box.upper[k]);
        }
    };

    LMResult out;
    out.x = x0;
    clamp_to_box(out.x);

    std::vector<Real> r, r_trial;
    residual(out.x, r);
    ++out.residual_evaluations;
    const std::size_t m = r.size();
    require(m >= n, "levenberg_marquardt: fewer residuals than parameters");

    auto cost_of = [](const std::vector<Real>& v) {
        Real c = 0.0;
        for (Real e : v) c += e * e;
        return 0.5 * c;
    };

    Real cost = cost_of(r);
    Matrix jac(m, n, 0.0), normal(n, n, 0.0);
    std::vector<Real> grad(n), step(n), x_trial(n), diag(n);
    Real lambda = opt.lambda_init;

    for (int iter = 1; iter <= opt.max_iterations; ++iter) {
        out.iterations = iter;
        jacobian(out.x, jac);
        ++out.jacobian_evaluations;

        // grad = J^T r, normal = J^T J
        for (std::size_t a = 0; a < n; ++a) {
            Real g = 0.0;
            for (std::size_t i = 0; i < m; ++i) g += jac(i, a) * r[i];
            grad[a] = g;
            for (std::size_t b = a; b < n; ++b) {
                Real s = 0.0;
                for (std::size_t i = 0; i < m; ++i) s += jac(i, a) * jac(i, b);
                normal(a, b) = s;
                normal(b, a) = s;
            }
        }

        Real gmax = 0.0;
        for (std::size_t a = 0; a < n; ++a) gmax = std::fmax(gmax, std::fabs(grad[a]));
        if (gmax < opt.gradient_tolerance) { out.status = LMStatus::GradientTolerance; break; }

        // Marquardt scaling: damp along the diagonal of J^T J so that badly
        // scaled parameters are damped in proportion to their own curvature.
        for (std::size_t a = 0; a < n; ++a) {
            diag[a] = normal(a, a) > 0.0 ? normal(a, a) : 1.0;
        }

        bool accepted = false;
        while (!accepted && lambda <= opt.lambda_max) {
            Matrix damped = normal;
            for (std::size_t a = 0; a < n; ++a) damped(a, a) += lambda * diag[a];

            std::vector<Real> neg_grad(n);
            for (std::size_t a = 0; a < n; ++a) neg_grad[a] = -grad[a];

            if (!spd_solve(damped, neg_grad, step)) {
                lambda *= opt.lambda_up;
                continue;
            }

            for (std::size_t a = 0; a < n; ++a) x_trial[a] = out.x[a] + step[a];
            clamp_to_box(x_trial);

            residual(x_trial, r_trial);
            ++out.residual_evaluations;
            const Real trial_cost = cost_of(r_trial);

            if (trial_cost < cost) {
                Real step_norm = 0.0, x_norm = 0.0;
                for (std::size_t a = 0; a < n; ++a) {
                    step_norm += sqr(x_trial[a] - out.x[a]);
                    x_norm += sqr(out.x[a]);
                }
                const Real rel_decrease = (cost - trial_cost) / std::fmax(cost, 1e-300);

                out.x = x_trial;
                r = r_trial;
                cost = trial_cost;
                lambda = std::fmax(lambda * opt.lambda_down, 1e-14);
                accepted = true;

                if (rel_decrease < opt.cost_tolerance) out.status = LMStatus::CostTolerance;
                else if (std::sqrt(step_norm) < opt.step_tolerance * (std::sqrt(x_norm) + opt.step_tolerance))
                    out.status = LMStatus::StepTolerance;
                else
                    out.status = LMStatus::MaxIterations;   // provisional
            } else {
                lambda *= opt.lambda_up;
            }
        }

        if (!accepted) { out.status = LMStatus::Singular; break; }
        if (out.status == LMStatus::CostTolerance || out.status == LMStatus::StepTolerance) break;
    }

    out.cost = cost;
    out.rms = std::sqrt(2.0 * cost / Real(m));
    out.at_bound.assign(n, false);
    if (!box.empty()) {
        for (std::size_t k = 0; k < n; ++k) {
            const Real scale = std::fmax(std::fabs(out.x[k]), 1e-12);
            if (!box.lower.empty() && std::fabs(out.x[k] - box.lower[k]) <= 1e-10 * scale)
                out.at_bound[k] = true;
            if (!box.upper.empty() && std::fabs(out.x[k] - box.upper[k]) <= 1e-10 * scale)
                out.at_bound[k] = true;
        }
    }
    return out;
}

}  // namespace vse
