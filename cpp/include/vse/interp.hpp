// vse/interp.hpp — cubic spline interpolation.
//
// Present for one reason: to be the control in the experiment. The claim that an
// arbitrage-free parameterisation is worth its constraints only means something
// if you have measured what the unconstrained alternative does, and the
// unconstrained alternative everyone actually uses is a cubic spline through the
// smile in delta space. It fits better -- it interpolates, so its RMSE is zero
// at the knots -- and it produces negative risk-neutral densities. That
// comparison is the point.
#pragma once

#include "vse/common.hpp"
#include "vse/linalg.hpp"

#include <algorithm>
#include <vector>

namespace vse {

/// Natural cubic spline with linear extrapolation outside the knots.
///
/// Extrapolation is linear rather than cubic on purpose: a cubic continued
/// beyond its last knot diverges, and in vol space that means a wing that goes
/// negative a short distance past the last quoted strike.
class CubicSpline {
public:
    CubicSpline() = default;

    CubicSpline(std::vector<Real> x, std::vector<Real> y) : x_(std::move(x)), y_(std::move(y)) {
        const std::size_t n = x_.size();
        require(n >= 2, "CubicSpline: need at least two knots");
        require(y_.size() == n, "CubicSpline: x and y differ in length");
        for (std::size_t i = 1; i < n; ++i) {
            require(x_[i] > x_[i - 1], "CubicSpline: knots must be strictly increasing");
        }

        m_.assign(n, 0.0);
        if (n == 2) return;   // a straight line; second derivatives are zero

        // Tridiagonal system for the second derivatives, natural end conditions.
        std::vector<Real> sub(n - 2, 0.0), diag(n - 2, 0.0), sup(n - 2, 0.0), rhs(n - 2, 0.0);
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const Real hm = x_[i] - x_[i - 1];
            const Real hp = x_[i + 1] - x_[i];
            const std::size_t r = i - 1;
            sub[r]  = hm / 6.0;
            diag[r] = (hm + hp) / 3.0;
            sup[r]  = hp / 6.0;
            rhs[r]  = (y_[i + 1] - y_[i]) / hp - (y_[i] - y_[i - 1]) / hm;
        }
        std::vector<Real> sol, work;
        thomas_solve(sub, diag, sup, rhs, sol, work);
        for (std::size_t i = 1; i + 1 < n; ++i) m_[i] = sol[i - 1];
    }

    Real operator()(Real x) const {
        const std::size_t n = x_.size();
        if (x <= x_.front()) {
            const Real slope = derivative_at_knot(0);
            return y_.front() + slope * (x - x_.front());
        }
        if (x >= x_.back()) {
            const Real slope = derivative_at_knot(n - 1);
            return y_.back() + slope * (x - x_.back());
        }
        const std::size_t i = segment(x);
        const Real h = x_[i + 1] - x_[i];
        const Real A = (x_[i + 1] - x) / h, B = (x - x_[i]) / h;
        return A * y_[i] + B * y_[i + 1] +
               ((A * A * A - A) * m_[i] + (B * B * B - B) * m_[i + 1]) * h * h / 6.0;
    }

    Real derivative(Real x) const {
        const std::size_t n = x_.size();
        if (x <= x_.front()) return derivative_at_knot(0);
        if (x >= x_.back()) return derivative_at_knot(n - 1);
        const std::size_t i = segment(x);
        const Real h = x_[i + 1] - x_[i];
        const Real A = (x_[i + 1] - x) / h, B = (x - x_[i]) / h;
        return (y_[i + 1] - y_[i]) / h +
               ((-3.0 * A * A + 1.0) * m_[i] + (3.0 * B * B - 1.0) * m_[i + 1]) * h / 6.0;
    }

    Real second_derivative(Real x) const {
        if (x <= x_.front() || x >= x_.back()) return 0.0;   // linear extrapolation
        const std::size_t i = segment(x);
        const Real h = x_[i + 1] - x_[i];
        const Real A = (x_[i + 1] - x) / h, B = (x - x_[i]) / h;
        return A * m_[i] + B * m_[i + 1];
    }

    bool empty() const { return x_.empty(); }
    const std::vector<Real>& knots() const { return x_; }

private:
    std::size_t segment(Real x) const {
        const auto it = std::upper_bound(x_.begin(), x_.end(), x);
        std::size_t i = std::size_t(it - x_.begin());
        if (i == 0) i = 1;
        if (i >= x_.size()) i = x_.size() - 1;
        return i - 1;
    }

    Real derivative_at_knot(std::size_t i) const {
        const std::size_t n = x_.size();
        if (n < 2) return 0.0;
        const std::size_t j = (i == 0) ? 0 : n - 2;
        const Real h = x_[j + 1] - x_[j];
        if (i == 0) return (y_[1] - y_[0]) / h - h * (2.0 * m_[0] + m_[1]) / 6.0;
        return (y_[n - 1] - y_[n - 2]) / h + h * (m_[n - 2] + 2.0 * m_[n - 1]) / 6.0;
    }

    std::vector<Real> x_, y_, m_;
};

}  // namespace vse
