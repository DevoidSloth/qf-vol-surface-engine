// vse/linalg.hpp — the small dense and banded solvers this library needs.
//
// Deliberately not a linear algebra library. Calibration solves 2x2 to 8x8
// symmetric systems and the PDE engine solves tridiagonal ones; both are small
// enough that a BLAS call would cost more in dispatch than in arithmetic, and
// pulling in Eigen to get them would be the tail wagging the dog.
//
// What does matter is that the failure modes are explicit. A Cholesky that
// silently returns garbage on an indefinite matrix turns a Levenberg-Marquardt
// step into a random walk that still looks like it is converging, so the
// factorisation reports whether it succeeded and the caller is expected to act.
#pragma once

#include "vse/common.hpp"

#include <algorithm>
#include <vector>

namespace vse {

/// Row-major dense matrix, sized at construction.
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, Real fill = 0.0)
        : rows_(rows), cols_(cols), a_(rows * cols, fill) {}

    Real& operator()(std::size_t i, std::size_t j) { return a_[i * cols_ + j]; }
    Real operator()(std::size_t i, std::size_t j) const { return a_[i * cols_ + j]; }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    Real* data() noexcept { return a_.data(); }
    const Real* data() const noexcept { return a_.data(); }

    void fill(Real v) { std::fill(a_.begin(), a_.end(), v); }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::vector<Real> a_;
};

/// In-place Cholesky, A = L L^T, lower triangle only. Returns false if A is not
/// positive definite -- which for a normal-equations matrix means the Jacobian
/// has lost rank, i.e. two parameters have become indistinguishable from the
/// data. That is a real and common calibration outcome, not an exception.
inline bool cholesky_factor(Matrix& a) {
    const std::size_t n = a.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            Real sum = a(i, j);
            for (std::size_t k = 0; k < j; ++k) sum -= a(i, k) * a(j, k);
            if (i == j) {
                if (!(sum > 0.0)) return false;
                a(i, i) = std::sqrt(sum);
            } else {
                a(i, j) = sum / a(j, j);
            }
        }
    }
    return true;
}

/// Solve L L^T x = b given the factor produced by cholesky_factor.
inline void cholesky_solve_in_place(const Matrix& l, std::vector<Real>& b) {
    const std::size_t n = l.rows();
    for (std::size_t i = 0; i < n; ++i) {           // forward substitution
        Real sum = b[i];
        for (std::size_t k = 0; k < i; ++k) sum -= l(i, k) * b[k];
        b[i] = sum / l(i, i);
    }
    for (std::size_t ii = n; ii-- > 0;) {           // back substitution
        Real sum = b[ii];
        for (std::size_t k = ii + 1; k < n; ++k) sum -= l(k, ii) * b[k];
        b[ii] = sum / l(ii, ii);
    }
}

/// Symmetric positive-definite solve. Returns false and leaves x untouched if A
/// is not positive definite.
inline bool spd_solve(Matrix a, const std::vector<Real>& b, std::vector<Real>& x) {
    if (!cholesky_factor(a)) return false;
    x = b;
    cholesky_solve_in_place(a, x);
    return true;
}

/// Thomas algorithm for a tridiagonal system.
///
/// sub[0] and sup[n-1] are ignored. Overwrites the work vectors, not the inputs,
/// so it is safe to call repeatedly with the same coefficient arrays -- which is
/// exactly what a PDE time-stepper does, thousands of times, with the same
/// matrix and a different right-hand side.
///
/// No pivoting. That is safe here and only here: the operators this solves are
/// diagonally dominant by construction (see pde.hpp, where the grid is built to
/// keep them so), and Thomas without pivoting is stable for diagonally dominant
/// systems. The precondition is checked in the tests, not assumed.
inline void thomas_solve(const std::vector<Real>& sub, const std::vector<Real>& diag,
                         const std::vector<Real>& sup, const std::vector<Real>& rhs,
                         std::vector<Real>& out, std::vector<Real>& work) {
    const std::size_t n = diag.size();
    out.resize(n);
    work.resize(n);

    Real beta = diag[0];
    out[0] = rhs[0] / beta;
    for (std::size_t i = 1; i < n; ++i) {
        work[i] = sup[i - 1] / beta;
        beta = diag[i] - sub[i] * work[i];
        out[i] = (rhs[i] - sub[i] * out[i - 1]) / beta;
    }
    for (std::size_t i = n - 1; i-- > 0;) out[i] -= work[i + 1] * out[i + 1];
}

/// Ordinary least squares by normal equations with a scale-free ridge.
///
/// The ridge is applied as A(j,j) *= (1 + ridge_rel), not as A(j,j) += ridge_rel
/// * trace(A). That distinction cost real time to find. With an absolute ridge
/// proportional to the trace, a design matrix whose columns differ in scale --
/// a column of ones next to a column of strikes near 4000, which is precisely
/// the put-call parity regression -- has its trace dominated by the large
/// column, so the intercept receives a perturbation that is enormous relative to
/// its own diagonal entry. Fed through a system with condition number ~1e9, that
/// produced a discount factor wrong in the third decimal place on input data
/// that satisfied put-call parity to 3e-13.
///
/// Normal equations square the condition number, which is the usual argument for
/// QR instead. Here the design matrices have at most three columns, and callers
/// are expected to centre any column with a large mean (see
/// implied_forward_from_parity), which is what actually keeps kappa in hand.
inline bool least_squares(const Matrix& design, const std::vector<Real>& y,
                          std::vector<Real>& beta, Real ridge_rel = 1e-13) {
    const std::size_t m = design.rows(), n = design.cols();
    require(y.size() == m, "least_squares: design and response disagree on length");

    Matrix ata(n, n, 0.0);
    std::vector<Real> atb(n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            atb[j] += design(i, j) * y[i];
            for (std::size_t k = j; k < n; ++k) ata(j, k) += design(i, j) * design(i, k);
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        ata(j, j) = ata(j, j) * (1.0 + ridge_rel) + ridge_rel * 1e-300;
        for (std::size_t k = 0; k < j; ++k) ata(j, k) = ata(k, j);
    }
    return spd_solve(ata, atb, beta);
}

}  // namespace vse
