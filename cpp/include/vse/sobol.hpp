// vse/sobol.hpp — Sobol low-discrepancy sequence and the Brownian bridge.
//
// These two belong together because neither is worth much without the other.
//
// A Sobol sequence fills the unit cube more evenly than random points, and the
// error of quasi-Monte Carlo integration falls like (log N)^d / N rather than
// 1/sqrt(N) -- but only in the dimensions where the sequence is actually well
// distributed. Beyond the first few dozen dimensions its projections degrade,
// and a 256-step simulation that draws its increments in time order spreads the
// variance evenly across 256 dimensions, so most of it lands where the sequence
// is no better than random. Measured on a Heston MC, plain Sobol with sequential
// sampling is barely distinguishable from pseudo-random.
//
// The Brownian bridge changes what the dimensions mean. Rather than drawing
// increments in time order, it draws the terminal value first, then the midpoint
// conditional on it, then the two quarter points, and so on. The first dimension
// then carries the terminal value of the Brownian motion -- which for a European
// payoff is nearly all of the variance -- and the later, worse-distributed
// dimensions carry only fine detail of the path. That is what makes the
// combination effective, and why the variance reduction from Sobol is quoted
// with the bridge and not without it.
#pragma once

#include "vse/common.hpp"
#include "vse/normal.hpp"
#include "vse/sobol_data.hpp"

#include <cstdint>
#include <vector>

namespace vse {

/// Sobol generator with Joe-Kuo direction numbers, Gray-code ordering.
class Sobol {
public:
    explicit Sobol(int dimensions) : dim_(std::size_t(dimensions)) {
        require(dimensions >= 1, "Sobol: need at least one dimension");
        require(dimensions <= sobol_data::kMaxDimensions,
                "Sobol: more dimensions requested than the direction-number table holds "
                "(regenerate with scripts/gen_sobol.py --dimensions N)");

        v_.assign(dim_ * kBits, 0);
        x_.assign(dim_, 0);

        // Dimension 1: v[i] = 2^{32-i}.
        for (std::size_t i = 0; i < kBits; ++i) {
            v_[0 * kBits + i] = std::uint32_t(1) << (31 - i);
        }

        for (std::size_t d = 1; d < dim_; ++d) {
            const std::uint32_t a = sobol_data::kPolynomial[d - 1];
            const std::size_t s = sobol_data::kDegree[d - 1];
            const std::uint32_t* minit = sobol_data::kInitial + sobol_data::kOffset[d - 1];

            for (std::size_t i = 0; i < s && i < kBits; ++i) {
                v_[d * kBits + i] = minit[i] << (31 - i);
            }
            for (std::size_t i = s; i < kBits; ++i) {
                std::uint32_t value = v_[d * kBits + i - s];
                value ^= value >> s;
                for (std::size_t k = 1; k < s; ++k) {
                    if ((a >> (s - 1 - k)) & 1u) value ^= v_[d * kBits + i - k];
                }
                v_[d * kBits + i] = value;
            }
        }
    }

    std::size_t dimensions() const { return dim_; }

    /// Advance the generator; the raw 32-bit coordinates are left in state().
    ///
    /// Raw integers rather than doubles, because a randomised QMC estimator
    /// applies a digital shift -- an XOR with a random word per dimension --
    /// which has to happen before the conversion to a real. Doing it in floating
    /// point instead (add a uniform, take the fractional part) is the
    /// Cranley-Patterson shift, which works for lattices but destroys the
    /// net structure of a digital sequence like Sobol.
    void next_raw() {
        std::uint32_t c = 0;
        std::uint64_t value = count_;
        while (value & 1u) { value >>= 1; ++c; }
        require(c < kBits, "Sobol: sequence exhausted (2^32 points)");
        for (std::size_t d = 0; d < dim_; ++d) x_[d] ^= v_[d * kBits + c];
        ++count_;
    }

    const std::vector<std::uint32_t>& state() const { return x_; }

    /// Convert a raw coordinate to (0, 1), optionally digitally shifted.
    static Real to_unit(std::uint32_t raw, std::uint32_t shift = 0) {
        return (Real(raw ^ shift) + 0.5) * (1.0 / 4294967296.0);
    }

    /// Next point, written into `out`, componentwise in (0, 1).
    ///
    /// The first point of the sequence is the origin, which maps to an infinite
    /// normal under inversion and would poison a path. It is skipped, as is
    /// standard; `reset` returns to the same starting state so runs remain
    /// reproducible.
    void next(std::vector<Real>& out) {
        next_raw();
        out.resize(dim_);
        for (std::size_t d = 0; d < dim_; ++d) out[d] = to_unit(x_[d]);
    }

    /// Next point mapped to standard normals by inversion.
    ///
    /// Inversion rather than Box-Muller, and not as a matter of taste: Box-Muller
    /// consumes two uniforms to make two normals through a polar transformation
    /// that destroys the low-discrepancy structure, so applying it to a Sobol
    /// sequence throws away the entire benefit.
    void next_normal(std::vector<Real>& out) {
        next(out);
        for (Real& u : out) u = norm_inv_cdf(u);
    }

    void reset() {
        std::fill(x_.begin(), x_.end(), 0u);
        count_ = 1;
    }

private:
    static constexpr std::size_t kBits = 32;
    std::size_t dim_;
    std::vector<std::uint32_t> v_, x_;
    std::uint64_t count_ = 1;   // 1, so the origin is never emitted
};

/// Brownian bridge reordering of a vector of independent normals.
///
/// Construction follows Glasserman: build a binary subdivision of the time axis,
/// filling in each point conditional on the two already-known neighbours. Point
/// zero is the terminal value, so dimension zero of the Sobol sequence carries
/// the most important direction.
class BrownianBridge {
public:
    explicit BrownianBridge(const std::vector<Real>& times) : times_(times) {
        const std::size_t n = times.size();
        require(n >= 1, "BrownianBridge: need at least one time");
        for (std::size_t i = 1; i < n; ++i) {
            require(times[i] > times[i - 1], "BrownianBridge: times must be increasing");
        }

        bridge_.resize(n);
        left_.resize(n);
        right_.resize(n);
        left_weight_.resize(n);
        right_weight_.resize(n);
        std_dev_.resize(n);

        std::vector<std::size_t> map(n, 0);
        map[n - 1] = 1;
        bridge_[0] = n - 1;
        left_[0] = 0;
        right_[0] = n - 1;
        std_dev_[0] = std::sqrt(times[n - 1]);
        left_weight_[0] = 0.0;
        right_weight_[0] = 0.0;

        std::size_t j = 0;
        for (std::size_t i = 1; i < n; ++i) {
            while (map[j]) ++j;                 // first unfilled point
            std::size_t k = j;
            while (!map[k]) ++k;                // next filled point to its right
            const std::size_t l = j + ((k - 1 - j) >> 1);   // midpoint

            map[l] = i + 1;
            bridge_[i] = l;
            left_[i] = j;
            right_[i] = k;

            const Real t_left = (j > 0) ? times[j - 1] : 0.0;
            const Real t_mid = times[l];
            const Real t_right = times[k];
            left_weight_[i] = (t_right - t_mid) / (t_right - t_left);
            right_weight_[i] = (t_mid - t_left) / (t_right - t_left);
            std_dev_[i] = std::sqrt((t_mid - t_left) * (t_right - t_mid) / (t_right - t_left));

            j = k + 1;
            if (j >= n) j = 0;
        }
    }

    std::size_t size() const { return times_.size(); }

    /// Turn `normals` into a Brownian path sampled at the configured times.
    void build(const std::vector<Real>& normals, std::vector<Real>& path) const {
        const std::size_t n = times_.size();
        require(normals.size() >= n, "BrownianBridge: not enough normals");
        path.assign(n, 0.0);

        path[bridge_[0]] = std_dev_[0] * normals[0];
        for (std::size_t i = 1; i < n; ++i) {
            const std::size_t l = left_[i], k = bridge_[i], r = right_[i];
            const Real left_value = (l > 0) ? path[l - 1] : 0.0;
            path[k] = left_weight_[i] * left_value + right_weight_[i] * path[r] +
                      std_dev_[i] * normals[i];
        }
    }

    /// The increments of the path, which is what a simulation steps with.
    void build_increments(const std::vector<Real>& normals, std::vector<Real>& increments,
                          std::vector<Real>& scratch) const {
        build(normals, scratch);
        const std::size_t n = times_.size();
        increments.resize(n);
        Real previous = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            increments[i] = scratch[i] - previous;
            previous = scratch[i];
        }
    }

private:
    std::vector<Real> times_;
    std::vector<std::size_t> bridge_, left_, right_;
    std::vector<Real> left_weight_, right_weight_, std_dev_;
};

}  // namespace vse
