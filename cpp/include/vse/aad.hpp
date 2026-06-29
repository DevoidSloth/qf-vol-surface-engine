// vse/aad.hpp — reverse-mode adjoint algorithmic differentiation.
//
// WHY REVERSE MODE
//
// Forward mode (dual.hpp) carries one derivative slot per INPUT, so a gradient
// with respect to n parameters costs n times the value computation. That is the
// right trade for the Heston characteristic function, where n is five and the
// function is small.
//
// Reverse mode carries one adjoint per intermediate VALUE and sweeps backwards
// once, so a gradient with respect to n parameters costs a constant multiple of
// the value computation -- typically three to four -- INDEPENDENT OF n. That is
// the right trade for a Monte Carlo pricer with ten or fifty risk factors, where
// the alternative is 2n + 1 full repricings.
//
// The asymmetry is not an implementation detail. It is why every derivatives
// desk that computes a full risk ladder does it this way, and why the cost of
// adding a risk factor to an adjoint pricer is essentially zero.
//
// HOW IT WORKS
//
// Every arithmetic operation on an ADouble appends a node to a tape recording
// its parents and the local partial derivatives. After the value is computed,
// seeding the output adjoint with 1 and sweeping the tape backwards accumulates
// dOutput/dx for every recorded variable at once, by the chain rule.
//
// WHAT IT DOES NOT DO
//
// Pathwise differentiation -- which is what this is, applied to a Monte Carlo --
// requires the payoff to be almost surely differentiable in the parameters. A
// vanilla payoff has a kink at the strike, which is a measure-zero event and so
// harmless for FIRST derivatives. It is not harmless for second derivatives:
// pathwise gamma of a call is identically zero on every path and the true gamma
// is not. Gamma needs a likelihood-ratio or mixed estimator, and this file does
// not pretend otherwise -- the test suite checks first-order Greeks against
// central differences and states the restriction.
#pragma once

#include "vse/common.hpp"
#include "vse/normal.hpp"

#include <vector>

namespace vse {

/// The tape.
///
/// One node per operation: two parent indices and two local partial derivatives,
/// stored as a single array of 32-byte structs so that a push is one capacity
/// check and the backward sweep reads each node from one cache line. (Four
/// parallel vectors were tried and measured no slower on this machine; the
/// single array is kept because it is simpler, not because it was faster.)
class Tape {
public:
    static constexpr std::size_t kNoParent = std::size_t(-1);

    struct Node {
        std::size_t parent1 = kNoParent;
        std::size_t parent2 = kNoParent;
        Real weight1 = 0.0;
        Real weight2 = 0.0;
    };

    std::size_t push_constant() {
        nodes_.push_back(Node{});
        return nodes_.size() - 1;
    }

    std::size_t push_unary(std::size_t parent, Real weight) {
        nodes_.push_back(Node{parent, kNoParent, weight, 0.0});
        return nodes_.size() - 1;
    }

    std::size_t push_binary(std::size_t p1, Real w1, std::size_t p2, Real w2) {
        nodes_.push_back(Node{p1, p2, w1, w2});
        return nodes_.size() - 1;
    }

    /// Reset for the next pricing without releasing memory.
    ///
    /// std::vector::clear keeps capacity, so after the first path a Monte Carlo
    /// inner loop never allocates again. Constructing a fresh Tape per path
    /// instead -- the obvious thing to write -- spends most of its time in the
    /// allocator.
    void clear() {
        nodes_.clear();
        adjoint_.clear();
    }

    void reserve(std::size_t n) {
        nodes_.reserve(n);
        adjoint_.reserve(n);
    }

    std::size_t size() const { return nodes_.size(); }

    /// Sweep backwards from `output`, seeding its adjoint with `seed`.
    void backpropagate(std::size_t output, Real seed = 1.0) {
        adjoint_.assign(nodes_.size(), 0.0);
        require(output < adjoint_.size(), "Tape: output index out of range");
        adjoint_[output] = seed;
        for (std::size_t i = nodes_.size(); i-- > 0;) {
            const Real a = adjoint_[i];
            if (a == 0.0) continue;              // most of a tape is never touched
            const Node& node = nodes_[i];
            if (node.parent1 != kNoParent) adjoint_[node.parent1] += a * node.weight1;
            if (node.parent2 != kNoParent) adjoint_[node.parent2] += a * node.weight2;
        }
    }

    Real adjoint(std::size_t index) const {
        return index < adjoint_.size() ? adjoint_[index] : 0.0;
    }

    static Tape& active();

private:
    std::vector<Node> nodes_;
    std::vector<Real> adjoint_;
};

namespace detail {
/// The default tape for the calling thread.
///
/// Consulted when an ADouble is built from a bare Real; every operation between
/// existing ADoubles uses the pointer they already carry, which is what keeps
/// thread-local storage out of the hot path. See the note on ADouble.
///
/// Namespace scope rather than function-local static: a function-local
/// `static thread_local` also pays a construction-guard check on every call.
inline thread_local Tape g_active_tape;
}  // namespace detail

inline Tape& Tape::active() { return detail::g_active_tape; }

/// A differentiable double.
///
/// Carries a pointer to its tape rather than looking one up.
///
/// The obvious design is a thread_local active tape consulted by every
/// operation, and it is four times slower. A tape node is a few instructions, so
/// a thread-local storage lookup on each one dominates -- measured at 7.4x the
/// cost of an undifferentiated price for a function-local `static thread_local`
/// (which pays a construction-guard check as well), 4.4x for a namespace-scope
/// one (lookup only), and 1.7x with the pointer carried here.
///
/// The cost is eight bytes per value. Since an ADouble already carries a double
/// and an index, that takes it from sixteen bytes to twenty-four, which is a
/// trade worth making by a wide margin: the tape itself is far larger than the
/// live values at any moment.
///
/// Thread safety comes free with it. Each thread constructs its own Tape and the
/// values built from it point at it, so there is no shared mutable state and no
/// synchronisation.
class ADouble {
public:
    ADouble() : ADouble(0.0) {}
    ADouble(Real v)                                                       // NOLINT
        : value_(v), tape_(&Tape::active()), index_(tape_->push_constant()) {}
    ADouble(Real v, Tape* tape, std::size_t index) : value_(v), tape_(tape), index_(index) {}

    /// Seed an independent variable on a specific tape.
    static ADouble variable(Real v, Tape& tape) {
        return {v, &tape, tape.push_constant()};
    }

    Real value() const { return value_; }
    std::size_t index() const { return index_; }
    Tape* tape() const { return tape_; }
    Real adjoint() const { return tape_->adjoint(index_); }

private:
    Real value_;
    Tape* tape_;
    std::size_t index_;
};

/// See common.hpp; a non-template overload so it wins over the generic one.
inline Real value_of(const ADouble& a) { return a.value(); }

inline ADouble operator+(const ADouble& a, const ADouble& b) {
    return {a.value() + b.value(), a.tape(),
            a.tape()->push_binary(a.index(), 1.0, b.index(), 1.0)};
}
inline ADouble operator-(const ADouble& a, const ADouble& b) {
    return {a.value() - b.value(), a.tape(),
            a.tape()->push_binary(a.index(), 1.0, b.index(), -1.0)};
}
inline ADouble operator-(const ADouble& a) {
    return {-a.value(), a.tape(), a.tape()->push_unary(a.index(), -1.0)};
}
inline ADouble operator*(const ADouble& a, const ADouble& b) {
    return {a.value() * b.value(), a.tape(),
            a.tape()->push_binary(a.index(), b.value(), b.index(), a.value())};
}
inline ADouble operator/(const ADouble& a, const ADouble& b) {
    const Real inv = 1.0 / b.value();
    return {a.value() * inv, a.tape(),
            a.tape()->push_binary(a.index(), inv, b.index(), -a.value() * inv * inv)};
}

// Mixed operations record a UNARY node, not a constant node plus a binary one.
//
// The naive version -- promote the scalar to an ADouble and reuse the binary
// operator -- is correct and doubles the tape. A pricer is full of
// multiplications by dt, by 0.5, by a fixed random draw, and that alone was a
// third of the tape length and a third of the backward sweep.
inline ADouble operator+(const ADouble& a, Real b) {
    return {a.value() + b, a.tape(), a.tape()->push_unary(a.index(), 1.0)};
}
inline ADouble operator+(Real a, const ADouble& b) { return b + a; }
inline ADouble operator-(const ADouble& a, Real b) {
    return {a.value() - b, a.tape(), a.tape()->push_unary(a.index(), 1.0)};
}
inline ADouble operator-(Real a, const ADouble& b) {
    return {a - b.value(), b.tape(), b.tape()->push_unary(b.index(), -1.0)};
}
inline ADouble operator*(const ADouble& a, Real b) {
    return {a.value() * b, a.tape(), a.tape()->push_unary(a.index(), b)};
}
inline ADouble operator*(Real a, const ADouble& b) { return b * a; }
inline ADouble operator/(const ADouble& a, Real b) {
    const Real inv = 1.0 / b;
    return {a.value() * inv, a.tape(), a.tape()->push_unary(a.index(), inv)};
}
inline ADouble operator/(Real a, const ADouble& b) {
    const Real inv = 1.0 / b.value();
    return {a * inv, b.tape(), b.tape()->push_unary(b.index(), -a * inv * inv)};
}

inline bool operator<(const ADouble& a, const ADouble& b) { return a.value() < b.value(); }
inline bool operator>(const ADouble& a, const ADouble& b) { return a.value() > b.value(); }
inline bool operator<(const ADouble& a, Real b) { return a.value() < b; }
inline bool operator>(const ADouble& a, Real b) { return a.value() > b; }

inline ADouble exp(const ADouble& a) {
    const Real v = std::exp(a.value());
    return {v, a.tape(), a.tape()->push_unary(a.index(), v)};
}
inline ADouble log(const ADouble& a) {
    return {std::log(a.value()), a.tape(), a.tape()->push_unary(a.index(), 1.0 / a.value())};
}
inline ADouble sqrt(const ADouble& a) {
    const Real v = std::sqrt(a.value());
    return {v, a.tape(), a.tape()->push_unary(a.index(), v > 0.0 ? 0.5 / v : 0.0)};
}

/// The normal CDF as a single tape node.
///
/// Differentiating through Cody's rational approximation would work and would be
/// wasteful: thirty nodes to recover a derivative that is known in closed form.
/// One node with weight phi(x) is faster and more accurate, and this is what
/// user-defined primitives are for.
inline ADouble norm_cdf_ad(const ADouble& a) {
    return {norm_cdf(a.value()), a.tape(), a.tape()->push_unary(a.index(), norm_pdf(a.value()))};
}

/// max(a, 0), with the subgradient convention that the kink contributes zero.
///
/// This is where pathwise differentiation earns its restriction: the derivative
/// is an indicator, correct almost surely, undefined on a measure-zero set, and
/// identically zero one order up. First-order Greeks are fine; gamma is not.
inline ADouble max_zero(const ADouble& a) {
    const bool positive = a.value() > 0.0;
    return {positive ? a.value() : 0.0, a.tape(),
            a.tape()->push_unary(a.index(), positive ? 1.0 : 0.0)};
}

/// Black-Scholes, written once and instantiated for both Real and ADouble.
///
/// Templating rather than duplicating is the point: an adjoint pricer that is a
/// second implementation of the model will eventually disagree with the first,
/// and the disagreement will be found by a trader rather than by a test.
template <class T>
inline T bs_price_generic(const T& spot, const T& strike, const T& expiry, const T& rate,
                          const T& dividend, const T& sigma, OptionType type) {
    using std::exp;
    using std::log;
    using std::sqrt;
    const T srt = sigma * sqrt(expiry);
    const T d1 = (log(spot / strike) + (rate - dividend + T(0.5) * sigma * sigma) * expiry) / srt;
    const T d2 = d1 - srt;
    const T dfq = exp(-(dividend * expiry));
    const T dfr = exp(-(rate * expiry));

    if constexpr (std::is_same_v<T, ADouble>) {
        if (type == OptionType::Call) {
            return spot * dfq * norm_cdf_ad(d1) - strike * dfr * norm_cdf_ad(d2);
        }
        return strike * dfr * norm_cdf_ad(-d2) - spot * dfq * norm_cdf_ad(-d1);
    } else {
        if (type == OptionType::Call) {
            return spot * dfq * norm_cdf(d1) - strike * dfr * norm_cdf(d2);
        }
        return strike * dfr * norm_cdf(-d2) - spot * dfq * norm_cdf(-d1);
    }
}

}  // namespace vse
