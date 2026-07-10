// vse/common.hpp — shared types, constants and small utilities.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace vse {

using Real = double;

inline constexpr Real PI          = 3.14159265358979323846;
inline constexpr Real TWO_PI      = 6.28318530717958647692;
inline constexpr Real SQRT_TWO_PI = 2.50662827463100050242;
inline constexpr Real INV_SQRT_2PI= 0.39894228040143267794;
inline constexpr Real SQRT_2      = 1.41421356237309504880;
inline constexpr Real INV_SQRT_2  = 0.70710678118654752440;
inline constexpr Real SQRT_HALF_PI= 1.25331413731550025121;

inline constexpr Real DBL_EPS  = std::numeric_limits<Real>::epsilon();
inline constexpr Real DBL_HUGE = std::numeric_limits<Real>::max();

enum class OptionType : int { Call = 1, Put = -1 };

inline constexpr Real omega(OptionType t) noexcept {
    return t == OptionType::Call ? 1.0 : -1.0;
}

/// Thrown for genuinely invalid inputs. Numerical non-convergence gets its own
/// type so callers can distinguish "you gave me nonsense" from "I gave up".
struct DomainError : std::invalid_argument {
    explicit DomainError(const std::string& what) : std::invalid_argument(what) {}
};

struct ConvergenceError : std::runtime_error {
    explicit ConvergenceError(const std::string& what) : std::runtime_error(what) {}
};

inline void require(bool cond, const char* msg) {
    if (!cond) throw DomainError(msg);
}

template <typename T>
constexpr T sqr(T x) noexcept { return x * x; }

/// Underlying numeric value of a scalar or of a differentiable wrapper.
///
/// Lets one templated pricer branch on magnitude without knowing whether it is
/// being differentiated -- which is the whole point of writing the pricer once
/// and instantiating it for Real, Dual and ADouble rather than three times.
/// dual.hpp and aad.hpp add their own overloads.
template <typename T>
constexpr const T& value_of(const T& x) noexcept { return x; }

template <typename T>
constexpr T clampv(T x, T lo, T hi) noexcept { return x < lo ? lo : (x > hi ? hi : x); }

/// A market observation of a European option, expressed on the forward.
/// Everything downstream works on (forward, strike, expiry, discount factor) —
/// spot never appears, because the implied forward is what a surface is fitted
/// against. See docs/forward.md.
struct Quote {
    Real forward;    ///< F(T), implied from put-call parity where possible
    Real strike;     ///< K
    Real expiry;     ///< T in years
    Real discount;   ///< P(0,T)
    Real price;      ///< option premium (undiscounted price = price / discount)
    OptionType type;
};

}  // namespace vse
