// vse/dual.hpp — forward-mode automatic differentiation.
//
// Used to differentiate the Heston characteristic function with respect to its
// five parameters. The alternative is to hand-derive dd/drho, dg2/dsigma and the
// rest, which is roughly two pages of algebra that has to be right in every term
// -- and when it is wrong the calibration still converges, just to the wrong
// place and slowly, which is the worst kind of wrong.
//
// Two points worth being precise about, because "analytic Jacobian" is a claim
// people make loosely:
//
//   * These derivatives are EXACT, not approximations. Forward-mode AD applies
//     the chain rule to the same arithmetic the value computation performs; the
//     result differs from a hand-derived closed form only in rounding. It is not
//     a finite difference and has no step size.
//   * The value type is a template parameter, so the same code differentiates
//     through complex arithmetic. That matters here: the Heston CF lives in the
//     complex plane, complex-step differentiation cannot be used on a function
//     that is already complex, and a real-valued AD framework would not compose
//     with it at all.
//
// Cost is one extra value-sized slot per active parameter, so a five-parameter
// gradient costs about six characteristic-function evaluations' worth of
// arithmetic in a single pass -- against eleven full repricings for a central
// difference, each of them accurate to only ~1e-8.
#pragma once

#include "vse/common.hpp"

#include <array>
#include <complex>

namespace vse {

/// Forward-mode dual number carrying N first-order derivatives.
template <class T, int N>
struct Dual {
    T v{};
    std::array<T, N> d{};

    Dual() = default;
    Dual(const T& value) : v(value) { d.fill(T{}); }          // NOLINT: implicit is intended
    Dual(const T& value, int seed_index) : v(value) {
        d.fill(T{});
        d[std::size_t(seed_index)] = T(1);
    }

    static Dual constant(const T& value) { return Dual(value); }
    static Dual variable(const T& value, int index) { return Dual(value, index); }
};

template <class T, int N>
Dual<T, N> operator+(const Dual<T, N>& a, const Dual<T, N>& b) {
    Dual<T, N> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = a.d[std::size_t(i)] + b.d[std::size_t(i)];
    return r;
}

template <class T, int N>
Dual<T, N> operator-(const Dual<T, N>& a, const Dual<T, N>& b) {
    Dual<T, N> r;
    r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = a.d[std::size_t(i)] - b.d[std::size_t(i)];
    return r;
}

template <class T, int N>
Dual<T, N> operator-(const Dual<T, N>& a) {
    Dual<T, N> r;
    r.v = -a.v;
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = -a.d[std::size_t(i)];
    return r;
}

template <class T, int N>
Dual<T, N> operator*(const Dual<T, N>& a, const Dual<T, N>& b) {
    Dual<T, N> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) {
        r.d[std::size_t(i)] = a.d[std::size_t(i)] * b.v + a.v * b.d[std::size_t(i)];
    }
    return r;
}

template <class T, int N>
Dual<T, N> operator/(const Dual<T, N>& a, const Dual<T, N>& b) {
    Dual<T, N> r;
    const T inv = T(1) / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) {
        r.d[std::size_t(i)] = (a.d[std::size_t(i)] - r.v * b.d[std::size_t(i)]) * inv;
    }
    return r;
}

// Mixed arithmetic with the underlying scalar type.
template <class T, int N> Dual<T, N> operator+(const Dual<T, N>& a, const T& b) { return a + Dual<T, N>(b); }
template <class T, int N> Dual<T, N> operator+(const T& a, const Dual<T, N>& b) { return Dual<T, N>(a) + b; }
template <class T, int N> Dual<T, N> operator-(const Dual<T, N>& a, const T& b) { return a - Dual<T, N>(b); }
template <class T, int N> Dual<T, N> operator-(const T& a, const Dual<T, N>& b) { return Dual<T, N>(a) - b; }
template <class T, int N> Dual<T, N> operator*(const Dual<T, N>& a, const T& b) { return a * Dual<T, N>(b); }
template <class T, int N> Dual<T, N> operator*(const T& a, const Dual<T, N>& b) { return Dual<T, N>(a) * b; }
template <class T, int N> Dual<T, N> operator/(const Dual<T, N>& a, const T& b) { return a / Dual<T, N>(b); }
template <class T, int N> Dual<T, N> operator/(const T& a, const Dual<T, N>& b) { return Dual<T, N>(a) / b; }

template <class T, int N>
Dual<T, N> exp(const Dual<T, N>& a) {
    using std::exp;
    Dual<T, N> r;
    r.v = exp(a.v);
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = r.v * a.d[std::size_t(i)];
    return r;
}

template <class T, int N>
Dual<T, N> log(const Dual<T, N>& a) {
    using std::log;
    Dual<T, N> r;
    r.v = log(a.v);
    const T inv = T(1) / a.v;
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = a.d[std::size_t(i)] * inv;
    return r;
}

template <class T, int N>
Dual<T, N> sqrt(const Dual<T, N>& a) {
    using std::sqrt;
    Dual<T, N> r;
    r.v = sqrt(a.v);
    const T half_inv = T(0.5) / r.v;
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = a.d[std::size_t(i)] * half_inv;
    return r;
}

/// Real part of a complex dual, with its derivatives.
template <int N>
inline Dual<Real, N> real(const Dual<std::complex<Real>, N>& a) {
    Dual<Real, N> r;
    r.v = a.v.real();
    for (int i = 0; i < N; ++i) r.d[std::size_t(i)] = a.d[std::size_t(i)].real();
    return r;
}

/// See common.hpp for the scalar case.
template <class T, int N> inline const T& value_of(const Dual<T, N>& x) { return x.v; }

/// Convenience aliases for the two shapes this library uses.
template <int N> using RealDual = Dual<Real, N>;
template <int N> using ComplexDual = Dual<std::complex<Real>, N>;

}  // namespace vse
