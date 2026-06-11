// vse/fft.hpp — radix-2 fast Fourier transform.
//
// Iterative Cooley-Tukey, decimation in time. Not a library dependency because
// this is thirty lines and pulling in FFTW to price options would mean shipping
// a GPL dependency and a plan cache for transforms that take 40 microseconds.
//
// Precision note: the twiddle factors are computed by direct evaluation of
// cos/sin at each stage rather than by recurrence. The recurrence is faster and
// accumulates error like O(sqrt(log N)) per element, which is invisible at
// N = 4096 -- but the FFT here feeds a Carr-Madan pricer whose output is
// cross-checked against a quadrature at 1e-10, and it is not worth spending any
// of that budget on the transform.
#pragma once

#include "vse/common.hpp"

#include <complex>
#include <vector>

namespace vse {

/// In-place forward DFT, sum_j x_j e^{-2 pi i jk / N}. Size must be a power of two.
inline void fft_in_place(std::vector<std::complex<Real>>& a) {
    const std::size_t n = a.size();
    if (n <= 1) return;
    require((n & (n - 1)) == 0, "fft_in_place: size must be a power of two");

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const Real angle = -TWO_PI / Real(len);
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t k = 0; k < len / 2; ++k) {
                const Real phi = angle * Real(k);
                const std::complex<Real> w(std::cos(phi), std::sin(phi));
                const std::complex<Real> u = a[i + k];
                const std::complex<Real> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }
}

/// In-place inverse DFT, normalised by 1/N.
inline void ifft_in_place(std::vector<std::complex<Real>>& a) {
    for (auto& z : a) z = std::conj(z);
    fft_in_place(a);
    const Real inv = 1.0 / Real(a.size());
    for (auto& z : a) z = std::conj(z) * inv;
}

}  // namespace vse
