// vse/rng.hpp — pseudo-random generators.
//
// std::mt19937 is not used anywhere in this library. Two reasons:
//
//   * Mersenne Twister carries 2.5 KB of state, which evicts more useful things
//     from L1 in a Monte Carlo inner loop, and it fails several of the BigCrush
//     linear-complexity tests. xoshiro256++ is four words of state, passes
//     BigCrush, and is several times faster.
//   * More importantly, std::normal_distribution and friends are not specified
//     to produce the same values across standard libraries. A Monte Carlo price
//     that differs between Linux and Windows because libstdc++ and MSVC disagree
//     on how to consume uniforms is not reproducible, and reproducibility is the
//     only thing that makes an MC regression test meaningful. Everything here is
//     specified bit for bit.
#pragma once

#include "vse/common.hpp"
#include "vse/normal.hpp"

#include <cstdint>

namespace vse {

/// xoshiro256++ (Blackman & Vigna, 2018). Period 2^256 - 1.
class Xoshiro256pp {
public:
    explicit Xoshiro256pp(std::uint64_t seed = 0x2545F4914F6CDD1DULL) { reseed(seed); }

    /// SplitMix64 to expand a single seed into the four words of state. Seeding
    /// xoshiro with a low-entropy value directly (all zeros being the extreme)
    /// leaves it correlated for thousands of draws.
    void reseed(std::uint64_t seed) {
        std::uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ULL;
            std::uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            s_[i] = x ^ (x >> 31);
        }
    }

    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    /// Uniform on (0, 1). Open at both ends, because the inverse normal CDF is
    /// infinite at 0 and 1 and a single unlucky draw would poison a whole path.
    Real uniform() {
        // 53 significant bits, shifted off zero by half an ulp.
        const std::uint64_t bits = next() >> 11;
        return (Real(bits) + 0.5) * (1.0 / 9007199254740992.0);
    }

    Real uniform(Real lo, Real hi) { return lo + (hi - lo) * uniform(); }

    /// Standard normal by inversion, not Box-Muller or ziggurat.
    ///
    /// Inversion is the only method that maps one uniform to one normal
    /// monotonically, which is what lets the same generator be swapped for a
    /// Sobol sequence without changing anything else -- and is what makes
    /// antithetic pairing exact rather than approximate.
    Real normal() { return norm_inv_cdf(uniform()); }

    /// Jump 2^128 draws ahead. Gives non-overlapping streams per thread without
    /// having to reason about whether two seeds happen to collide.
    void jump() {
        static constexpr std::uint64_t kJump[4] = {
            0x180EC6D33CFD0ABAULL, 0xD5A61266F0C9392CULL,
            0xA9582618E03FC9AAULL, 0x39ABDC4529B1661CULL};
        std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 64; ++b) {
                if (kJump[i] & (std::uint64_t(1) << b)) {
                    s0 ^= s_[0]; s1 ^= s_[1]; s2 ^= s_[2]; s3 ^= s_[3];
                }
                next();
            }
        }
        s_[0] = s0; s_[1] = s1; s_[2] = s2; s_[3] = s3;
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4] = {};
};

}  // namespace vse
