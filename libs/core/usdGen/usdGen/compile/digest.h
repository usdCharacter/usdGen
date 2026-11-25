// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — internal multi-lane digest for capture-class hashing.
//
// FNV-1a is one serial xor/multiply chain per byte, so a single lane runs
// at multiply latency (~2-4 cycles/byte). Four independent lanes over
// striped inputs let the CPU overlap four chains and run at ~4x the
// throughput; UsdGenDigestCombine4 folds the lanes order-sensitively.
//
// Contract: digests are INTERNAL cache keys, compared for equality only
// (scheduler capture epochs, cook digests). Values are NOT stable across
// usdGen versions and must never be persisted or golden-tested — only
// determinism (same bytes -> same digest in every process) and input
// sensitivity are guaranteed. Groom outputs are unaffected by the lane
// count.
#ifndef USDGEN_DIGEST_H
#define USDGEN_DIGEST_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace usdGen {

constexpr uint64_t UsdGenDigestOffset = 14695981039346656037ull;
constexpr uint64_t UsdGenDigestPrime = 1099511628211ull;

// One word feed: xor the whole word, then the FNV prime (one multiply
// instead of eight — xor-then-multiply-by-odd is bijective, so every
// input bit still flips the lane). Values are internal keys (see
// above), never persisted: the word feed is endian-sensitive.
inline void UsdGenDigestMixWord(uint64_t &h, uint64_t word)
{
    h ^= word;
    h *= UsdGenDigestPrime;
}

// Order-sensitive 4-lane fold with a splitmix-style final avalanche.
// Non-crypto, as FNV-1a itself is.
inline uint64_t UsdGenDigestCombine4(uint64_t h0, uint64_t h1, uint64_t h2,
                                     uint64_t h3)
{
    uint64_t h = UsdGenDigestOffset;
    h ^= h0 + 0x9E3779B97F4A7C15ull;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h1 + 0xBF58476D1CE4E5B9ull;
    h *= 0x94D049BB133111EBull;
    h ^= h2 + 0x94D049BB133111EBull;
    h *= 0x9E3779B97F4A7C15ull;
    h ^= h3 + 0xD1B54A32D192ED03ull;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return h;
}

// 4-lane word-wise FNV over raw bytes: bulk words feed lane (w & 3) a
// whole 64-bit word per multiply (8x fewer multiplies than the byte
// loop at the same per-byte sensitivity — xor-then-multiply-by-odd is
// bijective, so every input byte still flips its lane), tail bytes
// feed byte-wise in order. Lanes start domain-separated from `seed`
// so empty/short inputs still mix. Values are internal keys (see
// above), never persisted: the word feed is endian-sensitive.
inline uint64_t UsdGenDigestBytes(void const *data, size_t n, uint64_t seed)
{
    auto const *b = static_cast<unsigned char const *>(data);
    uint64_t h[4] = {seed, seed ^ 0x9E3779B97F4A7C15ull,
                     seed ^ 0xBF58476D1CE4E5B9ull,
                     seed ^ 0x94D049BB133111EBull};
    size_t const nWords = n / 8;
    size_t w = 0;
    size_t const n4 = nWords & ~size_t(3);
    for (; w < n4; w += 4) {
        uint64_t w0, w1, w2, w3;
        std::memcpy(&w0, b + w * 8, 8);
        std::memcpy(&w1, b + (w + 1) * 8, 8);
        std::memcpy(&w2, b + (w + 2) * 8, 8);
        std::memcpy(&w3, b + (w + 3) * 8, 8);
        h[0] ^= w0;
        h[0] *= UsdGenDigestPrime;
        h[1] ^= w1;
        h[1] *= UsdGenDigestPrime;
        h[2] ^= w2;
        h[2] *= UsdGenDigestPrime;
        h[3] ^= w3;
        h[3] *= UsdGenDigestPrime;
    }
    for (; w < nWords; ++w) {
        uint64_t word;
        std::memcpy(&word, b + w * 8, 8);
        h[w & 3] ^= word;
        h[w & 3] *= UsdGenDigestPrime;
    }
    size_t lane = nWords & 3;
    for (size_t i = nWords * 8; i < n; ++i) {
        h[lane] ^= uint64_t(b[i]);
        h[lane] *= UsdGenDigestPrime;
        lane = (lane + 1) & 3;
    }
    return UsdGenDigestCombine4(h[0], h[1], h[2], h[3]);
}

}  // namespace usdGen

#endif  // USDGEN_DIGEST_H
