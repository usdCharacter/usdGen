// usdGenMath — pinned hash family (02-schema.md §2.19.1, R12).
//
// Header-only, no dependencies: the engine core and usdGenMath both include
// it; changing the function or any salt is a look change for every existing
// asset. Determinism (E-8) requires the exact
// SplitMix64 finalizer below.
#ifndef USDGEN_MATH_HASH_H
#define USDGEN_MATH_HASH_H

#include <cstdint>

namespace usdGen {

// SplitMix64 finalizer over the salted key.
inline uint64_t UsdGenHash64(uint64_t key, uint32_t salt)
{
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
/// Multi-key fold (02-schema.md §2.19.1): left-associative, one hash per
/// key — the documented spelling of
///   curveId = UsdGenHash64(seed, faceIndex, k, kSaltScatter).
inline uint64_t UsdGenHash64(uint64_t a, uint64_t b, uint32_t salt)
{
    return UsdGenHash64(UsdGenHash64(a, salt) ^ b, salt);
}
inline uint64_t UsdGenHash64(uint64_t a, uint64_t b, uint64_t c, uint32_t salt)
{
    return UsdGenHash64(UsdGenHash64(a, b, salt) ^ c, salt);
}
/// Truncated 32-bit hash (02-schema.md §2.19.1): the high 32 bits of the
/// 64-bit finalizer — `Hash32(key, salt) = uint32_t(Hash64(key, salt) >> 32)`.
inline uint32_t UsdGenHash32(uint64_t key, uint32_t salt)
{
    return uint32_t(UsdGenHash64(key, salt) >> 32);
}

/// Uniform float in [0,1) from the top 24 mantissa bits (§2.19.1):
/// `Hash01(key, salt) = float(Hash32(key, salt) >> 8) * 0x1.0p-24f`.
inline float UsdGenHash01(uint64_t key, uint32_t salt)
{
    return float(UsdGenHash32(key, salt) >> 8) * 0x1.0p-24f;
}

/// The call-site spelling of the per-curve draw (04-operators.md §0.6):
/// the seed enters the hash directly, so a re-seed deliberately renames
/// every curve; a density-scale / LOD / chunking change renames nothing.
inline float UsdGenDraw01(int seed, uint64_t curveId, uint32_t salt)
{
    const uint64_t key = UsdGenHash64(uint64_t(uint32_t(seed)), salt) ^ curveId;
    const uint32_t h = uint32_t(UsdGenHash64(key, salt) >> 32);
    return float(h >> 8) * 0x1.0p-24f;  // high 24 bits -> [0,1)
}

// hairId (uniform float published on tiles, ADR §1 S29): UsdGenHash32(curveId, 0)/2^32.
inline float UsdGenHairId(uint64_t curveId)
{
    const uint32_t h = uint32_t(UsdGenHash64(curveId, 0u) >> 32);
    return float(h) * 0x1.0p-32f;
}

// Per-use compile-time salts (R12; 04-operators.md §0.6 owns the per-operator
// table). kSaltDensity != 0 so the surviving set of the R13 decimation
// predicate is never {hairId < keepFraction}.
constexpr uint32_t kSaltScatter      = 0x52C4A11Eu;  // "ratt"
constexpr uint32_t kSaltDensity      = 0xDE51732Eu;  // never 0
constexpr uint32_t kSaltNoise        = 0x4E01523Eu;  // "Noiz"
constexpr uint32_t kSaltGrow         = 0x47726F77u;  // "Grow" (04 §0.6)
constexpr uint32_t kSaltGrowAzimuth  = 0x4772417Au;  // "GrAz": independent of length
constexpr uint32_t kSaltLength       = 0x4C656E67u;  // "Leng" (04 §0.6)
constexpr uint32_t kSaltMaskRandom   = 0x4D41534Eu;  // "MASK"
constexpr uint32_t kSaltClump        = 0x43174D50u;  // "CLMP"; per level use kSaltClump + level
constexpr uint32_t kSaltSculpt       = 0x53C17150u;  // "SCUL"
constexpr uint32_t kSaltScatterBary    = 0x52C4A11Fu;  // scatter barycentric draws (kSaltScatter + 1)
constexpr uint32_t kSaltScaleRandom  = 0x5343414Cu;  // "SCAL"
constexpr uint32_t kSaltCurl         = 0x4375726Cu;  // "Curl": per-curve phase draws
constexpr uint32_t kSaltBend         = 0x42656E64u;  // "Bend": per-curve angle draws
constexpr uint32_t kSaltScale        = 0x5363616Cu;  // "Scal": per-curve scale draws
constexpr uint32_t kSaltClumpLevel(int level) { return kSaltClump + uint32_t(level); }
/// Scatter curve id (02-schema.md §2.19.1):
/// `CurveId(seed, faceIndex, k) = Hash64(uint64_t(uint32_t(seed)),
///                                       uint64_t(faceIndex), uint64_t(k),
///                                       kSaltScatter)`.
inline uint64_t UsdGenCurveId(int seed, uint32_t faceIndex, uint32_t k)
{
    return UsdGenHash64(
        uint64_t(uint32_t(seed)), uint64_t(faceIndex), uint64_t(k), kSaltScatter);
}


}  // namespace usdGen

#endif  // USDGEN_MATH_HASH_H
