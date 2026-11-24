// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — whole-strand length math shared by UsdGenLength's randomRange mode
// and UsdGenTrim: the per-strand random scale, the clump-mean reference
// curves, and the "resample to n points uniformly in arc length over
// [0, target], extrapolating straight past the tip" kernel.
//
// Reference behaviour (imported grooms): the strand (or, with
// followClumpMean, its clump's mean curve) is the reference polyline R of
// length L. The target length is L * scale with
//   lo = clamp(min, 0.01, 2), hi = clamp(max, 0.01, 2), swapped if hi < lo,
//   r  = clamp(randomize, 0, 1),
//   scale = hi                       (the default target is max, not 1),
//   scale = (lo + u (hi - lo)) r + (1 - r) hi   when r > 0 and hi - lo > 1e-6,
//   u = U01(splitmix64(int64(int32(seed)))),
//   seed = clumpId_0 when shareByClump and the strand is clumped, else
//          the strand's random key (strandSeed),
//   scale *= clamp(mask, 0, 1)       (length maps sampled at the root).
// Point j lands at arc length x_j = j / (n - 1) * target along R (linear
// between points; past the tip R[last] + tip (x - L), tip = the last
// segment's direction or +Y). With followClumpMean every member keeps its
// per-index offset from the mean: Q_j += P_j - M_min(j, nM-1).
#ifndef USDGEN_OPS_LENGTH_RANGE_H
#define USDGEN_OPS_LENGTH_RANGE_H

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"

#include <cstddef>
#include <cstdint>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenWorkDispatcher;

namespace lengthRange {

struct Settings
{
    float min = 0.01f;
    float max = 1.0f;
    float randomize = 0.0f;
    bool shareByClump = false;
    bool followClumpMean = false;
};

/// The per-strand inputs of Apply(). All vectors are indexed by curve.
struct Strands
{
    std::vector<uint32_t> const *spans = nullptr;   // R + 1 CV offsets
    std::vector<int32_t> clumpId;                   // -1 = unclumped (empty: none)
    std::vector<uint32_t> seed;                     // random key per strand
    std::vector<uint32_t> order;                    // generation order (clump-mean sums)
    std::vector<float> mask;                        // [0,1] root mask (empty: 1)
};

/// The scale multiplier of one strand before the mask (see header comment).
float RandomScale(Settings const &settings, uint32_t seed);

/// Point at arc length x on polyline R (nR points, cumulative lengths cum,
/// total L): linear on the first segment k with cum[k+1] >= x, searched
/// forward from *k (monotone callers pass the previous k); past the tip
/// R[last] + tip (x - L).
GfVec3f SampleForward(float const *R, size_t nR, std::vector<float> const &cum,
                      float L, GfVec3f const &tip, float x, size_t *k);

/// Writes n points at x_j = j / (n - 1) * target along R into dst (3n floats).
void ResampleTo(float const *R, size_t nR, std::vector<float> const &cum, float L,
                GfVec3f const &tip, float target, size_t n, float *dst);

/// normalize(R[n-1] - R[n-2]), (0,1,0) when degenerate or n < 2.
GfVec3f TipDirection(float const *R, size_t n);

/// Applies the randomRange length scale to every strand of `points`
/// (3 * totalCvs, edited in place) and writes each strand's reference
/// (pre-scale) length to restLength (sized R). Order-independent: the only
/// cross-strand data (clump means) is summed in `order`.
void Apply(Settings const &settings, Strands const &strands, std::vector<float> *points,
           std::vector<float> *restLength, UsdGenWorkDispatcher *dispatcher = nullptr);

}  // namespace lengthRange
}  // namespace usdGen

#endif  // USDGEN_OPS_LENGTH_RANGE_H
