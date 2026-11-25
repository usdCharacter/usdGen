// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — the reference clump modes behind UsdGenClump (ops/clump.cpp):
// count-based random clumps, guide clumps, guide-seeded clumps, the Bezier
// root/mid/tip weight profile, flyaways, saved per-strand overrides and
// sub-clumps. These reproduce the imported groom's clump stage; every
// function is deterministic and independent of thread order.
//
// Conventions shared by every helper:
//   * Strands are addressed by CURVE index c (the buffer's order). The
//     active list `A` holds the curves in strand order (upstream
//     strandIndex plane, else curve index); every order-dependent rule
//     (first random pick, first-max farthest point, "first minimum" nearest,
//     sort tie breaks) follows A, never the curve index.
//   * `roots` / `liveRoots` are xyz per curve. Selection (seeds, nearest
//     guide / centre, override centres) reads `roots` (rest roots, stable
//     under deformation); geometry (sub-clump targets) reads `liveRoots`.
//   * Points are one flat xyz array over all CVs, curve c spanning
//     [spans[c], spans[c+1]).
#ifndef USDGEN_OPS_CLUMP_MODES_H
#define USDGEN_OPS_CLUMP_MODES_H

#include "usdGen/curveBuffer.h"
#include "usdGen/ops/strandUtil.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace clumpRef {

// ---------------------------------------------------------------------------
// Profile and flyaways
// ---------------------------------------------------------------------------

/// Root / mid / tip control values of the quadratic Bezier weight profile.
struct BezierControls
{
    float root = 0.0f;
    float mid = 0.5f;
    float midShift = 0.5f;
    float tip = 1.0f;
};

/// Reference clump weight: u = clamp01((u_i * L) / L) of point parameter
/// u_i = i/(n-1); y = Bezier((0,root),(midShift,mid),(1,tip)) at u (controls
/// clamped to [0,1], midShift to [0.1,0.9]), or `mid` when L <= 1e-6;
/// returns clamp01(y * clamp01(amount)).
float ClumpWeight(float u, float length, float amount, BezierControls const &controls);

/// Flyaway test: clamp01(fraction) > U01(SplitMix64(strandIndex)), compared
/// in double. Unseeded, so the same strand indices fly away everywhere.
bool IsFlyaway(uint64_t strandIndex, float fraction);

// ---------------------------------------------------------------------------
// Nearest-point queries
// ---------------------------------------------------------------------------

/// Squared distance between two xyz triples, float, (dx*dx + dy*dy) + dz*dz.
inline float Dist2(float const *a, float const *b)
{
    float const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

/// Nearest of a fixed candidate list with "first minimum" semantics: the
/// smallest squared distance, ties to the earliest candidate. Brute force
/// for small lists, a kd-tree (refined exactly) for large ones.
class NearestFinder
{
public:
    NearestFinder();
    ~NearestFinder();
    NearestFinder(NearestFinder &&) noexcept;
    NearestFinder &operator=(NearestFinder &&) noexcept;
    /// `points`: xyz per candidate, in candidate order.
    void Build(std::vector<float> points);
    size_t Size() const { return _points.size() / 3; }
    /// Candidate index (or -1 when empty) and its squared distance.
    int32_t Nearest(float const *q, float *d2 = nullptr) const;
private:
    struct Tree;
    std::vector<float> _points;
    std::unique_ptr<Tree> _tree;
};

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------

/// Random clump seeds (reference "Random"): K = min(count, |A|) centre
/// curves. K <= 255: the first pick is A[rng() % |A|], then farthest-point
/// sampling over A (strict '>' so the first maximum wins). K > 255: the first
/// K of an MSVC-style shuffle of A. With `partKey` (per curve, may be null)
/// and more than one key, keys are visited ascending and key group j gets
/// max(1, round(|grp| K / |A|)) seeds (the last group the remainder), the
/// farthest-point distances shared across groups. Returns curve indices.
std::vector<uint32_t> RandomSeedCenters(std::vector<uint32_t> const &A, float const *roots,
                                        int const *partKey, size_t count, std::mt19937 &rng);

/// Per-bucket sampler of the guide-seeded mode: `quota` seeds from
/// `bucket`, first = bucket[rng() % |bucket|], then farthest point with
/// distances to this call's seeds only, skipping curves already `chosen`;
/// quota >= 256 takes the first unchosen of an MSVC-style shuffle.
void SampleBucket(std::vector<uint32_t> const &bucket, size_t quota, float const *roots,
                  std::mt19937 &rng, std::vector<char> *chosen,
                  std::vector<uint32_t> *centers);

/// Guide-seeded seeds (reference "Random Experimental"): A bucketed by
/// `nearestGuide` (per curve, -1 none). Fewer non-empty guides than K: one
/// seed each, the rest greedily to max |bucket| / (quota + 1) while
/// quota < |bucket|. Otherwise K guides, the largest bucket first, then
/// greedily max sqrt(|bucket|) * min distance to the chosen guide roots.
/// Buckets are sampled in guide order with SampleBucket. Without guides:
/// the parting groups (as RandomSeedCenters' quotas), else all of A.
std::vector<uint32_t> GuideSeededCenters(std::vector<uint32_t> const &A, float const *roots,
                                         int const *nearestGuide, float const *guideRoots,
                                         size_t guideCount, int const *partKey,
                                         size_t count, std::mt19937 &rng);

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// Nearest-centre assignment.
///   groups[g]      candidate centre indices (into `centers`), ascending;
///   groupOf[pos]   the candidate group of A[pos].
/// Each strand takes its group's nearest centre (first minimum). Writes
/// clumpOf[curve] (index into `centers`).
/// The reference follows this with a capacity pass (ceil(N/K) members per
/// centre when K <= 255), but a strand leaves a full centre only for one
/// with room that is strictly nearer than its nearest centre. No such
/// centre exists, so that pass never moves a strand and is not reproduced.
void AssignCenters(std::vector<uint32_t> const &A, float const *roots,
                   std::vector<uint32_t> const &centers,
                   std::vector<std::vector<uint32_t>> const &groups,
                   std::vector<uint32_t> const &groupOf,
                   std::vector<int32_t> *clumpOf);

// ---------------------------------------------------------------------------
// Guide curves
// ---------------------------------------------------------------------------

/// Guides resampled to `ng` points uniform in arc length (linear between the
/// original guide points), sampled by arc length: linear between table
/// points at index clamp01(a/Lg)*(ng-1); past the tip the last table
/// segment's direction is extrapolated by (a - Lg); Lg <= 1e-6 gives the
/// guide root.
class GuideTable
{
public:
    /// Current positions of `guides`; ng is clamped to [2, ...].
    bool Build(UsdGenCurveBuffer const &guides, size_t ng, std::string *error);
    size_t Count() const { return _length.size(); }
    float Length(size_t g) const { return _length[g]; }
    GfVec3f Sample(size_t g, float a) const;
private:
    size_t _ng = 0;
    std::vector<float> _table;     // 3 * ng per guide
    std::vector<float> _length;
};

/// Guide-seeded centre curve: C[0] = root; C[i] = C[i-1] + (G(a_i) -
/// G(a_{i-1})) with a_i = i/(n-1) * length on guide g of the table; a
/// segment pointing against the running direction is bent back to
/// 0.25 seg + 0.75 |seg| dir, and the running direction follows
/// normalize(0.5 dir + 0.5 unit(seg)). `out` has 3n floats.
void SynthesizeCenterCurve(GuideTable const &table, size_t g, GfVec3f const &root,
                           float length, size_t n, float *out);

// ---------------------------------------------------------------------------
// Overrides and sub-clumps
// ---------------------------------------------------------------------------

/// Clump weight of CV i of curve c with n CVs, before flyaway scaling.
using WeightFn = std::function<float(uint32_t c, size_t i, size_t n)>;

/// Point i of a curve of n CVs read from another curve of m CVs at the
/// same parameter (exact when n == m).
GfVec3f SampleAtParameter(float const *p, size_t m, size_t i, size_t n);

/// Saved per-strand overrides, indexed by strandIndex (`strandIndex` per
/// curve): INT_MIN keeps the result, -1 unclumps (restore toward the
/// snapshot by clamp01(flyawayAmount): <= 1e-6 restores, < 0.999 lerps,
/// clumpId -1), v >= 0 joins custom clump v: a custom clump of more than one
/// member restores its members and pulls them toward the snapshot curve of
/// the member rooted nearest the members' mean root (weight * flyaway
/// amount for flyaways). The caller checks the saved count.
void ApplyOverrides(std::vector<uint32_t> const &A, std::vector<int> const &strandIndex,
                    VtIntArray const &overrides, float const *roots,
                    std::vector<float> const &snapshot, std::vector<float> *points,
                    std::vector<uint32_t> const &spans, std::vector<int32_t> *clumpId,
                    std::vector<char> const &flyaway, float flyawayAmount,
                    WeightFn const &weight);

struct SubClumpSettings
{
    int count = 2;
    float spread = 0.5f;
    float amount = 1.0f;
    bool ids = false;
};

/// Sub-clumps: every clump (clumpId >= 0, non-flyaway members of A, keyed
/// by clumpId) of m >= 2 members splits into min(count, m) angular sectors
/// around the mean live root; sector j's centre curve is its members' mean
/// curve moved so its root sits on the surface below c + o_j (closest point
/// on `surface`, else c + o_j) and its tip by c + o_j - root, o_j =
/// lerp(0.15, 2.7, clamp01(spread)) * rmsRadius toward the sector's mean
/// direction. Members pull toward it with weight * clamp(amount, 0.1, 1);
/// `ids` writes clumpId = key * count + j.
void ApplySubClumps(std::vector<uint32_t> const &A, float const *liveRoots,
                    std::vector<float> *points, std::vector<uint32_t> const &spans,
                    std::vector<int32_t> *clumpId, std::vector<char> const &flyaway,
                    SubClumpSettings const &settings, WeightFn const &weight,
                    strand::TriangleBvh const *surface);

}  // namespace clumpRef
}  // namespace usdGen

#endif  // USDGEN_OPS_CLUMP_MODES_H
