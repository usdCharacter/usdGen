// usdGenPomade — P4 check kernels: K12 root intersections, K13 smoothness
// (plan/17 §4.1). CPU twins + CUDA lanes share the USDGEN_POMADE_HD inlines
// below, so TN-6 parity is bit-exact (arithmetic + comparisons only, and
// CUDA builds with --fmad=false).
//
// Where the Pomade publications are silent (both kernels' exact tests), the
// math here is our own reconstruction and is documented as such:
//   * K13 scores each interior center CV 1 - cos(turning angle) (0 for a
//     straight run, -> 2 for a hairpin); ends score 0, and a CV beside a
//     zero-length segment scores 0 (degenerate, never a spike). A score
//     above kPomadeSmoothnessSpike (1 - cos(15 deg)) is a spike: gentle
//     bends stay quiet, kinks spike. RelaxCenter weights its Laplacian
//     correction per CV by a ramp over [threshold/2, threshold].
//   * K12 flags per tube whether its root ring overlaps any other tube's
//     root ring (contract 6). Broad phase: world AABBs (inclusive).
//     Narrow phase: both rings projected into the lower-id tube's root
//     chart, shrunk 1e-4 toward their vertex means, then proper edge
//     crossings or strict containment of vertices, edge midpoints, and
//     the vertex mean (the midpoints catch collinear strip overlaps,
//     the mean catches identical rings). The shrink kills resampling
//     and chart round-trip noise (~1e-7 relative) so subdivided
//     siblings, which share cut edges, read clear; touching (zero-area)
//     never flags, only positive-area overlap. Runs post-gesture, never
//     mid drag (§4.2).
#ifndef USDGEN_POMADE_CHECK_H
#define USDGEN_POMADE_CHECK_H

#include "usdGenPomade/api.h"
#include "usdGenPomade/pomadeTube.h"

#include <string>
#include <vector>

// Local spelling (pomadeTube.h/pomadeTessellate.h undef theirs at the end,
// so each header defines its own).
#ifdef __CUDACC__
#define USDGEN_POMADE_HD __host__ __device__
#else
#define USDGEN_POMADE_HD
#endif

namespace usdGenPomade {

// A K13 score at or above this is a spike (1 - cos(15 deg)).
constexpr float kPomadeSmoothnessSpike = 0.0340742f;

// Relative shrink applied to both rings before the K12 narrow phase.
constexpr float kPomadeOverlapShrink = 1e-4f;

// Max root-ring verts the narrow phase handles (the K5 cap).
constexpr int kPomadeCheckRingMax = 32;

// K13 kink score of the middle CV: 1 - cos of the turning angle between
// (p1 - p0) and (p2 - p1). Zero on any zero-length leg.
USDGEN_POMADE_HD inline float PomadeKinkScore(float const p0[3],
                                            float const p1[3],
                                            float const p2[3])
{
    float const ax = p1[0] - p0[0];
    float const ay = p1[1] - p0[1];
    float const az = p1[2] - p0[2];
    float const bx = p2[0] - p1[0];
    float const by = p2[1] - p1[1];
    float const bz = p2[2] - p1[2];
    float const la =
        sqrtf(ax * ax + ay * ay + az * az);
    float const lb =
        sqrtf(bx * bx + by * by + bz * bz);
    if (!(la > 0.0f) || !(lb > 0.0f)) {
        return 0.0f;
    }
    float c = (ax * bx + ay * by + az * bz) / (la * lb);
    c = c < -1.0f ? -1.0f : (c > 1.0f ? 1.0f : c);
    return 1.0f - c;
}

// Twice the signed area of (a, b, c): > 0 left, < 0 right, == 0 collinear.
USDGEN_POMADE_HD inline float PomadeOrient2(float ax, float ay, float bx,
                                          float by, float cx, float cy)
{
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Exact on-segment test (collinear + within the bounding box, inclusive).
USDGEN_POMADE_HD inline bool PomadeOnSegExact2(float ax, float ay, float bx,
                                             float by, float px, float py)
{
    if (PomadeOrient2(ax, ay, bx, by, px, py) != 0.0f) {
        return false;
    }
    float const loX = ax < bx ? ax : bx, hiX = ax > bx ? ax : bx;
    float const loY = ay < by ? ay : by, hiY = ay > by ? ay : by;
    return px >= loX && px <= hiX && py >= loY && py <= hiY;
}

// Proper crossing: interiors intersect (shared endpoints don't count).
USDGEN_POMADE_HD inline bool PomadeSegsCrossProper2(float ax, float ay,
                                                  float bx, float by,
                                                  float cx, float cy,
                                                  float dx, float dy)
{
    float const o1 = PomadeOrient2(ax, ay, bx, by, cx, cy);
    float const o2 = PomadeOrient2(ax, ay, bx, by, dx, dy);
    float const o3 = PomadeOrient2(cx, cy, dx, dy, ax, ay);
    float const o4 = PomadeOrient2(cx, cy, dx, dy, bx, by);
    return ((o1 > 0.0f) != (o2 > 0.0f)) && ((o1 < 0.0f) != (o2 < 0.0f)) &&
           ((o3 > 0.0f) != (o4 > 0.0f)) && ((o3 < 0.0f) != (o4 < 0.0f));
}

// Strict point-in-polygon: strictly inside, never on the boundary.
USDGEN_POMADE_HD inline bool PomadePointStrictlyInPoly2(float px, float py,
                                                      float const *u,
                                                      float const *v, int n)
{
    bool inside = false;
    for (int i = 0; i < n; ++i) {
        float const ax = u[i], ay = v[i];
        float const bx = u[(i + 1) % n], by = v[(i + 1) % n];
        if (PomadeOnSegExact2(ax, ay, bx, by, px, py)) {
            return false;
        }
        if ((ay > py) != (by > py)) {
            float const xin = ax + (py - ay) / (by - ay) * (bx - ax);
            if (px < xin) {
                inside = !inside;
            }
        }
    }
    return inside;
}

// World point into a root chart (center + N/B plane basis).
USDGEN_POMADE_HD inline void PomadeProjectChart2(
    float const w[3], float const c[3], float const n[3], float const b[3],
    float *u, float *v)
{
    float const dx = w[0] - c[0], dy = w[1] - c[1], dz = w[2] - c[2];
    *u = dx * n[0] + dy * n[1] + dz * n[2];
    *v = dx * b[0] + dy * b[1] + dz * b[2];
}

// World AABB of a ring (outMin/outMin + 3). Sequential scan, so the CPU
// twin and the device lane agree bit-exactly.
USDGEN_POMADE_HD inline void PomadeRingAabb3(float const *w, int n,
                                           float *outMin, float *outMax)
{
    outMin[0] = w[0];
    outMin[1] = w[1];
    outMin[2] = w[2];
    outMax[0] = w[0];
    outMax[1] = w[1];
    outMax[2] = w[2];
    for (int i = 1; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            float const x = w[size_t(i) * 3 + size_t(k)];
            if (x < outMin[k]) {
                outMin[k] = x;
            }
            if (x > outMax[k]) {
                outMax[k] = x;
            }
        }
    }
}

// Inclusive AABB overlap (touching counts: the narrow phase decides).
USDGEN_POMADE_HD inline bool PomadeAabbsOverlap3(float const *minA,
                                               float const *maxA,
                                               float const *minB,
                                               float const *maxB)
{
    return minA[0] <= maxB[0] && minB[0] <= maxA[0] &&
           minA[1] <= maxB[1] && minB[1] <= maxA[1] &&
           minA[2] <= maxB[2] && minB[2] <= maxA[2];
}

// Signed 2D polygon area (zero for degenerate rings).
USDGEN_POMADE_HD inline float PomadePolyArea2(float const *u, float const *v,
                                            int n)
{
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        int const j = (i + 1) % n;
        acc += u[i] * v[j] - u[j] * v[i];
    }
    return 0.5f * acc;
}

// Strict containment of one ring's sample points (every vertex, every
// edge midpoint, the vertex mean) in the other ring. The midpoints catch
// collinear strip overlaps (no proper crossing, no strictly-inside
// vertex); the mean catches identical rings.
USDGEN_POMADE_HD inline bool PomadeSamplesStrictlyInPoly2(float const *uS,
                                                        float const *vS,
                                                        int nS,
                                                        float const *uP,
                                                        float const *vP,
                                                        int nP)
{
    for (int i = 0; i < nS; ++i) {
        if (PomadePointStrictlyInPoly2(uS[i], vS[i], uP, vP, nP)) {
            return true;
        }
    }
    for (int i = 0; i < nS; ++i) {
        int const j = (i + 1) % nS;
        float const mx = (uS[i] + uS[j]) * 0.5f;
        float const my = (vS[i] + vS[j]) * 0.5f;
        if (PomadePointStrictlyInPoly2(mx, my, uP, vP, nP)) {
            return true;
        }
    }
    float su = 0.0f, sv = 0.0f;
    for (int i = 0; i < nS; ++i) {
        su += uS[i];
        sv += vS[i];
    }
    return PomadePointStrictlyInPoly2(su / float(nS), sv / float(nS), uP, vP,
                                     nP);
}

// 2D ring-vs-ring overlap: proper edge crossings or strict sample
// containment either way. Zero-area touching never reports, and a
// degenerate (zero area) ring overlaps nothing.
USDGEN_POMADE_HD inline int PomadeRingsOverlap2(float const *uA,
                                              float const *vA, int nA,
                                              float const *uB,
                                              float const *vB, int nB)
{
    if (PomadePolyArea2(uA, vA, nA) == 0.0f ||
        PomadePolyArea2(uB, vB, nB) == 0.0f) {
        return 0;
    }
    for (int i = 0; i < nA; ++i) {
        float const ax = uA[i], ay = vA[i];
        float const bx = uA[(i + 1) % nA], by = vA[(i + 1) % nA];
        for (int j = 0; j < nB; ++j) {
            float const cx = uB[j], cy = vB[j];
            float const dx = uB[(j + 1) % nB], dy = vB[(j + 1) % nB];
            if (PomadeSegsCrossProper2(ax, ay, bx, by, cx, cy, dx, dy)) {
                return 1;
            }
        }
    }
    if (PomadeSamplesStrictlyInPoly2(uA, vA, nA, uB, vB, nB)) {
        return 1;
    }
    if (PomadeSamplesStrictlyInPoly2(uB, vB, nB, uA, vA, nA)) {
        return 1;
    }
    return 0;
}

// Index of a tube id (-1 when absent). Ids should be unique; duplicates
// resolve to the first entry (deterministic either way).
USDGEN_POMADE_HD inline int PomadeIdIndex(int const *ids, int n, int id)
{
    for (int k = 0; k < n; ++k) {
        if (ids[k] == id) {
            return k;
        }
    }
    return -1;
}

// Ancestor/descendant test over (tubeId, parentTubeId) pairs: true when j
// is reachable walking up from i or vice versa (cycle-safe: bounded
// walks, -1 terminates). Related pairs nest by design and are skipped by
// K12; only unrelated tubes can intersect.
USDGEN_POMADE_HD inline bool PomadePairRelated(int const *ids,
                                             int const *parents, int n, int i,
                                             int j)
{
    for (int pass = 0; pass < 2; ++pass) {
        int a = pass == 0 ? i : j;
        int const target = pass == 0 ? j : i;
        for (int s = 0; s <= n; ++s) {
            if (a < 0 || a >= n) {
                break;
            }
            if (a == target) {
                return true;
            }
            a = PomadeIdIndex(ids, n, parents[a]);
        }
    }
    return false;
}

// K12 narrow phase for one ordered pair: both world rings projected into
// tube I's root chart, shrunk toward their vertex means, overlapped.
// Both rings round-trip through world (never chart shortcuts), so the CPU
// twin and the CUDA lane execute identical operations. Counts must be in
// [3, kPomadeCheckRingMax] (callers validate).
USDGEN_POMADE_HD inline int PomadeRootPairNarrowHit(
    float const cI[3], float const nI[3], float const bI[3],
    float const *wI, int countI, float const *wJ, int countJ)
{
    float uI[kPomadeCheckRingMax], vI[kPomadeCheckRingMax];
    float uJ[kPomadeCheckRingMax], vJ[kPomadeCheckRingMax];
    for (int k = 0; k < countI; ++k) {
        PomadeProjectChart2(wI + size_t(k) * 3, cI, nI, bI, &uI[k], &vI[k]);
    }
    for (int k = 0; k < countJ; ++k) {
        PomadeProjectChart2(wJ + size_t(k) * 3, cI, nI, bI, &uJ[k], &vJ[k]);
    }
    // Shrink toward the vertex means (noise guard, see header).
    float const keep = 1.0f - kPomadeOverlapShrink;
    {
        float su = 0.0f, sv = 0.0f;
        for (int k = 0; k < countI; ++k) {
            su += uI[k];
            sv += vI[k];
        }
        float const mu = su / float(countI), mv = sv / float(countI);
        for (int k = 0; k < countI; ++k) {
            uI[k] = mu + (uI[k] - mu) * keep;
            vI[k] = mv + (vI[k] - mv) * keep;
        }
    }
    {
        float su = 0.0f, sv = 0.0f;
        for (int k = 0; k < countJ; ++k) {
            su += uJ[k];
            sv += vJ[k];
        }
        float const mu = su / float(countJ), mv = sv / float(countJ);
        for (int k = 0; k < countJ; ++k) {
            uJ[k] = mu + (uJ[k] - mu) * keep;
            vJ[k] = mv + (vJ[k] - mv) * keep;
        }
    }
    return PomadeRingsOverlap2(uI, vI, countI, uJ, vJ, countJ);
}

#ifndef __CUDA_ARCH__

// -- K13: per-CV kink scores ---------------------------------------------
// `out` takes nCv scores (ends 0). Needs >= 1 CV (a lone CV scores 0).
USDGENPOMADE_API bool PomadeSmoothnessScoreCpu(float const *cx, float const *cy,
                             float const *cz, int nCv, float *out,
                             std::string *err);

// -- K12: root-ring overlap flags ----------------------------------------
// Root chart of one tube: the root section plane (center + N/B) plus the
// root ring placed in world coords (3 floats per vert, ringVerts of
// them). `frames` are the K4 frames at the tube's center CVs.
USDGENPOMADE_API bool PomadeRootChartCpu(PomadeTubeDesc const &tube,
                       std::vector<PomadeFrame> const &frames, float center[3],
                       float nrm[3], float bin[3],
                       std::vector<float> *worldRing, std::string *err);

// One ordered pair (J projected into I's chart): *outHit is the narrow
// verdict, *broadHit the world-AABB precheck (either may be null to skip
// that phase). Counts must be in [3, kPomadeCheckRingMax].
USDGENPOMADE_API bool PomadeRootPairOverlapCpu(float const centerI[3], float const nrmI[3],
                             float const binI[3], float const *worldRingI,
                             int countI, float const *worldRingJ, int countJ,
                             int *outHit, int *broadHit, std::string *err);

// All tubes: outFlags[i] is 1 when tube i's root overlaps another tube's
// root (positive area), else 0. Ancestor/descendant pairs are skipped
// (nesting by design); zero tubes is a no-op success.
USDGENPOMADE_API bool PomadeTubeIntersectCpu(PomadeTubeDesc const *tubes, int tubeCount,
                           int *outFlags, std::string *err);

#endif  // !__CUDA_ARCH__

}  // namespace usdGenPomade

#undef USDGEN_POMADE_HD

#endif  // USDGEN_POMADE_CHECK_H
