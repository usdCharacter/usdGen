// usdGen — UsdGenCollideOp implementation. 02-schema.md §2.8, 04 §4.
//
// Iterative push-out from collider meshes; see collide.h for the contract.
// Transport mirrors the bound surface, not a map: the builders append the
// usdGen:colliders targets to the Collide node's surfaces AFTER the inherited
// bound surface, so front/root-surface semantics stay intact for every other
// consumer (compiler ctx.surface, RootSurface, scatter) while this kernel
// collides against the whole list — 02 §1's "out of colliders and the skin".
// Only deformed points are read; worldMatrix is ignored exactly like
// UsdGenScatter (post-flattening groom space, S4).
//
// Closest-point search runs over fan-triangulated faces through a
// Morton-LBVH per collider (plan/04 §4 cost C), rebuilt by Capture every
// surface epoch; colliders at or under the brute-force threshold keep the
// brute-force loop. Both paths share the triangle test and the
// (collider, face, fan) ordinal tiebreak, so every visit order reproduces
// the deterministic first-win in node order / ascending faces, in double
// precision, with no allocation in Evaluate. Capture resolves and validates
// every collider mesh; an animated collider re-captures (surfaceGeneration
// digest terms) and re-evaluates every cook with no stale geometry.
// Value-class edits (offset/pushAmount/iterations)
// sweep without recapturing (displace precedent), so Evaluate additionally
// sanitizes: a non-finite connected value reads as 0, iterations < 1 copies
// the input, and an unknown resolveType falls back to flexible. Capture
// still hard-errors on every one of those states; the Evaluate fallbacks
// only cover a literal edited after capture and can never write NaN.
#include "usdGen/ops/collide.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

const TfToken sOffset{"offset"}, sPushAmount{"pushAmount"};
const TfToken sIterations{"iterations"}, sResolveType{"resolveType"};
const TfToken sFlexible{"flexible"}, sStiff{"stiff"}, sMask{"mask"};

struct CollideTri
{
    double ax, ay, az, bx, by, bz, cx, cy, cz;  // float verts widened once
    int face = 0;  // ascending-face ordinal (first-win major)
    int k = 0;     // fan-triangle index within the face (first-win minor)
};

struct CollideBvhNode
{
    double minX, minY, minZ, maxX, maxY, maxZ;  // exact float-vertex bounds
    int left = -1, right = -1;  // interior children, else a leaf run
    int begin = 0, end = 0;     // leaf run over the Morton-ordered tris
};

struct CollideBvh
{
    std::vector<CollideTri> tris;      // Morton order; leaf runs index this
    std::vector<CollideBvhNode> nodes;
    int root = -1;
};

struct UsdGenCollideCollider
{
    uint32_t geometry = 0;      // desc.surfaces index with points/topology
    std::vector<int> faces;     // empty == whole mesh, else sorted unique ids
    CollideBvh bvh;             // valid only when useBvh
    bool useBvh = false;
};

std::atomic<bool> sForceBrute{false};
std::atomic<size_t> sBruteThreshold{16};
struct BruteEnvInit
{
    BruteEnvInit()
    {
        if (std::getenv("USDGEN_COLLIDE_BRUTE"))
            sForceBrute.store(true, std::memory_order_relaxed);
    }
};
BruteEnvInit sBruteEnvInit;

struct UsdGenCollideCapture final : public UsdGenCapture
{
    std::vector<UsdGenCollideCollider> colliders;  // node.surfaces order
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenCollideCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

bool Finite(double v) { return std::isfinite(v); }
double Sanitize(double v) { return Finite(v) ? v : 0.0; }

double NowMs()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Phase-probe registry: one thread-local accumulator per worker thread,
// registered on first probe use (the lock runs once per thread, never on
// the hot path) and unregistered at thread exit, so Read/Reset from any
// thread sums exactly the live workers.
struct ProbeRegistry
{
    std::mutex mutex;
    std::vector<UsdGenCollidePhaseStats *> stats;
};
ProbeRegistry &Registry()
{
    static ProbeRegistry registry;
    return registry;
}
thread_local UsdGenCollidePhaseStats tProbeStats;
struct ProbeRegistrar
{
    ProbeRegistrar()
    {
        ProbeRegistry &r = Registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        r.stats.push_back(&tProbeStats);
    }
    ~ProbeRegistrar()
    {
        ProbeRegistry &r = Registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        auto &v = r.stats;
        v.erase(std::remove(v.begin(), v.end(), &tProbeStats), v.end());
    }
};
thread_local ProbeRegistrar tProbeRegistrar;
std::atomic<bool> sProbeOn{false};

UsdGenCollidePhaseStats *ProbeForThisThread()
{
    if (!sProbeOn.load(std::memory_order_relaxed)) return nullptr;
    (void)tProbeRegistrar;  // register this worker on first probe use
    return &tProbeStats;
}

// Unit face normal of a non-degenerate triangle; +Z when degenerate (the
// winning triangle is non-degenerate by construction, so the fallback is
// unreachable — it only keeps the hot path total).
GfVec3d UnitNormal(GfVec3d const &a, GfVec3d const &b, GfVec3d const &c)
{
    GfVec3d const n = GfCross(b - a, c - a);
    double const l = n.GetLength();
    return (l > 0.0 && Finite(l)) ? n / l : GfVec3d(0.0, 0.0, 1.0);
}

bool FindSurface(UsdGenGraphDesc const *desc, SdfPath const &path,
                 uint32_t *index)
{
    if (!desc) return false;
    for (uint32_t i = 0; i < desc->surfaces.size(); ++i) {
        if (desc->surfaces[i].path == path) {
            *index = i;
            return true;
        }
    }
    return false;
}

// The ClosestOnTriangle degeneracy verdict, shared with the BVH flattener
// so both paths skip exactly the same triangles.
bool DegenerateTri(GfVec3d const &a, GfVec3d const &b, GfVec3d const &c)
{
    GfVec3d const n = GfCross(b - a, c - a);
    double const len = n.GetLength();
    return !(len > 0.0) || !Finite(len);
}

// Ericson 5.1.5 closest point on triangle abc. False when the triangle is
// degenerate (zero area); the caller skips those. Non-degenerate edges are
// all nonzero, so the three interior divisions are exact and safe. The BVH
// path instantiates CheckDegenerate=false: the flattener already skipped
// every degenerate with this same predicate over the same widened doubles,
// so the check provably never fires there and skipping it only drops a
// cross product plus a square root per test.
template <bool CheckDegenerate>
bool ClosestOnTriangleT(GfVec3d const &p, GfVec3d const &a, GfVec3d const &b,
                        GfVec3d const &c, GfVec3d *closest, double *dist2)
{
    if constexpr (CheckDegenerate) {
        if (DegenerateTri(a, b, c)) return false;
    }
    GfVec3d const ab = b - a, ac = c - a, ap = p - a;
    double const d1 = GfDot(ab, ap), d2 = GfDot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        *closest = a;
    } else {
        GfVec3d const bp = p - b;
        double const d3 = GfDot(ab, bp), d4 = GfDot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) {
            *closest = b;
        } else {
            double const vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                *closest = a + ab * (d1 / (d1 - d3));
            } else {
                GfVec3d const cp = p - c;
                double const d5 = GfDot(ab, cp), d6 = GfDot(ac, cp);
                if (d6 >= 0.0 && d5 <= d6) {
                    *closest = c;
                } else {
                    double const vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        *closest = a + ac * (d2 / (d2 - d6));
                    } else {
                        double const va = d3 * d6 - d5 * d4;
                        double const d43 = d4 - d3, d56 = d5 - d6;
                        if (va <= 0.0 && d43 >= 0.0 && d56 >= 0.0) {
                            *closest = b + (c - b) * (d43 / (d43 + d56));
                        } else {
                            double const denom = 1.0 / (va + vb + vc);
                            *closest = a + ab * (vb * denom) + ac * (vc * denom);
                        }
                    }
                }
            }
        }
    }
    GfVec3d const d = p - *closest;
    *dist2 = GfDot(d, d);
    return true;
}

bool ClosestOnTriangle(GfVec3d const &p, GfVec3d const &a, GfVec3d const &b,
                       GfVec3d const &c, GfVec3d *closest, double *dist2)
{
    return ClosestOnTriangleT<true>(p, a, b, c, closest, dist2);
}

struct ClosestHit
{
    bool found = false;
    GfVec3d q, a, b, c;   // closest point + its winning corners (for dist 0)
    double dist2 = 0.0;
    int collider = -1, face = -1, k = -1;  // winner's first-win ordinal
};

// Strictly-less wins; exact ties (same double bits, e.g. a shared vertex
// evaluated from either triangle) break by (collider, face, fan) ordinal,
// so any visit order — brute-force ascending, BVH Morton, or mixed —
// reproduces the brute-force first-win exactly.
bool BetterHit(double d2, int collider, int face, int k,
               ClosestHit const &best)
{
    if (!best.found || d2 < best.dist2) return true;
    if (d2 != best.dist2) return false;
    if (collider != best.collider) return collider < best.collider;
    if (face != best.face) return face < best.face;
    return k < best.k;
}

// Brute-force closest point over one collider: node order, ascending faces,
// first win on ties. The ordinal tiebreak never fires here (ascending visit
// order), but tracking it lets BVH colliders compose with brute-force ones
// bit-exactly. Reads live points through captured indices; never allocates.
// A hit with non-finite distance is impossible after Capture validated the
// meshes (the desc is immutable within a run); the guard below keeps it from
// ever producing NaN output regardless.
void BruteCollider(UsdGenSurfaceDesc const &surf,
                   UsdGenCollideCollider const &collider, int colliderIdx,
                   GfVec3d const &p, ClosestHit *best,
                   UsdGenCollidePhaseStats *probe)
{
    size_t const F = surf.faceVertexCounts.size();
    if (!surf.points.empty() && !surf.faceVertexIndices.empty()) {
        size_t fi = 0;  // merge cursor over the sorted restricted faces
        size_t off = 0;
        size_t const Nfvi = surf.faceVertexIndices.size();
        for (size_t f = 0; f < F; ++f) {
            int const corners = surf.faceVertexCounts[f];
            if (corners < 0 || off + size_t(corners) > Nfvi) break;
            bool const use = collider.faces.empty()
                ? true
                : (fi < collider.faces.size() && size_t(collider.faces[fi]) == f);
            if (use) {
                ++fi;
                if (corners >= 3) {
                    int const v0 = surf.faceVertexIndices[off];
                    for (int k = 1; k + 1 < corners; ++k) {
                        int const v1 = surf.faceVertexIndices[off + size_t(k)];
                        int const v2 = surf.faceVertexIndices[off + size_t(k) + 1];
                        if (v0 < 0 || v1 < 0 || v2 < 0 ||
                            size_t(v0) >= surf.points.size() ||
                            size_t(v1) >= surf.points.size() ||
                            size_t(v2) >= surf.points.size())
                            continue;
                        GfVec3d const a(surf.points[size_t(v0)]);
                        GfVec3d const b(surf.points[size_t(v1)]);
                        GfVec3d const c(surf.points[size_t(v2)]);
                        GfVec3d q;
                        double d2 = 0.0;
                        if (probe) ++probe->triTests;
                        if (!ClosestOnTriangle(p, a, b, c, &q, &d2)) continue;
                        if (!(d2 >= 0.0)) continue;
                        if (BetterHit(d2, colliderIdx, int(f), k, *best)) {
                            best->found = true;
                            best->q = q;
                            best->a = a;
                            best->b = b;
                            best->c = c;
                            best->dist2 = d2;
                            best->collider = colliderIdx;
                            best->face = int(f);
                            best->k = k;
                        }
                    }
                }
            }
            off += size_t(corners);
        }
    }
}

// Morton helpers: 10 bits per axis over the centroid bbox (the Pomade K1
// spelling). Codes only steer the tree shape, never the result.
uint32_t ExpandBits10(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint32_t MortonCode(double x, double y, double z, double const lo[3],
                    double const scale[3])
{
    auto q = [](double t) -> uint32_t {
        if (!(t > 0.0)) return 0;
        if (!(t < 1.0)) return 1023;
        uint32_t const v = uint32_t(t * 1024.0);
        return v < 1023 ? v : 1023;
    };
    return (ExpandBits10(q((x - lo[0]) * scale[0])) << 2) |
           (ExpandBits10(q((y - lo[1]) * scale[1])) << 1) |
           ExpandBits10(q((z - lo[2]) * scale[2]));
}

// Per-triangle centroid + bounds, computed once per build: the SAH levels
// below re-read them instead of recomputing centroids 4x and bounds 3x
// per level. The cached values are the same expressions' doubles, so
// every bin, cost, and partition — and the whole tree — is bit-identical
// to the recompute. Indexed by flatten-order triangle; transient to the
// build (leaf bounds still come from FitLeaf's exact per-vertex walk).
struct BvhTriCache
{
    double cx, cy, cz;
    double mnx, mny, mnz, mxx, mxy, mxz;
};

// Binned-SAH tree over the Morton-sorted tris (the Morton order seeds
// spatially coherent spans; SAH picks the tightest split per span, which
// the Karras prefix rule cannot see). A split past the depth cap, over a
// degenerate span, or with an empty side falls back to the midpoint
// (halving every level), so the tree depth is bounded by construction
// (48 SAH + 32 halving levels) and the 256-entry traversal stack below
// provably suffices. Shape only guides pruning; the ordinal tiebreak keeps
// every shape bit-identical. Build is deterministic (fixed Morton order,
// deterministic partition), so rebuilt captures reproduce exactly.
class BvhBuilder
{
public:
    BvhBuilder(CollideBvh *bvh, std::vector<CollideTri> const *srcTris,
               std::vector<int> *perm, std::vector<BvhTriCache> const *cache)
        : _bvh(bvh), _srcTris(srcTris), _perm(perm), _cache(cache) {}

    // [l, r) over _perm; returns the node. The span's centroid extent
    // arrives from the parent's partition (haveExtents) instead of a fresh
    // measuring pass; the root and midpoint-fallback spans measure their
    // own (haveExtents false, clo/chi ignored).
    int Build(int l, int r, int depth, double const clo[3], double const chi[3],
              bool haveExtents)
    {
        int const node = int(_bvh->nodes.size());
        _bvh->nodes.push_back(CollideBvhNode());
        // Leaf capacity 2 (measured optimum: tri tests dominate visits, so
        // the smaller leaf's -40% tests beat its +7% visits; 1 loses on
        // depth, 4+ lose on tests — see the round-4 leaf sweep).
        if (r - l <= 2) {
            CollideBvhNode &leaf = _bvh->nodes[size_t(node)];
            leaf.begin = l;
            leaf.end = r;
            FitLeaf(&leaf, l, r);
            return node;
        }
        double cloL[3], chiL[3], cloR[3], chiR[3];
        int m = -1;
        bool haveChild = false;
        if (depth < 48) {
            m = SahSplit(l, r, clo, chi, haveExtents,
                         cloL, chiL, cloR, chiR);
            haveChild = m >= 0;
        }
        if (m < 0) m = (l + r) / 2;  // [l, m) + [m, r)
        int const left = Build(l, m, depth + 1, cloL, chiL, haveChild);
        int const right = Build(m, r, depth + 1, cloR, chiR, haveChild);
        CollideBvhNode &interior = _bvh->nodes[size_t(node)];
        interior.left = left;
        interior.right = right;
        CollideBvhNode const &a = _bvh->nodes[size_t(left)];
        CollideBvhNode const &b = _bvh->nodes[size_t(right)];
        interior.minX = std::min(a.minX, b.minX);
        interior.minY = std::min(a.minY, b.minY);
        interior.minZ = std::min(a.minZ, b.minZ);
        interior.maxX = std::max(a.maxX, b.maxX);
        interior.maxY = std::max(a.maxY, b.maxY);
        interior.maxZ = std::max(a.maxZ, b.maxZ);
        return node;
    }

private:
    static double BoundsArea(double const mn[3], double const mx[3])
    {
        double const dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
        return 2.0 * (dx * dy + dy * dz + dz * dx);
    }

    // Best binned-SAH split of [l, r): 16 bins per axis over the span's
    // centroid bounds, sweeping area x count (the parent area is constant
    // across candidates, so it drops out). Partitions _perm in place and
    // returns the split point, or -1 when the span is degenerate (no
    // centroid extent on any axis, or every candidate has an empty side).
    // The span extent arrives inherited (haveExtents) except at the root
    // and under midpoint fallbacks, which measure their own with the same
    // pass as before; the partition accumulates both children's extents
    // over the cache entries it already loads. Same member sets feed the
    // same min/max selections, so every split — and the whole tree — is
    // bit-identical to the measuring pass (the accumulation order can only
    // swap -0.0 for +0.0, and min/max share that order, so ext is never
    // -0.0 and every int() bin matches).
    int SahSplit(int l, int r, double const cloIn[3], double const chiIn[3],
                 bool haveExtents, double cloL[3], double chiL[3],
                 double cloR[3], double chiR[3])
    {
        double clo[3], chi[3];
        if (haveExtents) {
            for (int a = 0; a < 3; ++a) {
                clo[a] = cloIn[a];
                chi[a] = chiIn[a];
            }
        } else {
            clo[0] = clo[1] = clo[2] = HUGE_VAL;
            chi[0] = chi[1] = chi[2] = -HUGE_VAL;
            for (int i = l; i < r; ++i) {
                BvhTriCache const &e =
                    (*_cache)[size_t((*_perm)[size_t(i)])];
                clo[0] = std::min(clo[0], e.cx);
                clo[1] = std::min(clo[1], e.cy);
                clo[2] = std::min(clo[2], e.cz);
                chi[0] = std::max(chi[0], e.cx);
                chi[1] = std::max(chi[1], e.cy);
                chi[2] = std::max(chi[2], e.cz);
            }
        }
        struct Bin
        {
            double mn[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
            double mx[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
            int n = 0;
        };
        // One bin-fill pass feeds all three axes (the old spelling streamed
        // the span once per axis). Each axis's bins accumulate the same
        // entries in the same order with the same index expressions, so
        // every bin — and the whole tree — is bit-identical with ~2/3
        // fewer span loads. Degenerate axes stay skipped exactly as before
        // (haveAxis is the same `ext > 0.0` verdict, NaN included).
        Bin allBins[3][16];
        bool haveAxis[3] = {false, false, false};
        double axisScale[3] = {0.0, 0.0, 0.0};
        for (int axis = 0; axis < 3; ++axis) {
            double const ext = chi[axis] - clo[axis];
            if (ext > 0.0) {
                haveAxis[axis] = true;
                axisScale[axis] = 16.0 / ext;
            }
        }
        for (int i = l; i < r; ++i) {
            BvhTriCache const &e =
                (*_cache)[size_t((*_perm)[size_t(i)])];
            if (haveAxis[0]) {
                int b = int((e.cx - clo[0]) * axisScale[0]);
                if (b < 0) b = 0;
                else if (b > 15) b = 15;
                Bin &bin = allBins[0][b];
                bin.mn[0] = std::min(bin.mn[0], e.mnx);
                bin.mn[1] = std::min(bin.mn[1], e.mny);
                bin.mn[2] = std::min(bin.mn[2], e.mnz);
                bin.mx[0] = std::max(bin.mx[0], e.mxx);
                bin.mx[1] = std::max(bin.mx[1], e.mxy);
                bin.mx[2] = std::max(bin.mx[2], e.mxz);
                ++bin.n;
            }
            if (haveAxis[1]) {
                int b = int((e.cy - clo[1]) * axisScale[1]);
                if (b < 0) b = 0;
                else if (b > 15) b = 15;
                Bin &bin = allBins[1][b];
                bin.mn[0] = std::min(bin.mn[0], e.mnx);
                bin.mn[1] = std::min(bin.mn[1], e.mny);
                bin.mn[2] = std::min(bin.mn[2], e.mnz);
                bin.mx[0] = std::max(bin.mx[0], e.mxx);
                bin.mx[1] = std::max(bin.mx[1], e.mxy);
                bin.mx[2] = std::max(bin.mx[2], e.mxz);
                ++bin.n;
            }
            if (haveAxis[2]) {
                int b = int((e.cz - clo[2]) * axisScale[2]);
                if (b < 0) b = 0;
                else if (b > 15) b = 15;
                Bin &bin = allBins[2][b];
                bin.mn[0] = std::min(bin.mn[0], e.mnx);
                bin.mn[1] = std::min(bin.mn[1], e.mny);
                bin.mn[2] = std::min(bin.mn[2], e.mnz);
                bin.mx[0] = std::max(bin.mx[0], e.mxx);
                bin.mx[1] = std::max(bin.mx[1], e.mxy);
                bin.mx[2] = std::max(bin.mx[2], e.mxz);
                ++bin.n;
            }
        }
        int bestAxis = -1, bestBin = -1;
        double bestCost = HUGE_VAL;
        for (int axis = 0; axis < 3; ++axis) {
            if (!haveAxis[axis]) continue;
            Bin (&bins)[16] = allBins[axis];
            double pmn[16][3], pmx[16][3];
            double smn[16][3], smx[16][3];
            int pcount[16], scount[16];
            double amn[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
            double amx[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
            int acc = 0;
            for (int b = 0; b < 16; ++b) {
                if (bins[b].n) {
                    acc += bins[b].n;
                    for (int a = 0; a < 3; ++a) {
                        amn[a] = std::min(amn[a], bins[b].mn[a]);
                        amx[a] = std::max(amx[a], bins[b].mx[a]);
                    }
                }
                pcount[b] = acc;
                for (int a = 0; a < 3; ++a) {
                    pmn[b][a] = amn[a];
                    pmx[b][a] = amx[a];
                }
            }
            amn[0] = amn[1] = amn[2] = HUGE_VAL;
            amx[0] = amx[1] = amx[2] = -HUGE_VAL;
            acc = 0;
            for (int b = 15; b >= 0; --b) {
                if (bins[b].n) {
                    acc += bins[b].n;
                    for (int a = 0; a < 3; ++a) {
                        amn[a] = std::min(amn[a], bins[b].mn[a]);
                        amx[a] = std::max(amx[a], bins[b].mx[a]);
                    }
                }
                scount[b] = acc;
                for (int a = 0; a < 3; ++a) {
                    smn[b][a] = amn[a];
                    smx[b][a] = amx[a];
                }
            }
            for (int s = 0; s < 15; ++s) {
                if (!pcount[s] || !scount[s + 1]) continue;
                double const cost =
                    BoundsArea(pmn[s], pmx[s]) * pcount[s] +
                    BoundsArea(smn[s + 1], smx[s + 1]) * scount[s + 1];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBin = s;
                }
            }
        }
        if (bestAxis < 0) return -1;
        double const scale = 16.0 / (chi[bestAxis] - clo[bestAxis]);
        double const lo = clo[bestAxis];
        cloL[0] = cloL[1] = cloL[2] = HUGE_VAL;
        chiL[0] = chiL[1] = chiL[2] = -HUGE_VAL;
        cloR[0] = cloR[1] = cloR[2] = HUGE_VAL;
        chiR[0] = chiR[1] = chiR[2] = -HUGE_VAL;
        auto it = std::partition(
            _perm->begin() + l, _perm->begin() + r, [&](int tri) {
                BvhTriCache const &e = (*_cache)[size_t(tri)];
                double const c =
                    bestAxis == 0 ? e.cx : bestAxis == 1 ? e.cy : e.cz;
                int b = int((c - lo) * scale);
                if (b < 0) b = 0;
                else if (b > 15) b = 15;
                if (b <= bestBin) {
                    cloL[0] = std::min(cloL[0], e.cx);
                    cloL[1] = std::min(cloL[1], e.cy);
                    cloL[2] = std::min(cloL[2], e.cz);
                    chiL[0] = std::max(chiL[0], e.cx);
                    chiL[1] = std::max(chiL[1], e.cy);
                    chiL[2] = std::max(chiL[2], e.cz);
                    return true;
                }
                cloR[0] = std::min(cloR[0], e.cx);
                cloR[1] = std::min(cloR[1], e.cy);
                cloR[2] = std::min(cloR[2], e.cz);
                chiR[0] = std::max(chiR[0], e.cx);
                chiR[1] = std::max(chiR[1], e.cy);
                chiR[2] = std::max(chiR[2], e.cz);
                return false;
            });
        int const m = int(it - _perm->begin());
        return (m == l || m == r) ? -1 : m;
    }

    void FitLeaf(CollideBvhNode *leaf, int l, int r)
    {
        bool firstTri = true;
        for (int i = l; i < r; ++i) {
            CollideTri const &t = (*_srcTris)[size_t((*_perm)[size_t(i)])];
            double const xs[3] = {t.ax, t.bx, t.cx};
            double const ys[3] = {t.ay, t.by, t.cy};
            double const zs[3] = {t.az, t.bz, t.cz};
            for (int v = 0; v < 3; ++v) {
                if (firstTri && v == 0) {
                    leaf->minX = leaf->maxX = xs[0];
                    leaf->minY = leaf->maxY = ys[0];
                    leaf->minZ = leaf->maxZ = zs[0];
                    firstTri = false;
                } else {
                    leaf->minX = std::min(leaf->minX, xs[v]);
                    leaf->minY = std::min(leaf->minY, ys[v]);
                    leaf->minZ = std::min(leaf->minZ, zs[v]);
                    leaf->maxX = std::max(leaf->maxX, xs[v]);
                    leaf->maxY = std::max(leaf->maxY, ys[v]);
                    leaf->maxZ = std::max(leaf->maxZ, zs[v]);
                }
            }
        }
    }

    CollideBvh *_bvh;
    std::vector<CollideTri> const *_srcTris;  // flatten order (pre-permute)
    std::vector<int> *_perm;                  // Morton sort permutation
    std::vector<BvhTriCache> const *_cache;   // centroid+bounds per tri
};

void BuildColliderBvh(std::vector<CollideTri> *tris, CollideBvh *bvh)
{
    bvh->tris.clear();
    bvh->nodes.clear();
    bvh->root = -1;
    size_t const n = tris->size();
    if (!n) return;
    // Centroid + bounds once per triangle; the Morton normalization and
    // every SAH level re-read these instead of recomputing them.
    std::vector<BvhTriCache> cache(n);
    for (size_t i = 0; i < n; ++i) {
        CollideTri const &t = (*tris)[i];
        BvhTriCache &e = cache[i];
        e.cx = (t.ax + t.bx + t.cx) / 3.0;
        e.cy = (t.ay + t.by + t.cy) / 3.0;
        e.cz = (t.az + t.bz + t.cz) / 3.0;
        e.mnx = std::min(t.ax, std::min(t.bx, t.cx));
        e.mny = std::min(t.ay, std::min(t.by, t.cy));
        e.mnz = std::min(t.az, std::min(t.bz, t.cz));
        e.mxx = std::max(t.ax, std::max(t.bx, t.cx));
        e.mxy = std::max(t.ay, std::max(t.by, t.cy));
        e.mxz = std::max(t.az, std::max(t.bz, t.cz));
    }
    // Centroid bbox for the Morton normalization (shape only). The seed
    // stays the first triangle's vertex A, exactly as before.
    double lo[3] = {(*tris)[0].ax, (*tris)[0].ay, (*tris)[0].az};
    double hi[3] = {lo[0], lo[1], lo[2]};
    for (BvhTriCache const &e : cache) {
        lo[0] = std::min(lo[0], e.cx); hi[0] = std::max(hi[0], e.cx);
        lo[1] = std::min(lo[1], e.cy); hi[1] = std::max(hi[1], e.cy);
        lo[2] = std::min(lo[2], e.cz); hi[2] = std::max(hi[2], e.cz);
    }
    double scale[3];
    for (int a = 0; a < 3; ++a)
        scale[a] = hi[a] > lo[a] ? 1.0 / (hi[a] - lo[a]) : 0.0;
    std::vector<uint32_t> codes(n);
    std::vector<int> perm(n);
    for (size_t i = 0; i < n; ++i) {
        BvhTriCache const &e = cache[i];
        codes[i] = MortonCode(e.cx, e.cy, e.cz, lo, scale);
        perm[i] = int(i);
    }
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (codes[size_t(a)] != codes[size_t(b)])
            return codes[size_t(a)] < codes[size_t(b)];
        return a < b;
    });
    bvh->nodes.reserve(2 * n);
    BvhBuilder builder(bvh, tris, &perm, &cache);
    double clo[3] = {0.0, 0.0, 0.0}, chi[3] = {0.0, 0.0, 0.0};
    bvh->root = builder.Build(0, int(n), 0, clo, chi, false);
    // Reorder the triangles into Morton order so leaf runs index them
    // directly (ordinals travel with each triangle).
    std::vector<CollideTri> ordered(n);
    for (size_t i = 0; i < n; ++i) ordered[i] = (*tris)[size_t(perm[i])];
    bvh->tris.swap(ordered);
}

// Squared distance from p to a node's AABB (the prune lower bound).
// Branchless (fmax lowers to fmaxnm; std::max keeps its NaN branches):
// at most one max arm is positive per axis (min <= max), so the nested
// max is bit-identical to the two-arm sum it replaces (max(a, b, 0),
// with fmax's NaN-missing and +-0 rules matching arm by arm).
double NodeBound(CollideBvhNode const &node, double px, double py, double pz)
{
    double const dx =
        fmax(fmax(node.minX - px, px - node.maxX), 0.0);
    double const dy =
        fmax(fmax(node.minY - py, py - node.maxY), 0.0);
    double const dz =
        fmax(fmax(node.minZ - pz, pz - node.maxZ), 0.0);
    return dx * dx + dy * dy + dz * dz;
}

// The prune ceiling (the tighter of the running best and the squared push
// radius), computed once per interior node for both children — best cannot
// change between the two verdicts, so the shared cap keeps both bit-exact.
double PruneCap(ClosestHit const &best, double radius2)
{
    double cap = radius2;
    if (best.found && best.dist2 < cap) cap = best.dist2;
    return cap;
}

// True when a node bound prunes against the cap: a pruned node holds only
// triangles with true d2 strictly above best and radius2, so skipping them
// is behavior-free — past-best triangles can never win the ordinal race,
// and past-radius triangles have pen = offset - sqrt(d2) < 0 strictly (a
// correctly-rounded sqrt cannot round a true value above the offset down
// below it), so they can never push. Best only ever improves, so a node
// pruned now stays pruned. Queries whose true closest sits past the offset
// prune the whole far tree instead of running an exact search they discard.
bool PruneBelow(double lb, double cap)
{
    if (!(lb > cap)) return false;
    // Ulp-margin prune test: nonnegative doubles order as int64, so the bit
    // gap is the ulp distance — 4.5M ulps is ~1e-9 relative at every normal
    // scale (bound and triangle-test error are ~1e-15) and an absolute ~2e-317
    // floor below normal range, where the old max(lb, 1e-300) floor lived.
    // Ties (gap ~1e3 ulps) still fall through to the ordinal test. NaN lbs
    // exit at the float compare above; infinite lbs prune, which matches the
    // response (infinite distance never pushes). One int subtract in place of
    // max+multiply+subtract.
    int64_t lbBits, capBits;
    std::memcpy(&lbBits, &lb, sizeof(double));
    std::memcpy(&capBits, &cap, sizeof(double));
    return lbBits - capBits > 4500000;
}

// Closest point over one BVH collider into the shared best. The prune test
// provably skips no winner, tie, or push candidate: the AABB bound's FP
// error is ~1e-15 relative while the slop is 1e-9, so a pruned node holds
// only triangles with true d2 strictly above best and radius2 (ties, at
// gap zero, always fall through to the exact ordinal test). Children visit
// nearer-first: the near child descends directly (no push/pop roundtrip,
// no redundant pop-side recheck — best cannot improve between a push and
// its immediate pop), the far child pushes and re-checks at pop against
// the improved best, so a strong best lands early and far subtrees prune.
// The visit order matches push-both-then-pop-near exactly. Never allocates.
void QueryColliderBvh(CollideBvh const &bvh, int colliderIdx,
                      GfVec3d const &p, ClosestHit *best, double radius2,
                      UsdGenCollidePhaseStats *probe)
{
    if (bvh.root < 0) return;
    struct Entry
    {
        int node;
        double lb;
    };
    Entry stack[256];
    int sp = 0;
    double const px = p[0], py = p[1], pz = p[2];
    // The root checks once (whole-tree prune for far queries); descended
    // nodes skip the loop-top recheck — best cannot improve between the
    // parent's push-side verdict and the descent — while popped nodes
    // re-check against the improved best. Same prunes, same visit order.
    double const rootLb =
        NodeBound(bvh.nodes[size_t(bvh.root)], px, py, pz);
    if (PruneBelow(rootLb, PruneCap(*best, radius2))) return;
    int nodeIdx = bvh.root;
    for (;;) {
        CollideBvhNode const &node = bvh.nodes[size_t(nodeIdx)];
        if (probe) ++probe->nodeVisits;
        if (node.left >= 0) {
            double const lbL =
                NodeBound(bvh.nodes[size_t(node.left)], px, py, pz);
            double const lbR =
                NodeBound(bvh.nodes[size_t(node.right)], px, py, pz);
            double const cap = PruneCap(*best, radius2);
            bool const pruneL = PruneBelow(lbL, cap);
            bool const pruneR = PruneBelow(lbR, cap);
            if (!pruneL && !pruneR) {
                if (lbL <= lbR) {
                    stack[sp++] = {node.right, lbR};
                    nodeIdx = node.left;
                } else {
                    stack[sp++] = {node.left, lbL};
                    nodeIdx = node.right;
                }
                continue;
            }
            if (!pruneL) {
                nodeIdx = node.left;
                continue;
            }
            if (!pruneR) {
                nodeIdx = node.right;
                continue;
            }
        } else {
            for (int i = node.begin; i < node.end; ++i) {
                CollideTri const &t = bvh.tris[size_t(i)];
                GfVec3d const a(t.ax, t.ay, t.az);
                GfVec3d const b(t.bx, t.by, t.bz);
                GfVec3d const c(t.cx, t.cy, t.cz);
                GfVec3d q;
                double d2 = 0.0;
                if (probe) ++probe->triTests;
                if (!ClosestOnTriangleT<false>(p, a, b, c, &q, &d2)) continue;
                if (!(d2 >= 0.0)) continue;
                if (BetterHit(d2, colliderIdx, t.face, t.k, *best)) {
                    best->found = true;
                    best->q = q;
                    best->a = a;
                    best->b = b;
                    best->c = c;
                    best->dist2 = d2;
                    best->collider = colliderIdx;
                    best->face = t.face;
                    best->k = t.k;
                }
            }
        }
        for (;;) {
            if (sp == 0) return;
            Entry const entry = stack[--sp];
            if (!PruneBelow(entry.lb, PruneCap(*best, radius2))) {
                nodeIdx = entry.node;
                break;
            }
        }
    }
}

// Closest point over every captured collider, BVH or brute force per
// collider, sharing one running best with the ordinal tiebreak. The BVH
// path also prunes past the squared push radius (triangles there can never
// push, so skipping them is behavior-free); the brute path has no bounds
// and tests everything. Reads live points on the brute path and
// capture-frozen triangles on the BVH path (both the same epoch's
// geometry); never allocates.
ClosestHit QueryColliders(UsdGenGraphDesc const *desc,
                           UsdGenCollideCapture const &cap, GfVec3d const &p,
                           double radius2, UsdGenCollidePhaseStats *probe)
{
    ClosestHit best;
    if (!desc) return best;
    int colliderIdx = 0;
    for (UsdGenCollideCollider const &collider : cap.colliders) {
        if (collider.useBvh) {
            QueryColliderBvh(collider.bvh, colliderIdx, p, &best, radius2,
                             probe);
        } else {
            if (collider.geometry >= desc->surfaces.size()) continue;
            BruteCollider(desc->surfaces[collider.geometry], collider,
                          colliderIdx, p, &best, probe);
        }
        ++colliderIdx;
    }
    return best;
}

// Validates one resolved collider mesh and produces its restricted face
// list (empty == whole mesh). Every malformed shape fails closed; exactly
// degenerate triangles are skipped with a warning, and a collider with no
// valid triangle at all fails closed. The counting walk doubles as the
// fan-triangle flattener: it visits the same (face, fan) pairs in the
// same order the old separate pass did (its defensive guards provably
// never fire past the pre-checks above), so a non-null `tris` collects
// exactly the brute path's test set — ascending faces, ascending fan
// index, degenerates skipped — with one walk and one cross product per
// triangle instead of two. Null collects nothing (the brute-force path).
bool ValidateColliderMesh(SdfPath const &path, UsdGenSurfaceDesc const &entry,
                          UsdGenSurfaceDesc const &geometry,
                          std::vector<int> *faces, std::vector<CollideTri> *tris,
                          std::string *error, std::string *warning)
{
    if (geometry.points.empty()) {
        *error = "collider surface '" + path.GetString() + "' has no points";
        return false;
    }
    if (geometry.faceVertexCounts.empty()) {
        *error = "collider surface '" + path.GetString() + "' has no topology";
        return false;
    }
    size_t corners = 0;
    for (int n : geometry.faceVertexCounts) {
        if (n < 0) {
            *error = "collider surface '" + path.GetString() +
                     "' has a negative faceVertexCounts entry";
            return false;
        }
        corners += size_t(n);
    }
    if (corners != geometry.faceVertexIndices.size()) {
        *error = "collider surface '" + path.GetString() +
                 "' has a faceVertexCounts/faceVertexIndices mismatch";
        return false;
    }
    for (GfVec3f const &v : geometry.points) {
        if (!Finite(v[0]) || !Finite(v[1]) || !Finite(v[2])) {
            *error = "collider surface '" + path.GetString() +
                     "' has non-finite points";
            return false;
        }
    }
    for (int v : geometry.faceVertexIndices) {
        if (v < 0 || size_t(v) >= geometry.points.size()) {
            *error = "collider surface '" + path.GetString() +
                     "' has a faceVertexIndices entry out of range";
            return false;
        }
    }
    faces->clear();
    if (UsdGenSurfaceRestricted(entry)) {
        faces->assign(entry.subsetFaces.cbegin(), entry.subsetFaces.cend());
        std::sort(faces->begin(), faces->end());
        faces->erase(std::unique(faces->begin(), faces->end()), faces->end());
        for (int f : *faces) {
            if (f < 0 || size_t(f) >= geometry.faceVertexCounts.size()) {
                *error = "collider surface '" + path.GetString() +
                         "' names subset face " + std::to_string(f) +
                         " outside its mesh";
                return false;
            }
        }
        // Collider faces empty == whole mesh downstream: an empty GeomSubset
        // must not widen to its parent.
        if (faces->empty()) {
            *error = "collider subset '" + path.GetString() + "' names no face";
            return false;
        }
    }
    // At least one non-degenerate triangle must exist; count the skipped.
    // Single linear pass with a running index offset (the subset list is
    // sorted unique, so one merge cursor serves both cases) — the old
    // per-face offset re-walk was quadratic in the face count. Valid
    // triangles flatten into `tris` here when it is non-null.
    if (tris) tris->clear();
    size_t valid = 0, skipped = 0;
    bool const whole = faces->empty();
    size_t fi = 0;
    size_t const nFaces = faces->size();
    size_t off = 0;
    size_t const F = geometry.faceVertexCounts.size();
    for (size_t f = 0; f < F; ++f) {
        int const n = geometry.faceVertexCounts[f];  // >= 0 (checked above)
        bool const use =
            whole ? true : (fi < nFaces && size_t((*faces)[fi]) == f);
        if (use) {
            ++fi;
            if (n >= 3) {
                for (int k = 1; k + 1 < n; ++k) {
                    GfVec3d const a(
                        geometry.points[size_t(geometry.faceVertexIndices[off])]);
                    GfVec3d const b(
                        geometry.points[size_t(geometry.faceVertexIndices[off + size_t(k)])]);
                    GfVec3d const c(
                        geometry.points[size_t(geometry.faceVertexIndices[off + size_t(k) + 1])]);
                    GfVec3d const cr = GfCross(b - a, c - a);
                    double const l = cr.GetLength();
                    if (l > 0.0 && Finite(l)) {
                        ++valid;
                        if (tris) {
                            CollideTri tri;
                            tri.ax = a[0]; tri.ay = a[1]; tri.az = a[2];
                            tri.bx = b[0]; tri.by = b[1]; tri.bz = b[2];
                            tri.cx = c[0]; tri.cy = c[1]; tri.cz = c[2];
                            tri.face = int(f);
                            tri.k = k;
                            tris->push_back(tri);
                        }
                    } else {
                        ++skipped;
                    }
                }
            }
        }
        off += size_t(n);
    }
    if (!valid) {
        *error = "collider surface '" + path.GetString() +
                 "' has no valid triangle";
        return false;
    }
    if (skipped) {
        *warning = "collider surface '" + path.GetString() + "' skips " +
                   std::to_string(skipped) + " degenerate triangle(s)";
    }
    return true;
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("colliders"));    // rel retarget recompiles (02 §6)
    v.push_back(TfToken("resolveType"));  // structural kernel branch (C1)
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("mask"));
    v.push_back(TfToken("offset"));
    v.push_back(TfToken("pushAmount"));
    v.push_back(TfToken("iterations"));
    return v;
}();

TfSpan<const TfToken> UsdGenCollideOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenCollideOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenCollideOp::CreateCapture() const
{
    return std::make_unique<UsdGenCollideCapture>();
}
uint32_t UsdGenCollideOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenCollideOp::Bind(UsdGenParamView const &params,
                           UsdGenDiagnostics *diag)
{
    TfToken const resolveType = params.GetToken(sResolveType, sFlexible);
    if (resolveType != sFlexible && resolveType != sStiff) {
        if (diag) diag->Error("UsdGenCollide: unknown usdGen:resolveType '" +
                              resolveType.GetString() + "'");
        return false;
    }
    for (TfToken const *name : {&sOffset, &sPushAmount}) {
        double const v = params.GetDoubleLiteral(*name, 0.0);
        if (!Finite(v)) {
            if (diag) diag->Error("UsdGenCollide: usdGen:" + name->GetString() +
                                  " must be finite");
            return false;
        }
    }
    if (params.GetInt(sIterations, 1) < 1) {
        if (diag) diag->Error("UsdGenCollide: usdGen:iterations must be at least 1");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenCollideOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Structural resolveType plus the resolved collider geometry: an animated
    // collider (or a reordered surface pool, which moves the captured indices)
    // re-captures. Value edits (offset/pushAmount/iterations, and connected
    // values folded in separately by the scheduler) sweep without recapture.
    opUtil::Digest d;
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    d.Mix(p ? p->GetToken(sResolveType, sFlexible) : sFlexible);
    d.Mix(ctx.upstreamGeneration);
    d.Mix(uint64_t(ctx.seed));
    if (ctx.desc && p && p->node) {
        for (SdfPath const &path : p->node->surfaces) {
            uint32_t entry = 0;
            if (!FindSurface(ctx.desc, path, &entry)) {
                d.Mix(uint64_t(~0ull));
                continue;
            }
            d.Mix(uint64_t(entry));
            UsdGenSurfaceDesc const &e = ctx.desc->surfaces[entry];
            d.Mix(e.surfaceGeneration);
            d.Mix(uint64_t(e.subsetFaces.size()));
            if (e.faceVertexCounts.empty() && !e.subsetFaces.empty()) {
                uint32_t parent = 0;
                if (FindSurface(ctx.desc, path.GetParentPath(), &parent))
                    d.Mix(ctx.desc->surfaces[parent].surfaceGeneration);
                else
                    d.Mix(uint64_t(~0ull) - 1);
            }
        }
    }
    return d.Epoch(0x436F6C6C6964ull);
}

bool UsdGenCollideOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p || !p->node) {
        if (diag) diag->Error("UsdGenCollide: no node description");
        return false;
    }
    // Literals and the resolveType token are validated here (the CPU lane
    // never calls Bind).
    if (!Bind(*p, diag)) return false;
    if (!ctx.desc) {
        if (diag) diag->Error("UsdGenCollide: no graph description");
        return false;
    }
    std::vector<uint32_t> spans;
    std::string spansError;
    if (!opUtil::CurveSpans(upstream, &spans, &spansError)) {
        if (diag) diag->Error("UsdGenCollide: upstream CV topology " + spansError);
        return false;
    }
    if (p->node->surfaces.empty()) {
        if (diag) diag->Error("UsdGenCollide: no collider surfaces: author "
                              "usdGen:colliders (and/or a bound usdGen:surface)");
        return false;
    }
    auto &cap = *static_cast<UsdGenCollideCapture *>(out);
    cap.colliders.clear();
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;
    for (SdfPath const &path : p->node->surfaces) {
        uint32_t entry = 0;
        if (!FindSurface(ctx.desc, path, &entry)) {
            if (diag) diag->Error("UsdGenCollide: collider surface '" +
                                  path.GetString() + "' is not in the description");
            return false;
        }
        UsdGenSurfaceDesc const &e = ctx.desc->surfaces[entry];
        // A GeomSubset stub carries subsetFaces only; its parent mesh carries
        // the geometry (builder R15 pools; opUtil::RestSurfaceArea precedent).
        uint32_t geometry = entry;
        if (e.faceVertexCounts.empty() && !e.subsetFaces.empty()) {
            if (!FindSurface(ctx.desc, path.GetParentPath(), &geometry)) {
                if (diag) diag->Error("UsdGenCollide: collider subset '" +
                                      path.GetString() +
                                      "' has no parent mesh in the description");
                return false;
            }
        }
        UsdGenCollideCollider collider;
        collider.geometry = geometry;
        std::string error, warning;
        // The validation walk flattens the fan triangles as its byproduct,
        // so the threshold cut below reuses them instead of re-walking.
        std::vector<CollideTri> tris;
        bool const collectTris = !sForceBrute.load(std::memory_order_relaxed);
        if (!ValidateColliderMesh(path, e, ctx.desc->surfaces[geometry],
                                  &collider.faces, collectTris ? &tris : nullptr,
                                  &error, &warning)) {
            if (diag) diag->Error("UsdGenCollide: " + error);
            return false;
        }
        if (!warning.empty() && diag) diag->Warn("UsdGenCollide: " + warning);
        // Index colliders past the brute-force threshold (plan/04 §4: a BVH
        // rebuilt per surface epoch). The flattened set is exactly the set
        // the brute path tests, so the cut is behavior-free either way.
        if (collectTris &&
            tris.size() > sBruteThreshold.load(std::memory_order_relaxed)) {
            BuildColliderBvh(&tris, &collider.bvh);
            collider.useBvh = true;
        }
        cap.colliders.push_back(std::move(collider));
    }
    // Every connected value is scanned, so a non-finite expression fails
    // closed at capture instead of writing NaN points (displace precedent).
    // Uniform fields (literals, groom domain) carry one value for every CV,
    // so a single check is exactly equivalent to the per-CV loop — and an
    // empty groom skips it, matching the vacuous loop.
    UsdGenParamField const offset = p->GetScalarField(sOffset, 0.0);
    UsdGenParamField const pushAmount = p->GetScalarField(sPushAmount, 0.0);
    UsdGenParamField const mask = p->GetScalarField(sMask, 1.0);
    for (UsdGenParamField const *field : {&offset, &pushAmount, &mask}) {
        if (field->connected && field->components != 1) {
            if (diag) diag->Error("UsdGenCollide: connected offset/pushAmount/mask "
                                  "must be scalar float");
            return false;
        }
    }
    auto failNonFinite = [&] {
        if (diag) diag->Error("UsdGenCollide: per-CV offset/pushAmount/mask "
                              "must be finite");
        return false;
    };
    bool const scanOffset = !offset.Uniform();
    bool const scanPush = !pushAmount.Uniform();
    bool const scanMask = !mask.Uniform();
    if (upstream.totalCvs > 0) {
        if (!scanOffset && !Finite(offset.Value(0, 0))) return failNonFinite();
        if (!scanPush && !Finite(pushAmount.Value(0, 0))) return failNonFinite();
        if (!scanMask && !Finite(mask.Value(0, 0))) return failNonFinite();
    }
    if (scanOffset || scanPush || scanMask) {
        for (uint32_t c = 0; c != upstream.totalCurves; ++c) {
            for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
                if ((scanOffset && !Finite(offset.Value(c, cv))) ||
                    (scanPush && !Finite(pushAmount.Value(c, cv))) ||
                    (scanMask && !Finite(mask.Value(c, cv)))) {
                    return failNonFinite();
                }
            }
        }
    }
    return true;
}

void UsdGenCollideOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenCollideCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const offsetField =
        p ? p->GetScalarField(sOffset, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const pushField =
        p ? p->GetScalarField(sPushAmount, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    // Value-class fallbacks for literals edited after capture (Capture owns
    // the diagnostics; Evaluate degrades to identity and never writes NaN).
    int const iterations = p ? p->GetInt(sIterations, 1) : 1;
    bool const stiff =
        p && p->GetToken(sResolveType, sFlexible) == sStiff;
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    auto copyInput = [&](size_t o) {
        px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
    };
    auto current = [&](size_t o, int it) -> GfVec3d {
        if (it == 0) return GfVec3d(inPx[o], inPy[o], inPz[o]);
        return GfVec3d(px[o], py[o], pz[o]);
    };
    auto store = [&](size_t o, GfVec3d const &v) {
        px[o] = float(v[0]); py[o] = float(v[1]); pz[o] = float(v[2]);
    };
    if (iterations < 1 || cap.colliders.empty() || !ctx.desc) {
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            size_t const n =
                ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
            size_t const g =
                ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
            for (size_t i = 0; i < n; ++i) copyInput(g + i);
        }
        return;
    }
    UsdGenCollidePhaseStats *const ph = ProbeForThisThread();

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n =
            ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g =
            ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        for (int it = 0; it < iterations; ++it) {
            if (!stiff) {
                for (size_t i = 0; i < n; ++i) {
                    size_t const o = g + i;
                    size_t const cv = cvBase + o;
                    double const push =
                        Sanitize(pushField.Value(curve, cv));
                    double const m = std::clamp(
                        Sanitize(maskField.Value(curve, cv)), 0.0, 1.0);
                    if (push == 0.0 || m == 0.0) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    GfVec3d const pos = current(o, it);
                    // The offset doubles as the search radius (read once):
                    // triangles past it can never push, so the BVH prunes
                    // them without changing a single store. A non-positive
                    // offset prunes the query itself: pen = off - |p - q|
                    // <= 0 always, so the tail below would copy the input
                    // on iteration 0 and keep it after — the skip stores
                    // exactly that with no search.
                    double const off =
                        Sanitize(offsetField.Value(curve, cv));
                    if (off <= 0.0) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    double const q0 = ph ? NowMs() : 0.0;
                    ClosestHit const hit = QueryColliders(
                        ctx.desc, cap, pos, off * off, ph);
                    double const r0 = ph ? NowMs() : 0.0;
                    if (ph) {
                        ph->queryMs += r0 - q0;
                        ++ph->queries;
                    }
                    // Single-exit response tail (the pre-probe continues
                    // folded into if/else): identical stores, one phase stamp.
                    if (hit.found) {
                        double const pen = off - std::sqrt(hit.dist2);
                        if (pen > 0.0) {
                            double const s = pen * push * m;
                            if (s != 0.0) {
                                GfVec3d const dir = hit.dist2 == 0.0
                                    ? UnitNormal(hit.a, hit.b, hit.c)
                                    : (pos - hit.q) / std::sqrt(hit.dist2);
                                store(o, pos + dir * s);
                            } else if (it == 0) {
                                copyInput(o);
                            }
                        } else if (it == 0) {
                            copyInput(o);
                        }
                    } else if (it == 0) {
                        copyInput(o);
                    }
                    if (ph) ph->respMs += NowMs() - r0;
                }
            } else {
                // First penetrating CV hinges the strand: the root case
                // translates rigidly, otherwise the downstream CVs rotate
                // about the last clean CV. Both preserve segment lengths.
                // Single-exit per iteration (continues folded into if/else
                // below): identical stores, one response-phase stamp.
                double const it0 = ph ? NowMs() : 0.0;
                double qSum = 0.0;
                size_t hinge = n;  // first index with pen > 0, else n
                GfVec3d target(0.0);
                double hingeMask = 1.0;
                for (size_t i = 0; i < n; ++i) {
                    size_t const o = g + i;
                    size_t const cv = cvBase + o;
                    double const push =
                        Sanitize(pushField.Value(curve, cv));
                    double const m = std::clamp(
                        Sanitize(maskField.Value(curve, cv)), 0.0, 1.0);
                    GfVec3d const pos = current(o, it);
                    double const off =
                        Sanitize(offsetField.Value(curve, cv));
                    // Non-positive offsets can never hinge (pen <= 0
                    // always), so the query is useless — same hinge.
                    if (off <= 0.0) continue;
                    double const q0 = ph ? NowMs() : 0.0;
                    ClosestHit const hit = QueryColliders(
                        ctx.desc, cap, pos, off * off, ph);
                    if (ph) {
                        qSum += NowMs() - q0;
                        ++ph->queries;
                    }
                    double pen = 0.0;
                    GfVec3d dir(0.0, 0.0, 1.0);
                    if (hit.found) {
                        pen = off - std::sqrt(hit.dist2);
                        dir = hit.dist2 == 0.0
                            ? UnitNormal(hit.a, hit.b, hit.c)
                            : (pos - hit.q) / std::sqrt(hit.dist2);
                    }
                    if (pen > 0.0) {
                        hinge = i;
                        target = pos + dir * (pen * push * m);
                        hingeMask = m;
                        break;
                    }
                }
                if (hinge == n || target == current(g + hinge, it)) {
                    // No penetration, or a zero push (masked / zero
                    // pushAmount): identity.
                    if (it == 0)
                        for (size_t i = 0; i < n; ++i) copyInput(g + i);
                } else {
                    size_t const ho = g + hinge;
                    GfVec3d const hp = current(ho, it);
                    if (hinge == 0) {
                        GfVec3d const shift = target - hp;
                        for (size_t i = 0; i < n; ++i)
                            store(g + i, current(g + i, it) + shift);
                    } else {
                        if (it == 0)
                            for (size_t i = 0; i < hinge; ++i) copyInput(g + i);
                        GfVec3d const h = current(g + hinge - 1, it);
                        GfVec3d const from = hp - h, to = target - h;
                        if (from == GfVec3d(0.0) || to == GfVec3d(0.0)) {
                            // Collapsed hinge segment: rigid translation.
                            GfVec3d const shift = target - hp;
                            for (size_t i = hinge; i < n; ++i)
                                store(g + i, current(g + i, it) + shift);
                        } else {
                            GfVec3f const fromF{float(from[0]), float(from[1]),
                                               float(from[2])};
                            GfVec3f const toF{float(to[0]), float(to[1]),
                                             float(to[2])};
                            for (size_t i = hinge; i < n; ++i) {
                                size_t const o = g + i;
                                GfVec3d const rel = current(o, it) - h;
                                GfVec3f const relF{float(rel[0]), float(rel[1]),
                                                   float(rel[2])};
                                GfVec3f const rotated = opUtil::RotateOnto(
                                    relF, fromF, toF, float(hingeMask));
                                store(o, h + GfVec3d(double(rotated[0]),
                                                     double(rotated[1]),
                                                     double(rotated[2])));
                            }
                        }
                    }
                }
                if (ph) {
                    ph->queryMs += qSum;
                    ph->respMs += NowMs() - it0 - qSum;
                }
            }
        }
    }
}

void UsdGenCollideOp::SetPhaseProbe(bool enabled)
{
    sProbeOn.store(enabled, std::memory_order_relaxed);
}

void UsdGenCollideOp::ResetPhaseProbe()
{
    ProbeRegistry &r = Registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (UsdGenCollidePhaseStats *st : r.stats) *st = UsdGenCollidePhaseStats();
}

void UsdGenCollideOp::SetForceBruteForce(bool enabled)
{
    sForceBrute.store(enabled, std::memory_order_relaxed);
}

void UsdGenCollideOp::SetBruteForceThreshold(size_t tris)
{
    sBruteThreshold.store(tris, std::memory_order_relaxed);
}

UsdGenCollidePhaseStats UsdGenCollideOp::ReadPhaseProbe()
{
    UsdGenCollidePhaseStats total;
    ProbeRegistry &r = Registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (UsdGenCollidePhaseStats const *st : r.stats) {
        total.queryMs += st->queryMs;
        total.respMs += st->respMs;
        total.queries += st->queries;
        total.triTests += st->triTests;
        total.nodeVisits += st->nodeVisits;
    }
    return total;
}

}  // namespace usdGen
