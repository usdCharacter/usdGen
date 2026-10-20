// usdGen — UsdGenCollideOp implementation. 02-schema.md §2.8, 04 §4.
//
// Iterative push-out from collider meshes; see collide.h for the contract.
// Transport mirrors the bound surface, not a map: the builders append the
// usdGen:colliders targets to the Collide node's surfaces AFTER the inherited
// bound surface, so front/root-surface semantics stay intact for every other
// consumer (compiler ctx.surface, RootSurface, scatter) while this kernel
// collides against the whole list — 02 §1's "out of colliders and the skin".
// Local deformed collider points are mapped into description-local groom
// space at Capture using the builders' time-sampled world matrices.
//
// Polygon fallback searches fan-triangulated faces; supported subdivision
// schemes use BVH-accelerated proxy bounds and refined limit-surface queries.
// Ties are deterministic in node / ascending-face order and calculations use
// double precision. Capture resolves and validates every collider mesh; Evaluate
// reads captured groom-space points, so an animated collider re-captures
// (surfaceGeneration digest terms) and re-evaluates every cook
// with no stale geometry. Value-class edits (offset/pushAmount/iterations)
// sweep without recapturing (displace precedent), so Evaluate additionally
// sanitizes: a non-finite connected value reads as 0, iterations < 1 copies
// the input, and an unknown resolveType falls back to flexible. Capture
// still hard-errors on every one of those states; the Evaluate fallbacks
// only cover a literal edited after capture and can never write NaN.
#include "usdGen/ops/collide.h"
#include "usdGen/limitSurface.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>
#include <memory>
#include <limits>
#include <numeric>
#include <map>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

const TfToken sOffset{"offset"}, sPushAmount{"pushAmount"};
const TfToken sIterations{"iterations"}, sResolveType{"resolveType"};
const TfToken sFlexible{"flexible"}, sStiff{"stiff"}, sMask{"mask"};
const TfToken sDeepMode{"deepPenetrationMode"};
const TfToken sCollide{"collide"}, sCutThenCollide{"cutThenCollide"};
const TfToken sCutThreshold{"cutDepthThreshold"};
const TfToken sCutBlendDepth{"cutBlendDepth"};

struct UsdGenCollideCollider
{
    struct Triangle {
        GfVec3d a, b, c;
        GfVec2d ua, ub, uc;
        GfVec3d lo, hi;
        int face = -1;
    };
    struct Node {
        GfVec3d lo, hi;
        int left = -1, right = -1;
        uint32_t begin = 0, end = 0;
    };
    struct FaceCandidate {
        GfVec3d lo, hi;
        uint32_t begin = 0, end = 0;
        int face = -1;
        int root = -1;
    };
    uint32_t geometry = 0;      // desc.surfaces index with points/topology
    std::vector<int> faces;     // empty == whole mesh, else sorted unique ids
    bool hasFaces = true;       // false == authored selection contains only holes
    std::vector<GfVec3d> points; // posed vertices in description-local space
    bool closed = false;        // oriented, watertight selected face boundary
    GfVec3d center{0.0};        // bounds center for deterministic stiff fan
    std::shared_ptr<UsdGenCollisionLimitSurface const> limit;
    GfMatrix4d localToGroom{1.0};
    std::vector<Triangle> triangles;
    std::vector<uint32_t> order;
    std::vector<Node> bvh;
    std::vector<FaceCandidate> faceCandidates;
    GfVec3d controlLo{0.0}, controlHi{0.0};
    double proxyError = 0.0;
    double contactTolerance = 0.0;
    double normalSign = 1.0;
    double outwardSign = 1.0;
};

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

// Ericson 5.1.5 closest point on triangle abc. False when the triangle is
// degenerate (zero area); the caller skips those. Non-degenerate edges are
// all nonzero, so the three interior divisions are exact and safe.
bool ClosestOnTriangle(GfVec3d const &p, GfVec3d const &a, GfVec3d const &b,
                       GfVec3d const &c, GfVec3d *closest, double *dist2)
{
    GfVec3d const n = GfCross(b - a, c - a);
    double const len = n.GetLength();
    if (!(len > 0.0) || !Finite(len)) return false;
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

struct ClosestHit
{
    bool found = false;
    bool inside = false;
    size_t collider = 0;
    GfVec3d q{0.0}, a{0.0}, b{0.0}, c{0.0}; // point and winning corners
    GfVec3d normal{0.0, 0.0, 1.0};
    int face = -1;
    uint32_t proxyTriangle = ~uint32_t(0);
    GfVec2d uv{0.0};
    double dist2 = 0.0;
};

// Exact query memoization, owned by one Evaluate invocation. Repeated sweeps
// often revisit unchanged CVs and segments. Keep all coordinate bits (including
// signed zero), query modes and offsets in the key: collisions only evict work,
// never substitute another contact. No capture/frame state survives evaluation.
template <size_t N>
class ContactQueryCache
{
    struct Entry {
        std::array<uint64_t, N> key{};
        ClosestHit hit;
        bool valid = false;
    };
    std::vector<Entry> _entries{256};
public:
    template <class Query>
    ClosestHit Find(std::array<double, N> const &values, Query &&query)
    {
        std::array<uint64_t, N> key;
        uint64_t hash = 0;
        for (size_t i = 0; i < N; ++i) {
            std::memcpy(&key[i], &values[i], sizeof(double));
            uint64_t bits = key[i];
            bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ull;
            bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebull;
            hash ^= (bits ^ (bits >> 31)) + 0x9e3779b97f4a7c15ull +
                    (hash << 6) + (hash >> 2);
        }
        Entry &entry = _entries[hash & (_entries.size() - 1)];
        if (entry.valid && entry.key == key) return entry.hit;
        entry.hit = query();
        entry.key = key;
        entry.valid = true;
        return entry.hit;
    }
};

double BoxDistance2(GfVec3d const &p, GfVec3d const &lo,
                    GfVec3d const &hi)
{
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        double const d = std::max({lo[k] - p[k], 0.0, p[k] - hi[k]});
        d2 += d * d;
    }
    return d2;
}

double BoxBoxDistance2(GfVec3d const &aLo, GfVec3d const &aHi,
                       GfVec3d const &bLo, GfVec3d const &bHi)
{
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        double const d = std::max({bLo[k] - aHi[k], 0.0,
                                   aLo[k] - bHi[k]});
        d2 += d * d;
    }
    return d2;
}

// The limit evaluator's domain is the tessellator's face domain.  In
// particular a Catmull-Clark non-quad has disjoint tiled subface domains.
// Restrict each closest-point solve to its seed facet; adjacent facets are
// separate BVH candidates, so an optimizer never crosses a subface seam.
GfVec2d ClampToFacet(GfVec2d const &uv,
                     UsdGenCollideCollider::Triangle const &t)
{
    GfVec3d q;
    double d2 = 0.0;
    ClosestOnTriangle(GfVec3d(uv[0], uv[1], 0),
                      GfVec3d(t.ua[0], t.ua[1], 0),
                      GfVec3d(t.ub[0], t.ub[1], 0),
                      GfVec3d(t.uc[0], t.uc[1], 0), &q, &d2);
    return GfVec2d(q[0], q[1]);
}

GfVec2d ClampToSearchDomain(UsdGenCollideCollider const &c,
                           UsdGenCollideCollider::Triangle const &t,
                           GfVec2d const &uv)
{
    if (c.limit->FaceHasSubFaces(t.face))
        return ClampToFacet(uv,t);
    if (c.limit->FaceIsTriangle(t.face)) {
        GfVec3d closest;
        double d2 = 0.0;
        ClosestOnTriangle(GfVec3d(uv[0],uv[1],0),
            GfVec3d(0,0,0),GfVec3d(1,0,0),GfVec3d(0,1,0),
            &closest,&d2);
        return GfVec2d(closest[0],closest[1]);
    }
    return GfVec2d(std::clamp(uv[0],0.0,1.0),
                   std::clamp(uv[1],0.0,1.0));
}

bool FacetContains(UsdGenCollideCollider::Triangle const &t,
                   GfVec2d const &uv)
{
    GfVec2d const a=t.ub-t.ua,b=t.uc-t.ua,x=uv-t.ua;
    double const determinant=a[0]*b[1]-a[1]*b[0];
    if (determinant == 0.0) return false;
    double const u=(x[0]*b[1]-x[1]*b[0])/determinant;
    double const v=(a[0]*x[1]-a[1]*x[0])/determinant;
    return u>=-1e-10 && v>=-1e-10 && u+v<=1.0+1e-10;
}

uint32_t LocateLimitFacet(UsdGenCollideCollider const &c,
                          ClosestHit const &hit)
{
    if (hit.proxyTriangle < c.triangles.size() &&
        FacetContains(c.triangles[hit.proxyTriangle],hit.uv))
        return hit.proxyTriangle;
    for (auto const &face : c.faceCandidates) {
        if (face.face != hit.face) continue;
        for (uint32_t id=face.begin;id<face.end;++id)
            if (FacetContains(c.triangles[id],hit.uv)) return id;
    }
    // A seam may be numerically just outside both adjacent facets.  Choose
    // its nearest parametric facet rather than preserving a stale seed.
    double best=std::numeric_limits<double>::infinity();
    uint32_t winner=hit.proxyTriangle;
    for (auto const &face : c.faceCandidates) {
        if (face.face != hit.face) continue;
        for (uint32_t id=face.begin;id<face.end;++id) {
            GfVec2d const clamped=ClampToFacet(hit.uv,c.triangles[id]);
            double const d2=(clamped-hit.uv).GetLengthSq();
            if (d2<best) { best=d2; winner=id; }
        }
    }
    return winner;
}

bool LimitPoint(UsdGenCollideCollider const &c, int face,
                GfVec2d const &uv, GfVec3d *p,
                GfVec3d *du, GfVec3d *dv)
{
    GfVec3d local, localDu, localDv;
    if (!c.limit->Evaluate(face, uv[0], uv[1],
                           &local, &localDu, &localDv)) return false;
    *p = c.localToGroom.Transform(local);
    *du = c.localToGroom.TransformDir(localDu);
    *dv = c.localToGroom.TransformDir(localDv);
    return Finite((*p)[0]) && Finite((*p)[1]) && Finite((*p)[2]) &&
           Finite((*du)[0]) && Finite((*du)[1]) && Finite((*du)[2]) &&
           Finite((*dv)[0]) && Finite((*dv)[1]) && Finite((*dv)[2]);
}

bool RefineLimitHit(UsdGenCollideCollider const &c,
                    UsdGenCollideCollider::Triangle const &t,
                    GfVec3d const &p, ClosestHit *hit,
                    bool lockToFacet = false)
{
    GfVec3d proxy;
    double proxyD2 = 0.0;
    if (!ClosestOnTriangle(p, t.a, t.b, t.c, &proxy, &proxyD2)) return false;
    GfVec3d const e1 = t.b - t.a, e2 = t.c - t.a, x = proxy - t.a;
    double const aa = GfDot(e1, e1), ab = GfDot(e1, e2);
    double const bb = GfDot(e2, e2), ar = GfDot(e1, x), br = GfDot(e2, x);
    double const det = aa * bb - ab * ab;
    if (!(det > 0.0)) return false;
    double const s = std::clamp((ar * bb - br * ab) / det, 0.0, 1.0);
    double const r = std::clamp((br * aa - ar * ab) / det, 0.0, 1.0 - s);
    GfVec2d uv = t.ua + (t.ub - t.ua) * s + (t.uc - t.ua) * r;
    GfVec3d q, du, dv;
    if (!LimitPoint(c, t.face, uv, &q, &du, &dv)) return false;
    double d2 = (p - q).GetLengthSq();
    for (int iteration = 0; iteration < 12; ++iteration) {
        GfVec3d const residual = p - q;
        double const a = GfDot(du, du), b = GfDot(du, dv);
        double const d = GfDot(dv, dv), determinant = a * d - b * b;
        if (!(determinant > 1e-12 * a * d) || !Finite(determinant))
            break;
        GfVec2d const step((GfDot(du, residual) * d -
                            GfDot(dv, residual) * b) / determinant,
                           (GfDot(dv, residual) * a -
                            GfDot(du, residual) * b) / determinant);
        bool improved = false;
        for (double scale = 1.0; scale >= 1.0 / 128.0; scale *= 0.5) {
            GfVec2d const trial = lockToFacet
                ? ClampToFacet(uv + step * scale,t)
                : ClampToSearchDomain(c,t,uv + step * scale);
            // A clamped/rounded step at the current UV cannot improve d2.
            // Keep trying smaller steps, but do not evaluate the same patch.
            if (trial == uv) continue;
            GfVec3d tq, tu, tv;
            if (!LimitPoint(c, t.face, trial, &tq, &tu, &tv)) continue;
            double const trialD2 = (p - tq).GetLengthSq();
            if (trialD2 < d2) {
                uv = trial; q = tq; du = tu; dv = tv; d2 = trialD2;
                improved = true;
                break;
            }
        }
        if (!improved) break;
    }
    hit->found = true;
    hit->q = q; hit->a = t.a; hit->b = t.b; hit->c = t.c;
    hit->dist2 = d2; hit->face = t.face; hit->uv = uv;
    GfVec3d const n = GfCross(du, dv);
    double const length = n.GetLength();
    hit->normal = (length > 0.0 ? n / length :
                   UnitNormal(t.a, t.b, t.c)) * c.normalSign;
    return true;
}

bool ClosestOnHitFacet(UsdGenCollideCapture const &cap,
                       GfVec3d const &p, ClosestHit *hit)
{
    if (hit->collider >= cap.colliders.size()) return false;
    auto const &collider = cap.colliders[hit->collider];
    if (collider.limit) {
        if (hit->proxyTriangle >= collider.triangles.size()) return false;
        ClosestHit refined;
        if (!RefineLimitHit(collider,
                collider.triangles[hit->proxyTriangle], p, &refined,
                true))
            return false;
        refined.collider = hit->collider;
        refined.proxyTriangle = hit->proxyTriangle;
        refined.inside = hit->inside;
        *hit = refined;
        return true;
    }
    return ClosestOnTriangle(p, hit->a, hit->b, hit->c,
                             &hit->q, &hit->dist2);
}

bool BuildLimitProxy(UsdGenCollideCollider *c,
                     UsdGenSurfaceDesc const &mesh, std::string *error)
{
    auto surface = std::make_shared<UsdGenCollisionLimitSurface>();
    if (!surface->Build(mesh, error)) return false;
    c->limit = std::move(surface);
    GfVec3d lo(0.0), hi(0.0);
    bool seeded = false;
    for (GfVec3d const &point : c->points) {
        if (!seeded) { lo = hi = point; seeded = true; }
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], point[k]);
            hi[k] = std::max(hi[k], point[k]);
        }
    }
    double const diagonal = (hi - lo).GetLength();
    // The BVH is only a candidate accelerator.  Its chord may be coarser
    // than the requested contact precision because each candidate is solved
    // on the actual Bfr limit patch below.  A measured, inflated chord error
    // widens the BVH search; excessive curvature still hits an explicit cap.
    double const contactTolerance = std::max(1e-12, diagonal * 1e-5);
    double const proxyTolerance = std::max(contactTolerance,
                                           diagonal * 1e-3);
    size_t const triangleCap = 1000000;
    for (int rate = 2; rate <= 64; rate *= 2) {
        std::vector<UsdGenCollideCollider::Triangle> triangles;
        std::vector<UsdGenCollideCollider::FaceCandidate> faceCandidates;
        double maximumDeviation = 0.0;
        auto addFace = [&](int face) {
            if (!c->limit->HasFace(face)) return true;
            std::vector<GfVec2d> uv;
            std::vector<std::array<int, 3>> facets;
            if (!c->limit->TessellateFace(face, rate, &uv, &facets)) {
                *error = "OpenSubdiv could not tessellate limit face " +
                         std::to_string(face);
                return false;
            }
            if (triangles.size() + facets.size() > triangleCap) {
                *error = "limit-surface proxy exceeds one million triangles";
                return false;
            }
            GfVec3d controlLo, controlHi;
            if (!c->limit->FaceControlBounds(face,&controlLo,&controlHi)) {
                *error = "OpenSubdiv did not provide limit control bounds";
                return false;
            }
            GfVec3d faceLo(0.0), faceHi(0.0);
            bool boundsSeeded = false;
            for (int mask = 0; mask < 8; ++mask) {
                GfVec3d const local(mask & 1 ? controlHi[0] : controlLo[0],
                    mask & 2 ? controlHi[1] : controlLo[1],
                    mask & 4 ? controlHi[2] : controlLo[2]);
                GfVec3d const v = c->localToGroom.Transform(local);
                if (!boundsSeeded) { faceLo = faceHi = v; boundsSeeded = true; }
                for (int k = 0; k < 3; ++k) {
                    faceLo[k] = std::min(faceLo[k], v[k]);
                    faceHi[k] = std::max(faceHi[k], v[k]);
                }
            }
            std::vector<GfVec3d> positions(uv.size());
            for (size_t i = 0; i < uv.size(); ++i) {
                GfVec3d du, dv;
                if (!LimitPoint(*c, face, uv[i], &positions[i], &du, &dv)) {
                    *error = "OpenSubdiv returned a non-finite limit point";
                    return false;
                }
            }
            uint32_t const faceBegin = uint32_t(triangles.size());
            for (auto const &facet : facets) {
                UsdGenCollideCollider::Triangle t;
                t.face = face;
                t.ua = uv[size_t(facet[0])];
                t.ub = uv[size_t(facet[1])];
                t.uc = uv[size_t(facet[2])];
                t.a = positions[size_t(facet[0])];
                t.b = positions[size_t(facet[1])];
                t.c = positions[size_t(facet[2])];
                if (!(GfCross(t.b - t.a, t.c - t.a).GetLength() > 0.0))
                    continue;
                t.lo = t.hi = t.a;
                for (GfVec3d const &v : {t.b,t.c})
                    for (int k = 0; k < 3; ++k) {
                        t.lo[k] = std::min(t.lo[k],v[k]);
                        t.hi[k] = std::max(t.hi[k],v[k]);
                    }
                // Mid-edge and centroid chord error is measured in groom
                // space.  The safety factor keeps neighboring curved patches
                // in the search even when their proxy chord is farther away.
                for (auto const &sample : {
                         std::pair<GfVec2d, GfVec3d>{(t.ua+t.ub)*0.5,(t.a+t.b)*0.5},
                         {(t.ub+t.uc)*0.5,(t.b+t.c)*0.5},
                         {(t.uc+t.ua)*0.5,(t.c+t.a)*0.5},
                         {(t.ua+t.ub+t.uc)/3.0,(t.a+t.b+t.c)/3.0}}) {
                    GfVec3d actual, du, dv;
                    if (!LimitPoint(*c, face, sample.first, &actual, &du, &dv)) {
                        *error = "OpenSubdiv could not evaluate proxy convergence";
                        return false;
                    }
                    maximumDeviation = std::max(maximumDeviation,
                        (actual - sample.second).GetLength());
                }
                triangles.push_back(std::move(t));
            }
            if (triangles.size() > faceBegin)
                faceCandidates.push_back({faceLo,faceHi,faceBegin,
                                          uint32_t(triangles.size()),face});
            return true;
        };
        if (c->faces.empty()) {
            for (size_t f = 0; f < mesh.faceVertexCounts.size(); ++f)
                if (!addFace(int(f))) return false;
        } else {
            for (int f : c->faces) if (!addFace(f)) return false;
        }
        if (triangles.empty()) {
            c->triangles.clear(); c->order.clear(); c->bvh.clear();
            c->faceCandidates.clear();
            c->closed = false;
            return true; // all selected faces are holes
        }
        if (maximumDeviation * 4.0 > proxyTolerance) {
            if (rate == 64) {
                *error = "limit-surface proxy did not reach groom-scale "
                         "precision before tessellation cap";
                return false;
            }
            continue;
        }
        c->proxyError = std::max(contactTolerance, maximumDeviation * 4.0);
        c->contactTolerance = contactTolerance;
        c->triangles = std::move(triangles);
        c->faceCandidates = std::move(faceCandidates);
        c->controlLo = c->faceCandidates.front().lo;
        c->controlHi = c->faceCandidates.front().hi;
        for (auto const &face : c->faceCandidates)
            for (int k=0;k<3;++k) {
                c->controlLo[k]=std::min(c->controlLo[k],face.lo[k]);
                c->controlHi[k]=std::max(c->controlHi[k],face.hi[k]);
            }
        double signedVolume6 = 0.0;
        for (auto const &t : c->triangles)
            signedVolume6 += GfDot(t.a, GfCross(t.b,t.c));
        c->outwardSign = signedVolume6 < 0.0 ? -1.0 : 1.0;
        c->order.resize(c->triangles.size());
        std::iota(c->order.begin(), c->order.end(), uint32_t(0));
        c->bvh.clear();
        c->bvh.reserve(c->triangles.size() / 4 + 2);
        auto build = [&](auto &&self, uint32_t begin, uint32_t end) -> int {
            int const index = int(c->bvh.size());
            c->bvh.emplace_back();
            GfVec3d nlo = c->triangles[c->order[begin]].lo;
            GfVec3d nhi = c->triangles[c->order[begin]].hi;
            for (uint32_t i = begin + 1; i < end; ++i) {
                auto const &t = c->triangles[c->order[i]];
                for (int k = 0; k < 3; ++k) {
                    nlo[k] = std::min(nlo[k], t.lo[k]);
                    nhi[k] = std::max(nhi[k], t.hi[k]);
                }
            }
            c->bvh[index].lo = nlo; c->bvh[index].hi = nhi;
            c->bvh[index].begin = begin; c->bvh[index].end = end;
            if (end - begin <= 8) return index;
            GfVec3d const extent = nhi - nlo;
            int axis = extent[1] > extent[0] ? 1 : 0;
            if (extent[2] > extent[axis]) axis = 2;
            uint32_t const middle = begin + (end - begin) / 2;
            std::nth_element(c->order.begin() + begin,
                             c->order.begin() + middle,
                             c->order.begin() + end,
                [&](uint32_t a, uint32_t b) {
                    double const av = c->triangles[a].lo[axis] +
                                      c->triangles[a].hi[axis];
                    double const bv = c->triangles[b].lo[axis] +
                                      c->triangles[b].hi[axis];
                    return av < bv || (av == bv && a < b);
                });
            int const left = self(self, begin, middle);
            int const right = self(self, middle, end);
            c->bvh[index].left = left; c->bvh[index].right = right;
            return index;
        };
        build(build, 0, uint32_t(c->order.size()));
        // A separate hierarchy preserves each face's exact chord-box seed
        // ranking without scanning every tessellation cell at each query.
        for (auto &face : c->faceCandidates) {
            uint32_t const begin = uint32_t(c->order.size());
            for (uint32_t id = face.begin; id < face.end; ++id)
                c->order.push_back(id);
            face.root = build(build, begin, uint32_t(c->order.size()));
        }
        return true;
    }
    return false;
}

// Signed solid angle of a triangle as seen from p. On a consistently
// oriented closed mesh the sum has magnitude 4*pi inside and zero outside.
// Capture only enables this for watertight meshes, so open skins retain their
// unsigned surface-shell behavior.
double SolidAngle(GfVec3d const &p, GfVec3d const &a,
                  GfVec3d const &b, GfVec3d const &c)
{
    GfVec3d const u = a - p, v = b - p, w = c - p;
    double const lu = u.GetLength(), lv = v.GetLength(), lw = w.GetLength();
    double const numerator = GfDot(u, GfCross(v, w));
    double const denominator = lu * lv * lw + GfDot(u, v) * lw +
                               GfDot(v, w) * lu + GfDot(w, u) * lv;
    return 2.0 * std::atan2(numerator, denominator);
}

ClosestHit ClosestOnLimit(UsdGenCollideCollider const &c,
                          GfVec3d const &p)
{
    ClosestHit best;
    if (c.bvh.empty()) return best;
    uint32_t winning = ~uint32_t(0);
    std::array<std::pair<double,uint32_t>,8> proxySeeds;
    for (auto &entry : proxySeeds)
        entry={std::numeric_limits<double>::infinity(),~uint32_t(0)};
    double proxyBest=std::numeric_limits<double>::infinity();
    auto visit = [&](auto &&self, int index) -> void {
        auto const &node = c.bvh[size_t(index)];
        if (Finite(proxyBest)) {
            double const reach = std::sqrt(proxyBest) + c.proxyError;
            if (BoxDistance2(p, node.lo, node.hi) > reach * reach) return;
        }
        if (node.left < 0) {
            for (uint32_t i = node.begin; i < node.end; ++i) {
                uint32_t const id = c.order[i];
                auto const &t = c.triangles[id];
                if (Finite(proxyBest)) {
                    double const reach = std::sqrt(proxyBest) + c.proxyError;
                    if (BoxDistance2(p, t.lo, t.hi) > reach * reach) continue;
                }
                GfVec3d chord;
                double distance2=0.0;
                if (!ClosestOnTriangle(p,t.a,t.b,t.c,&chord,&distance2))
                    continue;
                proxyBest=std::min(proxyBest,distance2);
                for (size_t slot=0;slot<proxySeeds.size();++slot)
                    if (distance2 < proxySeeds[slot].first ||
                        (distance2 == proxySeeds[slot].first &&
                         id < proxySeeds[slot].second)) {
                        for (size_t j=proxySeeds.size()-1;j>slot;--j)
                            proxySeeds[j]=proxySeeds[j-1];
                        proxySeeds[slot]={distance2,id};
                        break;
                    }
            }
            return;
        }
        double const dl = BoxDistance2(p, c.bvh[size_t(node.left)].lo,
                                      c.bvh[size_t(node.left)].hi);
        double const dr = BoxDistance2(p, c.bvh[size_t(node.right)].lo,
                                      c.bvh[size_t(node.right)].hi);
        if (dl <= dr) { self(self, node.left); self(self, node.right); }
        else { self(self, node.right); self(self, node.left); }
    };
    visit(visit, 0);
    for (auto const &seed : proxySeeds) {
        if (seed.second == ~uint32_t(0)) break;
        ClosestHit candidate;
        if (!RefineLimitHit(c,c.triangles[seed.second],p,&candidate))
            continue;
        candidate.proxyTriangle=seed.second;
        if (!best.found || candidate.dist2 < best.dist2 ||
            (candidate.dist2 == best.dist2 && seed.second < winning)) {
            best=candidate;
            winning=seed.second;
        }
    }
    // The proxy-cell BVH provides cheap seeds.  Every Bfr control hull that
    // could beat the current limit hit contributes nearby exact patch seeds.
    for (auto const &face : c.faceCandidates) {
        if (best.found && BoxDistance2(p,face.lo,face.hi) > best.dist2)
            continue;
        std::array<std::pair<double,uint32_t>,8> seeds;
        for (auto &entry : seeds)
            entry = {std::numeric_limits<double>::infinity(),~uint32_t(0)};
        auto rankFace = [&](auto &&self, int index) -> void {
            auto const &node = c.bvh[size_t(index)];
            if (BoxDistance2(p,node.lo,node.hi) > seeds.back().first)
                return;
            if (node.left < 0) {
                for (uint32_t i=node.begin; i<node.end; ++i) {
                    uint32_t const id = c.order[i];
                    auto const &t = c.triangles[id];
                    double const d2 = BoxDistance2(p,t.lo,t.hi);
                    for (size_t slot=0; slot<seeds.size(); ++slot)
                        if (d2 < seeds[slot].first ||
                            (d2 == seeds[slot].first &&
                             id < seeds[slot].second)) {
                            for (size_t j=seeds.size()-1; j>slot; --j)
                                seeds[j]=seeds[j-1];
                            seeds[slot]={d2,id};
                            break;
                        }
                }
                return;
            }
            auto const &left = c.bvh[size_t(node.left)];
            auto const &right = c.bvh[size_t(node.right)];
            double const dl = BoxDistance2(p,left.lo,left.hi);
            double const dr = BoxDistance2(p,right.lo,right.hi);
            if (dl <= dr) {
                self(self,node.left); self(self,node.right);
            } else {
                self(self,node.right); self(self,node.left);
            }
        };
        rankFace(rankFace, face.root);
        for (auto const &entry : seeds) {
            if (entry.second == ~uint32_t(0)) break;
            // The global seeds have already run this identical patch solve.
            // Preserve every distinct seed and its deterministic tie order.
            if (std::any_of(proxySeeds.begin(), proxySeeds.end(),
                    [&](auto const &seed) { return seed.second == entry.second; }))
                continue;
            ClosestHit candidate;
            if (!RefineLimitHit(c,c.triangles[entry.second],p,&candidate))
                continue;
            candidate.proxyTriangle = entry.second;
            if (!best.found || candidate.dist2 < best.dist2 ||
                (candidate.dist2 == best.dist2 &&
                 entry.second < winning)) {
                best = candidate;
                winning = entry.second;
            }
        }
    }
    if (best.found)
        best.proxyTriangle=LocateLimitFacet(c,best);
    return best;
}

bool RayBox(GfVec3d const &start, GfVec3d const &direction,
            GfVec3d const &lo, GfVec3d const &hi,
            double tLimit, double padding)
{
    double near = 0.0, far = tLimit;
    for (int k = 0; k < 3; ++k) {
        if (direction[k] == 0.0) {
            if (start[k] < lo[k] - padding || start[k] > hi[k] + padding)
                return false;
        } else {
            double a = (lo[k] - padding - start[k]) / direction[k];
            double b = (hi[k] + padding - start[k]) / direction[k];
            if (a > b) std::swap(a, b);
            near = std::max(near, a);
            far = std::min(far, b);
            if (near > far) return false;
        }
    }
    return true;
}

bool SegmentTriangleHit(GfVec3d const &, GfVec3d const &,
                        GfVec3d const &, GfVec3d const &,
                        GfVec3d const &, double *);

// Ray parity uses the accelerated proxy only for classification.  Contact
// positions and normals are always refined against Bfr's posed limit.
bool InsideLimit(UsdGenCollideCollider const &c, GfVec3d const &p,
                 ClosestHit const *nearest = nullptr)
{
    if (!c.closed || c.bvh.empty()) return false;
    ClosestHit local;
    if (!nearest) { local = ClosestOnLimit(c,p); nearest = &local; }
    if (nearest->found && nearest->dist2 <=
        c.proxyError*c.proxyError*4.0) {
        double const side = GfDot(p-nearest->q,
            nearest->normal*c.outwardSign*c.normalSign);
        if (std::abs(side) > std::max(1e-12,c.proxyError*1e-5))
            return side < 0.0;
    }
    auto const &root = c.bvh[0];
    double const reach = (root.hi - root.lo).GetLength() * 4.0;
    if (!(reach > 0.0)) return false;
    GfVec3d const direction = GfVec3d(1.0, 0.371391, 0.117647) * reach;
    std::vector<double> crossings;
    auto visit = [&](auto &&self, int index) -> void {
        auto const &node = c.bvh[size_t(index)];
        if (!RayBox(p, direction, node.lo, node.hi, 1.0, c.proxyError))
            return;
        if (node.left >= 0) {
            self(self, node.left); self(self, node.right);
            return;
        }
        for (uint32_t i = node.begin; i < node.end; ++i) {
            auto const &t = c.triangles[c.order[i]];
            double fraction = 0.0;
            if (SegmentTriangleHit(p, p + direction, t.a, t.b, t.c,
                                   &fraction) && fraction > 1e-7)
                crossings.push_back(fraction);
        }
    };
    visit(visit, 0);
    std::sort(crossings.begin(),crossings.end());
    size_t uniqueCrossings = 0;
    double previous = -1.0;
    for (double t : crossings)
        if (previous < 0.0 || t - previous > 1e-8) {
            ++uniqueCrossings;
            previous = t;
        }
    return (uniqueCrossings & 1) != 0;
}

// Closest point over every captured collider: node order, ascending faces,
// first win on ties. Reads capture-time groom-space points; never
// allocates. A hit with non-finite distance is impossible after Capture
// validated the meshes (the desc is immutable within a run); the guard
// below keeps it from ever producing NaN output regardless.
ClosestHit ClosestOnColliders(UsdGenGraphDesc const *desc,
                              UsdGenCollideCapture const &cap, GfVec3d const &p,
                              double maxOffset = std::numeric_limits<double>::infinity())
{
    ClosestHit best;
    if (!desc) return best;
    for (size_t ci = 0; ci < cap.colliders.size(); ++ci) {
        UsdGenCollideCollider const &collider = cap.colliders[ci];
        if (!collider.hasFaces || collider.geometry >= desc->surfaces.size())
            continue;
        if (collider.limit && !collider.bvh.empty() &&
            Finite(maxOffset) && maxOffset >= 0.0 &&
            BoxDistance2(p,collider.controlLo,collider.controlHi) >
                maxOffset*maxOffset)
            continue;
        ClosestHit local;
        local.collider = ci;
        if (collider.limit) {
            local = ClosestOnLimit(collider, p);
            local.collider = ci;
            local.inside = InsideLimit(collider, p, &local);
            if (local.found && (!best.found ||
                (local.inside && !best.inside) ||
                (local.inside == best.inside && local.dist2 < best.dist2)))
                best = local;
            continue;
        }
        double winding = 0.0;
        UsdGenSurfaceDesc const &surf = desc->surfaces[collider.geometry];
        size_t const F = surf.faceVertexCounts.size();
        if (collider.points.empty() || surf.faceVertexIndices.empty()) continue;
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
                            size_t(v0) >= collider.points.size() ||
                            size_t(v1) >= collider.points.size() ||
                            size_t(v2) >= collider.points.size())
                            continue;
                        GfVec3d const &a = collider.points[size_t(v0)];
                        GfVec3d const &b = collider.points[size_t(v1)];
                        GfVec3d const &c = collider.points[size_t(v2)];
                        GfVec3d q;
                        double d2 = 0.0;
                        if (!ClosestOnTriangle(p, a, b, c, &q, &d2)) continue;
                        if (!(d2 >= 0.0)) continue;
                        if (collider.closed)
                            winding += SolidAngle(p, a, b, c);
                        if (!local.found || d2 < local.dist2) {
                            local.found = true;
                            local.q = q;
                            local.a = a;
                            local.b = b;
                            local.c = c;
                            local.dist2 = d2;
                            local.normal = UnitNormal(a, b, c);
                        }
                    }
                }
            }
            off += size_t(corners);
        }
        // A closed collider containing p takes precedence over any nearer
        // open skin or exterior surface: its own boundary must eject p.
        local.inside = collider.closed && std::abs(winding) > 6.283185307179586;
        if (local.found && (!best.found ||
            (local.inside && !best.inside) ||
            (local.inside == best.inside && local.dist2 < best.dist2)))
            best = local;
    }
    return best;
}

// Signed depth for one closed collider: positive inside, negative outside.
// The distance and classification use the same refined limit/polygon query
// as contact resolution, so cut detection never substitutes proxy depth.
bool SignedDepthOnClosed(UsdGenGraphDesc const *desc,
                         UsdGenCollideCapture const &cap, size_t ci,
                         GfVec3d const &p, double *depth, GfVec3d *boundary)
{
    if (!desc || ci >= cap.colliders.size())
        return false;
    UsdGenCollideCollider const &collider = cap.colliders[ci];
    if (!collider.closed || !collider.hasFaces ||
        collider.geometry >= desc->surfaces.size())
        return false;
    ClosestHit local;
    local.collider = ci;
    if (collider.limit) {
        local = ClosestOnLimit(collider,p);
        if (!local.found) return false;
        local.inside = InsideLimit(collider,p,&local);
        double const distance = std::sqrt(std::max(0.0,local.dist2));
        *depth = local.inside ? distance : -distance;
        *boundary=local.q;
        return true;
    }
    double winding = 0.0;
    UsdGenSurfaceDesc const &surf = desc->surfaces[collider.geometry];
    size_t fi=0,off=0;
    for (size_t f=0;f<surf.faceVertexCounts.size();++f) {
        int const corners=surf.faceVertexCounts[f];
        if (corners<0 || off+size_t(corners)>surf.faceVertexIndices.size()) break;
        bool const use=collider.faces.empty() ||
            (fi<collider.faces.size() && size_t(collider.faces[fi])==f);
        if (use) {
            ++fi;
            if (corners>=3) {
                int const v0=surf.faceVertexIndices[off];
                for (int k=1;k+1<corners;++k) {
                    int const v1=surf.faceVertexIndices[off+size_t(k)];
                    int const v2=surf.faceVertexIndices[off+size_t(k)+1];
                    if (v0<0 || v1<0 || v2<0 ||
                        size_t(v0)>=collider.points.size() ||
                        size_t(v1)>=collider.points.size() ||
                        size_t(v2)>=collider.points.size()) continue;
                    GfVec3d const &a=collider.points[size_t(v0)];
                    GfVec3d const &b=collider.points[size_t(v1)];
                    GfVec3d const &c=collider.points[size_t(v2)];
                    GfVec3d q; double d2=0.0;
                    if (!ClosestOnTriangle(p,a,b,c,&q,&d2)) continue;
                    winding += SolidAngle(p,a,b,c);
                    if (!local.found || d2<local.dist2) {
                        local.found=true; local.dist2=d2; local.q=q;
                    }
                }
            }
        }
        off+=size_t(corners);
    }
    if (!local.found) return false;
    double const distance=std::sqrt(std::max(0.0,local.dist2));
    *depth = std::abs(winding)>6.283185307179586 ? distance : -distance;
    *boundary=local.q;
    return true;
}

void PublishEvalFailure(UsdGenEvalContext const &ctx,uint32_t code)
{
    if (!ctx.failureCode) return;
    uint32_t current=ctx.failureCode->load(std::memory_order_relaxed);
    while ((current==0 || code<current) &&
           !ctx.failureCode->compare_exchange_weak(
               current,code,std::memory_order_release,std::memory_order_relaxed)) {}
}

GfVec3d OutwardAtBoundary(UsdGenGraphDesc const *desc,
                          UsdGenCollideCapture const &cap,
                          ClosestHit const &hit, GfVec3d const &p)
{
    GfVec3d const normal = hit.normal;
    if (hit.collider >= cap.colliders.size() ||
        !cap.colliders[hit.collider].closed)
        return normal;
    double const edge = std::max({(hit.b - hit.a).GetLength(),
                                  (hit.c - hit.b).GetLength(),
                                  (hit.a - hit.c).GetLength()});
    double const step = edge * 1e-5;
    if (!(step > 0.0)) return normal;
    ClosestHit const positive = ClosestOnColliders(desc, cap, p + normal * step);
    ClosestHit const negative = ClosestOnColliders(desc, cap, p - normal * step);
    bool const positiveInside = positive.inside &&
                                positive.collider == hit.collider;
    bool const negativeInside = negative.inside &&
                                negative.collider == hit.collider;
    if (positiveInside != negativeInside)
        return positiveInside ? -normal : normal;
    return normal;
}

GfVec3d HitDirection(UsdGenGraphDesc const *desc,
                     UsdGenCollideCapture const &cap,
                     ClosestHit const &hit, GfVec3d const &p,
                     bool inside)
{
    double const edge = std::max({(hit.b - hit.a).GetLength(),
                                  (hit.c - hit.b).GetLength(),
                                  (hit.a - hit.c).GetLength()});
    // The source positions are float. Fan interpolation of a point exactly
    // on a face can leave a residual at float precision, so treat that narrow
    // band as the boundary before choosing an outward direction.
    double const boundaryTol = edge * 1e-6;
    bool const closed = hit.collider < cap.colliders.size() &&
                        cap.colliders[hit.collider].closed;
    if (hit.dist2 == 0.0 ||
        (closed && hit.dist2 <= boundaryTol * boundaryTol))
        return OutwardAtBoundary(desc, cap, hit, p);
    double const distance = std::sqrt(hit.dist2);
    return (inside ? hit.q - p : p - hit.q) / distance;
}

bool SegmentTriangleHit(GfVec3d const &start, GfVec3d const &end,
                        GfVec3d const &a, GfVec3d const &b,
                        GfVec3d const &c, double *fraction)
{
    GfVec3d const ray = end - start, ab = b - a, ac = c - a;
    GfVec3d const cross = GfCross(ray, ac);
    double const det = GfDot(ab, cross);
    double const scale = ab.GetLength() * ray.GetLength() * ac.GetLength();
    if (!(scale > 0.0) || std::abs(det) <= 1e-12 * scale)
        return false;
    double const invDet = 1.0 / det;
    GfVec3d const s = start - a;
    double const u = GfDot(s, cross) * invDet;
    if (u < -1e-10 || u > 1.0 + 1e-10) return false;
    GfVec3d const q = GfCross(s, ab);
    double const v = GfDot(ray, q) * invDet;
    if (v < -1e-10 || u + v > 1.0 + 1e-10) return false;
    double const t = GfDot(ac, q) * invDet;
    if (t < -1e-7 || t >= 1.0 - 1e-7) return false;
    *fraction = std::max(0.0, t);
    return true;
}

bool RefineLimitCrossing(UsdGenCollideCollider const &collider,
                          UsdGenCollideCollider::Triangle const &triangle,
                          GfVec3d const &start, GfVec3d const &end,
                          double seed, double *fraction,
                          ClosestHit *hit)
{
    GfVec3d const ray = end - start;
    GfVec3d proxy = start + ray * seed;
    double proxyD2 = 0.0;
    if (!ClosestOnTriangle(proxy, triangle.a, triangle.b, triangle.c,
                           &proxy, &proxyD2)) return false;
    GfVec3d const e1 = triangle.b - triangle.a;
    GfVec3d const e2 = triangle.c - triangle.a;
    GfVec3d const x = proxy - triangle.a;
    double const aa = GfDot(e1,e1), ab = GfDot(e1,e2), bb = GfDot(e2,e2);
    double const det = aa*bb-ab*ab;
    if (!(det > 0.0)) return false;
    GfVec2d uv = triangle.ua + (triangle.ub-triangle.ua)*
        ((GfDot(e1,x)*bb-GfDot(e2,x)*ab)/det) +
        (triangle.uc-triangle.ua)*
        ((GfDot(e2,x)*aa-GfDot(e1,x)*ab)/det);
    uv = ClampToFacet(uv,triangle);
    double t = seed;
    GfVec3d q, du, dv;
    for (int iteration = 0; iteration < 12; ++iteration) {
        if (!LimitPoint(collider, triangle.face, uv, &q, &du, &dv))
            return false;
        GfVec3d const residual = start + ray*t - q;
        if (residual.GetLength() <= collider.contactTolerance*0.1) break;
        double const denominator = GfDot(du, GfCross(dv, ray));
        double const scale = du.GetLength()*dv.GetLength()*ray.GetLength();
        if (!(scale > 0.0) ||
            std::abs(denominator) < 1e-12*scale) break;
        GfVec2d const step(GfDot(residual, GfCross(dv, ray))/denominator,
                           GfDot(du, GfCross(residual, ray))/denominator);
        double const dt = GfDot(du, GfCross(dv, residual))/denominator;
        GfVec2d const next = ClampToFacet(uv + step,triangle);
        double const nt = std::clamp(t - dt, 0.0, 1.0);
        if (next == uv && nt == t) break;
        uv = next; t = nt;
    }
    if (!LimitPoint(collider, triangle.face, uv, &q, &du, &dv))
        return false;
    if ((start + ray*t - q).GetLength() >
        collider.contactTolerance || t >= 1.0-1e-7)
        return false;
    hit->found = true; hit->q = q;
    hit->a = triangle.a; hit->b = triangle.b; hit->c = triangle.c;
    hit->face = triangle.face; hit->uv = uv;
    hit->dist2 = 0.0;
    GfVec3d const n = GfCross(du,dv);
    double const normalLength = n.GetLength();
    hit->normal = (normalLength > 0.0 ? n/normalLength :
        UnitNormal(triangle.a,triangle.b,triangle.c)) *
        collider.normalSign;
    *fraction = t;
    return true;
}

// First crossing of a closed collider's selected triangles along a segment.
// The geometric intersection catches off-center crossings that sampling the
// midpoint misses. A boundary hit at the starting CV does not count; a later
// exit hit does, so a boundary-to-opposite-side segment is caught.
ClosestHit FirstClosedSegmentHit(UsdGenGraphDesc const *desc,
                                UsdGenCollideCapture const &cap,
                                GfVec3d const &start, GfVec3d const &end,
                                bool completePolygonHit=false)
{
    ClosestHit best;
    double first = 1.0;
    if (!desc) return best;
    for (size_t ci = 0; ci < cap.colliders.size(); ++ci) {
        UsdGenCollideCollider const &collider = cap.colliders[ci];
        if (!collider.hasFaces || !collider.closed ||
            collider.geometry >= desc->surfaces.size())
            continue;
        if (collider.limit) {
            for (auto const &face : collider.faceCandidates) {
                if (!RayBox(start,end-start,face.lo,face.hi,first,0.0))
                    continue;
                GfVec3d const ray = end-start;
                double const length2 = ray.GetLengthSq();
                if (!(length2 > 0.0)) continue;
                auto refine = [&](uint32_t id, double seed) {
                    auto const &triangle = collider.triangles[id];
                    double fraction = 0.0;
                    ClosestHit candidate;
                    if (!RefineLimitCrossing(collider,triangle,start,end,seed,
                                             &fraction,&candidate) ||
                        fraction >= first) return;
                    if (completePolygonHit) {
                        // Refined normals include the authored orientation.
                        // Recover geometric outward orientation without a
                        // sub-precision inside/outside probe at grazing entry.
                        candidate.normal *= collider.outwardSign *
                                            collider.normalSign;
                        if (!(GfDot(candidate.normal,ray) < 0.0)) return;
                    } else if (fraction == 0.0 &&
                        !InsideLimit(collider,start+ray*1e-6)) return;
                    first = fraction;
                    best = candidate;
                    best.collider = ci;
                    best.proxyTriangle = id;
                };
                std::array<std::pair<double,uint32_t>,8> seeds;
                for (auto &entry : seeds)
                    entry = {std::numeric_limits<double>::infinity(),
                             ~uint32_t(0)};
                // Gather possible chord hits through the face hierarchy,
                // then refine them in their original triangle order.
                double const gatherPadding = collider.proxyError +
                    ray.GetLength()*1e-7 + collider.contactTolerance;
                std::vector<uint32_t> candidates;
                auto gather = [&](auto &&self, int index) -> void {
                    auto const &node = collider.bvh[size_t(index)];
                    if (!RayBox(start,ray,node.lo,node.hi,1.0,
                                gatherPadding)) return;
                    if (node.left >= 0) {
                        self(self,node.left); self(self,node.right);
                        return;
                    }
                    for (uint32_t i=node.begin; i<node.end; ++i)
                        candidates.push_back(collider.order[i]);
                };
                gather(gather,face.root);
                std::sort(candidates.begin(),candidates.end());
                std::vector<uint32_t> processed;
                processed.reserve(candidates.size());
                for (uint32_t id : candidates) {
                    auto const &triangle = collider.triangles[id];
                    double seed = 0.0;
                    if (SegmentTriangleHit(start,end,triangle.a,
                                           triangle.b,triangle.c,&seed) ||
                        RayBox(start,ray,triangle.lo,triangle.hi,first,
                               collider.proxyError)) {
                        processed.push_back(id);
                        if (seed == 0.0) {
                            GfVec3d const center =
                                (triangle.a+triangle.b+triangle.c)/3.0;
                            seed = std::clamp(GfDot(center-start,ray)/length2,
                                              0.0,1.0);
                        }
                        refine(id,seed);
                    }
                }
                // For a missed chord, segment-box distance is a lower bound
                // on its unchanged center-projection score.  Once eight
                // closer seeds exist, skip entire farther subtrees.
                GfVec3d segLo, segHi;
                for (int k=0; k<3; ++k) {
                    segLo[k]=std::min(start[k],end[k]);
                    segHi[k]=std::max(start[k],end[k]);
                }
                auto rankMissed = [&](auto &&self, int index) -> void {
                    auto const &node = collider.bvh[size_t(index)];
                    auto lowerBound = [&](UsdGenCollideCollider::Node const &n) {
                        return BoxBoxDistance2(segLo,segHi,n.lo,n.hi);
                    };
                    if (lowerBound(node) > seeds.back().first) return;
                    if (node.left >= 0) {
                        double const dl=lowerBound(collider.bvh[size_t(node.left)]);
                        double const dr=lowerBound(collider.bvh[size_t(node.right)]);
                        if (dl <= dr) {
                            self(self,node.left); self(self,node.right);
                        } else {
                            self(self,node.right); self(self,node.left);
                        }
                        return;
                    }
                    for (uint32_t i=node.begin; i<node.end; ++i) {
                    uint32_t const id=collider.order[i];
                    if (std::binary_search(processed.begin(),processed.end(),id))
                        continue;
                    auto const &triangle = collider.triangles[id];
                    GfVec3d const center =
                        (triangle.a+triangle.b+triangle.c)/3.0;
                    double const projected = std::clamp(
                        GfDot(center-start,ray)/length2,0.0,1.0);
                    GfVec3d proxy;
                    double distance2 = 0.0;
                    if (!ClosestOnTriangle(start+ray*projected,
                            triangle.a,triangle.b,triangle.c,
                            &proxy,&distance2)) continue;
                    for (size_t slot=0; slot<seeds.size(); ++slot)
                        if (distance2 < seeds[slot].first ||
                            (distance2 == seeds[slot].first &&
                             id < seeds[slot].second)) {
                            for (size_t j=seeds.size()-1; j>slot; --j)
                                seeds[j]=seeds[j-1];
                            seeds[slot]={distance2,id};
                            break;
                        }
                    }
                };
                rankMissed(rankMissed,face.root);
                for (auto const &entry : seeds) {
                    if (entry.second == ~uint32_t(0)) break;
                    auto const &triangle = collider.triangles[entry.second];
                    GfVec3d const center =
                        (triangle.a+triangle.b+triangle.c)/3.0;
                    double const seed = std::clamp(
                        GfDot(center-start,ray)/length2,0.0,1.0);
                    refine(entry.second,seed);
                }
            }
            continue;
        }
        UsdGenSurfaceDesc const &surf = desc->surfaces[collider.geometry];
        size_t fi = 0, off = 0;
        for (size_t f = 0; f < surf.faceVertexCounts.size(); ++f) {
            int const n = surf.faceVertexCounts[f];
            bool const use = collider.faces.empty() ||
                (fi < collider.faces.size() && size_t(collider.faces[fi]) == f);
            if (use) {
                ++fi;
                if (n >= 3) {
                    GfVec3d const &a = collider.points[size_t(surf.faceVertexIndices[off])];
                    for (int k = 1; k + 1 < n; ++k) {
                        GfVec3d const &b = collider.points[size_t(
                            surf.faceVertexIndices[off + size_t(k)])];
                        GfVec3d const &c = collider.points[size_t(
                            surf.faceVertexIndices[off + size_t(k) + 1])];
                        double t = 0.0;
                        if (!SegmentTriangleHit(start, end, a, b, c, &t) ||
                            t >= first)
                            continue;
                        GfVec3d const geometricNormal =
                            UnitNormal(a,b,c) * collider.outwardSign;
                        if (completePolygonHit) {
                            if (!(GfDot(geometricNormal,end-start) < 0.0))
                                continue;
                        } else if (t == 0.0) {
                            ClosestHit const probe = ClosestOnColliders(
                                desc, cap, start + (end - start) * 1e-6);
                            if (!probe.inside || probe.collider != ci)
                                continue;
                        }
                        first = t;
                        best.found = true;
                        best.collider = ci;
                        best.a = a; best.b = b; best.c = c;
                        if (completePolygonHit) {
                            best.face = int(f);
                            best.q = start + (end - start) * t;
                            best.dist2 = 0.0;
                            best.normal = geometricNormal;
                        }
                    }
                }
            }
            off += size_t(n);
        }
    }
    return best;
}

// Validates one resolved collider mesh and produces its restricted face
// list (empty == whole mesh). Every malformed shape fails closed; exactly
// degenerate triangles are skipped with a warning, and a collider with no
// valid triangle at all fails closed.
bool ValidateColliderMesh(SdfPath const &path, UsdGenSurfaceDesc const &entry,
                          UsdGenSurfaceDesc const &geometry,
                          std::vector<int> *faces, bool *hasFaces,
                          bool *closed, std::string *error, std::string *warning)
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
    std::vector<int> holes(geometry.holeIndices.cbegin(),
                           geometry.holeIndices.cend());
    std::sort(holes.begin(), holes.end());
    holes.erase(std::unique(holes.begin(), holes.end()), holes.end());
    for (int f : holes) {
        if (f < 0 || size_t(f) >= geometry.faceVertexCounts.size()) {
            *error = "collider surface '" + path.GetString() +
                     "' names hole face " + std::to_string(f) +
                     " outside its mesh";
            return false;
        }
    }
    if (!holes.empty()) {
        if (faces->empty()) {
            faces->reserve(geometry.faceVertexCounts.size() - holes.size());
            for (size_t f = 0; f < geometry.faceVertexCounts.size(); ++f)
                if (!std::binary_search(holes.begin(), holes.end(), int(f)))
                    faces->push_back(int(f));
        } else {
            faces->erase(std::remove_if(faces->begin(), faces->end(),
                [&](int f) { return std::binary_search(holes.begin(),
                                                        holes.end(), f); }),
                faces->end());
        }
    }
    *hasFaces = true;
    if (faces->empty() && !holes.empty()) {
        *hasFaces = false;
        *closed = false;
        return true;
    }
    // At least one non-degenerate triangle must exist; count the skipped.
    size_t valid = 0, skipped = 0;
    auto scanFace = [&](size_t f) {
        size_t off = 0;
        for (size_t k = 0; k < f; ++k)
            off += size_t(geometry.faceVertexCounts[k]);
        int const n = geometry.faceVertexCounts[f];
        if (n < 3) return;
        for (int k = 1; k + 1 < n; ++k) {
            GfVec3d const a(geometry.points[size_t(geometry.faceVertexIndices[off])]);
            GfVec3d const b(geometry.points[size_t(geometry.faceVertexIndices[off + size_t(k)])]);
            GfVec3d const c(geometry.points[size_t(geometry.faceVertexIndices[off + size_t(k) + 1])]);
            GfVec3d const cr = GfCross(b - a, c - a);
            double const l = cr.GetLength();
            if (l > 0.0 && Finite(l)) ++valid;
            else ++skipped;
        }
    };
    if (faces->empty()) {
        for (size_t f = 0; f < geometry.faceVertexCounts.size(); ++f) scanFace(f);
    } else {
        for (int f : *faces) scanFace(size_t(f));
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
    // A selected polygon soup defines a solid only when its directed
    // boundary edges pair exactly once in opposite directions. Fan diagonals
    // are internal to each polygon and do not enter this test.
    *closed = skipped == 0;
    std::map<std::pair<int, int>, std::pair<int, int>> edges;
    auto scanEdges = [&](size_t f) {
        size_t off = 0;
        for (size_t k = 0; k < f; ++k)
            off += size_t(geometry.faceVertexCounts[k]);
        int const n = geometry.faceVertexCounts[f];
        if (n < 3) { *closed = false; return; }
        for (int k = 0; k < n; ++k) {
            int const a = geometry.faceVertexIndices[off + size_t(k)];
            int const b = geometry.faceVertexIndices[off + size_t((k + 1) % n)];
            if (a == b) { *closed = false; continue; }
            auto &counts = edges[std::minmax(a, b)];
            if (a < b) ++counts.first;
            else ++counts.second;
        }
    };
    if (faces->empty()) {
        for (size_t f = 0; f < geometry.faceVertexCounts.size(); ++f) scanEdges(f);
    } else {
        for (int f : *faces) scanEdges(size_t(f));
    }
    for (auto const &edge : edges)
        if (edge.second.first != 1 || edge.second.second != 1)
            *closed = false;
    return true;
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("colliders"));    // rel retarget recompiles (02 §6)
    v.push_back(TfToken("resolveType"));  // structural kernel branch (C1)
    v.push_back(sDeepMode);               // structural plane-routing branch
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("mask"));
    v.push_back(TfToken("offset"));
    v.push_back(TfToken("pushAmount"));
    v.push_back(TfToken("iterations"));
    v.push_back(sCutThreshold);
    v.push_back(sCutBlendDepth);
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
void UsdGenCollideOp::Configure(UsdGenParamView const &params)
{
    _cutThenCollide =
        params.GetToken(sDeepMode, sCollide) == sCutThenCollide;
}
uint32_t UsdGenCollideOp::PlanesTouched() const
{
    return kPlanePoints | (_cutThenCollide ? kPlaneWidths : 0u);
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
    TfToken const deepMode = params.GetToken(sDeepMode, sCollide);
    if (deepMode != sCollide && deepMode != sCutThenCollide) {
        if (diag) diag->Error("UsdGenCollide: unknown usdGen:deepPenetrationMode '" +
                              deepMode.GetString() + "'");
        return false;
    }
    double const cutThreshold = params.GetDoubleLiteral(sCutThreshold, 0.0);
    double const cutBlendDepth = params.GetDoubleLiteral(sCutBlendDepth, 0.02);
    if (!Finite(cutThreshold) || cutThreshold < 0.0) {
        if (diag) diag->Error(
            "UsdGenCollide: usdGen:cutDepthThreshold must be finite and >= 0");
        return false;
    }
    if (!Finite(cutBlendDepth) || !(cutBlendDepth > 0.0)) {
        if (diag) diag->Error(
            "UsdGenCollide: usdGen:cutBlendDepth must be finite and > 0");
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
    d.Mix(p ? p->GetToken(sDeepMode,sCollide) : sCollide);
    d.Mix(ctx.upstreamGeneration);
    d.Mix(uint64_t(ctx.seed));
    auto mixSurface = [&](UsdGenSurfaceDesc const &s) {
        d.Mix(s.subdivisionScheme); d.Mix(s.orientation);
        d.Mix(s.interpolateBoundary);
        d.Mix(s.faceVaryingLinearInterpolation);
        d.Mix(s.triangleSubdivisionRule); d.Mix(s.creaseMethod);
        for (auto const *values : {&s.faceVertexCounts, &s.faceVertexIndices,
                 &s.holeIndices, &s.creaseIndices, &s.creaseLengths,
                 &s.cornerIndices, &s.subsetFaces}) {
            d.Mix(uint64_t(values->size()));
            for (int v : *values) d.Mix(uint64_t(uint32_t(v)));
        }
        for (auto const *values : {&s.creaseSharpnesses,
                                   &s.cornerSharpnesses}) {
            d.Mix(uint64_t(values->size()));
            for (float v : *values) d.Mix(double(v));
        }
        d.Mix(uint64_t(s.points.size()));
        for (GfVec3f const &v : s.points)
            for (int k = 0; k < 3; ++k) d.Mix(double(v[k]));
        d.Mix(uint64_t(s.isSubset));
    };
    if (ctx.desc)
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                d.Mix(ctx.desc->xformMatrix[row][col]);
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
            mixSurface(e);
            for (int row = 0; row < 4; ++row)
                for (int col = 0; col < 4; ++col)
                    d.Mix(e.worldMatrix[row][col]);
            d.Mix(uint64_t(e.subsetFaces.size()));
            if (e.faceVertexCounts.empty() && !e.subsetFaces.empty()) {
                uint32_t parent = 0;
                if (FindSurface(ctx.desc, path.GetParentPath(), &parent)) {
                    d.Mix(ctx.desc->surfaces[parent].surfaceGeneration);
                    mixSurface(ctx.desc->surfaces[parent]);
                    for (int row = 0; row < 4; ++row)
                        for (int col = 0; col < 4; ++col)
                            d.Mix(ctx.desc->surfaces[parent].worldMatrix[row][col]);
                } else
                    d.Mix(uint64_t(~0ull) - 1);
            }
        }
    }
    return d.Epoch(0x436F6C6C6964ull);
}

void UsdGenCollideOp::WarnOnce(UsdGenDiagnostics *diag, SdfPath const &path,
                               UsdGenSurfaceDesc const &mesh,
                               std::vector<int> const &faces,
                               std::string const &message) const
{
    if (!diag) return;
    opUtil::Digest d;
    auto mixText = [&](std::string const &s) {
        for (unsigned char c : s) d.Mix(uint64_t(c));
    };
    mixText(path.GetString());
    mixText(message);
    d.Mix(mesh.subdivisionScheme);
    d.Mix(mesh.orientation); d.Mix(mesh.interpolateBoundary);
    d.Mix(mesh.faceVaryingLinearInterpolation);
    d.Mix(mesh.triangleSubdivisionRule); d.Mix(mesh.creaseMethod);
    for (int n : mesh.faceVertexCounts) d.Mix(uint64_t(uint32_t(n)));
    for (int n : mesh.faceVertexIndices) d.Mix(uint64_t(uint32_t(n)));
    for (int n : mesh.holeIndices) d.Mix(uint64_t(uint32_t(n)));
    for (int n : mesh.creaseIndices) d.Mix(uint64_t(uint32_t(n)));
    for (int n : mesh.creaseLengths) d.Mix(uint64_t(uint32_t(n)));
    for (int n : mesh.cornerIndices) d.Mix(uint64_t(uint32_t(n)));
    for (float n : mesh.creaseSharpnesses) d.Mix(double(n));
    for (float n : mesh.cornerSharpnesses) d.Mix(double(n));
    for (int n : faces) d.Mix(uint64_t(uint32_t(n)));
    uint64_t const key = d.h ? d.h : 1;
    auto &slot = _warned[size_t(key % _warned.size())];
    if (slot.exchange(key, std::memory_order_relaxed) != key)
        diag->Warn(message);
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
    // USD/Hydra builders sample worldMatrix at desc.time. The groom buffer
    // and published tile points are description-local; map collider geometry
    // into that frame once so every collision query and offset uses groom
    // units, including under parent transforms and nonuniform scale.
    GfMatrix4d const &groomWorld = ctx.desc->xformMatrix;
    auto affine = [](GfMatrix4d const &m) {
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                if (!Finite(m[row][col])) return false;
        return m[0][3] == 0.0 && m[1][3] == 0.0 &&
               m[2][3] == 0.0 && m[3][3] == 1.0;
    };
    if (!affine(groomWorld)) {
        if (diag) diag->Error("UsdGenCollide: description world matrix must be finite and affine");
        return false;
    }
    double groomDeterminant = 0.0;
    GfMatrix4d const worldToGroom = groomWorld.GetInverse(&groomDeterminant);
    if (!Finite(groomDeterminant) || groomDeterminant == 0.0) {
        if (diag) diag->Error("UsdGenCollide: description world matrix must be invertible");
        return false;
    }
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
        if (!ValidateColliderMesh(path, e, ctx.desc->surfaces[geometry],
                                  &collider.faces, &collider.hasFaces,
                                  &collider.closed,
                                  &error, &warning)) {
            if (diag) diag->Error("UsdGenCollide: " + error);
            return false;
        }
        if (!warning.empty() && diag) diag->Warn("UsdGenCollide: " + warning);
        auto const &mesh = ctx.desc->surfaces[geometry];
        if (!affine(mesh.worldMatrix)) {
            if (diag) diag->Error("UsdGenCollide: collider '" + path.GetString() +
                                  "' world matrix must be finite and affine");
            return false;
        }
        GfMatrix4d const colliderToGroom = mesh.worldMatrix * worldToGroom;
        double const colliderDeterminant = colliderToGroom.GetDeterminant();
        if (!Finite(colliderDeterminant) || colliderDeterminant == 0.0) {
            if (diag) diag->Error("UsdGenCollide: collider '" + path.GetString() +
                                  "' world matrix must be invertible");
            return false;
        }
        // The product of two affine matrices is affine mathematically, but
        // an inverted rotated/scaled parent may leave roundoff in the last
        // column. Validate source matrices above, then require this derived
        // matrix only to stay finite.
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                if (!Finite(colliderToGroom[row][col])) {
                    if (diag) diag->Error("UsdGenCollide: collider '" + path.GetString() +
                                          "' relative world matrix must be finite");
                    return false;
                }
        collider.points.reserve(mesh.points.size());
        for (GfVec3f const &local : mesh.points) {
            GfVec3d const point = colliderToGroom.Transform(GfVec3d(local));
            if (!Finite(point[0]) || !Finite(point[1]) || !Finite(point[2])) {
                if (diag) diag->Error("UsdGenCollide: collider '" + path.GetString() +
                                      "' has non-finite transformed points");
                return false;
            }
            collider.points.push_back(point);
        }
        collider.localToGroom = colliderToGroom;
        collider.normalSign = mesh.orientation == TfToken("leftHanded")
            ? -1.0 : 1.0;
        if (collider.closed) {
            // Polygon entry normals need the same geometric winding contract
            // as limit entries. Use groom-space points, including reflections,
            // and an origin on the mesh to avoid translation cancellation.
            GfVec3d const origin = collider.points.front();
            double volume6 = 0.0;
            size_t offset = 0, selected = 0;
            for (size_t face=0;face<mesh.faceVertexCounts.size();++face) {
                int const count = mesh.faceVertexCounts[face];
                bool const use = collider.faces.empty() ||
                    (selected<collider.faces.size() &&
                     size_t(collider.faces[selected])==face);
                if (use && count>=3) {
                    ++selected;
                    auto const &a=collider.points[size_t(mesh.faceVertexIndices[offset])];
                    for (int k=1;k+1<count;++k) {
                        auto const &b=collider.points[size_t(mesh.faceVertexIndices[offset+size_t(k)])];
                        auto const &c=collider.points[size_t(mesh.faceVertexIndices[offset+size_t(k+1)])];
                        volume6 += GfDot(a-origin,GfCross(b-origin,c-origin));
                    }
                }
                offset += size_t(count);
            }
            collider.outwardSign = volume6 < 0.0 ? -1.0 : 1.0;
        }
        TfToken const scheme = mesh.subdivisionScheme;
        if (scheme == TfToken("none")) {
            WarnOnce(diag, path, mesh, collider.faces,
                "UsdGenCollide: collider '" + path.GetString() +
                "' subdivisionScheme=none: authored polygon surface; "
                "using polygon collision");
        } else if (scheme == TfToken("catmullClark") ||
                   scheme == TfToken("loop") ||
                   scheme == TfToken("bilinear")) {
            std::string reason;
            if (!BuildLimitProxy(&collider, mesh, &reason)) {
                bool const fallback =
                    reason.rfind("OpenSubdiv rejected",0) == 0 ||
                    reason.rfind("OpenSubdiv failed to create a face",0) == 0 ||
                    reason.rfind("invalid face size",0) == 0;
                if (!fallback) {
                    if (diag) diag->Error("UsdGenCollide: collider '" +
                        path.GetString() + "' subdivisionScheme=" +
                        scheme.GetString() + ": " + reason);
                    return false;
                }
                collider.limit.reset();
                collider.triangles.clear(); collider.bvh.clear();
                WarnOnce(diag, path, mesh, collider.faces,
                    "UsdGenCollide: collider '" +
                    path.GetString() + "' subdivisionScheme=" +
                    scheme.GetString() + ": " + reason +
                    "; using polygon collision");
            } else if (!mesh.holeIndices.empty()) {
                // The selected limit surface has an open boundary at a hole.
                collider.closed = false;
            }
        } else {
            WarnOnce(diag, path, mesh, collider.faces,
                "UsdGenCollide: collider '" +
                path.GetString() + "' subdivisionScheme=" +
                scheme.GetString() + ": unsupported subdivision scheme; "
                "using polygon collision");
        }
        if (collider.hasFaces && collider.closed) {
            GfVec3d lo(0.0), hi(0.0);
            bool seeded = false;
            size_t fi = 0, off = 0;
            for (size_t f = 0; f < mesh.faceVertexCounts.size(); ++f) {
                int const n = mesh.faceVertexCounts[f];
                bool const use = collider.faces.empty() ||
                    (fi < collider.faces.size() &&
                     size_t(collider.faces[fi]) == f);
                if (use) {
                    ++fi;
                    for (int k = 0; k < n; ++k) {
                        GfVec3d const &point = collider.points[size_t(
                            mesh.faceVertexIndices[off + size_t(k)])];
                        if (!seeded) { lo = point; hi = point; seeded = true; }
                        for (int axis = 0; axis < 3; ++axis) {
                            lo[axis] = std::min(lo[axis], point[axis]);
                            hi[axis] = std::max(hi[axis], point[axis]);
                        }
                    }
                }
                off += size_t(n);
            }
            collider.center = (lo + hi) * 0.5;
            // The authored control hull conservatively contains supported
            // subdivision limits and is exact for polygon fallback.
            collider.controlLo=lo; collider.controlHi=hi;
            if (!(collider.contactTolerance>0.0))
                collider.contactTolerance=std::max(1e-12,(hi-lo).GetLength()*1e-5);
        }
        cap.colliders.push_back(std::move(collider));
    }
    // Every connected value is scanned, so a non-finite expression fails
    // closed at capture instead of writing NaN points (displace precedent).
    UsdGenParamField const offset = p->GetScalarField(sOffset, 0.0);
    UsdGenParamField const pushAmount = p->GetScalarField(sPushAmount, 0.0);
    UsdGenParamField const mask = p->GetScalarField(sMask, 1.0);
    UsdGenParamField const cutThreshold = p->GetScalarField(sCutThreshold, 0.0);
    UsdGenParamField const cutBlendDepth = p->GetScalarField(sCutBlendDepth, 0.02);
    for (UsdGenParamField const *field : {&offset, &pushAmount, &mask,
                                          &cutThreshold, &cutBlendDepth}) {
        if (field->connected && field->components != 1) {
            if (diag) diag->Error("UsdGenCollide: connected offset/pushAmount/mask "
                                  "must be scalar float");
            return false;
        }
    }
    for (uint32_t c = 0; c != upstream.totalCurves; ++c) {
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(offset.Value(c, cv)) ||
                !Finite(pushAmount.Value(c, cv)) ||
                !Finite(mask.Value(c, cv)) || cutThreshold.Value(c,cv)<0.0 ||
                !Finite(cutThreshold.Value(c,cv)) ||
                !(cutBlendDepth.Value(c,cv)>0.0) ||
                !Finite(cutBlendDepth.Value(c,cv))) {
                if (diag) diag->Error("UsdGenCollide: per-CV offset/pushAmount/mask "
                    "must be finite; cutDepthThreshold must be finite and >= 0; "
                    "cutBlendDepth must be finite and > 0");
                return false;
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
    UsdGenParamField const cutThresholdField = p
        ? p->GetScalarField(sCutThreshold,0.0) : UsdGenParamField{0.0};
    UsdGenParamField const cutBlendDepthField = p
        ? p->GetScalarField(sCutBlendDepth,0.02) : UsdGenParamField{0.02};
    bool const cutMode = _cutThenCollide;
    // Value-class fallbacks for literals edited after capture (Capture owns
    // the diagnostics; Evaluate degrades to identity and never writes NaN).
    int const iterations = p ? p->GetInt(sIterations, 1) : 1;
    bool const stiff =
        p && p->GetToken(sResolveType, sFlexible) == sStiff;
    bool const hasClosed = std::any_of(
        cap.colliders.begin(), cap.colliders.end(),
        [](UsdGenCollideCollider const &c) { return c.closed; });
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    bool cutPrepared = false;
    constexpr size_t kMaxDepthIntervals = 256;
    constexpr size_t kMaxDepthEvaluations = 1024;
    GfVec3d *shortenedSource=view->pointScratch;
    auto copyInput = [&](size_t o) {
        if (!cutPrepared) { px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o]; }
    };
    auto current = [&](size_t o, int it) -> GfVec3d {
        if (it == 0 && !cutPrepared)
            return GfVec3d(inPx[o], inPy[o], inPz[o]);
        return GfVec3d(px[o], py[o], pz[o]);
    };
    auto source = [&](size_t o) -> GfVec3d {
        return cutPrepared ? shortenedSource[o]
                           : GfVec3d(inPx[o],inPy[o],inPz[o]);
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
            for (size_t i = 0; i < n; ++i) {
                size_t const local=g+i;
                copyInput(local);
                if (cutMode && view->width && view->inWidth)
                    view->width[local]=view->inWidth[local];
                if (cutMode) for (uint32_t plane=0;plane<view->extraCvCount;++plane) {
                    UsdGenPlane &out=view->extraCv[plane];
                    UsdGenPlane const &in=view->inExtraCv[plane];
                    uint8_t const arity=std::max<uint8_t>(out.arity,1);
                    size_t const first=(cvBase+local)*arity;
                    for (uint8_t component=0;component<arity;++component) {
                        size_t const value=first+component;
                        if (out.type==TfToken("int"))
                            const_cast<int *>(out.i.cdata())[value]=in.i[value];
                        else
                            const_cast<float *>(out.f.cdata())[value]=in.f[value];
                    }
                }
            }
        }
        return;
    }

    ContactQueryCache<4> pointQueries;
    ContactQueryCache<7> segmentQueries;
    auto closest = [&](GfVec3d const &point, double offset) {
        return pointQueries.Find({point[0], point[1], point[2], offset}, [&] {
            return ClosestOnColliders(ctx.desc, cap, point, offset);
        });
    };
    auto firstCrossing = [&](GfVec3d const &start, GfVec3d const &end,
                             bool completePolygonHit = false) {
        return segmentQueries.Find({start[0], start[1], start[2],
                                    end[0], end[1], end[2],
                                    double(completePolygonHit)}, [&] {
            return FirstClosedSegmentHit(ctx.desc, cap, start, end,
                                         completePolygonHit);
        });
    };

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n =
            ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g =
            ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        double const cutThreshold=Sanitize(
            cutThresholdField.Value(curve,cvBase+g));
        double const cutBlendDepth=Sanitize(
            cutBlendDepthField.Value(curve,cvBase+g));
        cutPrepared = false;
        if (cutMode) {
            for (size_t i=0;i<n;++i) {
                size_t const cv=g+i;
                if (view->width && view->inWidth) view->width[cv]=view->inWidth[cv];
                for (uint32_t plane=0;plane<view->extraCvCount;++plane) {
                    UsdGenPlane &out=view->extraCv[plane];
                    UsdGenPlane const &in=view->inExtraCv[plane];
                    uint8_t const arity=std::max<uint8_t>(out.arity,1);
                    size_t const first=(cvBase+cv)*arity;
                    for (uint8_t component=0;component<arity;++component) {
                        size_t const value=first+component;
                        if (out.type==TfToken("int"))
                            const_cast<int *>(out.i.cdata())[value]=in.i[value];
                        else
                            const_cast<float *>(out.f.cdata())[value]=in.f[value];
                    }
                }
            }
        }
        if (cutMode && hasClosed && n > 1) {
            if (!shortenedSource || !view->scalarScratch) {
                PublishEvalFailure(ctx,2);
                return;
            }
            double *arc=view->scalarScratch+g;
            arc[0]=0.0;
            for (size_t i=1;i<n;++i) {
                GfVec3d const a(inPx[g+i-1],inPy[g+i-1],inPz[g+i-1]);
                GfVec3d const b(inPx[g+i],inPy[g+i],inPz[g+i]);
                arc[i]=arc[i-1]+(b-a).GetLength();
            }
            double const total=arc[n-1];
            double cut=0.0;
            for (size_t i=0;i+1<n;++i) {
                GfVec3d const a(inPx[g+i],inPy[g+i],inPz[g+i]);
                GfVec3d const b(inPx[g+i+1],inPy[g+i+1],inPz[g+i+1]);
                double const segmentLength=(b-a).GetLength();
                if (!(segmentLength>0.0)) continue;
                double const wa=std::clamp(
                    Sanitize(pushField.Value(curve,cvBase+g+i))*
                    std::clamp(Sanitize(maskField.Value(curve,cvBase+g+i)),0.0,1.0),
                    0.0,1.0);
                double const wb=std::clamp(
                    Sanitize(pushField.Value(curve,cvBase+g+i+1))*
                    std::clamp(Sanitize(maskField.Value(curve,cvBase+g+i+1)),0.0,1.0),
                    0.0,1.0);
                if (wa==0.0 && wb==0.0) continue;
                // Blend is at most one. Once an earlier contact has removed
                // more length than this entire weighted tail can remove,
                // no depth sample on this segment can change the result.
                // Pad roundoff outward; this is not a contact tolerance.
                double const boundPad = 32.0 *
                    std::numeric_limits<double>::epsilon() * std::max(1.0,total);
                double const tailUpper = std::max(wa,wb) *
                    std::max(0.0,total-arc[i]);
                if (cut > tailUpper + boundPad) continue;
                for (size_t ci=0;ci<cap.colliders.size();++ci) {
                    if (!cap.colliders[ci].closed) continue;
                    auto const &closedCollider=cap.colliders[ci];
                    GfVec3d const segmentLo(
                        std::min(a[0],b[0]),std::min(a[1],b[1]),std::min(a[2],b[2]));
                    GfVec3d const segmentHi(
                        std::max(a[0],b[0]),std::max(a[1],b[1]),std::max(a[2],b[2]));
                    if (segmentHi[0]<closedCollider.controlLo[0] ||
                        segmentLo[0]>closedCollider.controlHi[0] ||
                        segmentHi[1]<closedCollider.controlLo[1] ||
                        segmentLo[1]>closedCollider.controlHi[1] ||
                        segmentHi[2]<closedCollider.controlLo[2] ||
                        segmentLo[2]>closedCollider.controlHi[2]) continue;
                    struct DepthSample { double d=0.0; GfVec3d q{0.0}; };
                    struct Interval { double a,b,upper; DepthSample sa,sb,sm; };
                    bool queryFailed=false;
                    size_t depthEvaluations=0;
                    auto depth=[&](double t) {
                        DepthSample sample;
                        if (++depthEvaluations>kMaxDepthEvaluations ||
                            !SignedDepthOnClosed(ctx.desc,cap,ci,a+(b-a)*t,
                                &sample.d,&sample.q))
                            queryFailed=true;
                        return sample;
                    };
                    auto objective=[&](double t,double d) {
                        double const w=wa+(wb-wa)*t;
                        double const x=std::clamp(
                            (d-cutThreshold)/cutBlendDepth,0.0,1.0);
                        double const blend=x*x*(3.0-2.0*x);
                        double const s=arc[i]+segmentLength*t;
                        return w*std::max(0.0,total-s)*blend;
                    };
                    DepthSample const d0=depth(0.0),d1=depth(1.0);
                    if (queryFailed) {
                        PublishEvalFailure(ctx,1);
                        return;
                    }
                    cut=std::max({cut,objective(0,d0.d),objective(1,d1.d)});
                    std::array<Interval,kMaxDepthIntervals> work;
                    size_t workCount=0;
                    auto add=[&](double ta,double tb,DepthSample const &da,
                                 DepthSample const &db) {
                        double const weightBound=std::max(
                            wa+(wb-wa)*ta,wa+(wb-wa)*tb);
                        double const tailBound=std::max(
                            0.0,total-(arc[i]+segmentLength*ta));
                        if (cut > weightBound*tailBound + boundPad) return;
                        double const tm=(ta+tb)*0.5; DepthSample const dm=depth(tm);
                        if (queryFailed) {
                            PublishEvalFailure(ctx,
                                depthEvaluations>kMaxDepthEvaluations ? 2u : 1u);
                            return;
                        }
                        cut=std::max(cut,objective(tm,dm.d));
                        double const length=segmentLength*(tb-ta);
                        double const half=length*0.5;
                        // A 1-Lipschitz function bounded at both endpoints
                        // reaches at most (left+right+intervalLength)/2.
                        // Apply that bound to both cached half intervals.
                        double const splitUpper=std::max(
                            (da.d+dm.d+half)*0.5,(dm.d+db.d+half)*0.5);
                        GfVec3d const pa=a+(b-a)*ta,pb=a+(b-a)*tb;
                        auto fixedBoundaryUpper=[&](GfVec3d const &q) {
                            return std::max((pa-q).GetLength(),(pb-q).GetLength());
                        };
                        double const boundaryUpper=std::min({
                            fixedBoundaryUpper(da.q),fixedBoundaryUpper(dm.q),
                            fixedBoundaryUpper(db.q)});
                        double const depthUpper=std::min({da.d+length,
                            db.d+length,dm.d+half,(da.d+db.d+length)*0.5,
                            splitUpper,boundaryUpper});
                        double const maxWeight=std::max(
                            wa+(wb-wa)*ta,wa+(wb-wa)*tb);
                        double const upperX=std::clamp(
                            (depthUpper-cutThreshold)/cutBlendDepth,0.0,1.0);
                        double const upperBlend=upperX*upperX*(3.0-2.0*upperX);
                        double const remaining=std::max(
                            0.0,total-(arc[i]+segmentLength*ta));
                        double const upper=maxWeight*remaining*upperBlend;
                        double const tolerance=std::max(
                            1e-10,cap.colliders[ci].contactTolerance);
                        if (upper>cut+tolerance) {
                            if (workCount==work.size()) {
                                PublishEvalFailure(ctx,2);
                                return;
                            }
                            work[workCount++]={ta,tb,upper,da,db,dm};
                        }
                    };
                    add(0,1,d0,d1);
                    while (workCount && (!ctx.failureCode ||
                           ctx.failureCode->load(std::memory_order_acquire)==0)) {
                        size_t winner=0;
                        for (size_t wi=1;wi<workCount;++wi)
                            if (work[winner].upper<work[wi].upper) winner=wi;
                        Interval const q=work[winner];
                        work[winner]=work[--workCount];
                        double const tolerance=std::max(
                            1e-10,cap.colliders[ci].contactTolerance);
                        if (q.upper<=cut+tolerance) continue;
                        double const tm=(q.a+q.b)*0.5;
                        add(q.a,tm,q.sa,q.sm); add(tm,q.b,q.sm,q.sb);
                    }
                    if (ctx.failureCode &&
                        ctx.failureCode->load(std::memory_order_acquire)!=0) return;
                }
            }
            double const target=std::max(0.0,total-cut);
            if (target<total) {
                double const ratio=total>0.0 ? target/total : 0.0;
                size_t lo=0;
                for (size_t i=0;i<n;++i) {
                    double const sample=arc[i]*ratio;
                    while (lo+1<n && arc[lo+1]<sample) ++lo;
                    size_t const hi=std::min(lo+1,n-1);
                    double const span=arc[hi]-arc[lo];
                    double const alpha=span>0.0 ? (sample-arc[lo])/span : 0.0;
                    size_t const dst=g+i,src0=g+lo,src1=g+hi;
                    px[dst]=float(double(inPx[src0])+(double(inPx[src1])-inPx[src0])*alpha);
                    py[dst]=float(double(inPy[src0])+(double(inPy[src1])-inPy[src0])*alpha);
                    pz[dst]=float(double(inPz[src0])+(double(inPz[src1])-inPz[src0])*alpha);
                    if (view->width && view->inWidth)
                        view->width[dst]=float(double(view->inWidth[src0])+
                            (double(view->inWidth[src1])-view->inWidth[src0])*alpha);
                    for (uint32_t plane=0;plane<view->extraCvCount;++plane) {
                        UsdGenPlane &out=view->extraCv[plane];
                        UsdGenPlane const &in=view->inExtraCv[plane];
                        uint8_t const arity=std::max<uint8_t>(out.arity,1);
                        for (uint8_t component=0;component<arity;++component) {
                            size_t const d=(cvBase+dst)*arity+component;
                            size_t const x=(cvBase+src0)*arity+component;
                            size_t const y=(cvBase+src1)*arity+component;
                            if (out.type==TfToken("int"))
                                const_cast<int *>(out.i.cdata())[d]=
                                    alpha<0.5 ? in.i[x] : in.i[y];
                            else
                                const_cast<float *>(out.f.cdata())[d]=float(
                                    double(in.f[x])+(double(in.f[y])-in.f[x])*alpha);
                        }
                    }
                }
                for (size_t i=0;i<n;++i)
                    shortenedSource[g+i]=GfVec3d(px[g+i],py[g+i],pz[g+i]);
                cutPrepared=true;
            }
        }
        for (int it = 0; it < iterations; ++it) {
            if (!stiff) {
                bool flexibleChanged=false;
                auto storeFlexible=[&](size_t o,GfVec3d const &v) {
                    GfVec3d const before=current(o,it);
                    float const x=float(v[0]),y=float(v[1]),z=float(v[2]);
                    flexibleChanged = flexibleChanged ||
                        x!=float(before[0]) || y!=float(before[1]) ||
                        z!=float(before[2]);
                    px[o]=x; py[o]=y; pz[o]=z;
                };
                ClosestHit anchor;
                bool anchored = false;
                for (size_t i = 0; i < n; ++i) {
                    size_t const o = g + i;
                    size_t const cv = cvBase + o;
                    // Keep adjacent CVs on the first face entered by this
                    // strand. Independent nearest-face queries can send a
                    // lower CV to the bottom of a box and its neighbor to
                    // the top, leaving their connecting segment inside it.
                    GfVec3d const original=source(o);
                    double const off = Sanitize(offsetField.Value(curve, cv));
                    if (hasClosed && !cutMode) {
                        ClosestHit const reference =
                            closest(original, std::max(0.0,off));
                        if (reference.inside) {
                            if (!anchored || anchor.collider != reference.collider) {
                                ClosestHit entry;
                                if (i > 0) {
                                    GfVec3d const previous=source(o-1);
                                    entry = firstCrossing(previous, original);
                                }
                                anchor = entry.found &&
                                    entry.collider == reference.collider
                                    ? entry : reference;
                            }
                            anchored = true;
                        } else if (i > 0) {
                            GfVec3d const previous=source(o-1);
                            ClosestHit const crossing =
                                firstCrossing(previous, original);
                            ClosestHit correctedCrossing;
                            bool onEnteredFace = false;
                            if (anchored) {
                                GfVec3d const correctedPrevious(
                                    px[o - 1], py[o - 1], pz[o - 1]);
                                GfVec3d const candidate = current(o, it);
                                correctedCrossing = firstCrossing(
                                    correctedPrevious, candidate);
                                ClosestHit anchoredFace = anchor;
                                onEnteredFace = ClosestOnHitFacet(
                                    cap, candidate, &anchoredFace) &&
                                    anchoredFace.dist2 <= 1e-10;
                            }
                            ClosestHit const selected = crossing.found
                                ? crossing : correctedCrossing;
                            if (selected.found &&
                                (!anchored || selected.collider != anchor.collider))
                                anchor = selected;
                            anchored = selected.found || onEnteredFace;
                        } else {
                            anchored = false;
                        }
                    }
                    double const push =
                        Sanitize(pushField.Value(curve, cv));
                    double const m = std::clamp(
                        Sanitize(maskField.Value(curve, cv)), 0.0, 1.0);
                    if (push == 0.0 || m == 0.0) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    GfVec3d const pos = current(o, it);
                    if (cutMode) {
                        // Cut-flex collision is a projected iteration over the
                        // geometry actually stored by this sweep.  Correct the
                        // current endpoint for the first incoming-segment
                        // constraint, or its endpoint constraint when there is
                        // no crossing. Other colliders are re-evaluated on the
                        // next sweep instead of selecting a discontinuous
                        // closest point on the original entry facet.
                        ClosestHit entry;
                        GfVec3d previous=pos;
                        if (i>0) {
                            previous=GfVec3d(px[o-1],py[o-1],pz[o-1]);
                            entry=firstCrossing(previous,pos,true);
                        }
                        if (entry.found) {
                            GfVec3d normal=entry.normal;
                            double const length=normal.GetLength();
                            if (length>0.0) normal/=length;
                            double const side=GfDot(pos-entry.q,normal);
                            double const amount=std::max(0.0,off-side);
                            storeFlexible(o,pos+normal*(amount*push*m));
                            continue;
                        }
                        ClosestHit const hit=closest(pos, std::max(0.0,off));
                        if (!hit.found) {
                            if (it==0) copyInput(o);
                            continue;
                        }
                        double const distance=std::sqrt(hit.dist2);
                        double const pen=hit.inside ? off+distance
                                                   : off-distance;
                        if (!(pen>0.0)) {
                            if (it==0) copyInput(o);
                            continue;
                        }
                        GfVec3d const dir=HitDirection(
                            ctx.desc,cap,hit,pos,hit.inside);
                        storeFlexible(o,pos+dir*(pen*push*m));
                        continue;
                    }
                    if (anchored) {
                        ClosestHit face = anchor;
                        bool const faceReady=ClosestOnHitFacet(
                            cap,original,&face);
                        if (faceReady) {
                            GfVec3d const outward = HitDirection(
                                ctx.desc, cap, face, original, true);
                            GfVec3d const target = face.q + outward * off;
                            storeFlexible(o, pos + (target - pos) * (push * m));
                            continue;
                        }
                    }
                    ClosestHit const hit =
                        closest(pos, std::max(0.0,off));
                    if (!hit.found) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    double const distance = std::sqrt(hit.dist2);
                    double const pen = hit.inside ? off + distance
                                                  : off - distance;
                    if (!(pen > 0.0)) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    double const s = pen * push * m;
                    if (s == 0.0) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    GfVec3d const dir = HitDirection(
                        ctx.desc, cap, hit, pos, hit.inside);
                    storeFlexible(o, pos + dir * s);
                }
                // Exact fixed point only: finish the complete sweep, then
                // stop when every stored float CV is numerically unchanged.
                // Oscillation and over-relaxation continue to the authored
                // iteration count because any changed component keeps going.
                if (!flexibleChanged) break;
            } else {
                // First penetrating CV hinges the strand: the root case
                // translates rigidly, otherwise the downstream CVs rotate
                // about the last clean CV. Both preserve segment lengths.
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
                    double const off = Sanitize(offsetField.Value(curve, cv));
                    ClosestHit const hit =
                        closest(pos, std::max(0.0,off));
                    ClosestHit entry;
                    if (i > 0)
                        entry = firstCrossing(current(g + i - 1, it), pos);
                    bool const solidContact = entry.found || hit.inside;
                    double pen = 0.0;
                    GfVec3d dir(0.0, 0.0, 1.0);
                    ClosestHit face = entry.found ? entry : hit;
                    if (face.found) {
                        if (entry.found)
                            ClosestOnHitFacet(cap, pos, &face);
                        double const distance = std::sqrt(face.dist2);
                        pen = solidContact ? off + distance : off - distance;
                        dir = HitDirection(
                            ctx.desc, cap, face, pos, solidContact);
                    }
                    if (pen > 0.0) {
                        hinge = i;
                        target = pos + dir * (pen * push * m);
                        bool lateralOverride = false;
                        // Stiff resolution retains its established plane
                        // swivel and whole-span mirror preference in both modes.
                        if (entry.found && i > 0 && push * m > 0.0) {
                            // A closest point clamped to an entry triangle's
                            // edge can put the endpoint outside while its
                            // connecting segment still cuts through the
                            // solid. Keep the whole hinge segment on the
                            // exterior side of the actual entry face plane.
                            // The fixed-length target is the smallest swivel
                            // satisfying that one-sided constraint.
                            GfVec3d const h = current(g + i - 1, it);
                            GfVec3d const from = pos - h;
                            double const length = from.GetLength();
                            GfVec3d normal = entry.normal;
                            if (GfDot(normal, from) > 0.0) normal = -normal;
                            double const startSide =
                                GfDot(h - entry.q, normal);
                            double const edge = std::max({
                                (entry.b - entry.a).GetLength(),
                                (entry.c - entry.b).GetLength(),
                                (entry.a - entry.c).GetLength()});
                            double const margin =
                                1e-5 * std::max(edge, length);
                            double const normalTarget = off + margin - startSide;
                            double const normalNow = GfDot(from, normal);
                            if (length > 0.0 &&
                                startSide >= -margin &&
                                normalTarget > normalNow &&
                                normalTarget < length) {
                                GfVec3d tangent =
                                    from - normal * normalNow;
                                if (tangent.GetLength() < 1e-5 * length) {
                                    tangent = source(o)-source(o-1);
                                    tangent -= normal * GfDot(tangent, normal);
                                }
                                if (tangent.GetLength() < 1e-5 * length) {
                                    tangent = h -
                                        cap.colliders[entry.collider].center;
                                    tangent -= normal * GfDot(tangent, normal);
                                }
                                if (tangent.GetLength() < 1e-8) {
                                    GfVec3d const axis =
                                        std::abs(normal[0]) < 0.9
                                        ? GfVec3d(1, 0, 0)
                                        : GfVec3d(0, 0, 1);
                                    tangent = GfCross(normal, axis);
                                }
                                tangent.Normalize();
                                GfVec3d const planeTarget =
                                    h + normal * normalTarget +
                                    tangent * std::sqrt(length * length -
                                        normalTarget * normalTarget);
                                // The minimal hinge swivel can route the
                                // downstream span through another part of a
                                // broad solid. Use the established longer-
                                // reach swivel if any candidate CV or whole
                                // segment would still cross that solid.
                                GfVec3f const fromF{
                                    float(from[0]), float(from[1]),
                                    float(from[2])};
                                GfVec3d const to = planeTarget - h;
                                GfVec3f const toF{
                                    float(to[0]), float(to[1]),
                                    float(to[2])};
                                bool spanClear = true;
                                GfVec3d previous = h;
                                for (size_t j = i; j < n; ++j) {
                                    GfVec3d const rel = current(g + j, it) - h;
                                    GfVec3f const relF{
                                        float(rel[0]), float(rel[1]),
                                        float(rel[2])};
                                    GfVec3f const rotated = opUtil::RotateOnto(
                                        relF, fromF, toF,
                                        float(std::clamp(push * m, 0.0, 1.0)));
                                    GfVec3d const candidate = h +
                                        GfVec3d(double(rotated[0]),
                                                 double(rotated[1]),
                                                 double(rotated[2]));
                                    ClosestHit const candidateHit =
                                        closest(candidate,0.0);
                                    ClosestHit const crossing =
                                        firstCrossing(previous, candidate);
                                    if ((candidateHit.inside &&
                                         candidateHit.collider ==
                                             entry.collider) ||
                                        (crossing.found &&
                                         crossing.collider == entry.collider)) {
                                        spanClear = false;
                                        break;
                                    }
                                    previous = candidate;
                                }
                                // Ordinary collide retains its established
                                // entry-collider whole-tail rule here.
                                if (spanClear) {
                                    target = planeTarget;
                                    lateralOverride = true;
                                }
                            }
                        }
                        if (!lateralOverride && solidContact && i > 0 &&
                            push * m > 0.0) {
                            GfVec3d const h = current(g + i - 1, it);
                            GfVec3d const from = pos - h, toward = target - h;
                            GfVec3f const fromF{
                                float(from[0]),float(from[1]),float(from[2])};
                            double const lf = from.GetLength();
                            double const lt = toward.GetLength();
                            if (lf > 0.0 &&
                                (lt == 0.0 ||
                                 GfDot(from, -dir) > 1e-8 ||
                                 (lt > 0.0 &&
                                  GfDot(from, toward) / (lf * lt) > 0.98))) {
                                // Shortening the hinge vector cannot rotate
                                // a rigid strand out. Preserve its original
                                // travel along the entry face where possible;
                                // radial distance from the solid center only
                                // breaks an axial tie (upright bottom hit).
                                GfVec3d const outward = dir;
                                GfVec3d lateral=source(o)-source(o-1);
                                lateral -= outward * GfDot(lateral, outward);
                                if (lateral.GetLength() < 1e-5 * lf) {
                                    lateral = h -
                                        cap.colliders[face.collider].center;
                                    lateral -= outward * GfDot(lateral, outward);
                                }
                                if (lateral.GetLength() < 1e-8) {
                                    GfVec3d const axis =
                                        std::abs(outward[0]) < 0.9
                                        ? GfVec3d(1, 0, 0)
                                        : GfVec3d(0, 0, 1);
                                    lateral = GfCross(outward, axis);
                                }
                                lateral.Normalize();
                                double reach = lf;
                                for (size_t j = i + 1; j < n; ++j)
                                    reach = std::max(reach,
                                        (current(g + j, it) - h).GetLength());
                                double const clearance = std::max(0.0,
                                    GfDot(face.q - h, -outward) - 1e-5);
                                double const axial = std::clamp(
                                    clearance / reach, 0.0, 0.95);
                                GfVec3d const desired =
                                    lateral * std::sqrt(1.0 - axial * axial) -
                                    outward * axial;
                                target = h + desired * lf;
                                auto spanClears = [&](GfVec3d const &hingeTarget,
                                                      size_t *badIndex) {
                                    GfVec3f const hingeToF{
                                        float(hingeTarget[0] - h[0]),
                                        float(hingeTarget[1] - h[1]),
                                        float(hingeTarget[2] - h[2])};
                                    GfVec3d previous = h;
                                    for (size_t j = i; j < n; ++j) {
                                        GfVec3d const rel =
                                            current(g + j, it) - h;
                                        GfVec3f const relF{
                                            float(rel[0]), float(rel[1]),
                                            float(rel[2])};
                                        GfVec3f const rotated = opUtil::RotateOnto(
                                            relF, fromF, hingeToF,
                                            float(std::clamp(push * m, 0.0, 1.0)));
                                        GfVec3d const candidate = h + GfVec3d(
                                            double(rotated[0]),double(rotated[1]),
                                            double(rotated[2]));
                                        ClosestHit const candidateHit =
                                            closest(candidate,0.0);
                                        ClosestHit const crossing =
                                            firstCrossing(previous, candidate);
                                        if (candidateHit.inside || crossing.found) {
                                            *badIndex = j;
                                            return false;
                                        }
                                        previous = candidate;
                                    }
                                    return true;
                                };
                                size_t primaryBad=n, mirrorBad=n;
                                bool const primaryClear = spanClears(
                                    target,&primaryBad);
                                bool mirrorClear = false;
                                if (!primaryClear) {
                                    GfVec3d const mirrored =
                                        -lateral * std::sqrt(
                                            1.0 - axial * axial) -
                                        outward * axial;
                                    GfVec3d const mirroredTarget =
                                        h + mirrored * lf;
                                    mirrorClear = spanClears(
                                        mirroredTarget,&mirrorBad);
                                    // Prefer a fully clear span.  When neither
                                    // direction clears the whole tail, accept
                                    // the direction whose all-collider check
                                    // advances the first blocked CV farther
                                    // downstream.  This makes verified local
                                    // progress without accepting the blocked
                                    // portion as solved; the next sweep handles
                                    // that later hinge.
                                    if (mirrorClear || mirrorBad > primaryBad)
                                        target = mirroredTarget;
                                }
                                lateralOverride = true;
                            }
                        }
                        hingeMask = lateralOverride
                            ? std::clamp(push * m, 0.0, 1.0) : m;
                        break;
                    }
                }
                if (hinge == n) {
                    // Nothing changed this sweep, so later identical sweeps
                    // cannot find a new hinge on this strand.
                    if (it == 0)
                        for (size_t i = 0; i < n; ++i) copyInput(g + i);
                    break;
                }
                if (target == current(g + hinge, it)) {
                    // Zero push (masked / zero pushAmount): identity.
                    if (it == 0)
                        for (size_t i = 0; i < n; ++i) copyInput(g + i);
                    continue;
                }
                size_t const ho = g + hinge;
                GfVec3d const hp = current(ho, it);
                if (hinge == 0) {
                    GfVec3d const shift = target - hp;
                    for (size_t i = 0; i < n; ++i)
                        store(g + i, current(g + i, it) + shift);
                    continue;
                }
                if (it == 0)
                    for (size_t i = 0; i < hinge; ++i) copyInput(g + i);
                GfVec3d const h = current(g + hinge - 1, it);
                GfVec3d const from = hp - h, to = target - h;
                if (from == GfVec3d(0.0) || to == GfVec3d(0.0)) {
                    // Collapsed hinge segment: rigid translation instead.
                    GfVec3d const shift = target - hp;
                    for (size_t i = hinge; i < n; ++i)
                        store(g + i, current(g + i, it) + shift);
                    continue;
                }
                GfVec3f const fromF{float(from[0]), float(from[1]), float(from[2])};
                GfVec3f const toF{float(to[0]), float(to[1]), float(to[2])};
                for (size_t i = hinge; i < n; ++i) {
                    size_t const o = g + i;
                    GfVec3d const rel = current(o, it) - h;
                    GfVec3f const relF{float(rel[0]), float(rel[1]), float(rel[2])};
                    GfVec3f const rotated =
                        opUtil::RotateOnto(relF, fromF, toF, float(hingeMask));
                    store(o, h + GfVec3d(double(rotated[0]),
                                         double(rotated[1]),
                                         double(rotated[2])));
                }
            }
        }
    }
}

}  // namespace usdGen
