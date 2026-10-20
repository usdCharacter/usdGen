// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenTrimOp implementation (reference behaviour of imported
// grooms).
//
// Blades: every curve of the `trim:curves` reference set (>= 2 points) gets a
// ribbon standing on it. Per curve point k:
//   T  = normalize(P[min(k+1,m-1)] - P[max(k-1,0)])           (fallback +Y)
//   N  = the set's per-point `normal` plane (vertex float3) when present,
//        else the bound surface's normal at the closest point within
//        usdGen:trim:normalSearch, else +Y
//   N' = normalize(N - (N.T) T), B = normalize(T x N')
//   R  = rotate(B about T by theta), D = normalize(R x T)
//        (= cos(theta) N' + sin(theta) B)
//   Q  = P + d D
// with d = max(1e-4, clamp(distance, 0.01, 2)) scene units and
// theta = clamp(angle, 0, 89) deg * (invert ? -1 : +1). Each curve segment
// longer than 1e-6 contributes the quad {P[k-1], P[k], Q[k-1], Q[k]} as the
// triangles (P[k-1], Q[k], Q[k-1]) and (P[k-1], P[k], Q[k]). Per-curve
// overrides come from the set's uniform planes trimAngle / trimInvert /
// trimDistance (a non-finite value falls back to the operator parameter).
//
// Strands: s_cut = the smallest arc length acc + t*len (> 1e-6) at which a
// strand segment crosses a blade triangle, i.e. the FIRST crossing from the
// root. When 1e-6 < s_cut < L - 1e-6 the strand is resampled to its own CV
// count uniformly over [0, max(s_cut, 1e-4)]; strands that miss every blade
// are untouched.
#include "usdGen/ops/trim.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/lengthRange.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

TfTokenVector const kTrimReferences{TfToken("trim:curves")};
TfToken const kAngle("trim:angle"), kInvert("trim:invert"), kDistance("trim:distance"),
    kNormalSearch("trim:normalSearch");
TfToken const kPlaneAngle("trimAngle"), kPlaneInvert("trimInvert"),
    kPlaneDistance("trimDistance"), kPlaneNormal("normal");

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    if (!dispatcher || count < 256) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload);
}

struct Triangle
{
    GfVec3f a, b, c;
};

/// Median-split AABB tree over blade triangles (deterministic, small).
class BladeTree
{
public:
    void Build(std::vector<Triangle> triangles)
    {
        _tris = std::move(triangles);
        _nodes.clear();
        _order.resize(_tris.size());
        for (size_t i = 0; i < _order.size(); ++i) _order[i] = int(i);
        if (_tris.empty()) return;
        _nodes.reserve(2 * _tris.size());
        BuildRange(0, int(_tris.size()));
    }
    bool Empty() const { return _tris.empty(); }
    /// Calls f(triangle) for every triangle whose box overlaps [lo, hi].
    template <class F>
    void Query(GfVec3f const &lo, GfVec3f const &hi, F const &f) const
    {
        if (_nodes.empty()) return;
        int stack[96];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            Node const &node = _nodes[size_t(stack[--top])];
            bool overlap = true;
            for (int k = 0; k < 3 && overlap; ++k)
                overlap = node.lo[k] <= hi[k] && node.hi[k] >= lo[k];
            if (!overlap) continue;
            if (node.left < 0) {
                for (int i = node.first; i < node.first + node.count; ++i)
                    f(_tris[size_t(_order[size_t(i)])]);
                continue;
            }
            if (top + 2 > 96) continue;  // median split keeps depth ~log2
            stack[top++] = node.right;
            stack[top++] = node.left;
        }
    }

private:
    struct Node { float lo[3], hi[3]; int left = -1, right = -1, first = 0, count = 0; };
    int BuildRange(int first, int count)
    {
        int const index = int(_nodes.size());
        _nodes.push_back(Node());
        Node node;
        float clo[3], chi[3];
        for (int k = 0; k < 3; ++k) {
            node.lo[k] = clo[k] = std::numeric_limits<float>::max();
            node.hi[k] = chi[k] = -std::numeric_limits<float>::max();
        }
        for (int i = first; i < first + count; ++i) {
            Triangle const &t = _tris[size_t(_order[size_t(i)])];
            for (GfVec3f const *q : {&t.a, &t.b, &t.c})
                for (int k = 0; k < 3; ++k) {
                    node.lo[k] = std::min(node.lo[k], (*q)[k]);
                    node.hi[k] = std::max(node.hi[k], (*q)[k]);
                }
            GfVec3f const centroid = (t.a + t.b + t.c) / 3.0f;
            for (int k = 0; k < 3; ++k) {
                clo[k] = std::min(clo[k], centroid[k]);
                chi[k] = std::max(chi[k], centroid[k]);
            }
        }
        if (count <= 4) {
            node.first = first;
            node.count = count;
            _nodes[size_t(index)] = node;
            return index;
        }
        int axis = 0;
        for (int k = 1; k < 3; ++k)
            if (chi[k] - clo[k] > chi[axis] - clo[axis]) axis = k;
        auto key = [&](int i) {
            Triangle const &t = _tris[size_t(i)];
            return (t.a[axis] + t.b[axis] + t.c[axis]) / 3.0f;
        };
        int const mid = first + count / 2;
        std::nth_element(_order.begin() + first, _order.begin() + mid,
                         _order.begin() + first + count, [&](int a, int b) {
                             float const ka = key(a), kb = key(b);
                             return ka < kb || (ka == kb && a < b);
                         });
        node.left = BuildRange(first, mid - first);
        node.right = BuildRange(mid, first + count - mid);
        _nodes[size_t(index)] = node;
        return index;
    }
    std::vector<Triangle> _tris;
    std::vector<int> _order;
    std::vector<Node> _nodes;
};

UsdGenPlane const *FindPlane(std::vector<UsdGenPlane> const &planes, TfToken const &name)
{
    for (UsdGenPlane const &p : planes)
        if (p.name == name) return &p;
    return nullptr;
}

/// A per-curve value of a uniform (or constant) float/int plane; false when
/// absent or non-finite.
bool CurveValue(UsdGenPlane const *plane, size_t curve, size_t curves, double *out)
{
    if (!plane || plane->arity < 1) return false;
    bool const constant = plane->interpolation == TfToken("constant");
    if (!constant && plane->interpolation != TfToken("uniform")) return false;
    size_t const element = constant ? 0 : curve;
    if (!constant && curve >= curves) return false;
    size_t const index = element * plane->arity;
    double v;
    if (plane->type == TfToken("int")) {
        if (index >= plane->i.size()) return false;
        v = double(plane->i[index]);
    } else {
        if (index >= plane->f.size()) return false;
        v = double(plane->f[index]);
    }
    if (!std::isfinite(v)) return false;
    *out = v;
    return true;
}

/// Segment p0 -> p1 against triangle abc (Moller-Trumbore, double precision);
/// writes the segment parameter in [0, 1] of the hit.
bool SegmentHitsTriangle(GfVec3f const &p0, GfVec3f const &p1, Triangle const &tri, float *t)
{
    GfVec3d const a(tri.a), b(tri.b), c(tri.c), o(p0);
    GfVec3d const dir = GfVec3d(p1) - o;
    GfVec3d const e1 = b - a, e2 = c - a;
    GfVec3d const h = GfCross(dir, e2);
    double const det = GfDot(e1, h);
    if (!(std::fabs(det) > 1e-30)) return false;  // parallel to the blade
    double const f = 1.0 / det;
    GfVec3d const s = o - a;
    double const u = f * GfDot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    GfVec3d const q = GfCross(s, e1);
    double const v = f * GfDot(dir, q);
    if (v < 0.0 || u + v > 1.0) return false;
    double const tt = f * GfDot(e2, q);
    if (tt < 0.0 || tt > 1.0) return false;
    *t = float(tt);
    return true;
}

}  // namespace

TfSpan<const TfToken> UsdGenTrimOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(kTrimReferences.data(), kTrimReferences.size());
}

TfTokenVector UsdGenTrimOp::StructuralParameters() const
{
    return {kAngle, kInvert, kDistance, kNormalSearch};
}

std::vector<UsdGenStrandPlaneSpec> UsdGenTrimOp::OutputPlaneSpecs() const
{
    return {};
}

bool UsdGenTrimOp::Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                           std::vector<UsdGenStrandPlaneData> &, UsdGenDiagnostics *diag)
{
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error(Label() + ": " + message);
        return false;
    };
    UsdGenParamView const *p = ctx.params;
    double const angle = p ? p->GetDouble(kAngle, 0.0) : 0.0;
    bool const invert = p ? p->GetBool(kInvert, false) : false;
    double const distance = p ? p->GetDouble(kDistance, 0.1) : 0.1;
    double const normalSearch = p ? p->GetDouble(kNormalSearch, 0.08) : 0.08;
    if (!std::isfinite(angle) || !std::isfinite(distance) || !std::isfinite(normalSearch) ||
        normalSearch < 0.0)
        return fail("usdGen:trim:angle/distance/normalSearch must be finite (normalSearch >= 0)");

    // No trim curves bound: nothing to cut.
    if (ctx.referenceCount == 0 || !ctx.resolvedReferences || !ctx.resolvedReferences[0] ||
        !ctx.resolvedReferences[0]->value)
        return true;
    UsdGenCurveBuffer const &curves = ctx.resolvedReferences[0]->value->buffer;
    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(curves, &spans, &error))
        return fail("trim curve topology: " + error);
    if (curves.px.size() != curves.totalCvs || curves.py.size() != curves.totalCvs ||
        curves.pz.size() != curves.totalCvs)
        return fail("trim curve point planes do not match their CV count");
    size_t const curveCount = curves.totalCurves;

    UsdGenPlane const *anglePlane = FindPlane(curves.extraCurve, kPlaneAngle);
    UsdGenPlane const *invertPlane = FindPlane(curves.extraCurve, kPlaneInvert);
    UsdGenPlane const *distancePlane = FindPlane(curves.extraCurve, kPlaneDistance);
    UsdGenPlane const *normalPlane = FindPlane(curves.extraCv, kPlaneNormal);
    bool const storedNormals = normalPlane && normalPlane->type == TfToken("float") &&
        normalPlane->arity == 3 && normalPlane->f.size() == size_t(curves.totalCvs) * 3;

    // Closest-point normals come from the bound surface (rest pose, where the
    // trim curves are drawn); built only when some point needs one.
    strand::TriangleBvh bvh;
    bool bvhBuilt = false;
    float const searchSq = float(normalSearch * normalSearch);
    auto surfaceNormal = [&](GfVec3f const &at, GfVec3f *n) {
        if (!bvhBuilt) {
            bvhBuilt = true;
            if (ctx.desc && ctx.surface < ctx.desc->surfaces.size())
                strand::BuildSurfaceBvh(ctx.desc->surfaces[ctx.surface], true, &bvh);
        }
        if (bvh.Empty()) return false;
        strand::ClosestHit const hit = bvh.Closest(at, searchSq);
        if (!hit.hit) return false;
        *n = bvh.Normal(hit.triangle, hit.bary);
        return true;
    };

    GfVec3f const up(0.0f, 1.0f, 0.0f);
    std::vector<Triangle> blades;
    std::vector<GfVec3f> Q;
    for (size_t c = 0; c < curveCount; ++c) {
        size_t const first = spans[c], m = size_t(spans[c + 1] - spans[c]);
        if (m < 2) continue;
        double a = angle, d = distance, inv = invert ? 1.0 : 0.0;
        CurveValue(anglePlane, c, curveCount, &a);
        CurveValue(distancePlane, c, curveCount, &d);
        CurveValue(invertPlane, c, curveCount, &inv);
        float const height = std::max(1e-4f, std::min(std::max(float(d), 0.01f), 2.0f));
        float const theta = std::min(std::max(float(a), 0.0f), 89.0f) * 0.017453292f *
                            (inv != 0.0 ? -1.0f : 1.0f);
        auto P = [&](size_t k) { return opUtil::Point(curves, first + k); };
        Q.resize(m);
        for (size_t k = 0; k < m; ++k) {
            GfVec3f const T = strand::Normalize(P(std::min(k + 1, m - 1)) - P(k > 0 ? k - 1 : 0),
                                                up, 1e-12f);
            GfVec3f N = up;
            bool haveNormal = false;
            if (storedNormals) {
                float const *nf = normalPlane->f.cdata() + 3 * (first + k);
                GfVec3f const stored(nf[0], nf[1], nf[2]);
                if (stored.GetLength() > 1e-12f) {
                    N = stored;
                    haveNormal = true;
                }
            }
            if (!haveNormal) {
                GfVec3f hitNormal;
                if (surfaceNormal(P(k), &hitNormal)) N = hitNormal;
            }
            GfVec3f const Np = strand::Normalize(N - T * GfDot(N, T), up, 1e-12f);
            GfVec3f const B = strand::Normalize(GfCross(T, Np), GfVec3f(1.0f, 0.0f, 0.0f), 1e-12f);
            GfVec3f const R = strand::Rotate(B, T, theta);
            GfVec3f const D = strand::Normalize(GfCross(R, T), Np, 1e-12f);
            Q[k] = P(k) + D * height;
        }
        for (size_t k = 1; k < m; ++k) {
            if (!((P(k) - P(k - 1)).GetLength() > 1e-6f)) continue;
            blades.push_back({P(k - 1), Q[k], Q[k - 1]});
            blades.push_back({P(k - 1), P(k), Q[k]});
        }
    }
    if (blades.empty()) return true;
    BladeTree tree;
    tree.Build(std::move(blades));

    uint32_t const R = strands.Count();
    ParallelFor(ctx.dispatcher, R, [&](size_t c) {
        size_t const n = strands.Size(uint32_t(c));
        if (n < 2) return;
        float *pts = strands.P(uint32_t(c));
        float sCut = std::numeric_limits<float>::infinity();
        float acc = 0.0f;
        for (size_t i = 1; i < n; ++i) {
            if (acc >= sCut) break;  // later crossings are farther from the root
            GfVec3f const a = strand::Load(pts, i - 1), b = strand::Load(pts, i);
            float const len = (b - a).GetLength();
            if (len > 1e-8f) {
                GfVec3f lo, hi;
                for (int k = 0; k < 3; ++k) {
                    lo[k] = std::min(a[k], b[k]);
                    hi[k] = std::max(a[k], b[k]);
                }
                tree.Query(lo, hi, [&](Triangle const &tri) {
                    float t = 0.0f;
                    if (!SegmentHitsTriangle(a, b, tri, &t)) return;
                    float const s = acc + t * len;
                    if (s > 1e-6f) sCut = std::min(sCut, s);
                });
            }
            acc += len;
        }
        if (!std::isfinite(sCut)) return;
        thread_local std::vector<float> original, cum;
        original.assign(pts, pts + 3 * n);
        float const L = strand::ArcLengths(original.data(), n, &cum);
        if (!(sCut > 1e-6f && sCut < L - 1e-6f)) return;
        sCut = std::max(sCut, 1e-4f);
        lengthRange::ResampleTo(original.data(), n, cum, L,
                                lengthRange::TipDirection(original.data(), n), sCut, n, pts);
    });
    return true;
}

}  // namespace usdGen
