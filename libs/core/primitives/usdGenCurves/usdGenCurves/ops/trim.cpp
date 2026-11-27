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
#include "usdGenCurves/ops/trim.h"
#include "usdGenSurface/surfaceInput.h"

#include "usdGenCurves/ops/lengthRange.h"
#include "usdGenCurves/ops/opUtil.h"
#include "usdGenCurves/ops/strandUtil.h"
#include "usdGen/exec/scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "usdGen/registries/registrar.h"
#include "usdGen/api/token.h"
#include "usdGenMath/math.h"
#include "usdGen/api/span.h"


namespace usdGen {

namespace {

using namespace trimP;
using D = expr::Domain;
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
// input, surface, seed, the structural parameters, then enabled and mask.
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"trim:angle", UsdGenParamClass::Capture},
    {"trim:invert", UsdGenParamClass::Capture},
    {"trim:distance", UsdGenParamClass::Capture},
    {"trim:normalSearch", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
};
constexpr char const *kReferences[] = {"trim:curves"};
constexpr char const *kTokens[] = {"trimAngle", "trimInvert", "trimDistance", "normal"};
std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenTrimOp>(r);
}

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
    UsdGenVec3f a, b, c;
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
    void Query(UsdGenVec3f const &lo, UsdGenVec3f const &hi, F const &f) const
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
            for (UsdGenVec3f const *q : {&t.a, &t.b, &t.c})
                for (int k = 0; k < 3; ++k) {
                    node.lo[k] = std::min(node.lo[k], (*q)[k]);
                    node.hi[k] = std::max(node.hi[k], (*q)[k]);
                }
            UsdGenVec3f const centroid = (t.a + t.b + t.c) / 3.0f;
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

/// The named plane among the per-curve (cv false) or per-CV planes, in
/// 'storage'; null when absent.
UsdGenPlaneView const *FindPlane(UsdGenCurveSetView const &curves, UsdGenToken const &name, bool cv,
                                 UsdGenPlaneView *storage)
{
    bool perCv = false;
    return curves.FindPlane(name, storage, &perCv) && perCv == cv ? storage : nullptr;
}

/// A per-curve value of a uniform (or constant) float/int plane; false when
/// absent or non-finite.
bool CurveValue(UsdGenPlaneView const *plane, size_t curve, size_t curves, double *out)
{
    if (!plane || plane->arity < 1) return false;
    bool const constant = plane->interpolation == UsdGenTokens().constant;
    if (!constant && plane->interpolation != UsdGenTokens().uniform) return false;
    size_t const element = constant ? 0 : curve;
    if (!constant && curve >= curves) return false;
    size_t const index = element * plane->arity;
    double v;
    if (plane->type == UsdGenTokens().int_) {
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
bool SegmentHitsTriangle(UsdGenVec3f const &p0, UsdGenVec3f const &p1, Triangle const &tri, float *t)
{
    UsdGenVec3d const a(tri.a), b(tri.b), c(tri.c), o(p0);
    UsdGenVec3d const dir = UsdGenVec3d(p1) - o;
    UsdGenVec3d const e1 = b - a, e2 = c - a;
    UsdGenVec3d const h = UsdGenCross(dir, e2);
    double const det = UsdGenDot(e1, h);
    if (!(std::fabs(det) > 1e-30)) return false;  // parallel to the blade
    double const f = 1.0 / det;
    UsdGenVec3d const s = o - a;
    double const u = f * UsdGenDot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    UsdGenVec3d const q = UsdGenCross(s, e1);
    double const v = f * UsdGenDot(dir, q);
    if (v < 0.0 || u + v > 1.0) return false;
    double const tt = f * UsdGenDot(e2, q);
    if (tt < 0.0 || tt > 1.0) return false;
    *t = float(tt);
    return true;
}

}  // namespace

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
    UsdGenCurveSetView const curves = UsdGenCurveSetView::Of(*ctx.resolvedReferences[0]->value);
    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(curves, &spans, &error))
        return fail("trim curve topology: " + error);
    if (curves.px.size() != curves.totalCvs || curves.py.size() != curves.totalCvs ||
        curves.pz.size() != curves.totalCvs)
        return fail("trim curve point planes do not match their CV count");
    size_t const curveCount = curves.totalCurves;

    UsdGenPlaneView planes[4];
    UsdGenPlaneView const *anglePlane = FindPlane(curves, Record().Token(kTokAngle), false, &planes[0]);
    UsdGenPlaneView const *invertPlane = FindPlane(curves, Record().Token(kTokInvert), false, &planes[1]);
    UsdGenPlaneView const *distancePlane =
        FindPlane(curves, Record().Token(kTokDistance), false, &planes[2]);
    UsdGenPlaneView const *normalPlane = FindPlane(curves, Record().Token(kTokNormal), true, &planes[3]);
    bool const storedNormals = normalPlane && normalPlane->type == UsdGenTokens().float_ &&
        normalPlane->arity == 3 && normalPlane->f.size() == size_t(curves.totalCvs) * 3;

    // Closest-point normals come from the bound surface (rest pose, where the
    // trim curves are drawn); built only when some point needs one.
    strand::TriangleBvh bvh;
    bool bvhBuilt = false;
    float const searchSq = float(normalSearch * normalSearch);
    auto surfaceNormal = [&](UsdGenVec3f const &at, UsdGenVec3f *n) {
        if (!bvhBuilt) {
            bvhBuilt = true;
            if (ctx.desc && ctx.surface < UsdGenSurfaceInputs(*ctx.desc).size())
                strand::BuildSurfaceBvh(UsdGenSurfaceAt(*ctx.desc, ctx.surface), true, &bvh);
        }
        if (bvh.Empty()) return false;
        strand::ClosestHit const hit = bvh.Closest(at, searchSq);
        if (!hit.hit) return false;
        *n = bvh.Normal(hit.triangle, hit.bary);
        return true;
    };

    UsdGenVec3f const up(0.0f, 1.0f, 0.0f);
    std::vector<Triangle> blades;
    std::vector<UsdGenVec3f> Q;
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
            UsdGenVec3f const T = strand::Normalize(P(std::min(k + 1, m - 1)) - P(k > 0 ? k - 1 : 0),
                                                up, 1e-12f);
            UsdGenVec3f N = up;
            bool haveNormal = false;
            if (storedNormals) {
                float const *nf = normalPlane->f.cdata() + 3 * (first + k);
                UsdGenVec3f const stored(nf[0], nf[1], nf[2]);
                if (stored.GetLength() > 1e-12f) {
                    N = stored;
                    haveNormal = true;
                }
            }
            if (!haveNormal) {
                UsdGenVec3f hitNormal;
                if (surfaceNormal(P(k), &hitNormal)) N = hitNormal;
            }
            UsdGenVec3f const Np = strand::Normalize(N - T * UsdGenDot(N, T), up, 1e-12f);
            UsdGenVec3f const B = strand::Normalize(UsdGenCross(T, Np), UsdGenVec3f(1.0f, 0.0f, 0.0f), 1e-12f);
            UsdGenVec3f const R = strand::Rotate(B, T, theta);
            UsdGenVec3f const D = strand::Normalize(UsdGenCross(R, T), Np, 1e-12f);
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
            UsdGenVec3f const a = strand::Load(pts, i - 1), b = strand::Load(pts, i);
            float const len = (b - a).GetLength();
            if (len > 1e-8f) {
                UsdGenVec3f lo, hi;
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

UsdGenOpId UsdGenRegisterTrimOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenTrim";
    d.displayName = "Trim";
    d.create = &Create;
    d.params = UsdGenSpan(kParams);
    d.referenceInputs = UsdGenSpan(kReferences);
    d.tokens = UsdGenSpan(kTokens);
    return r.RegisterOp(d);
}

}  // namespace usdGen
