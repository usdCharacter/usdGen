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
// Closest-point search: Capture resolves and validates every collider mesh
// and indexes the CURRENT collider points — flexible/stiff in one exact BVH
// over all colliders (ops/collideBvh.h: the former brute-force scan's
// results, first-win ties in node order / ascending faces, double
// precision), relax in one strand::TriangleBvh per collider. Evaluate only
// queries the captured indices (no allocation). An animated collider
// re-captures (surfaceGeneration digest terms), so the indices are never
// stale. Value-class edits (offset/pushAmount/iterations/clearance) sweep
// without recapturing (displace precedent), so Evaluate additionally
// sanitizes: a non-finite connected value reads as 0, iterations < 1 copies
// the input, and an unknown resolveType falls back to flexible. Capture
// still hard-errors on every one of those states; the Evaluate fallbacks
// only cover a literal edited after capture and can never write NaN.
#include "usdGen/ops/collide.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/collideBvh.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

const TfToken sOffset{"offset"}, sPushAmount{"pushAmount"};
const TfToken sIterations{"iterations"}, sResolveType{"resolveType"};
const TfToken sFlexible{"flexible"}, sStiff{"stiff"}, sMask{"mask"};
const TfToken sRelax{"relax"};
const TfToken sClearanceRoot{"clearance:root"}, sClearanceTip{"clearance:tip"};
const TfToken sRelaxIterations{"relax:iterations"};
const TfToken sUnresolved{"collideUnresolved"};

// Relax (imported-groom push-out) constants, reference values.
constexpr float kRelaxMinClearance = 1e-4f;   // clearance floor, scene units
constexpr float kRelaxGain = 0.8f;            // per-iteration relaxation
constexpr float kRelaxUnresolved = 1e-4f;     // "still penetrating" distance
constexpr float kRelaxMinSegment = 1e-6f;     // length-preservation guard
constexpr int kRelaxDefaultIterations = 24;
constexpr int kRelaxMaxWindow = 10;           // min(n,1000)/min(n,100) <= 10

struct UsdGenCollideCollider
{
    uint32_t geometry = 0;      // desc.surfaces index with points/topology
    std::vector<int> faces;     // empty == whole mesh, else sorted unique ids
};

struct UsdGenCollideCapture final : public UsdGenCapture
{
    std::vector<UsdGenCollideCollider> colliders;  // node.surfaces order
    bool relax = false;
    UsdGenCollideExactBvh exact;                   // flexible / stiff
    std::vector<strand::TriangleBvh> relaxBvhs;    // relax, one per collider
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

// First face-vertex index of every face (validated topology: counts >= 0
// and summing to the index count).
std::vector<size_t> FaceOffsets(UsdGenSurfaceDesc const &geometry)
{
    std::vector<size_t> offsets(geometry.faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < geometry.faceVertexCounts.size(); ++f)
        offsets[f + 1] = offsets[f] + size_t(std::max(0, geometry.faceVertexCounts[f]));
    return offsets;
}

// Calls emit(a, b, c) for every fan triangle of the collider in scan order:
// ascending (restricted) faces, fan triangles (c0, ck, ck+1).
template <class F>
void ForEachFanTriangle(UsdGenSurfaceDesc const &geometry,
                        std::vector<int> const &faces, F const &emit)
{
    std::vector<size_t> const offsets = FaceOffsets(geometry);
    auto face = [&](size_t f) {
        int const corners = geometry.faceVertexCounts[f];
        if (corners < 3) return;
        size_t const off = offsets[f];
        int const v0 = geometry.faceVertexIndices[off];
        for (int k = 1; k + 1 < corners; ++k) {
            int const v1 = geometry.faceVertexIndices[off + size_t(k)];
            int const v2 = geometry.faceVertexIndices[off + size_t(k) + 1];
            emit(v0, v1, v2);
        }
    };
    if (faces.empty()) {
        for (size_t f = 0; f < geometry.faceVertexCounts.size(); ++f) face(f);
    } else {
        for (int f : faces) face(size_t(f));
    }
}

// Validates one resolved collider mesh and produces its restricted face
// list (empty == whole mesh). Every malformed shape fails closed; exactly
// degenerate triangles are skipped with a warning, and a collider with no
// valid triangle at all fails closed.
bool ValidateColliderMesh(SdfPath const &path, UsdGenSurfaceDesc const &entry,
                          UsdGenSurfaceDesc const &geometry,
                          std::vector<int> *faces, std::string *error,
                          std::string *warning)
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
    size_t valid = 0, skipped = 0;
    ForEachFanTriangle(geometry, *faces, [&](int v0, int v1, int v2) {
        GfVec3d const a(geometry.points[size_t(v0)]);
        GfVec3d const b(geometry.points[size_t(v1)]);
        GfVec3d const c(geometry.points[size_t(v2)]);
        GfVec3d const cr = GfCross(b - a, c - a);
        double const l = cr.GetLength();
        if (l > 0.0 && Finite(l)) ++valid;
        else ++skipped;
    });
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

// One strand::TriangleBvh over the collider's CURRENT points: the
// (restricted) fan triangles minus the degenerate ones the validation
// skipped; corner normals are area-weighted vertex normals of those points.
void BuildRelaxBvh(UsdGenSurfaceDesc const &geometry, std::vector<int> const &faces,
                   strand::TriangleBvh *bvh)
{
    std::vector<strand::FanTriangle> all, kept;
    VtIntArray restricted(faces.begin(), faces.end());
    strand::FanTriangulate(geometry.faceVertexCounts, geometry.faceVertexIndices, &all,
                           faces.empty() ? nullptr : &restricted);
    kept.reserve(all.size());
    for (strand::FanTriangle const &t : all) {
        GfVec3d const a(geometry.points[size_t(t.v[0])]);
        GfVec3d const b(geometry.points[size_t(t.v[1])]);
        GfVec3d const c(geometry.points[size_t(t.v[2])]);
        double const l = GfCross(b - a, c - a).GetLength();
        if (l > 0.0 && Finite(l)) kept.push_back(t);
    }
    bvh->Build(geometry.points, kept);
}

// --- relax ------------------------------------------------------------------

struct RelaxSample
{
    bool hit = false;
    float d = 0.0f;              // signed distance (negative inside)
    GfVec3f n{0.0f, 0.0f, 1.0f}; // unit gradient of d (outward)
};

// Signed distance to the union of the colliders (minimum over colliders,
// first collider on ties). The closest point is exact; the side comes from
// the interpolated vertex normal at it, and the gradient is the unit
// direction away from (outside) / toward (inside) the closest point. A
// point on the surface (within ~1e-6 relative) takes the vertex normal.
RelaxSample SignedDistance(std::vector<strand::TriangleBvh> const &bvhs, GfVec3f const &p)
{
    RelaxSample best;
    float const scale = 1.0f + std::max(std::fabs(p[0]), std::max(std::fabs(p[1]), std::fabs(p[2])));
    for (strand::TriangleBvh const &bvh : bvhs) {
        if (bvh.Empty()) continue;
        strand::ClosestHit const h = bvh.Closest(p);
        if (!h.hit) continue;
        GfVec3f const surfaceN = bvh.Normal(h.triangle, h.bary);
        GfVec3f const diff = p - h.point;
        float const len = diff.GetLength();
        RelaxSample s;
        s.hit = true;
        if (len > 1e-6f * scale) {
            float const sign = GfDot(diff, surfaceN) < 0.0f ? -1.0f : 1.0f;
            s.d = sign * len;
            float const k = sign / len;
            s.n = GfVec3f(diff[0] * k, diff[1] * k, diff[2] * k);
        } else {
            s.d = GfDot(diff, surfaceN);
            s.n = surfaceN;
        }
        if (!best.hit || s.d < best.d) best = s;
    }
    return best;
}

// The push-out window w = max(1, round(r)), r = steps(cap 1000) /
// steps(cap 100) with steps = clamp(n, 3, cap): 1 for n <= 100.
int RelaxWindow(size_t n)
{
    float const steps1000 = float(std::clamp<size_t>(n, 3, 1000));
    float const steps100 = float(std::clamp<size_t>(n, 3, 100));
    long const w = std::lroundf(steps1000 / steps100);
    return int(std::clamp<long>(w, 1, kRelaxMaxWindow));
}

inline GfVec3f Mul(GfVec3f const &v, float s) { return GfVec3f(v[0] * s, v[1] * s, v[2] * s); }

// One strand, in place: `in` is the upstream strand (A), `out` receives the
// result. Between iterations `out[k]` (k >= 1) holds Wk; inside an iteration
// it holds the displacement Disp (the reference keeps both arrays; Disp is
// always Wk - A at an iteration boundary, so one array suffices). The
// smoothing is a Jacobi pass done in place: a ring of the last w overwritten
// displacements supplies the k - w tap. Returns the "still penetrating"
// flag. Allocation-free.
bool RelaxStrand(std::vector<strand::TriangleBvh> const &bvhs,
                 float const *ix, float const *iy, float const *iz,
                 float *ox, float *oy, float *oz, size_t n,
                 float clearRoot, float clearTip, int iterations)
{
    auto in = [&](size_t k) { return GfVec3f(ix[k], iy[k], iz[k]); };
    auto out = [&](size_t k) { return GfVec3f(ox[k], oy[k], oz[k]); };
    auto put = [&](size_t k, GfVec3f const &v) { ox[k] = v[0]; oy[k] = v[1]; oz[k] = v[2]; };
    for (size_t k = 0; k < n; ++k) { ox[k] = ix[k]; oy[k] = iy[k]; oz[k] = iz[k]; }
    if (n < 2 || iterations < 1) return false;
    float const span = float(n - 1);
    auto clearance = [&](size_t k) {
        return clearRoot + (clearTip - clearRoot) * (float(k) / span);
    };
    bool penetrating = false;
    for (size_t k = 1; k < n && !penetrating; ++k) {
        RelaxSample const s = SignedDistance(bvhs, in(k));
        penetrating = s.hit && s.d < clearance(k);
    }
    if (!penetrating) return false;

    size_t const w = size_t(RelaxWindow(n));
    GfVec3f ring[kRelaxMaxWindow];
    for (int it = 0; it < iterations; ++it) {
        // Disp[k] = (Wk[k] - A[k]) + n * 0.8 * (clear(k) - d) when inside the
        // clearance, sampled at the current Wk.
        for (size_t k = 1; k < n; ++k) {
            GfVec3f const wk = out(k);
            GfVec3f disp = wk - in(k);
            RelaxSample const s = SignedDistance(bvhs, wk);
            float const c = clearance(k);
            if (s.hit && s.d < c) disp += Mul(s.n, kRelaxGain * (c - s.d));
            put(k, disp);
        }
        // S[k] = 0.25 Disp[max(0,k-w)] + 0.5 Disp[k] + 0.25 Disp[min(n-1,k+w)]
        // for k = 1..n-2; S[n-1] = 0.5 Disp[n-1] + 0.5 Disp[max(0,n-1-w)];
        // Disp[0] == 0 (the root never moves).
        for (size_t k = 1; k + 1 < n; ++k) {
            size_t const lo = k > w ? k - w : 0;
            GfVec3f const a = lo == 0 ? GfVec3f(0.0f) : ring[lo % w];
            GfVec3f const b = out(k);
            GfVec3f const c = out(std::min(n - 1, k + w));
            ring[k % w] = b;
            put(k, Mul(a, 0.25f) + Mul(b, 0.5f) + Mul(c, 0.25f));
        }
        {
            size_t const last = n - 1;
            size_t const lo = last > w ? last - w : 0;
            GfVec3f const a = lo == 0 ? GfVec3f(0.0f) : ring[lo % w];
            put(last, Mul(out(last), 0.5f) + Mul(a, 0.5f));
        }
        // Wk = A + S, then follow-the-leader from the fixed root restores
        // every rest segment length |A[k] - A[k-1]|.
        GfVec3f prev = in(0);
        for (size_t k = 1; k < n; ++k) {
            GfVec3f const a = in(k);
            float const seg = (a - in(k - 1)).GetLength();
            GfVec3f const target = a + out(k);
            GfVec3f const dir = target - prev;
            float const len = dir.GetLength();
            GfVec3f const unit = len > kRelaxMinSegment
                ? GfVec3f(dir[0] / len, dir[1] / len, dir[2] / len)
                : GfVec3f(0.0f, 1.0f, 0.0f);
            GfVec3f const wk = prev + Mul(unit, seg);
            put(k, wk);
            prev = wk;
        }
    }
    for (size_t k = 1; k < n; ++k) {
        RelaxSample const s = SignedDistance(bvhs, out(k));
        if (s.hit && s.d <= kRelaxUnresolved) return true;
    }
    return false;
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
    v.push_back(sClearanceRoot);          // relax only
    v.push_back(sClearanceTip);           // relax only
    v.push_back(sRelaxIterations);        // relax only
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
TfSpan<const TfToken> UsdGenCollideOp::OutputPrimvars() const
{
    return TfSpan<const TfToken>(_outputs.data(), _outputs.size());
}
void UsdGenCollideOp::Configure(UsdGenParamView const &params)
{
    _outputs.clear();
    if (params.GetToken(sResolveType, sFlexible) == sRelax)
        _outputs.push_back(sUnresolved);
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
    if (resolveType != sFlexible && resolveType != sStiff && resolveType != sRelax) {
        if (diag) diag->Error("UsdGenCollide: unknown usdGen:resolveType '" +
                              resolveType.GetString() + "'");
        return false;
    }
    for (TfToken const *name : {&sOffset, &sPushAmount, &sClearanceRoot, &sClearanceTip}) {
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
    if (params.GetInt(sRelaxIterations, kRelaxDefaultIterations) < 1) {
        if (diag) diag->Error("UsdGenCollide: usdGen:relax:iterations must be at least 1");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenCollideOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Structural resolveType plus the resolved collider geometry: an animated
    // collider (or a reordered surface pool, which moves the captured indices)
    // re-captures. Value edits (offset/pushAmount/iterations/clearance, and
    // connected values folded in separately by the scheduler) sweep without
    // recapture.
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
    cap.exact.Clear();
    cap.relaxBvhs.clear();
    cap.relax = p->GetToken(sResolveType, sFlexible) == sRelax;
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
        if (!ValidateColliderMesh(path, e, ctx.desc->surfaces[geometry],
                                  &collider.faces, &error, &warning)) {
            if (diag) diag->Error("UsdGenCollide: " + error);
            return false;
        }
        if (!warning.empty() && diag) diag->Warn("UsdGenCollide: " + warning);
        cap.colliders.push_back(std::move(collider));
    }
    // Every connected value is scanned, so a non-finite expression fails
    // closed at capture instead of writing NaN points (displace precedent).
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
    for (uint32_t c = 0; c != upstream.totalCurves; ++c) {
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(offset.Value(c, cv)) ||
                !Finite(pushAmount.Value(c, cv)) ||
                !Finite(mask.Value(c, cv))) {
                if (diag) diag->Error("UsdGenCollide: per-CV offset/pushAmount/mask "
                                      "must be finite");
                return false;
            }
        }
    }
    // Index the current collider points (validated above).
    if (cap.relax) {
        cap.relaxBvhs.resize(cap.colliders.size());
        for (size_t i = 0; i < cap.colliders.size(); ++i)
            BuildRelaxBvh(ctx.desc->surfaces[cap.colliders[i].geometry],
                          cap.colliders[i].faces, &cap.relaxBvhs[i]);
    } else {
        for (UsdGenCollideCollider const &collider : cap.colliders) {
            UsdGenSurfaceDesc const &geometry = ctx.desc->surfaces[collider.geometry];
            ForEachFanTriangle(geometry, collider.faces, [&](int v0, int v1, int v2) {
                cap.exact.Add(geometry.points[size_t(v0)], geometry.points[size_t(v1)],
                              geometry.points[size_t(v2)]);
            });
        }
        cap.exact.Build();
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
    // The optional per-curve unresolved flag (relax nodes only).
    float *flags = (view->outCount > 0 && view->outF) ? view->outF[0] : nullptr;
    auto copyInput = [&](size_t o) {
        px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
    };

    if (cap.relax) {
        double const rootClear = p ? Sanitize(p->GetDouble(sClearanceRoot, 0.0)) : 0.0;
        double const tipClear = p ? Sanitize(p->GetDouble(sClearanceTip, 0.0)) : 0.0;
        float const cR = std::max(kRelaxMinClearance, float(rootClear));
        float const cT = std::max(kRelaxMinClearance, float(tipClear));
        int const relaxIterations =
            p ? p->GetInt(sRelaxIterations, kRelaxDefaultIterations) : kRelaxDefaultIterations;
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            size_t const n =
                ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
            size_t const g =
                ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
            size_t const curve = curveBase + c;
            bool const unresolved = RelaxStrand(
                cap.relaxBvhs, inPx + g, inPy + g, inPz + g, px + g, py + g, pz + g,
                n, cR, cT, relaxIterations);
            // usdGen:mask envelopes the relaxed result per CV.
            for (size_t i = 0; i < n; ++i) {
                size_t const o = g + i;
                float const m = float(std::clamp(
                    Sanitize(maskField.Value(curve, cvBase + o)), 0.0, 1.0));
                if (m == 1.0f) continue;
                if (m == 0.0f) { copyInput(o); continue; }
                px[o] = inPx[o] + (px[o] - inPx[o]) * m;
                py[o] = inPy[o] + (py[o] - inPy[o]) * m;
                pz[o] = inPz[o] + (pz[o] - inPz[o]) * m;
            }
            if (flags) flags[c] = unresolved ? 1.0f : 0.0f;
        }
        return;
    }
    if (flags)
        for (uint32_t c = 0; c < view->curveCount; ++c) flags[c] = 0.0f;

    if (iterations < 1 || cap.colliders.empty() || cap.exact.Empty() || !ctx.desc) {
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            size_t const n =
                ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
            size_t const g =
                ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
            for (size_t i = 0; i < n; ++i) copyInput(g + i);
        }
        return;
    }
    auto current = [&](size_t o, int it) -> GfVec3d {
        if (it == 0) return GfVec3d(inPx[o], inPy[o], inPz[o]);
        return GfVec3d(px[o], py[o], pz[o]);
    };
    auto store = [&](size_t o, GfVec3d const &v) {
        px[o] = float(v[0]); py[o] = float(v[1]); pz[o] = float(v[2]);
    };
    // Only a hit closer than the offset can push (pen = off - dist > 0), so
    // the search is bounded just above off^2: the exact closest triangle is
    // returned whenever it can push, and "no hit" otherwise — the same
    // outcome as the unbounded scan's non-positive penetration.
    auto closest = [&](GfVec3d const &pos, double off) {
        UsdGenCollideHit none;
        if (!(off > 0.0)) return none;
        return cap.exact.Closest(pos, off * off * (1.0 + 1e-6));
    };

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
                    double const off = Sanitize(offsetField.Value(curve, cv));
                    UsdGenCollideHit const hit = closest(pos, off);
                    if (!hit.found) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    double const pen = off - std::sqrt(hit.dist2);
                    if (!(pen > 0.0)) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    double const s = pen * push * m;
                    if (s == 0.0) {
                        if (it == 0) copyInput(o);
                        continue;
                    }
                    GfVec3d const dir = hit.dist2 == 0.0
                        ? UnitNormal(hit.a, hit.b, hit.c)
                        : (pos - hit.q) / std::sqrt(hit.dist2);
                    store(o, pos + dir * s);
                }
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
                    UsdGenCollideHit const hit = closest(pos, off);
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
