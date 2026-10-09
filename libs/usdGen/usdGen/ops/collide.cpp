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
// Closest-point search is brute force over fan-triangulated faces (04 §4
// notes the production BVH this waits on): deterministic first-win ties in
// node order / ascending faces, double precision, no allocation in
// Evaluate. Capture resolves and validates every collider mesh; Evaluate
// reads live points through the captured indices, so an animated collider
// re-captures (surfaceGeneration digest terms) and re-evaluates every cook
// with no stale geometry. Value-class edits (offset/pushAmount/iterations)
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

struct UsdGenCollideCollider
{
    uint32_t geometry = 0;      // desc.surfaces index with points/topology
    std::vector<int> faces;     // empty == whole mesh, else sorted unique ids
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
    GfVec3d q, a, b, c;   // closest point + its winning corners (for dist 0)
    double dist2 = 0.0;
};

// Closest point over every captured collider: node order, ascending faces,
// first win on ties. Reads live points through captured indices; never
// allocates. A hit with non-finite distance is impossible after Capture
// validated the meshes (the desc is immutable within a run); the guard
// below keeps it from ever producing NaN output regardless.
ClosestHit ClosestOnColliders(UsdGenGraphDesc const *desc,
                              UsdGenCollideCapture const &cap, GfVec3d const &p,
                              UsdGenCollidePhaseStats *probe)
{
    ClosestHit best;
    if (!desc) return best;
    for (UsdGenCollideCollider const &collider : cap.colliders) {
        if (collider.geometry >= desc->surfaces.size()) continue;
        UsdGenSurfaceDesc const &surf = desc->surfaces[collider.geometry];
        size_t const F = surf.faceVertexCounts.size();
        if (surf.points.empty() || surf.faceVertexIndices.empty()) continue;
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
                        if (!best.found || d2 < best.dist2) {
                            best.found = true;
                            best.q = q;
                            best.a = a;
                            best.b = b;
                            best.c = c;
                            best.dist2 = d2;
                        }
                    }
                }
            }
            off += size_t(corners);
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
                    double const q0 = ph ? NowMs() : 0.0;
                    ClosestHit const hit =
                        ClosestOnColliders(ctx.desc, cap, pos, ph);
                    double const q1 = ph ? NowMs() : 0.0;
                    if (ph) {
                        ph->narrowMs += q1 - q0;
                        ++ph->queries;
                    }
                    // Single-exit response tail (the pre-probe continues
                    // folded into if/else): identical stores, one phase stamp.
                    if (hit.found) {
                        double const off =
                            Sanitize(offsetField.Value(curve, cv));
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
                    if (ph) ph->respMs += NowMs() - q1;
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
                    double const q0 = ph ? NowMs() : 0.0;
                    ClosestHit const hit =
                        ClosestOnColliders(ctx.desc, cap, pos, ph);
                    if (ph) {
                        qSum += NowMs() - q0;
                        ++ph->queries;
                    }
                    double pen = 0.0;
                    GfVec3d dir(0.0, 0.0, 1.0);
                    if (hit.found) {
                        double const off = Sanitize(offsetField.Value(curve, cv));
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
                    ph->narrowMs += qSum;
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

UsdGenCollidePhaseStats UsdGenCollideOp::ReadPhaseProbe()
{
    UsdGenCollidePhaseStats total;
    ProbeRegistry &r = Registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (UsdGenCollidePhaseStats const *st : r.stats) {
        total.broadMs += st->broadMs;
        total.narrowMs += st->narrowMs;
        total.respMs += st->respMs;
        total.queries += st->queries;
        total.triTests += st->triTests;
        total.nodeVisits += st->nodeVisits;
    }
    return total;
}

}  // namespace usdGen
