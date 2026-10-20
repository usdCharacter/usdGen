// usdGen — UsdGenSmoothOp implementation. 02-schema.md §2.7.1, 04 §2.12.
//
// mode = alongCurve (pure-evaluate): `iterations` Jacobi passes of
//   new[i] = lerp(old[i], neighborAvg, strength * mask)
// over each strand, where neighborAvg is the mean of the two adjacent CVs.
// Endpoints use the clamped boundary (the single adjacent CV); lockRoot
// pins CV0. strength is sampled per curve root, the mask per CV.
// mode = neighbours (capture): each strand blends toward the same-index
// CVs of up to numNeighbors nearest strands whose roots lie within
// searchRadius of its own root (self excluded, deformed positions, queried
// over a nanoflann kd-tree at capture). Strands with no in-radius
// neighbour are copied unchanged. The k passes toward the fixed capture
// average collapse to one lerp with alpha = 1 - (1 - s*m)^iterations.
// lockRoot pins CV0 in both modes. searchRadius <= 0 disables the query,
// so neighbours mode at schema defaults is a pass-through.
// mode = box (pure-evaluate, reference behaviour of imported grooms): per
// Jacobi pass (`iterations`, one by default -- the reference runs one),
//   P[k] = (1 - a) O[k] + a * mean(O[max(0, k-w) .. min(n-1, k+w)]),
//   a = clamp(strength, 0, 1) * mask, w = usdGen:window (inclusive box,
//   includes self), for k = 1 .. n-2; both endpoints are always fixed (no
//   lockRoot), and there is no length preservation.
#include "usdGen/ops/smooth.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

#include <nanoflann.hpp>

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

struct UsdGenSmoothCapture final : public UsdGenCapture
{
    bool neighbours = false;
    std::vector<uint32_t> spans;                 // R+1 curve spans (neighbours only)
    std::vector<float> positions;                // 3 * totalCvs snapshot (neighbours only)
    std::vector<std::vector<uint32_t>> neighborLists;  // per strand, ascending (neighbours only)
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenSmoothCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        // alongCurve reads the current chunk input, so it never goes stale;
        // neighbours reads the capture snapshot, which tracks deformed roots
        // and is rebuilt on any upstream change.
        if (!neighbours) return true;
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

struct Cloud {
    std::vector<float> points;   // xyz per strand root
    size_t kdtree_get_point_count() const { return points.size() / 3; }
    float kdtree_get_pt(size_t i, size_t d) const { return points[i * 3 + d]; }
    template <class B> bool kdtree_get_bbox(B &) const { return false; }
};
using CloudIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t>;

bool Finite(double v) { return std::isfinite(v); }

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("mode"));            // structural: the kernel branch (02 §6.1)
    v.push_back(TfToken("searchRadius"));    // capture: rebuilds the neighbour lists
    v.push_back(TfToken("numNeighbors"));    // capture: rebuilds the neighbour lists
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("strength"));
    v.push_back(TfToken("iterations"));
    v.push_back(TfToken("lockRoot"));
    v.push_back(TfToken("mask"));
    v.push_back(TfToken("window"));          // box half-width (pure evaluate)
    return v;
}();

TfSpan<const TfToken> UsdGenSmoothOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenSmoothOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenSmoothOp::CreateCapture() const
{
    return std::make_unique<UsdGenSmoothCapture>();
}
uint32_t UsdGenSmoothOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenSmoothOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sMode, sAlongCurve);
    if (mode != sAlongCurve && mode != sNeighbours && mode != sBox) {
        if (diag) diag->Error("UsdGenSmooth: unknown usdGen:mode '" + mode.GetString() + "'");
        return false;
    }
    int const window = params.GetInt(sWindow, 2);
    if (window < 1 || window > int(kUsdGenMaxCvsPerCurve)) {
        if (diag) diag->Error("UsdGenSmooth: usdGen:window must be in [1, 4096]");
        return false;
    }
    double const strength = params.GetDoubleLiteral(sStrength, 0.5);
    if (!Finite(strength) || strength < -1.0 || strength > 1.0) {
        if (diag) diag->Error("UsdGenSmooth: usdGen:strength must be finite and in [-1, 1]");
        return false;
    }
    int const iterations = params.GetInt(sIterations, 1);
    if (iterations < 1 || iterations > 16) {
        if (diag) diag->Error("UsdGenSmooth: usdGen:iterations must be in [1, 16]");
        return false;
    }
    double const radius = params.GetDoubleLiteral(sSearchRadius, 0.0);
    if (!Finite(radius) || radius < 0.0) {
        if (diag) diag->Error("UsdGenSmooth: usdGen:searchRadius must be finite and >= 0");
        return false;
    }
    int const neighbors = params.GetInt(sNumNeighbors, 4);
    if (neighbors < 1 || neighbors > 32) {
        if (diag) diag->Error("UsdGenSmooth: usdGen:numNeighbors must be in [1, 32]");
        return false;
    }
    if (!Finite(params.GetDoubleLiteral(sMask, 1.0))) {
        if (diag) diag->Error("UsdGenSmooth: usdGen:mask must be finite");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenSmoothOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Structural tokens plus upstream generation plus seed only. Value edits
    // sweep without recapturing; searchRadius/numNeighbors edits recapture
    // through their capture dirty class (UsdGenDirtyCapture sets
    // captureNeeded), and connected-value edits are folded into the capture
    // identity separately by the scheduler.
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    feed(p ? uint64_t(p->GetToken(sMode, sAlongCurve).Hash()) : 0u);
    feed(ctx.upstreamGeneration);
    feed(uint64_t(ctx.seed));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenSmoothOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenSmoothCapture *>(out);
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenSmooth: " + message);
        return false;
    };
    // Literals are validated here (the CPU lane never calls Bind).
    if (p && !Bind(*p, diag)) return false;
    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(upstream, &spans, &error))
        return fail("upstream CV topology: " + error);
    size_t const R = upstream.totalCurves;
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;

    TfToken const mode = p ? p->GetToken(sMode, sAlongCurve) : sAlongCurve;
    cap.neighbours = mode == sNeighbours;
    if (!cap.neighbours) {
        cap.spans.clear();
        cap.positions.clear();
        cap.neighborLists.clear();
    }
    UsdGenParamField const strengthField =
        p ? p->GetScalarField(sStrength, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const radiusField =
        p ? p->GetScalarField(sSearchRadius, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const neighborsField =
        p ? p->GetScalarField(sNumNeighbors, 4.0) : UsdGenParamField{4.0};
    if (strengthField.connected && strengthField.components != 1)
        return fail("connected strength must be float");
    if (maskField.connected && maskField.components != 1)
        return fail("connected mask must be float");
    if (radiusField.connected && radiusField.components != 1)
        return fail("connected searchRadius must be float");
    if (neighborsField.connected && neighborsField.components != 1)
        return fail("connected numNeighbors must be int");
    // Every connected value is scanned in both modes, so a non-finite or
    // out-of-range expression fails closed at capture instead of writing
    // NaN points per frame.
    for (size_t c = 0; c < R; ++c) {
        uint32_t const root = spans[c];
        double const strength = strengthField.Value(c, root);
        if (!Finite(strength) || strength < -1.0 || strength > 1.0)
            return fail("per-strand strength must be finite and in [-1, 1]");
        double const radius = radiusField.Value(c, root);
        if (!Finite(radius) || radius < 0.0)
            return fail("per-strand searchRadius must be finite and >= 0");
        double const want = neighborsField.Value(c, root);
        long long const k = Finite(want) ? std::llround(want) : 0;
        if (!Finite(want) || k < 1 || k > 32)
            return fail("per-strand numNeighbors must round to [1, 32]");
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(maskField.Value(c, cv)))
                return fail("per-CV mask must be finite");
        }
    }
    if (p) {
        int const iterations = p->GetInt(sIterations, 1);
        if (iterations < 1 || iterations > 16)
            return fail("usdGen:iterations must be in [1, 16]");
    }
    if (!cap.neighbours || R == 0) return true;

    // Neighbours mode: snapshot the deformed positions and resolve each
    // strand's neighbour list (up to numNeighbors nearest roots within
    // searchRadius, self excluded, stored ascending for deterministic
    // summation in Evaluate).
    cap.spans = spans;
    cap.positions.resize(3 * size_t(upstream.totalCvs));
    for (uint32_t cv = 0; cv < upstream.totalCvs; ++cv) {
        cap.positions[3 * size_t(cv)] = upstream.px[cv];
        cap.positions[3 * size_t(cv) + 1] = upstream.py[cv];
        cap.positions[3 * size_t(cv) + 2] = upstream.pz[cv];
    }
    Cloud cloud;
    cloud.points.resize(3 * R);
    for (size_t c = 0; c < R; ++c) {
        uint32_t const root = spans[c];
        cloud.points[3 * c] = upstream.px[root];
        cloud.points[3 * c + 1] = upstream.py[root];
        cloud.points[3 * c + 2] = upstream.pz[root];
    }
    CloudIndex index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    cap.neighborLists.assign(R, {});
    std::vector<nanoflann::ResultItem<uint32_t, float>> hits;
    for (size_t c = 0; c < R; ++c) {
        uint32_t const root = spans[c];
        double const radius = radiusField.Value(c, root);
        if (radius <= 0.0) continue;   // disabled: the strand keeps its shape
        long long const k = std::llround(neighborsField.Value(c, root));
        float const query[3] = {cloud.points[3 * c], cloud.points[3 * c + 1],
                                cloud.points[3 * c + 2]};
        double const r2 = radius * radius;
        float const searchR2 = float(std::min(r2, double(std::numeric_limits<float>::max())));
        hits.clear();
        (void)index.radiusSearch(query, searchR2, hits);
        std::sort(hits.begin(), hits.end(), [](auto const &a, auto const &b) {
            return a.second != b.second ? a.second < b.second : a.first < b.first;
        });
        std::vector<uint32_t> &list = cap.neighborLists[c];
        for (auto const &hit : hits) {
            if (hit.first == c) continue;
            if (list.size() >= size_t(k)) break;
            list.push_back(hit.first);
        }
        std::sort(list.begin(), list.end());
    }
    return true;
}

void UsdGenSmoothOp::EvaluateBox(UsdGenEvalContext const &ctx, UsdGenChunkView *view) const
{
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const strengthField =
        p ? p->GetScalarField(sStrength, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    int const iterations = std::clamp(p ? p->GetInt(sIterations, 1) : 1, 1, 16);
    size_t const window =
        size_t(std::clamp(p ? p->GetInt(sWindow, 2) : 2, 1, int(kUsdGenMaxCvsPerCurve)));
    thread_local std::vector<float> tOld;
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        float const strength = std::clamp(
            static_cast<float>(strengthField.Value(curve, cvBase + g)), 0.0f, 1.0f);
        if (tOld.size() < 3 * n) tOld.resize(3 * n);
        float *O = tOld.data();
        for (size_t i = 0; i < n; ++i) {
            O[3 * i] = inPx[g + i]; O[3 * i + 1] = inPy[g + i]; O[3 * i + 2] = inPz[g + i];
        }
        for (int pass = 0; pass < iterations; ++pass) {
            if (pass > 0) {
                for (size_t i = 0; i < n; ++i) {
                    O[3 * i] = px[g + i]; O[3 * i + 1] = py[g + i]; O[3 * i + 2] = pz[g + i];
                }
            }
            for (size_t k = 0; k < n; ++k) {
                size_t const o = g + k;
                float x = O[3 * k], y = O[3 * k + 1], z = O[3 * k + 2];
                if (k > 0 && k + 1 < n) {
                    float const a = strength * std::clamp(
                        static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
                    if (a != 0.0f) {
                        size_t const lo = k >= window ? k - window : 0;
                        size_t const hi = std::min(n - 1, k + window);
                        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
                        for (size_t i = lo; i <= hi; ++i) {
                            sx += O[3 * i]; sy += O[3 * i + 1]; sz += O[3 * i + 2];
                        }
                        float const count = float(hi - lo + 1);
                        x = (1.0f - a) * x + a * (sx / count);
                        y = (1.0f - a) * y + a * (sy / count);
                        z = (1.0f - a) * z + a * (sz / count);
                    }
                }
                px[o] = x; py[o] = y; pz[o] = z;
            }
        }
    }
}

void UsdGenSmoothOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    TfToken const mode = p ? p->GetToken(sMode, sAlongCurve) : sAlongCurve;
    auto const &cap = static_cast<UsdGenSmoothCapture const &>(captureIn);
    if (mode == sBox) {
        EvaluateBox(ctx, view);
        return;
    }
    if (mode != sAlongCurve && mode != sNeighbours) {
        // Defensive pass-through; Capture already failed closed.
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
            size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
            for (size_t i = 0; i < n; ++i) {
                px[g + i] = inPx[g + i]; py[g + i] = inPy[g + i]; pz[g + i] = inPz[g + i];
            }
        }
        return;
    }
    UsdGenParamField const strengthField =
        p ? p->GetScalarField(sStrength, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    int iterations = p ? p->GetInt(sIterations, 1) : 1;
    iterations = std::clamp(iterations, 1, 16);
    bool const lockRoot = p ? p->GetBool(sLockRoot, true) : true;

    if (mode == sNeighbours && !cap.neighbours) {
        // No capture payload (a mode switch always recaptures, so this is
        // only reachable defensively): pass through, never read stale lists.
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
            size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
            for (size_t i = 0; i < n; ++i) {
                px[g + i] = inPx[g + i]; py[g + i] = inPy[g + i]; pz[g + i] = inPz[g + i];
            }
        }
        return;
    }
    bool const neighbours = mode == sNeighbours;

    if (!neighbours) {
        // alongCurve: Jacobi passes ping-ponging between the scratch buffer
        // (first read) and the output planes.
        thread_local std::vector<float> tScratch;
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
            size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
            if (!n) continue;
            size_t const curve = curveBase + c;
            float const strength =
                static_cast<float>(strengthField.Value(curve, cvBase + g));
            if (tScratch.size() < 3 * n) tScratch.assign(3 * n, 0.0f);
            float *sx = tScratch.data(), *sy = sx + n, *sz = sy + n;
            for (size_t i = 0; i < n; ++i) {
                sx[i] = inPx[g + i]; sy[i] = inPy[g + i]; sz[i] = inPz[g + i];
            }
            for (int pass = 0; pass < iterations; ++pass) {
                bool const readScratch = (pass % 2) == 0;
                for (size_t i = 0; i < n; ++i) {
                    size_t const o = g + i;
                    float const mask = std::clamp(
                        static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
                    float const alpha = strength * mask;
                    float rx, ry, rz;
                    if (readScratch) { rx = sx[i]; ry = sy[i]; rz = sz[i]; }
                    else { rx = px[o]; ry = py[o]; rz = pz[o]; }
                    auto read = [&](size_t j, float &x, float &y, float &z) {
                        if (readScratch) { x = sx[j]; y = sy[j]; z = sz[j]; }
                        else { x = px[g + j]; y = py[g + j]; z = pz[g + j]; }
                    };
                    float ox = rx, oy = ry, oz = rz;
                    bool const pinned = i == 0 && lockRoot;
                    if (!pinned && alpha != 0.0f && n > 1) {
                        size_t const a = i == 0 ? 1 : i - 1;
                        size_t const b = i + 1 < n ? i + 1 : n - 2;
                        float ax, ay, az, bx, by, bz;
                        read(a, ax, ay, az);
                        if (a == b) { bx = ax; by = ay; bz = az; }
                        else read(b, bx, by, bz);
                        float const nx = 0.5f * (ax + bx);
                        float const ny = 0.5f * (ay + by);
                        float const nz = 0.5f * (az + bz);
                        ox = rx + alpha * (nx - rx);
                        oy = ry + alpha * (ny - ry);
                        oz = rz + alpha * (nz - rz);
                    }
                    if (readScratch) { px[o] = ox; py[o] = oy; pz[o] = oz; }
                    else { sx[i] = ox; sy[i] = oy; sz[i] = oz; }
                }
            }
            if (iterations % 2 == 0) {
                for (size_t i = 0; i < n; ++i) {
                    px[g + i] = sx[i]; py[g + i] = sy[i]; pz[g + i] = sz[i];
                }
            }
        }
        return;
    }

    // neighbours: one lerp per CV toward the capture-time neighbour average.
    // k passes toward a fixed target equal one pass with 1 - (1 - a)^k.
    size_t const R = cap.neighborLists.size();
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        std::vector<uint32_t> const *list =
            curve < R ? &cap.neighborLists[curve] : nullptr;
        float const strength =
            static_cast<float>(strengthField.Value(curve, cvBase + g));
        if (!list || list->empty()) {
            for (size_t i = 0; i < n; ++i) {
                px[g + i] = inPx[g + i]; py[g + i] = inPy[g + i]; pz[g + i] = inPz[g + i];
            }
            continue;
        }
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            if (i == 0 && lockRoot) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            float const a = strength * mask;
            if (a == 0.0f) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            double sx = 0.0, sy = 0.0, sz = 0.0;
            size_t hits = 0;
            for (uint32_t d : *list) {
                if (size_t(d) >= R || size_t(d) + 1 >= cap.spans.size()) continue;
                uint32_t const first = cap.spans[d], last = cap.spans[d + 1];
                if (i >= size_t(last - first)) continue;   // ragged: no CV i
                size_t const s = 3 * (size_t(first) + i);
                if (s + 2 >= cap.positions.size()) continue;
                sx += cap.positions[s]; sy += cap.positions[s + 1]; sz += cap.positions[s + 2];
                ++hits;
            }
            if (!hits) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            float const nx = float(sx / double(hits));
            float const ny = float(sy / double(hits));
            float const nz = float(sz / double(hits));
            double eff = 1.0;
            for (int k = 0; k < iterations; ++k) eff *= 1.0 - double(a);
            float const alpha = float(1.0 - eff);
            px[o] = inPx[o] + alpha * (nx - inPx[o]);
            py[o] = inPy[o] + alpha * (ny - inPy[o]);
            pz[o] = inPz[o] + alpha * (nz - inPz[o]);
        }
    }
}

}  // namespace usdGen
