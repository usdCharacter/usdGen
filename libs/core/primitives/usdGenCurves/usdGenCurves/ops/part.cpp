// usdGen — UsdGenPartOp implementation. 02-schema.md §2.7.2, 04 §3.
// Parting lines from a curve set (usdGen:part:curves, a reference input):
// every strand is assigned the side of its nearest parting-curve sample,
// emitted as partId (uniform int, m*2+side for parting curve m). A strand
// masked out (mask == 0 at its root), with part:strength == 0, or rooted
// farther than part:radius*part:strength from any sample is unparted
// (partId -1). Points pass through untouched: the part reads downstream,
// where guide/clump consumers weight by it. Sampling is on rest positions
// when the buffers carry them, so partIds are stable under deformation.
// TopologyEffect = None.
#include "usdGen/ops/part.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/ops/opUtil.h"

#include <nanoflann.hpp>

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "usdGen/registries/registrar.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

struct UsdGenPartCapture final : public UsdGenCapture
{
    std::vector<int32_t> partId;   // per strand; -1 = unparted
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenPartCapture>(*this);
    }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        // Nearest-side assignment reads positions, like Clump's centres.
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves;
    }
};

struct Cloud {
    std::vector<float> points;   // xyz per parting sample
    size_t kdtree_get_point_count() const { return points.size() / 3; }
    float kdtree_get_pt(size_t i, size_t d) const { return points[i * 3 + d]; }
    template <class B> bool kdtree_get_bbox(B &) const { return false; }
};
using CloudIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t>;

bool Finite(double v) { return std::isfinite(v); }

}  // namespace

namespace {

using namespace partP;
using D = expr::Domain;
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kFloatRoot{expr::ScalarType::Float32, 1, "float", D::Groom | D::Primitive};
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};

// Topology list (input, surface, seed, part:curves, part:radius), then the
// value list (enabled, part:strength, mask), as the static lists were.
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"part:curves", UsdGenParamClass::Structural},   // the parting set (02 §2.7.2)
    {"part:radius", UsdGenParamClass::Capture, true, kFloatRoot},   // rebuilds the side assignment
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"part:strength", UsdGenParamClass::Value, true, kFloatRoot},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
};
char const *const kReferences[] = {"part:curves"};
UsdGenPlaneDecl const kOutputs[] = {{"partId", UsdGenPlaneRate::Uniform, UsdGenPlaneScalar::Int, 1}};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenPartOp>(r);
}

}  // namespace

std::unique_ptr<UsdGenCapture> UsdGenPartOp::CreateCapture() const
{
    return std::make_unique<UsdGenPartCapture>();
}
uint32_t UsdGenPartOp::PlanesTouched() const
{
    // partId only; points/widths/hairT pass through aliased.
    return 0;
}

bool UsdGenPartOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    double const radius = params.GetDoubleLiteral(kRadius, 0.05);
    if (!Finite(radius) || radius <= 0.0) {
        if (diag) diag->Error("UsdGenPart: usdGen:part:radius must be finite and > 0");
        return false;
    }
    double const strength = params.GetDoubleLiteral(kStrength, 1.0);
    if (!Finite(strength) || strength < 0.0 || strength > 1.0) {
        if (diag) diag->Error("UsdGenPart: usdGen:part:strength must be finite and in [0, 1]");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenPartOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Radius is capture-class; strength and the mask gate per strand at
    // capture too, but their live values re-resolve through the scheduler's
    // expression identity (a strength edit re-captures this cheap assignment
    // and re-sweeps). The part:curves identity arrives via the resolved
    // reference values, as it does for GuideInterpolate.
    opUtil::Digest d;
    if (UsdGenParamView const *p = ctx.params) {
        d.Mix(Record().ParamName(kRadius));
        d.Mix(p->GetValue(kRadius, VtValue()));
    }
    d.Mix(uint64_t(ctx.seed));
    d.Mix(ctx.upstreamGeneration);
    return d.Epoch(0x5061727431ull);
}

bool UsdGenPartOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &in,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in.Primary());
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenPartCapture *>(out);
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenPart: " + message);
        return false;
    };
    if (p && !Bind(*p, diag)) return false;
    if (ctx.referenceCount != 1 || !ctx.resolvedReferences || !ctx.resolvedReferences[0] ||
        !ctx.resolvedReferences[0]->value)
        return fail("usdGen:part:curves must target one parting curve set");
    UsdGenCurveBuffer const &parting = ctx.resolvedReferences[0]->value->buffer;
    size_t const M = parting.totalCurves;
    if (M == 0) return fail("the parting curve set has no curves");

    std::vector<uint32_t> partSpans, rootSpans;
    std::string error;
    if (!opUtil::CurveSpans(parting, &partSpans, &error))
        return fail("parting topology: " + error);
    if (!opUtil::CurveSpans(upstream, &rootSpans, &error))
        return fail("strand topology: " + error);
    size_t const R = upstream.totalCurves;
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;

    // Parting samples: rest positions (current when no rest plane) plus a
    // per-sample tangent and owning curve index.
    size_t const nSamples = parting.totalCvs;
    Cloud cloud;
    cloud.points.resize(nSamples * 3);
    std::vector<float> tangents(nSamples * 3, 0.0f);
    std::vector<uint32_t> sampleCurve(nSamples, 0);
    for (size_t m = 0; m < M; ++m) {
        size_t const first = partSpans[m], count = partSpans[m + 1] - first;
        for (size_t j = 0; j < count; ++j) {
            size_t const s = first + j;
            GfVec3f const pos = opUtil::RestPoint(parting, s);
            cloud.points[s * 3] = pos[0];
            cloud.points[s * 3 + 1] = pos[1];
            cloud.points[s * 3 + 2] = pos[2];
            sampleCurve[s] = uint32_t(m);
            size_t const a = first + (j > 0 ? j - 1 : 0);
            size_t const b = first + (j + 1 < count ? j + 1 : j);
            GfVec3f const pa = opUtil::RestPoint(parting, a);
            GfVec3f const pb = opUtil::RestPoint(parting, b);
            GfVec3f t = pb - pa;
            float const l = t.GetLength();
            if (l <= 1e-12f) t = GfVec3f(0.0f, 1.0f, 0.0f);
            else t /= l;
            tangents[s * 3] = t[0];
            tangents[s * 3 + 1] = t[1];
            tangents[s * 3 + 2] = t[2];
        }
    }
    CloudIndex index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    UsdGenParamField const radiusField =
        p ? p->GetScalarField(kRadius, 0.05) : UsdGenParamField{0.05};
    UsdGenParamField const strengthField =
        p ? p->GetScalarField(kStrength, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    cap.partId.assign(R, -1);
    for (size_t c = 0; c < R; ++c) {
        uint32_t const root = rootSpans[c];
        double const radius = radiusField.Value(c, root);
        double const strength = strengthField.Value(c, root);
        if (!Finite(radius) || radius <= 0.0)
            return fail("per-strand part:radius must be finite and > 0");
        if (!Finite(strength) || strength < 0.0 || strength > 1.0)
            return fail("per-strand part:strength must be finite and in [0, 1]");
        float const mask = std::clamp(
            static_cast<float>(maskField.Value(c, root)), 0.0f, 1.0f);
        if (mask == 0.0f || strength == 0.0) continue;   // unparted: -1
        GfVec3f const rootPos = opUtil::RestPoint(upstream, root);
        float const query[3] = {rootPos[0], rootPos[1], rootPos[2]};
        uint32_t found = 0;
        float distSq = 0.0f;
        if (!index.knnSearch(query, 1, &found, &distSq) || found >= nSamples)
            return fail("parting query failed");
        double const effR = radius * strength;
        if (double(std::sqrt(distSq)) > effR) continue;   // too far: -1
        GfVec3f const u(tangents[size_t(found) * 3], tangents[size_t(found) * 3 + 1],
                        tangents[size_t(found) * 3 + 2]);
        GfVec3f const toStrand = rootPos - GfVec3f(cloud.points[size_t(found) * 3],
                                                   cloud.points[size_t(found) * 3 + 1],
                                                   cloud.points[size_t(found) * 3 + 2]);
        GfVec3f N = upstream.rootN.size() == R ? upstream.rootN[c]
                                               : GfVec3f(0.0f, 1.0f, 0.0f);
        if (N.GetLength() <= 1e-12f) N = GfVec3f(0.0f, 1.0f, 0.0f);
        else N.Normalize();
        GfVec3f const sideVec = GfCross(u, toStrand);
        int const side = GfDot(sideVec, N) >= 0.0f ? 0 : 1;
        cap.partId[c] = int32_t(sampleCurve[found] * 2 + uint32_t(side));
    }
    return true;
}

void UsdGenPartOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    TF_UNUSED(ctx);
    auto const &cap = static_cast<UsdGenPartCapture const &>(captureIn);
    // Points pass through: this kernel only publishes the side assignment.
    // view->outI[slot][c] is chunk-local (uniform plane sliced at firstCurve).
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    if (view->outCount < 1 || !view->outI || !view->outI[0]) return;
    int *plane = view->outI[0];
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const curve = curveBase + c;
        plane[c] = curve < cap.partId.size() ? cap.partId[curve] : -1;
    }
}

UsdGenOpId UsdGenRegisterPartOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenPart";
    d.displayName = "Part";
    d.create = &Create;
    d.topoFx = UsdGenTopoFx::None;
    d.params = UsdGenSpan(kParams);
    d.referenceInputs = UsdGenSpan(kReferences);
    d.outputPlanes = UsdGenSpan(kOutputs);
    return r.RegisterOp(d);
}

}  // namespace usdGen
