// usdGen — ordered binary WidthBlend operator.
#include "usdGen/ops/widthBlend.h"

#include <cmath>
#include <cstdint>
#include "usdGen/registries/registrar.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

TfTokenVector const kTopologyParameters{};
TfTokenVector const kValueParameters{TfToken("enabled"),
                                     TfToken("widthBlend:weight")};

TfToken const kWeight{"widthBlend:weight"};

/// The ordered interpolation weight, clamped to exact endpoints.
bool ReadWeight(UsdGenParamView const *params, float *out,
                UsdGenDiagnostics *diag)
{
    double const weight = params ? params->GetDouble(kWeight, 1.0) : 1.0;
    if (!std::isfinite(weight) || weight < 0.0 || weight > 1.0) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: usdGen:widthBlend:weight must be finite and in [0, 1]");
        return false;
    }
    *out = static_cast<float>(weight);
    return true;
}

bool SamePlane(UsdGenPlane const &a, UsdGenPlane const &b)
{
    return a.name == b.name && a.interpolation == b.interpolation &&
        a.type == b.type && a.arity == b.arity && a.f == b.f && a.i == b.i;
}

bool SamePlaneList(std::vector<UsdGenPlane> const &a,
                   std::vector<UsdGenPlane> const &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i != a.size(); ++i)
        if (!SamePlane(a[i], b[i])) return false;
    return true;
}

/// WidthBlend's right input may differ only in the width plane.  Compare all
/// other transported values and topology metadata explicitly so a malformed
/// fan-in never silently pairs unlike curve streams.
bool SameNonWidthData(UsdGenCurveBuffer const &left,
                      UsdGenCurveBuffer const &right)
{
    if (left.totalCurves != right.totalCurves || left.totalCvs != right.totalCvs ||
        left.px != right.px || left.py != right.py || left.pz != right.pz ||
        left.rest != right.rest ||
        left.hairT != right.hairT || left.curveId != right.curveId ||
        left.rootPrim != right.rootPrim || left.rootUV != right.rootUV ||
        left.rootT != right.rootT || left.rootN != right.rootN ||
        left.rootB != right.rootB || left.cvOffsets != right.cvOffsets ||
        !SamePlaneList(left.extraCv, right.extraCv) ||
        !SamePlaneList(left.extraCurve, right.extraCurve) ||
        left.chunks.size() != right.chunks.size())
        return false;
    for (size_t i = 0; i != left.chunks.size(); ++i) {
        UsdGenChunkDesc const &a = left.chunks[i];
        UsdGenChunkDesc const &b = right.chunks[i];
        if (a.firstCurve != b.firstCurve || a.curveCount != b.curveCount ||
            a.liveCount != b.liveCount || a.firstCv != b.firstCv ||
            a.cvCount != b.cvCount || a.tile != b.tile || a.surface != b.surface)
            return false;
    }
    return true;
}

struct UsdGenWidthBlendCapture final : public UsdGenCapture
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWidthBlendCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.totalCurves != 0 && upstream.totalCvs != 0;
    }
};

}  // namespace

TfSpan<const TfToken> UsdGenWidthBlendOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(kTopologyParameters.data(),
                                 kTopologyParameters.size());
}

TfSpan<const TfToken> UsdGenWidthBlendOp::ValueParameters() const
{
    return TfSpan<const TfToken>(kValueParameters.data(), kValueParameters.size());
}

std::unique_ptr<UsdGenCapture> UsdGenWidthBlendOp::CreateCapture() const
{
    return std::make_unique<UsdGenWidthBlendCapture>();
}

bool UsdGenWidthBlendOp::Bind(UsdGenParamView const &params,
                              UsdGenDiagnostics *diag)
{
    UsdGenNodeDesc const *node = params.node;
    float weight = 1.0f;
    if (!ReadWeight(&params, &weight, diag)) return false;
    bool paramsOk = true;
    if (node)
        for (UsdGenParamValue const &param : node->params)
            paramsOk = paramsOk && param.name == kWeight;
    if (!node || !node->enabled ||
        !node->mode.IsEmpty() || !paramsOk ||
        !node->ramps.empty() || !node->expressionBindings.empty() ||
        !node->references.empty() || !node->curves.empty() ||
        !node->surfaces.empty() || !node->maps.empty() ||
        !node->mapBindings.empty()) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: requires enabled=true and no auxiliary inputs");
        return false;
    }
    if (node->inputs.size() != 2 || node->inputs[0] == node->inputs[1]) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: requires two distinct ordered geometry inputs");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenWidthBlendOp::CaptureDigest(
    UsdGenCaptureContext const &ctx) const
{
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(ctx.upstreamCount);
    if (ctx.upstreams) {
        for (uint32_t i = 0; i != ctx.upstreamCount; ++i)
            mix(ctx.upstreams[i] ? ctx.upstreams[i]->topologyVersion : UINT64_MAX);
    } else {
        mix(ctx.upstreamGeneration);
    }
    return {h, h ^ 0x9E3779B97F4A7C15ULL};
}

bool UsdGenWidthBlendOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    TF_UNUSED(out);
    float weight = 1.0f;
    if (!ReadWeight(ctx.params, &weight, diag)) return false;
    if (ctx.upstreamCount != 2 || !ctx.upstreams || !ctx.upstreams[0] ||
        !ctx.upstreams[1]) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: exactly two ordered geometry inputs are required");
        return false;
    }
    // Keep the legacy primary argument honest: the scheduler passes the first
    // ordered input here, and direct callers can still provide a value copy.
    UsdGenCurveBuffer const &left = *ctx.upstreams[0];
    UsdGenCurveBuffer const &right = *ctx.upstreams[1];
    if (&upstream != &left &&
        (upstream.totalCurves != left.totalCurves || upstream.totalCvs != left.totalCvs)) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: primary geometry input disagrees with ordered input 0");
        return false;
    }
    if (!SameNonWidthData(left, right)) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: ordered inputs must have identical non-width geometry");
        return false;
    }
    if (left.width.size() != left.totalCvs || right.width.size() != right.totalCvs) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: both ordered inputs must provide one width per CV");
        return false;
    }
    return true;
}

void UsdGenWidthBlendOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &capture,
    UsdGenChunkView *view) const
{
    TF_UNUSED(capture);
    if (!view || !view->width || !view->inWidth || !view->inWidth2) return;
    float weight = ctx.params
        ? static_cast<float>(ctx.params->GetDouble(kWeight, 1.0)) : 1.0f;
    if (!(weight > 0.0f)) weight = 0.0f;
    else if (weight >= 1.0f) weight = 1.0f;
    for (uint32_t c = 0; c != view->curveCount; ++c) {
        uint32_t const count = view->cvCount != 0 ? view->cvCount
            : (view->cvOffsets && view->desc
               ? uint32_t(view->cvOffsets[c + 1] - view->cvOffsets[c]) : 0);
        size_t const base = view->cvCount != 0 ? size_t(c) * view->cvCount
            : (view->cvOffsets && view->desc
               ? size_t(view->cvOffsets[c]) - view->desc->firstCv : 0);
        for (uint32_t i = 0; i != count; ++i) {
            size_t const o = base + i;
            float const left = view->inWidth[o];
            float const right = view->inWidth2[o];
            view->width[o] = weight <= 0.0f ? left
                : weight >= 1.0f ? right
                : left + (right - left) * weight;
        }
    }
}

UsdGenOpId UsdGenRegisterWidthBlendOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenWidthBlendOp().Type(), [] { return std::make_unique<UsdGenWidthBlendOp>(); });
}

}  // namespace usdGen
