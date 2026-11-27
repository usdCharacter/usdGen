// usdGen — ordered binary WidthBlend operator.
#include "usdGen/ops/widthBlend.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include "usdGenCurves/types/curveSet.h"
#include "usdGen/registries/registrar.h"


namespace usdGen {
namespace {

using namespace widthBlendP;

UsdGenParamDecl const kParams[kCount] = {
    {"enabled", UsdGenParamClass::Value},
    {"widthBlend:weight", UsdGenParamClass::Value},
};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenWidthBlendOp>(r);
}

/// The closed descriptor ABI of the first binary value fan-in. Disabled
/// operators bypass runtime Bind(), so the compiler runs this at compile.
bool ValidateNode(UsdGenGraphSpec const &, UsdGenOpNode const &node,
                  UsdGenParamView const &view, std::string *error)
{
    bool paramsOk = true;
    UsdGenToken const &weightName = view.op->ParamName(kWeight);
    for (UsdGenParamValue const &param : node.params)
        paramsOk = paramsOk && param.name == weightName;
    double const weight = view.GetDoubleLiteral(kWeight, 1.0);
    if (!node.enabled || !std::isfinite(weight) || weight < 0.0 ||
        weight > 1.0 || !node.mode.IsEmpty() ||
        !paramsOk || !node.ramps.empty() ||
        !node.expressionBindings.empty() || !node.references.empty() ||
        !node.inputSets.empty() || !node.surfaces.empty() ||
        !node.maps.empty() || !node.mapBindings.empty()) {
        if (error) *error =
            "UsdGenCompiler: WidthBlend requires enabled=true, a finite "
            "usdGen:widthBlend:weight in [0,1], and no auxiliary inputs";
        return false;
    }
    if (node.inputs[0] == node.inputs[1]) {
        if (error) *error =
            "UsdGenCompiler: WidthBlend requires two distinct ordered geometry inputs";
        return false;
    }
    return true;
}

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

template <class T>
bool SameSpan(UsdGenSpan<const T> a, UsdGenSpan<const T> b)
{
    if (a.size() != b.size()) return false;
    if (a.data() == b.data()) return true;
    return std::equal(a.begin(), a.end(), b.begin());
}

bool SamePlane(UsdGenPlaneView const &a, UsdGenPlaneView const &b)
{
    return a.name == b.name && a.interpolation == b.interpolation &&
        a.type == b.type && a.arity == b.arity &&
        SameSpan<float>(a.f, b.f) && SameSpan<int>(a.i, b.i);
}

/// WidthBlend's right input may differ only in the width plane.  Compare all
/// other transported values and topology metadata explicitly so a malformed
/// fan-in never silently pairs unlike curve streams. (The chunk partition is
/// the graph's, a function of these counts and offsets.)
bool SameNonWidthData(UsdGenCurveSetView const &left,
                      UsdGenCurveSetView const &right)
{
    if (left.totalCurves != right.totalCurves || left.totalCvs != right.totalCvs ||
        !SameSpan<float>(left.px, right.px) || !SameSpan<float>(left.py, right.py) ||
        !SameSpan<float>(left.pz, right.pz) ||
        !SameSpan<UsdGenVec3f>(left.rest, right.rest) ||
        !SameSpan<float>(left.hairT, right.hairT) ||
        !SameSpan<uint64_t>(left.curveId, right.curveId) ||
        !SameSpan<int>(left.rootPrim, right.rootPrim) ||
        !SameSpan<UsdGenVec2f>(left.rootUV, right.rootUV) ||
        !SameSpan<UsdGenVec3f>(left.rootT, right.rootT) ||
        !SameSpan<UsdGenVec3f>(left.rootN, right.rootN) ||
        !SameSpan<UsdGenVec3f>(left.rootB, right.rootB) ||
        !SameSpan<int>(left.cvOffsets, right.cvOffsets) ||
        left.ExtraCvCount() != right.ExtraCvCount() ||
        left.ExtraCurveCount() != right.ExtraCurveCount())
        return false;
    for (uint32_t i = 0; i != left.ExtraCvCount(); ++i)
        if (!SamePlane(left.ExtraCv(i), right.ExtraCv(i))) return false;
    for (uint32_t i = 0; i != left.ExtraCurveCount(); ++i)
        if (!SamePlane(left.ExtraCurve(i), right.ExtraCurve(i))) return false;
    return true;
}

struct UsdGenWidthBlendCapture final : public UsdGenCapture
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWidthBlendCapture>(*this);
    }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        return upstream.totalCurves != 0 && upstream.totalCvs != 0;
    }
};

}  // namespace

std::unique_ptr<UsdGenCapture> UsdGenWidthBlendOp::CreateCapture() const
{
    return std::make_unique<UsdGenWidthBlendCapture>();
}

bool UsdGenWidthBlendOp::Bind(UsdGenParamView const &params,
                              UsdGenDiagnostics *diag)
{
    UsdGenOpNode const *node = params.node;
    float weight = 1.0f;
    if (!ReadWeight(&params, &weight, diag)) return false;
    bool paramsOk = true;
    if (node)
        for (UsdGenParamValue const &param : node->params)
            paramsOk = paramsOk && param.name == params.op->ParamName(kWeight);
    if (!node || !node->enabled ||
        !node->mode.IsEmpty() || !paramsOk ||
        !node->ramps.empty() || !node->expressionBindings.empty() ||
        !node->references.empty() || !node->inputSets.empty() ||
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
    mix(ctx.inputs.count);
    if (ctx.inputs.sets) {
        for (uint32_t i = 0; i != ctx.inputs.count; ++i)
            mix(ctx.inputs.sets[i].topologyVersion);
    } else {
        mix(ctx.upstreamGeneration);
    }
    return {h, h ^ 0x9E3779B97F4A7C15ULL};
}

bool UsdGenWidthBlendOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &in,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    (void)out;
    float weight = 1.0f;
    if (!ReadWeight(ctx.params, &weight, diag)) return false;
    if (in.count != 2 || !in.sets) {
        if (diag) diag->Error(
            "UsdGenWidthBlend: exactly two ordered geometry inputs are required");
        return false;
    }
    UsdGenCurveSetView const left = UsdGenCurveSetView::Of(in.sets[0]);
    UsdGenCurveSetView const right = UsdGenCurveSetView::Of(in.sets[1]);
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
    UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    (void)capture;
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
    UsdGenOpDef d;
    d.name = "UsdGenWidthBlend";
    d.displayName = "WidthBlend";
    d.create = &Create;
    d.countChanges = 0;
    d.geometryInputArity = 2;
    d.traits = kUsdGenOpGeometryInputsOrdered;
    d.params = UsdGenSpan(kParams);
    d.hooks.validateNode = &ValidateNode;
    return r.RegisterOp(d);
}

}  // namespace usdGen
