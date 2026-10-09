// usdGen — UsdGenWidthOp implementation (M1). 02-schema.md §2.7.1, 04 §2.11.
// Authors `widths` from a base width, a root->tip ramp (usdGen:width:knots,
// S11) and taper; usdGen:replace selects set (true) vs multiply (false).
// TopologyEffect = None.
#include "usdGen/ops/width.h"

#include "usdGen/opParams.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

/// Read a float2[] ramp-knots property (02 §2.17). Missing or empty -> {}
/// (UsdGenBuildRampLut maps {} to the flat constant ramp).
VtVec2fArray ReadRampKnots(UsdGenParamView const *p, TfToken const &name)
{
    if (!p) return {};
    if (VtValue const v = p->GetVtValue(name, VtValue());
        v.IsHolding<VtVec2fArray>()) {
        return v.UncheckedGet<VtVec2fArray>();
    }
    return {};
}

struct UsdGenWidthCapture final : public UsdGenCapturePayload
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWidthCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        // Width never changes topology; hairT must exist (generator written it).
        return !upstream.hairT.empty();
    }
};

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("replace"));       // §6.1 structural: set vs multiply changes
                                          // nothing about topology, but it is a
                                          // capture-class authoring choice
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));       // toggle: value-class (02 §6.3)
    v.push_back(TfToken("width"));
    v.push_back(TfToken("width:knots"));
    // C1 property is usdGen:width:interpolation (schema.usda:369, 02 §2.7.1);
    // the param name is the prefix-stripped form. Value-class: the 257-entry
    // LUT is rebuilt from the live parameters every evaluate (see below).
    v.push_back(TfToken("width:interpolation"));
    v.push_back(TfToken("mask"));        // operator envelope (02 §2.13)
    return v;
}();

TfSpan<const TfToken> UsdGenWidthOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenWidthOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenWidthOp::CreateCapture() const
{
    return std::make_unique<UsdGenWidthCapture>();
}
uint32_t UsdGenWidthOp::PlanesTouched() const
{
    return kPlaneWidths;
}

bool UsdGenWidthOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TF_UNUSED(params);
    TF_UNUSED(diag);
    return true;
}

UsdGenEpoch UsdGenWidthOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Only capture-class terms: `replace` (structural, 02 §6.1) and the
    // upstream generation. `width:knots` is value-class (02 :665) and is
    // deliberately excluded — its ramp is rebuilt every evaluate (gate SI-2).
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](std::string const &k, uint64_t v) {
        h ^= v; h *= 0x100000001b3ULL;
        h ^= k.size(); h *= 0x100000001b3ULL;
    };
    if (p) {
        feed("replace", p->GetBool(TfToken("replace"), true) ? 1 : 0);
    }
    feed("upstream", ctx.upstreamGeneration);
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenWidthOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    TF_UNUSED(ctx);
    TF_UNUSED(upstream);
    TF_UNUSED(out);
    TF_UNUSED(diag);
    // `width:knots` is value-class (02 :665): the ramp LUT is rebuilt every
    // evaluate from the live parameters (review M-5). Baking it into the
    // capture payload here would freeze a value-class edit out of the frame
    // response, and the digest above deliberately excludes it.
    // usdGen:mask is value-class too and is read in Evaluate.
    return true;
}

void UsdGenWidthOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    TF_UNUSED(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    // Parameter tokens are operator-owned: no per-chunk interning and no
    // function-static destruction racing an asynchronous cook at exit.
    // usdGen:width and usdGen:mask are CONNECTABLE: an authored literal is a
    // uniform field, a connected expression is a groom/primitive/point field
    // indexed by the shared rule in UsdGenParamField::Value.
    UsdGenParamField const widthField =
        p ? p->GetScalarField(sWidth, 0.01) : UsdGenParamField{0.01};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    const bool replace = p ? p->GetBool(sReplace, true) : true;
    auto const *hairT = view->hairT;
    auto const *inWidth = view->inWidth;
    // Chunk-local plane indices are rebased to the whole-buffer indices the
    // expression fields are published in.
    const size_t curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    const size_t cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    float defaultInputWidth = ctx.desc ? ctx.desc->defaultWidth : 0.01f;
    if (!std::isfinite(defaultInputWidth) || defaultInputWidth < 0.0f)
        defaultInputWidth = 0.01f;
    // Reuse worker-local LUT storage for nonempty knots. Flat ramps use the
    // exact scalar 1.0 below and require no allocated table.
    float const *lutPtr = nullptr;
    VtVec2fArray const widthKnots = ReadRampKnots(p, sKnots);
    bool const magFlat = widthKnots.empty();
    thread_local std::vector<float> tLut;
    if (!magFlat) {
        if (tLut.size() != kUsdGenRampLutSize) tLut.assign(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(widthKnots, p ? p->GetToken(sKnotsInterp, sCatmullRom) : sCatmullRom, tLut.data(), kUsdGenRampLutSize);
        lutPtr = tLut.data();
    }
    // Uniform fast path: a flat ramp with uniform width/mask fields makes
    // target and envelope loop-invariant, so they hoist out of the element
    // loop and the t evaluation (dead: it feeds only the LUT branch) drops
    // with its hairT read. The per-element math below is the generic body's
    // with the same operands, so the stores are bitwise identical.
    if (magFlat && widthField.Uniform() && maskField.Uniform()) {
        float const width =
            static_cast<float>(widthField.Value(curveBase, cvBase));
        float const target = width * 1.0f;
        float const envelope = std::clamp(
            static_cast<float>(maskField.Value(curveBase, cvBase)), 0.0f, 1.0f);
        if (replace) {
            for (uint32_t c = 0; c < view->curveCount; ++c) {
                size_t const cv = view->cvCount ? size_t(view->cvCount)
                    : (view->cvOffsets ? size_t(view->cvOffsets[c + 1] - view->cvOffsets[c]) : 0);
                size_t const first = view->cvCount ? size_t(c) * view->cvCount
                    : (view->cvOffsets ? size_t(view->cvOffsets[c]) -
                         (view->desc ? view->desc->firstCv : 0) : 0);
                for (size_t i = 0; i < cv; ++i) {
                    const size_t o = first + i;
                    const float input = inWidth ? inWidth[o] : defaultInputWidth;
                    float const w = input + (target - input) * envelope;
                    view->width[o] = std::max(0.0f, w);
                }
            }
        } else {
            for (uint32_t c = 0; c < view->curveCount; ++c) {
                size_t const cv = view->cvCount ? size_t(view->cvCount)
                    : (view->cvOffsets ? size_t(view->cvOffsets[c + 1] - view->cvOffsets[c]) : 0);
                size_t const first = view->cvCount ? size_t(c) * view->cvCount
                    : (view->cvOffsets ? size_t(view->cvOffsets[c]) -
                         (view->desc ? view->desc->firstCv : 0) : 0);
                for (size_t i = 0; i < cv; ++i) {
                    const size_t o = first + i;
                    const float input = inWidth ? inWidth[o] : defaultInputWidth;
                    float const w = input * (1.0f + (target - 1.0f) * envelope);
                    view->width[o] = std::max(0.0f, w);
                }
            }
        }
        return;
    }
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const cv = view->cvCount ? size_t(view->cvCount)
            : (view->cvOffsets ? size_t(view->cvOffsets[c + 1] - view->cvOffsets[c]) : 0);
        size_t const first = view->cvCount ? size_t(c) * view->cvCount
            : (view->cvOffsets ? size_t(view->cvOffsets[c]) -
                 (view->desc ? view->desc->firstCv : 0) : 0);
        for (size_t i = 0; i < cv; ++i) {
            const size_t o = first + i;
            const float t = hairT ? hairT[o] : (cv > 1 ? float(i) / float(cv - 1) : 0.0f);
            // Interpolated ramp sampling — never a nearest LUT bin (04 §2.17).
            // Flat ramp (no knots) is exactly 1.0: skip the call (E-1).
            // `target` is the unmasked width profile.  The mask is the
            // operator-local envelope, not a multiplier on the target: set
            // and multiply have different identity values (04 §2.11).
            const float width =
                static_cast<float>(widthField.Value(curveBase + c, cvBase + o));
            float const target = width * (magFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t));
            // usdGen:mask IS the envelope, so a zero mask is a true
            // pass-through of the input width.
            const float envelope = std::clamp(
                static_cast<float>(maskField.Value(curveBase + c, cvBase + o)), 0.0f, 1.0f);
            // A style input is always defined.  Grow normally materializes
            // one, but a generic source may not; use the description default
            // in that case so the same local mask envelope still applies.
            const float input = inWidth ? inWidth[o] : defaultInputWidth;
            float const w = replace
                ? input + (target - input) * envelope
                : input * (1.0f + (target - 1.0f) * envelope);
            view->width[o] = std::max(0.0f, w);
        }
    }
}

}  // namespace usdGen
