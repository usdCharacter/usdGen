// usdGen — UsdGenWidthOp implementation (M1). 02-schema.md §2.7.1, 04 §2.11.
// Authors `widths` from a base width, a root->tip ramp (usdGen:width:knots,
// S11) and taper; usdGen:replace selects set (true) vs multiply (false).
// usdGen:width:profile:mode = "bezier" replaces the knots ramp with the
// imported groom's thickness profile: the quadratic Bezier through
// (0, root), (midShift, mid), (1, tip) at t = i/(n-1) (controls clamped to
// [0,1], midShift to [0.1,0.9]; all three controls 0 -> mid = 1), and
// target = max(width, 0) * clamp01(f) with f <= 1e-6 -> 1.
// TopologyEffect = None.
#include "usdGen/ops/width.h"
#include "usdGenCurves/types/curveCapture.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/ops/strandUtil.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include "usdGen/registries/registrar.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"


namespace usdGen {

using namespace widthP;

namespace {

UsdGenValue const kNoValue;

/// Read a float2[] ramp-knots property (02 §2.17). Missing or empty -> {}
/// (UsdGenBuildRampLut maps {} to the flat constant ramp).
UsdGenVec2fArray ReadRampKnots(UsdGenParamView const *p, UsdGenParamSlot slot)
{
    if (!p) return {};
    if (UsdGenValue const &v = p->GetValue(slot, kNoValue);
        v.IsHolding<UsdGenVec2fArray>()) {
        return v.UncheckedGet<UsdGenVec2fArray>();
    }
    return {};
}

// Views built outside the compiler (the GPU plan builders, until S4-6/7)
// carry no slots; read the node's params by spelling, as before.
UsdGenValue const &RawParam(UsdGenParamView const &p, UsdGenParamSlot slot, char const *name)
{
    if (p.op) return p.GetValue(slot, kNoValue);
    if (p.node)
        for (UsdGenParamValue const &q : p.node->params)
            if (q.name.GetString() == name) return q.value;
    return kNoValue;
}

double RawDouble(UsdGenParamView const &p, UsdGenParamSlot slot, char const *name,
                 double fallback)
{
    if (p.op) return p.GetDouble(slot, fallback);
    UsdGenValue const &v = RawParam(p, slot, name);
    if (v.IsHolding<double>()) return v.UncheckedGet<double>();
    if (v.IsHolding<float>()) return v.UncheckedGet<float>();
    return fallback;
}

bool IsBezierMode(UsdGenParamView const &p)
{
    if (p.op) return p.GetEnum(kProfileMode, kProfileRamp) == kProfileBezier;
    UsdGenValue const &v = RawParam(p, kProfileMode, "width:profile:mode");
    if (v.IsHolding<UsdGenToken>()) return v.UncheckedGet<UsdGenToken>().GetString() == "bezier";
    if (v.IsHolding<std::string>()) return v.UncheckedGet<std::string>() == "bezier";
    return false;
}

/// The thickness profile of usdGen:width:profile:mode = "bezier", read once
/// per evaluate (value-class, groom evaluation).
struct BezierWidth
{
    bool on = false;
    float root = 1.0f, mid = 1.0f, midShift = 0.5f, tip = 1.0f;

    /// clamp01(f) of point i of n, f <= 1e-6 -> 1.
    float At(size_t i, size_t n) const
    {
        float const t = n > 1 ? float(i) / float(n - 1) : 0.0f;
        float const f = strand::BezierProfile(t, root, mid, midShift, tip);
        return f <= 1e-6f ? 1.0f : strand::Clamp01(f);
    }
};

BezierWidth ReadBezierWidth(UsdGenParamView const *p)
{
    BezierWidth b;
    if (!p || !IsBezierMode(*p))
        return b;
    b.on = true;
    auto read = [&](UsdGenParamSlot slot, char const *name, float fallback) {
        float const v = float(RawDouble(*p, slot, name, fallback));
        return std::isfinite(v) ? v : fallback;
    };
    b.root = strand::Clamp01(read(kRoot, "width:root", 1.0f));
    b.mid = strand::Clamp01(read(kMid, "width:mid", 1.0f));
    b.midShift = read(kMidShift, "width:midShift", 0.5f);
    b.tip = strand::Clamp01(read(kTip, "width:tip", 1.0f));
    if (b.root + b.mid + b.tip <= 1e-6f) b.mid = 1.0f;
    return b;
}

struct UsdGenWidthCapture final : public UsdGenCapturePayload
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWidthCapture>(*this);
    }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        // Width never changes topology; hairT must exist (generator written it).
        return !upstream.hairT.empty();
    }
};

}  // namespace

bool UsdGenWidthBezierLut(UsdGenParamView const &params, float *lut, size_t size)
{
    BezierWidth const bezier = ReadBezierWidth(&params);
    if (!bezier.on || !lut || size < 2) return false;
    for (size_t k = 0; k < size; ++k) lut[k] = bezier.At(k, size);
    return true;
}

namespace {

constexpr char const *kProfileModes[] = {"ramp", "bezier"};
constexpr char const *kTokens[] = {"catmullRom"};
// Width reads no map; descriptors that bind a plain usdGen:map to it (the
// imaging and CUDA session fixtures) were accepted before and stay so.
constexpr UsdGenMapSlotDecl kMaps[] = {{"usdGen:map"}};

using D = expr::Domain;
UsdGenExprAdmission const kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
UsdGenExprAdmission const kBoolAll{expr::ScalarType::Bool, 1, "bool", D::All};
UsdGenExprAdmission const kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};

UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    // §6.1 structural: set vs multiply changes nothing about topology, but it
    // is a capture-class authoring choice.
    {"replace", UsdGenParamClass::Structural, true, kBoolAll},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},        // toggle: value-class (02 §6.3)
    {"width", UsdGenParamClass::Value, true, kFloatAll},
    {"width:knots", UsdGenParamClass::Value},
    // C1 property is usdGen:width:interpolation (schema.usda:369, 02 §2.7.1).
    // Value-class: the 257-entry LUT is rebuilt from the live parameters every
    // evaluate (see below).
    {"width:interpolation", UsdGenParamClass::Value},
    // Bezier thickness profile (value-class like the knots ramp).
    {"width:profile:mode", UsdGenParamClass::Value, true, {}, UsdGenSpan(kProfileModes)},
    {"width:root", UsdGenParamClass::Value},
    {"width:mid", UsdGenParamClass::Value},
    {"width:midShift", UsdGenParamClass::Value},
    {"width:tip", UsdGenParamClass::Value},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},           // operator envelope (02 §2.13)
};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenWidthOp>(r);
}

}  // namespace

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
    (void)params;
    (void)diag;
    return true;
}

UsdGenEpoch UsdGenWidthOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Only capture-class terms: `replace` (structural, 02 §6.1) and the
    // upstream generation. `width:knots` is value-class (02 :665) and is
    // deliberately excluded — its ramp is rebuilt every evaluate (gate SI-2).
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](std::string_view k, uint64_t v) {
        h ^= v; h *= 0x100000001b3ULL;
        h ^= k.size(); h *= 0x100000001b3ULL;
    };
    if (p) {
        feed("replace", p->GetBool(kReplace, true) ? 1 : 0);
    }
    feed("upstream", ctx.upstreamGeneration);
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenWidthOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &in,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    (void)ctx;
    (void)in;
    (void)out;
    (void)diag;
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
    UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    (void)captureIn;
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    // usdGen:width and usdGen:mask are CONNECTABLE: an authored literal is a
    // uniform field, a connected expression is a groom/primitive/point field
    // indexed by the shared rule in UsdGenParamField::Value.
    UsdGenParamField const widthField =
        p ? p->GetScalarField(kWidth, 0.01) : UsdGenParamField{0.01};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    const bool replace = p ? p->GetBool(kReplace, true) : true;
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
    BezierWidth const bezier = ReadBezierWidth(p);
    UsdGenVec2fArray const widthKnots = bezier.on ? UsdGenVec2fArray() : ReadRampKnots(p, kKnots);
    bool const magFlat = widthKnots.empty() && !bezier.on;
    thread_local std::vector<float> tLut;
    if (!widthKnots.empty()) {
        if (tLut.size() != kUsdGenRampLutSize) tLut.assign(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(widthKnots, p ? p->GetToken(kKnotsInterp, Record().Token(kTokCatmullRom))
                : Record().Token(kTokCatmullRom), tLut.data(), kUsdGenRampLutSize);
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
            float const target = bezier.on
                ? std::max(width, 0.0f) * bezier.At(i, cv)
                : width * (magFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t));
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

UsdGenOpId UsdGenRegisterWidthOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenWidth";
    d.displayName = "Width";
    d.create = &Create;
    d.countChanges = 0;
    d.params = UsdGenSpan(kParams);
    d.mapSlots = UsdGenSpan(kMaps);
    d.tokens = UsdGenSpan(kTokens);
    return r.RegisterOp(d);
}

}  // namespace usdGen
