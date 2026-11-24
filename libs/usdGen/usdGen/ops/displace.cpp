// usdGen — UsdGenDisplaceOp implementation. 02-schema.md §2.7, 04 §3.
// Height-field displacement along the rest root normal:
//   P[i] += N_root * amount * ((sample - base) * scale + offset) * mask.
//
// Map transport mirrors Clump: the CPU kernel never samples a map prim
// itself. UsdGenResolvedMapValue carries identity only, and the only CPU
// sampling path (cpuParameters ResolveSamplers, ptex()) runs inside the
// expression evaluator, so the rel target arrives here as an ordinary
// connected scalar field on displace:map, sampled per curve root (04 §3:
// "map sampled per root (CPU, capture, S37)"). Unconnected sample is 0.
// amount/base/scale/offset/mask are sampled per CV; mask clamps to [0, 1].
// Every CV including the root moves (no lockRoots on this type), so at zero
// effect (amount 0, mask 0, or a zero field) Evaluate copies the input
// bit-for-bit. mode = vector fails closed: only UsdGenPtexMap is consumable
// and only through single scalar channels (r/g/b/a/luminance), so no vector
// map type reaches the kernel. All work is per curve/CV in Evaluate (no
// capture payload); Capture only validates literals and connected values.
#include "usdGen/ops/displace.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

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

struct UsdGenDisplaceCapture final : public UsdGenCapturePayload
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenDisplaceCapture>(*this);
    }
};

bool Finite(double v) { return std::isfinite(v); }

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("mode"));          // structural kernel branch (02 §6)
    v.push_back(TfToken("displace:map"));  // rel retarget recompiles (02 §6)
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("mask"));
    v.push_back(TfToken("displace:amount"));
    v.push_back(TfToken("displace:base"));
    v.push_back(TfToken("displace:scale"));
    v.push_back(TfToken("displace:offset"));
    return v;
}();

TfSpan<const TfToken> UsdGenDisplaceOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenDisplaceOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenDisplaceOp::CreateCapture() const
{
    return std::make_unique<UsdGenDisplaceCapture>();
}
uint32_t UsdGenDisplaceOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenDisplaceOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sMode, sHeight);
    if (mode != sHeight && mode != sVector) {
        if (diag) diag->Error("UsdGenDisplace: unknown usdGen:mode '" + mode.GetString() + "'");
        return false;
    }
    if (mode == sVector) {
        if (diag) diag->Error("UsdGenDisplace: usdGen:mode 'vector' is not supported: "
                              "only UsdGenPtexMap height maps are consumable, through "
                              "single scalar channels");
        return false;
    }
    for (TfToken const *name : {&sAmount, &sBase, &sScale, &sOffset}) {
        double const v = params.GetDoubleLiteral(*name, 0.0);
        if (!Finite(v)) {
            if (diag) diag->Error("UsdGenDisplace: usdGen:" + name->GetString() +
                                  " must be finite");
            return false;
        }
    }
    return true;
}

UsdGenEpoch UsdGenDisplaceOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Pure-evaluate kernel: the digest is the structural mode token plus the
    // upstream generation and the seed. Value edits (including connected-value
    // edits, which the scheduler folds into the capture identity separately)
    // sweep without recapturing anything.
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    feed(p ? uint64_t(p->GetToken(sMode, sHeight).Hash()) : 0u);
    feed(ctx.upstreamGeneration);
    feed(uint64_t(ctx.seed));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenDisplaceOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    TF_UNUSED(out);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    // Literals and the mode token are validated here (the CPU lane never
    // calls Bind) and every connected value is scanned, so a non-finite
    // expression fails closed at capture instead of writing NaN points.
    if (!Bind(*p, diag)) return false;
    std::vector<uint32_t> spans;
    std::string spansError;
    if (!opUtil::CurveSpans(upstream, &spans, &spansError)) {
        if (diag) diag->Error("UsdGenDisplace: upstream CV topology " + spansError);
        return false;
    }
    uint32_t const R = upstream.totalCurves;
    UsdGenParamField const map = p->GetScalarField(sMap, 0.0);
    UsdGenParamField const amount = p->GetScalarField(sAmount, 0.0);
    UsdGenParamField const base = p->GetScalarField(sBase, 0.5);
    UsdGenParamField const scale = p->GetScalarField(sScale, 1.0);
    UsdGenParamField const offset = p->GetScalarField(sOffset, 0.0);
    UsdGenParamField const mask = p->GetScalarField(sMask, 1.0);
    for (UsdGenParamField const *field : {&map, &amount, &base, &scale, &offset, &mask}) {
        if (field->connected && field->components != 1) {
            if (diag) diag->Error("UsdGenDisplace: connected height controls must be "
                                  "scalar float");
            return false;
        }
    }
    for (uint32_t c = 0; c != R; ++c) {
        if (!Finite(map.Value(c, spans[c]))) {
            if (diag) diag->Error("UsdGenDisplace: per-strand displace:map must be finite");
            return false;
        }
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(amount.Value(c, cv)) || !Finite(base.Value(c, cv)) ||
                !Finite(scale.Value(c, cv)) || !Finite(offset.Value(c, cv)) ||
                !Finite(mask.Value(c, cv))) {
                if (diag) diag->Error("UsdGenDisplace: per-CV amount/base/scale/offset/mask "
                                      "must be finite");
                return false;
            }
        }
    }
    return true;
}

void UsdGenDisplaceOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    TF_UNUSED(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const mapField =
        p ? p->GetScalarField(sMap, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const amountField =
        p ? p->GetScalarField(sAmount, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const baseField =
        p ? p->GetScalarField(sBase, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const scaleField =
        p ? p->GetScalarField(sScale, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const offsetField =
        p ? p->GetScalarField(sOffset, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *rootN = view->rootN;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        // The map sample is per strand root (04 §3: sampled per root); the
        // remaining controls vary per CV.
        float const sample = static_cast<float>(mapField.Value(curve, cvBase + g));
        GfVec3f const N = rootN ? rootN[c] : GfVec3f(0.0f, 0.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            size_t const cv = cvBase + o;
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cv)), 0.0f, 1.0f);
            float const amount = static_cast<float>(amountField.Value(curve, cv));
            float const base = static_cast<float>(baseField.Value(curve, cv));
            float const scale = static_cast<float>(scaleField.Value(curve, cv));
            float const offset = static_cast<float>(offsetField.Value(curve, cv));
            float const d = amount * ((sample - base) * scale + offset) * mask;
            if (d == 0.0f) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            px[o] = inPx[o] + N[0] * d;
            py[o] = inPy[o] + N[1] * d;
            pz[o] = inPz[o] + N[2] * d;
        }
    }
}

UsdGenOpId UsdGenRegisterDisplaceOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenDisplaceOp().Type(), [] { return std::make_unique<UsdGenDisplaceOp>(); });
}

}  // namespace usdGen
