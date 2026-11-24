// usdGen — UsdGenScaleOp implementation. 02-schema.md §2.7.1, 04 §2.14.
// Global length multiplier about each root: with the ramp R, the per-curve
// target T_c = scale_c * mult_c and the per-CV mask w,
//   L(c,i) = lerp(1, T_c * R(t_i), w(c,i))
//   P[i]   = root + (P_in[i] - root) * L(c,i).
// The root offset is the zero vector, so roots never move. scale is sampled
// per curve root, the ramp over hairT per CV, the mask per CV (clamped to
// [0,1]); mult_c is the UsdGenDraw01 draw over the per-root scaleRandom
// range, or the scalar lo when the range is degenerate. widthToo scales
// widths by the same L. All work is per curve/CV in Evaluate (no capture
// payload — the draws are pure functions of seed, curveId and salt, so a
// scaleRandom edit re-rolls by re-evaluating); Capture only validates.
#include "usdGen/ops/scale.h"
#include "usdGen/ops/stylerParams.h"
#include "usdGen/registries/engineTokens.h"

#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"

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

using namespace scaleP;

namespace {

struct UsdGenScaleCapture final : public UsdGenCapturePayload
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenScaleCapture>(*this);
    }
};

bool Finite(double v) { return std::isfinite(v); }

VtVec2fArray ReadRampKnots(UsdGenParamView const *p, UsdGenParamSlot name)
{
    if (!p) return {};
    if (VtValue const v = p->GetValue(name, VtValue());
        v.IsHolding<VtVec2fArray>()) {
        return v.UncheckedGet<VtVec2fArray>();
    }
    return {};
}

inline void ReadVec2Prop(UsdGenParamView const &p, UsdGenParamSlot name,
                         double &lo, double &hi)
{
    lo = 1.0; hi = 1.0;
    if (VtValue const v = p.GetValue(name, VtValue()); v.IsHolding<VtFloatArray>()) {
        VtFloatArray const a = v.UncheckedGet<VtFloatArray>();
        if (a.size() >= 2) { lo = a[0]; hi = a[1]; }
    } else if (v.IsHolding<GfVec2f>()) {
        GfVec2f const r = v.UncheckedGet<GfVec2f>();
        lo = r[0]; hi = r[1];
    }
}

}  // namespace

namespace {
using D = expr::Domain;
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kFloatRoot{expr::ScalarType::Float32, 1, "float", D::Groom | D::Primitive};
constexpr UsdGenExprAdmission kFloat2Root{expr::ScalarType::Float32, 2, "float2", D::Groom | D::Primitive};
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"widthToo", UsdGenParamClass::Capture, true, kBoolGroom},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"scale", UsdGenParamClass::Value, true, kFloatRoot},
    {"scale:knots", UsdGenParamClass::Value},
    {"scale:interpolation", UsdGenParamClass::Value},
    {"scaleRandom", UsdGenParamClass::Value, true, kFloat2Root},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
};
std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *) { return std::make_unique<UsdGenScaleOp>(r); }
}  // namespace
std::unique_ptr<UsdGenCapture> UsdGenScaleOp::CreateCapture() const
{
    return std::make_unique<UsdGenScaleCapture>();
}
uint32_t UsdGenScaleOp::PlanesTouched() const
{
    return kPlanePoints | kPlaneWidths;
}

bool UsdGenScaleOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    double const scale = params.GetDoubleLiteral(kScale, 1.0);
    if (!Finite(scale) || scale < 0.0) {
        if (diag) diag->Error("UsdGenScale: usdGen:scale must be finite and >= 0");
        return false;
    }
    double lo = 1.0, hi = 1.0;
    ReadVec2Prop(params, kScaleRandom, lo, hi);
    if (!Finite(lo) || !Finite(hi) || lo < 0.0 || hi < 0.0) {
        if (diag) diag->Error("UsdGenScale: usdGen:scaleRandom must be finite and >= 0");
        return false;
    }
    // usdGen:scale:interpolation is consumed by the R11 LUT builder, whose
    // documented fallback maps an unknown token to linear (02 §2.17) — the
    // same contract every sibling ramp op implements, so Bind stays silent
    // here instead of failing closed. usdGen:widthToo is a plain bool with no
    // range to validate; usdGen:enabled is scheduler-owned (the node gate)
    // and usdGen:mask is the clamped envelope, neither needing a literal check.
    return true;
}

UsdGenEpoch UsdGenScaleOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Pure-evaluate kernel: the digest covers the structural token plus the
    // upstream generation plus the seed only. Value parameters (including the
    // per-curve draw range) are excluded, so literal edits sweep without
    // recapturing anything.
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    feed(p && p->GetBool(kWidthToo, false) ? 1u : 0u);
    feed(ctx.upstreamGeneration);
    feed(uint64_t(ctx.seed));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenScaleOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    TF_UNUSED(out);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    if (!Bind(*p, diag)) return false;
    uint32_t const R = upstream.totalCurves;
    uint32_t const nCvs = upstream.totalCvs;
    std::vector<uint32_t> spans(size_t(R) + 1, 0);
    if (!upstream.cvOffsets.empty()) {
        if (upstream.cvOffsets.size() != size_t(R) + 1) {
            if (diag) diag->Error("UsdGenScale: upstream CV topology has invalid ragged offsets");
            return false;
        }
        for (uint32_t c = 0; c != R; ++c) {
            int const first = upstream.cvOffsets[c], last = upstream.cvOffsets[c + 1];
            if (first < 0 || last < first) {
                if (diag) diag->Error("UsdGenScale: upstream CV topology has decreasing ragged offsets");
                return false;
            }
            spans[c] = uint32_t(first);
            spans[c + 1] = uint32_t(last);
        }
    } else if (R && nCvs % R != 0) {
        if (diag) diag->Error("UsdGenScale: upstream CV topology is non-uniform without offsets");
        return false;
    } else {
        uint32_t const perCurve = R ? nCvs / R : 0;
        for (uint32_t c = 0; c != R; ++c) spans[c + 1] = spans[c] + perCurve;
    }
    UsdGenParamField const scale = p->GetScalarField(kScale, 1.0);
    UsdGenParamField const random = p->GetScalarField(kScaleRandom, 1.0);
    if ((scale.connected && scale.components != 1) ||
        (random.connected && random.components != 2)) {
        if (diag) diag->Error("UsdGenScale: connected scale must be float and "
                              "scaleRandom must be float2");
        return false;
    }
    // Both fields are sampled per curve root (02 §2.7.1 / schema evaluation),
    // so the scan covers exactly the root element of every strand.
    for (uint32_t c = 0; c != R; ++c) {
        uint32_t const root = spans[c];
        double const s = scale.Value(c, root);
        if (!Finite(s) || s < 0.0) {
            if (diag) diag->Error("UsdGenScale: per-strand scale must be finite and >= 0");
            return false;
        }
        if (random.connected) {
            double const lo = random.Value(c, root, 0);
            double const hi = random.Value(c, root, 1);
            if (!Finite(lo) || !Finite(hi) || lo < 0.0 || hi < 0.0) {
                if (diag) diag->Error("UsdGenScale: per-strand scaleRandom must be finite and >= 0");
                return false;
            }
        }
    }
    return true;
}

void UsdGenScaleOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    TF_UNUSED(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const scaleField =
        p ? p->GetScalarField(kScale, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const randomField =
        p ? p->GetScalarField(kScaleRandom, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    double litLo = 1.0, litHi = 1.0;
    bool const widthToo = p ? p->GetBool(kWidthToo, false) : false;
    if (p) {
        ReadVec2Prop(*p, kScaleRandom, litLo, litHi);
        if (litLo > litHi) std::swap(litLo, litHi);
    }
    float const *lutPtr = nullptr;
    VtVec2fArray const knots = ReadRampKnots(p, kScaleKnots);
    bool const rampFlat = knots.empty();
    thread_local std::vector<float> tLut;
    if (!rampFlat) {
        if (tLut.size() != kUsdGenRampLutSize) tLut.assign(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(knots,
            p ? p->GetToken(kScaleInterp, UsdGenTokens().catmullRom) : UsdGenTokens().catmullRom,
            tLut.data(), kUsdGenRampLutSize);
        lutPtr = tLut.data();
    }
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *hairT = view->hairT;
    auto const *curveId = view->curveId;
    auto const *inWidth = view->inWidth;
    float *width = view->width;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    float defaultInputWidth = ctx.desc ? ctx.desc->defaultWidth : 0.01f;
    if (!std::isfinite(defaultInputWidth) || defaultInputWidth < 0.0f)
        defaultInputWidth = 0.01f;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        size_t const rootAbs = cvBase + g;
        // Per-curve length factor: the scale control sampled at the root
        // times the per-curve draw over the scaleRandom range.
        float const scaleC = static_cast<float>(scaleField.Value(curve, rootAbs));
        double lo = litLo, hi = litHi;
        if (randomField.connected) {
            lo = randomField.Value(curve, rootAbs, 0);
            hi = randomField.Value(curve, rootAbs, 1);
            if (lo > hi) std::swap(lo, hi);
        }
        float mult;
        if (lo == hi) {
            mult = static_cast<float>(lo);  // degenerate range: the scalar itself (1.0 by default)
        } else {
            uint64_t const id = curveId ? curveId[c] : 0;
            float const draw = UsdGenDraw01(int(ctx.seed), id, kSaltScale);
            mult = float(lo + (hi - lo) * double(draw));
        }
        float const target = scaleC * mult;
        float const rx = inPx[g], ry = inPy[g], rz = inPz[g];
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            float const t = hairT ? hairT[o]
                : (n > 1 ? float(i) / float(n - 1) : 0.0f);
            float const ramp = rampFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t);
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            float const L = 1.0f + (target * ramp - 1.0f) * mask;
            float const inW = inWidth ? inWidth[o] : defaultInputWidth;
            if (L == 1.0f) {
                // Bitwise pass-through at zero effect (mask 0 or an identity
                // factor): no rounding dust on points or widths.
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                if (width) width[o] = inW;
                continue;
            }
            px[o] = rx + (inPx[o] - rx) * L;
            py[o] = ry + (inPy[o] - ry) * L;
            pz[o] = rz + (inPz[o] - rz) * L;
            // The widths plane is scheduler-fresh (zeros), never upstream
            // storage, so the !widthToo path must copy through; the widthToo
            // path scales by the same L and clamps (a negative ramp factor
            // mirrors points through the root but must not emit < 0 widths).
            if (width) width[o] = widthToo ? std::max(0.0f, inW * L) : inW;
        }
    }
}

UsdGenOpId UsdGenRegisterScaleOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenScale";
    d.displayName = "Scale";
    d.create = &Create;
    d.topoFx = UsdGenTopoFx::None;
    d.params = UsdGenSpan(kParams);
    return r.RegisterOp(d);
}

}  // namespace usdGen
