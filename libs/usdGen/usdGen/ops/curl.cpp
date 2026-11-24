// usdGen — UsdGenCurlOp implementation. 02-schema.md §2.7, 04 §3.
// Coils each strand around its own tangent frame:
//   P[i] += r(t) * (cos(theta_i) N_i + sgn * sin(theta_i) B_i),
//   theta_i = 2*pi*f*s_i + phi_c, s = arc length from the root,
//   r(t) = radius * radiusRamp(t) * (taper ? t : 1).
// (N, B) are rotation-minimizing frames propagated along the input curve
// (axisMode = curveTangent) or the orthonormalized root frame (axisMode =
// guide, which coils around the root tangent). The per-curve phase is
// phi_c = phase + phaseRandom * draw01(seed, curveId) * 2*pi. Roots stay
// fixed under the default taper; without it the whole coil (roots included)
// offsets by r(0). All work is per curve/CV in Evaluate; Capture only
// validates.
#include "usdGen/ops/curl.h"
#include "usdGen/scratch.h"
#include "usdGen/clumpMotion.h"

#include "usdGen/opParams.h"
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

namespace {

struct UsdGenCurlCapture final : public UsdGenCapturePayload
{
    UsdGenClumpMotion clumpMotion;
    std::vector<std::vector<float>> groupDraws;
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenCurlCapture>(*this);
    }
};

bool Finite(double v) { return std::isfinite(v); }

VtVec2fArray ReadRampKnots(UsdGenParamView const *p, TfToken const &name)
{
    if (!p) return {};
    if (VtValue const v = p->GetVtValue(name, VtValue());
        v.IsHolding<VtVec2fArray>()) {
        return v.UncheckedGet<VtVec2fArray>();
    }
    return {};
}

inline GfVec3f Normalized(GfVec3f v, GfVec3f const &fallback)
{
    float const l = v.GetLength();
    return l > 1e-12f ? v / l : fallback;
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("axisMode"));   // structural: selects the frame (02 §2.7.2)
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("radius"));
    v.push_back(TfToken("radius:knots"));
    v.push_back(TfToken("radius:interpolation"));
    v.push_back(TfToken("frequency"));
    v.push_back(TfToken("phase"));
    v.push_back(TfToken("phaseRandom"));
    v.push_back(TfToken("taper"));
    v.push_back(TfToken("clockwise"));
    v.push_back(TfToken("mask"));
    return v;
}();

TfSpan<const TfToken> UsdGenCurlOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenCurlOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenCurlOp::CreateCapture() const
{
    return std::make_unique<UsdGenCurlCapture>();
}
uint32_t UsdGenCurlOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenCurlOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sAxisMode, sCurveTangent);
    if (mode != sCurveTangent && mode != sGuide) {
        if (diag) diag->Error("UsdGenCurl: unknown usdGen:axisMode '" + mode.GetString() + "'");
        return false;
    }
    for (char const *name : {"radius", "frequency", "phase", "phaseRandom"}) {
        if (!Finite(params.GetDoubleLiteral(TfToken(name), 0.0))) {
            if (diag) diag->Error(std::string("UsdGenCurl: usdGen:") + name + " must be finite");
            return false;
        }
    }
    return true;
}

UsdGenEpoch UsdGenCurlOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    feed(p ? uint64_t(p->GetToken(sAxisMode, sCurveTangent).Hash()) : 0u);
    feed(ctx.upstreamGeneration);
    feed(uint64_t(ctx.seed));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenCurlOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    auto &cap = *static_cast<UsdGenCurlCapture *>(out);
    std::string clumpError;
    auto const *prior = dynamic_cast<UsdGenCurlCapture const *>(ctx.previousCapture);
    auto const *previousMotion = prior ? &prior->clumpMotion :
        (cap.clumpMotion.levels.empty() ? nullptr : &cap.clumpMotion);
    if (!UsdGenBuildClumpMotion(upstream, &cap.clumpMotion, &clumpError,
                                previousMotion)) {
        if (diag) diag->Error("UsdGenCurl: " + clumpError);
        return false;
    }
    cap.groupDraws.clear();
    for (auto const &level : cap.clumpMotion.levels) {
        auto &draws = cap.groupDraws.emplace_back();
        draws.reserve(level.groups.size());
        for (auto const &group : level.groups)
            draws.push_back(UsdGenDraw01(int(ctx.seed), group.centerId, kSaltCurl));
    }
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    TfToken const mode = p->GetToken(sAxisMode, sCurveTangent);
    if (mode != sCurveTangent && mode != sGuide) {
        if (diag) diag->Error("UsdGenCurl: unknown usdGen:axisMode '" + mode.GetString() + "'");
        return false;
    }
    for (char const *name : {"radius", "frequency", "phase", "phaseRandom"}) {
        if (!Finite(p->GetDoubleLiteral(TfToken(name), 0.0))) {
            if (diag) diag->Error(std::string("UsdGenCurl: usdGen:") + name + " must be finite");
            return false;
        }
    }
    uint32_t const R = upstream.totalCurves;
    uint32_t const nCvs = upstream.totalCvs;
    std::vector<uint32_t> spans(size_t(R) + 1, 0);
    if (!upstream.cvOffsets.empty()) {
        if (upstream.cvOffsets.size() != size_t(R) + 1) {
            if (diag) diag->Error("UsdGenCurl: upstream CV topology has invalid ragged offsets");
            return false;
        }
        for (uint32_t c = 0; c != R; ++c) {
            int const first = upstream.cvOffsets[c], last = upstream.cvOffsets[c + 1];
            if (first < 0 || last < first) {
                if (diag) diag->Error("UsdGenCurl: upstream CV topology has decreasing ragged offsets");
                return false;
            }
            spans[c] = uint32_t(first);
            spans[c + 1] = uint32_t(last);
        }
    } else if (R && nCvs % R != 0) {
        if (diag) diag->Error("UsdGenCurl: upstream CV topology is non-uniform without offsets");
        return false;
    } else {
        uint32_t const perCurve = R ? nCvs / R : 0;
        for (uint32_t c = 0; c != R; ++c) spans[c + 1] = spans[c] + perCurve;
    }
    UsdGenParamField const radius = p->GetScalarField(sRadius, 0.02);
    UsdGenParamField const frequency = p->GetScalarField(sFrequency, 2.0);
    UsdGenParamField const phase = p->GetScalarField(sPhase, 0.0);
    UsdGenParamField const phaseRandom = p->GetScalarField(sPhaseRandom, 0.0);
    for (uint32_t c = 0; c != R; ++c) {
        uint32_t const root = spans[c];
        if (!Finite(frequency.Value(c, root)) || !Finite(phase.Value(c, root)) ||
            !Finite(phaseRandom.Value(c, root))) {
            if (diag) diag->Error("UsdGenCurl: per-strand frequency/phase must be finite");
            return false;
        }
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(radius.Value(c, cv))) {
                if (diag) diag->Error("UsdGenCurl: per-CV radius must be finite");
                return false;
            }
        }
    }
    return true;
}

void UsdGenCurlOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenCurlCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const radiusField =
        p ? p->GetScalarField(sRadius, 0.02) : UsdGenParamField{0.02};
    UsdGenParamField const frequencyField =
        p ? p->GetScalarField(sFrequency, 2.0) : UsdGenParamField{2.0};
    UsdGenParamField const phaseField =
        p ? p->GetScalarField(sPhase, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const phaseRandomField =
        p ? p->GetScalarField(sPhaseRandom, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    bool const taper = p ? p->GetBool(sTaper, true) : true;
    bool const clockwise = p ? p->GetBool(sClockwise, true) : true;
    bool const guideMode = p && p->GetToken(sAxisMode, sCurveTangent) == sGuide;
    float const *lutPtr = nullptr;
    VtVec2fArray const knots = ReadRampKnots(p, sRadiusKnots);
    bool const rampFlat = knots.empty();
    UsdGenScratchScope scratch;   // chunk-local; rewinds when Evaluate returns
    if (!rampFlat) {
        float *lut = scratch.Filled<float>(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(knots,
            p ? p->GetToken(sRadiusInterp, sCatmullRom) : sCatmullRom,
            lut, kUsdGenRampLutSize);
        lutPtr = lut;
    }
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *hairT = view->hairT;
    auto const *rootN = view->rootN, *rootB = view->rootB;
    auto const *curveId = view->curveId;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    constexpr float twoPi = 6.28318530717958647692f;
    float const sgn = clockwise ? -1.0f : 1.0f;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        float const freq = static_cast<float>(frequencyField.Value(curve, cvBase + g));
        float const phase = static_cast<float>(phaseField.Value(curve, cvBase + g));
        float const phaseRandom =
            static_cast<float>(phaseRandomField.Value(curve, cvBase + g));
        uint64_t const id = curveId ? curveId[c] : 0;
        float const individualDraw = UsdGenDraw01(int(ctx.seed), id, kSaltCurl);
        GfVec3f frameN = rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f);
        GfVec3f const frameB = rootB ? rootB[c] : GfVec3f(0.0f, 0.0f, 1.0f);
        // Seed the rotation-minimizing normal: any vector off the first
        // tangent, projected and normalized.
        {
            GfVec3f t0 = n > 1
                ? GfVec3f(inPx[g + 1] - inPx[g], inPy[g + 1] - inPy[g], inPz[g + 1] - inPz[g])
                : GfVec3f(0.0f, 1.0f, 0.0f);
            t0 = Normalized(t0, GfVec3f(0.0f, 1.0f, 0.0f));
            GfVec3f seed = std::fabs(GfDot(t0, GfVec3f(0.0f, 1.0f, 0.0f))) > 0.9f
                ? GfVec3f(1.0f, 0.0f, 0.0f) : GfVec3f(0.0f, 1.0f, 0.0f);
            seed -= t0 * GfDot(seed, t0);
            frameN = Normalized(seed, GfVec3f(1.0f, 0.0f, 0.0f));
        }
        float s = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            if (i > 0) {
                float const dx = inPx[o] - inPx[o - 1];
                float const dy = inPy[o] - inPy[o - 1];
                float const dz = inPz[o] - inPz[o - 1];
                s += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            // Tangent at this CV (central differences inside, one-sided at
            // the ends) from the INPUT curve, so the coil follows deforming
            // strands without a capture-time frame table.
            size_t const a = g + (i > 0 ? i - 1 : 0);
            size_t const b = g + (i + 1 < n ? i + 1 : i);
            GfVec3f const tangent = Normalized(
                GfVec3f(inPx[b] - inPx[a], inPy[b] - inPy[a], inPz[b] - inPz[a]),
                GfVec3f(0.0f, 1.0f, 0.0f));
            GfVec3f N, B;
            if (guideMode) {
                N = Normalized((rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f)) -
                    tangent * GfDot(rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f), tangent),
                    frameN);
                B = GfCross(tangent, N);
                if (B.GetLength() < 1e-12f) B = frameB;
                B = Normalized(B, frameB);
            } else {
                if (i > 0) {
                    GfVec3f proj = frameN - tangent * GfDot(frameN, tangent);
                    if (proj.GetLength() > 1e-12f) frameN = proj / proj.GetLength();
                }
                N = frameN;
                B = Normalized(GfCross(tangent, N), frameB);
            }
            float const t = hairT ? hairT[o]
                : (n > 1 ? float(i) / float(n - 1) : 0.0f);
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            float const radius =
                static_cast<float>(radiusField.Value(curve, cvBase + o));
            float r = radius * (rampFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t)) * mask;
            if (taper) r *= t;
            if (r == 0.0f) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            float draw = individualDraw;
            float remaining = 1.0f;
            // Blend the stochastic phase at each CV so the clump's root lock,
            // profile and mask remain effective at their authored positions.
            for (size_t li = cap.clumpMotion.levels.size(); li-- > 0 && remaining > 0.0f;) {
                auto const &level = cap.clumpMotion.levels[li];
                if (level.weight.empty() || curve >= level.groupForCurve.size()) continue;
                uint32_t const group = level.groupForCurve[curve];
                if (group == UINT32_MAX || group >= level.groups.size()) continue;
                float const portion = remaining * level.weight[cvBase + o];
                draw += portion * (cap.groupDraws[li][group] - individualDraw);
                remaining -= portion;
            }
            float const theta = twoPi * freq * s + phase + phaseRandom * draw * twoPi;
            float const co = std::cos(theta), si = std::sin(theta);
            px[o] = inPx[o] + r * (co * N[0] + sgn * si * B[0]);
            py[o] = inPy[o] + r * (co * N[1] + sgn * si * B[1]);
            pz[o] = inPz[o] + r * (co * N[2] + sgn * si * B[2]);
        }
    }
}

UsdGenOpId UsdGenRegisterCurlOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenCurlOp().Type(), [] { return std::make_unique<UsdGenCurlOp>(); });
}

}  // namespace usdGen
