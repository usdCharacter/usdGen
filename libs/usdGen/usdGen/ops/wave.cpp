// usdGen — UsdGenWaveOp implementation. 02-schema.md §2.7, 04 §3.
// Sinusoidal displacement in the root frame (T, N):
//   P[i] += A_T sin(2*pi*f_T*s_i) T + A_N sin(2*pi*f_N*s_i) N,
// s_i = arc length from the root along the INPUT curve. Roots (s = 0) never
// move, so the wave keeps every follicle. Frequencies are turns per stage
// unit, sampled per curve root; amplitudes and the mask envelope are sampled
// per CV. All work is per curve/CV in Evaluate (no capture payload); Capture
// only validates literals and fails closed on non-finite connected values.
// There is no per-strand stochastic term to decorrelate: identical authored
// wave controls and input arc lengths already give identical displacement.
// Clump motion metadata therefore does not alter this deterministic styler;
// changing its frequency or root frame would override authored wave shape.
#include "usdGen/ops/wave.h"
#include "usdGen/ops/stylerParams.h"
#include "usdGen/registries/engineTokens.h"


#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "usdGen/registries/registrar.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

using namespace waveP;

namespace {

struct UsdGenWaveCapture final : public UsdGenCapturePayload
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWaveCapture>(*this);
    }
};

bool Finite(double v) { return std::isfinite(v); }

}  // namespace

namespace {
using D = expr::Domain;
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"frequencyU", UsdGenParamClass::Value, true, kFloatAll},
    {"frequencyN", UsdGenParamClass::Value, true, kFloatAll},
    {"amplitudeU", UsdGenParamClass::Value, true, kFloatAll},
    {"amplitudeN", UsdGenParamClass::Value, true, kFloatAll},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
};
std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &, void *) { return std::make_unique<UsdGenWaveOp>(); }
}  // namespace
std::unique_ptr<UsdGenCapture> UsdGenWaveOp::CreateCapture() const
{
    return std::make_unique<UsdGenWaveCapture>();
}
uint32_t UsdGenWaveOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenWaveOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    for (UsdGenParamSlot s : {kFrequencyU, kFrequencyN, kAmplitudeU, kAmplitudeN}) {
        char const *const name = kParams[s].name;
        double const v = params.GetDoubleLiteral(s, 0.0);
        if (!Finite(v)) {
            if (diag) diag->Error(std::string("UsdGenWave: usdGen:") + name + " must be finite");
            return false;
        }
    }
    return true;
}

UsdGenEpoch UsdGenWaveOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Pure-evaluate kernel: no capture payload, so the digest is the upstream
    // generation only. Value edits (including connected-value edits, which the
    // scheduler folds into the capture identity separately) sweep without
    // recapturing anything.
    uint64_t h = 1469598103934665603ULL;
    h ^= ctx.upstreamGeneration; h *= 0x100000001b3ULL;
    h ^= uint64_t(ctx.seed); h *= 0x100000001b3ULL;
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenWaveOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    TF_UNUSED(out);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    // Literals are validated here (the CPU lane never calls Bind) and every
    // connected value is scanned, so a non-finite expression fails closed at
    // capture instead of writing NaN points per frame.
    for (UsdGenParamSlot s : {kFrequencyU, kFrequencyN, kAmplitudeU, kAmplitudeN}) {
        char const *const name = kParams[s].name;
        if (!Finite(p->GetDoubleLiteral(s, 0.0))) {
            if (diag) diag->Error(std::string("UsdGenWave: usdGen:") + name + " must be finite");
            return false;
        }
    }
    uint32_t const R = upstream.totalCurves;
    uint32_t const nCvs = upstream.totalCvs;
    std::vector<uint32_t> spans(size_t(R) + 1, 0);
    if (!upstream.cvOffsets.empty()) {
        if (upstream.cvOffsets.size() != size_t(R) + 1) {
            if (diag) diag->Error("UsdGenWave: upstream CV topology has invalid ragged offsets");
            return false;
        }
        for (uint32_t c = 0; c != R; ++c) {
            int const first = upstream.cvOffsets[c], last = upstream.cvOffsets[c + 1];
            if (first < 0 || last < first) {
                if (diag) diag->Error("UsdGenWave: upstream CV topology has decreasing ragged offsets");
                return false;
            }
            spans[c] = uint32_t(first);
            spans[c + 1] = uint32_t(last);
        }
    } else if (R && nCvs % R != 0) {
        if (diag) diag->Error("UsdGenWave: upstream CV topology is non-uniform without offsets");
        return false;
    } else {
        uint32_t const perCurve = R ? nCvs / R : 0;
        for (uint32_t c = 0; c != R; ++c) spans[c + 1] = spans[c] + perCurve;
    }
    UsdGenParamField const freqU = p->GetScalarField(kFrequencyU, 1.0);
    UsdGenParamField const freqN = p->GetScalarField(kFrequencyN, 1.0);
    UsdGenParamField const ampU = p->GetScalarField(kAmplitudeU, 0.0);
    UsdGenParamField const ampN = p->GetScalarField(kAmplitudeN, 0.0);
    for (uint32_t c = 0; c != R; ++c) {
        uint32_t const root = spans[c];
        if (!Finite(freqU.Value(c, root)) || !Finite(freqN.Value(c, root))) {
            if (diag) diag->Error("UsdGenWave: per-strand frequencies must be finite");
            return false;
        }
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(ampU.Value(c, cv)) || !Finite(ampN.Value(c, cv))) {
                if (diag) diag->Error("UsdGenWave: per-CV amplitudes must be finite");
                return false;
            }
        }
    }
    return true;
}

void UsdGenWaveOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    TF_UNUSED(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const freqUField =
        p ? p->GetScalarField(kFrequencyU, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const freqNField =
        p ? p->GetScalarField(kFrequencyN, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const ampUField =
        p ? p->GetScalarField(kAmplitudeU, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const ampNField =
        p ? p->GetScalarField(kAmplitudeN, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *rootT = view->rootT, *rootN = view->rootN;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    constexpr float twoPi = 6.28318530717958647692f;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        float const freqU = static_cast<float>(freqUField.Value(curve, cvBase + g));
        float const freqN = static_cast<float>(freqNField.Value(curve, cvBase + g));
        GfVec3f const T = rootT ? rootT[c] : GfVec3f(1.0f, 0.0f, 0.0f);
        GfVec3f const N = rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f);
        float s = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            if (i > 0) {
                float const dx = inPx[o] - inPx[o - 1];
                float const dy = inPy[o] - inPy[o - 1];
                float const dz = inPz[o] - inPz[o - 1];
                s += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            float const ampU = static_cast<float>(ampUField.Value(curve, cvBase + o));
            float const ampN = static_cast<float>(ampNField.Value(curve, cvBase + o));
            if (mask == 0.0f || (ampU == 0.0f && ampN == 0.0f)) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            float const dT = ampU * std::sin(twoPi * freqU * s) * mask;
            float const dN = ampN * std::sin(twoPi * freqN * s) * mask;
            px[o] = inPx[o] + T[0] * dT + N[0] * dN;
            py[o] = inPy[o] + T[1] * dT + N[1] * dN;
            pz[o] = inPz[o] + T[2] * dT + N[2] * dN;
        }
    }
}

UsdGenOpId UsdGenRegisterWaveOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenWave";
    d.displayName = "Wave";
    d.create = &Create;
    d.topoFx = UsdGenTopoFx::None;
    d.params = UsdGenSpan(kParams);
    return r.RegisterOp(d);
}

}  // namespace usdGen
