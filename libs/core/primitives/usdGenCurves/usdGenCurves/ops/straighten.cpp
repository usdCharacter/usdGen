// usdGen — UsdGenStraightenOp implementation. 02-schema.md §2.7, 04 §3.
// Blend each CV toward the root-to-tip straight line, decomposed in the root
// frame (04 §3: P[i] = lerp(P[i], root + chord*s_i, k) decomposed per plane):
//   linePos_i = root + (tip - root) * s_i,
//   d         = in[i] - linePos_i = dT*T + dN*N + dB*B,
//   out[i]    = linePos_i + T*dT*(1-kT) + N*dN*(1-kN) + B*dB,
// with kT = tangentStraightness * mask and kN = normalStraightness * mask.
// s_i is the arc-length fraction from the root along the INPUT curve (the
// wave-style arc-length parameterization; index fraction when the strand has
// zero length). Roots and tips are the line endpoints and are copied exactly,
// so they stay fixed. The binormal component passes through: the schema and
// plan name only the two controlled planes, so there is nothing to scale it
// by. All work is per curve/CV in Evaluate (no capture payload); Capture only
// validates literals and fails closed on non-finite connected values.
#include "usdGen/ops/straighten.h"
#include "usdGenCurves/types/curveCapture.h"
#include "usdGenCurves/types/curveSet.h"
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

using namespace straightenP;

namespace {

struct UsdGenStraightenCapture final : public UsdGenCapturePayload
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenStraightenCapture>(*this);
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
    {"tangentStraightness", UsdGenParamClass::Value, true, kFloatAll},
    {"normalStraightness", UsdGenParamClass::Value, true, kFloatAll},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
};
std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *) { return std::make_unique<UsdGenStraightenOp>(r); }
}  // namespace
std::unique_ptr<UsdGenCapture> UsdGenStraightenOp::CreateCapture() const
{
    return std::make_unique<UsdGenStraightenCapture>();
}
uint32_t UsdGenStraightenOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenStraightenOp::Bind(UsdGenParamView const &params,
                              UsdGenDiagnostics *diag)
{
    // Neither the schema nor 02 §2.7.2 states a range for the straightness
    // pair, so literals are validated finite only (overshoot/negative values
    // blend past the line or away from it, like wave amplitudes).
    for (UsdGenParamSlot s : {kTangentStraightness, kNormalStraightness}) {
        char const *const name = kParams[s].name;
        double const v = params.GetDoubleLiteral(s, 0.0);
        if (!Finite(v)) {
            if (diag) diag->Error(std::string("UsdGenStraighten: usdGen:") + name +
                                  " must be finite");
            return false;
        }
    }
    return true;
}

UsdGenEpoch UsdGenStraightenOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Pure-evaluate kernel with no structural tokens: the digest is the
    // upstream generation plus the seed only. Value edits (including
    // connected-value edits, which the scheduler folds into the capture
    // identity separately) sweep without recapturing anything.
    uint64_t h = 1469598103934665603ULL;
    h ^= ctx.upstreamGeneration; h *= 0x100000001b3ULL;
    h ^= uint64_t(ctx.seed); h *= 0x100000001b3ULL;
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenStraightenOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &in,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in.Primary());
    TF_UNUSED(out);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    // Literals are validated here (the CPU lane never calls Bind) and every
    // connected value is scanned, so a non-finite expression fails closed at
    // capture instead of writing NaN points per frame.
    if (!Bind(*p, diag)) return false;
    uint32_t const R = upstream.totalCurves;
    uint32_t const nCvs = upstream.totalCvs;
    std::vector<uint32_t> spans(size_t(R) + 1, 0);
    if (!upstream.cvOffsets.empty()) {
        if (upstream.cvOffsets.size() != size_t(R) + 1) {
            if (diag) diag->Error("UsdGenStraighten: upstream CV topology has invalid ragged offsets");
            return false;
        }
        for (uint32_t c = 0; c != R; ++c) {
            int const first = upstream.cvOffsets[c], last = upstream.cvOffsets[c + 1];
            if (first < 0 || last < first) {
                if (diag) diag->Error("UsdGenStraighten: upstream CV topology has decreasing ragged offsets");
                return false;
            }
            spans[c] = uint32_t(first);
            spans[c + 1] = uint32_t(last);
        }
    } else if (R && nCvs % R != 0) {
        if (diag) diag->Error("UsdGenStraighten: upstream CV topology is non-uniform without offsets");
        return false;
    } else {
        uint32_t const perCurve = R ? nCvs / R : 0;
        for (uint32_t c = 0; c != R; ++c) spans[c + 1] = spans[c] + perCurve;
    }
    // Both straightness controls are sampled per CV (the schema declares groom
    // evaluation; UsdGenParamField::Value broadcasts that correctly, and the
    // per-CV read tolerates primitive/point granularities the same way).
    UsdGenParamField const tangent = p->GetScalarField(kTangentStraightness, 0.0);
    UsdGenParamField const normal = p->GetScalarField(kNormalStraightness, 0.0);
    for (uint32_t c = 0; c != R; ++c) {
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(tangent.Value(c, cv)) || !Finite(normal.Value(c, cv))) {
                if (diag) diag->Error("UsdGenStraighten: per-CV straightness values must be finite");
                return false;
            }
        }
    }
    return true;
}

void UsdGenStraightenOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    TF_UNUSED(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const tangentField =
        p ? p->GetScalarField(kTangentStraightness, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const normalField =
        p ? p->GetScalarField(kNormalStraightness, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *rootT = view->rootT, *rootN = view->rootN, *rootB = view->rootB;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        GfVec3f const T = rootT ? rootT[c] : GfVec3f(1.0f, 0.0f, 0.0f);
        GfVec3f const N = rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f);
        GfVec3f const B = rootB ? rootB[c] : GfVec3f(0.0f, 0.0f, 1.0f);
        float const rootx = inPx[g], rooty = inPy[g], rootz = inPz[g];
        float const chordx = inPx[g + n - 1] - rootx;
        float const chordy = inPy[g + n - 1] - rooty;
        float const chordz = inPz[g + n - 1] - rootz;
        // Pass 1: total arc length, so pass 2 can normalize to a fraction.
        float total = 0.0f;
        for (size_t i = 1; i < n; ++i) {
            float const dx = inPx[g + i] - inPx[g + i - 1];
            float const dy = inPy[g + i] - inPy[g + i - 1];
            float const dz = inPz[g + i] - inPz[g + i - 1];
            total += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        // The root is a line endpoint: copy it exactly.
        px[g] = rootx; py[g] = rooty; pz[g] = rootz;
        float s = 0.0f;
        for (size_t i = 1; i < n; ++i) {
            size_t const o = g + i;
            {
                float const dx = inPx[o] - inPx[o - 1];
                float const dy = inPy[o] - inPy[o - 1];
                float const dz = inPz[o] - inPz[o - 1];
                s += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            // The tip is the other line endpoint: copy it exactly.
            if (i + 1 == n) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            float const frac = total > 0.0f ? s / total
                : float(i) / float(n - 1);
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            float const kT =
                static_cast<float>(tangentField.Value(curve, cvBase + o)) * mask;
            float const kN =
                static_cast<float>(normalField.Value(curve, cvBase + o)) * mask;
            if (kT == 0.0f && kN == 0.0f) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            float const lx = rootx + chordx * frac;
            float const ly = rooty + chordy * frac;
            float const lz = rootz + chordz * frac;
            float const dx = inPx[o] - lx;
            float const dy = inPy[o] - ly;
            float const dz = inPz[o] - lz;
            float const dT = dx * T[0] + dy * T[1] + dz * T[2];
            float const dN = dx * N[0] + dy * N[1] + dz * N[2];
            float const dB = dx * B[0] + dy * B[1] + dz * B[2];
            float const rT = dT * (1.0f - kT);
            float const rN = dN * (1.0f - kN);
            px[o] = lx + T[0] * rT + N[0] * rN + B[0] * dB;
            py[o] = ly + T[1] * rT + N[1] * rN + B[1] * dB;
            pz[o] = lz + T[2] * rT + N[2] * rN + B[2] * dB;
        }
    }
}

UsdGenOpId UsdGenRegisterStraightenOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenStraighten";
    d.displayName = "Straighten";
    d.create = &Create;
    d.countChanges = 0;
    d.params = UsdGenSpan(kParams);
    return r.RegisterOp(d);
}

}  // namespace usdGen
