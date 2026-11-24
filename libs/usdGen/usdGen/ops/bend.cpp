// usdGen — UsdGenBendOp implementation. 02-schema.md §2.7, 04 §3.
// Cumulative per-segment rotation about a per-curve axis: with the ramp R,
// segment k (CV k -> k+1) adds
//   delta_k = angle_k * mult_c * (R(t_k) - R(t_{k-1})) * mask_k   [degrees]
// to the cumulative bend, and CV k+1 is placed by rotating the input segment
// vector by the cumulative angle. A flat ramp bends rigidly about the root;
// a 0->1 ramp distributes the bend along the strand. Segment lengths are
// preserved by construction. The axis is usdGen:axis in description space
// (axisMode = uniform), in the root (T, N, B) frame (axisMode =
// rootDirection), or the strand's root tangent (axisMode = attribute). A
// zero-length axis bends nothing. All work is per curve/CV in Evaluate;
// Capture only validates.
#include "usdGen/ops/bend.h"
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

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

struct UsdGenBendCapture final : public UsdGenCapturePayload
{
    UsdGenClumpMotion clumpMotion;
    std::vector<std::vector<float>> groupDraws;
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenBendCapture>(*this);
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

inline void ReadVec2Prop(UsdGenParamView const &p, TfToken const &name,
                         double &lo, double &hi)
{
    lo = 1.0; hi = 1.0;
    if (VtValue const v = p.GetVtValue(name, VtValue()); v.IsHolding<VtFloatArray>()) {
        VtFloatArray const a = v.UncheckedGet<VtFloatArray>();
        if (a.size() >= 2) { lo = a[0]; hi = a[1]; }
    } else if (v.IsHolding<GfVec2f>()) {
        GfVec2f const r = v.UncheckedGet<GfVec2f>();
        lo = r[0]; hi = r[1];
    }
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("axisMode"));   // structural: selects the axis frame (02 §2.7.2)
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("angle"));
    v.push_back(TfToken("angle:knots"));
    v.push_back(TfToken("angle:interpolation"));
    v.push_back(TfToken("angleRandom"));
    v.push_back(TfToken("axis"));
    v.push_back(TfToken("mask"));
    return v;
}();

TfSpan<const TfToken> UsdGenBendOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenBendOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenBendOp::CreateCapture() const
{
    return std::make_unique<UsdGenBendCapture>();
}
uint32_t UsdGenBendOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenBendOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sAxisMode, sRootDirection);
    if (mode != sRootDirection && mode != sUniform && mode != sAttribute) {
        if (diag) diag->Error("UsdGenBend: unknown usdGen:axisMode '" + mode.GetString() + "'");
        return false;
    }
    if (!Finite(params.GetDoubleLiteral(sAngle, 0.0))) {
        if (diag) diag->Error("UsdGenBend: usdGen:angle must be finite");
        return false;
    }
    double lo = 1.0, hi = 1.0;
    ReadVec2Prop(params, sAngleRandom, lo, hi);
    if (!Finite(lo) || !Finite(hi)) {
        if (diag) diag->Error("UsdGenBend: usdGen:angleRandom must be finite");
        return false;
    }
    VtValue const axis = params.GetVtValue(sAxis, VtValue());
    if (!axis.IsEmpty() && !axis.IsHolding<GfVec3f>()) {
        if (diag) diag->Error("UsdGenBend: usdGen:axis must be vector3f");
        return false;
    }
    if (axis.IsHolding<GfVec3f>()) {
        GfVec3f const v = axis.UncheckedGet<GfVec3f>();
        if (!Finite(v[0]) || !Finite(v[1]) || !Finite(v[2])) {
            if (diag) diag->Error("UsdGenBend: usdGen:axis must be finite");
            return false;
        }
    }
    return true;
}

UsdGenEpoch UsdGenBendOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    feed(p ? uint64_t(p->GetToken(sAxisMode, sRootDirection).Hash()) : 0u);
    feed(ctx.upstreamGeneration);
    feed(uint64_t(ctx.seed));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenBendOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    auto &cap = *static_cast<UsdGenBendCapture *>(out);
    std::string clumpError;
    auto const *prior = dynamic_cast<UsdGenBendCapture const *>(ctx.previousCapture);
    auto const *previousMotion = prior ? &prior->clumpMotion :
        (cap.clumpMotion.levels.empty() ? nullptr : &cap.clumpMotion);
    if (!UsdGenBuildClumpMotion(upstream, &cap.clumpMotion, &clumpError,
                                previousMotion)) {
        if (diag) diag->Error("UsdGenBend: " + clumpError);
        return false;
    }
    cap.groupDraws.clear();
    for (auto const &level : cap.clumpMotion.levels) {
        auto &draws = cap.groupDraws.emplace_back();
        draws.reserve(level.groups.size());
        for (auto const &group : level.groups)
            draws.push_back(UsdGenDraw01(int(ctx.seed), group.centerId, kSaltBend));
    }
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    if (!Bind(*p, diag)) return false;
    uint32_t const R = upstream.totalCurves;
    uint32_t const nCvs = upstream.totalCvs;
    std::vector<uint32_t> spans(size_t(R) + 1, 0);
    if (!upstream.cvOffsets.empty()) {
        if (upstream.cvOffsets.size() != size_t(R) + 1) {
            if (diag) diag->Error("UsdGenBend: upstream CV topology has invalid ragged offsets");
            return false;
        }
        for (uint32_t c = 0; c != R; ++c) {
            int const first = upstream.cvOffsets[c], last = upstream.cvOffsets[c + 1];
            if (first < 0 || last < first) {
                if (diag) diag->Error("UsdGenBend: upstream CV topology has decreasing ragged offsets");
                return false;
            }
            spans[c] = uint32_t(first);
            spans[c + 1] = uint32_t(last);
        }
    } else if (R && nCvs % R != 0) {
        if (diag) diag->Error("UsdGenBend: upstream CV topology is non-uniform without offsets");
        return false;
    } else {
        uint32_t const perCurve = R ? nCvs / R : 0;
        for (uint32_t c = 0; c != R; ++c) spans[c + 1] = spans[c] + perCurve;
    }
    UsdGenParamField const angle = p->GetScalarField(sAngle, 0.0);
    UsdGenParamField const random = p->GetScalarField(sAngleRandom, 1.0);
    UsdGenParamField const axis = p->GetScalarField(sAxis, 0.0);
    if ((random.connected && random.components != 2) ||
        (axis.connected && axis.components != 3)) {
        if (diag) diag->Error("UsdGenBend: connected angleRandom must be float2 and "
                              "axis must be vector3f");
        return false;
    }
    for (uint32_t c = 0; c != R; ++c) {
        uint32_t const root = spans[c];
        if (random.connected &&
            (!Finite(random.Value(c, root, 0)) || !Finite(random.Value(c, root, 1)))) {
            if (diag) diag->Error("UsdGenBend: per-strand angleRandom must be finite");
            return false;
        }
        if (axis.connected &&
            (!Finite(axis.Value(c, root, 0)) || !Finite(axis.Value(c, root, 1)) ||
             !Finite(axis.Value(c, root, 2)))) {
            if (diag) diag->Error("UsdGenBend: per-strand axis must be finite");
            return false;
        }
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(angle.Value(c, cv))) {
                if (diag) diag->Error("UsdGenBend: per-CV angle must be finite");
                return false;
            }
        }
    }
    return true;
}

void UsdGenBendOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenBendCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const angleField =
        p ? p->GetScalarField(sAngle, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const randomField =
        p ? p->GetScalarField(sAngleRandom, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const axisField =
        p ? p->GetScalarField(sAxis, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    double litLo = 1.0, litHi = 1.0;
    GfVec3f litAxis(1.0f, 0.0f, 0.0f);
    TfToken mode = sRootDirection;
    if (p) {
        ReadVec2Prop(*p, sAngleRandom, litLo, litHi);
        if (litLo > litHi) std::swap(litLo, litHi);
        VtValue const v = p->GetVtValue(sAxis, VtValue());
        if (v.IsHolding<GfVec3f>()) litAxis = v.UncheckedGet<GfVec3f>();
        mode = p->GetToken(sAxisMode, sRootDirection);
    }
    bool const uniformMode = mode == sUniform;
    bool const attributeMode = mode == sAttribute;
    float const *lutPtr = nullptr;
    VtVec2fArray const knots = ReadRampKnots(p, sAngleKnots);
    bool const rampFlat = knots.empty();
    UsdGenScratchScope scratch;   // chunk-local; rewinds when Evaluate returns
    if (!rampFlat) {
        float *lut = scratch.Filled<float>(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(knots,
            p ? p->GetToken(sAngleInterp, sCatmullRom) : sCatmullRom,
            lut, kUsdGenRampLutSize);
        lutPtr = lut;
    }
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *hairT = view->hairT;
    auto const *rootT = view->rootT, *rootN = view->rootN, *rootB = view->rootB;
    auto const *curveId = view->curveId;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    constexpr float degToRad = 3.14159265358979323846f / 180.0f;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        // Per-curve random multiplier over the angleRandom range.
        double lo = litLo, hi = litHi;
        if (randomField.connected) {
            lo = randomField.Value(curve, cvBase + g, 0);
            hi = randomField.Value(curve, cvBase + g, 1);
            if (lo > hi) std::swap(lo, hi);
        }
        uint64_t const id = curveId ? curveId[c] : 0;
        float const individualDraw = UsdGenDraw01(int(ctx.seed), id, kSaltBend);
        // Per-curve bend axis. A zero-length axis bends nothing.
        GfVec3f axis(0.0f);
        if (attributeMode) {
            axis = rootT ? rootT[c] : GfVec3f(1.0f, 0.0f, 0.0f);
        } else {
            GfVec3f const a = axisField.connected
                ? GfVec3f(float(axisField.Value(curve, cvBase + g, 0)),
                          float(axisField.Value(curve, cvBase + g, 1)),
                          float(axisField.Value(curve, cvBase + g, 2)))
                : litAxis;
            if (uniformMode) {
                axis = a;
            } else {  // rootDirection: axis components in the root frame.
                GfVec3f const T = rootT ? rootT[c] : GfVec3f(1.0f, 0.0f, 0.0f);
                GfVec3f const N = rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f);
                GfVec3f const B = rootB ? rootB[c] : GfVec3f(0.0f, 0.0f, 1.0f);
                axis = T * a[0] + N * a[1] + B * a[2];
            }
        }
        float const axisLen = axis.GetLength();
        // Walk the segments, accumulating the bend and placing each CV by
        // rotating its input segment vector. Roots never move.
        px[g] = inPx[g]; py[g] = inPy[g]; pz[g] = inPz[g];
        if (axisLen <= 1e-12f || n < 2) {
            for (size_t i = 1; i < n; ++i) {
                px[g + i] = inPx[g + i]; py[g + i] = inPy[g + i]; pz[g + i] = inPz[g + i];
            }
            continue;
        }
        axis /= axisLen;
        auto rampAt = [&](size_t i) {
            float const t = hairT ? hairT[g + i]
                : (n > 1 ? float(i) / float(n - 1) : 0.0f);
            return rampFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t);
        };
        float cumul = 0.0f;   // cumulative bend in radians
        float prevR = 0.0f;   // R(t_{-1}) := 0, so a flat ramp bends at the root
        for (size_t k = 0; k + 1 < n; ++k) {
            float draw = individualDraw;
            float remaining = 1.0f;
            for (size_t li = cap.clumpMotion.levels.size(); li-- > 0 && remaining > 0.0f;) {
                auto const &level = cap.clumpMotion.levels[li];
                if (level.weight.empty() || curve >= level.groupForCurve.size()) continue;
                uint32_t const group = level.groupForCurve[curve];
                if (group == UINT32_MAX || group >= level.groups.size()) continue;
                // Segment k controls CV k+1; sample cohesion at that CV so
                // a root-locked clump can still move its first free segment.
                float const portion = remaining * level.weight[cvBase + g + k + 1];
                draw += portion * (cap.groupDraws[li][group] - individualDraw);
                remaining -= portion;
            }
            float const mult = float(lo + (hi - lo) * double(draw));
            float const angleK =
                static_cast<float>(angleField.Value(curve, cvBase + g + k));
            float const maskK = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + g + k)), 0.0f, 1.0f);
            float const rK = rampAt(k);
            cumul += angleK * mult * (rK - prevR) * maskK * degToRad;
            prevR = rK;
            float const co = std::cos(cumul), si = std::sin(cumul);
            float const vx = inPx[g + k + 1] - inPx[g + k];
            float const vy = inPy[g + k + 1] - inPy[g + k];
            float const vz = inPz[g + k + 1] - inPz[g + k];
            // Rodrigues' rotation of the input segment vector.
            float const dot = axis[0] * vx + axis[1] * vy + axis[2] * vz;
            float const cx = axis[1] * vz - axis[2] * vy;
            float const cy = axis[2] * vx - axis[0] * vz;
            float const cz = axis[0] * vy - axis[1] * vx;
            float const omc = 1.0f - co;
            size_t const o = g + k, q = g + k + 1;
            px[q] = px[o] + vx * co + cx * si + axis[0] * dot * omc;
            py[q] = py[o] + vy * co + cy * si + axis[1] * dot * omc;
            pz[q] = pz[o] + vz * co + cz * si + axis[2] * dot * omc;
        }
    }
}

}  // namespace usdGen
