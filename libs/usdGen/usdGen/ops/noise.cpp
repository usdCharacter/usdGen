// usdGen — UsdGenNoiseOp implementation (M1). 02-schema.md §2.7.1, 04 §2.9.
// Frizz: correlated fBm displacement in the ROOT frame (I3).
//
// 04 §2.9's evaluate pseudocode re-samples the fBm every frame, but its
// sample position — rootRest[c]*correlation + h_c + (0,0, hairT[i]*frequency)
// — depends only on rest data, the per-curve capture hash, hairT and the
// capture-class parameters. Noise reads the rest plane, so
// "the field is evaluated at the rest root and frizz does not swim when the
// surface animates". We therefore pin the field values into the capture
// payload (SeExpr2::FBM, S38 — the single noise implementation) and let
// Evaluate apply only the value-class scale (magnitude * magLUT * usdGen:mask
// envelope). The per-frame cost is an add-scale, not a noise sample, which is
// what keeps gate E-1 (full 5-op run <= 1.5 ms at 100 k x 8 CV) honest: a
// per-frame SeExpr FBM over 800 k CVs measures ~100 ms single-thread /
// ~12.5 ms at 8 threads on this host (MEASURED, fbmbench, M1 integration).
#include "usdGen/ops/noise.h"
#include "usdGen/clumpMotion.h"

#include "usdGen/opParams.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"
#include "SeExpr2/Noise.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {


/// Read a float2[] ramp-knots property (02 §2.17 scalar ramp). Missing or
/// differently typed values fall back to the caller's default.
VtVec2fArray ReadRampKnots(UsdGenParamView const *p, TfToken const &name)
{
    VtVec2fArray knots;
    if (p) {
        if (VtValue const v = p->GetVtValue(name, VtValue());
            v.IsHolding<VtVec2fArray>()) {
            knots = v.UncheckedGet<VtVec2fArray>();
        }
    }
    return knots;
}

struct UsdGenNoiseCapture final : public UsdGenCapturePayload
{
    UsdGenClumpMotion clumpMotion;
    // perCv[i]      = pinned fBm field value in [-1, 1] at CV i
    // perCurve[c]   = pinned field value at the curve root (recorded for the
    //                 correlation-1 limit and diagnostics)
    uint64_t upstreamTopologyVersion = 0;
    uint32_t upstreamCvs = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenNoiseCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.totalCvs == upstreamCvs;
    }
};

inline uint64_t doubleAsBits(double d)
{
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

/// Per-curve hash vector h_c (04 §2.9): three [0,1) draws over the stable
/// id, distinct salts kSaltNoise .. kSaltNoise + 2.
void HashVec3(uint32_t seed, uint64_t curveId, float out[3])
{
    out[0] = UsdGenDraw01(int(seed), curveId, kSaltNoise + 0u);
    out[1] = UsdGenDraw01(int(seed), curveId, kSaltNoise + 1u);
    out[2] = UsdGenDraw01(int(seed), curveId, kSaltNoise + 2u);
}

}  // namespace

#ifdef USDGEN_USE_GPU_NOISE
// GPU path for Capture (CUDA1b; noise_gpu.cu). Returns false if the
// device, a transfer, or the launch fails — Capture then falls through
// to the CPU loop below.
bool UsdGenNoiseCaptureGPU(const float *base,
                           const float *cvT,
                           float frequency,
                           int octaves,
                           float lacunarity,
                           float gain,
                           int nCurves,
                           int nCV,
                           float *field,
                           float *perCurve);
#endif

static TfTokenVector _topoParams = [] {
    return UsdGenBaseTopologyParams();
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));      // toggle: value-class (02 §6.3)
    v.push_back(TfToken("noise:magnitude"));
    v.push_back(TfToken("noise:magnitude:knots"));
    v.push_back(TfToken("noise:magnitude:interpolation"));
    v.push_back(TfToken("noise:frequency"));
    v.push_back(TfToken("noise:correlation"));
    v.push_back(TfToken("noise:octaves"));
    v.push_back(TfToken("noise:lacunarity"));
    v.push_back(TfToken("noise:gain"));
    v.push_back(TfToken("cumulative"));
    v.push_back(TfToken("preserveLength"));
    v.push_back(TfToken("mask"));        // operator envelope (02 §2.13)
    return v;
}();

TfSpan<const TfToken> UsdGenNoiseOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenNoiseOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenNoiseOp::CreateCapture() const
{
    return std::make_unique<UsdGenNoiseCapture>();
}
uint32_t UsdGenNoiseOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenNoiseOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    int octaves = params.GetInt(TfToken("noise:octaves"), 1);
    float lacunarity = static_cast<float>(params.GetDouble(TfToken("noise:lacunarity"), 2.0));
    float gain = static_cast<float>(params.GetDouble(TfToken("noise:gain"), 0.5));
    if (octaves < 1 || octaves > 6) {
        if (diag) diag->Error("UsdGenNoise: usdGen:noise:octaves must be in [1, 6]");
        return false;
    }
    if (lacunarity <= 1.0f) {
        if (diag) diag->Error("UsdGenNoise: usdGen:noise:lacunarity must be > 1");
        return false;
    }
    if (gain < 0.0f || gain > 1.0f) {
        if (diag) diag->Error("UsdGenNoise: usdGen:noise:gain must be in [0, 1]");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenNoiseOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // The pinned field is a function of: frequency, correlation, octaves,
    // lacunarity, gain, seed, and the upstream
    // topology (rest roots + hairT). Value-class parameters (magnitude, its
    // knots, cumulative, preserveLength) are applied at evaluate and are
    // deliberately NOT terms here — that is what keeps a magnitude edit out
    // of the capture (gate SI-2).
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](std::string_view k, uint64_t v) {
        h ^= v; h *= 0x100000001b3ULL;
        h ^= k.size(); h *= 0x100000001b3ULL;
    };
    if (p) {
        feed("frequency", static_cast<uint32_t>(
                              p->GetDouble(TfToken("noise:frequency"), 3.0) * 10000.0));
        feed("correlation", static_cast<uint32_t>(
                                p->GetDouble(TfToken("noise:correlation"), 0.5) * 100000.0));
        feed("octaves", uint64_t(p->GetInt(TfToken("noise:octaves"), 1)));
        feed("lacunarity", doubleAsBits(
                               p->GetDouble(TfToken("noise:lacunarity"), 2.0)));
        feed("gain", doubleAsBits(p->GetDouble(TfToken("noise:gain"), 0.5)));
    }
    feed("seed", ctx.seed);
    feed("upstreamTopology", ctx.upstreamGeneration);
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenNoiseOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    auto &cap = *static_cast<UsdGenNoiseCapture *>(out);
    std::string clumpError;
    auto const *prior = dynamic_cast<UsdGenNoiseCapture const *>(ctx.previousCapture);
    auto const *previousMotion = prior ? &prior->clumpMotion :
        (cap.clumpMotion.levels.empty() ? nullptr : &cap.clumpMotion);
    if (!UsdGenBuildClumpMotion(upstream, &cap.clumpMotion, &clumpError,
                                previousMotion)) {
        if (diag) diag->Error("UsdGenNoise: " + clumpError);
        return false;
    }
    bool const clumpActive = std::any_of(cap.clumpMotion.levels.begin(),
        cap.clumpMotion.levels.end(), [](auto const &level) {
            return !level.groups.empty() &&
                std::any_of(level.weight.begin(), level.weight.end(),
                    [](float weight) { return weight > 0.0f; });
        });

    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamCvs = upstream.totalCvs;

    // The fBm field is pinned here, so these controls are capture-class. They
    // are still connectable: their evaluated values are part of this node's
    // capture identity (scheduler WithExternalValueIdentity), so editing one
    // re-captures. Per-curve controls sample the curve's ROOT CV; the
    // frequency is sampled at each CV, which is where the CUDA kernel reads it.
    UsdGenParamField const frequencyField =
        p ? p->GetScalarField(sFrequency, 3.0) : UsdGenParamField{3.0};
    UsdGenParamField const correlationField =
        p ? p->GetScalarField(sCorrelation, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const lacunarityField =
        p ? p->GetScalarField(sLacunarity, 2.0) : UsdGenParamField{2.0};
    UsdGenParamField const gainField =
        p ? p->GetScalarField(sGain, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const octavesField =
        p ? p->GetScalarField(sOctaves, double(p->GetInt(sOctaves, 1)))
          : UsdGenParamField{1.0};

    // `noise:magnitude` and its knots are value-class (02-schema.md :638-639),
    // so the magnitude ramp is NOT baked into the capture payload: Evaluate()
    // rebuilds the 257-entry ramp LUT from the live parameters with
    // UsdGenBuildRampLut(), honouring the frozen <p>:interpolation token
    // (02 :598). Baking it here would freeze a value-class edit out of the
    // frame response (review M-5); the CaptureDigest above deliberately
    // excludes the knots for the same reason (gate SI-2).

    if (upstream.px.empty()) {
        cap.perCv.clear();
        cap.perCurve.clear();
        return true;
    }

    const size_t nCv = upstream.totalCvs;
    const size_t nCurve = upstream.totalCurves;
    // Ragged layout (03 §1.3): per-curve starts come from cvOffsets (size
    // nCurve+1); the c*cvCount stride is meaningless there, so cvCount is
    // forced to 0 and every indexing site below branches on `ragged`.
    const bool ragged = !upstream.cvOffsets.empty() &&
                        size_t(upstream.cvOffsets.size()) == nCurve + 1;
    auto const *offs = ragged ? upstream.cvOffsets.cdata() : nullptr;
    const int cvCount = ragged
        ? 0 : (nCurve ? int(nCv / nCurve) : 0);   // uniform-CV fast path (M1)

    auto const *px = upstream.px.data();
    auto const *py = upstream.py.data();
    auto const *pz = upstream.pz.data();
    auto const *hairT = upstream.hairT.empty() ? nullptr : upstream.hairT.data();
    auto const *ids = upstream.curveId.empty() ? nullptr : upstream.curveId.data();

    cap.perCv.resize(nCv);
    cap.perCurve.resize(nCurve);

    #ifdef USDGEN_USE_GPU_NOISE
    // CUDA path (CUDA1b): the kernel pins the same field as the CPU
    // loop below — one thread per curve; the host precomputes the
    // rest-pinned base positions and the per-CV t-values.
    // GPU is UNIFORM-STRIDE ONLY (plan/05): the kernel walks c*cvCount+i
    // with a fixed stride, so ragged buffers take the CPU fill below.
    // Without the !ragged test a ragged buffer passed the old gate with
    // the bogus mean cvCount = nCv/nCurve — false positive, wrong field.
    if (!clumpActive && cvCount > 0 && !ragged && frequencyField.Uniform() && correlationField.Uniform() &&
        lacunarityField.Uniform() && gainField.Uniform() && octavesField.Uniform()) {
        const float frequency = static_cast<float>(frequencyField.Value(0, 0));
        const float correlation = std::clamp(
            static_cast<float>(correlationField.Value(0, 0)), 0.0f, 1.0f);
        const int octaves = static_cast<int>(octavesField.Value(0, 0));
        const float lacunarity = static_cast<float>(lacunarityField.Value(0, 0));
        const float gain = static_cast<float>(gainField.Value(0, 0));
        std::vector<float> base(nCurve * 3u);
        std::vector<float> cvT(nCv);
        for (size_t c = 0; c < nCurve; ++c) {
            float hvec[3];
            HashVec3(ctx.seed, ids ? ids[c] : 0, hvec);
            const GfVec3f root =
                GfVec3f(px[c * cvCount], py[c * cvCount], pz[c * cvCount]);
            for (int k = 0; k < 3; ++k)
                base[c * 3 + k] = root[k] * correlation + (1.0f - correlation) * hvec[k];
        }
        for (size_t i = 0; i < nCv; ++i) {
            const int j = int(i % (size_t)cvCount);
            cvT[i] = hairT
                ? hairT[i]
                : (cvCount > 1 ? float(j) / float(cvCount - 1) : 0.0f);
        }
        if (UsdGenNoiseCaptureGPU(base.data(), cvT.data(), frequency, octaves,
                                  lacunarity, gain, int(nCurve), cvCount,
                                  cap.perCv.data(), cap.perCurve.data())) {
            return true;
        }
    }
    #endif

    // Pin the fBm field: sample positions are rest-pinned (I3) —
    //   pos[c, i] = rootRest[c] * correlation + (1 - correlation) * h_c
    //               + (0, 0, hairT[i] * frequency)
    // correlation = 1 samples every curve at its own rest root (a shared
    // spatial field, hence "correlated"); correlation = 0 is fully
    // decorrelated per curve by the hash offset.
    auto *field = cap.perCv.data();
    std::vector<GfVec3f> groupBases(cap.clumpMotion.levels.size());
    std::vector<bool> groupValid(cap.clumpMotion.levels.size(), false);
    for (size_t c = 0; c < nCurve; ++c) {
        float hvec[3];
        HashVec3(ctx.seed, ids ? ids[c] : 0, hvec);
        const size_t g = ragged ? size_t(offs[c]) : c * size_t(cvCount);
        const float correlation = std::clamp(
            static_cast<float>(correlationField.Value(c, g)), 0.0f, 1.0f);
        const int octaves = static_cast<int>(octavesField.Value(c, g));
        const float lacunarity = static_cast<float>(lacunarityField.Value(c, g));
        const float gain = static_cast<float>(gainField.Value(c, g));
        const GfVec3f root = ragged
            ? GfVec3f(px[g], py[g], pz[g])
            : (cvCount > 0 ? GfVec3f(px[g], py[g], pz[g])
                           : GfVec3f(0.0f, 0.0f, 0.0f));
        const float baseX = root[0] * correlation + (1.0f - correlation) * hvec[0];
        const float baseY = root[1] * correlation + (1.0f - correlation) * hvec[1];
        const float baseZ = root[2] * correlation + (1.0f - correlation) * hvec[2];
        for (size_t li = 0; li < cap.clumpMotion.levels.size(); ++li) {
            auto const &level = cap.clumpMotion.levels[li];
            groupValid[li] = !level.weight.empty() && c < level.groupForCurve.size() &&
                level.groupForCurve[c] != UINT32_MAX;
            if (!groupValid[li]) continue;
            auto const &group = level.groups[level.groupForCurve[c]];
            float hash[3];
            HashVec3(ctx.seed, group.centerId, hash);
            groupBases[li] = group.restAnchor * correlation +
                GfVec3f(hash[0], hash[1], hash[2]) * (1.0f - correlation);
        }

        float rootOut = 0.0f;
        {
            const float in3[3] = {baseX, baseY, baseZ};
            SeExpr2::FBM<3, 1, false, float>(in3, &rootOut,
                                             octaves, lacunarity, gain);
        }
        cap.perCurve[c] = 2.0f * rootOut - 1.0f;

        const int n = ragged ? int(offs[c + 1] - offs[c]) : cvCount;
        // cap.perCv stays GLOBAL-indexed (length totalCvs, plan/05:497):
        // curve c owns field [g, g+n).
        for (int i = 0; i < n; ++i) {
            const float t = hairT
                ? hairT[g + i]
                : (n > 1 ? float(i) / float(n - 1) : 0.0f);
            const float captureFrequency =
                static_cast<float>(frequencyField.Value(c, g + size_t(i)));
            const float in3[3] = {baseX, baseY, baseZ + t * captureFrequency};
            float out1 = 0.0f;
            SeExpr2::FBM<3, 1, false, float>(in3, &out1,
                                             octaves, lacunarity, gain);
            float const individual = 2.0f * out1 - 1.0f;
            float shared = individual;
            float remaining = 1.0f;
            // The group re-samples the same field at a stable rest anchor,
            // with the center's hash offset. Blend FIELD OUTPUTS rather than
            // sample coordinates: local detail and authored frequency remain.
            for (size_t li = cap.clumpMotion.levels.size();
                 li-- > 0 && remaining > 0.0f;) {
                auto const &level = cap.clumpMotion.levels[li];
                if (!groupValid[li]) continue;
                float const portion = remaining * level.weight[g + i];
                if (portion == 0.0f) continue;
                auto const &base = groupBases[li];
                float const groupIn[3] = {
                    base[0], base[1], base[2] + t * captureFrequency};
                float groupOut = 0.0f;
                SeExpr2::FBM<3, 1, false, float>(groupIn, &groupOut,
                                                 octaves, lacunarity, gain);
                shared += portion * ((2.0f * groupOut - 1.0f) - individual);
                remaining -= portion;
            }
            field[g + i] = shared;
        }
    }
    return true;
}

void UsdGenNoiseOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    UsdGenNoiseCapture const &cap =
        static_cast<UsdGenNoiseCapture const &>(captureIn);
    if (cap.perCv.empty()) return;

    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    // usdGen:noise:magnitude, usdGen:preserveLength, usdGen:cumulative and
    // usdGen:mask are CONNECTABLE value-class controls; an authored literal is
    // a uniform field and the arithmetic below is unchanged.
    UsdGenParamField const magnitudeField =
        p ? p->GetScalarField(sMagnitude, 0.05) : UsdGenParamField{0.05};
    UsdGenParamField const preserveField =
        p ? p->GetScalarField(sPreserveLength, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const cumulativeField =
        p ? p->GetScalarField(sCumulative, p->GetBool(sCumulative, false) ? 1.0 : 0.0)
          : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *hairT = view->hairT;
    auto const *rootN = view->rootN;
    const size_t curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    auto const *perCv = cap.perCv.data();
    // Worker-local storage avoids per-chunk allocations. Empty knots use
    // scalar 1.0 below; there is no shared table with static exit lifetime.
    float const *lutPtr = nullptr;
    VtVec2fArray const magKnots = ReadRampKnots(p, sMagKnots);
    bool const magFlat = magKnots.empty();
    thread_local std::vector<float> tLut;
    if (!magFlat) {
        if (tLut.size() != kUsdGenRampLutSize) tLut.assign(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(magKnots, p ? p->GetToken(sMagInterp, sCatmullRom) : sCatmullRom, tLut.data(), kUsdGenRampLutSize);
        lutPtr = tLut.data();
    }
    const size_t cv = size_t(view->cvCount);
    // Ragged chunk (03 §1.3): cvCount == 0 and view->cvOffsets is ALREADY
    // shifted by desc->firstCurve (scheduler SweepChunk, scheduler.cpp:213),
    // so chunk-local curve c spans [cvOffsets[c] - cvOffsets[0], + n) of the
    // chunk-local planes.
    auto const *cvOff = view->cvOffsets;
    const bool ragged = cvOff != nullptr;   // null == uniform fast path
    // cap.perCv is GLOBAL per-CV (length totalCvs): a chunk-local plane
    // index o addresses the field at firstCv + o (plan/05:497).
    const size_t firstCv = view->desc ? size_t(view->desc->firstCv) : 0;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        const size_t nCVs = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : cv;
        const size_t g = ragged ? size_t(cvOff[c] - cvOff[0])
                                : view->Cv(c, 0);
        const size_t curve = curveBase + c;
        const bool cumulative = cumulativeField.Value(curve, firstCv + g) != 0.0;
        const GfVec3f n = rootN ? rootN[c] : GfVec3f(0.0f, 1.0f, 0.0f);
        float runSum = 0.0f;
        for (size_t i = 0; i < nCVs; ++i) {
            const size_t o = g + i;
            const float magnitude =
                static_cast<float>(magnitudeField.Value(curve, firstCv + o));
            const float mask = std::clamp(
                static_cast<float>(maskField.Value(curve, firstCv + o)), 0.0f, 1.0f);
            const float w = magnitude * mask;
            if (w == 0.0f) {
                // Early-out: bitwise copy of the input (02 §2.13).
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            const float preserveLength =
                static_cast<float>(preserveField.Value(curve, firstCv + o));
            const float t = hairT ? hairT[o]
                                   : (nCVs > 1 ? float(i) / float(nCVs - 1) : 0.0f);
            // Flat magnitude ramp (no knots) evaluates to exactly 1.0, so
            // skip the out-of-line LUT call: bitwise-identical, fewer CV costs.
            const float magScale = magFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t);
            float disp = w * magScale * perCv[firstCv + o];
            if (cumulative) {
                runSum += disp;
                disp = runSum / float(i + 1);
            }
            const float dx = inPx[o] + n[0] * disp;
            const float dy = inPy[o] + n[1] * disp;
            const float dz = inPz[o] + n[2] * disp;
            if (preserveLength >= 0.999f && i > 0) {
                // Restore the rest segment length: rescale the new segment to
                // the magnitude of the input segment.
                const float prevX = inPx[o - 1], prevY = inPy[o - 1],
                            prevZ = inPz[o - 1];
                const float sx = inPx[o] - prevX, sy = inPy[o] - prevY,
                            sz = inPz[o] - prevZ;
                const float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
                const float qx = dx - prevX, qy = dy - prevY, qz = dz - prevZ;
                const float ql = std::sqrt(qx * qx + qy * qy + qz * qz);
                if (sl > 1e-9f && ql > 1e-9f) {
                    const float f = sl / ql;
                    px[o] = prevX + qx * f;
                    py[o] = prevY + qy * f;
                    pz[o] = prevZ + qz * f;
                    continue;
                }
            }
            px[o] = dx; py[o] = dy; pz[o] = dz;
        }
    }
}

}  // namespace usdGen
