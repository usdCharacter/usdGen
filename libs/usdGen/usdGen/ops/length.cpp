// usdGen — UsdGenLengthOp implementation (M1). 02-schema.md §2.7.1, 04 §2.10.
// Set/scale/cull strand length. The WHOLE type is topology-bumping
// (TopologyEffect() == CurveCount, ADR §9 R14) because cull mode can change
// the curve count; with cullThreshold == 0 (the default) it behaves as a
// value-class re-position.
//
// M1 kernel: a per-curve scale FACTOR captured against the upstream rest
// length at capture time; Evaluate scales the current (deformed) curve about
// its root by that factor. "set" and "scale" both reduce to a factor
// (set: target/restLen, scale: value*lerp(lo,hi,draw) — plan/04 §2.10's
// `length:random` float2, drawn per curve with kSaltLength (plan/04 §0.6)).
// "cutExtend" is the same radial re-position on M1's straight chains; culling
// zero-lengthens a curve (stable ids; a true curve removal is M2's topology
// work, and R14 already routes usdGen:cullThreshold as topology).
//
// "randomRange" (imported grooms, CPU lane only) is a whole-strand capture:
// Capture runs ops/lengthRange.h over every strand (clump means, root-sampled
// usdGen:lengthMap masks, arc-length resampling) and keeps the result plus
// the per-strand reference length; Evaluate lerps input -> result by the
// usdGen:mask envelope and writes the `restLength` plane. Its capture
// tracks upstream VALUES (positions), unlike the factor modes.
#include "usdGen/ops/length.h"


#include "usdGen/opParams.h"
#include "usdGen/ops/lengthRange.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/ops/wholeStrand.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

inline uint64_t doubleAsBits(double d)
{
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

// Reads a `float2` property — GfVec2f or 2-element VtFloatArray; lo/hi swapped
// when hi < lo. Returns false when absent/unparseable. (`length:random`,
// 02 §2.6 float2 — plan/04:1033.)
bool ReadVec2Prop(UsdGenParamView const *p, char const *name, double* lo, double* hi)
{
    if (!p)
        return false;
    VtValue v = p->GetVtValue(TfToken(name), VtValue());
    float x = 0.0f, y = 0.0f;
    if (v.IsHolding<GfVec2f>()) {
        const GfVec2f vec = v.UncheckedGet<GfVec2f>();
        x = vec[0]; y = vec[1];
    } else if (v.IsHolding<VtFloatArray>()) {
        const VtFloatArray& arr = v.UncheckedGet<VtFloatArray>();
        if (arr.size() != 2)
            return false;
        x = arr[0]; y = arr[1];
    } else {
        return false;
    }
    *lo = x;
    *hi = y;
    if (*hi < *lo)
        std::swap(*lo, *hi);
    return true;
}

struct UsdGenLengthCapture final : public UsdGenCapturePayload
{
    // perCurve[c] = lerp(length:random, draw01(seed, curveId, kSaltLength)) —
    // the position-independent part of §2.10's target formula. Rest length is
    // NOT sampled here: capture sees lazily-evaluated upstream buffers whose
    // chunk cvCount is not yet populated, so any restLen/target/restLen ratio
    // must be computed at Evaluate time from the chunk's real input planes.
    TfToken mode = TfToken("scale");
    double value = 1.0;
    double cullThreshold = 0.0;
    double minRemaining = 0.0;
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;
    // randomRange only: the whole-strand result (3 * totalCvs) and the
    // per-strand reference (pre-scale) length.
    bool range = false;
    bool rangeReady = false;
    std::vector<float> rangeResult;
    std::vector<float> restLength;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenLengthCapture>(*this);
    }
    bool OwnsBuffer() const override { return false; }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        if (range && (upstream.valueVersion != upstreamValueVersion ||
                      upstream.totalCvs != upstreamCvs))
            return false;   // randomRange resamples the upstream POSITIONS
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.totalCurves == upstreamCurves;
    }
};

const TfToken kRandomRange{"randomRange"};
const TfToken kLengthMap{"usdGen:lengthMap"};
const TfToken kRestLength{"restLength"};

bool FiniteParam(UsdGenParamView const &p, char const *name, double fallback)
{
    return std::isfinite(p.GetDouble(TfToken(name), fallback));
}


}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("enabled"));      // §6.2: Length is unconditionally topology
    v.push_back(TfToken("length:mode"));
    v.push_back(TfToken("length:method"));
    v.push_back(TfToken("rebuild"));
    v.push_back(TfToken("length:value"));
    v.push_back(TfToken("length:random"));
    v.push_back(TfToken("minRemainingLength"));
    v.push_back(TfToken("cullThreshold"));
    // randomRange (capture-class): the whole-strand result is recaptured.
    v.push_back(TfToken("length:min"));
    v.push_back(TfToken("length:max"));
    v.push_back(TfToken("length:randomize"));
    v.push_back(TfToken("length:shareByClump"));
    v.push_back(TfToken("length:followClumpMean"));
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("mask"));        // operator envelope (02 §2.13)
    return v;
}();

TfSpan<const TfToken> UsdGenLengthOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenLengthOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenLengthOp::CreateCapture() const
{
    return std::make_unique<UsdGenLengthCapture>();
}
uint32_t UsdGenLengthOp::PlanesTouched() const
{
    return kPlanePoints;
}
TfSpan<const TfToken> UsdGenLengthOp::OutputPrimvars() const
{
    return TfSpan<const TfToken>(_outputs.data(), _outputs.size());
}
void UsdGenLengthOp::Configure(UsdGenParamView const &params)
{
    _outputs.clear();
    if (params.GetToken(TfToken("length:mode"), TfToken("scale")) == kRandomRange)
        _outputs.push_back(kRestLength);
}

bool UsdGenLengthOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken mode = params.GetToken(TfToken("length:mode"), TfToken("scale"));
    TfToken method = params.GetToken(TfToken("length:method"), TfToken("scale"));
    if (mode != TfToken("set") && mode != TfToken("scale") && mode != TfToken("cull") &&
        mode != kRandomRange) {
        if (diag)
            diag->Error("UsdGenLength: unknown usdGen:length:mode '" + mode.GetString() + "'");
        return false;
    }
    if (method != TfToken("scale") && method != TfToken("cutExtend")) {
        if (diag)
            diag->Error("UsdGenLength: unknown usdGen:length:method '" + method.GetString() + "'");
        return false;
    }
    if (!FiniteParam(params, "length:min", 0.01) || !FiniteParam(params, "length:max", 1.0) ||
        !FiniteParam(params, "length:randomize", 0.0)) {
        if (diag)
            diag->Error("UsdGenLength: usdGen:length:min/max/randomize must be finite");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenLengthOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](std::string const &k, uint64_t v) {
        h ^= v; h *= 0x100000001b3ULL;
        h ^= k.size(); h *= 0x100000001b3ULL;
    };
    if (p) {
        feed("mode", p->GetToken(TfToken("length:mode"), TfToken("scale")).Hash());
        feed("method", p->GetToken(TfToken("length:method"), TfToken("scale")).Hash());
        feed("value", doubleAsBits(p->GetDouble(TfToken("length:value"), 1.0)));
        feed("cull", doubleAsBits(p->GetDouble(TfToken("cullThreshold"), 0.0)));
        feed("minRemaining", doubleAsBits(p->GetDouble(TfToken("minRemainingLength"), 0.0)));
        double lo = 1.0, hi = 1.0;
        ReadVec2Prop(p, "length:random", &lo, &hi);   // M-5: the range is a capture input
        feed("lenRandomLo", doubleAsBits(lo));
        feed("lenRandomHi", doubleAsBits(hi));
        feed("seed", ctx.seed);
        feed("rangeMin", doubleAsBits(p->GetDouble(TfToken("length:min"), 0.01)));
        feed("rangeMax", doubleAsBits(p->GetDouble(TfToken("length:max"), 1.0)));
        feed("rangeRandomize", doubleAsBits(p->GetDouble(TfToken("length:randomize"), 0.0)));
        feed("rangeShare", p->GetBool(TfToken("length:shareByClump"), false) ? 1u : 0u);
        feed("rangeFollow", p->GetBool(TfToken("length:followClumpMean"), false) ? 1u : 0u);
    }
    // A bound length map is sampled on the surface: its geometry is a
    // capture input (the map identity itself is mixed by the scheduler).
    if (ctx.desc && ctx.surface < ctx.desc->surfaces.size() &&
        !strand::BoundMaps(ctx.desc, ctx.mapBindings, ctx.mapBindingCount, kLengthMap).empty())
        feed("surface", ctx.desc->surfaces[ctx.surface].surfaceGeneration);
    feed("upstream", ctx.upstreamGeneration);
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

namespace {

// randomRange: the whole-strand length stage over every upstream strand.
// The upstream node has completed its sweep (dependency frontier), so its
// point planes hold this run's positions.
bool CaptureRandomRange(UsdGenCaptureContext const &ctx,
                        UsdGenCurveBuffer const &upstream,
                        UsdGenLengthCapture *cap, UsdGenDiagnostics *diag)
{
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenLength: " + message);
        return false;
    };
    UsdGenParamView const *p = ctx.params;
    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(upstream, &spans, &error))
        return fail("strand topology: " + error);
    size_t const R = upstream.totalCurves, totalCvs = upstream.totalCvs;
    if (upstream.px.size() != totalCvs || upstream.py.size() != totalCvs ||
        upstream.pz.size() != totalCvs)
        return fail("input point planes do not match the CV count");

    lengthRange::Settings settings;
    if (p) {
        settings.min = float(p->GetDouble(TfToken("length:min"), 0.01));
        settings.max = float(p->GetDouble(TfToken("length:max"), 1.0));
        settings.randomize = float(p->GetDouble(TfToken("length:randomize"), 0.0));
        settings.shareByClump = p->GetBool(TfToken("length:shareByClump"), false);
        settings.followClumpMean = p->GetBool(TfToken("length:followClumpMean"), false);
    }
    if (!std::isfinite(settings.min) || !std::isfinite(settings.max) ||
        !std::isfinite(settings.randomize))
        return fail("usdGen:length:min/max/randomize must be finite");

    // Strand identity: clumpId_0, strandSeed and strandIndex planes when the
    // upstream carries them (UsdGenStrandSet fallbacks otherwise).
    UsdGenStrandSet ids;
    ids.upstream = &upstream;
    lengthRange::Strands strands;
    strands.spans = &spans;
    if (int const *clump = ids.CurveInt(TfToken("clumpId_0")))
        strands.clumpId.assign(clump, clump + R);
    strands.seed.resize(R);
    strands.order.resize(R);
    std::vector<uint32_t> index(R);
    for (uint32_t c = 0; c < R; ++c) {
        strands.seed[c] = ids.Seed(c);
        index[c] = ids.Index(c);
        strands.order[c] = c;
    }
    std::stable_sort(strands.order.begin(), strands.order.end(),
                     [&](uint32_t a, uint32_t b) { return index[a] < index[b]; });

    // usdGen:lengthMap: the product of every bound mask, sampled at the root
    // (closest point on the rest surface to the strand's rest root).
    std::vector<UsdGenMapDesc const *> const maps =
        strand::BoundMaps(ctx.desc, ctx.mapBindings, ctx.mapBindingCount, kLengthMap);
    if (!maps.empty()) {
        strands.mask.assign(R, 1.0f);
        strand::TriangleBvh bvh;
        if (ctx.desc && ctx.surface < ctx.desc->surfaces.size())
            strand::BuildSurfaceBvh(ctx.desc->surfaces[ctx.surface], true, &bvh);
        if (bvh.Empty()) {
            if (diag) diag->Warn("UsdGenLength: usdGen:lengthMap needs a bound surface; "
                                 "the length mask is ignored");
        } else {
            for (size_t c = 0; c < R; ++c) {
                if (spans[c + 1] == spans[c]) continue;
                strand::ClosestHit const hit = bvh.Closest(opUtil::RestPoint(upstream, spans[c]));
                if (!hit.hit) continue;
                float m = 1.0f;
                for (UsdGenMapDesc const *map : maps)
                    m *= strand::SampleMaskMap(*map, bvh, hit.triangle, hit.bary, 1.0f);
                strands.mask[c] = strand::Clamp01(m);
            }
        }
    }

    cap->rangeResult.resize(3 * totalCvs);
    for (size_t i = 0; i < totalCvs; ++i) {
        cap->rangeResult[3 * i] = upstream.px[i];
        cap->rangeResult[3 * i + 1] = upstream.py[i];
        cap->rangeResult[3 * i + 2] = upstream.pz[i];
    }
    lengthRange::Apply(settings, strands, &cap->rangeResult, &cap->restLength, ctx.dispatcher);
    for (float v : cap->rangeResult)
        if (!std::isfinite(v)) return fail("randomRange computed a non-finite point");
    cap->rangeReady = true;
    return true;
}

}  // namespace

bool UsdGenLengthOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    auto &cap = *static_cast<UsdGenLengthCapture *>(out);
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.mode = p ? p->GetToken(TfToken("length:mode"), TfToken("scale")) : TfToken("scale");
    double value = p ? p->GetDouble(TfToken("length:value"), 1.0) : 1.0;
    double lo = 1.0, hi = 1.0;
    ReadVec2Prop(p, "length:random", &lo, &hi);
    double cullThreshold = p ? p->GetDouble(TfToken("cullThreshold"), 0.0) : 0.0;
    double minRemaining = p ? p->GetDouble(TfToken("minRemainingLength"), 0.0) : 0.0;

    cap.value = value;
    cap.cullThreshold = cullThreshold;
    cap.minRemaining = minRemaining;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCvs = upstream.totalCvs;
    cap.range = cap.mode == kRandomRange;
    cap.rangeReady = false;
    cap.rangeResult.clear();
    cap.restLength.clear();
    if (cap.range) {
        cap.perCurve.clear();
        return CaptureRandomRange(ctx, upstream, &cap, diag);
    }

    // Capture draws only the seed-sensitive multiplier; lengths are a runtime
    // quantity (see UsdGenLengthCapture comment).
    const uint32_t R = upstream.totalCurves;
    cap.perCurve.resize(R);
    auto *mult = cap.perCurve.data();
    auto const *ids = upstream.curveId.empty() ? nullptr : upstream.curveId.data();
    for (uint32_t c = 0; c < R; ++c) {
        // lerp(lo, hi, draw01) — plan/04 §2.10 (`length:random` per-curve
        // multiplier), draw per plan/04 §0.6.
        const float r = UsdGenDraw01(int(ctx.seed), uint64_t(ids ? ids[c] : 0), kSaltLength);
        mult[c] = float(lo + (hi - lo) * double(r));
    }
    return true;
}

void UsdGenLengthOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const

{
    UsdGenLengthCapture const &cap =
        static_cast<UsdGenLengthCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    // usdGen:length:value, usdGen:cullThreshold, usdGen:minRemainingLength and
    // usdGen:mask are CONNECTABLE. The first three shape ONE per-curve scale
    // factor, so they are sampled at the curve's root CV; the mask is a true
    // per-CV envelope. Nothing connected leaves these as uniform literals and
    // the arithmetic below bit-identical to the pre-expression kernel.
    UsdGenParamField const valueField =
        p ? p->GetScalarField(sValue, cap.value) : UsdGenParamField{cap.value};
    UsdGenParamField const cullField =
        p ? p->GetScalarField(sCull, cap.cullThreshold) : UsdGenParamField{cap.cullThreshold};
    UsdGenParamField const minField =
        p ? p->GetScalarField(sMinRemaining, cap.minRemaining)
          : UsdGenParamField{cap.minRemaining};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    // perCurve is a whole-buffer capture payload (the scheduler pre-offsets
    // only buffer planes) — index it absolutely, grow.cpp pattern.
    auto const *mult = cap.perCurve.empty() ? nullptr : cap.perCurve.cdata();
    const size_t curveBase = view->desc ? view->desc->firstCurve : 0;
    const size_t cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    // sSet is operator-owned: no per-curve token interning and no lazy
    // static destruction while an asynchronous evaluation is still running.
    // restLength slot: bound only on a node configured for randomRange.
    float *restOut = (view->outCount > 0 && view->outF) ? view->outF[0] : nullptr;

    if (cap.range) {
        // randomRange: lerp input -> captured whole-strand result by the
        // usdGen:mask envelope (mask == 0 is a bitwise pass-through).
        bool const ready = cap.rangeReady &&
            cap.rangeResult.size() == size_t(cap.upstreamCvs) * 3;
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            const size_t cv = view->cvCount ? size_t(view->cvCount)
                : (view->cvOffsets ? size_t(view->cvOffsets[c + 1] - view->cvOffsets[c]) : 0);
            const size_t base = view->cvCount ? size_t(c) * view->cvCount
                : (view->cvOffsets ? size_t(view->cvOffsets[c]) - cvBase : 0);
            const size_t curve = curveBase + c;
            for (size_t i = 0; i < cv; ++i) {
                const size_t o = base + i;
                const size_t abs = cvBase + o;
                const float m = ready ? std::clamp(
                    static_cast<float>(maskField.Value(curve, abs)), 0.0f, 1.0f) : 0.0f;
                if (m == 0.0f || abs * 3 + 2 >= cap.rangeResult.size()) {
                    px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                    continue;
                }
                float const *r = &cap.rangeResult[abs * 3];
                px[o] = m == 1.0f ? r[0] : inPx[o] + (r[0] - inPx[o]) * m;
                py[o] = m == 1.0f ? r[1] : inPy[o] + (r[1] - inPy[o]) * m;
                pz[o] = m == 1.0f ? r[2] : inPz[o] + (r[2] - inPz[o]) * m;
            }
            if (restOut)
                restOut[c] = ready && curve < cap.restLength.size()
                    ? cap.restLength[curve]
                    : (cv ? UsdGenPolylineRestLength(inPx, inPy, inPz, base, cv) : 0.0f);
        }
        return;
    }

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        const size_t cv = view->cvCount ? size_t(view->cvCount)
            : (view->cvOffsets ? size_t(view->cvOffsets[c + 1] - view->cvOffsets[c]) : 0);
        const size_t base = view->cvCount ? size_t(c) * view->cvCount
            : (view->cvOffsets ? size_t(view->cvOffsets[c]) -
                (view->desc ? size_t(view->desc->firstCv) : 0) : 0);
        // A node configured for randomRange and later switched to a factor
        // mode keeps its restLength slot: publish the pre-scale length.
        if (restOut)
            restOut[c] = cv ? UsdGenPolylineRestLength(inPx, inPy, inPz, base, cv) : 0.0f;
        if (!cv) continue;
        const float rx = inPx[base], ry = inPy[base], rz = inPz[base];
        const double m = mult ? double(mult[curveBase + c]) : 1.0;
        // The per-curve controls sample the curve's root CV: one curve gets
        // one length factor, whatever granularity the expression declares.
        const size_t curve = curveBase + c;
        const size_t root = cvBase + base;
        const double value = valueField.Value(curve, root);
        const double cullThreshold = cullField.Value(curve, root);
        const double minRemaining = minField.Value(curve, root);
        float f;
        if (cap.mode == sSet) {
            // target = value * mult, absolute; scale it in over the runtime
            // rest length of THIS (possibly deformed) curve — plan/04 §2.10.
            const float restLen = UsdGenPolylineRestLength(inPx, inPy, inPz, base, cv);
            f = restLen > 1e-9f ? float(value * m / double(restLen)) : 0.0f;
        } else {  // "scale" and "cull": the ratio restLen cancels — plan/04 §2.10.
            f = float(value * m);
        }
        if (cullThreshold > 0.0 || minRemaining > 0.0) {
            const float restLen = UsdGenPolylineRestLength(inPx, inPy, inPz, base, cv);
            const double target = cap.mode == sSet
                                    ? value * m
                                    : double(restLen) * value * m;
            if (target < cullThreshold || target < minRemaining) f = 0.0f;
        }
        for (size_t i = 0; i < cv; ++i) {
            const size_t o = base + i;
            // usdGen:mask is the envelope (§2.13): the scale factor eases
            // toward identity (1.0), so mask == 0 is a bitwise pass-through.
            const float mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            const float fe = 1.0f + (f - 1.0f) * mask;
            if (fe == 1.0f) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            px[o] = rx + (inPx[o] - rx) * fe;
            py[o] = ry + (inPy[o] - ry) * fe;
            pz[o] = rz + (inPz[o] - rz) * fe;
        }
    }
}

}  // namespace usdGen
