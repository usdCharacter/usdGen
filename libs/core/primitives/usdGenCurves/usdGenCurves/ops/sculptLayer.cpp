// usdGen — UsdGenSculptLayerOp implementation. 02-schema.md §2.10.
//
// P[j] += frame(deltas[first + j]) * weight * mask for every matched strand,
// where frame(d) = T*dx + B*dy + N*dz in space = rootFrame (the rest root
// frame, the same tangent/binormal/normal row convention the compiler
// publishes) and d as-is in space = object. Matching is by stable curveId
// over the sorted layer ids, so hand work survives upstream tweaks that keep
// ids; layer entries with no upstream strand are ignored, and strands with
// no entry (or a zero weight/mask product) are copied bit-for-bit.
//
// Dirty-class split (02 §2.10): deltas and weight are value-class and are
// read live in Evaluate, so a slider edit sweeps without recapturing; ids,
// offsets, epoch, locks and rebase metadata are capture-class and are
// snapshotted in Capture. A delta span whose length disagrees with its
// strand's CV count fails closed: silently reindexing hand work is worse
// than a diagnostic. rootPrims/rootUVs exist for the tool's "Rebase sculpt"
// action and are validated, never consumed; lockedCurves freezes downstream
// stylers, never this layer. Stage-free: ctx/desc/params only.
#include "usdGenCurves/ops/sculptLayer.h"

#include "usdGenCurves/ops/opUtil.h"

#include "usdGenMath/math.h"
#include "usdGenMath/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "usdGen/registries/registrar.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"


namespace usdGen {

namespace {

struct UsdGenSculptLayerCapture final : public UsdGenCapture
{
    std::vector<uint64_t> ids;       // strictly ascending layer curveIds
    std::vector<uint32_t> offsets;   // ids.size() + 1 prefix offsets into deltas
    uint32_t totalDeltas = 0;
    uint64_t upstreamTopologyVersion = 0;
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;

    bool HasLayer() const { return !ids.empty(); }

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenSculptLayerCapture>(*this);
    }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        // The id->span match and the span-vs-CV-count proof below track the
        // upstream topology only; deformed positions move freely beneath.
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

bool Finite(double v) { return std::isfinite(v); }

UsdGenValue FindValue(UsdGenParamView const &params, UsdGenParamSlot slot)
{
    return UsdGenAuthoredValue(params, slot);
}

using namespace sculptP;
using D = expr::Domain;
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kFloatGroom{expr::ScalarType::Float32, 1, "float", D::Groom};
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
// Decl order is the legacy topology list, then the value list.
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"sculpt:space", UsdGenParamClass::Structural},     // kernel branch (02 §2.10)
    {"sculpt:curveIds", UsdGenParamClass::Capture},     // re-match deltas to ids
    {"sculpt:cvOffsets", UsdGenParamClass::Capture},
    {"sculpt:epoch", UsdGenParamClass::Capture},        // authored-against epoch
    {"sculpt:lockedCurves", UsdGenParamClass::Capture}, // freeze-brush id set
    {"sculpt:rootPrims", UsdGenParamClass::Capture},    // rebase metadata
    {"sculpt:rootUVs", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
    {"sculpt:weight", UsdGenParamClass::Value, true, kFloatGroom},
    {"sculpt:deltas", UsdGenParamClass::Value},
};
constexpr char const *kTokens[] = {"rootFrame", "object"};
std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenSculptLayerOp>(r);
}

}  // namespace

std::unique_ptr<UsdGenCapture> UsdGenSculptLayerOp::CreateCapture() const
{
    return std::make_unique<UsdGenSculptLayerCapture>();
}
uint32_t UsdGenSculptLayerOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenSculptLayerOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenSculptLayer: " + message);
        return false;
    };
    UsdGenToken const space = params.GetToken(kSpace, Record().Token(kTokRootFrame));
    if (space != Record().Token(kTokRootFrame) && space != Record().Token(kTokObject))
        return fail("unknown usdGen:sculpt:space '" + space.GetString() + "'");
    double const weight = params.GetDoubleLiteral(kWeight, 1.0);
    if (!Finite(weight) || weight < 0.0 || weight > 1.0)
        return fail("usdGen:sculpt:weight must be finite and in [0, 1]");
    if (!Finite(params.GetDoubleLiteral(kMask, 1.0)))
        return fail("usdGen:mask must be finite");

    UsdGenValue const ids = FindValue(params, kCurveIds);
    UsdGenValue const offsets = FindValue(params, kCvOffsets);
    UsdGenValue const deltas = FindValue(params, kDeltas);
    UsdGenValue const epoch = FindValue(params, kEpoch);
    UsdGenValue const locked = FindValue(params, kLocked);
    UsdGenValue const rootPrims = FindValue(params, kRootPrims);
    UsdGenValue const rootUVs = FindValue(params, kRootUVs);
    if (!ids.IsEmpty() && !ids.IsHolding<UsdGenCowArray<uint64_t>>())
        return fail("usdGen:sculpt:curveIds must be uint64[]");
    if (!offsets.IsEmpty() && !offsets.IsHolding<UsdGenIntArray>())
        return fail("usdGen:sculpt:cvOffsets must be int[]");
    if (!deltas.IsEmpty() && !deltas.IsHolding<UsdGenVec3fArray>())
        return fail("usdGen:sculpt:deltas must be vector3f[]");
    if (!epoch.IsEmpty() && !epoch.IsHolding<std::string>())
        return fail("usdGen:sculpt:epoch must be string");
    if (!locked.IsEmpty() && !locked.IsHolding<UsdGenCowArray<uint64_t>>())
        return fail("usdGen:sculpt:lockedCurves must be uint64[]");
    if (!rootPrims.IsEmpty() && !rootPrims.IsHolding<UsdGenIntArray>())
        return fail("usdGen:sculpt:rootPrims must be int[]");
    if (!rootUVs.IsEmpty() && !rootUVs.IsHolding<UsdGenVec2fArray>())
        return fail("usdGen:sculpt:rootUVs must be texCoord2f[]");

    UsdGenCowArray<uint64_t> const emptyIds;
    UsdGenIntArray const emptyInts;
    UsdGenVec3fArray const emptyDeltas;
    UsdGenCowArray<uint64_t> const &idArray =
        ids.IsHolding<UsdGenCowArray<uint64_t>>() ? ids.UncheckedGet<UsdGenCowArray<uint64_t>>() : emptyIds;
    UsdGenIntArray const &offsetArray =
        offsets.IsHolding<UsdGenIntArray>() ? offsets.UncheckedGet<UsdGenIntArray>() : emptyInts;
    UsdGenVec3fArray const &deltaArray =
        deltas.IsHolding<UsdGenVec3fArray>() ? deltas.UncheckedGet<UsdGenVec3fArray>() : emptyDeltas;
    size_t const nIds = idArray.size();

    // Layout (02 §2.10): strictly ascending ids, N + 1 prefix offsets from 0
    // to deltas.size(). The schema-default layer (every array empty) is the
    // pass-through, never an error.
    for (size_t i = 1; i < nIds; ++i) {
        if (idArray[i] == idArray[i - 1])
            return fail("usdGen:sculpt:curveIds carries a duplicate curveId");
        if (idArray[i] < idArray[i - 1])
            return fail("usdGen:sculpt:curveIds must be sorted ascending");
    }
    if (nIds == 0) {
        if (!deltaArray.empty())
            return fail("usdGen:sculpt:deltas without usdGen:sculpt:curveIds is malformed");
        if (!offsetArray.empty() &&
            (offsetArray.size() != 1 || offsetArray[0] != 0))
            return fail("an empty usdGen:sculpt:curveIds needs empty usdGen:sculpt:cvOffsets");
    } else {
        if (offsetArray.size() != nIds + 1)
            return fail("usdGen:sculpt:cvOffsets must carry curveIds.size() + 1 entries");
        if (offsetArray[0] != 0)
            return fail("usdGen:sculpt:cvOffsets must start at 0");
        for (size_t i = 1; i < offsetArray.size(); ++i) {
            if (offsetArray[i] < offsetArray[i - 1])
                return fail("usdGen:sculpt:cvOffsets must be non-decreasing");
        }
        if (size_t(offsetArray.back()) != deltaArray.size())
            return fail("usdGen:sculpt:cvOffsets must end at deltas.size()");
    }
    for (UsdGenVec3f const &d : deltaArray) {
        if (!Finite(d[0]) || !Finite(d[1]) || !Finite(d[2]))
            return fail("usdGen:sculpt:deltas must be finite");
    }
    if (locked.IsHolding<UsdGenCowArray<uint64_t>>()) {
        UsdGenCowArray<uint64_t> const &locks = locked.UncheckedGet<UsdGenCowArray<uint64_t>>();
        std::vector<uint64_t> sorted(locks.begin(), locks.end());
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            return fail("usdGen:sculpt:lockedCurves carries a duplicate curveId");
    }
    // Rebase metadata is optional but, when present, parallel to curveIds.
    if (rootPrims.IsHolding<UsdGenIntArray>()) {
        UsdGenIntArray const &prims = rootPrims.UncheckedGet<UsdGenIntArray>();
        if (!prims.empty() && prims.size() != nIds)
            return fail("usdGen:sculpt:rootPrims must be empty or parallel to curveIds");
        for (int prim : prims) {
            if (prim < 0)
                return fail("usdGen:sculpt:rootPrims must carry non-negative face indices");
        }
    }
    if (rootUVs.IsHolding<UsdGenVec2fArray>()) {
        UsdGenVec2fArray const &uvs = rootUVs.UncheckedGet<UsdGenVec2fArray>();
        if (!uvs.empty() && uvs.size() != nIds)
            return fail("usdGen:sculpt:rootUVs must be empty or parallel to curveIds");
        for (UsdGenVec2f const &uv : uvs) {
            if (!Finite(uv[0]) || !Finite(uv[1]))
                return fail("usdGen:sculpt:rootUVs must be finite");
        }
    }
    return true;
}

UsdGenEpoch UsdGenSculptLayerOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Structural space token plus every capture-class array, the upstream
    // generation and the seed. Value edits (weight, deltas, mask, including
    // connected-value edits, which the scheduler folds into the capture
    // identity separately) sweep without recapturing.
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    opUtil::Digest digest;
    digest.Mix(p ? p->GetToken(kSpace, Record().Token(kTokRootFrame)) : Record().Token(kTokRootFrame));
    if (p) {
        digest.Mix(FindValue(*p, kCurveIds));
        digest.Mix(FindValue(*p, kCvOffsets));
        digest.Mix(FindValue(*p, kEpoch));
        digest.Mix(FindValue(*p, kLocked));
        digest.Mix(FindValue(*p, kRootPrims));
        digest.Mix(FindValue(*p, kRootUVs));
    }
    digest.Mix(ctx.upstreamGeneration);
    digest.Mix(uint64_t(ctx.seed));
    return digest.Epoch(0x9E3779B97F4A7C15ull);
}

bool UsdGenSculptLayerOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &in,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in.Primary());
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenSculptLayerCapture *>(out);
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenSculptLayer: " + message);
        return false;
    };
    cap = UsdGenSculptLayerCapture();
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;
    // Literals and the layer layout are validated here (the CPU lane never
    // calls Bind).
    if (p && !Bind(*p, diag)) return false;
    std::vector<uint32_t> spans;
    std::string spansError;
    if (!opUtil::CurveSpans(upstream, &spans, &spansError))
        return fail("upstream CV topology " + spansError);

    UsdGenValue const ids = p ? FindValue(*p, kCurveIds) : UsdGenValue();
    UsdGenValue const offsets = p ? FindValue(*p, kCvOffsets) : UsdGenValue();
    UsdGenCowArray<uint64_t> const emptyIds;
    UsdGenIntArray const emptyInts;
    UsdGenCowArray<uint64_t> const &idArray = ids.IsHolding<UsdGenCowArray<uint64_t>>()
        ? ids.UncheckedGet<UsdGenCowArray<uint64_t>>() : emptyIds;
    UsdGenIntArray const &offsetArray = offsets.IsHolding<UsdGenIntArray>()
        ? offsets.UncheckedGet<UsdGenIntArray>() : emptyInts;
    UsdGenToken const space = p ? p->GetToken(kSpace, Record().Token(kTokRootFrame)) : Record().Token(kTokRootFrame);

    // Connected weight and mask scans, so a non-finite or out-of-range
    // expression fails closed at capture instead of writing NaN points.
    UsdGenParamField const weightField =
        p ? p->GetScalarField(kWeight, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    if (weightField.connected && weightField.components != 1)
        return fail("connected sculpt:weight must be float");
    if (maskField.connected && maskField.components != 1)
        return fail("connected mask must be float");
    for (uint32_t c = 0; c < upstream.totalCurves; ++c) {
        double const weight = weightField.Value(c, spans[c]);
        if (!Finite(weight) || weight < 0.0 || weight > 1.0)
            return fail("per-strand sculpt:weight must be finite and in [0, 1]");
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(maskField.Value(c, cv)))
                return fail("per-CV mask must be finite");
        }
    }

    if (idArray.empty()) return true;
    // A non-empty layer keys on stable ids, and in rootFrame space on the
    // rest root frame the compiler publishes per curve.
    if (upstream.curveId.size() != upstream.totalCurves)
        return fail("a sculpt layer needs stable per-curve ids upstream");
    if (space == Record().Token(kTokRootFrame) &&
        (upstream.rootT.size() != upstream.totalCurves ||
         upstream.rootB.size() != upstream.totalCurves ||
         upstream.rootN.size() != upstream.totalCurves))
        return fail("sculpt:space 'rootFrame' needs a complete rest root frame upstream");
    // Upstream ids are chunk-ordered, not sorted: index them once so the
    // match below and every Evaluate binary search stay deterministic.
    std::vector<std::pair<uint64_t, uint32_t>> byId;
    byId.reserve(upstream.totalCurves);
    for (uint32_t c = 0; c < upstream.totalCurves; ++c)
        byId.emplace_back(upstream.curveId[c], c);
    std::sort(byId.begin(), byId.end(),
              [](auto const &a, auto const &b) { return a.first < b.first; });
    for (size_t k = 0; k < idArray.size(); ++k) {
        auto const found = std::lower_bound(
            byId.begin(), byId.end(), idArray[k],
            [](std::pair<uint64_t, uint32_t> const &entry, uint64_t id) {
                return entry.first < id;
            });
        if (found == byId.end() || found->first != idArray[k])
            continue;  // kept and ignored (02 §2.10), never dropped
        uint32_t const span = uint32_t(offsetArray[k + 1] - offsetArray[k]);
        uint32_t const strand = spans[found->second + 1] - spans[found->second];
        if (span != strand)
            return fail("delta span carries " + std::to_string(span) +
                        " deltas but its strand carries " + std::to_string(strand) + " CVs");
    }
    cap.ids.assign(idArray.begin(), idArray.end());
    cap.offsets.reserve(offsetArray.size());
    for (int o : offsetArray) cap.offsets.push_back(uint32_t(o));
    cap.totalDeltas = cap.offsets.back();
    return true;
}

void UsdGenSculptLayerOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    auto const &cap = static_cast<UsdGenSculptLayerCapture const &>(captureIn);
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    auto copyThrough = [&](uint32_t c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        for (size_t i = 0; i < n; ++i) {
            px[g + i] = inPx[g + i]; py[g + i] = inPy[g + i]; pz[g + i] = inPz[g + i];
        }
    };
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    // Value-class reads: weight and deltas sweep without recapturing. A
    // resized deltas array against a stale capture span is a router-level
    // inconsistency Evaluate cannot diagnose, so it passes through rather
    // than reading out of bounds.
    UsdGenValue const deltasValue = p ? FindValue(*p, kDeltas) : UsdGenValue();
    UsdGenVec3fArray const emptyDeltas;
    UsdGenVec3fArray const *deltas = deltasValue.IsHolding<UsdGenVec3fArray>()
        ? &deltasValue.UncheckedGet<UsdGenVec3fArray>() : &emptyDeltas;
    if (cap.HasLayer() && deltas->size() != cap.totalDeltas) {
        for (uint32_t c = 0; c < view->curveCount; ++c) copyThrough(c);
        return;
    }
    UsdGenToken const space = p ? p->GetToken(kSpace, Record().Token(kTokRootFrame)) : Record().Token(kTokRootFrame);
    bool const rootFrame = space == Record().Token(kTokRootFrame);
    if (space != Record().Token(kTokRootFrame) && space != Record().Token(kTokObject)) {
        // Defensive pass-through; Capture already failed closed.
        for (uint32_t c = 0; c < view->curveCount; ++c) copyThrough(c);
        return;
    }
    UsdGenParamField const weightField =
        p ? p->GetScalarField(kWeight, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *curveId = view->curveId;
    auto const *rootT = view->rootT, *rootB = view->rootB, *rootN = view->rootN;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        float const weight = static_cast<float>(weightField.Value(curve, cvBase + g));
        // Unmatched strands, zero weight, missing ids and missing frames all
        // copy through; Capture failed closed on the malformed cases already.
        auto const span = cap.ids.empty() || !curveId ? cap.ids.end()
            : std::lower_bound(cap.ids.begin(), cap.ids.end(), curveId[c]);
        if (weight == 0.0f || span == cap.ids.end() || *span != curveId[c]) {
            copyThrough(c);
            continue;
        }
        size_t const k = size_t(span - cap.ids.begin());
        uint32_t const first = cap.offsets[k];
        if (cap.offsets[k + 1] - first != n) {
            copyThrough(c);
            continue;
        }
        UsdGenVec3f T(1, 0, 0), B(0, 1, 0), N(0, 0, 1);
        if (rootFrame) {
            if (!rootT || !rootB || !rootN) {
                copyThrough(c);
                continue;
            }
            T = rootT[c]; B = rootB[c]; N = rootN[c];
        }
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + o)), 0.0f, 1.0f);
            float const s = weight * mask;
            if (s == 0.0f) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                continue;
            }
            UsdGenVec3f const d = (*deltas)[size_t(first) + i] * s;
            if (rootFrame) {
                px[o] = inPx[o] + T[0] * d[0] + B[0] * d[1] + N[0] * d[2];
                py[o] = inPy[o] + T[1] * d[0] + B[1] * d[1] + N[1] * d[2];
                pz[o] = inPz[o] + T[2] * d[0] + B[2] * d[1] + N[2] * d[2];
            } else {
                px[o] = inPx[o] + d[0];
                py[o] = inPy[o] + d[1];
                pz[o] = inPz[o] + d[2];
            }
        }
    }
}

UsdGenOpId UsdGenRegisterSculptLayerOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenSculptLayer";
    d.displayName = "SculptLayer";
    d.create = &Create;
    d.params = UsdGenSpan(kParams);
    d.tokens = UsdGenSpan(kTokens);
    return r.RegisterOp(d);
}

}  // namespace usdGen
