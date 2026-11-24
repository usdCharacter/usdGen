// usdGen — UsdGenFreezeOp implementation. 02-schema.md §2.9, 05 §5.5.
//
// Frozen mode deep-copies the snapshot (explicit frozen:curves reference, or
// the chain input at first capture) into an owning capture; the scheduler
// installs it as this node's buffer, so frozen Evaluate is a no-op and later
// upstream edits can neither recapture (digest excludes upstream;
// ValidForTopology is unconditional) nor corrupt the private planes. Live
// mode owns nothing and Evaluate memcpys the chain input to the output.
// TopologyEffect = CurveCount (static worst case; see freeze.h).
#include "usdGen/ops/freeze.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

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

// Private copy: VtArray assignment is a CoW share, and the scheduler writes
// upstream planes through const-cast raw pointers (SweepChunk), which never
// detaches. A shared "snapshot" would corrupt on the next upstream commit,
// so every snapshot plane is a fresh allocation filled by memcpy.
template <typename Array>
void DeepCopyArray(Array const &src, Array *dst)
{
    dst->resize(src.size());
    if (!src.empty()) std::copy(src.begin(), src.end(), dst->begin());
}

void DeepCopyPlanes(std::vector<UsdGenPlane> const &src,
                    std::vector<UsdGenPlane> *dst)
{
    dst->resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        (*dst)[i].name = src[i].name;
        (*dst)[i].interpolation = src[i].interpolation;
        (*dst)[i].type = src[i].type;
        (*dst)[i].arity = src[i].arity;
        DeepCopyArray(src[i].f, &(*dst)[i].f);
        DeepCopyArray(src[i].i, &(*dst)[i].i);
    }
}

void DeepCopyBuffer(UsdGenCurveBuffer const &src, UsdGenCurveBuffer *dst)
{
    dst->totalCurves = src.totalCurves;
    dst->totalCvs = src.totalCvs;
    dst->topologyVersion = src.topologyVersion;
    dst->valueVersion = src.valueVersion;
    DeepCopyArray(src.px, &dst->px);
    DeepCopyArray(src.py, &dst->py);
    DeepCopyArray(src.pz, &dst->pz);
    DeepCopyArray(src.rest, &dst->rest);
    DeepCopyArray(src.width, &dst->width);
    DeepCopyArray(src.hairT, &dst->hairT);
    DeepCopyArray(src.curveId, &dst->curveId);
    DeepCopyArray(src.rootPrim, &dst->rootPrim);
    DeepCopyArray(src.rootUV, &dst->rootUV);
    DeepCopyArray(src.rootT, &dst->rootT);
    DeepCopyArray(src.rootN, &dst->rootN);
    DeepCopyArray(src.rootB, &dst->rootB);
    DeepCopyArray(src.cvOffsets, &dst->cvOffsets);
    // Named planes ARE snapshot state: PrepareNodeForEval keeps private
    // owners for frozen Freeze (captureOwnsTransformedPlanes, alongside
    // Grow/Resample), so installed planes survive upstream edits exactly
    // like the rest channel. A frozen node that passed live extras through
    // would leak the chain it caps (02 §2.9).
    DeepCopyPlanes(src.extraCv, &dst->extraCv);
    DeepCopyPlanes(src.extraCurve, &dst->extraCurve);
}

struct UsdGenFreezeCapture final : public UsdGenCapture
{
    bool live = false;
    std::string epoch;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenFreezeCapture>(*this);
    }
    bool OwnsBuffer() const override { return !live; }
    bool ValidForTopology(UsdGenCurveBuffer const &) const override
    {
        // The freeze contract: frozen output NEVER follows upstream edits,
        // so an upstream topology/value revision alone never invalidates.
        // Mode/epoch edits move CaptureDigest and recapture explicitly.
        return true;
    }
};

}  // namespace

namespace {

// Schema: uniform string usdGen:frozen:epoch = "". Missing/empty is the
// default; TfToken is tolerated exactly as GetToken tolerates std::string.
bool ReadEpoch(UsdGenParamView const *p, TfToken const &name,
               std::string *epoch, UsdGenDiagnostics *diag, char const *what)
{
    epoch->clear();
    if (!p) return true;
    VtValue const v = p->GetVtValue(name, VtValue());
    if (v.IsEmpty()) return true;
    if (v.IsHolding<std::string>()) {
        *epoch = v.UncheckedGet<std::string>();
        return true;
    }
    if (v.IsHolding<TfToken>()) {
        *epoch = v.UncheckedGet<TfToken>().GetString();
        return true;
    }
    if (diag)
        diag->Error(std::string(what) +
                    ": usdGen:frozen:epoch must be a string (schema: uniform string)");
    return false;
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("enabled"));        // 02 §6.2: topology (CurveCount)
    v.push_back(TfToken("frozen:curves"));  // structural: the snapshot edge
    v.push_back(TfToken("frozen:mode"));    // structural: the unfreeze gesture
    v.push_back(TfToken("frozen:epoch"));   // capture: the recapture trigger
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("frozen:tier"));    // cosmetic/advisory: never recaptures
    return v;
}();

static TfTokenVector const _referenceInputs{TfToken("frozen:curves")};

TfSpan<const TfToken> UsdGenFreezeOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenFreezeOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
TfSpan<const TfToken> UsdGenFreezeOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(_referenceInputs.data(), _referenceInputs.size());
}
std::unique_ptr<UsdGenCapture> UsdGenFreezeOp::CreateCapture() const
{
    return std::make_unique<UsdGenFreezeCapture>();
}
uint32_t UsdGenFreezeOp::PlanesTouched() const
{
    return kPlanePoints | kPlaneWidths;
}

bool UsdGenFreezeOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sMode, sFrozen);
    if (mode != sFrozen && mode != sLive) {
        if (diag) diag->Error("UsdGenFreeze: unknown usdGen:frozen:mode '" +
                              mode.GetString() + "'");
        return false;
    }
    TfToken const tier = params.GetToken(sTier, sSession);
    if (tier != sSession && tier != sSublayer && tier != sPayload) {
        if (diag) diag->Error("UsdGenFreeze: unknown usdGen:frozen:tier '" +
                              tier.GetString() + "'");
        return false;
    }
    std::string epoch;
    return ReadEpoch(&params, sEpoch, &epoch, diag, "UsdGenFreeze");
}

UsdGenEpoch UsdGenFreezeOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Immunity by construction: the upstream generation is NOT a digest term,
    // so upstream value/topology edits never recapture. Mode and epoch are:
    // the unfreeze/re-freeze gestures re-snapshot explicitly. Tier is cosmetic
    // (02 §2.9: the evaluator never reads it) and stays out of the digest.
    opUtil::Digest d;
    TfToken mode = sFrozen;
    std::string epoch;
    if (ctx.params) {
        mode = ctx.params->GetToken(sMode, sFrozen);
        std::string validated;
        if (ReadEpoch(ctx.params, sEpoch, &validated, nullptr, ""))
            epoch = validated;
    }
    d.Mix(TfToken("frozen:mode"));
    d.Mix(mode);
    d.Mix(TfToken("frozen:epoch"));
    for (char c : epoch) d.Mix(uint64_t(uint8_t(c)));
    d.Mix(uint64_t(epoch.size()));
    d.Mix(uint64_t(ctx.seed));
    return d.Epoch(0x467265657A6531ull);
}

bool UsdGenFreezeOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenFreezeCapture *>(out);
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenFreeze: " + message);
        return false;
    };
    if (p && !Bind(*p, diag)) return false;

    TfToken const mode = p ? p->GetToken(sMode, sFrozen) : sFrozen;
    TfToken const tier = p ? p->GetToken(sTier, sSession) : sSession;
    std::string epoch;
    if (!ReadEpoch(p, sEpoch, &epoch, diag, "UsdGenFreeze")) return false;
    // SESSION TIER ONLY: sublayer/payload have no in-engine landing place;
    // lower them to session loudly (warning, comment, test, report) rather
    // than failing an asset the tool is free to relocate.
    if (tier == sSublayer || tier == sPayload) {
        if (diag) diag->Warn("UsdGenFreeze: usdGen:frozen:tier '" +
                             tier.GetString() +
                             "' is not implemented; lowering to 'session'");
    }

    std::string topoError;
    auto checkPlanes = [&](UsdGenCurveBuffer const &b, char const *whose) -> bool {
        std::vector<uint32_t> spans;
        if (!opUtil::CurveSpans(b, &spans, &topoError))
            return fail(std::string(whose) + topoError);
        if (b.px.size() != b.totalCvs || b.py.size() != b.totalCvs ||
            b.pz.size() != b.totalCvs)
            return fail(std::string(whose) +
                        "point plane cardinality does not match its CV topology");
        if (!b.width.empty() && b.width.size() != b.totalCvs)
            return fail(std::string(whose) +
                        "width plane cardinality does not match its CV topology");
        return true;
    };

    if (mode == sLive) {
        // Passthrough owns nothing; the snapshot (if any) is retained-but-
        // stale per 02 §2.9. Upstream planes are still validated so a
        // malformed input fails here with a message instead of mis-copying
        // silently in Evaluate (which cannot fail).
        if (!checkPlanes(upstream, "live input ")) return false;
        cap.live = true;
        cap.epoch = epoch;
        return true;
    }

    // Frozen: the explicit reference wins; else the chain input. The compiler
    // enforces 0-or-1 resolved references; direct callers are checked here.
    if (ctx.referenceCount > 1)
        return fail("usdGen:frozen:curves must target at most one snapshot curve set");
    UsdGenCurveBuffer const *snapshot = nullptr;
    bool const explicitSnapshot =
        ctx.referenceCount == 1 && ctx.resolvedReferences && ctx.resolvedReferences[0] &&
        ctx.resolvedReferences[0]->value;
    if (ctx.referenceCount == 1 && !explicitSnapshot)
        return fail("usdGen:frozen:curves names a snapshot with no resolved value");
    if (explicitSnapshot)
        snapshot = &ctx.resolvedReferences[0]->value->buffer;
    else
        snapshot = &upstream;

    if (snapshot->totalCurves == 0 || snapshot->totalCvs == 0)
        return fail(explicitSnapshot ? "the frozen:curves snapshot has no curves"
                                     : "nothing to freeze: the chain input has no curves");
    if (!checkPlanes(*snapshot, "snapshot ")) return false;
    if (explicitSnapshot && upstream.totalCurves > 0) {
        // Session-tier layout rule (freeze.h): the graph-wide partition is
        // sized from the chain, so a differently-shaped snapshot cannot be
        // installed. Counts AND ragged offsets must agree.
        if (snapshot->totalCurves != upstream.totalCurves ||
            snapshot->totalCvs != upstream.totalCvs ||
            !(snapshot->cvOffsets == upstream.cvOffsets))
            return fail("the frozen:curves snapshot layout (curves/CVs/offsets) "
                        "must match the chain input layout in the session tier");
    }

    UsdGenCurveBuffer &buf = cap.MutableBuffer();
    DeepCopyBuffer(*snapshot, &buf);
    // Widths follow CurveSource's rule: a snapshot without a width plane
    // materializes the description default instead of inventing zeros.
    if (buf.width.empty()) {
        float fallback = 0.01f;
        if (ctx.desc) {
            if (!std::isfinite(ctx.desc->defaultWidth) || ctx.desc->defaultWidth < 0.0f)
                return fail("description defaultWidth must be finite and >= 0 "
                            "to freeze a snapshot without widths");
            fallback = ctx.desc->defaultWidth;
        }
        buf.width.assign(buf.totalCvs, fallback);
    }
    // hairT is layout-derived (i/(n-1)); canonicalize it when the source did
    // not carry a complete plane so the installed snapshot is self-consistent.
    if (buf.hairT.size() != buf.totalCvs) {
        std::vector<uint32_t> spans;
        if (!opUtil::CurveSpans(buf, &spans, &topoError))
            return fail(std::string("snapshot ") + topoError);
        buf.hairT.resize(buf.totalCvs);
        for (uint32_t c = 0; c < buf.totalCurves; ++c) {
            uint32_t const first = spans[c], n = spans[c + 1] - first;
            for (uint32_t i = 0; i < n; ++i)
                buf.hairT[first + i] = n > 1 ? float(i) / float(n - 1) : 0.0f;
        }
    }
    cap.live = false;
    cap.epoch = epoch;
    return true;
}

void UsdGenFreezeOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenFreezeCapture const &>(captureIn);
    if (cap.live) {
        // Passthrough: chain input to private output. Layouts agree by
        // construction (a non-owning node aliases upstream totals); every
        // span below is still clamped so a mismatched direct caller corrupts
        // nothing. Widths follow the Width-op rule: a missing input plane
        // reads as the description default.
        float defaultWidth = 0.01f;
        if (ctx.desc && std::isfinite(ctx.desc->defaultWidth) &&
            ctx.desc->defaultWidth >= 0.0f)
            defaultWidth = ctx.desc->defaultWidth;
        int const firstCv = view->desc ? view->desc->firstCv : 0;
        for (uint32_t c = 0; c < view->curveCount; ++c) {
            int const nOut = view->cvOffsets
                ? view->cvOffsets[c + 1] - view->cvOffsets[c]
                : int(view->cvCount);
            int const gOut = view->cvOffsets
                ? view->cvOffsets[c] - firstCv
                : int(c * view->cvCount);
            int const nIn = view->inCvOffsets
                ? view->inCvOffsets[c + 1] - view->inCvOffsets[c]
                : int(view->inCvCount);
            int const gIn = view->inCvOffsets
                ? view->inCvOffsets[c] - int(view->inFirstCv)
                : int(c * view->inCvCount);
            if (nOut <= 0 || nIn <= 0 || gOut < 0 || gIn < 0) continue;
            int const n = std::min(nOut, nIn);
            for (int i = 0; i < n; ++i) {
                size_t const o = size_t(gOut + i), in = size_t(gIn + i);
                if (view->px && view->inPx) view->px[o] = view->inPx[in];
                if (view->py && view->inPy) view->py[o] = view->inPy[in];
                if (view->pz && view->inPz) view->pz[o] = view->inPz[in];
                if (view->width)
                    view->width[o] = view->inWidth ? view->inWidth[in] : defaultWidth;
            }
        }
        return;
    }
    // Frozen: intentional no-op. The deep-copied snapshot was installed as
    // this node's buffer at capture and prepare keeps it (sizes match, and it
    // aliases neither upstream nor any older generation). Rewriting it here
    // would write through const-cast pointers into capture-shared planes.
}

UsdGenOpId UsdGenRegisterFreezeOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenFreezeOp().Type(), [] { return std::make_unique<UsdGenFreezeOp>(); });
}

}  // namespace usdGen
