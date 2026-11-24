// usdGen — UsdGenFreezeOp. 02-schema.md §2.9, 05-static-curves-and-deformation.md §5.5.
//
// Caps the chain at a snapshot. frozen:mode = "frozen" evaluates the
// deep-copied snapshot and ignores upstream values; "live" passes the chain
// input through (the snapshot is retained-but-stale per 02 §2.9, i.e. a
// live->frozen transition re-snapshots). The snapshot source is the optional
// frozen:curves reference input (Reference role, part:curves precedent) when
// bound, else the chain input at first capture.
//
// SESSION TIER ONLY (02 §2.9 tier is advisory): usdGen:frozen:tier
// "sublayer"/"payload" are lowered to "session" with a capture warning; the
// evaluator never persists snapshot data outside the capture.
//
// Session-tier layout rule: an explicit snapshot must match the chain-input
// layout exactly (curve count, CV count, ragged offsets) or Capture fails
// closed. Mid-chain curve-count changes are not representable in the
// scheduler's graph-wide partition (UsdGenLength degenerates culled strands
// for the same reason); a count-changing snapshot is the sublayer/payload
// tier's re-topology job, out of scope here.
//
// Frozen mode is IMMUNE to upstream edits: CaptureDigest excludes the
// upstream generation and ValidForTopology always returns true, so upstream
// value or generation edits never recapture. Mode and epoch edits DO
// recapture (both are digest terms). There is intentionally no stale check
// against the target's frozenEpoch: 02 §2.9 grants the stale-policy knob to
// UsdGenCurveSource alone, and the reference transport does not carry the
// epoch string.
//
// Framework notes (scheduler-owned, not kernel bugs): PrepareNodeForEval
// lets only Grow/Resample author rest and named planes, so v1 frozen output
// carries the snapshot's points/widths/per-curve ids but the upstream rest
// and upstream extra planes. hairT is layout-derived (i/(n-1) in both the
// reference builder and CurveSource), hence identical to the snapshot's
// under the enforced layout match, so the kernel does not touch that plane.
// CPU-only: the CUDA capability matrix has no UsdGenFreeze row and the
// planner rejects the graph explicitly; there is no .cu backend.
//
// TopologyEffect = CurveCount: the static worst case (an explicit snapshot
// may change the curve count, exactly like Length's cull mode), which also
// routes usdGen:enabled as structural (topology) per 02 §6.2.
#ifndef USDGEN_OP_FREEZE_H
#define USDGEN_OP_FREEZE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenFreezeOp final : public UsdGenOp
{
public:
    UsdGenFreezeOp() = default;
    ~UsdGenFreezeOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenFreeze"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CurveCount; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;
    TfSpan<const TfToken> ReferenceInputs() const override;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenCurveBuffer const &upstream,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenChunkView *view) const override;
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override;
};

namespace freezeP {
/// Declaration order == today's TopologyParameters() then ValueParameters().
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kEnabled, kCurves, kMode, kEpoch, kTier, kCount
};
enum Tok : uint16_t { kTokFrozen, kTokLive, kTokSession, kTokSublayer, kTokPayload };
}  // namespace freezeP

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterFreezeOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_FREEZE_H
