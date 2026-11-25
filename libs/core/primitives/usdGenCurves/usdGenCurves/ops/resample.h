// usdGen — UsdGenResampleOp. 02-schema.md §2.7.1, 04-operators.md §2.11.
// Changes the CV count: every strand is resampled to usdGen:cvCount CVs,
// either uniformly by arc length (distribution = uniform) or by linear
// interpolation in input-index space (distribution = keepParam, which keeps
// the input parameterization). usdGen:restoreSegmentLengths rescales the
// resampled strand about its root to the input total length. The mask
// blends per output CV between the resampled position and the nearest input
// CV (mask = 0 holds the input shape); roots are exact in all modes.
// Widths and rest resample with the same weights at capture; points evaluate
// per chunk so mask edits sweep without recapture.
// usdGen:spacing > 0 replaces the fixed count by a per-strand one,
// clamp(ceil(L / spacing) + 1, 2, kUsdGenMaxCvsPerCurve) with L the strand's
// current (upstream) length, and publishes a ragged layout (cvOffsets).
// TopologyEffect = CvCount.
#ifndef USDGEN_OP_RESAMPLE_H
#define USDGEN_OP_RESAMPLE_H

#include "usdGen/ops/recordedOp.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace resampleP {
/// Slots, in decl order.
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kCvCount, kDistribution, kRestore, kSpacing, kEnabled,
    kMask, kCount
};
enum Tok : uint16_t { kTokUniform, kTokKeepParam };
}  // namespace resampleP

class UsdGenResampleOp final : public UsdGenOp
{
public:
    explicit UsdGenResampleOp(UsdGenOpRecord const &record) : UsdGenOp(record) {}
    ~UsdGenResampleOp() override = default;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenCurveBuffer const &upstream,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenLegacyChunkView *view) const override;
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override;
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterResampleOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_RESAMPLE_H
