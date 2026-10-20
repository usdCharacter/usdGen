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

#include "usdGen/op.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenResampleOp final : public UsdGenOp
{
public:
    UsdGenResampleOp() = default;
    ~UsdGenResampleOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenResample"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CvCount; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;

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
private:
    const TfToken sCvCount{"cvCount"}, sDistribution{"distribution"};
    const TfToken sUniform{"uniform"}, sKeepParam{"keepParam"};
    const TfToken sRestore{"restoreSegmentLengths"}, sMask{"mask"};
    const TfToken sSpacing{"spacing"};
};

}  // namespace usdGen

#endif  // USDGEN_OP_RESAMPLE_H
