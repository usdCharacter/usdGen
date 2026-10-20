// usdGen — UsdGenWidthOp (M1). 02-schema.md §2.7.1, 04-operators.md §2.11.
// Authors `widths` from a base width and a root->tip ramp
// (usdGen:width:knots, S11), or the root/mid/tip Bezier thickness profile
// (usdGen:width:profile:mode = "bezier"); usdGen:replace selects set (true)
// vs multiply (false).
// TopologyEffect = None.
#ifndef USDGEN_OP_WIDTH_H
#define USDGEN_OP_WIDTH_H

#include "usdGen/op.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenWidthOp final : public UsdGenOp
{
public:
    UsdGenWidthOp() = default;
    ~UsdGenWidthOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenWidth"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }

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
    // Intern once per operator, with the same lifetime as its scheduled work.
    const TfToken sWidth{"width"}, sReplace{"replace"}, sMask{"mask"};
    const TfToken sKnots{"width:knots"}, sKnotsInterp{"width:interpolation"};
    const TfToken sCatmullRom{"catmullRom"};
};

/// For lanes that sample a width profile LUT over hairT (CUDA, Vulkan): when
/// `params` select usdGen:width:profile:mode = "bezier", fills `lut` (`size`
/// entries at t = k/(size-1)) with the Bezier thickness factor the CPU lane
/// applies per point (clamp01(f), f <= 1e-6 -> 1) and returns true; returns
/// false and leaves `lut` untouched in ramp mode.
bool UsdGenWidthBezierLut(UsdGenParamView const &params, float *lut, size_t size);

}  // namespace usdGen

#endif  // USDGEN_OP_WIDTH_H
