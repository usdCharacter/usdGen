// usdGen — UsdGenWidthOp (M1). 02-schema.md §2.7.1, 04-operators.md §2.11.
// Authors `widths` from a base width and a root->tip ramp
// (usdGen:width:knots, S11), or the root/mid/tip Bezier thickness profile
// (usdGen:width:profile:mode = "bezier"); usdGen:replace selects set (true)
// vs multiply (false).
// TopologyEffect = None.
#ifndef USDGEN_OP_WIDTH_H
#define USDGEN_OP_WIDTH_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace widthP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kReplace, kEnabled, kWidth, kKnots, kKnotsInterp,
    kProfileMode, kRoot, kMid, kMidShift, kTip, kMask,
    kCount
};
enum ProfileMode : int { kProfileRamp, kProfileBezier };
enum Tok : uint16_t { kTokCatmullRom };
}  // namespace widthP

class UsdGenWidthOp final : public UsdGenOp
{
public:
    explicit UsdGenWidthOp(UsdGenOpRecord const &record) : UsdGenOp(record) {}
    ~UsdGenWidthOp() override = default;

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

/// For lanes that sample a width profile LUT over hairT (CUDA, Vulkan): when
/// `params` select usdGen:width:profile:mode = "bezier", fills `lut` (`size`
/// entries at t = k/(size-1)) with the Bezier thickness factor the CPU lane
/// applies per point (clamp01(f), f <= 1e-6 -> 1) and returns true; returns
/// false and leaves `lut` untouched in ramp mode.
bool UsdGenWidthBezierLut(UsdGenParamView const &params, float *lut, size_t size);

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterWidthOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_WIDTH_H
