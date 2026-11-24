// usdGen — UsdGenScaleOp. 02-schema.md §2.7.1, 04-operators.md §2.14.
// Global length multiplier about each root:
//   P[i] = root + (P_in[i] - root) * lerp(1, scale * ramp(t_i) * mult_c, mask_i),
// where ramp is the scale:knots LUT over hairT and mult_c is the per-curve
// draw over the scaleRandom range. Roots never move; usdGen:widthToo also
// scales widths by the same per-CV factor.
// TopologyEffect = None.
#ifndef USDGEN_OP_SCALE_H
#define USDGEN_OP_SCALE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace scaleP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kWidthToo,
    kEnabled, kScale, kScaleKnots, kScaleInterp, kScaleRandom, kMask,
    kCount
};
}  // namespace scaleP

class UsdGenScaleOp final : public UsdGenOp
{
public:
    UsdGenScaleOp() = default;
    ~UsdGenScaleOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenScale"); }

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

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterScaleOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_SCALE_H
