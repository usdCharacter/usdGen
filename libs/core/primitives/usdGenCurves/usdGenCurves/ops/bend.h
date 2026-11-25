// usdGen — UsdGenBendOp. 02-schema.md §2.7, 04-operators.md §3.
// Cumulative per-segment rotation: the CVs at and after segment k rotate
// about P[k] by the curve's angle share for that segment, so a straight
// strand becomes an arc and the segment lengths are preserved by
// construction. The ramp distributes the total angle along the strand; a
// flat ramp bends rigidly about the root.
// TopologyEffect = None.
#ifndef USDGEN_OP_BEND_H
#define USDGEN_OP_BEND_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace bendP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kAxisMode,
    kEnabled, kAngle, kAngleKnots, kAngleInterp, kAngleRandom, kAxis, kMask,
    kCount
};
enum AxisMode : int { kAxisRootDirection, kAxisUniform, kAxisAttribute };
}  // namespace bendP

class UsdGenBendOp final : public UsdGenOp
{
public:
    explicit UsdGenBendOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenBendOp() override = default;


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
UsdGenOpId UsdGenRegisterBendOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_BEND_H
