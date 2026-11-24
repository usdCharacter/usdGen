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

class UsdGenBendOp final : public UsdGenOp
{
public:
    UsdGenBendOp() = default;
    ~UsdGenBendOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenBend"); }
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
    const TfToken sAngle{"angle"}, sAngleKnots{"angle:knots"};
    const TfToken sAngleInterp{"angle:interpolation"}, sCatmullRom{"catmullRom"};
    const TfToken sAngleRandom{"angleRandom"}, sAxis{"axis"};
    const TfToken sAxisMode{"axisMode"}, sRootDirection{"rootDirection"};
    const TfToken sUniform{"uniform"}, sAttribute{"attribute"}, sMask{"mask"};
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterBendOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_BEND_H
