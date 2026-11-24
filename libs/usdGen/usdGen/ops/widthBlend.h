// usdGen — ordered binary WidthBlend operator.
//
// WidthBlend consumes two geometry inputs with identical topology and
// immutable non-width planes.  It writes a fresh width plane containing the
// ordered lerp of the left and right widths, weighted by its own
// usdGen:widthBlend:weight parameter; every other plane is inherited from the
// left input by the scheduler's normal CoW preparation.
#ifndef USDGEN_OP_WIDTH_BLEND_H
#define USDGEN_OP_WIDTH_BLEND_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenWidthBlendOp final : public UsdGenOp
{
public:
    UsdGenWidthBlendOp() = default;
    ~UsdGenWidthBlendOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenWidthBlend"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }
    size_t GeometryInputArity() const override { return 2; }
    bool GeometryInputsOrdered() const override { return true; }

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
    uint32_t PlanesTouched() const override { return kPlaneWidths; }
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterWidthBlendOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_WIDTH_BLEND_H
