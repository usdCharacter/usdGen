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


namespace usdGen {

namespace widthBlendP {
enum : UsdGenParamSlot { kEnabled, kWeight, kCount };
}  // namespace widthBlendP

class UsdGenWidthBlendOp final : public UsdGenOp
{
public:
    explicit UsdGenWidthBlendOp(UsdGenOpRecord const &record) : UsdGenOp(record) {}
    ~UsdGenWidthBlendOp() override = default;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenOpInputs const &in,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenChunkView const &chunk) const override;
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override { return kPlaneWidths; }
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterWidthBlendOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_WIDTH_BLEND_H
