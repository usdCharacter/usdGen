#ifndef USDGEN_OP_CURVESOURCE_H
#define USDGEN_OP_CURVESOURCE_H

#include "usdGen/registries/op.h"
#include "usdGen/registries/handles.h"


namespace usdGen {

class UsdGenRestSurfaceBindingCache;

/// UsdGenCurveSourceOp — loads curves from the scene index into a buffer.
/// Produces uniform or ragged chunks depending on the source data.
class UsdGenCurveSourceOp final : public UsdGenOp
{
public:
   explicit UsdGenCurveSourceOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
   ~UsdGenCurveSourceOp() override = default;

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
   uint32_t PlanesTouched() const override;

private:
   // Capture is serialized by the graph owner; consumers only see immutable
   // indices, never a process-global mutable cache shared between sessions.
   std::shared_ptr<const UsdGenRestSurfaceBindingCache> bindingCache_;
};

/// Parameter slots: the order of UsdGenRegisterCurveSourceOp's decl table.
namespace curveSourceP {
enum : UsdGenParamSlot {
   kInput, kSurface, kSeed, kEnabled, kMode, kUseRest, kUsdGenUseRest, kInterpolationMode,
   kDensityMultiplier, kRegionMapChannel, kExpectMapGeneration, kResampleTo, kIdSource,
   kStaleAction, kExpectEpoch, kRebind,
   kUsdGenResampleTo,    // read, not routed
   kCount
};
enum Tok : uint16_t { kTokLoad, kTokNone, kTokSurfaceCage };
}  // namespace curveSourceP

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterCurveSourceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_CURVESOURCE_H
