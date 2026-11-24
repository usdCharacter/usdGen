#ifndef USDGEN_OP_CURVESOURCE_H
#define USDGEN_OP_CURVESOURCE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenRestSurfaceBindingCache;

/// UsdGenCurveSourceOp — loads curves from the scene index into a buffer.
/// Produces uniform or ragged chunks depending on the source data.
class UsdGenCurveSourceOp final : public UsdGenOp
{
public:
   explicit UsdGenCurveSourceOp(UsdGenOpRecord const &record) noexcept : _record(&record) {}
   ~UsdGenCurveSourceOp() override = default;

   UsdGenOpRecord const &Record() const noexcept { return *_record; }
   // Registered values served from the record; removed with the legacy virtuals (S4-8).
   TfToken Type() const override { return _record->Name(); }
   UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CurveCount; }
   UsdGenRole Role() const override { return UsdGenRole::Curves; }
   bool IsGenerator() const override { return true; }
   size_t GeometryInputArity() const override { return 0; }

   TfSpan<const TfToken> TopologyParameters() const override { return {}; }
   TfSpan<const TfToken> ValueParameters() const override { return {}; }
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
   UsdGenOpRecord const *_record;
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
