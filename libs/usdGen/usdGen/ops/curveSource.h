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
   UsdGenCurveSourceOp();
   ~UsdGenCurveSourceOp() override = default;

   TfToken Type() const override { return TfToken("UsdGenCurveSource"); }
   UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CurveCount; }
   UsdGenRole Role() const override { return UsdGenRole::Curves; }
   bool IsGenerator() const override { return true; }
   size_t GeometryInputArity() const override { return 0; }

   TfSpan<const TfToken> TopologyParameters() const override;
   TfSpan<const TfToken> ValueParameters() const override;
   TfSpan<const TfToken> ReferenceInputs() const override { return {}; }
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
   TfTokenVector topologyParameters_;
   TfTokenVector valueParameters_;
   // Capture is serialized by the graph owner; consumers only see immutable
   // indices, never a process-global mutable cache shared between sessions.
   std::shared_ptr<const UsdGenRestSurfaceBindingCache> bindingCache_;
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterCurveSourceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_CURVESOURCE_H
