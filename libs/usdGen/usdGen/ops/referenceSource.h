// usdGen — immutable authored reference source (CPU reference lane).
#ifndef USDGEN_OP_REFERENCESOURCE_H
#define USDGEN_OP_REFERENCESOURCE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenReferenceSourceOp final : public UsdGenOp
{
public:
    UsdGenReferenceSourceOp() = default;
    ~UsdGenReferenceSourceOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenReferenceSource"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CurveCount; }
    UsdGenRole Role() const override { return UsdGenRole::Reference; }
    bool IsGenerator() const override { return true; }
    size_t GeometryInputArity() const override { return 0; }

    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }
    TfSpan<const TfToken> ReferenceInputs() const override;

    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override;
    bool Capture(UsdGenCaptureContext const &, UsdGenCurveBuffer const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override;
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView *) const override {}
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override { return 0; }
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterReferenceSourceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_REFERENCESOURCE_H
