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
    explicit UsdGenReferenceSourceOp(UsdGenOpRecord const &record) noexcept : _record(&record) {}
    ~UsdGenReferenceSourceOp() override = default;

    UsdGenOpRecord const &Record() const noexcept { return *_record; }
    // Registered values served from the record; removed with the legacy virtuals (S4-8).
    TfToken Type() const override { return _record->Name(); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CurveCount; }
    UsdGenRole Role() const override { return UsdGenRole::Reference; }
    bool IsGenerator() const override { return true; }
    size_t GeometryInputArity() const override { return 0; }

    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }
    TfSpan<const TfToken> ReferenceInputs() const override { return _record->ReferenceInputs(); }

    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override;
    bool Capture(UsdGenCaptureContext const &, UsdGenCurveBuffer const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override;
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView *) const override {}
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override { return 0; }
private:
    UsdGenOpRecord const *_record;
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterReferenceSourceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_REFERENCESOURCE_H
