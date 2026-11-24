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
    explicit UsdGenReferenceSourceOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenReferenceSourceOp() override = default;

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
UsdGenOpId UsdGenRegisterReferenceSourceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_REFERENCESOURCE_H
