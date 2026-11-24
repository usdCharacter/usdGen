// usdGen — UsdGenPartOp. 02-schema.md §2.7.2, 04-operators.md §3.
// Parting lines from a curve set: every strand is assigned the side of its
// nearest parting curve (emitted as partId, a uniform int) and strands within
// usdGen:part:radius of the line are pushed apart along the surface, scaled
// by usdGen:part:strength and the mask envelope, so the part reads even
// without a downstream guide/clump consumer. TopologyEffect = None.
#ifndef USDGEN_OP_PART_H
#define USDGEN_OP_PART_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenPartOp final : public UsdGenOp
{
public:
    UsdGenPartOp() = default;
    ~UsdGenPartOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenPart"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;
    TfSpan<const TfToken> ReferenceInputs() const override;
    TfSpan<const TfToken> OutputPrimvars() const override;

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
    const TfToken sRadius{"part:radius"}, sStrength{"part:strength"};
    const TfToken sMask{"mask"};
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterPartOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_PART_H
