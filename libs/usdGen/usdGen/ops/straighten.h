// usdGen — UsdGenStraightenOp. 02-schema.md §2.7, 04-operators.md §3.
// Blend each strand toward the straight root->tip line, decomposed per plane:
//   P[i] = lerp(P[i], root + chord*s_i, k) decomposed per plane,
// where s_i is the arc-length fraction from the root along the INPUT curve and
// the lerp is split across the root frame: usdGen:tangentStraightness scales
// the tangent-frame component of (P[i] - linePos), usdGen:normalStraightness
// the normal-frame component. The binormal-frame component passes through
// unchanged (no binormal straightness parameter exists). Roots and tips (the
// line endpoints) never move.
// TopologyEffect = None.
#ifndef USDGEN_OP_STRAIGHTEN_H
#define USDGEN_OP_STRAIGHTEN_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace straightenP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed,
    kEnabled, kTangentStraightness, kNormalStraightness, kMask,
    kCount
};
}  // namespace straightenP

class UsdGenStraightenOp final : public UsdGenOp
{
public:
    UsdGenStraightenOp() = default;
    ~UsdGenStraightenOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenStraighten"); }

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
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterStraightenOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_STRAIGHTEN_H
