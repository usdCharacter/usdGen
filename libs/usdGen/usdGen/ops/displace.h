// usdGen — UsdGenDisplaceOp. 02-schema.md §2.7, 04-operators.md §3.
// Height-field displacement along the rest root normal:
//   P[i] += N_root * amount * ((sample - base) * scale + offset) * mask,
// where sample is the displace:map value at the strand root (the rel target
// is sampled by the expression lane and arrives as a groom/primitive scalar
// field, exactly like clump:map; 0 when unconnected) and amount, base,
// scale, offset and mask are sampled per CV. Every CV including the root
// moves: this type declares no lockRoots. mode = height only; mode = vector
// fails closed because no vector map type is consumable through the existing
// transport (only UsdGenPtexMap, scalar channels). TopologyEffect = None.
#ifndef USDGEN_OP_DISPLACE_H
#define USDGEN_OP_DISPLACE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenDisplaceOp final : public UsdGenOp
{
public:
    UsdGenDisplaceOp() = default;
    ~UsdGenDisplaceOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenDisplace"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }

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
    uint32_t PlanesTouched() const override;
private:
    const TfToken sAmount{"displace:amount"}, sMap{"displace:map"};
    const TfToken sBase{"displace:base"}, sScale{"displace:scale"};
    const TfToken sOffset{"displace:offset"}, sMode{"mode"};
    const TfToken sHeight{"height"}, sVector{"vector"}, sMask{"mask"};
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterDisplaceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_DISPLACE_H
