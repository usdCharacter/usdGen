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

#include "usdGen/api/ext/op.h"
#include "usdGen/api/ids.h"


namespace usdGen {

namespace displaceP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kMode, kMap,
    kEnabled, kMask, kAmount, kBase, kScale, kOffset,
    kCount
};
enum Mode : int { kModeHeight, kModeVector };
}  // namespace displaceP

class UsdGenDisplaceOp final : public UsdGenOp
{
public:
    explicit UsdGenDisplaceOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenDisplaceOp() override = default;


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
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterDisplaceOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_DISPLACE_H
