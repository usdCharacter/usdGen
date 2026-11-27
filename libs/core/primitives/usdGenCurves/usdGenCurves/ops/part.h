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

#include <vector>


namespace usdGen {

namespace partP {
// Declaration order of ops/part.cpp kParams (topology list, then value list).
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kCurves, kRadius,
    kEnabled, kStrength, kMask,
    kCount
};
}  // namespace partP

class UsdGenPartOp final : public UsdGenOp
{
public:
    explicit UsdGenPartOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenPartOp() override = default;

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
UsdGenOpId UsdGenRegisterPartOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_PART_H
