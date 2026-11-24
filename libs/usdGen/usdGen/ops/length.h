// usdGen — UsdGenLengthOp (M1). 02-schema.md §2.7.1, 04-operators.md §2.10.
// Set, scale, cut or cull strand length. The WHOLE type is topology-bumping
// (ADR §9 R14): static TopologyEffect() == CurveCount because cull mode
// exists, so the classification does not depend on usdGen:length:mode.
//
// usdGen:length:mode = "randomRange" (CPU lane only) is the imported-groom
// length stage: a whole-strand capture (ops/lengthRange.h) that rescales
// every strand to a random multiple of its own (or its clump mean's) length,
// resampled uniformly in arc length, and emits the `restLength` plane (the
// pre-scale reference length). The emittedPlanes hook declares the plane
// only for a node in randomRange mode.
#ifndef USDGEN_OP_LENGTH_H
#define USDGEN_OP_LENGTH_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace lengthP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kEnabled, kMode, kMethod, kRebuild, kValue, kRandom,
    kMinRemaining, kCull, kMin, kMax, kRandomize, kShareByClump, kFollowClumpMean,
    kMask,
    kCount
};
enum Mode : int { kModeScale, kModeSet, kModeCull, kModeRandomRange };
enum Method : int { kMethodScale, kMethodCutExtend };
enum Tok : uint16_t { kTokClumpId0, kTokLengthMap, kTokRestLength };
}  // namespace lengthP

class UsdGenLengthOp final : public UsdGenOp
{
public:
    explicit UsdGenLengthOp(UsdGenOpRecord const &record) : UsdGenOp(record) {}
    ~UsdGenLengthOp() override = default;

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
UsdGenOpId UsdGenRegisterLengthOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_LENGTH_H
