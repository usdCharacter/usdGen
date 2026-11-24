// usdGen — UsdGenSmoothOp. 02-schema.md §2.7.1, 04-operators.md §2.12.
// Laplacian smoothing in two modes: alongCurve relaxes each strand toward
// its own neighbours (new[i] = lerp(old[i], neighborAvg, strength * mask),
// endpoints clamped to the single adjacent CV, CV0 pinned by lockRoot);
// neighbours blends each CV toward the same-index CVs of nearby strands
// (roots within searchRadius, up to numNeighbors nearest, self excluded);
// box is the imported-groom smoother: a Jacobi pass of
// P[k] = (1 - s) O[k] + s mean(O[k-w .. k+w]) with both endpoints fixed
// (s = clamp(strength, 0, 1) * mask, w = usdGen:window).
// TopologyEffect = None.
#ifndef USDGEN_OP_SMOOTH_H
#define USDGEN_OP_SMOOTH_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace smoothP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kMode, kSearchRadius, kNumNeighbors,
    kEnabled, kStrength, kIterations, kLockRoot, kMask, kWindow,
    kCount
};
enum Mode : int { kModeAlongCurve, kModeNeighbours, kModeBox };
}  // namespace smoothP

class UsdGenSmoothOp final : public UsdGenOp
{
public:
    UsdGenSmoothOp() = default;
    ~UsdGenSmoothOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenSmooth"); }

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
    void EvaluateBox(UsdGenEvalContext const &ctx, UsdGenChunkView *view) const;
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterSmoothOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_SMOOTH_H
