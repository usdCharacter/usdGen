// usdGen — UsdGenCurlOp. 02-schema.md §2.7, 04-operators.md §3.
// Coils each strand around its own tangent frame at a radius and frequency:
//   P[i] += r(t) * (cos(theta_i) N_i + sin(theta_i) B_i),
//   theta_i = 2*pi*f*s_i + phi_c, s = arc length from the root.
// (N, B) are rotation-minimizing frames along the input curve
// (axisMode = curveTangent) or the root frame (axisMode = guide).
// TopologyEffect = None.
#ifndef USDGEN_OP_CURL_H
#define USDGEN_OP_CURL_H

#include "usdGen/registries/op.h"
#include "usdGen/registries/handles.h"


namespace usdGen {

namespace curlP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kAxisMode,
    kEnabled, kRadius, kRadiusKnots, kRadiusInterp, kFrequency, kPhase, kPhaseRandom, kTaper,
    kClockwise, kMask,
    kCount
};
enum AxisMode : int { kAxisCurveTangent, kAxisGuide };
}  // namespace curlP

class UsdGenCurlOp final : public UsdGenOp
{
public:
    explicit UsdGenCurlOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenCurlOp() override = default;


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
UsdGenOpId UsdGenRegisterCurlOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_CURL_H
