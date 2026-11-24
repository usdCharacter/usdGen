// usdGen — UsdGenWaveOp. 02-schema.md §2.7, 04-operators.md §3.
// Sinusoidal displacement in the root tangent/normal directions:
//   P[i] += A_T sin(2*pi*f_T*s_i) T + A_N sin(2*pi*f_N*s_i) N,
// where s is the arc length from the root. Roots (s = 0) never move.
// TopologyEffect = None.
#ifndef USDGEN_OP_WAVE_H
#define USDGEN_OP_WAVE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace waveP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed,
    kEnabled, kFrequencyU, kFrequencyN, kAmplitudeU, kAmplitudeN, kMask,
    kCount
};
}  // namespace waveP

class UsdGenWaveOp final : public UsdGenOp
{
public:
    explicit UsdGenWaveOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenWaveOp() override = default;


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
UsdGenOpId UsdGenRegisterWaveOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_WAVE_H
