// usdGen — UsdGenNoiseOp (M1). 02-schema.md §2.7.1, 04-operators.md §2.9.
// Frizz: correlated fBm displacement in the ROOT frame so it follows the
// deforming surface (I3 — the fBm is sampled on rest roots at capture via
// SeExpr's noise (S38); per-frame only the mask envelope re-runs).
// TopologyEffect = None (topology-preserving styler).
#ifndef USDGEN_OP_NOISE_H
#define USDGEN_OP_NOISE_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace noiseP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kEnabled, kMagnitude, kMagKnots, kMagInterp, kFrequency,
    kCorrelation, kOctaves, kLacunarity, kGain, kCumulative, kPreserveLength, kMask,
    kNoiseSeed,                          // admission only; unrouted, never read
    kCount
};
enum Tok : uint16_t { kTokCatmullRom };
}  // namespace noiseP

class UsdGenNoiseOp final : public UsdGenOp
{
public:
    explicit UsdGenNoiseOp(UsdGenOpRecord const &record) : UsdGenOp(record) {}
    ~UsdGenNoiseOp() override = default;

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
UsdGenOpId UsdGenRegisterNoiseOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_NOISE_H
