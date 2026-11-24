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

class UsdGenNoiseOp final : public UsdGenOp
{
public:
    UsdGenNoiseOp() = default;
    ~UsdGenNoiseOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenNoise"); }
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
    bool CaptureDigestComplete() const override { return true; }
    uint32_t PlanesTouched() const override;
private:
    const TfToken sMagnitude{"noise:magnitude"}, sCumulative{"cumulative"};
    const TfToken sPreserveLength{"preserveLength"}, sMagKnots{"noise:magnitude:knots"};
    const TfToken sMagInterp{"noise:magnitude:interpolation"}, sCatmullRom{"catmullRom"};
    const TfToken sMask{"mask"};
    const TfToken sFrequency{"noise:frequency"}, sCorrelation{"noise:correlation"};
    const TfToken sLacunarity{"noise:lacunarity"}, sGain{"noise:gain"};
    const TfToken sOctaves{"noise:octaves"};
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterNoiseOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_NOISE_H
