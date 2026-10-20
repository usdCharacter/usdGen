// usdGen — UsdGenGuideInterpolateOp. 02-schema.md §2.6, 04-operators.md §2.3.
// Roots -> strands shaped by a sparse guide curve set (usdGen:guides).
// usdGen:weighting = "falloff" (default): every strand blends the root-local
// shapes of its nearest guides, in each guide's root frame, constrained by a
// region field (usdGen:region, a connectable per-strand id: a ptex region
// map, a geoSampler voronoi over the guide roots, any expression).
// "nearest" / "inverseDistance" / "gaussian": the imported-groom guide blend
// (ops/guideBlend.h) with bend-follow frames and parting.
// Emits guideIndex / guideWeight (uniform, arity 3) and the uniform int
// planes primaryGuide, nearestGuide, partId, mirrorSign; upstream uniform
// planes (strandIndex, strandSeed) are carried through.
// TopologyEffect = CvCount, like Grow.
#ifndef USDGEN_OP_GUIDE_INTERPOLATE_H
#define USDGEN_OP_GUIDE_INTERPOLATE_H

#include "usdGen/op.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenGuideInterpolateOp final : public UsdGenOp
{
public:
    UsdGenGuideInterpolateOp() = default;
    ~UsdGenGuideInterpolateOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenGuideInterpolate"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CvCount; }
    bool IsGenerator() const override { return true; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;
    TfSpan<const TfToken> ReferenceInputs() const override;

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

}  // namespace usdGen

#endif  // USDGEN_OP_GUIDE_INTERPOLATE_H
