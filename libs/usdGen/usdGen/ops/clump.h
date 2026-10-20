// usdGen — UsdGenClumpOp. 02-schema.md §2.7, 04-operators.md §2.8.
// Pulls strands toward clump centre strands. Clump membership comes from a
// connected usdGen:clump:map (one clump per distinct map value: a ptex region
// map, a geoSampler voronoi, any expression) or, unconnected, from centres
// scattered at usdGen:clump:density. Further levels split every clump into
// smaller ones (usdGen:clump:levels). usdGen:clump:centers selects the
// reference modes instead (count / guides / guideSeeded, see clump.cpp and
// ops/clumpModes.h), with Bezier profile, flyaways, overrides and
// sub-clumps. Emits clumpId_<n> (uniform int) per level. TopologyEffect =
// None.
#ifndef USDGEN_OP_CLUMP_H
#define USDGEN_OP_CLUMP_H

#include "usdGen/op.h"

#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenClumpOp final : public UsdGenOp
{
public:
    UsdGenClumpOp() = default;
    ~UsdGenClumpOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenClump"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;
    TfSpan<const TfToken> OutputPrimvars() const override;
    /// "guides": the guide set of the guides / guideSeeded centre modes.
    /// Optional: a node with no usdGen:guides target resolves none.
    TfSpan<const TfToken> ReferenceInputs() const override;
    void Configure(UsdGenParamView const &params) override;

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
    // Four native motion planes per level, set by Configure().
    // Default (level 0) outputs; Configure() recomputes per settings.
    std::vector<TfToken> _outputs{TfToken("clumpId_0"), TfToken("clumpCenter_0"),
                                  TfToken("clumpCenterId_0"), TfToken("clumpWeight_0")};
    int _levelBaseOverride = -1;
public:
    // The compiler supplies the chain-wide topological allocation. Direct
    // standalone Configure callers retain the descriptor-order fallback.
    void SetLevelBase(int base) { _levelBaseOverride = base; }
private:
    const TfToken sMask{"mask"};
};

}  // namespace usdGen

#endif  // USDGEN_OP_CLUMP_H
