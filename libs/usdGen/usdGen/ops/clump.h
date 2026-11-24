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
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace clumpP {
// Declaration order of ops/clump.cpp kParams: the topology list (every
// control but the envelope is capture-class), then the value list.
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed,
    kAmount, kMap, kDensity, kSize, kLevels, kLevel, kSizeReduction, kTightnessReduction,
    kMethod, kProfileKnots, kProfileInterpolation, kStrayAmount, kStrayRate, kStrayFalloff,
    kPreserveLength,
    // Reference modes and the guide relationship.
    kGuides, kCenters, kClumpCount, kProfileMode, kRoot, kMid, kMidShift, kTip,
    kFlyawayFraction, kFlyawayAmount, kSubEnabled, kSubCount, kSubSpread, kSubAmount,
    kSubIds, kOverrides, kOverrideCount,
    kEnabled, kMask,
    kParamCount
};
enum MethodEnum : int { kLinearBlend, kExtrudeAndBlend };
enum CenterModeEnum : int { kCentersDensity, kCentersCount, kCentersGuides, kCentersGuideSeeded };
enum ProfileModeEnum : int { kProfileRamp, kProfileBezier };
// Record().Token(i)
enum Tok : uint16_t {
    kTokClumpId0, kTokClumpCenter0, kTokClumpCenterId0, kTokClumpWeight0,
    kTokStrandIndex, kTokPartId, kTokNearestGuide
};
}  // namespace clumpP

class UsdGenClumpOp final : public UsdGenOp
{
public:
    explicit UsdGenClumpOp(UsdGenOpRecord const &record);
    ~UsdGenClumpOp() override = default;

    UsdGenOpRecord const &Record() const noexcept { return *_record; }

    // Static description, answered from the record until S4-8 removes the
    // virtuals (tests and the engine still ask the instance).
    TfToken Type() const override { return _record->Name(); }
    UsdGenTopoFx TopologyEffect() const override { return _record->Def().topoFx; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;
    /// The planes of the levels Configure() numbered (4a-1 moves this to a hook).
    TfSpan<const TfToken> OutputPrimvars() const override;
    /// "guides": the guide set of the guides / guideSeeded centre modes.
    /// Optional: a node with no usdGen:guides target resolves none.
    TfSpan<const TfToken> ReferenceInputs() const override { return _record->ReferenceInputs(); }
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
    UsdGenOpRecord const *_record;
    std::vector<TfToken> _topology, _value;
    // Four native motion planes per level, set by Configure().
    // Default (level 0) outputs; Configure() recomputes per settings.
    std::vector<TfToken> _outputs;
    int _levelBaseOverride = -1;
public:
    // The compiler supplies the chain-wide topological allocation. Direct
    // standalone Configure callers retain the descriptor-order fallback.
    void SetLevelBase(int base) { _levelBaseOverride = base; }
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterClumpOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_CLUMP_H
