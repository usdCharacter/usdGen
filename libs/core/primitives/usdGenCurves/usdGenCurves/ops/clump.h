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
    explicit UsdGenClumpOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenClumpOp() override = default;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenCurveBuffer const &upstream,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenLegacyChunkView *view) const override;
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override;
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterClumpOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_CLUMP_H
