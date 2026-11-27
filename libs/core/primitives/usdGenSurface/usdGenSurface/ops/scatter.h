// usdGen — UsdGenScatterOp. 02-schema.md §2.6, 04-operators.md §2.1.
// Emits random roots on the rest surface with stable ids:
//   curveId = UsdGenHash64(seed, faceIndex, k, kSaltScatter) (R12).
// usdGen:distribution / countMode / proximity:* / densityMap select the
// generation-order modes in ops/scatterModes.h; the defaults keep this path.
#ifndef USDGEN_OP_SCATTER_H
#define USDGEN_OP_SCATTER_H

#include "usdGen/op.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/registries/handles.h"
#include "usdGenCurves/export.h"
#include "usdGenSurface/types/rootSet.h"


namespace usdGen {

class UsdGenScatterOp final : public UsdGenOp
{
public:
    explicit UsdGenScatterOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenScatterOp() override = default;

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

private:
    // Static rest/topology array digests, memoized across poses (the steady
    // state rehashes megabytes of unchanged rest data every frame).
    mutable opUtil::ContentDigestCache<UsdGenVec3f> restPointsDigest_;
    mutable opUtil::ContentDigestCache<int> faceCountsDigest_;
    mutable opUtil::ContentDigestCache<int> faceIndicesDigest_;
    mutable opUtil::ContentDigestCache<int> subsetFacesDigest_;
    mutable opUtil::ContentDigestCache<UsdGenVec2f> uvDigest_;
    mutable opUtil::ContentDigestCache<float> densityMultDigest_;
    // Rest normals feed the root frames of the generation-order modes.
    mutable opUtil::ContentDigestCache<UsdGenVec3f> restNormalsDigest_;
    // UsdGenSubdivisionDigest re-hashes the topology arrays on top of the
    // array digests above; memoize it keyed on the same buffer identities
    // plus the scheme tokens (a miss recomputes exactly as before).
    struct SubdivisionDigestCache {
        uint64_t Digest(UsdGenSurfaceInputView const &surface);
        UsdGenRef<const UsdGenElementSet> owner_;   // holds the keyed columns
        UsdGenSpan<const int> counts_, indices_, holes_, creaseIndices_, creaseLengths_,
            cornerIndices_;
        UsdGenSpan<const float> creaseSharpnesses_, cornerSharpnesses_;
        UsdGenToken scheme_, orientation_, interpolateBoundary_,
            faceVaryingLinearInterpolation_, creaseMethod_,
            triangleSubdivisionRule_;
        uint64_t digest_ = 0;
        bool valid_ = false;
    };
    mutable SubdivisionDigestCache subdivisionDigest_;
};

/// Parameter slots, enumerants and tokens: the order of UsdGenRegisterScatterOp's tables.
namespace scatterP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kEnabled, kDensity, kFlip, kSubdivisionLevel, kDistribution,
    kCountMode, kProximityMode, kProximityDistance, kProximityCoverage, kGuides, kDensityMap,
    kCount
};
// Order matches scatter::Distribution / CountMode / ProximityMode.
enum Distribution : int { kDistRandom, kDistUvGrid, kDistVertex, kDistEven, kDistPerGuide };
enum CountMode : int { kCountPerFace, kCountSurface };
enum Proximity : int { kProxNone, kProxDistance, kProxCoverage };
enum Tok : uint16_t { kTokRandom, kTokPerFace, kTokNone, kTokLeftHanded, kTokDensityMap,
                      kTokStrandIndex, kTokStrandSeed };
}  // namespace scatterP

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterScatterOp(UsdGenRegistrar &);

/// A Scatter capture's roots as a RootSet view (static channels; no copy,
/// valid until the capture is re-captured or recycled). Invalid when the
/// capture holds no consistent roots.
// Built in usdGenCurves with the op (cmake/targets/primitives.cmake).
USDGEN_CURVES_API UsdGenRootSetView UsdGenScatterRoots(UsdGenCapture const &capture) noexcept;

}  // namespace usdGen

#endif  // USDGEN_OP_SCATTER_H
