// usdGen — UsdGenScatterOp. 02-schema.md §2.6, 04-operators.md §2.1.
// Emits random roots on the rest surface with stable ids:
//   curveId = UsdGenHash64(seed, faceIndex, k, kSaltScatter) (R12).
// usdGen:distribution / countMode / proximity:* / densityMap select the
// generation-order modes in ops/scatterModes.h; the defaults keep this path.
#ifndef USDGEN_OP_SCATTER_H
#define USDGEN_OP_SCATTER_H

#include "usdGen/op.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenScatterOp final : public UsdGenOp
{
public:
    explicit UsdGenScatterOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenScatterOp() override = default;

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

private:
    // Static rest/topology array digests, memoized across poses (the steady
    // state rehashes megabytes of unchanged rest data every frame).
    mutable opUtil::ContentDigestCache<VtVec3fArray> restPointsDigest_;
    mutable opUtil::ContentDigestCache<VtIntArray> faceCountsDigest_;
    mutable opUtil::ContentDigestCache<VtIntArray> faceIndicesDigest_;
    mutable opUtil::ContentDigestCache<VtIntArray> subsetFacesDigest_;
    mutable opUtil::ContentDigestCache<VtVec2fArray> uvDigest_;
    mutable opUtil::ContentDigestCache<VtFloatArray> densityMultDigest_;
    // Rest normals feed the root frames of the generation-order modes.
    mutable opUtil::ContentDigestCache<VtVec3fArray> restNormalsDigest_;
    // UsdGenSubdivisionDigest re-hashes the topology arrays on top of the
    // array digests above; memoize it keyed on the same buffer identities
    // plus the scheme tokens (a miss recomputes exactly as before).
    struct SubdivisionDigestCache {
        uint64_t Digest(UsdGenSurfaceDesc const &surface);
        VtIntArray counts_, indices_, holes_, creaseIndices_, creaseLengths_,
            cornerIndices_;
        VtFloatArray creaseSharpnesses_, cornerSharpnesses_;
        TfToken scheme_, orientation_, interpolateBoundary_,
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

}  // namespace usdGen

#endif  // USDGEN_OP_SCATTER_H
