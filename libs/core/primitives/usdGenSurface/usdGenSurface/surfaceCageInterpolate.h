// Runtime interpolation of a sparse, surface-owned cage into transient C3 hair.
//
// This is deliberately an engine-only value helper.  It receives a captured
// rest mesh and immutable Ptex sampler, never a USD stage, scene-index, or
// Pomade model.  CurveSource and CUDA source preparation may therefore share
// its exact CPU admission and source topology without importing UI code.
#ifndef USDGEN_SURFACE_CAGE_INTERPOLATE_H
#define USDGEN_SURFACE_CAGE_INTERPOLATE_H

#include "usdGen/export.h"
#include "usdGen/graphDesc.h"
#include "usdGen/maps/ptexMap.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// One captured sparse rail.  `normalizedT` names each stored point's tube
/// parameter, is strictly increasing from 0 to 1, and is sampled linearly;
/// it is never inferred from geometric arc length.  Empty `rest` means that
/// the captured points are also the rail's C3 rest positions.  Empty widths
/// use the owning cage's defaultWidth.
struct UsdGenSurfaceCageGuide
{
    int ownerTubeId = -1;
    uint64_t curveId = 0;
    VtVec3fArray points;
    VtVec3fArray rest;
    VtFloatArray normalizedT;
    VtFloatArray widths;
};

/// Exactly three sparse rails define a cage triangle.  The explicit owner is
/// not inferred from hierarchy or nearest geometry; all three rails must
/// carry the same ownerTubeId.
struct UsdGenSurfaceCageTriangle
{
    int ownerTubeId = -1;
    std::array<uint32_t, 3> guides{{0, 0, 0}};
    // Captured chart positions of the three sparse root rails.  They are
    // used only for an owner's authored radial length profile; Ptex lookup
    // still comes from the sampled rest-surface point.
    std::array<GfVec2f, 3> rootChart{{GfVec2f(0.0f, 0.0f),
                                       GfVec2f(0.0f, 0.0f),
                                       GfVec2f(0.0f, 0.0f)}};
};

/// Captured per-leaf output controls.  The owner-indexed flattened length
/// profiles live on UsdGenSurfaceCageInput so descriptor and CUDA-preparation
/// payloads use one directly serialisable layout.
struct UsdGenSurfaceCageOwner
{
    int ownerTubeId = -1;
    float density = 0.0f;
    int seed = 0;
    uint32_t cvCount = 0;
    float defaultWidth = 0.01f;
    float edgeBias = 0.0f;
    GfVec2f chartCentroid{0.0f, 0.0f};
    float chartMeanRadius = 1.0f;
};

/// Immutable categorical owner map.  The opened texture must correspond to
/// `options`; channel zero stores `ownerTubeId + 1`, while zero is empty.
/// `firstFaceIds` follows UsdGenPtexFirstFaceIds for a quad/n-gon texture and
/// is unused for an mt_triangle texture.  The caller owns map generation and
/// cache identity; this helper owns no global map state.
struct UsdGenSurfaceCageOwnerMap
{
    std::shared_ptr<const UsdGenPtexTexture> texture;
    UsdGenPtexMapOptions options;
    std::vector<int> firstFaceIds;
    uint32_t channel = 0;
};

/// Fully captured, standalone runtime input.  Scatter always evaluates
/// `surface.restPoints`; a current-pose fallback is rejected.  `densityMultiplier`
/// scales every owner's density before the deterministic per-face count rule.
struct UsdGenSurfaceCageInput
{
    std::vector<UsdGenSurfaceCageGuide> guides;
    std::vector<UsdGenSurfaceCageTriangle> triangles;
    std::vector<UsdGenSurfaceCageOwner> owners;
    // offsets has owners.size()+1 entries and indexes `lengthProfile` as
    // float2 (normalised radial input, profile length multiplier).  Every
    // owner needs at least one finite pair; x is strictly increasing in
    // [0,1].  Linear interpolation clamps outside its authored domain.
    VtIntArray ownerLengthProfileOffsets;
    VtVec2fArray ownerLengthProfile;
    UsdGenSurfaceDesc surface;
    UsdGenSurfaceCageOwnerMap ownerMap;
    float densityMultiplier = 1.0f;
};

/// Flat, CurveSource-ready C3 data.  All arrays are curve-major and have
/// one uniform `cvCount` span per `curveVertexCounts` entry.  `ownerTubeId`
/// remains uniform output ownership rather than an implicit nearest-guide
/// decision.  `points[base]` and `rest[base]` equal the sampled rest root.
struct UsdGenSurfaceCageResult
{
    VtIntArray curveVertexCounts;
    VtVec3fArray points;
    VtVec3fArray rest;
    VtFloatArray widths;
    VtFloatArray hairT;
    VtArray<uint64_t> curveId;
    VtIntArray rootPrim;
    VtVec2fArray rootUV;
    VtIntArray ownerTubeId;
};

/// Scatters rest roots, accepts only categorical Ptex owner values exactly
/// equal to `ownerTubeId + 1`, selects one triangle of that same owner, and
/// barycentrically interpolates its three rails.  At output progress q it
/// applies the owner radial length profile, samples rail t=q*length, and
/// applies the owner edge-bias weights with a q fade so c0 remains exact.
/// Roots with owner-map zero, another owner, invalid Ptex coordinates, or no
/// owner triangle are rejected; there is no cross-owner or nearest fallback.
/// Structural errors return false and leave `out` unchanged.  A valid input
/// whose map rejects every candidate returns true with an empty result.
USDGEN_CORE_API bool UsdGenInterpolateSurfaceCage(
    UsdGenSurfaceCageInput const &input, UsdGenSurfaceCageResult *out,
    std::string *error = nullptr);

}  // namespace usdGen

#endif  // USDGEN_SURFACE_CAGE_INTERPOLATE_H
