// Deterministic material-coordinate transfer into a simple concave polygon.
#ifndef USDGEN_CONCAVE_MATERIAL_REMAP_H
#define USDGEN_CONCAVE_MATERIAL_REMAP_H

#include "usdGen/export.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Deterministically ear-triangulates a simple actual material polygon while
/// retaining its original input slot IDs.  Collinear boundary slots remain in
/// the resulting nondegenerate triangles.  `triangles` is unchanged on
/// failure, including self-intersecting or collapsed input polygons.
USDGEN_CORE_API bool UsdGenTriangulateConcaveMaterialSlots(
    std::vector<GfVec2f> const &actual, std::vector<std::array<int, 3>> *triangles,
    std::string *error = nullptr);

/// Resolves a canonical point through a previously triangulated N-slot
/// polygon. This lets dense runtime callers cache a target ring triangulation
/// for a shared sampled station while retaining the same boundary/tie policy
/// as the one-shot remap below.
USDGEN_CORE_API bool UsdGenLocateConcaveMaterialPoint(
    size_t slotCount, std::vector<std::array<int, 3>> const &triangles,
    GfVec2f const &canonicalPoint, std::array<int, 3> *slots,
    GfVec3f *weights, std::string *error = nullptr);

/// Maps a point in the canonical unit regular N-gon to a simple actual
/// N-slot polygon. Slot IDs remain the input order; no geometry sorting or
/// nearest-point fallback is used. `canonicalPoint` is accepted only inside
/// a canonical triangle corresponding to a nondegenerate ear of `actual`.
/// `slots` and `weights` identify that actual triangle. Boundary points clamp
/// only floating-point roundoff and are deterministic on a shared diagonal.
/// On failure outputs remain unchanged.
USDGEN_CORE_API bool UsdGenRemapConcaveMaterialPoint(
    std::vector<GfVec2f> const &actual, GfVec2f const &canonicalPoint,
    std::array<int, 3> *slots, GfVec3f *weights,
    std::string *error = nullptr);

} // namespace usdGen

#endif // USDGEN_CONCAVE_MATERIAL_REMAP_H
