// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_NON_WIDTH_METADATA_H
#define USDGEN_VULKAN_NON_WIDTH_METADATA_H

// Host-side admission check for a Vulkan WidthBlend fan-in.  This deliberately
// compares only the immutable *non-width* contract: it has no dependency on a
// device allocation, buffer identity, or generation revision.
#include "sourceGeneration.h"

namespace usdGen::vulkan {
namespace detail {

inline bool SameNonWidthChannelMetadata(UsdGenDeviceChannelMetadata const& a,
                                        UsdGenDeviceChannelMetadata const& b)
{
    return a.name == b.name && a.type == b.type && a.domain == b.domain &&
        a.elementCount == b.elementCount && a.arity == b.arity &&
        a.strideBytes == b.strideBytes && a.readOnly == b.readOnly &&
        a.semantic == b.semantic;
}

inline bool SameNonWidthTileMetadata(UsdGenDeviceTileMetadata const& a,
                                     UsdGenDeviceTileMetadata const& b)
{
    if (a.tile != b.tile || a.firstCurve != b.firstCurve ||
        a.curveCount != b.curveCount || a.firstPoint != b.firstPoint ||
        a.pointCount != b.pointCount || a.boundsValid != b.boundsValid)
        return false;
    for (size_t axis = 0; axis != a.extentMin.size(); ++axis) {
        if (a.extentMin[axis] != b.extentMin[axis] ||
            a.extentMax[axis] != b.extentMax[axis])
            return false;
    }
    return true;
}

inline bool SameNonWidthGeometryMetadata(UsdGenDeviceGeometryMetadata const& a,
                                         UsdGenDeviceGeometryMetadata const& b)
{
    // topologyVersion/valueVersion are revision IDs, not fan-in semantics.
    if (a.curveCount != b.curveCount || a.pointCount != b.pointCount ||
        a.alreadyDeformed != b.alreadyDeformed ||
        a.curveTopology.type != b.curveTopology.type ||
        a.curveTopology.basis != b.curveTopology.basis ||
        a.curveTopology.wrap != b.curveTopology.wrap ||
        a.tiles.size() != b.tiles.size())
        return false;
    for (size_t i = 0; i != a.tiles.size(); ++i)
        if (!SameNonWidthTileMetadata(a.tiles[i], b.tiles[i])) return false;
    return true;
}

inline bool SameNonWidthChunks(std::vector<UsdGenChunkDesc> const& a,
                               std::vector<UsdGenChunkDesc> const& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i != a.size(); ++i) {
        UsdGenChunkDesc const& x = a[i];
        UsdGenChunkDesc const& y = b[i];
        if (x.firstCurve != y.firstCurve || x.curveCount != y.curveCount ||
            x.liveCount != y.liveCount || x.firstCv != y.firstCv ||
            x.cvCount != y.cvCount || x.tile != y.tile ||
            x.surface != y.surface || x.boundsRest != y.boundsRest)
            return false;
    }
    return true;
}

template <class Plane>
inline bool SameNonWidthPlaneSequence(std::vector<Plane> const& a,
                                      std::vector<Plane> const& b,
                                      bool excludeWidths)
{
    size_t ai = 0, bi = 0;
    for (;;) {
        while (ai != a.size() && excludeWidths &&
               a[ai].metadata.semantic == UsdGenDeviceChannelSemantic::Widths)
            ++ai;
        while (bi != b.size() && excludeWidths &&
               b[bi].metadata.semantic == UsdGenDeviceChannelSemantic::Widths)
            ++bi;
        if (ai == a.size() || bi == b.size()) return ai == a.size() && bi == b.size();
        // bytes is the descriptor's exposed payload span, not its allocation.
        if (!SameNonWidthChannelMetadata(a[ai].metadata, b[bi].metadata) ||
            a[ai].bytes != b[bi].bytes)
            return false;
        ++ai;
        ++bi;
    }
}

} // namespace detail

// Returns true only if these immutable snapshots can be joined by a width
// blend without changing any other geometry contract.  It compares counts,
// all geometry fields except revisions, ordered chunks (including live counts
// and rest bounds), public non-width plane schemas/spans, and private source
// frame plane schemas/spans.  It intentionally does not compare plane buffer
// handles, allocation ownership/identity, or any plane payload values.
inline bool SameNonWidthMetadata(VulkanSourceGeneration const& left,
                                 VulkanSourceGeneration const& right)
{
    return left.curveCount() == right.curveCount() &&
        left.pointCount() == right.pointCount() &&
        detail::SameNonWidthGeometryMetadata(left.geometry(), right.geometry()) &&
        detail::SameNonWidthChunks(left.chunks(), right.chunks()) &&
        detail::SameNonWidthPlaneSequence(left.planes(), right.planes(), true) &&
        detail::SameNonWidthPlaneSequence(left.sourceFrames(), right.sourceFrames(), false);
}

} // namespace usdGen::vulkan

#endif
