// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The RootSet primitive: surface roots, one level (one element per root).
// Scatter declares it as its output type; root consumers (Grow,
// GuideInterpolate) and the GPU scatter inputs read roots through
// UsdGenRootSetView. No pxr include and no curves header here.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGenSurface/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"
#include <cstddef>
#include <cstdint>

PXR_NAMESPACE_OPEN_SCOPE
class UsdGenVec2f;
class UsdGenVec3f;
PXR_NAMESPACE_CLOSE_SCOPE

namespace usdGen {

using UsdGenVec2f;
using UsdGenVec3f;

class UsdGenRegistrar;

/// Static channels of the RootSet layout (slot == enumerator), all level 1.
enum UsdGenRootSlot : UsdGenSlot {
    kUsdGenRootPx, kUsdGenRootPy, kUsdGenRootPz,              // F32 Position, comp 0/1/2 (planar)
    kUsdGenRootId,                                            // U64 Id
    kUsdGenRootPrim,                                          // I32 surface face
    kUsdGenRootUV,                                            // F32x2 packed Uv
    kUsdGenRootT, kUsdGenRootN, kUsdGenRootB,                 // F32x3 packed Frame
    kUsdGenRootStaticSlots
};

/// "UsdGenRootSet": op ports name the type through this.
USDGEN_SURFACE_API char const *UsdGenRootSetTypeName() noexcept;
/// Valid once UsdGenRegisterRootSetType ran (the op registry's first use).
USDGEN_SURFACE_API UsdGenTypeId UsdGenRootSetType() noexcept;
USDGEN_SURFACE_API UsdGenElementLayout const &UsdGenRootSetLayout() noexcept;
/// Interns the static channel names and registers "UsdGenRootSet". Idempotent.
USDGEN_SURFACE_API UsdGenTypeId UsdGenRegisterRootSetType(UsdGenRegistrar &);

/// Partition hook: chunks of params.chunkSize roots; no sub-items.
USDGEN_SURFACE_API uint32_t UsdGenRootSetPartition(uint32_t elements, UsdGenElementSetView const &set,
                                                UsdGenPartitionParams const &,
                                                UsdGenSpan<UsdGenChunkRange> out);
/// Group hook: groupTarget chunks per group, the last group takes the rest.
USDGEN_SURFACE_API uint32_t UsdGenRootSetGroups(uint32_t chunks, UsdGenPartitionParams const &,
                                             UsdGenSpan<UsdGenChunkGroup> out);

/// A per-root dynamic channel (levels 0-1, name order).
struct UsdGenRootPlaneView {
    UsdGenToken name;
    UsdGenScalar scalar = UsdGenScalar::F32;
    uint8_t arity = 1;
    uint8_t level = 1;                     // 0 constant, 1 per root
    UsdGenSpan<const float> f;             // F32
    UsdGenSpan<const int> i;               // I32
};

/// Read view with the curve view's field names for what root consumers read.
/// Of() reads a RootSet, or a set whose static channels carry the root names
/// with exactly one position per element (until the scheduler keeps per-node
/// layouts, the engine stores Scatter's roots in the graph's partition layout).
/// Anything else yields an empty view (totalCurves 0, valid false).
struct UsdGenRootSetView {
    USDGEN_SURFACE_API static UsdGenRootSetView Of(UsdGenElementSetView const &) noexcept;
    static UsdGenRootSetView Of(UsdGenElementSet const &s) noexcept { return Of(s.View()); }
    UsdGenSpan<const float> px, py, pz;
    UsdGenSpan<const uint64_t> curveId;
    UsdGenSpan<const int> rootPrim;
    UsdGenSpan<const UsdGenVec2f> rootUV;
    UsdGenSpan<const UsdGenVec3f> rootT, rootN, rootB;
    uint32_t totalCurves = 0;              // roots
    uint64_t topologyVersion = 0, valueVersion = 0;
    bool valid = false;
    USDGEN_SURFACE_API uint32_t PlaneCount() const noexcept;
    USDGEN_SURFACE_API UsdGenRootPlaneView Plane(uint32_t) const noexcept;
private:
    UsdGenElementSetView _set{};
};

/// Copies a view into a fresh RootSet (one arena, static channels and
/// per-root planes); null when the view is invalid. Allocates: capture,
/// compile and test time only.
USDGEN_SURFACE_API UsdGenRef<const UsdGenElementSet>
UsdGenRootSetFromView(UsdGenRootSetView const &, uint64_t topologyVersion, uint64_t valueVersion);

}  // namespace usdGen
