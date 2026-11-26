// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Curve publication (contract C2, docs/freezes/C2.md): the tiles, scalp cap and
// prim-set signature one cook publishes, and the dirty report between two of
// them. One UsdGenTilePublication per tile prim
// <description>/__usdGenRender/tile_NNNN. Built by the curve publisher
// (curvePublisher.h); hosts read UsdGenCurvePublicationOf(generation).
// Interim home until step 9 moves it to usdGenHydra/primitives/curves.
#pragma once

#include "usdGenCurves/export.h"
#include "usdGen/api/primRef.h"
#include "usdGen/compile/types.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/publisher.h"

#include "pxr/base/gf/matrix4d.h"   // also GfVec3d
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

struct UsdGenGeneration;
struct UsdGenRbfAnimationBinding;   // usdGenCurves/rbfAnimation.h

struct UsdGenTilePublication
{
    UsdGenTileId  tile = 0;
    UsdGenPrimRef prim;                       // <description>/__usdGenRender/tile_NNNN (4-digit zero pad)

    // --- topology (C2) ---
    VtIntArray    curveVertexCounts;          // basisCurves/topology/curveVertexCounts
    std::string   basis = "bspline";          // basisCurves/topology/basis (usdGen:curve:basis)
    // Contract constants, never authorable: type="cubic", wrap="pinned".
    int           refineLevel = 2;            // displayStyle/refineLevel (S-9: no M1 tumble tier)

    // --- exact-size primvar arrays (C2): points.size() == sum(curveVertexCounts) (SI-1)
    VtVec3fArray  points;                     // primvars/points, vertex
    VtFloatArray  widths;                     // primvars/widths, vertex or constant; never varying
    VtFloatArray  hairT;                      // primvars/hairT, vertex, root 0 -> tip 1
    VtFloatArray  hairId;                     // primvars/hairId, uniform float [0,1)
    VtVec2fArray  st;                         // primvars/st, uniform (root UV)
    VtVec3fArray  displayColor;               // uniform (perCurve bake) or vertex (perCV)
    VtVec3fArray  bakeColor;                  // optional primvars:<bakePrimvar> (bakeTarget=primvar)
    // Operator-emitted planes (C2), preserving their constant/uniform/vertex
    // interpolation, float/int type, and element arity.  The historical member
    // name predates publication of per-CV extra planes.
    std::vector<UsdGenPlane> extraUniform;

    // Motion profile P1 only (post-M1); empty otherwise,
    // and the primvar is BLOCKED (HdBlockDataSource) in P0/P2.
    VtVec3fArray  velocities;

    // --- per-tile data (recomputed every deforming frame; C2 extent rule) ---
    GfVec3d       extentMin, extentMax;       // extent/min, extent/max

    // --- hand-authored per tile, post-flattening (ADR §5.3) ---
    GfMatrix4d    xformMatrix;                // xform/matrix, surface world matrix
    // xform/resetXformStack = true (constant), never a payload.
    TfToken       purpose;                    // inherited by hand from the description
    TfToken       visibility;                 // inherited by hand from the description
    UsdGenPrimRef material;                   // description's Material (or the preview material)
    TfToken       materialPurpose = UsdGenTokens().allPurpose; // the empty-token container child
    UsdGenPrimRef origin;                     // the description prim; absolute outside prototypes
    UsdGenPrimRef dependency;                 // __dependencies: dependedOnPrimPath
};

/// The scalp-shadow cap: the haired part of a groom's emitting surface,
/// offset a hair's breadth along its own normal and carrying the same
/// furTauP/furTauN a strand carries, so a material can darken whatever skin
/// shader the user bound underneath by the fraction of light the hair above it
/// absorbs. Points and normals are WORLD space and the prim's xform is
/// identity with resetXformStack, exactly as a tile's is. Empty when the
/// description has no occluder or no hair over it.
///
/// Each description bakes its own cap, but the groom scene index publishes
/// one per emitting surface: caps with the same `latticeKey` are sampled on
/// the same lattice, vertex for vertex, so they are summed
/// (UsdGenMergeScalpShadows) into a single cap at
/// <owner description>/__usdGenRender/scalpShadow.
struct UsdGenScalpShadowPublication
{
    UsdGenPrimRef prim;
    VtVec3fArray  points;                     // mesh/points, world space
    VtVec3fArray  normals;                    // primvars/normals, vertex
    VtIntArray    faceVertexCounts;           // all 3
    VtIntArray    faceVertexIndices;
    // furTauP / furTauN, vertex, arity 3 -- the same planes a tile publishes,
    // so the publisher packs them through the same path. A merged cap over
    // hair of different colours also carries furCapAlbedo (vertex, arity 3).
    std::vector<UsdGenPlane> extraUniform;
    GfVec3d       extentMin{0.0}, extentMax{0.0};
    TfToken       purpose;
    TfToken       visibility;
    UsdGenPrimRef material;                   // the synthetic scalp-shadow material
    // Content identity: the scene index dirties the prim when this changes and
    // shares the previous immutable arrays when it does not.
    uint64_t      digest = 0;
    // The lattice the cap was sampled on. It depends on the emitting surface
    // alone (its triangles, their refinement, the lift and which side is
    // solid), never on the description's own voxel grid, so two descriptions
    // over the same surface at the same frame produce equal keys and
    // identical points. Zero means the cap cannot be merged.
    uint64_t      latticeKey = 0;
    // Per point: its slot in that lattice (source triangle * slots per
    // triangle + slot), so caps that dropped different triangles still line
    // up when merged.
    VtIntArray    latticeIds;
    // How far the points sit above the skin along their normals.
    float         lift = 0.f;

    bool IsEmpty() const { return points.empty() || faceVertexIndices.empty(); }
};

/// M6-only; empty in M1 (UsdGenInstance publishes instancers instead of tiles).
struct UsdGenInstancerPublication
{
    UsdGenPrimRef prim;                       // <description>/__usdGenRender/inst_NNNN
    std::vector<UsdGenPrimRef> prototypes;    // instancerTopology/prototypes (namespace children)
    std::vector<VtIntArray> instanceIndices;  // instancerTopology/instanceIndices, one per prototype
    VtBoolArray   mask;                       // instancerTopology/mask; empty == all true
};

/// Structural signature for the generation diff (03 §6.1): what changed in the
/// published prim set, independent of payload values.
struct UsdGenPrimSetSignature
{
    int           tileCount = 0;
    int           instancerCount = 0;
    std::vector<UsdGenPrimRef> prims;         // sorted
    // per prim (parallel to prims): sorted list of published primvar names
    std::vector<std::vector<std::string>> primvarNames;
    // per prim (parallel to prims): prim type string
    std::vector<std::string> primTypes;

    bool operator==(UsdGenPrimSetSignature const &rhs) const;
};

/// Diff result consumed by the imaging notice emitter (06 §5.1 rules).
struct UsdGenTileDirty
{
    UsdGenTileId tile = 0;
    UsdGenPrimRef prim;
    bool         removed = false;            // -> PrimsRemoved
    bool         added   = false;            // -> PrimsAdded
    bool         pointsDirty = false;        // -> primvars/points/primvarValue + extent/min|max
    bool         widthsDirty = false;        // -> primvars/widths/primvarValue
    bool         xformDirty  = false;        // -> xform/matrix
    std::vector<TfToken> newPrimvars;        // first appearance or new interpolation -> primvars/<name> once
    std::vector<TfToken> dirtyPrimvars;      // -> primvars/<name>/primvarValue (never with points)
};

/// What one cook publishes for a curve terminal. Immutable once built;
/// unchanged tiles share their arrays with the previous publication.
class UsdGenCurvePublication final : public UsdGenPublication
{
public:
    USDGEN_CURVES_API UsdGenCurvePublication() noexcept;
    std::vector<UsdGenTilePublication> tiles;        // sorted by tile id
    std::vector<UsdGenTilePublication> guides;       // guides/<setName> prims (§4.2)
    std::vector<UsdGenInstancerPublication> instancers; // M6
    // The scalp-shadow cap: hair shadowing the emitting surface. Empty when
    // the description has no occluder, no hair over it, or the bake is off.
    UsdGenScalpShadowPublication scalpShadow;
    UsdGenPrimSetSignature signature;
    // Identity of the usdGen:preview colours the tiles carry (0 == none), so a
    // preview edit rebuilds tiles whose geometry did not move.
    uint64_t colorDigest = 0;
    // RBF playback of a static groom over a deforming surface; null otherwise.
    std::shared_ptr<const UsdGenRbfAnimationBinding> animationBinding;
};

/// Diff of two curve publications, consumed by the imaging notice emitter
/// (06 §5.1 rules).
class UsdGenCurveDirtyReport final : public UsdGenPublication
{
public:
    USDGEN_CURVES_API UsdGenCurveDirtyReport() noexcept;
    std::vector<UsdGenTileDirty> tiles;
    bool surfaceXformDirty = false;         // description-level xform moved
};

/// UsdGenPublication::kind of the two classes above.
enum : uint8_t { kUsdGenCurvePublicationKind = 0, kUsdGenCurveDirtyReportKind = 1 };

/// A new, empty, mutable curve publication (tests and hosts that stage one).
USDGEN_CURVES_API UsdGenRef<UsdGenCurvePublication> UsdGenNewCurvePublication();

/// The curve publication of `p`; an empty one when `p` is null or not curves.
USDGEN_CURVES_API UsdGenCurvePublication const &
UsdGenCurvePublicationOf(UsdGenPublication const *p) noexcept;
inline UsdGenCurvePublication const &
UsdGenCurvePublicationOf(UsdGenRef<const UsdGenPublication> const &p) noexcept
{ return UsdGenCurvePublicationOf(p.Get()); }
UsdGenCurvePublication const &
UsdGenCurvePublicationOf(UsdGenRef<const UsdGenPublication> &&) = delete;   // would dangle
/// The generation's curve publication; empty on the device lane.
USDGEN_CURVES_API UsdGenCurvePublication const &
UsdGenCurvePublicationOf(UsdGenGeneration const &generation) noexcept;

/// The curve dirty report of `p`; an empty one when `p` is null or not curves.
USDGEN_CURVES_API UsdGenCurveDirtyReport const &
UsdGenCurveDirtyReportOf(UsdGenPublication const *p) noexcept;
inline UsdGenCurveDirtyReport const &
UsdGenCurveDirtyReportOf(UsdGenRef<const UsdGenPublication> const &p) noexcept
{ return UsdGenCurveDirtyReportOf(p.Get()); }
UsdGenCurveDirtyReport const &
UsdGenCurveDirtyReportOf(UsdGenRef<const UsdGenPublication> &&) = delete;   // would dangle

}  // namespace usdGen
