#ifndef USDGEN_FUR_OCCLUSION_H
#define USDGEN_FUR_OCCLUSION_H

#include "usdGen/curveBuffer.h"

namespace usdGen {
class UsdGenWorkDispatcher;

/// Opaque geometry injected into the fur density volume, a host renderer's
/// `r.HairStrands.Voxelization.InjectOpaqueDepth`. Without it light leaks
/// through the scalp and hair on the unlit side of a head stays lit. `points`
/// are in the mesh's own object space; `worldMatrix` places them in the same
/// world space as the tiles' transformed points. Polygons are fan-triangulated
/// and their shell is marked `opaqueBiasVoxels` INSIDE the surface so roots
/// sitting on it are not self-shadowed.
struct UsdGenFurOccluder
{
    VtVec3fArray points;
    VtIntArray   faceVertexCounts;
    VtIntArray   faceVertexIndices;
    GfMatrix4d   worldMatrix{1.0};
};

struct UsdGenFurOcclusionParams
{
    /// Opaque blockers, normally the groom's emitting surfaces. Only the part
    /// that falls inside the groom's own bounds is voxelized: the grid is
    /// never grown to fit an occluder, so a ground plane cannot destroy the
    /// resolution of a head.
    std::vector<UsdGenFurOccluder> occluders;

    /// Target voxel edge in world units when `resolution` is 0. A host renderer uses
    /// 0.3 cm (`Voxelization.Virtual.VoxelWorldSize`) and this scene
    /// convention is centimetres. The bake never chooses a voxel coarser than
    /// the historical 48-cube fitted to the longest axis, and never a grid
    /// dimension above `maxDimension`.
    float voxelSize = 0.3f;
    int   maxDimension = 128;

    /// > 0 overrides the derived grid with a cube of this many voxels fitted
    /// to the longest axis (the pre-2026-09 behaviour and the bake CLI's
    /// explicit argument). Clamped to [8, maxDimension].
    int   resolution = 0;

    /// a host renderer's `InjectOpaque.BiasCount` / `MarkCount`, in voxels: the opaque
    /// shell starts this far below the surface and is this thick.
    int   opaqueBiasVoxels = 2;
    int   opaqueMarkVoxels = 4;

    /// Also build the scalp-shadow cap (UsdGenScalpShadowPublication): the
    /// occluders' haired area, tessellated on a lattice fixed by the occluders
    /// alone, carrying HAIR-ONLY optical depth so it darkens the skin by what
    /// the hair above absorbs and not by the head shadowing itself, which the
    /// skin shader's own N.L already does.
    bool scalpShadow = false;
    /// Ceiling on the cap's triangle count; the tessellation level is the
    /// largest that fits it and still does not refine past an edge of 1/128
    /// of the occluders' bounding diagonal.
    int  scalpMaxTriangles = 200000;

    /// Also publish `furDensity` (vertex, float, arity 1) on every tile: the
    /// LOCAL hair density at each CV, for a hair shading model that darkens
    /// ambient light by local hair density. It is the per-voxel optical depth
    /// of the hair-only density grid -- the expected fibre crossings of a ray
    /// traversing one voxel along an axis, averaged over x, y and z --
    /// trilinearly gathered at the CV. Units are crossings per voxel, so the
    /// value scales with the voxel edge (`voxelSize`). HAIR ONLY: the opaque
    /// occluder shell is not in it, so roots resting on a surface do not read
    /// as saturated. Clamped at UsdGenFurTauClamp. Off by default, so a groom
    /// whose shading does not ask for it carries no extra primvar.
    bool localDensity = false;

    /// With a dispatcher, tiles are splatted and gathered in parallel on its
    /// arena and the occluder shell is marked in parallel slabs.
    UsdGenWorkDispatcher* dispatcher = nullptr;
};

/// Bakes geometry-derived directional optical depth for a whole groom.
///
/// SEMANTICS (matched to a host renderer's `FHairTransmittanceMask::HairCount`, which
/// is what `ComputeDualScatteringTerms` consumes):
///   furTauP[k] / furTauN[k] = the EXPECTED NUMBER OF FIBRE CROSSINGS for a
///   ray leaving the CV toward +axis[k] / -axis[k] and running to the edge of
///   the volume -- the line integral of projected fibre area per unit volume.
/// A single fibre of width w crossing the ray's voxel column contributes w/h,
/// not 1, exactly as a voxel density volume does: these are expectations over
/// the voxel's cross-section, so a dense groom reads tens of crossings and an
/// isolated strand reads a few hundredths.
///
/// The receiver's OWN contribution IS included (half of its own voxel, the
/// correct discretisation of an integral that starts at the CV). That matches
/// a host renderer, whose ray march also starts inside the shaded strand's voxel, so
/// the consumer must apply a host renderer's shift, `HairCount = max(0, tau - 1)`,
/// before `Tf = pow(A_front, HairCount)`.
///
/// RECONSTRUCTION. For a world-space direction L the consumer must use the
/// normalised cosine-power-4 blend of the six stored depths:
///   vec3 d = mix(furTauN, furTauP, step(vec3(0), L));
///   vec3 w = L*L; w *= w;                       // cos^4, per axis
///   float tau = dot(w, d) / (w.x + w.y + w.z);  // >= 1/9 for a unit L
/// Power 4 beats the older power-2 blend (`dot(L*L, d)`, which needed no
/// normalisation) by ~17% RMS on transmittance for the same data; see
/// docs/storm-fur.md for the measured table. Values saturate at 64.
///
/// Opaque occluders are folded into the same hair count as a saturated shell,
/// so the consumer needs no separate visibility channel: one crossing of the
/// shell already reaches the clamp.
///
/// All tiles of a description participate in one volume, so tile boundaries do
/// not become lighting boundaries. There is no light or camera dependency.
///
/// With `params.localDensity`, every tile also carries `furDensity` (vertex
/// float, one value per CV): the hair-only per-voxel optical depth around the
/// CV, averaged over the three axes. It is a non-directional measure of how
/// crowded the CV's neighbourhood is, not a path integral; see the parameter.
/// Without it any `furDensity` plane already on a tile is removed.
///
/// Returns false when nothing changed and the previous COW planes were shared.
/// `previous` alone cannot see a parameter or occluder edit, so pass
/// `volumeKey`: caller-owned state, 0 on the first call, written back every
/// call. Without it, occluder/resolution edits are silently reused.
///
/// `scalpShadow` receives the cap when `params.scalpShadow` is set AND the bake
/// rebuilds. On the reuse path it is left untouched, so a caller that wants the
/// COW share seeds it from the previous generation before calling. Its depths
/// are HAIR ONLY and carry no self shift: the receiver is skin, not a fibre, so
/// the consumer uses the count as it stands rather than a host renderer's
/// `max(0, HairCount - 1)`.
///
/// Throws std::invalid_argument for malformed or non-finite geometry.
bool UsdGenBuildFurOcclusion(
    std::vector<UsdGenTilePublication>* tiles,
    std::vector<UsdGenTilePublication> const* previous = nullptr,
    UsdGenFurOcclusionParams const& params = UsdGenFurOcclusionParams(),
    uint64_t* volumeKey = nullptr,
    UsdGenScalpShadowPublication* scalpShadow = nullptr);

/// The saturation ceiling of the published depths.
inline constexpr float UsdGenFurTauClamp = 64.0f;

/// One description's cap and the root albedo of its hair, for
/// UsdGenMergeScalpShadows.
struct UsdGenScalpShadowContribution
{
    UsdGenScalpShadowPublication const* cap = nullptr;
    GfVec3f albedo{0.f};
};

/// Sums caps baked over the same emitting surface into one cap.
///
/// Every cap must carry the same nonzero `latticeKey`; parts that do not are
/// ignored. The result keeps every triangle any part kept, and each point's
/// depths are the sums of the parts' depths there (a part that dropped the
/// point contributes nothing, which is what dropping meant: no hair over it).
/// Summing is exact for the skin: the light that reaches it has crossed the
/// hair of every description in turn.
///
/// When the parts' albedos differ, the result also carries `furCapAlbedo`
/// (vertex, float3): the parts' albedos weighted by how much of each one's
/// hair the point sees. Otherwise the material's baseColor stands for all.
///
/// The identity fields (primPath, purpose, visibility, materialPath) come from
/// the first contributing part. Returns an empty cap when no part contributes.
UsdGenScalpShadowPublication UsdGenMergeScalpShadows(
    std::vector<UsdGenScalpShadowContribution> const& parts);
}
#endif
