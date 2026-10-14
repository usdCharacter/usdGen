#ifndef USDGEN_FUR_OCCLUSION_H
#define USDGEN_FUR_OCCLUSION_H

#include "usdGen/curveBuffer.h"

#include <array>
#include <cstdint>
#include <vector>

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

/// One world-space occluder triangle. The direction that points into the
/// solid is resolved once for the whole mesh set. The corner ids index a
/// per-bake vertex array, which is what lets the scalp cap carry smooth
/// normals: a face normal would band the shadow along every edge of the
/// scalp mesh.
struct UsdGenFurOccluderTriangle
{
    GfVec3f a, b, c, normal;
    uint32_t ia, ib, ic;
};

/// Caller-owned reuse for the world-space occluder mesh (the fan
/// triangulation plus area-weighted vertex normals and their vertices).
/// A deform timeline
/// re-cooks every frame while its emitting surfaces sit still, so the mesh
/// is identical cook to cook; the bake keys it on the occluder bytes alone
/// and skips the rebuild on a match. Like `volumeKey`, this is live in
/// exactly one cook at a time and is never shared between sessions.
struct UsdGenFurOccluderBuild
{
    uint64_t key = 0;
    bool valid = false;
    std::vector<UsdGenFurOccluderTriangle> triangles;
    // World-space vertices, carried across rebuilds for their capacity: a
    // deforming surface moves its points every cook but never their count,
    // so the steady-state rebuild overwrites these in place with no resize
    // and no re-fault. Build-only temporaries (every later stage reads the
    // triangles); sized exactly after each rebuild.
    std::vector<GfVec3f> vertices;
    std::vector<GfVec3f> vertexNormals;
    // Max edge length per triangle, filled by the same build: the k/n slits
    // in the edges, vote, and shell loops read this instead of recomputing
    // three lengths per triangle per cook. Same floats the recomputation
    // yields (max selects, never rounds); valid whenever the mesh is.
    std::vector<float> extents;
    // Memoized occluder-byte digest (the occluderKey the volume key folds
    // and the mesh carry compares): the occluder arrays are VtArray CoW
    // shares of the emitting surfaces, so equal (size, cdata) is equal
    // content and the multi-megabyte rehash is skipped. Same rule as
    // opUtil::ContentDigestCache, and holding the shares is what keeps it
    // sound: a caller's mutating subscript detaches (new cdata, a miss),
    // so in-place content changes under a live share cannot happen. The
    // digest feeds equality-only keys, never goldens.
    std::vector<UsdGenFurOccluder> digestArrays;
    uint64_t digest = 0;
    bool digestValid = false;
    // Cached fan-triangulation index triples, one entry per occluder
    // ordinal: a deforming surface moves its points every cook but never
    // its topology, so the fan expansion and its validation run once and
    // later rebuilds emit triangles straight from the triples (the
    // degenerate skip still re-evaluates per cook: it reads positions).
    // Keyed on (size, cdata) of the counts/indices shares plus the point
    // count, and the entry holds those shares (same CoW rule as digest:
    // a mutating subscript detaches, so content under a live share is
    // immutable and an address can never alias different content).
    struct TopoEntry {
        VtIntArray counts, indices;
        size_t pointsSize = 0;
        struct Tri { uint32_t a, b, c; };
        std::vector<Tri> fan;
        bool valid = false;
    };
    std::vector<TopoEntry> topo;
};

/// Caller-owned scratch for the scalp-shadow cap build: the per-chunk
/// outputs, carried across cooks so a deform timeline reuses their capacity
/// instead of re-faulting hundreds of megabytes of fresh pages every frame.
/// Never published (the merged cap arrays are still built fresh every bake),
/// and live in exactly one cook at a time like `volumeKey`.
struct UsdGenScalpShadowScratch
{
    struct Chunk {
        VtVec3fArray points, normals;
        VtFloatArray tauP, tauN;
        VtIntArray counts, indices;
    };
    std::vector<Chunk> chunks;
};

/// One splat job's contribution to the density grid, over the sub-box of
/// cells its samples touch. Lives here (rather than next to the splat) so a
/// caller-owned bake scratch can carry the sub-box storage across cooks.
struct UsdGenSplatJobDensity
{
    int lo[3] = {0, 0, 0}, dims[3] = {0, 0, 0};
    std::vector<GfVec3f> density;
};

/// Caller-owned scratch for the volume bake's temporaries: the six sweep
/// planes, the interleaved fixed-point depths, the splat density grid, the
/// per-job splat sub-boxes, and the opaque-shell masks. A deform timeline
/// re-cooks every frame at (nearly) the same grid size, so carrying these
/// reuses tens of megabytes of already-faulted storage instead of
/// re-zeroing fresh pages per cook; the sweep, interleave, and merged-mask
/// fills overwrite every element, so carried planes skip the fill entirely.
/// Never published, and live in exactly one cook at a time like `volumeKey`.
struct UsdGenFurBakeScratch
{
    std::array<std::vector<float>,6> tau;
    std::vector<uint16_t> depths;
    std::vector<GfVec3f> density;
    std::vector<float> mask;
    std::vector<std::vector<float>> chunkMasks;
    // Appended last: tools link the bake across the libusdGen boundary, so
    // new carried temporaries go at the end, keeping earlier offsets stable.
    std::vector<UsdGenSplatJobDensity> jobDensity;
};

struct UsdGenFurOcclusionParams
{
    /// Opaque blockers, normally the groom's emitting surfaces. Only the part
    /// that falls inside the groom's own bounds is voxelized: the grid is
    /// never grown to fit an occluder, so a ground plane cannot destroy the
    /// resolution of a head.
    std::vector<UsdGenFurOccluder> occluders;

    /// Optional reuse for the world-space occluder mesh (see above). Null
    /// keeps the historical behaviour of rebuilding it every bake.

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
    /// occluders' haired area, tessellated to about the voxel size, carrying
    /// HAIR-ONLY optical depth so it darkens the skin by what the hair above
    /// absorbs and not by the head shadowing itself, which the skin shader's
    /// own N.L already does.
    bool scalpShadow = false;
    /// Ceiling on the cap's triangle count; the tessellation level is the
    /// largest that fits it and still does not exceed the voxel size.
    int  scalpMaxTriangles = 200000;

    /// With a dispatcher, tiles are splatted and gathered in parallel on its
    /// arena and the occluder shell is marked in parallel slabs.
    UsdGenWorkDispatcher* dispatcher = nullptr;

    /// Caller-owned occluder-mesh reuse (see UsdGenFurOccluderBuild). The
    /// bake reads and writes it; the caller must not touch it mid-cook.
    UsdGenFurOccluderBuild* occluderCache = nullptr;

    /// Caller-owned cap-build scratch (see UsdGenScalpShadowScratch). Null
    /// keeps the historical behaviour of allocating the chunk outputs fresh
    /// every bake.
    UsdGenScalpShadowScratch* scalpScratch = nullptr;

    /// Caller-owned volume-bake scratch (see UsdGenFurBakeScratch). Null
    /// keeps the historical behaviour of allocating the sweep, density,
    /// and shell temporaries fresh every bake.
    UsdGenFurBakeScratch* bakeScratch = nullptr;
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
}
#endif
