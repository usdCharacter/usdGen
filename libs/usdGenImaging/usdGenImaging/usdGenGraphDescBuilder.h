// usdGen imaging — graph description builder (06-imaging.md §2.2/§3.10;
// 03-execution-engine.md §2.5, the S14 pull-everything rule).
//
// Builds the pure-value usdGen::UsdGenGraphDesc from Hydra data sources
// (production path, 13 §7 V2-7–V2-11).  The offline UsdStage oracle is
// deliberately declared in usdGenGraphDescBuilderStage.h: keeping this
// production interface stage-free is the B-2 staging fence.
#ifndef USDGEN_IMAGING_GRAPH_DESC_BUILDER_H
#define USDGEN_IMAGING_GRAPH_DESC_BUILDER_H

#include "usdGen/graphDesc.h"

#include "pxr/pxr.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/usd/sdf/path.h"

#include <memory>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

/// Opaque, immutable capture-side cache.  It contains operator reads and,
/// for reuseGeometry, the pooled surface, curve-set, map and
/// expression-geometry descs; description-level fields are always pulled
/// from Hydra.
class UsdGenGraphDescCaptureCache;

struct UsdGenGraphDescBuildOptions
{
    /// Hydra shutter offset for sampled values (0 = the scene's current
    /// frame). The offline UsdStage oracle interprets this as absolute time.
    double time = 0.0;
    /// Explicit opt-in only.  A first/topology capture leaves this false so
    /// S14 still pulls every mapped operator property. The caller must use
    /// the same input scene and report all intervening dirties; caches have
    /// no cross-scene identity. Disable reuse after structural/time changes
    /// or whenever continuity of the dirty stream cannot be established.
    bool reuseNodes = false;
    /// Explicit opt-in only.  Same contract as reuseNodes, but for the
    /// geometry pools.  Unlike operator params (which can read the frame
    /// procedurally), geometry flows only from Hydra data, so a pooled
    /// desc survives a frame change while its prim stays clean.
    bool reuseGeometry = false;
    std::shared_ptr<const UsdGenGraphDescCaptureCache> previousCache;
    /// Prim paths from the caller-boundary dirty packet.  A dirty path that
    /// is an ancestor of, or equal to, an operator forces that node to read;
    /// the same rule re-reads a pooled surface, curve set, map or geometry.
    SdfPathVector dirtyPrimPaths;
};

struct UsdGenGraphDescCapture
{
    usdGen::UsdGenGraphDesc desc;
    std::shared_ptr<const UsdGenGraphDescCaptureCache> cache;
    /// Prim paths whose pooled descs were reused without an upstream read:
    /// the recording caller unions these into the groom's dependencies, or
    /// their next dirty would not recapture it.
    SdfPathVector reusedInputs;
};

/// Capture a pure descriptor and its immutable read cache.  The cache
/// is usable only when options explicitly request reuse; description-level
/// fields are rebuilt on every call.
UsdGenGraphDescCapture CaptureGraphDescFromHydra(
    HdSceneIndexBase &input,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options = UsdGenGraphDescBuildOptions());

/// Walk one UsdGenDescription:
///  - the Description adapter supplies usdGen:operatorOrder, a composed
///    reverse-sibling post-order aggregate (children before their parent;
///    lower sibling before upper sibling).  It is the sole source of stack
///    topology: input edges and terminal are derived, never authored;
///  - EVERY mapped usdGen:* property of every node is pulled at least once
///    per topology generation (S14) so time-varying and mode-irrelevant
///    parameters still register dependencies;
///  - usdGen:surface targets accept Mesh and GeomSubset prims (ADR §9 R15);
///    a GeomSubset yields UsdGenSurfaceDesc::subsetFaces in PARENT-mesh face
///    indices (02 §2.20 rules 1-2);
///  - usdGen:guides / usdGen:curves / usdGen:frozen:curves yield
///    UsdGenCurveSetDesc entries (rest points at UsdTimeCode::Default(),
///    via the UsdGenCurveAPI adapter where applied);
///    usdGen:reference / usdGen:references yield Reference-role entries
///    consumed through the node's reference lane;
///  - usdGen:colliders on UsdGenCollide appends Mesh/GeomSubset targets to
///    that node's surfaces AFTER the inherited bound surface (front/root
///    semantics unchanged); their descs join the shared surface pool.
///
/// Fill-in policy: C1 schema defaults (docs/freezes/C1.md) are materialized
/// for missing properties so the engine's param partition (S14 union rule)
/// sees a complete set.
///
/// Hydra-sourced staging: operator discovery consumes the Description's
/// composed hierarchy order; parameters are read from flat relative-locator sampled
/// data sources (the adapter overlays its mapped source at the prim root,
/// so usdGen:curve:basis -> curve/basis)
/// with S14 pull-all preserved (every mapped locator pulled at least once
/// per topology generation so dependencies register; stock subtrees sharing
/// the overlaid root are pruned from the sweep). Sibling of
usdGen::UsdGenGraphDesc BuildGraphDescFromHydra(
    HdSceneIndexBase &input,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options = UsdGenGraphDescBuildOptions());

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_GRAPH_DESC_BUILDER_H
