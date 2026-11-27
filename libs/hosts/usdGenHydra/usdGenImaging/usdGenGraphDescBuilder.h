// usdGen imaging — graph description builder (06-imaging.md §2.2/§3.10;
// 03-execution-engine.md §2.5, the S14 pull-everything rule).
//
// Builds the pure-value usdGen::UsdGenGraphDesc from Hydra data sources
// (production path, 13 §7 V2-7–V2-11).  The offline UsdStage oracle is
// deliberately declared in usdGenGraphDescBuilderStage.h: keeping this
// production interface stage-free is the B-2 staging fence.
#ifndef USDGEN_IMAGING_GRAPH_DESC_BUILDER_H
#define USDGEN_IMAGING_GRAPH_DESC_BUILDER_H

#include "usdGenUsd/graphDesc.h"
#include "usdGenImaging/export.h"
#include "usdGenUsd/walk/buildOptions.h"

#include "pxr/pxr.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/usd/sdf/path.h"

#include <memory>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

struct UsdGenGraphDescCapture
{
    usdGen::UsdGenGraphDesc desc;
    std::shared_ptr<const UsdGenGraphDescCaptureCache> cache;
};

/// Capture a pure descriptor and its immutable node-read cache.  The cache
/// is usable only when options explicitly request node reuse; all other
/// descriptor sections are rebuilt on every call.
USDGEN_HYDRA_API UsdGenGraphDescCapture CaptureGraphDescFromHydra(
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
USDGEN_HYDRA_API usdGen::UsdGenGraphDesc BuildGraphDescFromHydra(
    HdSceneIndexBase &input,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options = UsdGenGraphDescBuildOptions());

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_GRAPH_DESC_BUILDER_H
