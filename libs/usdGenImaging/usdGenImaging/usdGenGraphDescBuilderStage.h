// usdGen imaging — stage-sourced graph description builder (13 §7 V2-11).
//
// The staging rule (B-2) fences the UsdStage walk out of the engine-facing
// usdGenImaging TUs; this header is its designated home. The production path
// is BuildGraphDescFromHydra in usdGenGraphDescBuilder.h — this entry point
// is fenced to T0/offline/tests (unit tests, fixture authoring) and is the
// SI-12 parity oracle.
#ifndef USDGEN_IMAGING_GRAPH_DESC_BUILDER_STAGE_H
#define USDGEN_IMAGING_GRAPH_DESC_BUILDER_STAGE_H

#include "usdGenUsd/graphDesc.h"

#include "pxr/pxr.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/stage.h"

#include "usdGenImaging/usdGenGraphDescBuilder.h"  // UsdGenGraphDescBuildOptions

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

/// Stage-sourced staging, fenced to T0/offline/tests (unit tests, fixture
/// authoring) per V2-11; the production path is BuildGraphDescFromHydra.
/// UsdGenCollide's usdGen:colliders targets append to that node's surfaces
/// after the inherited bound surface (same rule as the Hydra builder).
/// Carries the V2-9a absence semantics by hand: Hydra data sources carry
/// only authored opinions, so stage reads whose fallback differs from the
/// desired absent-value must gate on HasAuthoredValueOpinion() (notably
/// `purpose`, whose "default" fallback is render-tag poison).
usdGen::UsdGenGraphDesc BuildGraphDescFromStage(
    UsdStageRefPtr const &stage,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options = UsdGenGraphDescBuildOptions());

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_GRAPH_DESC_BUILDER_STAGE_H
