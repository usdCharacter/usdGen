// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Graph-desc build options shared by the Hydra builder (usdGenImaging) and
// the stage builder (walk/usdGenGraphDescBuilderStage.h). Stage-free.
#ifndef USDGEN_USD_WALK_BUILD_OPTIONS_H
#define USDGEN_USD_WALK_BUILD_OPTIONS_H

#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include <memory>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

/// Opaque, immutable capture-side cache.  It deliberately contains only
/// operator reads: geometry, maps and description fields are always pulled
/// from Hydra by this first incremental slice.
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
    std::shared_ptr<const UsdGenGraphDescCaptureCache> previousCache;
    /// Prim paths from the caller-boundary dirty packet.  A dirty path that
    /// is an ancestor of, or equal to, an operator forces that node to read.
    SdfPathVector dirtyPrimPaths;
};

}  // namespace usdGenImaging

#endif  // USDGEN_USD_WALK_BUILD_OPTIONS_H
