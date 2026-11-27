// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The paths of the prims a groom publishes (step 7.2: moved out of the
// engine, which names them by UsdGenPrimRef). Byte-identical to the
// engine's former formulas: <description>/__usdGenRender/tile_NNNN etc.
#ifndef USDGENUSD_PUBLISH_PATHS_H
#define USDGENUSD_PUBLISH_PATHS_H

#include "usdGenUsd/lower.h"

#include "usdGen/api/primRef.h"
#include "usdGen/api/types.h"

#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"

namespace usdGenUsd {

/// <description>/__usdGenRender
PXR_NS::SdfPath RenderNamespace(PXR_NS::SdfPath const &description);
/// tile_NNNN (4-digit zero pad), interned once per tile index.
PXR_NS::TfToken TileName(usdGen::UsdGenTileId tile);
/// <description>/__usdGenRender/tile_NNNN
PXR_NS::SdfPath TilePath(PXR_NS::SdfPath const &description, usdGen::UsdGenTileId tile);
/// <description>/__usdGenRender/material_preview (lit) or material_preview_flat.
PXR_NS::SdfPath PreviewMaterialPath(PXR_NS::SdfPath const &description,
                                    PXR_NS::TfToken const &shading);
/// The path of 'prim' whose node is at 'node'.
PXR_NS::SdfPath PrimPath(usdGen::UsdGenPrimRef const &prim, PXR_NS::SdfPath const &node);
/// The path of 'prim', its node named through 'table' (empty when unknown).
PXR_NS::SdfPath PrimPath(usdGen::UsdGenPrimRef const &prim, PathTable const *table);
/// The same through the table the generation carries.
PXR_NS::SdfPath PrimPath(usdGen::UsdGenGeneration const &generation,
                         usdGen::UsdGenPrimRef const &prim);

}  // namespace usdGenUsd

#endif  // USDGENUSD_PUBLISH_PATHS_H
