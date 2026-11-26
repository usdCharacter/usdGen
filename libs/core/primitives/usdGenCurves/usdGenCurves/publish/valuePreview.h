// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGen value preview colours (usdGen:preview:*, 07 §7.3): how a previewed
// value (preview/valuePreview.h evaluates it) maps to a strand's colour.
#ifndef USDGEN_VALUE_PREVIEW_H
#define USDGEN_VALUE_PREVIEW_H

#include "usdGen/api/primRef.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Linear colour of one previewed value. The heat, viridis and gray maps and
/// the ids palette are defined in sRGB and converted, so a mid-range value
/// looks mid-range on screen; "rgb" shows the normalised components as a
/// linear colour, so an expression that computes a colour shows that colour.
GfVec3f UsdGenPreviewColor(TfToken const &colorMap, GfVec2f const &range,
                           double const *values, unsigned components);

/// The colour of a strand the source has no value for.
GfVec3f UsdGenPreviewMissingColor();

/// The synthetic material a previewing description's tiles bind:
/// <description>/__usdGenRender/material_preview (lit) or
/// material_preview_flat (served by UsdGenGroomSceneIndex; usdGenUsd names it).
UsdGenPublishRole UsdGenPreviewMaterialRole(TfToken const &shading);

} // namespace usdGen

#endif
