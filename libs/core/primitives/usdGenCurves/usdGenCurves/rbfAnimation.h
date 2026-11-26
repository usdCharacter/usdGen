// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_RBF_ANIMATION_H
#define USDGEN_RBF_ANIMATION_H
#include "usdGen/ops/rbfField.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/gf/vec3f.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/primRef.h"
#include <memory>
#include <vector>

namespace usdGen {
struct UsdGenRbfAnimationTile {
    UsdGenPrimRef prim;
    VtVec3fArray restPoints;
    double widthPadding = 0;
};
// Immutable tail-deform input. Generated once with the groom, then retained
// by a renderer while only the sparse surface driver positions change.
struct UsdGenRbfAnimationBinding {
    UsdGenNodeKey surface;
    std::vector<size_t> sampleIndices;
    rbf::CubicField field;
    std::vector<UsdGenRbfAnimationTile> tiles;
};
bool UsdGenRbfAnimationEnabled();
}
#endif
