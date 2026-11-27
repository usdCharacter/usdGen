// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// A prim the engine names without a path (step 7.2): a host node, or a prim
// the host publishes under one (its render namespace, then the role name).
// usdGenUsd/publishPaths.h composes the path.
#ifndef USDGEN_API_PRIM_REF_H
#define USDGEN_API_PRIM_REF_H

#include "usdGen/api/nodeKey.h"

#include <cstdint>

namespace usdGen {

enum class UsdGenPublishRole : uint8_t {
    Scene = 0,             // the node itself
    Tile,                  // tile_NNNN ('index' is the tile id)
    ScalpShadow,           // the scalp-shadow cap
    ScalpShadowMaterial,   // its synthetic shader
    PreviewMaterial,       // usdGen:preview shader, lit
    PreviewMaterialFlat,   // usdGen:preview shader, flat
    Instancer              // inst_NNNN (M6)
};

struct UsdGenPrimRef
{
    UsdGenNodeKey node;
    UsdGenPublishRole role = UsdGenPublishRole::Scene;
    uint32_t index = 0;

    constexpr bool IsEmpty() const noexcept { return !node.IsValid(); }
    friend constexpr bool operator==(UsdGenPrimRef const &a, UsdGenPrimRef const &b) noexcept {
        return a.node == b.node && a.role == b.role && a.index == b.index;
    }
    friend constexpr bool operator!=(UsdGenPrimRef const &a, UsdGenPrimRef const &b) noexcept {
        return !(a == b);
    }
    friend constexpr bool operator<(UsdGenPrimRef const &a, UsdGenPrimRef const &b) noexcept {
        return a.node != b.node ? a.node < b.node
             : a.role != b.role ? uint8_t(a.role) < uint8_t(b.role)
             : a.index < b.index;
    }
};

}  // namespace usdGen

#endif  // USDGEN_API_PRIM_REF_H
