// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/publishPaths.h"
#include "usdGenUsd/graphDesc.h"

#include "usdGen/registries/engineTokens.h"
#include "usdGenUsd/walk/usdGenTokens.h"

#include <atomic>
#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenUsd {

using namespace usdGen;

namespace {

// tile_NNNN names, interned once per tile index for the process and
// published by CAS, never freed. Indices past the table intern per call.
constexpr size_t kTileNameCache = size_t(1) << 16;
std::atomic<TfToken *> gTileNames[kTileNameCache];

TfToken InternTileName(UsdGenTileId tile)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "tile_%04d", static_cast<int>(tile));
    return TfToken(buf);
}

TfToken const kMaterialLit("material_preview");
TfToken const kMaterialFlat("material_preview_flat");
TfToken const kShadingFlat("flat");
TfToken const kScalpShadow("scalpShadow");
TfToken const kMaterialScalpShadow("material_scalpShadow");

TfToken InstancerName(uint32_t index)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "inst_%04u", index);
    return TfToken(buf);
}

}  // namespace

SdfPath RenderNamespace(SdfPath const &description)
{
    return description.AppendChild(::usdGenImaging::UsdGenRenderNamespaceToken());
}

TfToken TileName(UsdGenTileId tile)
{
    size_t const index = size_t(tile);
    if (index >= kTileNameCache) return InternTileName(tile);
    if (TfToken *name = gTileNames[index].load(std::memory_order_acquire)) return *name;
    auto *fresh = new TfToken(InternTileName(tile));
    TfToken *winner = nullptr;
    if (gTileNames[index].compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

SdfPath TilePath(SdfPath const &description, UsdGenTileId tile)
{
    return RenderNamespace(description).AppendChild(TileName(tile));
}

SdfPath PreviewMaterialPath(SdfPath const &description, TfToken const &shading)
{
    return RenderNamespace(description).AppendChild(shading == kShadingFlat ? kMaterialFlat
                                                                            : kMaterialLit);
}

SdfPath PrimPath(UsdGenPrimRef const &prim, SdfPath const &node)
{
    if (node.IsEmpty()) return SdfPath();
    switch (prim.role) {
    case UsdGenPublishRole::Scene:
        return node;
    case UsdGenPublishRole::Tile:
        return TilePath(node, UsdGenTileId(prim.index));
    case UsdGenPublishRole::ScalpShadow:
        return RenderNamespace(node).AppendChild(kScalpShadow);
    case UsdGenPublishRole::ScalpShadowMaterial:
        return RenderNamespace(node).AppendChild(kMaterialScalpShadow);
    case UsdGenPublishRole::PreviewMaterial:
        return RenderNamespace(node).AppendChild(kMaterialLit);
    case UsdGenPublishRole::PreviewMaterialFlat:
        return RenderNamespace(node).AppendChild(kMaterialFlat);
    case UsdGenPublishRole::Instancer:
        return RenderNamespace(node).AppendChild(InstancerName(prim.index));
    }
    return SdfPath();
}

SdfPath PrimPath(UsdGenPrimRef const &prim, PathTable const *table)
{
    if (prim.IsEmpty() || !table) return SdfPath();
    return PrimPath(prim, table->Path(prim.node));
}

SdfPath PrimPath(UsdGenGeneration const &generation, UsdGenPrimRef const &prim)
{
    return PrimPath(prim, Paths(generation));
}

}  // namespace usdGenUsd

namespace usdGen {

UsdGenLookDesc UsdGenSchemaLook()
{
    UsdGenLookDesc look;
    look.bakeMode = ::usdGenImaging::UsdGenLookBakeModeFallbackToken();
    return look;
}

}  // namespace usdGen
