// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Routing input for hosts: the graph-independent snapshot a host dirty router
// is built from (UsdGenSession::Routing(), UsdGenGraph::RoutingSnapshot()).
#pragma once
#include "usdGen/api/graphSpec.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace usdGen {

/// Owning, read-only routing input detached from a compiled graph.  The
/// imaging router may retain this value while the graph is recompiled or
/// destroyed; none of its fields point into UsdGenGraph or an operator.
/// One UsdGenPaintMap a node samples: the surface primvar whose edits must
/// re-capture the node (the brush bakes the primvar, never the map prim, so
/// the map-prim row alone would leave the snapshot stale). `surface` indexes
/// UsdGenGraphRoutingSnapshot::meshPaths, the prim that carries the primvar.
struct UsdGenPaintRoutingRef
{
    UsdGenSurfaceId surface = 0;
    TfToken primvar;
};

struct UsdGenGraphRoutingNode
{
    UsdGenNodeId id = kUsdGenInvalidNode;
    UsdGenNodeKey key;
    TfToken type;
    std::vector<TfToken> topologyParameters;
    std::vector<TfToken> valueParameters;
    std::vector<std::pair<TfToken, uint32_t>> paramRouting;
    UsdGenSurfaceId surface = 0;
    bool hasSurface = false;
    std::vector<UsdGenNodeKey> inputRefs;
    std::vector<UsdGenNodeKey> mapRefs;
    std::vector<UsdGenMapBinding> mapBindingRefs;
    std::vector<UsdGenNodeKey> geometryRefs;
    std::vector<UsdGenPaintRoutingRef> paintRefs;
};

struct UsdGenGraphRoutingSnapshot
{
    UsdGenNodeKey description;
    /// Per surface input: the bound target, and the Mesh prim whose
    /// points, topology and primvars it reads. They differ only for a
    /// GeomSubset target, whose own prim carries just indices and type (R15).
    std::vector<UsdGenNodeKey> surfaceKeys;
    std::vector<UsdGenNodeKey> meshKeys;
    std::vector<UsdGenGraphRoutingNode> nodes;
    UsdGenNodeId terminal = kUsdGenInvalidNode;
    /// The compiled spec's opaque host data (UsdGenGraphSpec::host).
    std::shared_ptr<const void> host;
};

}  // namespace usdGen
