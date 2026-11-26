// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenGraphDesc (paths) -> UsdGenGraphSpec (node keys), and the host-side
// conveniences around the engine calls that take a spec (step 7.2).
#ifndef USDGENUSD_LOWER_H
#define USDGENUSD_LOWER_H

#include "usdGenUsd/graphDesc.h"
#include "usdGenUsd/nodeKeyMap.h"

#include "usdGen/api/graphSpec.h"
#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGen/exec/session.h"

#include "pxr/usd/sdf/path.h"

#include <memory>
#include <utility>
#include <vector>

namespace usdGenUsd {

/// The key -> path table of one lowered spec. Immutable once made; a spec
/// carries it as UsdGenGraphSpec::host, and the engine copies that into the
/// generations and routing snapshots it makes from the spec.
class PathTable {
public:
    explicit PathTable(std::vector<std::pair<usdGen::UsdGenNodeKey, PXR_NS::SdfPath>> entries);
    /// The path of 'key', else the empty path.
    PXR_NS::SdfPath const &Path(usdGen::UsdGenNodeKey key) const;
    /// The key of 'path', else invalid.
    usdGen::UsdGenNodeKey Key(PXR_NS::SdfPath const &path) const;
    /// Keys of 'prim' and every path under it, in path order.
    void KeysUnder(PXR_NS::SdfPath const &prim, std::vector<usdGen::UsdGenNodeKey> *out) const;
    size_t Size() const noexcept { return _byKey.size(); }
    /// Every (key, path), sorted by key.
    std::vector<std::pair<usdGen::UsdGenNodeKey, PXR_NS::SdfPath>> const &Entries() const noexcept
    {
        return _byKey;
    }

private:
    std::vector<std::pair<usdGen::UsdGenNodeKey, PXR_NS::SdfPath>> _byKey;   // sorted by key
    std::vector<std::pair<PXR_NS::SdfPath, usdGen::UsdGenNodeKey>> _byPath;  // sorted by path
};

/// Keys paths for a hand-built engine value (a routing snapshot, a cache key)
/// and remembers them, so Table() can serve as that value's host data.
class PathRecorder {
public:
    usdGen::UsdGenNodeKey operator()(PXR_NS::SdfPath const &path);
    std::shared_ptr<const PathTable> Table() const;

private:
    NodeKeyMap _map;
    std::vector<std::pair<usdGen::UsdGenNodeKey, PXR_NS::SdfPath>> _entries;
};

/// UsdGenStackParams over host types: owns lowered copies of the graph and
/// node it views (tests and tools that drive one op by hand).
class LoweredParams {
public:
    /// Resolved slots point into the host node's params, so in-place value
    /// edits of 'node' show through the view (as with UsdGenStackParams).
    LoweredParams(usdGen::UsdGenGraphDesc const &desc, usdGen::UsdGenNodeDesc const &node,
                  usdGen::UsdGenOpRecord const &record);
    LoweredParams(LoweredParams const &) = delete;
    LoweredParams &operator=(LoweredParams const &) = delete;
    usdGen::UsdGenParamView const &View() const noexcept { return _view; }
    usdGen::UsdGenGraphSpec const &Spec() const noexcept { return _spec; }
    usdGen::UsdGenOpNode const &Node() const noexcept { return _node; }

private:
    usdGen::UsdGenGraphSpec _spec;
    usdGen::UsdGenOpNode _node;
    usdGen::UsdGenResolvedParam _slots[usdGen::kUsdGenMaxParamsPerOp];
    usdGen::UsdGenParamView _view;
};

/// Lowers 'desc', keying its paths through 'map'. The spec's labels are the
/// path texts and its host data is the PathTable of exactly these keys.
usdGen::UsdGenGraphSpec Lower(usdGen::UsdGenGraphDesc const &desc, NodeKeyMap &map);
/// Lowers with a fresh map (keys are then KeyOf(path)).
usdGen::UsdGenGraphSpec Lower(usdGen::UsdGenGraphDesc const &desc);
std::shared_ptr<const usdGen::UsdGenGraphSpec> LowerShared(
    usdGen::UsdGenGraphDesc const &desc, NodeKeyMap *map = nullptr);

/// One input or node on its own, keyed by KeyOf (as a fresh map keys it):
/// for tests and tools that drive an engine function with a single item.
usdGen::UsdGenOpNode Lower(usdGen::UsdGenNodeDesc const &node);
usdGen::UsdGenCurveSetInput Lower(usdGen::UsdGenCurveSetDesc const &curves);
usdGen::UsdGenSurfaceInput Lower(usdGen::UsdGenSurfaceDesc const &surface);
usdGen::UsdGenMapInput Lower(usdGen::UsdGenMapDesc const &map);
usdGen::UsdGenGeometryInput Lower(usdGen::UsdGenGeometryDesc const &geometry);
usdGen::UsdGenExpressionInput Lower(usdGen::UsdGenExpressionDesc const &expression);
usdGen::UsdGenParamBinding Lower(usdGen::UsdGenExpressionBinding const &binding);

/// The address of a temporary, valid to the end of the full expression: a
/// lowered host item passed where an engine call takes a pointer.
template <class T>
T const *Addr(T const &value) noexcept { return &value; }

/// The key a map gives 'path' when no other path collides with it.
usdGen::UsdGenNodeKey KeyOf(PXR_NS::SdfPath const &path);

/// The table a spec, generation or routing snapshot carries, or null.
PathTable const *Paths(usdGen::UsdGenGraphSpec const &spec);
PathTable const *Paths(usdGen::UsdGenGeneration const &generation);
PathTable const *Paths(usdGen::UsdGenGraphRoutingSnapshot const &routing);
PathTable const *Paths(std::shared_ptr<const void> const &host);
/// Path of 'key' through 'table', else the empty path.
PXR_NS::SdfPath PathOf(PathTable const *table, usdGen::UsdGenNodeKey key);
/// Key of 'path' through 'table'; KeyOf(path) when the table lacks it.
usdGen::UsdGenNodeKey KeyFor(PathTable const *table, PXR_NS::SdfPath const &path);

/// Lower + UsdGenCompiler::Compile / Recompile.
usdGen::UsdGenCompileResult Compile(usdGen::UsdGenCompiler &compiler,
                                    usdGen::UsdGenGraphDesc const &desc,
                                    usdGen::UsdGenGraph *graph, NodeKeyMap *map = nullptr);
usdGen::UsdGenCompileResult Recompile(usdGen::UsdGenCompiler &compiler,
                                      usdGen::UsdGenGraphDesc const &desc,
                                      usdGen::UsdGenGraph *graph, NodeKeyMap *map = nullptr);
usdGen::UsdGenCompileResult CompileInjectedDevice(
    usdGen::UsdGenCompiler &compiler, usdGen::UsdGenGraphDesc const &desc,
    usdGen::UsdGenGraph *graph, usdGen::UsdGenCompiler::DevicePlanCompiler const &device,
    std::shared_ptr<const usdGen::UsdGenExecutionPlanHandle> *plan, NodeKeyMap *map = nullptr);

/// Lower + UsdGenSession::SetGraphSpec / PostGraphSpec.
void SetGraphDesc(usdGen::UsdGenSession &session, usdGen::UsdGenGraphDesc const &desc,
                  NodeKeyMap *map = nullptr);
bool PostGraphDesc(usdGen::UsdGenSession &session, usdGen::UsdGenGraphDesc const &desc,
                   NodeKeyMap *map = nullptr);

/// The node of 'path' in a compiled graph or a session graph snapshot.
usdGen::UsdGenNodeId NodeIdForPath(usdGen::UsdGenGraph const &graph, PXR_NS::SdfPath const &path);
usdGen::UsdGenNodeId NodeIdForPath(usdGen::UsdGenSessionGraphSnapshot const &graph,
                                   PXR_NS::SdfPath const &path);
/// A graph's node path (its key through the graph's table).
PXR_NS::SdfPath NodePath(usdGen::UsdGenGraph const &graph, usdGen::UsdGenNodeId id);

/// Dirties every node reading geometry at or under 'prim' (UsdGenGraph::DirtyGeometry).
void DirtyGeometry(usdGen::UsdGenGraph &graph, PXR_NS::SdfPath const &prim);
void DirtyInput(usdGen::UsdGenGraph &graph, PXR_NS::SdfPath const &input,
                uint64_t generation = UINT64_MAX);
void DirtyMap(usdGen::UsdGenGraph &graph, PXR_NS::SdfPath const &map,
              uint64_t generation = UINT64_MAX);

}  // namespace usdGenUsd

#endif  // USDGENUSD_LOWER_H
