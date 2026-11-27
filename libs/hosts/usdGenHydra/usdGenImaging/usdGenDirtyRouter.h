// usdGen imaging — UsdGenDirtyRouter (03-execution-engine.md §5.1/§5.2 hop 1;
// 02-schema.md §6 the dirty-class table it is generated from).
//
// Hydra-side routing only: turns observer entries into the engine's pure
// value UsdGenPendingDirty. Routing is one hash lookup per notice entry plus
// a longest-prefix walk over that prim's entry list — no search, no cook
// (I7, S17). The bare `usdGen` container locator (an adapter resync) routes
// as graph-structural for that prim (02 §6.6 rule 2).
//
// Namespace note: like M0's plugin classes this sits in the pxr namespace so
// TfType/registry machinery can reference it; it forwards to the global
// usdGen:: engine types (03 §5.1: "the router lives in usdGenImaging").
#ifndef USDGEN_IMAGING_DIRTY_ROUTER_H
#define USDGEN_IMAGING_DIRTY_ROUTER_H

#include "usdGenImaging/api.h"
#include "usdGen/api/routing.h"
#include "usdGenUsd/lower.h"
#include "usdGen/session.h"

#include "pxr/pxr.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/usd/sdf/path.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace usdGenImaging {

/// std::hash adapter for SdfPath (26.08 ships no SdfPath::Hash).
struct SdfPathHash
{
    std::size_t operator()(SdfPath const &path) const
    {
        return std::hash<std::string>{}(path.GetText());
    }
};

class UsdGenDirtyRouter
{
public:
    /// One routing entry: a locator prefix under `usdGen/` (or a known
    /// surface locator) -> (engine node, dirty bits, optional surface).
    struct Entry
    {
        usdGen::UsdGenNodeId node;
        uint32_t bits;               // OR of usdGen::UsdGenDirty* bits
        usdGen::UsdGenSurfaceId surface = 0;
        bool surfaceScoped = false;
        // A structural row (the authored operatorOrder): the match sets
        // PendingDirty::structural for a recompile; node/bits are unused.
        bool structural = false;
    };

    /// Rebuild the routing table from an owning, graph-independent routing
    /// snapshot (UsdGenSession::Routing()): one entry list per prim path,
    /// generated from each node's parameter routing plus the 02 §6 rows,
    /// longest prefix first (a dirty on usdGen/clump matches every
    /// usdGen:clump:* leaf, but usdGen/mode does NOT match usdGen/length/mode).
    void Rebuild(usdGen::UsdGenGraphRoutingSnapshot const &snapshot);

    /// Hop-1 routing from _PrimsDirtied. O(entries): one hash lookup each,
    /// longest-prefix walk per hit. NEVER cooks (I7, S17). A usdGen locator
    /// on a routed prim that matches no row warns once, or, when \p unrouted
    /// is given, sets it instead (the caller falls back to a full recapture).
    void Route(
        HdSceneIndexObserver::DirtiedPrimEntries const &entries,
        usdGen::UsdGenPendingDirty *out, bool *unrouted = nullptr) const;
    void RouteAdded(
        HdSceneIndexObserver::AddedPrimEntries const &entries,
        usdGen::UsdGenPendingDirty *out) const;
    void RouteRemoved(
        HdSceneIndexObserver::RemovedPrimEntries const &entries,
        usdGen::UsdGenPendingDirty *out) const;

    size_t EntryCount() const noexcept { return _entryCount; }

private:
    struct PerPrim
    {
        std::vector<std::pair<HdDataSourceLocator, Entry>> prefixes; // longest first
    };

    /// True when `path` is, contains, or is contained by any prim the
    /// compiled graph references (structural-notice filter).
    bool _touchesGraphPath(SdfPath const &path) const;
    std::unordered_map<SdfPath, PerPrim, SdfPathHash> _byPrim;
    size_t _entryCount = 0;
};

}  // namespace usdGenImaging

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_IMAGING_DIRTY_ROUTER_H
