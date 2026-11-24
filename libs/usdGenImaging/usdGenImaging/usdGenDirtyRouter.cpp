// usdGen imaging — UsdGenDirtyRouter implementation (06-imaging.md §3.3,
// 03 §5.1/§5.2 hop 1). Table construction from a compiled graph; routing is
// one hash lookup per notice entry plus a longest-prefix walk (I7, S17).
#include "usdGenImaging/usdGenDirtyRouter.h"

#include "usdGenImaging/atomicShared.h"
#include "usdGenImaging/primAdapter.h"
#include "usdGenImaging/usdGenTokens.h"
#include "usdGen/executionPipeline.h"

#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <map>
#include <set>

PXR_NAMESPACE_OPEN_SCOPE

namespace usdGenImaging {

namespace {

HdDataSourceLocator
_Loc(std::initializer_list<TfToken> tokens)
{
    HdDataSourceLocator loc;
    for (TfToken const &t : tokens) {
        loc = loc.Append(t);
    }
    return loc;
}

/// Locator for a routed parameter. Node paramRouting stores the STRIPPED
/// name (usdGen: removed by the builder, 02 §0.7), so re-add the prefix
/// before handing it to the shared property->locator mapping.
/// Contract §3.2: routes through the SAME pure function as the adapter
/// (LocatorForProperty), so the §3.1 debug assert below guards both paths.
/// The sibling set here is the node's own compiled param-name set: ancestor
/// status is evaluated against the names this node actually routes, which is
/// exactly the set the adapter maps for the node's prim type. NO
/// key-matching logic changes (longest-prefix + Intersects is
/// shape-agnostic).
HdDataSourceLocator
_ParamLocator(TfToken const &strippedName,
              TfTokenVector const &siblingStrippedNames)
{
    std::string s = strippedName.GetString();
    if (s.compare(0, 7, "usdGen:") != 0) {
        s = "usdGen:" + s;
    }
    // Rebuild the sibling set in full usdGen:-prefixed form for the
    // ancestor pass (contract §3.1 pseudocode takes full property names).
    TfTokenVector siblings;
    siblings.reserve(siblingStrippedNames.size());
    for (TfToken const &n : siblingStrippedNames) {
        std::string ns = n.GetString();
        if (ns.compare(0, 7, "usdGen:") != 0) {
            ns = "usdGen:" + ns;
        }
        siblings.push_back(TfToken(ns));
    }
    // Routed params are attributes (relationships route as graph edges via
    // curveRefs/mapRefs below, never through paramRouting).
    return HdDataSourceLocator(UsdGenContainerToken())
        .Append(UsdGenPrimAdapterBase::LocatorForProperty(
            TfToken(s), /*isRelationship=*/false, siblings));
}

size_t
_LocatorDepth(HdDataSourceLocator const &loc)
{
    return loc.GetElementCount();
}

// One TF_WARN per (prim, container-leaf) that routes nowhere (06 §9:
// "reported once per groom", not per frame).
void
_WarnUnknownRoute(SdfPath const &path, TfToken const &leaf)
{
    struct Warnings {
        usdGen::UsdGenExecutionRuntime runtime{8};
        std::unordered_map<SdfPath, std::set<TfToken>, SdfPathHash> seen;
        usdGen::UsdGenExecutionPipeline owner{runtime};
    };
    // Leaked on purpose, like every process-lifetime arena owner in this
    // library (see ImagingRuntime in usdGenImagingSession.cpp): a static
    // destructor here runs at DLL_PROCESS_DETACH on Windows, after ExitProcess
    // has killed the TBB workers, so a drain could never complete. A warning
    // still queued when the process exits is the only thing lost.
    static std::atomic<Warnings*> warningsOwner{nullptr};  // constant-initialized
    Warnings& warnings = ::usdGenImaging::UsdGenLazyLeaked(warningsOwner);
    // Diagnostics do not make notice routing await the warning registry.
    // The owner deduplicates immutable keys and emits each warning once.
    warnings.owner.PostCommand([path, leaf, &warnings] {
        if (warnings.seen[path].insert(leaf).second)
            TF_WARN("usdGen: dirty locator '%s' on %s matches no routed property "
                    "(compiled graph may be stale, or the property is not part "
                    "of any operator).", leaf.GetText(), path.GetText());
    });
}

}  // namespace

void
UsdGenDirtyRouter::Rebuild(usdGen::UsdGenGraph const &graph)
{
    auto snapshot = graph.RoutingSnapshot();
    Rebuild(*snapshot);
}

void
UsdGenDirtyRouter::Rebuild(usdGen::UsdGenGraphRoutingSnapshot const &snapshot)
{
    _byPrim.clear();
    _entryCount = 0;
    if (snapshot.nodes.empty()) {
        return;  // nothing compiled: route nothing, warn on demand
    }

    std::map<SdfPath, PerPrim> table;
    // Hand-built snapshots may omit meshPaths: every target is its own mesh.
    auto const meshOf = [&](size_t surface) -> SdfPath const & {
        return surface < snapshot.meshPaths.size() ? snapshot.meshPaths[surface]
                                                   : snapshot.surfacePaths[surface];
    };

    for (auto const &node : snapshot.nodes) {

        // 02 §6 value rows: every routed parameter of the node.
        // Sibling set = the node's own compiled param names (contract
        // §3.2): the ancestor pass sees exactly the routed set.
        TfTokenVector siblingNames;
        siblingNames.reserve(node.paramRouting.size());
        for (auto const &routing : node.paramRouting) {
            siblingNames.push_back(routing.first);
        }
        for (auto const &routing : node.paramRouting) {
            HdDataSourceLocator const loc =
                _ParamLocator(routing.first, siblingNames);
            // Contract §3.1 debug assert: every paramRouting name resolves
            // to a mapped absolute locator. The adapter's cached
            // Mappings(type) is authoritative — flag staleness loudly.
            // (Both the container path and this dirty path route through
            // the one pure LocatorForProperty, so this assert guards both.)
            TF_VERIFY(!loc.IsEmpty(),
                      "usdGen: routed param '%s' has no mapped locator",
                      routing.first.GetText());
            table[node.path].prefixes.emplace_back(
                loc, Entry{node.id, routing.second});
        }

        // 02 §6 surface rows: deformation of the node's bound surface and
        // the C3 source behind each curve / map ref (06 §3.6: authored curve
        // prims dirty via their own prims, routed as capture/map re-capture).
        if (node.hasSurface && node.surface < snapshot.surfacePaths.size()) {
            // A GeomSubset target (R15) reads its parent Mesh's points,
            // topology and primvars, so those rows key on the mesh; the
            // subset's own prim carries only indices and type, a recapture
            // (02 §2.20 rule 5: the adapter dirties the bare leaves, the
            // prefixed spellings are accepted too).
            SdfPath const &target = snapshot.surfacePaths[node.surface];
            SdfPath const &mesh = meshOf(node.surface);
            if (mesh != target) {
                PerPrim &subset = table[target];
                Entry const recapture{node.id, usdGen::UsdGenDirtyCapture};
                for (char const *leaf : {"indices", "type"}) {
                    subset.prefixes.emplace_back(_Loc({TfToken(leaf)}), recapture);
                    subset.prefixes.emplace_back(
                        _Loc({TfToken("geomSubset"), TfToken(leaf)}), recapture);
                }
            }
            PerPrim &pp = table[mesh];
            Entry const e{node.id, usdGen::UsdGenDirtySurfacePoints,
                          node.surface, true};
            pp.prefixes.emplace_back(
                _Loc({TfToken("primvars"), TfToken("points")}), e);
            pp.prefixes.emplace_back(
                _Loc({TfToken("primvars"), TfToken("points"),
                      TfToken("primvarValue")}),
                e);
            pp.prefixes.emplace_back(
                HdDataSourceLocator(TfToken("extComputationPrimvars")), e);

            // RestAPI normals define the stable root frame used by surface-
            // bound consumers.  They are a capture dependency rather than a
            // current-frame deformation: a Default-time normal/interpolation
            // edit must rebuild the consuming node, while primvars:normals
            // remains intentionally unrouted here.
            Entry const restNormals{node.id, usdGen::UsdGenDirtyCapture};
            pp.prefixes.emplace_back(
                _Loc({TfToken("usdGen"), TfToken("rest"),
                      TfToken("normals")}), restNormals);
            pp.prefixes.emplace_back(
                _Loc({TfToken("usdGen"), TfToken("rest"),
                      TfToken("normalsInterpolation")}), restNormals);
            // Paint density the brush bakes on the bound surface: scatter
            // captures its face means, so a density edit re-captures the
            // node (the capture digest decides the recook). A bare coarse
            // `primvars` dirty keeps its pre-existing SurfacePoints route
            // (the deeper points prefix wins); the bake always carries the
            // fine primvar locator too, so the re-capture still lands.
            Entry const paintDensity{node.id, usdGen::UsdGenDirtyCapture};
            Entry const limitTopology{node.id, usdGen::UsdGenDirtySurfaceTopo,
                                      node.surface, true};
            // A scheme/boundary/crease edit changes the limit surface even
            // when points and face indices remain identical.
            pp.prefixes.emplace_back(
                _Loc({TfToken("mesh"),TfToken("subdivisionScheme")}), limitTopology);
            pp.prefixes.emplace_back(
                _Loc({TfToken("mesh"),TfToken("subdivisionTags")}), limitTopology);
            pp.prefixes.emplace_back(
                _Loc({TfToken("mesh"),TfToken("topology")}), limitTopology);
            pp.prefixes.emplace_back(
                _Loc({TfToken("primvars"),
                      TfToken("usdGen:paint:density")}),
                paintDensity);
        }
        // C3 curves changed -> re-capture the consuming node (no curve scope
        // in UsdGenPendingDirty: the capture class carries it, 02 §6 row 5).
        for (SdfPath const &curveRef : node.curveRefs) {
            table[curveRef].prefixes.emplace_back(
                HdDataSourceLocator(),
                Entry{node.id, usdGen::UsdGenDirtyCapture});
        }
        for (SdfPath const &mapRef : node.mapRefs) {
            table[mapRef].prefixes.emplace_back(
                HdDataSourceLocator(UsdGenContainerToken()),
                Entry{node.id, usdGen::UsdGenDirtyMap});
        }
        // A paint map's primvar lives on the surface, not on the map prim:
        // the brush bake edits primvars:<name> on the mesh, so route that
        // locator to the sampling node or the re-captured snapshot never
        // cooks (length/width/clump/curl paints would do nothing). Same
        // class as graph.DirtyMap (Map | Capture): the fresh snapshot
        // re-captures the node and re-sweeps its chunks. The primvar-deep
        // locator wins over the coarse primvars SurfacePoints row; a bare
        // coarse primvars dirty keeps its pre-existing route.
        for (usdGen::UsdGenPaintRoutingRef const &paint : node.paintRefs) {
            if (paint.primvar.IsEmpty() ||
                paint.surface >= snapshot.surfacePaths.size())
                continue;
            table[meshOf(paint.surface)].prefixes.emplace_back(
                _Loc({TfToken("primvars"), paint.primvar}),
                Entry{node.id, usdGen::UsdGenDirtyMap |
                                   usdGen::UsdGenDirtyCapture});
        }
        // Geometry and maps a connected expression samples: any edit of the
        // prim re-captures the consumer, whose expression values then move.
        for (SdfPath const &geometryRef : node.geometryRefs) {
            table[geometryRef].prefixes.emplace_back(
                HdDataSourceLocator(),
                Entry{node.id, usdGen::UsdGenDirtyCapture});
        }
    }

    // usdGen:tileTarget / usdGen:curve:basis on the description (R21): the
    // pending struct has no topology-only scope, so tileTarget routes as
    // Topology on the terminal node — the engine's Repartition consumes it
    // without a digest bump (03 §5.2, types.h UsdGenDirtyTopology comment).
    {
        usdGen::UsdGenNodeId const terminal = snapshot.terminal;
        PerPrim &pp = table[snapshot.description];
        pp.prefixes.emplace_back(
            _Loc({UsdGenContainerToken(), TfToken("tileTarget")}),
            Entry{terminal, usdGen::UsdGenDirtyTopology});
        pp.prefixes.emplace_back(
            _Loc({UsdGenContainerToken(), TfToken("curve"), TfToken("basis")}),
            Entry{terminal, usdGen::UsdGenDirtyTopology});
        pp.prefixes.emplace_back(
            _Loc({UsdGenContainerToken(), TfToken("curve"), TfToken("basis")}),
            Entry{terminal, usdGen::UsdGenDirtyTopology});
        // The authored execution order (usdGen:operatorOrder): reordering
        // the chain is a recompile, not a value sweep.
        pp.prefixes.emplace_back(
            _Loc({UsdGenContainerToken(), TfToken("operatorOrder")}),
            Entry{terminal, 0u, {}, {}, true});
    }

    for (auto &primEntry : table) {
        // Longest prefix first: a dirty on usdGen/clump matches every
        // usdGen/clump/* leaf; the Universal/marker prefixes sink to the end.
        std::sort(primEntry.second.prefixes.begin(),
                  primEntry.second.prefixes.end(),
                  [](std::pair<HdDataSourceLocator, Entry> const &a,
                     std::pair<HdDataSourceLocator, Entry> const &b) {
                      return _LocatorDepth(a.first) > _LocatorDepth(b.first);
                  });
        _entryCount += primEntry.second.prefixes.size();
        _byPrim.emplace(primEntry.first, std::move(primEntry.second));
    }
}

// Built at library load, not on first use (no guarded function statics).
HdDataSourceLocator const s_containerLocator(UsdGenContainerToken());

void
UsdGenDirtyRouter::Route(
    HdSceneIndexObserver::DirtiedPrimEntries const &entries,
    usdGen::UsdGenPendingDirty *out) const
{
    HdDataSourceLocator const &s_container = s_containerLocator;

    for (auto const &entry : entries) {
        auto const it = _byPrim.find(entry.primPath);
        if (it == _byPrim.end()) {
            // Not a routed graph prim. (Scene-frame dirty on '/' is detected
            // by the scene index itself — 06 §3.9 trigger (b).)
            continue;
        }
        PerPrim const &pp = it->second;

        bool const universal = entry.dirtyLocators.IsEmpty() ||
            entry.dirtyLocators == HdDataSourceLocatorSet::UniversalSet();
        if (universal) {
            // Full-prim resync (02 §6.6 rule 2): surfaces resync as surface
            // topology; node prims resync structurally.
            for (auto const &prefix : pp.prefixes) {
                if (prefix.second.surfaceScoped) {
                    out->surfaceBits[prefix.second.surface] |=
                        usdGen::UsdGenDirtySurfaceTopo;
                    out->surfaceTopology = true;
                } else {
                    out->structural = true;
                }
            }
            continue;
        }

        for (auto const &locator : entry.dirtyLocators) {
            if (locator == s_container) {
                // Adapter resync of the whole usdGen container (02 §6.6 r2)
                // subsumes every routed leaf beneath it.
                out->structural = true;
                continue;
            }
            bool matched = false;
            for (auto const &prefix : pp.prefixes) {
                // Intersects: locator is under the table prefix, equal to
                // it, or COARSER than it (a container resync dirties every
                // routed leaf beneath). Universal entries store the empty
                // locator, which is a prefix of everything.
                if (locator.Intersects(prefix.first)) {
                    if (prefix.second.structural) {
                        out->structural = true;
                    } else if (prefix.second.surfaceScoped) {
                        out->surfaceBits[prefix.second.surface] |=
                            prefix.second.bits;
                    } else {
                        out->nodeBits[prefix.second.node] |= prefix.second.bits;
                    }
                    matched = true;
                    break;  // longest prefix wins (sorted in Rebuild)
                }
            }
            if (matched) {
                continue;
            }
            if (locator.GetFirstElement() == UsdGenContainerToken()) {
                _WarnUnknownRoute(entry.primPath, locator.GetLastElement());
            }
        }
    }
}


void
UsdGenDirtyRouter::RouteAdded(
    HdSceneIndexObserver::AddedPrimEntries const &entries,
    usdGen::UsdGenPendingDirty *out) const
{
    for (auto const &entry : entries) {
        if (_touchesGraphPath(entry.primPath)) {
            out->structural = true;
        }
    }
}

void
UsdGenDirtyRouter::RouteRemoved(
    HdSceneIndexObserver::RemovedPrimEntries const &entries,
    usdGen::UsdGenPendingDirty *out) const
{
    for (auto const &entry : entries) {
        if (_touchesGraphPath(entry.primPath)) {
            out->structural = true;
        }
    }
}

bool
UsdGenDirtyRouter::_touchesGraphPath(SdfPath const &path) const
{
    for (auto const &primEntry : _byPrim) {
        SdfPath const &key = primEntry.first;
        if (path == key || key.HasPrefix(path) || path.HasPrefix(key)) {
            return true;
        }
    }
    return false;
}

}  // namespace usdGenImaging

PXR_NAMESPACE_CLOSE_SCOPE
