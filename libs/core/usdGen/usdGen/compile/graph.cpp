// usdGen engine — graph implementation (03-execution-engine.md §3, §5.2).
//
// The graph is the compiled, topologically sorted node list. It owns the
// per-node dirty bytes and the tile partition. All mutation happens on the
// commit thread; readers only ever observe the published generation
// (I7), never this structure.
#include "usdGen/graph.h"

#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/typeRegistry.h"
#include "usdGen/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>


namespace usdGen {

UsdGenCompiledNode &UsdGenGraph::Node(UsdGenNodeId id)
{
    if (id >= _nodes.size() || !_nodes[id]) {
        throw std::out_of_range("UsdGenGraph::Node: invalid node id");
    }
    return *_nodes[id];
}

UsdGenCompiledNode const &UsdGenGraph::Node(UsdGenNodeId id) const
{
    if (id >= _nodes.size() || !_nodes[id]) {
        throw std::out_of_range("UsdGenGraph::Node: invalid node id");
    }
    return *_nodes[id];
}

UsdGenOpNode const &UsdGenGraph::OpNode(UsdGenNodeId id) const
{
    if (id >= _nodes.size() || !_nodes[id] || !_nodes[id]->desc) {
        throw std::out_of_range("UsdGenGraph::OpNode: invalid node id");
    }
    return *_nodes[id]->desc;
}

namespace {

/// The surface role's kind and desc; null without one.
UsdGenInputDesc const *SurfaceDesc(UsdGenTypeId *kind)
{
    *kind = UsdGenInputKindWithRole(UsdGenInputRole::Surface);
    return UsdGenFindInputDesc(*kind);
}

UsdGenSpan<const UsdGenInputRecord> SurfaceRecords(UsdGenGraphSpec const &spec)
{
    UsdGenTypeId kind;
    return SurfaceDesc(&kind) ? spec.Inputs(kind) : UsdGenSpan<const UsdGenInputRecord>();
}

std::array<uint64_t, 3> RestSourceKey(UsdGenInputDesc const &input, UsdGenInputRecord const &record)
{
    std::array<uint64_t, 3> key{};
    if (input.restKey) input.restKey(record, key.data());
    return key;
}

}  // namespace

UsdGenGraphSpec const &UsdGenGraph::RestSpec() const
{
    if (!_desc) {
        if (!_restDesc) _restDesc = std::make_unique<UsdGenGraphSpec>();
        return *_restDesc;
    }
    UsdGenTypeId surfaceKind;
    UsdGenInputDesc const *const surfaceInput = SurfaceDesc(&surfaceKind);
    UsdGenSpan<const UsdGenInputRecord> const surfaces =
        surfaceInput ? _desc->Inputs(surfaceKind) : UsdGenSpan<const UsdGenInputRecord>();
    bool fresh = _restDesc && _restSource == _desc.get() &&
        _restSourceKeys.size() == surfaces.size();
    for (size_t s = 0; fresh && s < surfaces.size(); ++s)
        fresh = _restSourceKeys[s] == RestSourceKey(*surfaceInput, surfaces[s]);
    if (fresh) return *_restDesc;

    // Copy-assign into the retained view: its vectors keep their capacity.
    if (!_restDesc) _restDesc = std::make_unique<UsdGenGraphSpec>();
    UsdGenGraphSpec *rest = _restDesc.get();
    *rest = *_desc;
    UsdGenSpan<UsdGenInputRecord> const restSurfaces =
        surfaceInput ? rest->MutableInputs(surfaceKind) : UsdGenSpan<UsdGenInputRecord>();
    size_t const keep = 2 * restSurfaces.size() + 2;
    for (UsdGenInputRecord &surface : restSurfaces) {
        RestRecord const *hit = nullptr;
        for (RestRecord const &cached : _restRecords) {
            if (cached.posed == surface.data && cached.generation == surface.generation &&
                cached.rest.values.size() == surface.values.size() &&
                std::equal(surface.values.begin(), surface.values.end(), cached.rest.values.begin(),
                           [](UsdGenParamValue const &a, UsdGenParamValue const &b) {
                               return a.name == b.name && a.value == b.value;
                           })) {
                hit = &cached;
                break;
            }
        }
        if (!hit) {
            UsdGenInputRecord restRecord;
            // No authored rest: Default space is the current geometry.
            if (!surfaceInput->restVariant || !surfaceInput->restVariant(surface, &restRecord)) continue;
            if (_restRecords.size() >= keep) _restRecords.erase(_restRecords.begin());
            _restRecords.push_back(RestRecord{surface.data, surface.generation, std::move(restRecord)});
            hit = &_restRecords.back();
        }
        surface = hit->rest;
    }
    _restSourceKeys.clear();
    for (UsdGenInputRecord const &surface : surfaces)
        _restSourceKeys.push_back(RestSourceKey(*surfaceInput, surface));
    _restSource = _desc.get();
    return *_restDesc;
}

UsdGenSpan<const UsdGenResolvedReferenceValue> UsdGenGraph::ReferenceValues() const
{
    return UsdGenSpan<const UsdGenResolvedReferenceValue>(
        _referenceValues.data(), _referenceValues.size());
}

UsdGenSpan<const UsdGenResolvedMapValue> UsdGenGraph::MapValues() const
{
    return UsdGenSpan<const UsdGenResolvedMapValue>(_mapValues.data(),
                                                 _mapValues.size());
}

std::shared_ptr<const UsdGenGraphRoutingSnapshot>
UsdGenGraph::RoutingSnapshot() const
{
    auto snapshot = std::make_shared<UsdGenGraphRoutingSnapshot>();
    snapshot->terminal = _terminal;
    if (_desc) {
        snapshot->description = _desc->description;
        snapshot->host = _desc->host;
        UsdGenTypeId surfaceKind;
        UsdGenInputDesc const *const surfaceInput = SurfaceDesc(&surfaceKind);
        UsdGenSpan<const UsdGenInputRecord> const surfaces =
            surfaceInput ? _desc->Inputs(surfaceKind) : UsdGenSpan<const UsdGenInputRecord>();
        snapshot->surfaceKeys.reserve(surfaces.size());
        snapshot->meshKeys.reserve(surfaces.size());
        for (UsdGenInputRecord const &surface : surfaces) {
            snapshot->surfaceKeys.push_back(surface.key);
            snapshot->meshKeys.push_back(surfaceInput->routingKey ? surfaceInput->routingKey(surface)
                                                                  : surface.key);
        }
    }
    snapshot->nodes.reserve(_nodes.size());
    for (auto const &nodePtr : _nodes) {
        if (!nodePtr || !nodePtr->desc) continue;
        UsdGenCompiledNode const &node = *nodePtr;
        UsdGenGraphRoutingNode copy;
        copy.id = node.id;
        copy.type = node.record ? node.record->Name() : UsdGenToken();
        copy.hasSurface = node.hasSurface;
        copy.surface = node.surface;
        copy.paramRouting = node.paramRouting;
        copy.inputRefs = node.inputRefs;
        copy.mapRefs = node.mapRefs;
        copy.mapBindingRefs = node.mapBindingRefs;
        copy.geometryRefs = node.geometryRefs;
        // Paint refs: every mapRef or geometryRef naming a UsdGenPaintMap
        // becomes the (surface, primvar) whose bakes re-capture this
        // node. Both lists matter: direct map slots land in mapRefs, but
        // an expression input (the length wiring's ptex("lengthPaint"))
        // lands in geometryRefs. Linear over the desc maps (a handful per
        // description, at recompile only); a map whose paint surface is
        // not routed, or whose primvar is empty, keeps the map-prim row
        // as its only route.
        UsdGenTypeId const mapKind = UsdGenInputKindWithRole(UsdGenInputRole::Map);
        UsdGenInputDesc const *const mapInput = UsdGenFindInputDesc(mapKind);
        if (_desc && mapInput && mapInput->paintSource) {
            std::vector<UsdGenNodeKey> paintCandidates;
            paintCandidates.reserve(node.mapRefs.size() +
                                    node.geometryRefs.size());
            paintCandidates.insert(paintCandidates.end(), node.mapRefs.begin(),
                                   node.mapRefs.end());
            paintCandidates.insert(paintCandidates.end(),
                                   node.geometryRefs.begin(),
                                   node.geometryRefs.end());
            for (UsdGenNodeKey const mapRef : paintCandidates) {
                for (UsdGenInputRecord const &mapRecord : _desc->Inputs(mapKind)) {
                    if (mapRecord.key != mapRef) continue;
                    UsdGenNodeKey paintSurface;
                    UsdGenToken paintPrimvar;
                    if (!mapInput->paintSource(mapRecord, &paintSurface, &paintPrimvar)) break;
                    if (paintPrimvar.IsEmpty()) break;
                    for (size_t s = 0;
                         s < snapshot->meshKeys.size(); ++s) {
                        if (snapshot->meshKeys[s] != paintSurface)
                            continue;
                        UsdGenPaintRoutingRef ref;
                        ref.surface = static_cast<UsdGenSurfaceId>(s);
                        ref.primvar = paintPrimvar;
                        bool known = false;
                        for (auto const &prior : copy.paintRefs) {
                            if (prior.surface == ref.surface &&
                                prior.primvar == ref.primvar) {
                                known = true;
                                break;
                            }
                        }
                        if (!known) copy.paintRefs.push_back(ref);
                        break;
                    }
                    break;
                }
            }
        }
        if (node.desc) {
            copy.key = node.desc->key;
        }
        if (node.record) {
            // Routed decls: Value class vs the rest, in declaration order.
            UsdGenOpRecord const &rec = *node.record;
            for (UsdGenParamSlot s = 0; s < rec.ParamCount(); ++s) {
                UsdGenParamDecl const &decl = rec.Def().params[s];
                if (!decl.routed) continue;
                (decl.cls == UsdGenParamClass::Value ? copy.valueParameters
                                                     : copy.topologyParameters)
                    .push_back(rec.ParamName(s));
            }
        }
        snapshot->nodes.push_back(std::move(copy));
    }
    return snapshot;
}

UsdGenNodeId UsdGenGraph::NodeIdForKey(UsdGenNodeKey key) const
{
    auto it = _nodeByKey.find(key);
    return it == _nodeByKey.end() ? InvalidNode : it->second;
}

UsdGenElementSet const &UsdGenGraph::Output() const
{
    return NodeOutput(_terminal);
}

UsdGenElementSet const &UsdGenGraph::NodeOutput(UsdGenNodeId id) const
{
    if (id != InvalidNode && id < _nodes.size() && _nodes[id] && _nodes[id]->output)
        return *_nodes[id]->output;
    return EmptyOutput();
}

namespace {
/// The layout of an engine without a default primitive: no channels, no
/// partition (Repartition keeps zero chunks).
UsdGenElementLayout const kNoPrimitiveLayout{};
}  // namespace

UsdGenElementSet const &UsdGenGraph::EmptyOutput() const
{
    if (!_emptyOutput) {
        UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
        UsdGenTypeId const primitive = types.DefaultPrimitive();
        UsdGenElementLayout const *layout =
            primitive != UsdGenTypeId::Invalid ? types.Def(primitive).layout : nullptr;
        _emptyOutput = UsdGenElementSet::Empty(layout ? *layout : kNoPrimitiveLayout);
    }
    return *_emptyOutput;
}

UsdGenSpan<const UsdGenChunkRange> UsdGenGraph::Chunks(UsdGenNodeId id) const
{
    UsdGenCompiledNode const &n = Node(id);
    return UsdGenSpan<const UsdGenChunkRange>(n.chunks.data(), n.chunks.size());
}

UsdGenSpan<const UsdGenChunkGroup> UsdGenGraph::Groups() const
{
    return UsdGenSpan<const UsdGenChunkGroup>(_groups.data(), _groups.size());
}

bool UsdGenGraph::AnyDirty() const noexcept
{
    for (auto const &n : _nodes) {
        if (!n) continue;
        if (n->captureNeeded) return true;
        for (UsdGenChunkDirtyBits b : n->chunkDirty) {
            if (b != UsdGenDirtyNone) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Dirty routing (03 §5.2 hops 2-4)
// ---------------------------------------------------------------------------

void UsdGenGraph::MarkNode(UsdGenNodeId id, uint32_t bits)
{
    if (id >= _nodes.size() || !_nodes[id]) return;
    UsdGenCompiledNode &n = *_nodes[id];
    if (bits == UsdGenDirtyNone) return;

    if (bits & UsdGenDirtyCapture) n.captureNeeded = true;

    // Hop 3 (03 §5.2): every routed class bit lands in this node's chunk
    // bytes AND in every strict descendant's chunk bytes. Capture and
    // Structural are no exception: a descendant must keep its own dirty
    // state honest for AnyDirty()/recompile even though the Run drain
    // consumes capture at the owner and the session drains structural.
    // (Surface-major chunk order keeps a surface dirty a contiguous
    // range — see DirtySurface.)
    UsdGenChunkDirtyBits const byte = static_cast<UsdGenChunkDirtyBits>(bits);
    for (UsdGenChunkDirtyBits &b : n.chunkDirty) b |= byte;
    for (UsdGenNodeId d : n.descendants) {
        if (d >= _nodes.size() || !_nodes[d]) continue;
        for (UsdGenChunkDirtyBits &b : _nodes[d]->chunkDirty) b |= byte;
    }
}

void UsdGenGraph::DirtyParameter(UsdGenNodeId id, UsdGenToken const &param)
{
    if (id >= _nodes.size() || !_nodes[id]) return;
    UsdGenCompiledNode &n = *_nodes[id];
    uint32_t bits = 0;
    for (auto const &entry : n.paramRouting) {
        if (entry.first == param) { bits = entry.second; break; }
    }
    if (bits == 0) {
        // Unknown usdGen:* leaf on this prim: one-shot value-dirty is the
        // safe superset (02 §6: unknown properties are logged by the router,
        // not silently dropped).
        bits = UsdGenDirtyParameter;
    }
    MarkNode(id, bits);
}

void UsdGenGraph::DirtyValue(UsdGenNodeId id)
{
    if (id >= _nodes.size() || !_nodes[id]) return;
    MarkNode(id, UsdGenDirtyParameter);   // DirtyParameter's unrouted-name bits
}

void UsdGenGraph::DirtyCapture(UsdGenNodeId id)
{
    MarkNode(id, UsdGenDirtyCapture);
}

void UsdGenGraph::DirtyTopology(UsdGenNodeId id)
{
    MarkNode(id, UsdGenDirtyTopology | UsdGenDirtyCapture);
}

void UsdGenGraph::DirtySurface(UsdGenSurfaceId surface, uint32_t bits,
                               uint64_t generation)
{
    if (generation != UINT64_MAX && _desc) {
        UsdGenSpan<const UsdGenInputRecord> const surfaces = SurfaceRecords(*_desc);
        if (surface < surfaces.size())
            _inputVersions.UpdateSurface(surfaces[surface].key, generation);
    }
    // Hop 4: surface-major chunk order -> the chunks binding `surface` are a
    // contiguous range on every node; OR `bits` into them and into the
    // owning nodes + strict descendants.
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        UsdGenCompiledNode &n = *nPtr;
        bool any = false;
        for (size_t c = 0; c < n.chunks.size(); ++c) {
            if (n.chunks[c].domain != surface) continue;
            any = true;
            n.chunkDirty[c] |= static_cast<UsdGenChunkDirtyBits>(bits);
        }
        if (any) {
            uint32_t valueBits = bits & ~(UsdGenDirtyCapture | UsdGenDirtyStructural);
            for (UsdGenNodeId d : n.descendants) {
                if (d >= _nodes.size() || !_nodes[d]) continue;
                UsdGenCompiledNode &dn = *_nodes[d];
                for (size_t c = 0; c < dn.chunks.size(); ++c) {
                    if (dn.chunks[c].domain != surface) continue;
                    if (bits & UsdGenDirtyCapture) dn.captureNeeded = true;
                    dn.chunkDirty[c] |= static_cast<UsdGenChunkDirtyBits>(
                        valueBits ? valueBits : UsdGenDirtyParameter);
                }
            }
        }
    }
}

void UsdGenGraph::DirtyChunks(UsdGenNodeId id, UsdGenSpan<const UsdGenChunkId> chunks)
{
    if (id >= _nodes.size() || !_nodes[id]) return;
    UsdGenCompiledNode &n = *_nodes[id];
    for (UsdGenChunkId c : chunks) {
        if (c < n.chunkDirty.size()) n.chunkDirty[c] |= UsdGenDirtyParameter;
    }
    for (UsdGenNodeId d : n.descendants) {
        if (d >= _nodes.size() || !_nodes[d]) continue;
        for (UsdGenChunkId c : chunks) {
            if (c < _nodes[d]->chunkDirty.size())
                _nodes[d]->chunkDirty[c] |= UsdGenDirtyParameter;
        }
    }
}

void UsdGenGraph::DirtyMap(UsdGenNodeKey mapPrim, uint64_t generation)
{
    if (generation != UINT64_MAX)
        _inputVersions.UpdateMap(mapPrim, generation);
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        auto &n = *nPtr;
        for (UsdGenNodeKey const m : n.mapRefs)
            if (m == mapPrim) {
                // MarkNode propagates both capture and value dirt to every
                // strict descendant; map edits cannot leave a stale child
                // capture behind.
                MarkNode(n.id, UsdGenDirtyMap | UsdGenDirtyCapture);
                break;
            }
    }
}

void UsdGenGraph::DirtyInput(UsdGenNodeKey inputIdentity, uint64_t generation)
{
    if (generation != UINT64_MAX)
        _inputVersions.UpdateInput(inputIdentity, generation);
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        auto &n = *nPtr;
        for (UsdGenNodeKey const c : n.inputRefs)
            if (c == inputIdentity) {
                MarkNode(n.id, UsdGenDirtyCapture);
                break;
            }
    }
}

void UsdGenGraph::DirtyGeometry(UsdGenNodeKey prim)
{
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        for (UsdGenNodeKey const ref : nPtr->geometryRefs)
            if (ref == prim) {
                MarkNode(nPtr->id, UsdGenDirtyCapture);
                break;
            }
    }
}

// Effective set for a node: a pure passthrough node (no topo fx) presents
// its input's set — topology, item totals and offsets — so stale downstream
// totals never override a changed upstream layout; a topo fx regenerates
// elements and owns the layout again. Traced through passthrough inputs only;
// depth guards against accidental cycles in the input chain.
static UsdGenElementSet const &_EffSet(UsdGenGraph const &g,
                                       UsdGenNodeId id, int depth)
{
    UsdGenCompiledNode const &n = g.Node(id);
    // An op that follows its input until owning (kUsdGenOpFollowsInputUntilOwning)
    // declares counts it may change, but its non-owning capture only moves items.
    // Follow its input before capture too: its initially empty output must
    // not turn a ragged source into the partition's uniform fallback.
    // An actual owning capture remains an authoritative topology boundary.
    bool const nonOwningLength = n.op && n.record &&
        n.record->Has(kUsdGenOpFollowsInputUntilOwning) &&
        (!n.capture || !n.captureOutput);
    if ((n.countChanges != 0 && !nonOwningLength) ||
        n.input == kUsdGenInvalidNode || depth > 64)
        return g.NodeOutput(id);
    return _EffSet(g, n.input, depth + 1);
}

bool UsdGenGraph::Repartition(uint32_t elements, uint32_t fallbackStride)
{
    UsdGenElementLayout const &layout = EmptyOutput().Channels().Layout();
    if (!layout.partition || !layout.groups) return false;
    UsdGenPartitionParams params;
    params.chunkSize = static_cast<uint32_t>(std::max(1, _chunkSize));
    params.groupTarget = static_cast<uint32_t>(std::max(0, _tileTarget));
    UsdGenElementSetView none{};
    none.count[1] = elements;
    uint32_t const nChunks = layout.partition(elements, none, params, {});
    uint32_t const nGroups = layout.groups(nChunks, params, {});
    // 03 §1.4: tileTarget is clamped to [32, 256] for the partition.
    int const cpt = ComputeChunksPerTile(int(nChunks),
                                         std::clamp(_tileTarget, kUsdGenTileMin, kUsdGenTileMax));
    uint32_t const previousChunks = static_cast<uint32_t>(
        ComputeNumChunks(int(_partitionElements), _chunkSize));
    _groupScratch.resize(nGroups);
    layout.groups(nChunks, params, UsdGenSpan<UsdGenChunkGroup>(_groupScratch.data(), nGroups));
    bool groupsMoved = _groups.size() != nGroups;
    for (uint32_t g = 0; !groupsMoved && g < nGroups; ++g)
        groupsMoved = _groups[g].firstChunk != _groupScratch[g].firstChunk ||
                      _groups[g].chunkCount != _groupScratch[g].chunkCount;
    bool const tileLayoutChanged = previousChunks != nChunks || _chunksPerTile != cpt || groupsMoved;
    bool const changed = _partitionElements != elements || tileLayoutChanged;
    _partitionElements = elements;

    if (tileLayoutChanged) {
        _chunksPerTile = cpt;
        _groups.assign(nGroups, UsdGenChunkGroup());
        float const inf = std::numeric_limits<float>::max();
        for (uint32_t g = 0; g < nGroups; ++g) {
            _groups[g].id = _groupScratch[g].id;
            _groups[g].firstChunk = _groupScratch[g].firstChunk;
            _groups[g].chunkCount = _groupScratch[g].chunkCount;
            for (int k = 0; k < 3; ++k) {   // UsdGenRange3f's empty state
                _groups[g].extentMin[k] = inf;
                _groups[g].extentMax[k] = -inf;
            }
        }
    }

    bool layoutDiff = false;
    _rangeScratch.resize(nChunks);
    for (size_t ni = 0; ni < _nodes.size(); ++ni) {
        if (!_nodes[ni]) continue;
        UsdGenCompiledNode &n = *_nodes[ni];
        // Per-node sub-item layout: prefer the node's actual element data (its
        // effective set, possibly a pure passthrough input's) over the
        // fallback stride. Ragged sets carry their offsets verbatim (stride
        // 0, first item from the offsets).
        UsdGenNodeId const id = static_cast<UsdGenNodeId>(ni);
        UsdGenElementSetView view = _EffSet(*this, id, 0).View();
        if (!view.offsets[2] && (!view.count[1] || view.count[2] % view.count[1])) {
            view.count[1] = elements;
            view.count[2] = elements * fallbackStride;
        }
        layout.partition(elements, view, params,
                         UsdGenSpan<UsdGenChunkRange>(_rangeScratch.data(), nChunks));
        // Keep existing chunks only if the count matches and EVERY chunk
        // still has the desired element span AND sub-item span (a 10->20
        // root recook keeps 1 chunk and the same items per element, so comparing
        // item fields alone keeps a stale element span that under-emits groups when
        // growing -- or over-reads buffers when shrinking).
        bool keep = !n.chunks.empty() && n.chunks.size() == nChunks;
        for (uint32_t c = 0; keep && c < nChunks; ++c) {
            UsdGenChunkRange const &want = _rangeScratch[c];
            keep = n.chunks[c].first[0] == want.first[0] &&
                   n.chunks[c].count[0] == want.count[0] &&
                   n.chunks[c].stride[1] == want.stride[1] &&
                   n.chunks[c].first[1] == want.first[1];
        }
        if (keep) {
            // Group ids moved under a kept chunk set: refresh them.
            if (tileLayoutChanged)
                for (uint32_t c = 0; c < nChunks; ++c) n.chunks[c].group = _rangeScratch[c].group;
            continue;
        }
        n.chunks.assign(_rangeScratch.begin(), _rangeScratch.end());
        layoutDiff = true;
        n.chunkDirty.assign(nChunks, UsdGenDirtyParameter);
        n.chunkCaptured.assign(nChunks, 0);
    }
    return changed || layoutDiff;
}

void UsdGenGraph::_PropagateDescendantBits(uint32_t bits,
                                           std::vector<bool> const &)
{
    // Kept for interface symmetry with the pre-M1 scaffold; MarkNode() does
    // the actual descendant propagation.
    (void)bits;
}

}  // namespace usdGen
