// usdGen engine — graph implementation (03-execution-engine.md §3, §5.2).
//
// The graph is the compiled, topologically sorted node list. It owns the
// per-node dirty bytes and the tile partition. All mutation happens on the
// commit thread; readers only ever observe the published generation
// (I7), never this structure.
#include "usdGen/graph.h"

#include "usdGen/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

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

UsdGenNodeDesc const &UsdGenGraph::NodeDesc(UsdGenNodeId id) const
{
    if (id >= _nodes.size() || !_nodes[id] || !_nodes[id]->desc) {
        throw std::out_of_range("UsdGenGraph::NodeDesc: invalid node id");
    }
    return *_nodes[id]->desc;
}

namespace {

// Word-at-a-time FNV-1a over a surface's Default-space content: topology,
// rest points and normals, uv, subset and world matrix. The posed points,
// shutter samples and velocities are deliberately not terms.
class RestContentHash
{
public:
    void Bytes(void const *data, size_t size)
    {
        auto const *bytes = static_cast<unsigned char const *>(data);
        size_t i = 0;
        for (; i + 8 <= size; i += 8) {
            uint64_t word;
            std::memcpy(&word, bytes + i, 8);
            Mix(word);
        }
        uint64_t tail = 0;
        if (i < size) std::memcpy(&tail, bytes + i, size - i);
        Mix(tail);
        Mix(uint64_t(size));
    }
    template <class T> void Array(VtArray<T> const &a)
    {
        Bytes(a.empty() ? nullptr : a.cdata(), a.size() * sizeof(T));
    }
    void Mix(uint64_t word) { _h ^= word; _h *= 0x100000001b3ULL; }
    uint64_t Value() const { return _h ? _h : 1; }

private:
    uint64_t _h = 1469598103934665603ULL;
};

uint64_t RestSurfaceGeneration(UsdGenSurfaceDesc const &surface)
{
    RestContentHash h;
    h.Mix(0x5265737453726675ULL);   // distinct from any posed generation domain
    h.Array(surface.faceVertexCounts);
    h.Array(surface.faceVertexIndices);
    h.Array(surface.restPoints);
    h.Array(surface.restNormals);
    h.Mix(uint64_t(surface.restNormalDomain));
    h.Array(surface.uv);
    h.Array(surface.subsetFaces);
    h.Mix(surface.isSubset ? 1u : 0u);
    h.Bytes(surface.worldMatrix.GetArray(), 16 * sizeof(double));
    return h.Value();
}

std::array<uint64_t, 3> RestSourceKey(UsdGenSurfaceDesc const &surface)
{
    return {surface.surfaceGeneration,
            uint64_t(reinterpret_cast<uintptr_t>(surface.points.cdata())),
            uint64_t(reinterpret_cast<uintptr_t>(surface.restPoints.cdata()))};
}

}  // namespace

UsdGenGraphDesc const &UsdGenGraph::RestDesc() const
{
    if (!_desc) {
        if (!_restDesc) _restDesc = std::make_unique<UsdGenGraphDesc>();
        return *_restDesc;
    }
    bool fresh = _restDesc && _restSource == _desc.get() &&
        _restSourceKeys.size() == _desc->surfaces.size();
    for (size_t s = 0; fresh && s < _desc->surfaces.size(); ++s)
        fresh = _restSourceKeys[s] == RestSourceKey(_desc->surfaces[s]);
    if (fresh) return *_restDesc;

    auto rest = std::make_unique<UsdGenGraphDesc>(*_desc);
    for (UsdGenSurfaceDesc &surface : rest->surfaces) {
        if (surface.restFromCurrentPoints || surface.restPoints.empty() ||
            surface.restPoints.size() != surface.points.size())
            continue;   // no authored rest: Default space is the current geometry
        surface.points = surface.restPoints;
        if (!surface.samples.empty()) {
            surface.samples.resize(1);
            surface.samples[0].points = surface.restPoints;
        }
        surface.velocities = VtVec3fArray();
        surface.surfaceGeneration = RestSurfaceGeneration(surface);
    }
    _restSourceKeys.clear();
    for (UsdGenSurfaceDesc const &surface : _desc->surfaces)
        _restSourceKeys.push_back(RestSourceKey(surface));
    _restSource = _desc.get();
    _restDesc = std::move(rest);
    return *_restDesc;
}

TfSpan<const UsdGenResolvedReferenceValue> UsdGenGraph::ReferenceValues() const
{
    return TfSpan<const UsdGenResolvedReferenceValue>(
        _referenceValues.data(), _referenceValues.size());
}

TfSpan<const UsdGenResolvedMapValue> UsdGenGraph::MapValues() const
{
    return TfSpan<const UsdGenResolvedMapValue>(_mapValues.data(),
                                                 _mapValues.size());
}

std::shared_ptr<const UsdGenGraphRoutingSnapshot>
UsdGenGraph::RoutingSnapshot() const
{
    auto snapshot = std::make_shared<UsdGenGraphRoutingSnapshot>();
    snapshot->terminal = _terminal;
    if (_desc) {
        snapshot->description = _desc->description;
        snapshot->surfacePaths.reserve(_desc->surfaces.size());
        snapshot->meshPaths.reserve(_desc->surfaces.size());
        for (auto const &surface : _desc->surfaces) {
            snapshot->surfacePaths.push_back(surface.path);
            snapshot->meshPaths.push_back(UsdGenSurfaceMeshPath(surface));
        }
    }
    snapshot->nodes.reserve(_nodes.size());
    for (auto const &nodePtr : _nodes) {
        if (!nodePtr || !nodePtr->desc) continue;
        UsdGenCompiledNode const &node = *nodePtr;
        UsdGenGraphRoutingNode copy;
        copy.id = node.id;
        copy.type = node.type;
        copy.hasSurface = node.hasSurface;
        copy.surface = node.surface;
        copy.paramRouting = node.paramRouting;
        copy.curveRefs = node.curveRefs;
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
        if (_desc) {
            std::vector<SdfPath> paintCandidates;
            paintCandidates.reserve(node.mapRefs.size() +
                                    node.geometryRefs.size());
            paintCandidates.insert(paintCandidates.end(), node.mapRefs.begin(),
                                   node.mapRefs.end());
            paintCandidates.insert(paintCandidates.end(),
                                   node.geometryRefs.begin(),
                                   node.geometryRefs.end());
            for (SdfPath const &mapRef : paintCandidates) {
                for (auto const &map : _desc->maps) {
                    if (map.path != mapRef) continue;
                    if (map.type != TfToken("UsdGenPaintMap")) break;
                    if (map.paintPrimvar.IsEmpty()) break;
                    for (size_t s = 0;
                         s < snapshot->meshPaths.size(); ++s) {
                        if (snapshot->meshPaths[s] != map.paintSurface)
                            continue;
                        UsdGenPaintRoutingRef ref;
                        ref.surface = static_cast<UsdGenSurfaceId>(s);
                        ref.primvar = map.paintPrimvar;
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
            copy.path = node.desc->path;
        }
        if (node.op) {
            auto topology = node.op->TopologyParameters();
            copy.topologyParameters.assign(topology.begin(), topology.end());
            auto values = node.op->ValueParameters();
            copy.valueParameters.assign(values.begin(), values.end());
        }
        snapshot->nodes.push_back(std::move(copy));
    }
    return snapshot;
}

UsdGenNodeId UsdGenGraph::NodeIdForPath(SdfPath const &path) const
{
    auto it = _nodeByPath.find(path);
    return it == _nodeByPath.end() ? InvalidNode : it->second;
}

UsdGenCurveBuffer const &UsdGenGraph::Output() const
{
    if (_terminal == InvalidNode || _nodes.empty() || !_nodes[_terminal]) {
        return _emptyOutput;
    }
    return _nodes[_terminal]->buffer;
}

TfSpan<const UsdGenChunkDesc> UsdGenGraph::Chunks(UsdGenNodeId id) const
{
    UsdGenCompiledNode const &n = Node(id);
    return TfSpan<const UsdGenChunkDesc>(n.chunks.data(), n.chunks.size());
}

TfSpan<const UsdGenTileView> UsdGenGraph::Tiles() const
{
    return TfSpan<const UsdGenTileView>(_tiles.data(), _tiles.size());
}

bool UsdGenGraph::AnyDirty() const noexcept
{
    for (auto const &n : _nodes) {
        if (!n) continue;
        if (n->captureNeeded) return true;
        for (uint8_t b : n->chunkDirty) {
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
    uint8_t const byte = static_cast<uint8_t>(bits);
    for (uint8_t &b : n.chunkDirty) b |= byte;
    for (UsdGenNodeId d : n.descendants) {
        if (d >= _nodes.size() || !_nodes[d]) continue;
        for (uint8_t &b : _nodes[d]->chunkDirty) b |= byte;
    }
}

void UsdGenGraph::DirtyParameter(UsdGenNodeId id, TfToken const &param)
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
    if (generation != UINT64_MAX && _desc &&
        surface < _desc->surfaces.size())
        _inputVersions.UpdateSurface(
            _desc->surfaces[surface].path, generation);
    // Hop 4: surface-major chunk order -> the chunks binding `surface` are a
    // contiguous range on every node; OR `bits` into them and into the
    // owning nodes + strict descendants.
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        UsdGenCompiledNode &n = *nPtr;
        bool any = false;
        for (size_t c = 0; c < n.chunks.size(); ++c) {
            if (n.chunks[c].surface != surface) continue;
            any = true;
            n.chunkDirty[c] |= static_cast<uint8_t>(bits);
        }
        if (any) {
            uint32_t valueBits = bits & ~(UsdGenDirtyCapture | UsdGenDirtyStructural);
            for (UsdGenNodeId d : n.descendants) {
                if (d >= _nodes.size() || !_nodes[d]) continue;
                UsdGenCompiledNode &dn = *_nodes[d];
                for (size_t c = 0; c < dn.chunks.size(); ++c) {
                    if (dn.chunks[c].surface != surface) continue;
                    if (bits & UsdGenDirtyCapture) dn.captureNeeded = true;
                    dn.chunkDirty[c] |= static_cast<uint8_t>(
                        valueBits ? valueBits : UsdGenDirtyParameter);
                }
            }
        }
    }
}

void UsdGenGraph::DirtyChunks(UsdGenNodeId id, TfSpan<const UsdGenChunkId> chunks)
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

void UsdGenGraph::DirtyMap(SdfPath const &mapPrim, uint64_t generation)
{
    if (generation != UINT64_MAX)
        _inputVersions.UpdateMap(mapPrim, generation);
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        auto &n = *nPtr;
        for (SdfPath const &m : n.mapRefs)
            if (m == mapPrim) {
                // MarkNode propagates both capture and value dirt to every
                // strict descendant; map edits cannot leave a stale child
                // capture behind.
                MarkNode(n.id, UsdGenDirtyMap | UsdGenDirtyCapture);
                break;
            }
    }
}

void UsdGenGraph::DirtyCurves(SdfPath const &curvePrim, uint64_t generation)
{
    if (generation != UINT64_MAX)
        _inputVersions.UpdateCurve(curvePrim, generation);
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        auto &n = *nPtr;
        for (SdfPath const &c : n.curveRefs)
            if (c == curvePrim) {
                MarkNode(n.id, UsdGenDirtyCapture);
                break;
            }
    }
}

void UsdGenGraph::DirtyGeometry(SdfPath const &prim)
{
    for (auto &nPtr : _nodes) {
        if (!nPtr) continue;
        for (SdfPath const &ref : nPtr->geometryRefs)
            if (ref.HasPrefix(prim)) {
                MarkNode(nPtr->id, UsdGenDirtyCapture);
                break;
            }
    }
}

// Effective curve buffer for a node: a pure passthrough node (no topo fx)
// presents its input's buffer — topology, cv totals and offsets — so stale
// downstream totals never override a changed upstream layout; a topo fx
// regenerates curves and owns the layout again. Traced through passthrough
// inputs only; depth guards against accidental cycles in the input chain.
static UsdGenCurveBuffer const &_EffBuffer(UsdGenGraph const &g,
                                           UsdGenNodeId id, int depth)
{
    UsdGenCompiledNode const &n = g.Node(id);
    // Length's static CurveCount classification covers its whole family,
    // but the current non-owning capture only repositions existing CVs.
    // Follow its input before capture too: its initially empty output must
    // not turn a ragged source into the partition's uniform fallback.
    // An actual owning capture remains an authoritative topology boundary.
    bool const nonOwningLength = n.op && n.op->Type().GetString() == "UsdGenLength" &&
        (!n.capture || !n.capture->OwnsBuffer());
    if ((n.topoFx != UsdGenTopoFx::None && !nonOwningLength) ||
        n.input == kUsdGenInvalidNode || depth > 64)
        return n.buffer;
    return _EffBuffer(g, n.input, depth + 1);
}

bool UsdGenGraph::Repartition(int totalCurves, int cvCount)
{
    int const nChunks = ComputeNumChunks(totalCurves, _chunkSize);
    // 03 §1.4: tileTarget is clamped to [32, 256] for the partition.
    int const tileTarget =
        std::clamp(_tileTarget, kUsdGenTileMin, kUsdGenTileMax);
    int const cpt = ComputeChunksPerTile(nChunks, tileTarget);
    int const nTiles = ComputeNumTiles(nChunks, tileTarget);
    int const previousChunks = ComputeNumChunks(_partitionCurves, _chunkSize);
    bool const chunkLayoutChanged = previousChunks != nChunks;
    bool const tileLayoutChanged = (chunkLayoutChanged || _chunksPerTile != cpt ||
                                    int(_tiles.size()) != nTiles);
    bool const changed = (_partitionCurves != totalCurves || tileLayoutChanged);
    _partitionCurves = totalCurves;

    if (tileLayoutChanged) {
        _chunksPerTile = cpt;
        _tiles.assign(nTiles, UsdGenTileView());
        auto boundary = [=](int t) -> int {
            // Reserve one chunk for each remaining tile before taking the
            // packed upper bound; this keeps every tile non-empty while
            // preserving the cpt ceiling when nChunks is just above nTiles.
            int64_t const packed = int64_t(t) * int64_t(cpt);
            int64_t const tail = int64_t(nChunks) - int64_t(nTiles - t);
            return static_cast<int>(std::min(packed, tail));
        };
        for (int t = 0; t < nTiles; ++t) {
            _tiles[t].tile = static_cast<UsdGenTileId>(t);
            _tiles[t].firstChunk = static_cast<uint32_t>(boundary(t));
            _tiles[t].chunkCount =
                static_cast<uint32_t>(boundary(t + 1) - boundary(t));
        }
    }

    auto tileForChunk = [=](int chunk) -> UsdGenTileId {
        auto const it = std::upper_bound(
            _tiles.begin(), _tiles.end(), static_cast<uint32_t>(chunk),
            [](uint32_t value, UsdGenTileView const& tile) {
                return value < tile.firstChunk;
            });
        return static_cast<UsdGenTileId>(
            it == _tiles.begin() ? 0 : std::distance(_tiles.begin(), it) - 1);
    };

    bool layoutDiff = false;

    for (size_t ni = 0; ni < _nodes.size(); ++ni) {
        if (!_nodes[ni]) continue;
        UsdGenCompiledNode &n = *_nodes[ni];
        // Per-node cvp: prefer the node's actual curve data (its effective
        // buffer, possibly a pure passthrough input's) over the partition's
        // uniform override. Ragged buffers (non-empty cvOffsets) are carried
        // verbatim: per-chunk firstCv is the source offset, cvCount == 0
        // means "variable cv/curve" per UsdGenChunkDesc.
        UsdGenNodeId const id = static_cast<UsdGenNodeId>(ni);
        UsdGenCurveBuffer const &eff = _EffBuffer(*this, id, 0);
        VtIntArray const &offs = eff.cvOffsets;
        uint32_t cvpUniform = static_cast<uint32_t>(std::max(0, cvCount));
        if (eff.totalCurves > 0 &&
            eff.totalCvs % eff.totalCurves == 0) {
            cvpUniform = eff.totalCvs / eff.totalCurves;
        }
        uint32_t const cvp = offs.empty() ? cvpUniform : 0u;
        // Keep existing chunks only if the count matches and EVERY chunk
        // still equals the desired layout (curve span AND cv span: a 10->20
        // root recook keeps 1 chunk and the same cv/curve, so comparing cv
        // fields alone keeps a stale curve span that under-emits tiles when
        // growing -- or over-reads buffers when shrinking).
        bool keep = !n.chunks.empty() && int(n.chunks.size()) == nChunks;
        for (int c = 0; keep && c < nChunks; ++c) {
            uint32_t const firstCurve = static_cast<uint32_t>(c * _chunkSize);
            uint32_t const curveCount = static_cast<uint32_t>(
                std::min(_chunkSize, totalCurves - c * _chunkSize));
            uint32_t const wantFirstCv = offs.empty()
                ? firstCurve * cvp
                : static_cast<uint32_t>(
                      offs[std::min<size_t>(firstCurve, offs.size() - 1)]);
            keep = n.chunks[c].firstCurve == firstCurve &&
                   n.chunks[c].curveCount == curveCount &&
                   n.chunks[c].cvCount == cvp &&
                   n.chunks[c].firstCv == wantFirstCv;
        }
        if (keep) {
            if (tileLayoutChanged) {
                // Tile ids moved under a kept chunk set: refresh them.
                for (auto &cd : n.chunks)
                    cd.tile = tileForChunk(static_cast<int>(
                        cd.firstCurve / std::max<size_t>(1, _chunkSize)));
            }
            continue;
        }
        n.chunks.assign(nChunks, UsdGenChunkDesc());
        for (int c = 0; c < nChunks; ++c) {
            UsdGenChunkDesc &cd = n.chunks[c];
            cd.firstCurve = static_cast<uint32_t>(c * _chunkSize);
            cd.curveCount = static_cast<uint32_t>(
                std::min(_chunkSize, totalCurves - c * _chunkSize));
            if (offs.empty()) {
                cd.firstCv = cd.firstCurve * cvp;
                cd.cvCount = cvp;
            } else {
                // Ragged: cvCount == 0 = variable cv/curve; firstCv is the
                // source-curve offset (clamped for safety).
                cd.cvCount = 0;
                cd.firstCv = static_cast<uint32_t>(
                    offs[std::min<size_t>(cd.firstCurve, offs.size() - 1)]);
            }
            cd.liveCount = cd.curveCount;
            cd.tile = tileForChunk(c);
        }
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
    TF_UNUSED(bits);
}

}  // namespace usdGen
