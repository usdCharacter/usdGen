// usdGen engine — the compiled graph (03-execution-engine.md §3, §9.1).
//
// A UsdGenGraph is the compiled, topologically-sorted node list with per-node
// buffers, captures and chunk-dirty bytes. The compiler (§3) fills it; the
// scheduler (§5) mutates dirty state and evaluates; the session (§9.1) owns it.
#ifndef USDGEN_GRAPH_H
#define USDGEN_GRAPH_H

#include "usdGen/curveBuffer.h"
#include "usdGen/executionCache.h"
#include "usdGen/op.h"
#include "usdGen/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
class UsdGenCudaExecutionPlan;

/// Extra-plane pointer tables are stack-local in the chunk evaluator.  The
/// current R24 contract has only a handful of named outputs; keep an explicit
/// bound so binding them never turns Evaluate() into an allocating path.
constexpr uint32_t kUsdGenMaxExtraPlaneSlots = 16;

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
    SdfPath path;
    TfToken type;
    std::vector<TfToken> topologyParameters;
    std::vector<TfToken> valueParameters;
    std::vector<std::pair<TfToken, uint32_t>> paramRouting;
    UsdGenSurfaceId surface = 0;
    bool hasSurface = false;
    std::vector<SdfPath> curveRefs;
    std::vector<SdfPath> mapRefs;
    std::vector<UsdGenMapBindingDesc> mapBindingRefs;
    std::vector<SdfPath> geometryRefs;
    std::vector<UsdGenPaintRoutingRef> paintRefs;
};

struct UsdGenGraphRoutingSnapshot
{
    SdfPath description;
    /// Per desc.surfaces entry: the bound target, and the Mesh prim whose
    /// points, topology and primvars it reads. They differ only for a
    /// GeomSubset target, whose own prim carries just indices and type (R15).
    std::vector<SdfPath> surfacePaths;
    std::vector<SdfPath> meshPaths;
    std::vector<UsdGenGraphRoutingNode> nodes;
    UsdGenNodeId terminal = kUsdGenInvalidNode;
};

/// One compiled node. Owns its UsdGenOp, its output buffer, its capture, its
/// chunk dirty bytes and its digests (03 §3.2).
struct UsdGenCompiledNode
{
    UsdGenNodeDesc const *desc = nullptr;         // points into the graph's desc copy
    int                  descIdx = -1;            // index of that desc in graph Desc().nodes

    UsdGenNodeId           id = kUsdGenInvalidNode;
    TfToken                type;
    std::unique_ptr<UsdGenOp> op;

    UsdGenNodeId           input = kUsdGenInvalidNode;   // primary usdGen:input (dense range test)
    std::vector<UsdGenNodeId> inputs;                   // all usdGen:input targets

    UsdGenTopoFx           topoFx = UsdGenTopoFx::None;
    UsdGenRole             role = UsdGenRole::Curves;
    bool                   enabled = true;

    UsdGenEpoch            structuralDigest {};  // Merkle d(n) (03 §3.3)
    UsdGenEpoch            captureEpoch {};      // (03 §3.4)
    uint64_t               valueVersion = 0;     // (03 §3.2)
    uint64_t               topologySeq = 0;       // monotone install counter for generators

    UsdGenCurveBuffer      buffer;               // this node's output (S24)
    std::unique_ptr<UsdGenCapture> capture;       // epoch-keyed payload
    std::vector<UsdGenChunkDesc> chunks;          // this node's chunk partition
    std::vector<uint8_t>      chunkDirty;         // one dirty byte per chunk (UsdGenDirtyBits)
    std::vector<uint8_t>      chunkCaptured;      // capture-validity byte per chunk (reference lane: single flag)

    // --- compile-time routing data (03 §5.1; filled by UsdGenCompiler) ----
    UsdGenParamView        paramView;            // points into the graph's desc copy
    // Connected (expression) parameters evaluated for this node, once per
    // cook, over its INPUT geometry. paramView.expressions points here.
    UsdGenCpuParameters    expressions;
    uint64_t               lastExpressionDigest = 0;  // the digest last evaluated
    std::vector<std::pair<TfToken, uint32_t>> paramRouting;  // param name -> UsdGenDirtyBits
    std::vector<UsdGenNodeId> descendants;       // strict descendants, topological order
    // Slot order is the operator declaration order.  The scheduler resolves
    // these names to the current upstream planes on each preparation, since a
    // topology revision may replace the backing VtArrays.
    std::vector<TfToken> outputPrimvars;
    std::vector<TfToken> inputPrimvars;
    UsdGenSurfaceId        surface = 0;
    bool                   hasSurface = false;
    std::vector<SdfPath>   curveRefs;            // usdGen:curves/guides/frozen:curves targets
    std::vector<SdfPath>   mapRefs;              // legacy flat map targets
    std::vector<UsdGenMapBindingDesc> mapBindingRefs; // typed authored map slots
    // Prims a connected expression reads through its input:<name>
    // relationships (targets, the gprims and maps they resolve to).
    std::vector<SdfPath>   geometryRefs;
    std::vector<uint32_t>  referenceValues;       // indices into graph immutable reference values
    std::vector<uint32_t>  mapValues;             // indices into graph immutable map values
    bool                   captureNeeded = false; // set by the session, cleared by the capture step

    // --- evaluation-skip state (value-class no-op detection; keeps SI-2's
    // "nothing else" notice set exact and the sparse path cheap) ---
    uint64_t paramValueDigest = 0;  // hash of the value-class params + enabled
    uint64_t lastParamDigest  = 0;  // the paramValueDigest last evaluated
    double lastEvalTime = std::numeric_limits<double>::quiet_NaN();
    double lastEvalRate = std::numeric_limits<double>::quiet_NaN();
    bool hasEvaluationTime = false;
    double lastEvaluationTime = 0.0;
    double lastTimeCodesPerSecond = 0.0;
};

class UsdGenGraph
{
    friend class UsdGenCompiler;
public:
    using NodeId = UsdGenNodeId;
    static constexpr NodeId InvalidNode = kUsdGenInvalidNode;

    UsdGenGraph() = default;
    UsdGenGraph(UsdGenGraph &&) = default;
    UsdGenGraph &operator=(UsdGenGraph &&) = default;
    UsdGenGraph(const UsdGenGraph &) = delete;
    UsdGenGraph &operator=(const UsdGenGraph &) = delete;

    int NodeCount() const noexcept { return int(_nodes.size()); }
    UsdGenCompiledNode &Node(NodeId id);
    UsdGenCompiledNode const &Node(NodeId id) const;


    // --- dirty routing targets (03 §5.2 hops 2-3; the imaging router feeds these) ---
    void DirtyParameter(NodeId id, TfToken const &param);
    void DirtyCapture(NodeId id);
    void DirtyTopology(NodeId id);
    void DirtySurface(UsdGenSurfaceId surface, uint32_t bits,
                      uint64_t generation = UINT64_MAX);
    void DirtyChunks(NodeId id, TfSpan<const UsdGenChunkId> chunks);
    /// Bumps the capture epoch of every node whose UsdGenNodeDesc::maps names this prim
    /// (asset path edit, ReloadMaps(), textureGeneration bump). §3.6, UsdGenDirtyMap.
    void DirtyMap(SdfPath const &mapPrim, uint64_t generation = UINT64_MAX);
    /// Bumps the capture epoch of every node whose UsdGenNodeDesc::curves names this prim
    /// (a C3 BasisCurves changed: guide edit, re-freeze, re-import). §3.6, §4.5.
    void DirtyCurves(SdfPath const &curvePrim,
                     uint64_t generation = UINT64_MAX);
    /// Re-captures every node whose connected expressions sample this prim
    /// (geoSampler()/ptex() inputs), or a descendant of it.
    void DirtyGeometry(SdfPath const &prim);

    /// Immutable-after-publication input generations captured by the last
    /// compiled descriptor. Dirty routing updates this tuple before marking
    /// dependent nodes, so cache admission can compare exact versions.
    UsdGenExecutionInputVersions const& InputVersions() const noexcept {
        return _inputVersions;
    }

    /// The terminal node's OUTPUT buffer and its tile partition (03 §6.3).
    UsdGenCurveBuffer const &Output() const;
    TfSpan<const UsdGenChunkDesc> Chunks(NodeId id) const;
    TfSpan<const UsdGenTileView>  Tiles() const;

    /// True when any node carries a dirty byte the next commit must consume.
    bool AnyDirty() const noexcept;

    UsdGenGraphDesc const &Desc() const noexcept { return *_desc; }
    std::shared_ptr<const UsdGenCudaExecutionPlan> const& CudaPlan() const noexcept { return _cudaPlan; }
    UsdGenNodeDesc const &NodeDesc(NodeId id) const;
    /// Immutable descriptor-value identities resolved during compilation.
    TfSpan<const UsdGenResolvedReferenceValue> ReferenceValues() const;
    TfSpan<const UsdGenResolvedMapValue> MapValues() const;

    /// Make an owning routing snapshot.  This is intentionally a fresh copy:
    /// callers can hand the result to an asynchronous imaging owner without
    /// retaining this mutable graph or any operator/node pointers.
    /// Construct it on the graph's execution owner, never concurrently with
    /// graph mutation. Only the resulting snapshot is safe for readers.
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> RoutingSnapshot() const;

    // Chunk/tile partition state (R21; terminal-node topology in M1).
    int ChunkSize() const noexcept { return _chunkSize; }
    int TileTarget() const noexcept { return _tileTarget; }
    int ChunksPerTile() const noexcept { return _chunksPerTile; }
    int NumTiles() const noexcept { return int(_tiles.size()); }
    NodeId TerminalNodeId() const noexcept { return _terminal; }
    UsdGenNodeId NodeIdForPath(SdfPath const &path) const;
    /// OR bits into node id's chunks (all of them) and propagate the value
    /// bits to every strict descendant (03 §5.2 hop 3).
    void MarkNode(NodeId id, uint32_t bits);
    /// Re-partition every node's chunks + the tile set after a topology
    /// change. Per-node CV count is taken from the node's own buffer when
    /// known (a generator upstream of a CV-changing node keeps its own
    /// cv/curve); `cvCount` is the fallback for nodes whose buffers are not
    /// populated yet. Returns true when the layout (curve count, chunk
    /// count or any node's cv/curve) changed; unchanged nodes keep their
    /// chunk and dirty bytes.
    bool Repartition(int totalCurves, int cvCount);
private:
    void _PropagateDescendantBits(uint32_t bits, std::vector<bool> const &descendants);

    std::unique_ptr<UsdGenGraphDesc> _desc;  // owned copy taken at Compile
    std::shared_ptr<const UsdGenCudaExecutionPlan> _cudaPlan;
    std::vector<std::unique_ptr<UsdGenCompiledNode>> _nodes; // indexed by id
    std::vector<UsdGenTileView> _tiles;          // terminal tile partition
    std::vector<UsdGenResolvedReferenceValue> _referenceValues;
    std::vector<UsdGenResolvedMapValue> _mapValues;
    UsdGenNodeId      _terminal = InvalidNode;
    std::unordered_map<SdfPath, UsdGenNodeId, SdfPath::Hash> _nodeByPath;
    int _chunkSize = kUsdGenDefaultChunkSize;
    int _tileTarget = kUsdGenTileTargetDefault;
    int _chunksPerTile = 1;
    int _partitionCurves = 0;   // totalCurves of the last partition (0 == unpartitioned)
    // Fallback lifetime belongs to the graph, not process static teardown.
    // It is private and Output exposes it only as const.
    UsdGenCurveBuffer _emptyOutput;
    UsdGenExecutionInputVersions _inputVersions;
};

}  // namespace usdGen

#endif  // USDGEN_GRAPH_H
