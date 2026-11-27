// usdGen engine — the compiled graph (03-execution-engine.md §3, §9.1).
//
// A UsdGenGraph is the compiled, topologically-sorted node list with per-node
// buffers, captures and chunk-dirty bytes. The compiler (§3) fills it; the
// scheduler (§5) mutates dirty state and evaluates; the session (§9.1) owns it.
#ifndef USDGEN_GRAPH_H
#define USDGEN_GRAPH_H

#include "usdGen/api/compile.h"
#include "usdGen/api/elementSet.h"
#include "usdGen/api/routing.h"
#include "usdGen/exec/executionCache.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace usdGen {
class UsdGenOpRecord;

/// One compiled node. Owns its UsdGenOp, its output set, its capture, its
/// chunk dirty bytes and its digests (03 §3.2).
struct UsdGenCompiledNode : UsdGenPooled
{
    UsdGenOpNode const *desc = nullptr;         // points into the graph's desc copy
    int                  descIdx = -1;            // index of that desc in graph Desc().nodes

    UsdGenNodeId           id = kUsdGenInvalidNode;
    UsdGenOpId             opId = UsdGenOpId::Invalid;
    UsdGenOpRecord const  *record = nullptr;         // immutable, process lifetime
    std::unique_ptr<UsdGenOp> op;

    UsdGenNodeId           input = kUsdGenInvalidNode;   // primary usdGen:input (dense range test)
    std::vector<UsdGenNodeId> inputs;                   // all usdGen:input targets

    uint8_t                countChanges = 0;
    UsdGenLane             lane = UsdGenLane::Chunked;
    bool                   enabled = true;

    UsdGenEpoch            structuralDigest {};  // Merkle d(n) (03 §3.3)
    UsdGenEpoch            captureEpoch {};      // (03 §3.4)
    // The stamps of `output` (03 §3.2), kept here too so a node without an
    // output yet (or a rebuilt one, compiler) carries them: valueVersion is
    // this node's own counter, bumped by every run that writes;
    // topologyVersion follows the input, or topologySeq for an owning capture.
    uint64_t               valueVersion = 0;
    uint64_t               topologyVersion = 0;
    uint64_t               topologySeq = 0;       // monotone install counter for owning captures

    // This node's output (S24): sealed by the scheduler whenever it prepares
    // the node. Unwritten channels share the input's (or the capture's)
    // columns; the written ones live in one delta arena that the next run
    // rewrites in place (D9). Null before the first run.
    UsdGenRef<const UsdGenElementSet> output;
    // An owning capture's output, converted once per capture: the base the
    // node's output derives from. Null for every other capture.
    UsdGenRef<const UsdGenElementSet> captureOutput;
    // Slot cache: the output's channel table for the input table and prior
    // output table it was derived from (refreshed only when either moves).
    UsdGenRef<const UsdGenChannelTable> channels;
    uint64_t               channelsInput = 0, channelsPrior = 0;
    std::unique_ptr<UsdGenCapture> capture;       // epoch-keyed payload
    // The capture `capture` replaced, kept when Recyclable(): the next
    // re-capture refills it (by then nothing downstream shares its arrays).
    std::unique_ptr<UsdGenCapture> spareCapture;
    UsdGenPoolVector<UsdGenChunkRange> chunks;    // this node's chunk partition
    std::vector<UsdGenChunkDirtyBits> chunkDirty; // one dirty word per chunk
    std::vector<uint8_t>      chunkCaptured;      // capture-validity byte per chunk (reference lane: single flag)

    // --- compile-time routing data (03 §5.1; filled by UsdGenCompiler) ----
    UsdGenParamView        paramView;            // points into the graph's desc copy
    // record->ParamCount() resolved slots, re-resolved on every compile;
    // paramView.slots points here.
    UsdGenPoolVector<UsdGenResolvedParam> params;
    int                    indexedPlaneBase = -1;  // first indexed plane (clumpId_<base>)
    uint8_t                indexedPlaneCount = 0;
    bool                   limitSurfaceUv = false; // hooks.emitsLimitSurfaceUv at compile
    // kUsdGenOpRemapsVertexPlanes or hooks.remapsVertexPlanes, at compile.
    bool                   remapsVertexPlanes = false;
    // Plane slots resolved per run by the scheduler; negative: element rate.
    std::array<int16_t, kUsdGenMaxExtraPlaneSlots> outPlaneIndex{}, inPlaneIndex{};
    // Connected parameters evaluated for this node, once per cook, over its
    // INPUT geometry; null when no evaluator is registered.
    // paramView.expressions points here.
    std::unique_ptr<UsdGenParamEvaluatorState> expressions;
    uint64_t               lastExpressionDigest = 0;  // the digest last evaluated
    std::vector<std::pair<UsdGenToken, uint32_t>> paramRouting;  // param name -> UsdGenDirtyBits
    std::vector<UsdGenNodeId> descendants;       // strict descendants, topological order
    // Slot order is the operator declaration order.  The scheduler resolves
    // these names to the current upstream planes on each preparation, since a
    // topology revision may replace the backing VtArrays.
    std::vector<UsdGenToken> outputPrimvars;
    std::vector<UsdGenToken> inputPrimvars;
    UsdGenSurfaceId        surface = 0;
    bool                   hasSurface = false;
    std::vector<UsdGenNodeKey> inputRefs;            // UsdGenOpNode::inputSets targets
    std::vector<UsdGenNodeKey> mapRefs;              // legacy flat map targets
    std::vector<UsdGenMapBinding> mapBindingRefs; // typed authored map slots
    // Prims a connected expression reads through its input:<name>
    // relationships (targets, the gprims and maps they resolve to).
    std::vector<UsdGenNodeKey> geometryRefs;
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
    void DirtyParameter(NodeId id, UsdGenToken const &param);
    /// One-shot value dirty with no parameter name (a time reader's clock moved).
    void DirtyValue(NodeId id);
    void DirtyCapture(NodeId id);
    void DirtyTopology(NodeId id);
    void DirtySurface(UsdGenSurfaceId surface, uint32_t bits,
                      uint64_t generation = UINT64_MAX);
    void DirtyChunks(NodeId id, UsdGenSpan<const UsdGenChunkId> chunks);
    /// Bumps the capture epoch of every node whose UsdGenOpNode::maps names this prim
    /// (asset path edit, ReloadMaps(), textureGeneration bump). §3.6, UsdGenDirtyMap.
    void DirtyMap(UsdGenNodeKey mapPrim, uint64_t generation = UINT64_MAX);
    /// Bumps the capture epoch of every node whose UsdGenOpNode::inputSets names this
    /// input (its content changed: an edit, a re-freeze, a re-import). §3.6, §4.5.
    void DirtyInput(UsdGenNodeKey inputIdentity,
                     uint64_t generation = UINT64_MAX);
    /// Re-captures every node whose connected expressions sample this prim
    /// (geoSampler()/ptex() inputs). Exact key: the host expands a dirtied
    /// prim to the keys under it (usdGenUsd NodeKeyMap::KeysUnder).
    void DirtyGeometry(UsdGenNodeKey prim);

    /// Immutable-after-publication input generations captured by the last
    /// compiled descriptor. Dirty routing updates this tuple before marking
    /// dependent nodes, so cache admission can compare exact versions.
    UsdGenExecutionInputVersions const& InputVersions() const noexcept {
        return _inputVersions;
    }

    /// The terminal node's output set (an empty set before the first run)
    /// and its tile partition (03 §6.3).
    UsdGenElementSet const &Output() const;
    /// A node's output, or the empty set.
    UsdGenElementSet const &NodeOutput(NodeId id) const;
    /// A set with no elements (made on first use).
    UsdGenElementSet const &EmptyOutput() const;
    UsdGenSpan<const UsdGenChunkRange> Chunks(NodeId id) const;
    /// Publication groups over the terminal's chunks (a C2 tile per group).
    UsdGenSpan<const UsdGenChunkGroup> Groups() const;

    /// True when any node carries a dirty byte the next commit must consume.
    bool AnyDirty() const noexcept;

    UsdGenGraphSpec const &Spec() const noexcept { return *_desc; }
    /// Spec() in Default space. Operators evaluate in Default space until a
    /// deforming op: every node upstream of the first one on its path sees
    /// this view, that op and everything after it see Spec(). Here a
    /// surface with authored rest data (UsdGenRestAPI) reads its Default-time
    /// rest as its current points -- one shutter sample at the desc time, no
    /// velocities -- and its surfaceGeneration follows only that rest content,
    /// so animating the surface re-captures nothing upstream of a deform. A
    /// surface without authored rest is unchanged. Built on first use after
    /// the descriptor or any surface's content changes; commit thread only.
    UsdGenGraphSpec const &RestSpec() const;
    UsdGenOpNode const &OpNode(NodeId id) const;
    /// Immutable descriptor-value identities resolved during compilation.
    UsdGenSpan<const UsdGenResolvedReferenceValue> ReferenceValues() const;
    UsdGenSpan<const UsdGenResolvedMapValue> MapValues() const;

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
    int NumGroups() const noexcept { return int(_groups.size()); }
    NodeId TerminalNodeId() const noexcept { return _terminal; }
    UsdGenNodeId NodeIdForKey(UsdGenNodeKey key) const;
    /// OR bits into node id's chunks (all of them) and propagate the value
    /// bits to every strict descendant (03 §5.2 hop 3).
    void MarkNode(NodeId id, uint32_t bits);
    /// Re-partition every node's chunks + the group set after a topology
    /// change, through the output layout's partition and groups hooks. A
    /// node's sub-item layout comes from its effective set when known (a
    /// generator upstream of an item-count-changing node keeps its own stride);
    /// `fallbackStride` items per element stand in for a set not populated
    /// yet. Returns true when the layout (element count, chunk count or any
    /// node's chunk spans) changed; unchanged nodes keep their chunk and
    /// dirty bytes.
    bool Repartition(uint32_t elements, uint32_t fallbackStride);
private:
    void _PropagateDescendantBits(uint32_t bits, std::vector<bool> const &descendants);

    std::unique_ptr<UsdGenGraphSpec> _desc;  // owned copy taken at Compile
    // The previous descriptor, overwritten by the next Recompile, and the
    // node vector Recompile swaps out: both keep capacity across compiles.
    std::unique_ptr<UsdGenGraphSpec> _spareDesc;
    std::vector<std::unique_ptr<UsdGenCompiledNode>> _retiredNodes;
    // RestSpec()'s view and the source it was built from: the descriptor, and
    // per surface its generation and its points' and rest's buffers.
    mutable std::unique_ptr<UsdGenGraphSpec> _restDesc;
    mutable UsdGenGraphSpec const *_restSource = nullptr;
    mutable std::vector<std::array<uint64_t, 3>> _restSourceKeys;
    // Recent Default-space surface records, keyed by the posed record's
    // columns (held) and content: a recompile that keeps an input reuses its
    // rest record without allocating.
    struct RestRecord {
        UsdGenRef<const UsdGenElementSet> posed;
        uint64_t generation = 0;
        UsdGenInputRecord rest;
    };
    mutable std::vector<RestRecord> _restRecords;
    std::vector<std::unique_ptr<UsdGenCompiledNode>> _nodes; // indexed by id
    UsdGenPoolVector<UsdGenChunkGroup> _groups;  // terminal publication groups
    UsdGenPoolVector<UsdGenChunkRange> _rangeScratch;   // Repartition's desired chunks
    UsdGenPoolVector<UsdGenChunkGroup> _groupScratch;   // and groups
    std::vector<UsdGenResolvedReferenceValue> _referenceValues;
    std::vector<UsdGenResolvedMapValue> _mapValues;
    UsdGenNodeId      _terminal = InvalidNode;
    std::unordered_map<UsdGenNodeKey, UsdGenNodeId, UsdGenNodeKeyHash, std::equal_to<UsdGenNodeKey>,
                       UsdGenPoolAllocator<std::pair<const UsdGenNodeKey, UsdGenNodeId>>> _nodeByKey;
    int _chunkSize = kUsdGenDefaultChunkSize;
    int _tileTarget = kUsdGenTileTargetDefault;
    int _chunksPerTile = 1;
    uint32_t _partitionElements = 0;   // elements of the last partition (0 == unpartitioned)
    // Fallback lifetime belongs to the graph, not process static teardown.
    // Made on first use; Output exposes it only as const.
    mutable UsdGenRef<const UsdGenElementSet> _emptyOutput;
    UsdGenExecutionInputVersions _inputVersions;
};

}  // namespace usdGen

#endif  // USDGEN_GRAPH_H
