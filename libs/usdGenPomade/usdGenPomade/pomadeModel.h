// usdGenPomade — PomadeModel (plan/17 §2.1): the authoring state of one groom.
//
// The model owns the scalp binding and its graph, the tube hierarchy, the
// guide fill, picking and undo. Interaction never authors USD (D1); the model
// is the single source the Pomade scene index publishes from (D2) and the only
// thing the committer serialises.
//
// Tube identity (plan/17 §2.1 tubes[], Pomade contract 2): every tube has an
// id, a level, a region and a parent (-1 at L1), and there is one L1 tube per
// closed graph region. Ids encode the tree: a subdivided or imported child is
// `parent * 16 + 1 + childIndex` with childIndex in [0, 14], so every
// descendant id is 1..15 (mod 16) and decodes back to a unique root, while
// L1 roots are the multiples of 16 (0, 16, 32, …) and on-the-fly group
// parents mint negatives. The two families therefore cannot collide.
//
// Storage of the first L1 root (id 0) is the legacy member set below
// (_shape/_host/_sections/_roots/_guides/_fill and the CUDA mirror); every
// other tube lives in the _tubes store. The split is a storage detail, not a
// contract: TubeIds/L1TubeIds/GetTubeDesc/GetTubeRecord/SnapshotTubes and the
// per-tube ops present all tubes uniformly, and the tube-0 spellings of the
// single-tube ABI act on "the first L1 tube".
//
// Storage: planar host mirrors (std::vector) always exist; device mirrors
// (usdGen::gpu::DeviceBuffer) exist only under USDGEN_POMADE_HAS_CUDA and are
// hidden behind a pimpl so no CUDA type leaks into this header. The CPU
// tessellation in pomadeTessellate.h is authoritative; the CUDA kernel fills
// the device mirror with the same values.
#ifndef USDGEN_POMADE_MODEL_H
#define USDGEN_POMADE_MODEL_H

#include "usdGenPomade/api.h"
#include "usdGenPomade/pomadeGizmo.h"
#include "usdGenPomade/pomadeGraph.h"
#include "usdGenPomade/pomadeHierarchy.h"
#include "usdGenPomade/pomadeRegion.h"
#include "usdGenPomade/pomadeScalp.h"
#include "usdGenPomade/pomadeSelection.h"
#include "usdGenPomade/pomadeTessellate.h"
#include "usdGenPomade/pomadeTube.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace usdGenPomade {

// The immutable cross-thread view of a model (pomadeModelState.h, which
// only usdGenPomade's own sources include).
struct PomadeModelState;

#ifdef USDGEN_POMADE_HAS_CUDA
// Forward of the K11 reduction record (pomadeKernels.h); this header
// never includes the launch header, it only threads the pointer type.
struct PomadeDevicePickBest;
#endif

// What changed since the last published snapshot. The scene index maps these
// to leaf-exact Hydra dirty locators (points -> primvars/points/primvarValue
// + extent/min/max; topology -> mesh/topology/* as well).
enum PomadeDirty : uint32_t {
    PomadeDirty_Clean = 0,
    PomadeDirty_Points = 1u << 0,    // vertex positions/normals moved
    PomadeDirty_Topology = 1u << 1,  // ring/vertex counts or faces changed
    PomadeDirty_Graph = 1u << 2,     // scalp graph nodes/edges/regions changed
    PomadeDirty_Regions = 1u << 3,   // rasterised region maps changed
    PomadeDirty_Guides = 1u << 4,    // refilled guide preview changed
    PomadeDirty_Display = 1u << 5,   // level visibility / x-ray / focus changed
    PomadeDirty_Selection = 1u << 6,  // selection or hover changed (V1)
    PomadeDirty_Gizmo = 1u << 7,      // the gizmo record changed (V1)
    PomadeDirty_Brush = 1u << 8,      // the brush ring record changed (V1)
};

// The K8/K9/K10 product: guide CVs (guide-major), per-guide counts,
// stable curve ids and UsdGenCurveAPI root frames. Defined ahead of the
// model: the P3 guide cache lives on PomadeModel itself.
struct USDGENPOMADE_API PomadeGuideSet {
    std::vector<float> points;     // 3 floats per CV, guide-major
    std::vector<int> counts;       // CVs per guide (uniform)
    std::vector<uint64_t> ids;     // stable curve ids, 1000 + guide
    std::vector<double> frames;    // 16 doubles per root frame (row-major)
    std::vector<int> tubeIds;      // owning tube per guide (one entry per
                                   // guide; the pure fills stamp their tube,
                                   // the live refill stamps each producer)
    int guideCount = 0;
    int cvCount = 0;
};

class USDGENPOMADE_API PomadeModel {
public:
    PomadeModel();
    ~PomadeModel();
    PomadeModel(PomadeModel const &) = delete;
    PomadeModel &operator=(PomadeModel const &) = delete;

    // (Re)build the static test tube from `shape` (default: 5 rings x 8
    // verts, radius 0.5, length 4.0 along +Y). Resets center columns to the
    // straight pose, tessellates, bumps the version, marks topology dirty.
    bool BuildTestTube(PomadeTubeShape shape = PomadeTubeShape());
    // Translate one center ring in the section plane; re-tessellates on the
    // next Sync(), bumps the version, marks points dirty. Returns false for
    // an out-of-range ring.
    bool MoveCenterRing(int ring, float dx, float dz);

    // Recompute derived buffers (tessellation + device mirror). The host
    // mirror always holds the authoritative result; a CUDA alloc, upload
    // or launch failure drops the device mirror and Sync still succeeds
    // (the P6 OOM fallback), recording the reason below. Only host-side
    // failures return false with a diagnostic.
    bool Sync();
    bool HasCudaMirror() const;
    // True once the P6 fallback dropped the mirror (sticky per model).
    // The reason reads "" while the mirror is healthy.
    bool DeviceFallback() const;
    char const *DeviceFallbackReason() const;
    // Copy the device mirror into pinned `posOut`/`nrmOut` (each `floatCount`
    // floats). Synchronizes the model's stream, so the copy observes the last
    // Sync. Returns false — leaving the outputs untouched — when no CUDA
    // mirror exists, in which case the caller stages from the host mirror.
    bool CopyDeviceToHost(float *posOut, float *nrmOut,
                          size_t floatCount) const;

    uint64_t GetVersion() const { return _version; }
    uint32_t TakeDirty();  // returns and clears the pending dirty bits
    uint32_t PeekDirty() const { return _dirty; }

    PomadeTubeShape GetShape() const { return _shape; }

    // Host mirrors (authoritative). Valid after BuildTestTube; Sync refreshes.
    // Authoring thread only: the commit worker reads the published state.
    struct HostTubeMesh {
        std::vector<float> positions;   // 3 * vertexCount, ring-major
        std::vector<float> normals;     // 3 * vertexCount
        std::vector<float> centerX;     // per ring
        std::vector<float> centerY;
        std::vector<float> centerZ;
        std::vector<int> faceVertexCounts;   // quads: 4 * quadCount entries of 4
        std::vector<int> faceVertexIndices;  // 4 * quadCount
        float extentMin[3];
        float extentMax[3];
    };
    HostTubeMesh const &GetHostMesh() const { return _host; }

    char const *GetDiagnostic() const { return _diagnostic.c_str(); }

    // -- P1: fill, locks, subdivision, snapshots (plan/17 section 2-3) -----
    //
    // The committer serialises these; hydrate reads them back. All setters
    // bump the version (the committer's coalescing key) but mark no mesh
    // dirty bits: guides are not tube geometry, and the P0 scene index does
    // not publish them yet (P3 wires the guide preview).
    struct FillParams {
        float density = 100.0f;  // expected guides per unit root area
        int cvCount = 8;         // CVs per guide, in [2, 64]
        int seed = 0;
        float edgeBias = 0.0f;   // radial remap in [-1, 1] (K9)
        std::vector<float> lengthProfile;  // flattened (pos, value) pairs
                                           // over the root radial fraction;
                                           // empty = uniform (K9)
        // Legacy committed layers retain their byte-identical K9 root stream;
        // all new authoring uses the region-v3 material sampler.
        PomadeGuideSampler sampler = PomadeGuideSampler::RegionV3;
    };
    // Commit-only output generation is deliberately separate from the live
    // Fill preview. A multiplier of one means the authored leaf Fill density
    // (never the transient preview fraction); width is the authored curve
    // width in scene units.
    struct OutputSettings {
        bool enabled = false;
        float densityMultiplier = 1.0f;
        float width = 0.01f;
        // Mirrors the selected Bake texel override so OutputRegionMap is
        // built at the artist-selected resolution (-1 selects automatic).
        int ptexResolution = -1;
    };
    struct SubdivideParams {
        int count = 4;  // in [2, 8]
        int seed = 0;
        std::string splitMode = "kmeans";  // kmeans | edge
        // Edge mode: a*u + b*v + c = 0 in root-frame coords. Kept so a
        // merge re-derives the same two halves the split made.
        float edgeA = 1.0f;
        float edgeB = 0.0f;
        float edgeC = 0.0f;
    };
    bool SetFillParams(FillParams params);
    FillParams GetFillParams() const;
    // Atomic, undoable output settings. They change commit output only, not
    // the interactive guide preview or tube mesh.
    bool SetOutputSettings(OutputSettings settings);
    OutputSettings GetOutputSettings() const;
    bool SetOutputPtexResolution(int resOverride);
    bool SetSubdivideParams(SubdivideParams params);
    SubdivideParams GetSubdivideParams() const;
    void SetLockFlags(bool locked, bool lockParents, bool lockChildren);
    void GetLockFlags(bool *locked, bool *lockParents,
                      bool *lockChildren) const;

    // A plain-data copy of the tube for the commit worker: taken on the
    // authoring thread into the published state, then the SdfLayer is built
    // from it with no model access. Guides are NOT stored; both the committer and hydrate derive
    // them from the snapshot through PomadeGenerateGuides, which is what makes
    // the hydrate bit-equality assert meaningful.
    struct TubeSnapshot {
        PomadeTubeShape shape;
        std::vector<float> centerX;
        std::vector<float> centerY;
        std::vector<float> centerZ;
        std::vector<PomadeTubeSection> sections;  // P3: authored rings; empty
                                                 // in pre-P3 snapshots means
                                                 // default circles (compat)
        bool rootFramePinned = false;
        PomadeFrame rootFrame;
        std::array<float, 9> frameReference = {{
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f}};
        FillParams fill;
        SubdivideParams subdivide;
        bool locked = false;
        bool lockParents = false;
        bool lockChildren = false;
        uint64_t version = 0;
        bool hasTube = false;
    };
    TubeSnapshot Snapshot() const;

    // -- Threading ---------------------------------------------------------
    //
    // Every method above and below belongs to the authoring (UI) thread; the
    // model is single-writer and takes no lock. Another thread reads the
    // model only through its published state: an immutable, refcounted
    // copy of what a commit serialises, published by atomic pointer.
    //
    // PublishState (authoring thread) captures the current state and makes
    // it the published one; the displaced state is retired through the
    // usdGen epochs, so a reader that loaded it keeps it valid.
    // AcquireState (any thread) returns the published state with one
    // reference the caller owns (usdGen::UsdGenRef<>::Adopt it), or null
    // before the first publish. CaptureState (authoring thread) builds a
    // fresh state without publishing it, again with one owned reference.
    void PublishState();
    PomadeModelState const *AcquireState() const;
    PomadeModelState const *CaptureState() const;

    // Restore the tube from a snapshot (the hydrate write path). Rebuilds
    // the center column, re-tessellates, bumps the version, marks topology
    // dirty. Returns false when the snapshot holds no tube.
    bool Restore(TubeSnapshot const &snapshot);

    // -- P2: scalp binding, graph, region maps (plan/17 §2.1, §4.5) --------
    //
    // The scalp mesh is shared (never copied per gesture); the graph is the
    // live authoring state; the region maps are derived by K3. Graph edits
    // bump BOTH the version (the §3.1 coalescing key) and the map version
    // (the §3.1a bake key); tube/sculpt edits bump only the version, so a
    // sculpt drag never enqueues a bake. K3 runs explicitly (Rasterise, at
    // gesture end per §4.2), never inside a graph edit.
    //
    // `activeFaces` binds a face GeomSubset (plan/02 §2.20): the arrays are
    // the PARENT mesh's and the ids are parent-mesh face indices; empty
    // binds every face. See PomadeScalpMesh::activeFaces.
    bool BindScalp(std::vector<float> const &points,
                   std::vector<int> const &faceVertexCounts,
                   std::vector<int> const &faceVertexIndices,
                   std::vector<int> const &activeFaces = std::vector<int>());
    bool HasScalp() const;
    // Authoring-thread views (like GetHostMesh); the commit worker reads
    // the published state instead.
    PomadeScalpGraph const &GetGraph() const { return _graph; }
    PomadeRegionMaps const &GetRegionMaps() const { return _maps; }
    PomadeRegionLoops const &GetRegionLoops() const { return _loops; }
    std::shared_ptr<PomadeScalpMesh const> GetScalp() const;

    // K1 queries against the bound scalp (miss when unbound).
    PomadeHit Raycast(float const origin[3], float const dir[3]) const;
    PomadeHit ClosestPoint(float const p[3]) const;

    // Graph-mode edits (§5.1). Each bumps the version + map version and
    // marks graph + regions dirty. Mirror-X (when enabled) twins every
    // placed node across x = 0.
    int GraphAddNode(PomadeHit const &hit);
    // Atomically close one region. Stable non-negative ids reuse live graph
    // nodes; -1 creates at the corresponding surface hit. Returns the
    // extracted region id, or -1 with no graph/undo/version mutation.
    int GraphCreateRegion(std::vector<int> const &nodeIds,
                          std::vector<PomadeHit> const &hits);
    bool GraphMoveNode(int nodeId, PomadeHit const &hit);
    bool GraphMoveNodes(std::vector<int> const &nodeIds,
                        std::vector<PomadeHit> const &hits);
    int GraphConnect(int a, int b);
    int GraphSplitEdge(int edgeId, PomadeHit const &hit);
    bool GraphWeld(int keep, int drop);
    int GraphWeldAll(float radius);
    std::vector<int> GraphUnweld(int nodeId);
    bool GraphDeleteEdge(int edgeId);
    bool GraphDeleteNode(int nodeId);
    int GraphSnapNode(float const p[3], float radius) const;
    int GraphSnapEdge(float const p[3], float radius) const;
    bool GraphGetNode(int nodeId, PomadeGraphNode *out) const;
    bool GraphGetEdge(int edgeId, int outNodeIds[2]) const;
    // Viewport-only graph-node position.  The graph stores canonical scalp
    // coordinates; this applies the same small normal lift as the scene
    // index so interaction can target the dot the artist sees.
    bool GraphGetNodeDisplayPosition(int nodeId, float outP[3]) const;
    bool GraphLinkRegions(int r0, int r1);
    bool GraphUnlinkRegions(int r0, int r1);
    PomadeStrokeResult GraphStroke(std::vector<PomadeHit> const &samples,
                                  float snapRadius, float simplifyEps);
    std::vector<std::pair<int, int>> GraphMirrorX();

    void SetSnapRadius(float radius);
    float GetSnapRadius() const;
    void SetMirrorX(bool on);
    bool GetMirrorX() const;

    // Run K3 over the current graph (gesture end, §4.2). Marks regions
    // dirty. Returns false with a diagnostic when no scalp is bound.
    bool Rasterise();

    // Exact graph-region hit at a face-local scalp coordinate.  The return
    // value is the source graph region id (never its interpolation id), or
    // -1 for an uncovered/off-surface point.  Unlike the coarse K3 face map,
    // this tests the hit itself against each closed graph loop, so separate
    // subface regions on one coarse face remain distinguishable.
    int RegionAtSurface(int faceId, float u, float v) const;

    uint64_t GetMapVersion() const;

    // The bake swap records the file it pointed at, so §3.1 layer builds
    // re-author the current map file instead of wiping it (TransferContent
    // replaces the whole live layer).
    void NoteBakedMapFile(uint64_t mapVersion, std::string const &path);

    struct GraphSnapshot {
        std::vector<PomadeGraphNode> nodes;  // alive only
        std::vector<std::pair<int, int>> edges;  // alive (a, b) pairs
        std::vector<std::vector<int>> regionLoops;  // node ids per region
        // Exact on-surface boundary samples per region. The coarse
        // faceRegions primvar cannot distinguish two subface polygons.
        std::vector<std::vector<float>> regionBoundaries;
        std::vector<std::pair<int, int>> linked;
        float snapRadius = 0.05f;
        std::vector<int> faceRegions;  // live primvar (K3, interp ids)
        std::vector<int> faceRegionIds;  // picking: lowest claiming region
        int uncoveredCount = 0;  // HUD overlays: coverage + intersections
        int intersectedCount = 0;
        std::vector<float> regionColors;  // 3R
        uint64_t mapVersion = 0;
        int levelCount = 1;
        std::string bakedMapFile;  // re-authored by §3.1 builds
        uint64_t bakedMapVersion = 0;
        bool hasScalp = false;
        size_t scalpFaceCount = 0;
    };
    GraphSnapshot SnapshotGraph() const;

    // Restore the graph from a snapshot (the hydrate write path). Re-traces
    // edge polylines, re-extracts regions, re-rasterises, bumps the version
    // (+ map version), marks graph + regions dirty.
    bool RestoreGraph(GraphSnapshot const &snapshot);

    // -- P3: Tube mode (plan/17 §5.2, K4/K5) --------------------------------
    //
    // The test tube keeps its legacy cylinder display until a section op
    // runs; section ops switch the tube to the K5 Hermite path. Every op
    // below bumps the version; mesh-affecting ops re-tessellate via Sync
    // and mark points (moved) or topology (counts changed) dirty. Guides
    // are NOT refilled here: the Tube-mode move is MoveCenterCV (or a ring
    // op) + RefillGuides(preview), the release is RefillGuides(1.0) (§4.2).
    // Append or refresh the L1 tube of `regionId` (plan/17 Pomade contract 2,
    // plan/18 §7 G14). Every closed graph region owns exactly one L1 tube:
    // the first one built lands in the legacy tube-0 members, each further
    // region gets a store entry with parentTubeId -1 and level 1, and a
    // second call for a region that already has a tube refreshes that tube
    // in place (its children re-derive). `ringVerts=0` matches the canonical
    // graph-loop CV count; an explicit 3..32 resolves to at least that many
    // slots, adding only edge samples so every drawn corner remains exact.
    // A loop needing more than 32 slots is refused.
    // Pushes ONE undo step; it no longer clears the stack.
    bool BuildTubeFromRegion(int regionId, int centerCount, int ringVerts,
                             float length);
    // The tube rooted in `regionId`, or -1. The inverse maps a tube id to
    // the region it (or its L1 ancestor) is rooted in.
    int TubeForRegion(int regionId) const;
    int RegionForTube(int tubeId) const;
    // Every L1 root, ascending: tube 0 when it is built, then the store
    // entries with no parent that are not on-the-fly group parents.
    std::vector<int> L1TubeIds() const;
    // Re-attach the L1 tubes to the regions a graph edit left behind
    // (plan/17 §5.1 Connect/Delete). Regions are renumbered densely by
    // every extraction, so identity is carried by the region's node loop,
    // not its id: a region whose loop is unchanged keeps its tube, a region
    // split in two hands its tube to the part that kept most of the loop
    // and leaves the other part stub-less for BuildTubeFromRegion, and two
    // regions merged by a deleted edge keep the tube with the larger share
    // while the other tube is removed with its subtree.
    //
    // Refuses (returning false, with the reason in GetDiagnostic and no
    // change applied) when a tube that would be re-derived or removed
    // carries child deltas or imports: the artist is asked to merge the
    // children first, exactly as §5.1 specifies. `outRebuilt`/`outRemoved`
    // are optional.
    bool SyncRegionTubes(std::vector<int> *outRebuilt,
                         std::vector<int> *outRemoved);
    // Rebuild only the stored support-plane frame for every region-rooted
    // L1 tube.  Hydrate uses this after validating a legacy raw-K4 guide
    // payload, so subsequent live edits gain the pinned-root invariant.
    bool PinRegionRootFrames();
    bool MoveCenterCV(int cv, float dx, float dy, float dz);
    bool InsertCenterCV(int atIndex);
    bool DeleteCenterCV(int index);
    bool SetTubeLength(float length);
    bool MatchSurface();
    int GetCenterCVCount() const;
    bool GetCenterCV(int cv, float *x, float *y, float *z) const;
    int GetSectionCount() const;
    bool GetSection(int ring, PomadeTubeSection *section) const;
    bool MoveSectionRing(int ring, float du, float dv);
    bool ScaleSectionRing(int ring, float scale);
    bool TwistSectionRing(int ring, float radians);
    bool MoveSectionCV(int ring, int slot, float du, float dv);
    bool AddSectionRing(float t);
    bool RemoveSectionRing(int ring);
    bool CopySectionRing(int src, int dst);
    bool SetSoftSelection(float center, float radius);
    void GetSoftSelection(float *center, float *radius) const;
    bool RelaxCenter(float strength, int iterations);
    bool SnapRootToScalp();
    bool SetDisplaySegments(int segments);
    int GetDisplaySegments() const;
    // The scalp-graph region this tube is rooted in (-1 = unrooted, disc
    // fill). Set by BuildTubeFromRegion and hydrate; bumps the version.
    void SetTubeRegionId(int regionId);
    int GetTubeRegionId() const;
    // Faces at the tube's interpolation id (mesh-fill roots). Empty when
    // unbound, unrooted, or unclaimed; read from the last Rasterise.
    std::vector<int> TubeRegionFaces() const;
    PomadeTubeDesc BuildTubeDesc() const;
    std::vector<PomadeFrame> GetFrames() const;  // K4 cache after Sync

    // -- P4: hierarchy store (plan/17 §2.1 tubes[], §2.4, §5.4/§5.5) -------
    //
    // Tube 0 is the primary tube in the members above; subdivided children
    // live in the store below as (actual, derived, deltas) triples. Every
    // mutator bumps the version; subdivide/merge/group also bump the map
    // version (tube ids change), while pure sculpt moves skip the bake.
    struct HierarchyTube {
        PomadeTubeDesc actual;    // authored shape (derived + deltas)
        PomadeTubeDesc derived;   // re-derived from the parent (K14/K6)
        PomadeShapeDeltas deltas;  // in the derived frames (zero at split)
        PomadeSubdivideDesc subdivide;  // params this tube split with
        bool lockParents = false;
        bool lockChildren = false;
        bool transientParent = false;  // on-the-fly group parent
        bool persistent = false;       // kept (UsdGenTubeHierarchyAPI)
        // P5 bridge import: explicit shape (never K6/K7-written, still
        // K7-read). Deltas stay zero; a zero norm does NOT mean "matches
        // derivation" for imports (there is no derivation to match).
        bool imported = false;
        std::vector<int> members;      // group-parent member tube ids
        // V0b (plan/18 G2): fill is per tube (plan/17 §2.1 tubes[].fill).
        // Children inherit the parent's params at the split; a parent's own
        // fill is suspended while it has children (§2.3), so only leaves
        // produce guides.
        FillParams fill;
    };
    // Split tubeId into count children (K14). Fails when the tube already
    // has children (merge or re-subdivide instead). Bumps version + map.
    bool SubdivideTube(int tubeId, int count, const char *splitMode,
                       int seed, std::vector<int> *outChildren);
    // The same with the edge-mode split line carried (plan/18 §7 G12 /
    // audit §5.4 row 267: the coefficients were hard-coded to (1, 0, 0),
    // so "the split follows the drawn edge" was unreachable). `edgeA/B/C`
    // are a*u + b*v + c = 0 in the tube's ROOT FRAME coordinates (u along
    // the root frame normal, v along its binormal, origin at center CV 0),
    // and are ignored by kmeans mode.
    bool SubdivideTube(int tubeId, int count, const char *splitMode,
                       int seed, float edgeA, float edgeB, float edgeC,
                       std::vector<int> *outChildren);
    // Edge subdivide from a drawn stroke: two WORLD points across the root
    // region become the split line, projected into the root frame here so
    // no caller has to know the frame. Always two children (edge mode).
    bool SubdivideTubeAlongEdge(int tubeId, float const *worldA,
                                float const *worldB, int seed,
                                std::vector<int> *outChildren);
    // Merge tubeId's whole subtree back into it (K7 last pass over the
    // children with the persistent-parent hint, then removal). Bumps
    // version + map. Merging a childless tube is a no-op success.
    bool MergeChildren(int tubeId);
    // Fold sibling tubeIds into one child at their level (K7 over the
    // subset, no hint); the kept id is tubeIds[0]. Bumps version + map.
    bool MergeSelected(std::vector<int> const &tubeIds, int *outKept);
    // Whole-tube Delete (SL-03): drop every listed tube with its subtree
    // as ONE undo step (a gesture bracket suppresses the snapshot as
    // usual). A parent that loses its last child becomes a producing leaf
    // again. Refuses the whole call, model untouched, for an unknown id,
    // tube 0 or any other L1 root: an L1 tube belongs to its graph region
    // (plan/18 §7 G14) and the next region sync would only rebuild it.
    // `outRemoved` (optional) receives how many tubes went, descendants
    // included. Bumps version + map.
    bool RemoveTubes(std::vector<int> const &tubeIds, int *outRemoved);
    int GetTubeCount() const;  // tube 0 (when built) + the store
    // Level/children/centers of any tube (0 included). Children list in
    // ascending id order.
    int GetTubeLevel(int tubeId) const;  // -1 when missing
    std::vector<int> GetTubeChildren(int tubeId) const;
    int GetTubeCenterCount(int tubeId) const;  // -1 when missing
    bool GetTubeCenterCV(int tubeId, int cv, float *x, float *y,
                         float *z) const;
    // World position of the visible/editable center-CV handle. This is the
    // actual section polygon's area centroid at the CV parameter; it does
    // not replace the raw authored center returned by GetTubeCenterCV.
    bool GetTubeCenterHandle(int tubeId, int cv, float *x, float *y,
                             float *z) const;
    // Move one center CV with K6-down / K7-up propagation (locks gate
    // each direction). Bumps the version once. Tube 0 delegates to the
    // live members (soft selection included), then propagates.
    bool MoveTubeCenterCV(int tubeId, int cv, float dx, float dy, float dz);
    // Translate every center CV of one tube as one rigid center-cage edit.
    // Unlike N calls to MoveTubeCenterCV this takes one hierarchy snapshot
    // and runs K6/K7 once, so a parent whole-tube Move never derives its
    // children through intermediate bent parent poses. Soft selection is
    // intentionally not involved: whole-tube translation is exact.
    bool TranslateTube(int tubeId, float dx, float dy, float dz);
    // Flattened center deltas (du, dv, dw per CV) of tubeId; zeros for
    // tube 0 and unedited tubes.
    std::vector<float> ReadTubeDeltas(int tubeId) const;
    // Per-tube (-1 = global default) propagation gates. Effective lock =
    // global OR per-tube. Tube 0's per-tube flags are the existing
    // _lockParents/_lockChildren members (commit-visible).
    void SetTubeLockParents(int tubeId, bool on);
    void SetTubeLockChildren(int tubeId, bool on);
    // K7 on-the-fly parent over tubeIds; returns its (negative) tube id.
    // transient selects the transient flag. Bumps version + map.
    bool GroupTubes(std::vector<int> const &tubeIds, bool transientParent,
                    int *outParent);
    bool MakeTubePersistent(int tubeId);
    // P6 undo: restore the most recent pre-mutation snapshot (tube +
    // hierarchy + guides + the graph, V1). Empty stack is a no-op
    // success. Bumps version + map and dirties points, topology and
    // guides so workers re-derive; `outDirty` (optional) receives those
    // bits so the caller knows what to publish. The state it leaves goes
    // onto the redo stack.
    bool Undo(uint32_t *outDirty = nullptr);
    int GetUndoDepth() const;       // snapshots currently held
    uint64_t GetUndoBytes() const;  // their accounted bytes
    // Budget: max snapshots + max bytes (either bound evicts oldest;
    // depth 0 disables and clears). Rejects negative depth. Applies
    // immediately (shrinks the live stack to fit).
    bool SetUndoBudget(int maxDepth, uint64_t maxBytes);
    void ClearUndo();
    // Sculpt stroke: apply per-CV deltas (brush shapes arrive precomputed;
    // smooth adds one 0.5 relax pass), optionally rescale to the pre-stroke
    // length (mirrorX negates dx), then K6-down / K7-up like a move.
    bool SculptStroke(int tubeId, const char *brush, int const *cvIds,
                      float const *deltas, int cvCount, bool preserveLength,
                      bool mirrorX);
    // One SHAPED sculpt stroke (plan/18 §7 G13): the caller passes the
    // stroke, not per-CV deltas, and the brush maths lives here.
    //
    //   `viewProj`/`w`/`h`/`x`/`y`/`radiusPx` give the screen falloff (the
    //   same row-major projection and top-left pixels Pomade_Pick takes);
    //   an active gesture freezes Grab's footprint against its press-time
    //   center curve while each supplied delta still applies incrementally
    //   to the current curve; other brushes shape against the current curve;
    //   `tCenter`/`tRadius` the falloff by t along the curve (tRadius <= 0
    //   means no t bound); `deltaWorld` the drag (grab) or push direction
    //   (comb); `amount` the brush scalar -- comb push distance, smooth
    //   strength, lengthen fraction, twist radians -- at full weight.
    //
    // mirrorX applies the mirrored stroke to the tube symmetric about
    // x = 0 (the nearest root within kPomadeMirrorTolerance of the mirrored
    // root), which is what "symmetry" means to an artist; the old ABI only
    // negated dx on the same tube. *outTouched, when non-null, receives
    // the number of CVs the stroke actually moved.
    bool SculptStrokeShaped(int tubeId, const char *brush,
                            float const *viewProj, int w, int h, float x,
                            float y, float radiusPx, float const *deltaWorld,
                            float amount, float tCenter, float tRadius,
                            bool preserveLength, bool mirrorX,
                            int *outTouched);
    // K13 per-CV kink scores for tube 0 (empty when no tube is built).
    // The HUD warns when any score reaches kPomadeSmoothnessSpike.
    std::vector<float> SmoothnessScores() const;
    // K12 root-overlap check over tube 0 (when built) plus the store.
    // Caches per-tube flags (no version bump: a diagnostic, run
    // post-gesture by the caller, never mid-drag).
    bool CheckRootIntersections();
    // Tube ids flagged by the last CheckRootIntersections (ascending).
    std::vector<int> IntersectedTubes() const;

    // -- P5: bridges (plan/17 §5.7) --------------------------------------
    // All tube ids in the model (tube 0 when built, then the store),
    // ascending. Export enumerates a level through this + GetTubeLevel.
    std::vector<int> TubeIds() const;
    // Import explicit centers + sections as a locked child of parentId
    // (curves: centers from the curve, sections inherited; meshes: both
    // from the rings — the caller decomposes). secT takes nSec values,
    // secU/secV nSec * ringVerts row-major; scale/twist default to 1/0.
    // The child id is parentId * 16 + 1 + childIndex with the next free
    // childIndex, so imports never collide with subdivided siblings.
    // Bumps version + map. No propagation: existing shapes are untouched,
    // and the import itself is propagation-immune (K6/K7 read it, never
    // write it) while direct edits to it still propagate both ways.
    bool ImportLockedTube(int parentId, float const *cx, float const *cy,
                          float const *cz, int nCv, float const *secT,
                          float const *secU, float const *secV, int nSec,
                          int ringVerts, int *outTubeId);
    // Swept-mesh import: ring-major world points (ringCount * ringVerts,
    // 3 floats each, no caps). Centers are the ring centroids; sections
    // are the rings projected into the K4 charts at uniform t. Same
    // locked semantics, id scheme, and version/map bumps as above.
    bool ImportSweptMesh(int parentId, float const *points, int ringCount,
                         int ringVerts, int *outTubeId);
    // Section census + reads for any tube (tube 0 included). Count is -1
    // when the tube is unknown; GetTubeSection is false then too.
    int GetTubeSectionCount(int tubeId) const;
    bool GetTubeSection(int tubeId, int ring, PomadeTubeSection *out) const;
    // True when the tube is a bridge import (false when unknown).
    bool IsTubeImported(int tubeId) const;

    // -- V0b: the stage contract (plan/18 §7 G1–G3) ------------------------
    //
    // The committer serialises the WHOLE hierarchy and hydrate rebuilds it,
    // so the model has to hand out (and take back) one tube's complete
    // authoring state. GetTubeRecord/RestoreTubeRecord are that pair; every
    // field round-trips bit-exactly, which is what makes the commit ->
    // hydrate -> commit assertion in testUsdGenPomadeCommit meaningful.
    struct TubeRecord {
        PomadeTubeDesc actual;      // authored shape (derived + deltas)
        PomadeTubeDesc derived;     // re-derived from the parent (empty at L1)
        PomadeShapeDeltas deltas;   // in the derived frames
        SubdivideParams subdivide;  // params this tube splits its children with
        FillParams fill;
        bool locked = false;        // tube 0: the global lock; children: import
        bool lockParents = false;
        bool lockChildren = false;
        bool transientParent = false;
        bool persistent = false;
        bool imported = false;
        // False only while hydrating a legacy layer that did not author the
        // inherited-boundary binding attribute. Current empty vectors are
        // meaningful and remain authored.
        bool hasInheritedBoundaryBindings = true;
        std::vector<int> members;   // group-parent members (empty otherwise)
        bool hasTube = false;
    };
    bool GetTubeRecord(int tubeId, TubeRecord *out) const;
    // Drop every store tube (tube 0 is untouched) and reset the group-id
    // mint, so a hydrate rebuilds the hierarchy into a known-empty model
    // and re-mints the same ids. Clears the undo stack: its snapshots name
    // tubes that no longer exist. Bumps the version + map version.
    void ClearHierarchy();
    // The authored shape of any tube (0 included), with tubeId/level/
    // regionId/parentTubeId/childIndex filled in. False when unknown.
    bool GetTubeDesc(int tubeId, PomadeTubeDesc *out) const;
    // Install an authored shape + deltas + params on an EXISTING store tube
    // (the hydrate write path; tube 0 goes through Restore instead). The
    // deltas are stored verbatim, never recomputed, so a hydrated tube
    // reports exactly the bytes the committer wrote. Bumps the version.
    bool RestoreTubeRecord(int tubeId, TubeRecord const &record);
    // Install an L1 root read back from the stage (hydrate, plan/18 §7 G14).
    // `tubeId` must be a free, positive multiple of 16 (the L1 id family);
    // the first L1 root restores through Restore() instead.
    bool InstallL1Tube(int tubeId, TubeRecord const &record);

    // -- V0b: per-tube editing (plan/18 §7 G2) -----------------------------
    //
    // Every §5.2/§5.3 operation takes a tube id; the tube-0 spellings above
    // are thin wrappers over id 0 and keep their exact behaviour. On a
    // derived child the edit lands on `actual`, the deltas are recomputed
    // against `derived`, and K6/K7 propagate both ways as MoveTubeCenterCV
    // does. Operations that would change a derived child's LAYOUT (center CV
    // or section ring counts) are refused: the layout belongs to the
    // parent's subdivision, so the parent is the place to edit (an imported
    // tube has no derivation and accepts them).
    bool InsertTubeCenterCV(int tubeId, int atIndex);
    bool DeleteTubeCenterCV(int tubeId, int index);
    bool SetTubeLengthFor(int tubeId, float length);
    bool MatchTubeSurface(int tubeId);
    bool SnapTubeRootToScalp(int tubeId);
    bool RelaxTubeCenter(int tubeId, float strength, int iterations);
    bool MoveTubeSectionRing(int tubeId, int ring, float du, float dv);
    bool ScaleTubeSectionRing(int tubeId, int ring, float scale);
    bool TwistTubeSectionRing(int tubeId, int ring, float radians);
    bool MoveTubeSectionCV(int tubeId, int ring, int slot, float du, float dv);
    bool AddTubeSectionRing(int tubeId, float t);
    bool RemoveTubeSectionRing(int tubeId, int ring);
    bool CopyTubeSectionRing(int tubeId, int src, int dst);
    bool SetTubeFillParams(int tubeId, FillParams params);
    bool GetTubeFillParams(int tubeId, FillParams *out) const;
    // True while the tube has children: its own fill is suspended and the
    // children carry the density (§2.3). False for a leaf or unknown tube.
    bool IsTubeFillSuspended(int tubeId) const;

    // -- P3: Fill mode (plan/17 §5.3, K8–K10) --------------------------------
    //
    // RefillGuides regenerates roots + guides through K8/K9/K10. A
    // fraction in [0, 1) refills at preview-scaled density (the live
    // move); 1.0 refills at full density (the release). Negative selects
    // the stored preview fraction (default 0.25). With freeze-roots on,
    // the kept prefix survives and only the tail is re-sampled.
    bool SetPreviewFraction(float fraction);
    float GetPreviewFraction() const;
    void SetFreezeRoots(bool freeze);
    bool GetFreezeRoots() const;
    // Regenerate the live guide preview at `fraction` of full density
    // (negative = the panel's preview fraction). Every PRODUCING tube
    // fills from its own fill params -- a subdivided parent's fill is
    // suspended (§2.3), imports and group parents never fill -- and the
    // per-tube sets merge in ascending tube order with running curve ids
    // from 1000, so a one-tube groom refills bit-exactly as it always
    // has. Region faces partition from each L1 root down to the child
    // cell that claims them, exactly as the committer partitions them,
    // so the live preview and the committed Guides agree whenever the
    // region maps are current. True when at least one tube filled; a
    // degenerate tube is skipped, never fatal to its siblings.
    bool RefillGuides(float fraction);
    // The producing tubes the last refill skipped, as (tubeId, reason) in
    // fill order; empty when every tube filled. Skipped tubes contribute
    // no guides, so the dock warns about them instead of letting a
    // partial refill pass as a healthy one.
    std::vector<std::pair<int, std::string>> RefillDrops() const;
    // Explicit Fill after Clear: restores generation. Ordinary refills are
    // auto-refreshes and respect a cleared cache.
    bool GenerateGuides(float fraction);
    bool ClearGeneratedCurves();
    bool SetGeneratedCurvesVisible(bool visible);
    bool GetGeneratedCurvesVisible() const;
    bool GeneratedCurvesSuppressed() const;
    // Hydrate installs this persisted authoring state without an undo step.
    void SetGeneratedCurvesSuppressed(bool suppressed);
    struct GuidePreview {
        std::vector<float> points;  // 3 floats per CV, guide-major
        std::vector<int> counts;
        std::vector<int> tubeIds;  // owning tube per guide, ascending merge
        int guideCount = 0;
        int cvCount = 0;
    };
    GuidePreview GetGuidePreview() const;  // the current set (preview
                                          // density mid-drag, full on release)
    PomadeGuideSet const &GetGuides() const;  // UI-thread-only, current set
    std::vector<PomadeGuideRoot> const &GetRoots() const;  // UI-thread-only

    // -- P3: pick (plan/17 §4.6, K11) -----------------------------------------
    // Screen-space pick over the host mirrors with a kind mask (a
    // PomadePickKind OR). Guide candidates come from the preview set.
    // Heavyweight kinds (tube verts, guide CVs) reduce on the device
    // through the K11 lane when the mirror is healthy and the candidate
    // count clears the threshold (TN-2); small kinds always run the CPU
    // twin, and any device failure falls back to it wholesale. Same
    // rule and tie order either way; tube-vert winners can differ from
    // the CPU twin in photo-finishes (device tessellation carries
    // transcendental wobble vs the host mirror — and matches the
    // rendered pixels exactly, which is what the pick should hit).
    // Raw K11 candidate pick. Its TubeVert result is always an actual
    // tessellated vertex and `distPx` is that vertex's screen distance; the
    // C ABI Pomade_Pick exposes this record verbatim for GPU/CPU parity.
    PomadePickHit Pick(float const viewProj[16], int w, int h, float x,
                      float y, float radiusPx, uint32_t kindMask) const;
    // Selection/hover pick over the same candidates, with displayed-handle
    // priority and a visible tube-face owner fallback. This deliberately
    // does not share the raw Pomade_Pick distance/index contract.
    PomadePickHit PickItem(float const viewProj[16], int w, int h, float x,
                           float y, float radiusPx,
                           uint32_t kindMask) const;

    // -- V0: what the viewport publishes (plan/18 §2.1, §2.2) -------------
    //
    // The Pomade scene index no longer owns a model; it stages from the
    // registry's active one. These are the read-only views it needs, plus
    // the per-level display state the artist drives. Display state is
    // viewport-only: never authored to USD, never on the undo stack.

    // One tube as the publisher sees it. `desc` carries tubeId, level,
    // regionId, parentTubeId and childIndex, so the publisher needs no
    // further per-tube call. Ascending by id, tube 0 first when it is
    // built. One lock, one walk: a publisher that interleaved
    // GetTubeLevel/GetTubeCenterCV per tube could observe a torn model.
    struct TubeView {
        PomadeTubeDesc desc;
        bool imported = false;
        bool persistent = false;
        bool transientParent = false;
    };
    std::vector<TubeView> SnapshotTubes() const;
    // The deepest level any tube sits at; 0 when the model holds no tube.
    int GetMaxTubeLevel() const;

    struct LevelDisplay {
        bool visible = true;
        bool xray = false;
        // The alpha an x-rayed level draws with. plan/18 §2.4a puts the
        // focused level at 25 % and the levels around it fainter, so the
        // strength is per level, not one material parameter: a mode that
        // ghosts everything except the tube being edited needs two values
        // on screen at once. Ignored while `xray` is false.
        float xrayOpacity = kDefaultXrayOpacity;
        // Centers-only: the level draws its center curves and CV dots and
        // nothing else (no tube mesh, no rings, no guides). The fallback
        // ladder's fourth step (plan/18 §3.7) for every level outside the
        // edited subtree; never authored to USD.
        bool centersOnly = false;
        // Whether the center curves and CV dots draw at all. Graph hides
        // authoring helpers behind the region surface; Output hides its
        // authoring tubes and cage so committed amplified tiles remain
        // visible. The editable modes choose opaque surfaces or x-rayed
        // controls as their policy requires.
        bool centers = true;
        // The visible point controls name one explicit Tube sub-mode. The
        // center curve may remain as a guide while its inactive dot set is
        // hidden; these flags never change the underlying pick domains.
        bool centerCVDots = true;
        bool ringCVDots = true;
        // Whether the K9/K10 interactive guide preview draws. It appears
        // in Fill; Output presents the committed amplified tile output and
        // therefore hides this helper. This is ANDed with "show amplified
        // hair", which is the artist's switch and is not overridden here.
        bool guides = true;
    };
    // The plan/18 §2.4a x-ray strengths: the focused level, and every
    // other level behind it.
    static constexpr float kDefaultXrayOpacity = 0.25f;
    static constexpr float kFaintXrayOpacity = 0.10f;
    // Per-level (1-based) visibility and x-ray. A change marks Display
    // dirty and bumps the version; an untouched level reads the default
    // (visible, opaque). Levels below 1 are rejected.
    bool SetLevelDisplay(int level, bool visible, bool xray);
    // The same, plus the centers-only flag (the two-argument form leaves
    // whatever centers-only state the level already has alone).
    bool SetLevelDrawMode(int level, bool visible, bool xray,
                          bool centersOnly);
    // The whole record at once: what PomadeDisplayPolicy drives, and the
    // only entry that can set the x-ray strength or hide the centers.
    bool SetLevelDraw(int level, LevelDisplay const &draw);
    LevelDisplay GetLevelDisplay(int level) const;
    // Which tubes draw their cross-section rings and ring CV dots
    // (plan/18 §2.4a: "shown only in Tube mode (Ring/Section sub-modes) and
    // on selected tubes elsewhere"). Model state, because the publisher has
    // to know it while it lays the level out; the loops set it on every
    // mode/sub-mode change. Marks Display dirty and bumps the version.
    enum RingDisplay {
        Rings_Off = 0,       // no rings anywhere
        Rings_Selected = 1,  // the default: rings on selected tubes only
        Rings_All = 2,       // Tube mode's Ring and Section sub-modes
    };
    bool SetRingDisplay(int mode);
    int GetRingDisplay() const;
    // "Show amplified hair" (plan/17 §3.2, plan/18 §7 G7). When it is on,
    // the usdGen cook's amplified tiles over the committed guides are what
    // the artist looks at and the Pomade guide preview steps aside; when it
    // is off, the preview draws and the tiles are hidden. During a gesture
    // the tiles are hidden either way: they belong to a stage version older
    // than the drag, so leaving them up would draw last-release's hair over
    // this move's tubes. Marks Display dirty and bumps the version.
    bool SetAmplifiedHair(bool show);
    bool GetAmplifiedHair() const;
    // What the index should draw right now, resolved against the gesture
    // bracket: `tiles` false hides the cook's output, `guides` false hides
    // the Pomade preview.
    void ResolveHairDisplay(bool *tiles, bool *guides) const;
    // The focused level: thick center curves and large CV dots (plan/18
    // §2.4a). 0 = no focus. Marks Display dirty and bumps the version.
    bool SetFocusLevel(int level);
    int GetFocusLevel() const;
    // Per-branch hierarchy navigation is opt-in. While disabled, every
    // live tube remains visible and the legacy focus-level policy applies.
    // While enabled, the visible frontier contains each unexpanded tube
    // whose ancestors are all expanded; expanding a parent replaces it with
    // its direct children without affecting unrelated branches.
    bool SetActiveCutEnabled(bool enabled);
    bool GetActiveCutEnabled() const;
    bool SetTubeExpanded(int tubeId, bool expanded);
    bool GetTubeExpanded(int tubeId) const;
    bool IsTubeVisibleInActiveCut(int tubeId) const;
    // World units per screen pixel at the focus point, as the viewport
    // controller measures it (pomadeCamera.worldPerPixel). Storm sizes
    // points and curves in WORLD units, so this is the only way an overlay
    // dot can be a fixed number of pixels across: the publisher multiplies
    // the plan/18 §2.4a pixel targets (8 px / 5 px CV dots, 3 px / 1 px
    // curves) by this scale. 0 means "no camera has been resolved yet" —
    // a headless model, a record harness that never called it — and the
    // publisher falls back to sizing the overlays against each tube's own
    // root radius, which is what V0–V7 did everywhere. Marks Display dirty
    // and bumps the version, so a camera move restages four small width
    // arrays and nothing else.
    bool SetDisplayScale(float worldPerPixel);
    float GetDisplayScale() const;
    // The stage path of the groom this model commits to, e.g. "/World/Groom"
    // (PomadeSession sets it beside Pomade_CommitterCreate). It exists for
    // one reason: while the model is live, the committed `<groom>/Guides`
    // prim is last commit's curves drawn on top of this gesture's tubes,
    // so the scene index authors visibility = false over it and restores
    // it on deactivate. The override is a Hydra opinion only — the
    // amplifier reads the stage, so no cook is starved. Empty hides
    // nothing. Marks Display dirty and bumps the version.
    bool SetGroomPath(std::string const &path);
    std::string GetGroomPath() const;

    // -- V1: selection (plan/18 §2.3) --------------------------------------
    //
    // Model state, so it survives a commit and every mode reads one set.
    // Never on the undo stack. Each mutator marks Selection dirty and bumps
    // the version ONLY when something actually changed, so a re-click on an
    // already-selected CV publishes nothing.
    //
    // Reads prune themselves: any item naming a tube that no longer exists
    // (merged, undone, re-hydrated) is dropped when the selection is read,
    // which is why no topology path has to remember to clean up. The one
    // topology change that MOVES a selection rather than dropping it is a
    // subdivide, and SubdivideTube does that through PomadeSelection::
    // RemapTube (parent tube -> its children).
    void SelectionClear(uint32_t kindMask);
    void SelectionApply(PomadeSelectMode mode,
                        std::vector<PomadeSelectionItem> const &items);
    void SelectionSetHover(PomadeSelectionItem const &item);
    PomadeSelectionItem SelectionHover() const;
    std::vector<PomadeSelectionItem> SelectionItems(uint32_t kindMask) const;
    size_t SelectionCount(uint32_t kindMask) const;
    // Bumped by every real change; the publisher keys its per-level
    // selection hash off the items and the index off this.
    uint64_t SelectionGeneration() const;
    // World bounds of everything selected (gizmo placement, frame
    // selected). False — outputs untouched — when nothing selected has a
    // position in this model.
    bool SelectionBounds(float outMin[3], float outMax[3]) const;
    // The selection item one pick hit names: the pick reports candidate
    // ordinals (a graph node's array slot, a CV index), the selection
    // stores stable ids. One translation, used by the point pick and the
    // marquee alike, so they can never name different things.
    PomadeSelectionItem SelectionItemFromHit(PomadePickHit const &hit) const;

    // Marquee / lasso over the same candidate sets and projection Pick
    // uses (plan/18 §2.3). Pixels, top-left origin, like Pick. The
    // heavyweight kinds (tube verts, guide CVs) mask on the device when
    // the candidate count clears the pick threshold and fall back to the
    // CPU twin wholesale on any device failure. `mode` is set / add /
    // toggle. False only on bad arguments.
    bool SelectRect(float const viewProj[16], int w, int h, float x0,
                    float y0, float x1, float y1, uint32_t kindMask,
                    PomadeSelectMode mode);
    bool SelectPolygon(float const viewProj[16], int w, int h,
                       float const *xy, int pointCount, uint32_t kindMask,
                       PomadeSelectMode mode);

    // -- V1: gizmo and brush overlays (plan/18 §2.4) -----------------------
    //
    // Pure display records the tool sets and the scene index draws; never
    // authored to USD, never on the undo stack. A set that changes nothing
    // marks nothing dirty (the move loop sets the same brush ring on every
    // sample the cursor does not move).
    bool SetGizmo(PomadeGizmoRecord const &record);
    PomadeGizmoRecord GetGizmo() const;
    bool SetBrushRing(PomadeBrushRingRecord const &record);
    PomadeBrushRingRecord GetBrushRing() const;

    // -- V1: the gesture bracket (plan/18 §3.2, plan/17 §1.3) --------------
    //
    // Before V1 every mutation pushed its own full snapshot, so a 200-
    // sample drag pushed 200 of them and evicted the artist's history
    // (plan/18 §3.2). Begin takes ONE snapshot and suppresses the rest;
    // End seals it; Cancel restores the press-time base.
    //
    //   Begin  — error when a gesture is already open (nesting is a bug in
    //            the controller, not a stack to balance). `label` names the
    //            undo step ("" -> "edit").
    //   End    — seals and bumps the version once, so the committer has one
    //            coalescing key for the whole drag.
    //   Cancel — restores the base, drops the snapshot Begin pushed, does
    //            NOT bump the version (the stage never saw the drag), and
    //            reports the dirty bits the caller must publish: the
    //            viewport DID see it and has to be put back.
    bool BeginGesture(char const *label);
    bool EndGesture();
    bool CancelGesture(uint32_t *outDirty);
    int GetGestureDepth() const;

    // -- V1: redo and undo labels (plan/18 §2.5) ---------------------------
    //
    // Undo pushes the state it leaves onto the redo stack; any new mutation
    // clears it (the usual branch rule). Both report the dirty bits to
    // publish. GetUndoLabel(depth): depth 0 is the step Ctrl+Z would undo,
    // and NEGATIVE depths address the redo stack (-1 is the step Ctrl+Y
    // would redo), which is exactly what an Edit strip shows.
    bool Redo(uint32_t *outDirty);
    int GetRedoDepth() const;
    bool GetUndoLabel(int depth, std::string *out) const;

private:
    bool _TessellateHost();
    bool _TessellateSectionsHost();  // P3 K5 path (validates + fills _frames)
    // The tube desc from live members (internal twin of BuildTubeDesc).
    // The *Locked suffix on internals predates the single-writer rule; it
    // now means "called from inside a public entry", nothing more.
    PomadeTubeDesc _BuildTubeDescLocked() const;
    void _AttachTubeRootLocked(PomadeTubeDesc *desc) const;
    // P4 hierarchy store (authoring thread). Holds every
    // tube but the first L1 root: subdivided children, imports, on-the-fly
    // parents AND the second and later L1 roots (parentTubeId -1, level 1).
    std::map<int, HierarchyTube> _tubes;  // non-zero tubes by id
    // Region ownership (plan/18 §7 G14). _regionTube is keyed by the
    // CURRENT region id and rebuilt by SyncRegionTubes after every graph
    // edit; _tubeRegionKey holds the sorted node loop each L1 tube was
    // rooted in, which is the identity that survives re-extraction.
    std::map<int, int> _regionTube;
    std::map<int, std::vector<int>> _tubeRegionKey;
    // The smallest free positive multiple of 16 (the next L1 root id).
    int _NextL1TubeIdLocked() const;
    // Sorted alive node ids of region `regionId`'s loop, or empty.
    std::vector<int> _RegionKeyLocked(int regionId) const;
    // Root placement + boundary-fitted root section for a region: the
    // area-weighted centroid and normal of its claimed faces, a center
    // column of `centerCount` CVs along that normal, and a root ring whose
    // `ringVerts` CVs sit on the region boundary in the root plane (plan/17
    // §5.2 "fitted to the boundary shape", plan/18 §7 G12).
    // `outRadius` receives the mean fitted radius (the display width scale).
    bool _RegionTubeDescLocked(int regionId, int centerCount, int ringVerts,
                               float length, PomadeTubeDesc *out,
                               float *outRadius);
    // L1TubeIds under the caller's lock.
    std::vector<int> _L1TubeIdsLocked() const;
    // True when any descendant of `tubeId` carries a non-zero shape delta
    // or is a bridge import: re-deriving the root would throw that work
    // away, so §5.1 asks the artist to merge first instead.
    bool _SubtreeCarriesDeltasLocked(int tubeId) const;
    // Remove `tubeId` and every descendant from the store (never tube 0).
    void _DropSubtreeLocked(int tubeId);
    // True when `tubeId` has a non-imported child in the store: a bridge
    // import is not a subdivision, so a parent that only carries imports
    // keeps its own fill (the snapshot's rule, pomadeCommit.cpp).
    bool _FillSuspendedLocked(int tubeId) const;
    // Re-root an existing L1 tube on `regionId`, keeping its CV count,
    // ring CV count and length; children re-derive (K6 down, K7 up).
    bool _RerootTubeLocked(int tubeId, int regionId);
    // Write a desc into tube 0's legacy members (shape, centers, sections)
    // without touching undo or the version.
    bool _InstallTube0DescLocked(PomadeTubeDesc const &desc);
    // SyncRegionTubes under the caller's lock.
    bool _SyncRegionTubesLocked(std::vector<int> *outRebuilt,
                                std::vector<int> *outRemoved);
    std::map<int, int> _intersectFlags;   // K12 cache: tube id -> 0/1
    int _nextGroupId = -1;                // group parents mint negatives
    bool _globalLockParents = false;
    bool _globalLockChildren = false;
    // Desc of any tube (0 from members); false when tubeId is unknown.
    bool _TubeDescLocked(int tubeId, PomadeTubeDesc *out) const;
    // Absolute center write on tube 0 + Sync; no version bump (caller).
    bool _SetCenterCVLocked(int cv, float x, float y, float z);
    // Replace tube 0's members from a desc (merge-into-0); Syncs + marks
    // topology dirty; no version bump (caller).
    bool _WriteDescToTube0Locked(PomadeTubeDesc const &desc);
    // K6 re-derive of tubeId's whole subtree from its (new) actual;
    // K7 refresh of its ancestors. Authoring thread; no version bump.
    struct HierarchyRollback;
    bool _PropagateDownLocked(int tubeId);
    // Attachment-only K6: after a graph region edit changes an L1 root's
    // footprint, descendants retain their upper sculpt while their inherited
    // base center/section reattach to the freshly derived parent boundary.
    bool _PropagateAttachmentDownLocked(int tubeId);
    bool _ConformRegionRootSectionLocked(int tubeId,
                                         PomadeScalpGraph const &oldGraph,
                                         int oldRegionId, int newRegionId,
                                         PomadeTubeDesc const &oldActual,
                                         bool *outChanged);
    // `beforeEdit` is the immutable hierarchy at the start of a direct
    // child edit.  K7 uses it only to tell a moved inherited boundary slot
    // from an internal child edit; ordinary callers may omit it.
    bool _PropagateUpLocked(int tubeId,
                            HierarchyRollback const *beforeEdit = nullptr);
    // One sculpt stroke's mutation (deltas already shaped): apply, keep
    // the length, re-derive deltas, K6 down and K7 up. The CALLER
    // owns the rollback snapshot, the undo push and the version
    // bump, which is how a mirrored stroke stays one undo step.
    // `beforeEdit` is the hierarchy before the entire shaped stroke; K7
    // needs it to distinguish an inherited boundary from an internal CV.
    bool _SculptApplyLocked(int tubeId, std::string const &brush,
                            int const *cvIds, float const *deltas,
                            int cvCount, bool preserveLength, bool mirrorX,
                            HierarchyRollback const *beforeEdit = nullptr);
    // V0b per-tube edits (authoring thread). _ChildDescLocked hands out
    // the child's mutable record; _FinishChildEditLocked recomputes the
    // deltas against the derived shape (imports keep theirs at zero),
    // propagates K6 down and K7 up, and leaves the version bump to the
    // caller. `layoutChanged` is refused on a derived child.
    bool _FinishChildEditLocked(int tubeId, bool layoutChanged,
                                uint32_t dirtyBits,
                                HierarchyRollback const *beforeEdit = nullptr);
    // Take the lock, snapshot for rollback, run `edit` over the child's
    // authored shape, finish as above, seal one undo step and bump the
    // version once. Any failure restores the snapshot wholesale.
    bool _EditChildTube(int tubeId, bool layoutChanged, uint32_t dirtyBits,
                        std::function<bool(PomadeTubeDesc &)> const &edit);
    bool _RefillGuidesLocked(float fraction);
    // Import core (authoring thread): validates + inserts, bumps
    // version + map. Shared by ImportLockedTube/ImportSweptMesh.
    bool _ImportLockedTubeLocked(int parentId, float const *cx,
                                 float const *cy, float const *cz, int nCv,
                                 float const *secT, float const *secU,
                                 float const *secV, int nSec, int ringVerts,
                                 int *outTubeId);
    // The graph half of an undo step (V1, plan/18 §2.5). Held by shared
    // pointer and cached against _mapVersion: a tube edit's snapshot
    // copies the pointer (the graph cannot have changed), and only a
    // graph edit pays for a real copy. Restoring goes through the public
    // PomadeScalpGraph::Restore + a re-rasterise, exactly like hydrate.
    struct GraphUndoState {
        std::vector<PomadeGraphNode> nodes;
        std::vector<std::pair<int, int>> edges;
        std::vector<std::pair<int, int>> linked;
        float snapRadius = 0.05f;
    };
    // Rollback snapshot: moves/sculpts restore everything when a later
    // propagation step fails (no partial application).
    struct HierarchyRollback {
        std::map<int, HierarchyTube> tubes;
        std::vector<float> cx, cy, cz;
        std::vector<PomadeTubeSection> sections;
        PomadeTubeShape shape;
        bool useSections = false;
        std::vector<PomadeGuideRoot> roots;
        std::map<int, std::vector<PomadeGuideRoot>> tubeRoots;
        PomadeGuideSet guides;
        bool generatedCurvesSuppressed = false;
        OutputSettings output;
        // Tube 0's fill params live outside _tubes, so they travel here: a
        // Fill-panel density edit is an undo step (SS-02) and Ctrl+Z has
        // to put the density back, not only the guides it grew.
        FillParams fill;
        std::shared_ptr<GraphUndoState const> graph;  // null = no scalp
        // Region ownership travels with the step (plan/18 §7 G14): undoing
        // a stub build has to put the region->tube map back, or the next
        // build would append a second tube for the same region.
        std::map<int, int> regionTube;
        std::map<int, std::vector<int>> tubeRegionKey;
        int tube0Region = -1;
        bool tube0RootFramePinned = false;
        PomadeFrame tube0RootFrame;
        std::array<float, 9> tube0FrameReference = {{
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f}};
        std::string label;  // what the Edit strip calls this step
    };
    HierarchyRollback _SnapshotHierarchyLocked() const;
    void _RestoreHierarchyLocked(HierarchyRollback const &snap);
    // Move every L1 tube whose stable region loop survives a geometry-only
    // graph edit, along with its complete stored subtree. `reference` is the
    // press-time state for a gesture (or the immediately preceding state for
    // a one-shot edit), so successive samples never compound transport.
    bool _TransportRegionAttachmentsLocked(
        HierarchyRollback const &reference,
        std::vector<int> const &movedNodeIds);
    // P6 undo: bounded stack of pre-mutation snapshots (tube +
    // hierarchy + guides state — the sculpt/move/subdivide/merge
    // surface; graph/region state and deterministic refills are not
    // undo steps). _PushUndoLocked snapshots the CURRENT state and
    // enforces both bounds (oldest evicted first); the _Snapshot
    // variant pushes a snapshot the caller already took (the rollback
    // pattern reuses its failure snapshot for free). Builds and scalp
    // binds clear the stack (topology invalidation).
    void _PushUndoLocked();
    void _PushUndoSnapshotLocked(HierarchyRollback const &snap);
    void _EnforceUndoBudgetLocked();
    void _ClearUndoLocked();
    static uint64_t _UndoSnapshotBytes(HierarchyRollback const &snap);
    // V1 undo/redo internals (authoring thread). The graph state is
    // cached against _mapVersion; restoring a step re-derives the region
    // maps the way hydrate does.
    std::shared_ptr<GraphUndoState const> _GraphUndoStateLocked() const;
    void _RestoreGraphUndoLocked(
        std::shared_ptr<GraphUndoState const> const &state);
    // Push an undo step for a graph edit that has ALREADY been applied:
    // the tube half of the model is untouched by a graph edit, so the
    // current tube state is the pre-edit one, and `before` (captured at
    // the top of the edit) supplies the pre-edit graph. Two lines per
    // graph mutator instead of a full pre-edit snapshot each.
    void _PushGraphUndoLocked(
        std::shared_ptr<GraphUndoState const> const &before,
        char const *label);
    void _ClearRedoLocked();
    // Every tube id in the model, ascending (tube 0 when built, then the
    // store). The selection prunes against this on read.
    std::vector<int> _LiveTubeIdsLocked() const;
    // Active-cut helpers. The authoring thread. A collapsed tube owns the
    // frontier and hides its descendants; an expanded tube hides itself.
    bool _IsTubeVisibleInActiveCutLocked(int tubeId) const;
    bool _IsDescendantOfLocked(int tubeId, int ancestorTubeId) const;
    void _PruneActiveCutLocked();
    // V1 selection internals (authoring thread). _BuildPickSetsLocked
    // fills `sets` from the scratch vectors it also fills, so the point
    // pick, the marquee and the selection bounds all read one candidate
    // layout. _nodeIds maps a graph-node candidate ordinal to its node id.
    struct _PickScratch {
        // The displayed surface is a set of regular ring strips.  Point
        // candidates use the flattened vertex stream below, while a body
        // click walks these strips as quads so it can name a tube even when
        // the cursor is far from every tessellated vertex.
        struct SurfaceStrip {
            int tubeId = -1;
            int level = 0;
            int firstVertex = 0;
            int ringCount = 0;
            int ringVerts = 0;
        };
        std::vector<float> center;
        std::vector<float> section;
        std::vector<float> nodes;
        std::vector<float> edgeCVs;
        std::vector<float> regionCenters;
        std::vector<float> ringCenters;
        std::vector<int> nodeIds;
        std::vector<int> edgeIds;
        std::vector<int> regionIds;
        // Multi-tube pick maps: every center candidate names its
        // (tubeId, cv); every surface candidate names its tube.
        // tubeVertPositions is only built when children exist -- the
        // single-tube path keeps pointing at the host mirror, which is
        // what the device pick mirror stays in sync with.
        std::vector<int> centerTubeIds;
        std::vector<int> centerCvIds;
        // Section candidates are a concatenated, variable-ring-vertex
        // stream.  PomadePickSets preserves its compact (ring, slot) ABI;
        // these maps recover the actual child owner from that stream.
        std::vector<int> sectionTubeIds;
        std::vector<int> sectionRingIds;
        std::vector<int> sectionSlotIds;
        std::vector<int> ringTubeIds;
        std::vector<int> ringIds;
        std::vector<float> tubeVertPositions;
        std::vector<int> tubeVertTubeIds;
        std::vector<SurfaceStrip> surfaceStrips;
        int ringVerts = 0;  // section CVs per ring (0 = no authored rings)
    };
    PomadePickSets _BuildPickSetsLocked(
        _PickScratch *scratch, uint32_t displayGraphKinds = 0) const;
    PomadeSelectionItem _ItemFromCandidateLocked(
        _PickScratch const &scratch, uint32_t kind, int index,
        int subIndex) const;
    bool _ApplyRegionSelectLocked(std::vector<PomadePickCandidate> const &hits,
                                  _PickScratch const &scratch,
                                  PomadeSelectMode mode, uint32_t kindMask);
    // The world position of one selection item; false when the item names
    // nothing this model can place (a section CV of a child tube, say).
    bool _ItemPositionLocked(PomadeSelectionItem const &item,
                             _PickScratch const &scratch,
                             float out[3]) const;
#ifdef USDGEN_POMADE_HAS_CUDA
    // Device marquee mask over one candidate array (K11b). False means
    // "scan on the CPU instead", exactly like the point pick's fallback.
    bool _SelectMaskDeviceLocked(float const *devicePositions,
                                 int candidateCount, float const viewProj[16],
                                 int w, int h, float x0, float y0, float x1,
                                 float y1, float const *xy, int pointCount,
                                 std::vector<unsigned char> *out) const;
    // Run that mask for the two heavyweight kinds when the candidate count
    // earns it, and hand back a pointer per kind (null = the CPU scan
    // handles it). One helper, so the rect and the lasso split their work
    // the same way.
    void _MaskHeavyKindsLocked(PomadePickSets const &sets, uint32_t kindMask,
                               float const viewProj[16], int w, int h,
                               float x0, float y0, float x1, float y1,
                               float const *xy, int pointCount,
                               std::vector<unsigned char> *tubeMask,
                               std::vector<unsigned char> *guideMask,
                               unsigned char const **tubePtr,
                               unsigned char const **guidePtr) const;
#endif
    bool _SyncDevice();
#ifdef USDGEN_POMADE_HAS_CUDA
    // K11 production device path (authoring thread; const: the device
    // buffers are caches). _PickDeviceLocked reduces the heavyweight
    // kinds on the model's stream and folds the CPU small-kind winner in
    // kind order with the twin's exact rule; false (any device failure)
    // means "run the CPU twin instead". The guide upload is
    // version-stamped: RefillGuides-time content, not per-pick traffic.
    bool _PickDeviceLocked(PomadePickSets const &sets, uint32_t kindMask,
                           float const viewProj[16], int w, int h, float x,
                           float y, float radiusPx,
                           PomadePickHit *out) const;
    bool _ReducePickLocked(float const *devicePositions, int candidateCount,
                           float const viewProj[16], int w, int h, float x,
                           float y, float radiusPx,
                           PomadeDevicePickBest *out) const;
    bool _UploadGuideCVsLocked(PomadePickSets const &sets) const;
    bool _SyncSectionsDevice();  // P3 K4/K5 device lane
    // P6 OOM fallback (authoring thread): destroy the stream, release
    // every device buffer, mark streamOwned false and record `reason`.
    // Later syncs take the CPU-only early-out; the host mirror is
    // authoritative either way. Idempotent after the first call.
    void _DropDeviceLocked(char const *reason);
#endif

    PomadeTubeShape _shape;
    HostTubeMesh _host;
    // P3 authored rings (default circles until a section op runs) and the
    // K4 frame cache refreshed by Sync. _useSections selects the K5
    // display path; the legacy cylinder path stays until then (the P0/P1
    // position contract).
    std::vector<PomadeTubeSection> _sections;
    std::vector<PomadeFrame> _frames;
    bool _useSections = false;
    int _segmentsPerSpan = 8;
    float _softCenter = 0.0f;
    float _softRadius = 0.0f;
    // P3 guide cache: full-density roots + guides, refilled explicitly.
    // _roots is the merged census in ascending tube order (what GetRoots
    // and the root census report); each tube's freeze store lives in
    // _tubeRoots keyed by tube id, so a frozen leaf keeps its own prefix
    // while its siblings re-sample. A one-tube groom keeps its roots in
    // _tubeRoots[0] and the merged copy in _roots, identical in content.
    std::vector<PomadeGuideRoot> _roots;
    std::map<int, std::vector<PomadeGuideRoot>> _tubeRoots;
    PomadeGuideSet _guides;
    bool _generatedCurvesSuppressed = false;
    bool _generatedCurvesVisible = true;
    float _previewFraction = 0.25f;
    bool _freezeRoots = false;
    int _tubeRegionId = -1;
    bool _rootFramePinned = false;
    PomadeFrame _rootFrame;
    std::array<float, 9> _frameReference = {{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f}};
    FillParams _fill;
    OutputSettings _output;
    SubdivideParams _subdivide;
    bool _locked = false;
    bool _lockParents = false;
    bool _lockChildren = false;
    std::atomic<uint64_t> _version{0};
    uint64_t _mapVersion = 0;  // bumped by graph/hierarchy edits only (§3.1a)
    uint32_t _dirty = PomadeDirty_Clean;
    std::string _diagnostic;
    // (tubeId, reason) per tube the last refill skipped (RefillDrops).
    std::vector<std::pair<int, std::string>> _refillDrops;
    // Single writer: every member is owned by the authoring thread. Other
    // threads read only the published state below (GetVersion is atomic).
    std::atomic<PomadeModelState const *> _published{nullptr};
    // P2 state. The bake and commit workers never read the live model:
    // the bake takes a PomadeBakeInput, the committer the published state.
    std::shared_ptr<PomadeScalpMesh> _scalp;
    PomadeScalpBvh _bvh;
    PomadeScalpGraph _graph;
    PomadeRegionMaps _maps;
    PomadeRegionLoops _loops;
    float _snapRadius = 0.05f;
    bool _mirrorX = false;
    int _ringDisplay = Rings_Selected;  // plan/18 §2.4a
    bool _amplifiedHair = false;        // plan/17 §3.2
    std::string _bakedMapFile;
    uint64_t _bakedMapVersion = 0;

    struct _Device;
    std::unique_ptr<_Device> _device;
    // P6 fallback state (authoring thread).
    bool _deviceFallback = false;
    std::string _deviceFallbackReason;
    // Guide content generation: bumped at every wholesale _guides
    // assignment. The K11 device path uploads guide CVs once per
    // generation (version-stamped), never per pick.
    uint64_t _guideVersion = 0;
    // P6 undo stack (authoring thread): pre-mutation snapshots, oldest
    // first, bounded by count AND bytes (either bound evicts oldest).
    std::deque<HierarchyRollback> _undoStack;
    uint64_t _undoBytes = 0;  // running total over the stack
    int _undoMaxDepth = 50;
    uint64_t _undoMaxBytes = 256u * 1024u * 1024u;
    // V1 redo (plan/18 §2.5): the states undo left, newest last. Any new
    // mutation clears it. Budgeted like the undo stack.
    std::deque<HierarchyRollback> _redoStack;
    uint64_t _redoBytes = 0;
    // The label the next pushed step takes; BeginGesture sets it and the
    // push consumes it, so one-shot edits outside a gesture read "edit".
    std::string _nextUndoLabel;
    // Gesture bracket (plan/18 §3.2). Depth is 0 or 1 — a nested Begin is
    // an error, not a balance to keep — and while it is 1 every
    // _PushUndoSnapshotLocked is a no-op, so a 200-sample drag is one step.
    int _gestureDepth = 0;
    HierarchyRollback _gestureBase;  // what Cancel restores
    // V1 overlay + selection state. The selection is mutable because a
    // read prunes it against the live tube ids (see SelectionItems).
    mutable PomadeSelection _selection;
    PomadeGizmoRecord _gizmo;
    PomadeBrushRingRecord _brush;
    mutable std::shared_ptr<GraphUndoState const> _graphUndoCache;
    mutable uint64_t _graphUndoCacheMap = ~uint64_t(0);
    // V0 viewport display state (authoring thread).
    // Sparse: a level with no entry reads the LevelDisplay default.
    std::map<int, LevelDisplay> _levelDisplay;
    int _focusLevel = 0;
    // Viewport-only active cut. It is deliberately separate from authored
    // hierarchy state and defaults off for ABI/persistence compatibility.
    bool _activeCutEnabled = false;
    std::set<int> _expandedTubeIds;
    float _displayScale = 0.0f;
    std::string _groomPath;
};

// Display-only graph placement shared by the scene index and interactive
// picker.  It never changes PomadeGraphNode::p, which remains the canonical
// K11/scalp-authoring coordinate.
float USDGENPOMADE_API
PomadeGraphDisplayLift(PomadeScalpMesh const *scalp);
void USDGENPOMADE_API
PomadeGraphDisplayPosition(PomadeGraphNode const &node,
                          PomadeScalpMesh const *scalp, float outP[3]);
void USDGENPOMADE_API
PomadeGraphDisplaySurfacePosition(float const canonicalP[3],
                                 PomadeScalpMesh const *scalp,
                                 float outP[3]);

// -- P3 deterministic guide fill (K8/K9/K10) ----------------------------------
//
// PomadeGenerateGuides is a pure function of the snapshot: the committer calls
// it on the worker to author <groom>/Guides, hydrate calls it on the UI
// thread to assert bit-equality with the stored guides. Guide count is
// round(density) with the root area pinned to 1 (the no-scalp path the P1
// commit test pins); roots are K8 blue-noise disc samples carried up the
// tube by K9 with edgeBias/lengthProfile, resampled by K10 with root
// frames. PomadeGenerateGuidesOnScalp is the scalp-bound form: K8 samples
// per region face with area-weighted counts (round(density * area)).
// (PomadeGuideSet itself is declared above the model.)
int USDGENPOMADE_API PomadeGuideCountForDensity(float density);
PomadeGuideSet USDGENPOMADE_API
PomadeGenerateGuides(PomadeModel::TubeSnapshot const &snapshot);
PomadeGuideSet USDGENPOMADE_API PomadeGenerateGuidesOnScalp(
    PomadeModel::TubeSnapshot const &snapshot, PomadeScalpMesh const &scalp,
    int const *regionFaces, int regionFaceCount);
// V0b: the same two fills with the root hash stream keyed by the tube id
// (plan/17 §4.1: "hash streams keyed by (tubeId, seed)"), so sibling
// children with identical fill params do not draw identical root patterns.
// The tube-id-free spellings above are these with tubeId 0.
PomadeGuideSet USDGENPOMADE_API
PomadeGenerateGuidesForTube(PomadeModel::TubeSnapshot const &snapshot,
                           int tubeId, bool usePinnedRootFrame = true);
PomadeGuideSet USDGENPOMADE_API PomadeGenerateGuidesOnScalpForTube(
    PomadeModel::TubeSnapshot const &snapshot, PomadeScalpMesh const &scalp,
    int const *regionFaces, int regionFaceCount, int tubeId,
    bool usePinnedRootFrame = true);
// Exact closed-polygon mesh fill. `sourceRegionId` addresses the graph
// region, so this remains distinct when several regions share one face.
PomadeGuideSet USDGENPOMADE_API PomadeGenerateGuidesOnRegionForTube(
    PomadeModel::TubeSnapshot const &snapshot, PomadeScalpMesh const &scalp,
    PomadeRegionLoops const &loops, int sourceRegionId, int tubeId,
    std::vector<PomadeRootOwnershipCell> const *ownership = nullptr,
    bool usePinnedRootFrame = true);
// The snapshot form of one tube's authored shape: the committer and hydrate
// both turn a TubeRecord into the TubeSnapshot the guide fills take.
PomadeModel::TubeSnapshot USDGENPOMADE_API
PomadeSnapshotFromTubeRecord(PomadeModel::TubeRecord const &record);

// Default circle sections for a snapshot without authored rings (the
// commit bytes of an untouched test tube); shared by the model, the
// committer and hydrate so all three derive the same rings.
std::vector<PomadeTubeSection> USDGENPOMADE_API
PomadeDefaultSections(PomadeModel::TubeSnapshot const &snapshot);
// The K4/K5/K9 input for a snapshot: center CVs plus authored (or
// default) sections.
PomadeTubeDesc USDGENPOMADE_API
PomadeTubeDescFromSnapshot(PomadeModel::TubeSnapshot const &snapshot);

// -- Region and clump colours ----------------------------------------------
//
// Rooted tubes and scalp patches share PomadeRegionColor, converted from
// sRGB to linear RGB for Hydra. Children inherit their L1 region id and vary
// lightness by childIndex: 0 lightens, 1 darkens, 2 lightens further, etc.
// The variation shrinks by 0.8 per level below 2 and preserves the hue.
// The legacy 16-entry palette remains available for unrooted tubes (region
// -1 wraps to slot 15); valid region colours do not wrap every 16 entries.
struct PomadeRgb {
    float r = 0.0f, g = 0.0f, b = 0.0f;
};
int USDGENPOMADE_API PomadeClumpPaletteSize();
PomadeRgb USDGENPOMADE_API PomadeClumpPaletteEntry(int slot);
PomadeRgb USDGENPOMADE_API PomadeClumpColor(int regionId, int level,
                                         int childIndex);

// -- V9 display policy (plan/18 §2.4a) ---------------------------------------
//
// THE display table. Every mode's viewport look is decided here and nowhere
// else: the Python controller calls Pomade_SetDisplayPolicy on a mode,
// sub-mode or focus change and the model, publisher and scene index follow.
// Before V9 the tool set x-ray nowhere, so every level stayed opaque in
// every mode and the center curves, CV dots and guides — all of which live
// geometrically INSIDE the tube they describe — were never once visible.
//
// The rule the reference stills state (EG 2014 Fig. 1 (c) vs (d), SIGGRAPH
// 2018 Fig. 1 middle) is that Pomade shows EITHER opaque tubes OR the control
// curves through x-rayed tubes, never both:
//
//   mode / sub-mode      focused level         other levels       rings  guides
//   ---------------------------------------------------------------------------
//   Graph                opaque, no centers    opaque, no centers  off    off
//   Tube · Ring/Section  opaque, centers       x-ray 25 %          all    off
//   Tube · Center        x-ray 25 %, centers   x-ray 10 %          sel    off
//   Hierarchy            x-ray 25 %, centers   x-ray 10 %          sel    off
//   Sculpt               x-ray 25 %, centers   x-ray 10 %          sel    off
//   Fill                 x-ray 25 %, centers   x-ray 25 %          sel    ON
//   Output               hidden                hidden              off    off
//
// Tube · Ring keeps the focused tube opaque on purpose: its rings lie ON the
// surface, so they read against it, and an x-rayed tube would only make the
// rings of the tube behind it compete with them.
//
// Fill x-rays every level at the same 25 % because what has to read there is
// the guide preview, which starts at the surface of every tube at once.
//
// With no focused level (focusLevel <= 0) every level takes the focused
// row: "nothing is focused" must not mean "everything is a ghost".
//
// The policy is a BASE state. The Levels panel's Solo / Show<=n and the
// fallback ladder's centers-only step (plan/18 §3.7) are applied on top of
// it afterwards and are not overwritten by it — SetLevelDraw carries the
// level's existing centersOnly through.
enum PomadeDisplayMode {
    PomadeDisplayMode_Graph = 0,
    PomadeDisplayMode_TubeCenter = 1,
    PomadeDisplayMode_TubeRing = 2,
    PomadeDisplayMode_Fill = 3,
    PomadeDisplayMode_Hierarchy = 4,
    PomadeDisplayMode_Sculpt = 5,
    PomadeDisplayMode_Output = 6,
    // Append only: callers and recordings retain the established ids.
    PomadeDisplayMode_TubeObject = 7,
    PomadeDisplayMode_Count = 8,
};

// The mode ids are the ones pomadeModes.MODES spells ("graph", "tube",
// "fill", "hierarchy", "sculpt", "output") and the sub-mode ids the ones
// TUBE_SUBMODES spells ("tube", "center", "ring", "section"). A null or unknown
// sub-mode falls back to that mode's default row; an unknown mode answers
// -1 and leaves the caller's display state alone.
int USDGENPOMADE_API PomadeDisplayModeFromNames(char const *mode,
                                              char const *subMode);

// The resolved draw state of one level under one display mode. Pure: no
// model, no lock, so the T0 test states the table directly.
PomadeModel::LevelDisplay USDGENPOMADE_API
PomadePolicyLevelDisplay(int displayMode, int level, int focusLevel);

// The ring display (PomadeModel::RingDisplay) the mode asks for.
int USDGENPOMADE_API PomadePolicyRingDisplay(int displayMode);

} // namespace usdGenPomade

#endif // USDGEN_POMADE_MODEL_H
