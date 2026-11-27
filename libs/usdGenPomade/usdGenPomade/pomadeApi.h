/* usdGenPomade/pomadeApi.h — the C ABI the Pomade authoring tools call.
 *
 * Control and scalars only; no array ever crosses per element (the
 * 08-tools.md C4 rule, applied to the Pomade lane). Every entry point is
 * extern "C", returns int status (0 = success, non-zero = error) except the
 * const char * accessor, and never throws across the boundary.
 * Pomade_GetLastError() returns the message for the calling thread.
 *
 * Scope, in the order it was built: model lifetime and the static test
 * tube; the commit pipeline (committer create/enqueue/swap/status); the
 * scalp graph, regions and the Ptex bake; Tube and Fill mode; the tube
 * hierarchy; picking and undo; the bridges; the V0 viewport publication
 * (registry, activate, publish, level display) that joins all of it to
 * what the artist sees; and, at the end of the file, the V1 interaction
 * surface (selection, marquee and lasso, hover, the gizmo and brush-ring
 * overlays, the gesture bracket, redo and undo labels).
 */
#ifndef USDGEN_POMADE_API_C_ABI_H
#define USDGEN_POMADE_API_C_ABI_H

#include "usdGenPomade/api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes (08-tools.md C4: 0 = success, non-zero = error). */
#define POMADE_OK               0
#define POMADE_ERROR            1  /* bad args / engine error (see GetLastError)   */
#define POMADE_NOT_IMPLEMENTED  2  /* no C++ target on this build yet — honest fail */

/* Opaque model handle. */
typedef struct PomadeModelContext PomadeModelContext;

/* Create a model holding no tube. *outCtx receives the handle. */
int USDGENPOMADE_API Pomade_Create(PomadeModelContext **outCtx);
/* Destroy a model created by Pomade_Create. NULL is a no-op success. */
int USDGENPOMADE_API Pomade_Destroy(PomadeModelContext *ctx);

/* Build the static test tube: `rings` cross-sections of `ringVerts`
 * vertices, `radius`, `length` along +Y. Non-positive values select the
 * defaults (5, 8, 0.5, 4.0). */
int USDGENPOMADE_API Pomade_BuildTestTube(PomadeModelContext *ctx,
                                        int rings, int ringVerts,
                                        float radius, float length);

/* Translate one center ring in the section plane (test-tube sculpt stub). */
int USDGENPOMADE_API Pomade_MoveCenterRing(PomadeModelContext *ctx,
                                         int ring, float dx, float dz);

/* The model's version (bumps per sealed mutation) and pending dirty bits
 * (PomadeDirty_* from pomadeModel.h, as an int). */
unsigned long long USDGENPOMADE_API Pomade_GetVersion(
    PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_TakeDirty(PomadeModelContext *ctx);

/* Tube census for staging-size queries. Zero when no tube is built. */
int USDGENPOMADE_API Pomade_GetVertexCount(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_GetQuadCount(PomadeModelContext const *ctx);

/* Copy the staged tube positions (3 floats per vertex, ring-major) into
 * `outXYZ`, which must hold at least 3 * Pomade_GetVertexCount(ctx) floats.
 * Stages through the pinned host path, so this observes the device lane
 * where one exists. */
int USDGENPOMADE_API Pomade_ReadTubePoints(PomadeModelContext *ctx,
                                         float *outXYZ, int xyzLen);

/* 1 when the model holds a CUDA device mirror, else 0. */
int USDGENPOMADE_API Pomade_HasCudaMirror(PomadeModelContext const *ctx);

/* The P6 OOM-fallback reason: "" while the mirror is healthy (or the
 * context is null), otherwise why the mirror was dropped and the model
 * continues CPU-only. Valid for the model's lifetime (sticky). */
USDGENPOMADE_API const char *Pomade_GetDeviceFallbackReason(
    PomadeModelContext const *ctx);

/* P6 undo: restore the most recent pre-mutation snapshot (tube +
 * hierarchy + guides, and from V1 the scalp graph too). An empty stack is
 * a no-op success. Bumps the model version so workers re-derive, writes
 * the dirty bits the caller must publish into `outDirty` (NULL to
 * ignore), and leaves the state it replaced on the redo stack. */
int USDGENPOMADE_API Pomade_Undo(PomadeModelContext *ctx,
                               unsigned int *outDirty);
/* Snapshots currently held and their accounted bytes (0 on null). */
int USDGENPOMADE_API Pomade_GetUndoDepth(PomadeModelContext const *ctx);
unsigned long long USDGENPOMADE_API
Pomade_GetUndoBytes(PomadeModelContext const *ctx);
/* Budget: max snapshots + max bytes (either bound evicts oldest;
 * depth 0 disables and clears). Applies immediately. */
int USDGENPOMADE_API Pomade_SetUndoBudget(PomadeModelContext *ctx,
                                        int maxDepth,
                                        unsigned long long maxBytes);
int USDGENPOMADE_API Pomade_ClearUndo(PomadeModelContext *ctx);

/* P2 scalp graph + regions (plan/17 section 5.1, K1-K3/G1). Every edit
 * bumps the model version AND the map version; K3 runs only in
 * Pomade_Rasterise (gesture end, section 4.2), never inside an edit. Nodes
 * are addressed by stable model ids; (faceId, u, v) locates surface points
 * in the PomadeFacePosition encoding. */
int USDGENPOMADE_API Pomade_BindScalp(PomadeModelContext *ctx,
                                    float const *points, int pointFloats,
                                    int const *faceVertexCounts, int faceCount,
                                    int const *faceVertexIndices,
                                    int indexCount);
/* Bind a face GeomSubset scalp (plan/02 section 2.20): the arrays are the
 * PARENT mesh's, `activeFaces` are parent-mesh face indices (repeats are
 * fine, out-of-range ids and an empty list are errors). Face ids never
 * renumber: raycasts, closest points and region rasters skip the faces the
 * subset leaves out, while every per-face read stays parent-sized. */
int USDGENPOMADE_API Pomade_BindScalpSubset(PomadeModelContext *ctx,
                                          float const *points,
                                          int pointFloats,
                                          int const *faceVertexCounts,
                                          int faceCount,
                                          int const *faceVertexIndices,
                                          int indexCount,
                                          int const *activeFaces,
                                          int activeCount);
int USDGENPOMADE_API Pomade_HasScalp(PomadeModelContext const *ctx);
/* Per parent face: 1 on the growth surface, 0 outside a bound face subset
 * (all 1 for a whole-mesh bind). Same sizing rule as Pomade_ReadFaceRegions;
 * *outCount is 0 with no scalp bound. */
int USDGENPOMADE_API Pomade_ReadScalpFaceActive(PomadeModelContext *ctx,
                                              int *out, int maxOut,
                                              int *outCount);

/* K1 queries. *outHit is 1 on a hit (face/uv/position/normal written) and
 * 0 on a miss (outputs untouched). */
int USDGENPOMADE_API Pomade_Raycast(PomadeModelContext *ctx,
                                  float const origin[3], float const dir[3],
                                  int *outHit, int *outFace, float outUV[2],
                                  float outP[3], float outN[3]);
int USDGENPOMADE_API Pomade_ClosestPoint(PomadeModelContext *ctx,
                                       float const p[3], int *outHit,
                                       int *outFace, float outUV[2],
                                       float outP[3], float outN[3]);

/* Graph-mode edits. *outId receives the new node/edge id. */
int USDGENPOMADE_API Pomade_GraphAddNode(PomadeModelContext *ctx, int faceId,
                                       float u, float v, int *outId);
/* Atomically closes one graph region. `nodeIds[i] >= 0` reuses that exact
 * live stable node id and ignores faceIds/uvs at i; `nodeIds[i] == -1`
 * creates a node at faceIds[i], uvs[2*i..2*i+1]. Existing chain edges are
 * reused. `outRegionId` may be NULL. */
int USDGENPOMADE_API Pomade_GraphCreateRegion(
    PomadeModelContext *ctx, int const *nodeIds, int const *faceIds,
    float const *uvs, int count, int *outRegionId);
int USDGENPOMADE_API Pomade_GraphMoveNode(PomadeModelContext *ctx, int nodeId,
                                        int faceId, float u, float v);
/* Atomically moves existing graph nodes to K1 face/UV locations. Used by a
 * whole-edge drag: all targets are checked before its endpoints move, K2
 * retraces once, and any topology-changing target is rejected unchanged. */
int USDGENPOMADE_API Pomade_GraphMoveNodes(
    PomadeModelContext *ctx, int const *nodeIds, int const *faceIds,
    float const *uvs, int count);
int USDGENPOMADE_API Pomade_GraphConnect(PomadeModelContext *ctx, int a, int b,
                                       int *outEdge);
int USDGENPOMADE_API Pomade_GraphSplitEdge(PomadeModelContext *ctx, int edgeId,
                                         int faceId, float u, float v,
                                         int *outNode);
int USDGENPOMADE_API Pomade_GraphWeld(PomadeModelContext *ctx, int keep,
                                    int drop);
int USDGENPOMADE_API Pomade_GraphWeldAll(PomadeModelContext *ctx, float radius,
                                       int *outWelds);
int USDGENPOMADE_API Pomade_GraphUnweld(PomadeModelContext *ctx, int nodeId,
                                      int *outIds, int maxOut, int *outCount);
int USDGENPOMADE_API Pomade_GraphDeleteEdge(PomadeModelContext *ctx, int edgeId);
int USDGENPOMADE_API Pomade_GraphDeleteNode(PomadeModelContext *ctx, int nodeId);
/* Snap queries. Returns the node/edge id, or -1 when nothing is in range
 * (a null context also returns -1). */
int USDGENPOMADE_API Pomade_GraphSnapNode(PomadeModelContext const *ctx,
                                        float const p[3], float radius);
int USDGENPOMADE_API Pomade_GraphSnapEdge(PomadeModelContext const *ctx,
                                        float const p[3], float radius);
/* Reads one exact stable graph node; returns POMADE_ERROR for dead/unknown
 * ids. All output pointers are required. */
int USDGENPOMADE_API Pomade_GraphGetNode(PomadeModelContext const *ctx,
                                       int nodeId, int *outFaceId,
                                       float outUV[2], float outP[3]);
/* Reads the stable endpoint node ids of one live stable edge id. */
int USDGENPOMADE_API Pomade_GraphGetEdge(PomadeModelContext const *ctx,
                                       int edgeId, int outNodeIds[2]);
/* Reads the viewport position of one stable graph node.  This is its
 * canonical scalp point plus the display-only normal lift used by the
 * scene index; GraphGetNode remains the authoring-coordinate getter. */
int USDGENPOMADE_API Pomade_GraphGetNodeDisplayPosition(
    PomadeModelContext const *ctx, int nodeId, float outP[3]);
int USDGENPOMADE_API Pomade_GraphLinkRegions(PomadeModelContext *ctx, int r0,
                                           int r1);
int USDGENPOMADE_API Pomade_GraphUnlinkRegions(PomadeModelContext *ctx, int r0,
                                             int r1);
/* A Draw stroke: `faceIds`/`uvs` (2 floats per sample) are one K1 hit per
 * mouse sample. Writes the node chain (up to maxOut ids) and the weld/loop
 * flags. */
int USDGENPOMADE_API Pomade_GraphStroke(
    PomadeModelContext *ctx, int const *faceIds, float const *uvs, int samples,
    float snapRadius, float simplifyEps, int *outNodes, int maxOut,
    int *outCount, int *outClosed, int *outWeldedStart, int *outWeldedEnd);
/* Mirror-X over the whole graph. Writes (oldId, newId) pairs, 2 ints each. */
int USDGENPOMADE_API Pomade_GraphMirrorX(PomadeModelContext *ctx, int *outPairs,
                                       int maxPairs, int *outCount);

int USDGENPOMADE_API Pomade_SetSnapRadius(PomadeModelContext *ctx, float radius);
float USDGENPOMADE_API Pomade_GetSnapRadius(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_SetMirrorX(PomadeModelContext *ctx, int on);
int USDGENPOMADE_API Pomade_GetMirrorX(PomadeModelContext const *ctx);

int USDGENPOMADE_API Pomade_Rasterise(PomadeModelContext *ctx);
/* Exact graph-region query at a K1 surface hit. Returns the source graph
 * region id (not the linked interpolation id), or -1 when the coordinate is
 * uncovered or invalid. This intentionally does not use the coarse per-face
 * K3 map: multiple closed regions may occupy one scalp face. */
int USDGENPOMADE_API Pomade_RegionAtSurface(PomadeModelContext const *ctx,
                                          int faceId, float u, float v);
unsigned long long USDGENPOMADE_API Pomade_GetMapVersion(
    PomadeModelContext const *ctx);

/* Census + HUD data: alive nodes/edges/regions, uncovered faces,
 * root-intersected faces. Any out-param may be NULL. */
int USDGENPOMADE_API Pomade_GetGraphCounts(PomadeModelContext const *ctx,
                                         int *outNodes, int *outEdges,
                                         int *outRegions);
int USDGENPOMADE_API Pomade_GetRegionStats(PomadeModelContext const *ctx,
                                         int *outRegions, int *outUncovered,
                                         int *outIntersected);
/* The live region map (per-face interp ids, -1 uncovered). `out` must hold
 * the scalp face count ints; *outCount receives it. */
int USDGENPOMADE_API Pomade_ReadFaceRegions(PomadeModelContext *ctx, int *out,
                                          int maxOut, int *outCount);
/* Per-face lowest claiming region id (-1 uncovered): link picking reads
 * this, not the interp ids. Same sizing rule as Pomade_ReadFaceRegions. */
int USDGENPOMADE_API Pomade_ReadFaceRegionIds(PomadeModelContext *ctx, int *out,
                                            int maxOut, int *outCount);
/* Graph census reads for the scene-index publisher and the HUD. */
int USDGENPOMADE_API Pomade_ReadGraphNodes(PomadeModelContext *ctx,
                                         int *outFaceIds, float *outUV,
                                         float *outP, int maxNodes,
                                         int *outCount);
int USDGENPOMADE_API Pomade_ReadGraphEdges(PomadeModelContext *ctx,
                                         int *outPairs, int maxEdges,
                                         int *outCount);
int USDGENPOMADE_API Pomade_ReadRegionColors(PomadeModelContext *ctx,
                                           float *outRGB, int maxRegions,
                                           int *outCount);
/* Region loops as stable node ids (polygon-style): `outCounts` takes the
 * per-region node count (maxRegions entries), `outIndices` the flattened
 * loops (maxIndices entries). Either array may be NULL to query its size. */
int USDGENPOMADE_API Pomade_ReadRegionLoops(PomadeModelContext *ctx,
                                          int *outCounts, int maxRegions,
                                          int *outIndices, int maxIndices,
                                          int *outRegionCount,
                                          int *outIndexCount);

/* P3 Tube mode (plan/17 section 5.2, K4/K5). `ringVerts=0` matches the
 * canonical region-loop CV count. Explicit 3..32 retains every authored
 * corner and adds only deterministic samples along region edges. Center CVs,
 * sections, soft
 * selection and display density. Guides are NOT refilled here: the move
 * is an edit + Pomade_RefillGuides(preview), the release ends with
 * Pomade_RefillGuides(1.0). */
int USDGENPOMADE_API Pomade_BuildTubeFromRegion(PomadeModelContext *ctx,
                                              int regionId, int centerCount,
                                              int ringVerts, float length);
int USDGENPOMADE_API Pomade_MoveCenterCV(PomadeModelContext *ctx, int cv,
                                       float dx, float dy, float dz);
int USDGENPOMADE_API Pomade_InsertCenterCV(PomadeModelContext *ctx, int atIndex);
int USDGENPOMADE_API Pomade_DeleteCenterCV(PomadeModelContext *ctx, int index);
int USDGENPOMADE_API Pomade_SetTubeLength(PomadeModelContext *ctx, float length);
int USDGENPOMADE_API Pomade_MatchSurface(PomadeModelContext *ctx);
/* Center census. Null context returns -1. */
int USDGENPOMADE_API Pomade_GetCenterCVCount(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_GetCenterCV(PomadeModelContext *ctx, int cv,
                                      float outXYZ[3]);
/* Section census + reads. Null context returns -1. Pomade_GetSection
 * writes the ring's t, its (u, v) pairs (2 * ringVerts floats), scale
 * and twist; *outCount receives the ring CV count. */
int USDGENPOMADE_API Pomade_GetSectionCount(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_GetSection(PomadeModelContext *ctx, int ring,
                                     float *outT, float *outUV, int uvLen,
                                     int *outCount, float *outScale,
                                     float *outTwist);
int USDGENPOMADE_API Pomade_MoveSectionRing(PomadeModelContext *ctx, int ring,
                                          float du, float dv);
int USDGENPOMADE_API Pomade_ScaleSectionRing(PomadeModelContext *ctx, int ring,
                                           float scale);
int USDGENPOMADE_API Pomade_TwistSectionRing(PomadeModelContext *ctx, int ring,
                                           float radians);
int USDGENPOMADE_API Pomade_MoveSectionCV(PomadeModelContext *ctx, int ring,
                                        int slot, float du, float dv);
/* *outRing receives the new ring count (inserted rings sort by t). */
int USDGENPOMADE_API Pomade_AddSectionRing(PomadeModelContext *ctx, float t,
                                         int *outRing);
int USDGENPOMADE_API Pomade_RemoveSectionRing(PomadeModelContext *ctx, int ring);
int USDGENPOMADE_API Pomade_CopySectionRing(PomadeModelContext *ctx, int src,
                                          int dst);
int USDGENPOMADE_API Pomade_SetSoftSelection(PomadeModelContext *ctx,
                                           float center, float radius);
int USDGENPOMADE_API Pomade_GetSoftSelection(PomadeModelContext *ctx,
                                           float *outCenter,
                                           float *outRadius);
int USDGENPOMADE_API Pomade_RelaxCenter(PomadeModelContext *ctx, float strength,
                                      int iterations);
int USDGENPOMADE_API Pomade_SnapRootToScalp(PomadeModelContext *ctx);
int USDGENPOMADE_API Pomade_SetDisplaySegments(PomadeModelContext *ctx,
                                             int segments);
int USDGENPOMADE_API Pomade_GetDisplaySegments(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_SetTubeRegionId(PomadeModelContext *ctx, int regionId);
int USDGENPOMADE_API Pomade_GetTubeRegionId(PomadeModelContext const *ctx);
/* Faces at the tube's interpolation id. `out` must hold maxOut ints;
 * *outCount receives the face count. */
int USDGENPOMADE_API Pomade_ReadTubeRegionFaces(PomadeModelContext *ctx, int *out,
                                              int maxOut, int *outCount);

/* P3 Fill mode (plan/17 section 5.3, K8-K10). Pomade_SetFillParams takes
 * the density, CV count, seed, edge bias and the flattened length-profile
 * (position, value) pairs (pairFloats must be even). */
int USDGENPOMADE_API Pomade_SetFillParams(PomadeModelContext *ctx, float density,
                                        int cvCount, int seed, float edgeBias,
                                        float const *profilePairs,
                                        int pairFloats);
int USDGENPOMADE_API Pomade_GetFillParams(PomadeModelContext *ctx,
                                        float *outDensity, int *outCvCount,
                                        int *outSeed, float *outEdgeBias,
                                        float *outProfile, int maxFloats,
                                        int *outFloats);
/* Commit-only output. `densityMultiplier` scales the authored full Fill
 * density (not the interactive preview fraction); width is in scene units.
 * `enabled` is exactly 0 or 1. Get requires all three output pointers. */
int USDGENPOMADE_API Pomade_SetOutputSettings(PomadeModelContext *ctx,
                                            int enabled,
                                            float densityMultiplier,
                                            float width);
int USDGENPOMADE_API Pomade_GetOutputSettings(PomadeModelContext const *ctx,
                                            int *outEnabled,
                                            float *outDensityMultiplier,
                                            float *outWidth);
int USDGENPOMADE_API Pomade_SetPreviewFraction(PomadeModelContext *ctx,
                                             float fraction);
float USDGENPOMADE_API Pomade_GetPreviewFraction(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_SetFreezeRoots(PomadeModelContext *ctx, int on);
int USDGENPOMADE_API Pomade_GetFreezeRoots(PomadeModelContext const *ctx);
/* Refill through K8/K9/K10; negative fraction selects the stored preview
 * fraction, 1.0 refills at full density. */
int USDGENPOMADE_API Pomade_RefillGuides(PomadeModelContext *ctx, float fraction);
/* Explicit Fill after Clear. Auto-refresh RefillGuides stays suppressed. */
int USDGENPOMADE_API Pomade_GenerateGuides(PomadeModelContext *ctx,
                                         float fraction);
int USDGENPOMADE_API Pomade_ClearGeneratedCurves(PomadeModelContext *ctx);
int USDGENPOMADE_API Pomade_SetGeneratedCurvesVisible(PomadeModelContext *ctx,
                                                     int visible);
int USDGENPOMADE_API Pomade_GetGeneratedCurvesVisible(
    PomadeModelContext const *ctx, int *outVisible);
/* Guide census. Any out-param may be NULL. */
int USDGENPOMADE_API Pomade_GetGuideCounts(PomadeModelContext const *ctx,
                                         int *outGuides, int *outCv);
/* Tubes the last refill skipped (they produced no guides), in fill order
 * (two-call (NULL, 0) probe). A refill still returns POMADE_OK while any
 * tube filled, so this is how a partial drop reaches the dock. */
int USDGENPOMADE_API Pomade_ReadRefillDrops(PomadeModelContext const *ctx,
                                          int *out, int outCap,
                                          int *outCount);
/* The reason the index-th dropped tube was skipped ("" when out of
 * range). The string stays valid until the next call on this thread. */
USDGENPOMADE_API const char *Pomade_GetRefillDropReason(
    PomadeModelContext const *ctx, int index);
/* The current guide set (guide-major points + per-guide counts).
 * `outXYZ` must hold 3 * guides * cv floats, `outCounts` guides ints;
 * *outGuideCount receives the guide count. */
int USDGENPOMADE_API Pomade_ReadGuidePreview(PomadeModelContext *ctx,
                                           float *outXYZ, int xyzLen,
                                           int *outCounts, int countsLen,
                                           int *outGuideCount);
/* Root census: face ids, world positions (3 per root) and normalised
 * root-plane coords (2 per root). Any array may be NULL to query only
 * *outCount. */
int USDGENPOMADE_API Pomade_ReadGuideRoots(PomadeModelContext *ctx,
                                         int *outFaceIds, float *outXYZ,
                                         float *outRU, int maxRoots,
                                         int *outCount);

/* P3 pick (plan/17 section 4.6, K11). Screen-space pick with a kind mask
 * (a PomadePickKind OR from pomadeTube.h; PomadePick_All picks everything).
 * viewProj is row-major USD (the gpu/picking.cu convention), (x, y) is
 * top-left-origin pixels. *outHit is 1 on a hit (kind/index/subIndex +
 * distance/depth written) and 0 on a miss (outputs untouched). */
int USDGENPOMADE_API Pomade_Pick(PomadeModelContext *ctx,
                               float const viewProj[16], int w, int h, float x,
                               float y, float radiusPx, unsigned int kindMask,
                               int *outHit, unsigned int *outKind,
                               int *outIndex, int *outSubIndex,
                               float *outDistPx, float *outDepth);

/* P1 commit pipeline (plan/17 section 3). The committer owns a worker thread
 * that snapshots the model and builds anonymous SdfLayers; the UI thread
 * swaps them into the tool-owned live sublayer at idle. Layers cross as
 * identifiers (anonymous layers have them too), so this stays ctypes-safe:
 * no Sdf/Usd handle ever crosses. Returns PomadeCommitter_Swap* codes. */
typedef struct PomadeCommitterContext PomadeCommitterContext;

#define PomadeCommitter_Swapped 0
#define PomadeCommitter_PartialProgress 1
#define PomadeCommitter_NothingPending 2
#define PomadeCommitter_SkippedGesture 3
#define PomadeCommitter_SkippedStale 4
#define PomadeCommitter_Detached 5
/* Pomade_CommitterSwap failed (null argument, unknown live layer, a throw);
 * the reason is in Pomade_GetLastError. Negative so it can never be read as
 * PomadeCommitter_PartialProgress, which POMADE_ERROR (1) used to alias. */
#define PomadeCommitter_Error (-1)

/* Create a committer for `modelCtx` (which must outlive it) writing the
 * groom at `groomPath` and filling the description at `descPath` (NULL or
 * empty disables the GuideInterpolate fill-in). */
int USDGENPOMADE_API Pomade_CommitterCreate(
    PomadeModelContext *modelCtx, const char *groomPath, const char *descPath,
    PomadeCommitterContext **outCc);
int USDGENPOMADE_API Pomade_CommitterDestroy(PomadeCommitterContext *cc);

/* Enqueue a commit of the model's latest version. The GuideInterpolate
 * fill-in plan is computed by the caller over pxr (PomadePlanGuideInterpolate-
 * Fill documents the rule) and passed as flags; Enqueue() with all zeros
 * plans no fill-in for this version. `opPath` names the op the flags target
 * (existing or new); NULL selects the default. */
int USDGENPOMADE_API Pomade_CommitterEnqueue(
    PomadeCommitterContext *cc, int createOp, int setGuides, int setRegion,
    const char *opPath);

/* Swap the latest built layer into the live layer named `liveIdentifier`
 * (found via SdfLayer::Find). `gestureActive` nonzero holds the swap back.
 * Returns a PomadeCommitter_Swap* code, or PomadeCommitter_Error (with
 * GetLastError) when the live layer is unknown or the swap failed. */
int USDGENPOMADE_API Pomade_CommitterSwap(PomadeCommitterContext *cc,
                                        const char *liveIdentifier,
                                        int gestureActive);

/* Take the committer's last failure: a worker build that failed or threw,
 * or an enqueue the artist-owned-output guard refused. The text is cleared
 * by the call, so a tool pumping once per idle slot reports each failure
 * once. Copies at most cap-1 bytes plus a NUL into `out` and returns the
 * full length (0 when there is nothing to take; a return >= cap means the
 * copy was truncated). With out NULL or cap <= 0 the text is still taken
 * and only its length returned. -1 (with GetLastError) on a null
 * committer. */
int USDGENPOMADE_API Pomade_CommitterTakeDiagnostic(PomadeCommitterContext *cc,
                                                  char *out, int cap);

/* The model version whose build failed or whose enqueue was refused, or 0
 * when the latest enqueue has not failed. The worker never retries it on
 * its own; a pending version at or below it is not in flight, and only a
 * fresh Pomade_CommitterEnqueue tries again. */
unsigned long long USDGENPOMADE_API Pomade_CommitterFailedVersion(
    PomadeCommitterContext const *cc);

unsigned long long USDGENPOMADE_API Pomade_CommitterCommittedVersion(
    PomadeCommitterContext const *cc);
unsigned long long USDGENPOMADE_API Pomade_CommitterPendingVersion(
    PomadeCommitterContext const *cc);
double USDGENPOMADE_API Pomade_CommitterLastSwapMs(
    PomadeCommitterContext const *cc);
int USDGENPOMADE_API Pomade_CommitterPartialMode(
    PomadeCommitterContext const *cc);
int USDGENPOMADE_API Pomade_CommitterSetSwapBudgetMs(PomadeCommitterContext *cc,
                                                   double ms);

/* Abandon the usdGen cook the last swap started (plan/17 section 3.2 rule
 * 2, plan/18 section 7 G7). Called at gesture press: the cook in flight is
 * describing a groom the model has already left, and "show amplified hair"
 * has just hidden its tiles anyway. Returns the number of description
 * sessions whose cancellation token moved, or -1 on error. Safe with no
 * description linked (returns 0). */
int USDGENPOMADE_API Pomade_CommitterCancelCooks(PomadeCommitterContext *cc);

/* Retarget the scalp link (takes effect on the next build). NULL or empty
 * clears it. */
int USDGENPOMADE_API Pomade_CommitterSetScalpPath(PomadeCommitterContext *cc,
                                                const char *scalpPath);
/* Pomade_CommitterSetScalpPath for a face GeomSubset scalp (plan/02 section
 * 2.20): usdGen:pomade:scalp and Output's usdGen:surface name `scalpPath`,
 * while UsdGenRestAPI and primvars:usdGen:pomadeRegion land on `meshPath`,
 * the subset's parent Mesh. NULL/empty `meshPath` means `scalpPath` is the
 * Mesh itself; any other meshPath must be scalpPath's parent. */
int USDGENPOMADE_API Pomade_CommitterSetScalpTarget(PomadeCommitterContext *cc,
                                                  const char *scalpPath,
                                                  const char *meshPath);

/* Stage closed or reloaded under the tool (plan/17 section 3.4): Detach
 * idles the committer (the model survives); Swap reports Detached until
 * Reattach. Enqueues while detached are dropped; the tool enqueues fresh
 * after Reattach and the next swap re-hydrates (or re-creates) the groom
 * prim in the new live layer. */
int USDGENPOMADE_API Pomade_CommitterDetach(PomadeCommitterContext *cc);
int USDGENPOMADE_API Pomade_CommitterReattach(PomadeCommitterContext *cc);

/* P2 bake pipeline (plan/17 section 3.1a). The bake worker owns its thread
 * and its lowest-priority CUDA stream; the UI thread enqueues map versions
 * and swaps finished files with one attribute author each. */
typedef struct PomadeBakeContext PomadeBakeContext;

/* Create a bake worker for `modelCtx` (which must outlive it). Versioned
 * files land in `outDir` as `<baseName>.v<m>.ptx` (`baseName` NULL/empty
 * selects "regionMap"). */
int USDGENPOMADE_API Pomade_BakeCreate(PomadeModelContext *modelCtx,
                                     const char *outDir, const char *baseName,
                                     PomadeBakeContext **outBake);
int USDGENPOMADE_API Pomade_BakeDestroy(PomadeBakeContext *bake);

/* Bake options: `resOverride` forces one per-face texel resolution (log2,
 * -1 selects auto), `levelCount` sets the baked channel count (P2: 1). */
int USDGENPOMADE_API Pomade_BakeSetOptions(PomadeBakeContext *bake,
                                         int resOverride, int levelCount);

/* Enqueue a bake of the model's current graph at its map version. */
int USDGENPOMADE_API Pomade_BakeEnqueue(PomadeBakeContext *bake);

/* Take the latest finished bake (UI thread, at idle). Returns 1 with
 * *outVersion + the NUL-terminated path when a bake completed since the
 * last call, else 0. */
int USDGENPOMADE_API Pomade_BakeTakeCompleted(PomadeBakeContext *bake,
                                            unsigned long long *outVersion,
                                            char *outPath, int pathLen);

/* Swap a finished bake into the live layer (UI thread, at idle): authors
 * exactly one attribute (usdGen:map:file on `regionMapPath`), records the
 * file on the model (so section 3.1 builds re-author it), and tells the
 * worker to sweep stale versions. */
int USDGENPOMADE_API Pomade_BakeSwap(PomadeBakeContext *bake,
                                   unsigned long long mapVersion,
                                   const char *liveIdentifier,
                                   const char *regionMapPath,
                                   const char *file);

unsigned long long USDGENPOMADE_API Pomade_BakePendingVersion(
    PomadeBakeContext const *bake);
unsigned long long USDGENPOMADE_API Pomade_BakeCompletedVersion(
    PomadeBakeContext const *bake);

/* -- P4: hierarchy + sculpt (plan/17 §2.4, §5.4/§5.5) -------------------- */

/* Split tubeId into count children (K14); outIds takes the child ids. */
int USDGENPOMADE_API Pomade_SubdivideTube(PomadeModelContext *ctx, int tubeId,
                                        int count, const char *splitMode,
                                        int seed, int *outIds, int outCap,
                                        int *outCount);
/* Merge tubeId's children back into it (K7 last pass + removal). */
int USDGENPOMADE_API Pomade_MergeChildren(PomadeModelContext *ctx, int tubeId);
/* Fold sibling tubeIds into one child; the kept id is tubeIds[0]. */
int USDGENPOMADE_API Pomade_MergeSelected(PomadeModelContext *ctx,
                                        int const *tubeIds, int idCount,
                                        int *outKept);
/* Live tube count / level of tubeId (-1 when missing). Direct returns. */
int USDGENPOMADE_API Pomade_GetTubeCount(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_GetTubeLevel(PomadeModelContext const *ctx,
                                       int tubeId);
/* Child ids of tubeId in ascending order. */
int USDGENPOMADE_API Pomade_GetTubeChildren(PomadeModelContext const *ctx,
                                          int tubeId, int *outIds, int outCap,
                                          int *outCount);
/* Center-CV count of tubeId (-1 when missing). Direct return. */
int USDGENPOMADE_API Pomade_GetTubeCenterCount(PomadeModelContext const *ctx,
                                             int tubeId);
/* Center CV into outXYZ[3]. */
int USDGENPOMADE_API Pomade_GetTubeCenterCV(PomadeModelContext const *ctx,
                                          int tubeId, int cv, float *outXYZ);
/* Visible/editable center-CV handle into outXYZ[3]. This is the actual
 * section polygon area centroid; Pomade_GetTubeCenterCV remains raw authored
 * center-cage data. */
int USDGENPOMADE_API Pomade_GetTubeCenterHandle(PomadeModelContext const *ctx,
                                              int tubeId, int cv,
                                              float *outXYZ);
/* Move one center CV with K6-down / K7-up propagation. */
int USDGENPOMADE_API Pomade_MoveTubeCenterCV(PomadeModelContext *ctx, int tubeId,
                                           int cv, float dx, float dy,
                                           float dz);
/* Translate every center CV in one tube atomically, then run K6/K7 once. */
int USDGENPOMADE_API Pomade_TranslateTube(PomadeModelContext *ctx, int tubeId,
                                        float dx, float dy, float dz);
/* Flattened center deltas (du, dv, dw per CV); two-call (NULL, 0) probe. */
int USDGENPOMADE_API Pomade_ReadTubeDeltas(PomadeModelContext const *ctx,
                                         int tubeId, float *out, int outCap,
                                         int *outCount);
/* Propagation gates; tubeId -1 = global default. */
int USDGENPOMADE_API Pomade_SetLockParents(PomadeModelContext *ctx, int tubeId,
                                         int on);
int USDGENPOMADE_API Pomade_SetLockChildren(PomadeModelContext *ctx, int tubeId,
                                          int on);
/* K7 on-the-fly parent over tubeIds; returns its (negative) id. */
int USDGENPOMADE_API Pomade_GroupTubes(PomadeModelContext *ctx,
                                     int const *tubeIds, int idCount,
                                     int makeTransient, int *outParent);
int USDGENPOMADE_API Pomade_MakePersistent(PomadeModelContext *ctx, int tubeId);
/* Sculpt stroke over center CVs (brush shapes arrive precomputed). */
int USDGENPOMADE_API Pomade_SculptStroke(PomadeModelContext *ctx, int tubeId,
                                       const char *brush, int const *cvIds,
                                       float const *deltas, int cvCount,
                                       int preserveLength, int mirrorX);
/* K13 per-CV kink scores of tube 0 (two-call (NULL, 0) probe). Empty when
 * no tube is built. */
int USDGENPOMADE_API Pomade_ReadSmoothnessScores(PomadeModelContext const *ctx,
                                               float *out, int outCap,
                                               int *outCount);
/* K12 root-overlap check over tube 0 plus the store (post-gesture). */
int USDGENPOMADE_API Pomade_CheckRootIntersections(PomadeModelContext *ctx);
/* Tube ids flagged by the last check, ascending (two-call probe). */
int USDGENPOMADE_API Pomade_ReadIntersectedTubes(PomadeModelContext const *ctx,
                                               int *out, int outCap,
                                               int *outCount);
/* -- P5 bridges (plan/17 section 5.7) -------------------------------------- */
/* All tube ids in the model, ascending (two-call (NULL, 0) probe). */
int USDGENPOMADE_API Pomade_ReadTubeIds(PomadeModelContext const *ctx, int *out,
                                      int outCap, int *outCount);
/* -- V6: one L1 tube per region (plan/18 section 7 G14) -------------------- */
/* The L1 root ids, ascending (two-call (NULL, 0) probe). Tube 0 is first
 * when it is built; every further region's tube follows. */
int USDGENPOMADE_API Pomade_ReadL1TubeIds(PomadeModelContext const *ctx, int *out,
                                        int outCap, int *outCount);
/* The tube rooted in regionId, or -1 when the region has no stub yet. */
int USDGENPOMADE_API Pomade_TubeForRegion(PomadeModelContext const *ctx,
                                        int regionId);
/* The region a tube (or its L1 ancestor) is rooted in, or -1. */
int USDGENPOMADE_API Pomade_RegionForTube(PomadeModelContext const *ctx,
                                        int tubeId);
/* Re-attach the L1 tubes to the regions a graph edit left behind. Rasterise
 * already calls this; the entry point exists for a tool that edits the graph
 * and rasterises separately. outRebuilt/outRemoved may be NULL. Returns
 * POMADE_ERROR with the section 5.1 "merge the children first" diagnostic when
 * a tube that would move or go away carries child deltas. */
int USDGENPOMADE_API Pomade_SyncRegionTubes(PomadeModelContext *ctx,
                                          int *outRebuilt, int *outRemoved);
/* Import explicit centers + sections as a locked child of parentId. secT
 * takes nSec ascending values in [0, 1]; secU/secV take nSec * ringVerts
 * row-major (scale/twist default to 1/0). *outTubeId receives the child
 * id. Returns POMADE_OK/POMADE_ERROR (use Pomade_GetLastError). */
int USDGENPOMADE_API Pomade_ImportLockedTube(
    PomadeModelContext *ctx, int parentId, float const *cx, float const *cy,
    float const *cz, int nCv, float const *secT, float const *secU,
    float const *secV, int nSec, int ringVerts, int *outTubeId);
/* Swept-mesh import: ring-major world points (ringCount * ringVerts, 3
 * floats each, no caps) as a locked child of parentId. *outTubeId
 * receives the child id. */
int USDGENPOMADE_API Pomade_ImportSweptMesh(PomadeModelContext *ctx, int parentId,
                                          float const *points, int ringCount,
                                          int ringVerts, int *outTubeId);
/* Section census + reads for any tube (tube 0 included). Count is -1
 * when the tube is unknown. Pomade_GetTubeSection mirrors Pomade_GetSection
 * (ring t, (u, v) pairs, scale, twist). */
int USDGENPOMADE_API Pomade_GetTubeSectionCount(PomadeModelContext const *ctx,
                                              int tubeId);
int USDGENPOMADE_API Pomade_GetTubeSection(PomadeModelContext *ctx, int tubeId,
                                         int ring, float *outT, float *outUV,
                                         int uvLen, int *outCount,
                                         float *outScale, float *outTwist);
/* 1 when the tube is a bridge import, else 0 (unknown tubes read 0). */
int USDGENPOMADE_API Pomade_IsTubeImported(PomadeModelContext const *ctx,
                                         int tubeId);

/* -- V0 viewport publication (plan/18 section 2.1, 2.2) ---------------------
 *
 * Pomade_Create registers its model with the process-global PomadeRegistry;
 * the Pomade scene indices attach to the same registry. Pomade_Activate names
 * the one model the viewport draws, and Pomade_Publish is the ONLY way the
 * viewport learns about a change: the controller calls it after every
 * mutating entry point in a gesture, then asks the view to redraw. There is
 * no polling and no implicit publish, so a model the tool never publishes is
 * never on screen.
 *
 * Main thread only: publishing walks Hydra observers. */

/* The registry id of this model (> 0), or 0 when it is not registered. */
int USDGENPOMADE_API Pomade_GetModelId(PomadeModelContext const *ctx);
/* Make this model the one the viewport draws, and publish it. The static
 * test tube is removed on the first activation and never comes back. */
int USDGENPOMADE_API Pomade_Activate(PomadeModelContext *ctx);
/* Clear the active model (whatever it is) and publish the empty state. */
int USDGENPOMADE_API Pomade_Deactivate(PomadeModelContext *ctx);
/* Publish the active model to every attached scene index. `dirtyMask` is a
 * PomadeDirty_* OR the caller wants republished on top of whatever the model
 * reports pending; 0 publishes exactly the model's own pending bits.
 * Returns the number of indices that published something, or -1 on error
 * (unlike the status-returning entries above — the count IS the result). */
int USDGENPOMADE_API Pomade_Publish(PomadeModelContext *ctx,
                                  unsigned int dirtyMask);

/* Per-level display (1-based). Visibility, x-ray and focus are viewport
 * state: they re-publish, never re-tessellate, and are not authored to USD.
 * `visible`/`xray` are 0/1. The focused level draws thick center curves and
 * large CV dots; level 0 focuses nothing. */
/* V6 (plan/18 section 2.4a): which tubes draw their cross-section rings and
 * ring CV dots. 0 = none, 1 = selected tubes only (the default), 2 = every
 * tube (Tube mode's Ring and Section sub-modes). */
int USDGENPOMADE_API Pomade_SetRingDisplay(PomadeModelContext *ctx, int mode);
int USDGENPOMADE_API Pomade_GetRingDisplay(PomadeModelContext const *ctx);
/* V6 (plan/17 section 3.2, plan/18 section 7 G7): "show amplified hair".
 * On, the usdGen cook's amplified tiles draw and the Pomade guide preview
 * hides; off, the preview draws and the tiles are hidden. A live gesture
 * hides the tiles either way and the index restores them on release. */
int USDGENPOMADE_API Pomade_SetAmplifiedHair(PomadeModelContext *ctx, int show);
int USDGENPOMADE_API Pomade_GetAmplifiedHair(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_SetLevelDisplay(PomadeModelContext *ctx, int level,
                                          int visible, int xray);
int USDGENPOMADE_API Pomade_GetLevelDisplay(PomadeModelContext const *ctx,
                                          int level, int *outVisible,
                                          int *outXray);
int USDGENPOMADE_API Pomade_SetFocusLevel(PomadeModelContext *ctx, int level);
int USDGENPOMADE_API Pomade_GetFocusLevel(PomadeModelContext const *ctx);
/* Per-branch hierarchy display cut. Disabled is the legacy behavior: every
 * live tube is visible and focus level controls the edit frontier. Enabled
 * starts with L1 roots collapsed; expanding a non-leaf hides that tube and
 * exposes its direct children without changing unrelated branches. */
int USDGENPOMADE_API Pomade_SetActiveCutEnabled(PomadeModelContext *ctx,
                                              int enabled);
int USDGENPOMADE_API Pomade_GetActiveCutEnabled(PomadeModelContext const *ctx);
int USDGENPOMADE_API Pomade_SetTubeExpanded(PomadeModelContext *ctx, int tubeId,
                                          int expanded);
int USDGENPOMADE_API Pomade_GetTubeExpanded(PomadeModelContext const *ctx,
                                          int tubeId);
int USDGENPOMADE_API Pomade_IsTubeVisible(PomadeModelContext const *ctx,
                                        int tubeId);
/* V8 (plan/18 section 2.4a): world units per screen pixel at the focus
 * point, so the overlay dots and curves can be a fixed number of pixels
 * across. Storm sizes points and curves in WORLD units, so without this
 * a CV dot is as big as the tube it sits on. The controller sets it when
 * it resolves a camera and whenever the frustum changes; 0 restores the
 * radius-relative sizing a model with no camera uses. */
int USDGENPOMADE_API Pomade_SetDisplayScale(PomadeModelContext *ctx,
                                          float worldPerPixel);
int USDGENPOMADE_API Pomade_GetDisplayScale(PomadeModelContext const *ctx,
                                          float *outWorldPerPixel);
/* V9 (plan/18 section 2.4a): apply THE display policy in one call.
 * `mode` and `subMode` are the ids pomadeModes spells ("graph", "tube" +
 * "center"/"ring"/"section", "fill", "hierarchy", "sculpt", "output");
 * `focusLevel` is the level being edited, 0 for none. The call sets every
 * level's visibility, x-ray strength and centers flag, the ring display and
 * the focus level, so the viewport look of a mode is decided in exactly one
 * place (usdGenPomade::PomadePolicyLevelDisplay). An unknown mode is an error
 * and changes nothing. The per-level centers-only state the fallback ladder
 * owns is carried through untouched. */
int USDGENPOMADE_API Pomade_SetDisplayPolicy(PomadeModelContext *ctx,
                                           char const *mode,
                                           char const *subMode,
                                           int focusLevel);
/* The x-ray strength (0 when the level is opaque) and centers flag the
 * policy left on `level`. Null outputs are skipped. */
int USDGENPOMADE_API Pomade_GetLevelDraw(PomadeModelContext const *ctx,
                                       int level, float *outXrayOpacity,
                                       int *outCenters);
/* V9: the stage path of the groom this model commits to. While it is set
 * and the model is active, the scene index hides the committed
 * `<groom>/Guides` prim — last commit's curves, which would otherwise draw
 * plain white over this gesture's tubes. Hydra visibility only: the
 * amplifier reads the stage, so no cook is starved. "" hides nothing. */
int USDGENPOMADE_API Pomade_SetGroomPath(PomadeModelContext *ctx,
                                       char const *path);
int USDGENPOMADE_API Pomade_GetGroomPath(PomadeModelContext const *ctx,
                                       char *out, int cap);

/* What the attached scene indices actually published for `level`: the face,
 * point and tube counts of /__usdGenPomade/tubes/L<level>. POMADE_ERROR when
 * no attached index publishes that level (which is also how a headless
 * process reads: it has no index). Null outputs are skipped. */
int USDGENPOMADE_API Pomade_GetPublishedLevelInfo(PomadeModelContext const *ctx,
                                                int level, int *outFaceCount,
                                                int *outPointCount,
                                                int *outTubeCount);

/* -- V1: selection, overlays, gestures (plan/18 §2.3-§2.5) ----------------
 *
 * Selection is model state, so it survives a commit and every mode reads
 * one set. Kinds are the PomadePickKind bits from pomadeTube.h — the same
 * enum the pick returns — and the ids are spelled per kind:
 *
 *   PomadePick_TubeVert     tube        id = tube id
 *   PomadePick_CenterCV     center CV   id = tube, subId = CV
 *   PomadePick_SectionCV    section CV  id = tube, subId = ring,
 *                                      subSubId = slot
 *   PomadePick_SectionRing  ring        id = tube, subId = ring
 *   PomadePick_GraphNode    node        id = node id
 *   PomadePick_GraphEdge    edge        id = edge id
 *   PomadePick_Region       region      id = region id
 *   PomadePick_Guide        guide       id = guide index
 *   PomadePick_Level        level       id = level (selects every tube at it)
 *
 * Unused sub-ids are passed as -1 (or the array as NULL). Selection is NOT
 * undoable: an artist who undoes a move wants the geometry back, not the
 * click before it.
 */

/* How a select call combines with what is already selected. */
#define POMADE_SELECT_SET     0  /* replace the kinds being set */
#define POMADE_SELECT_ADD     1  /* union */
#define POMADE_SELECT_TOGGLE  2  /* symmetric difference */

/* Drop every selected item whose kind is in `kindMask` (0 = all kinds).
 * The hover item clears with its own kind. */
int USDGENPOMADE_API Pomade_SelectClear(PomadeModelContext *ctx,
                                      unsigned int kindMask);
/* Select `n` items of one kind. Set replaces that kind, Add unions,
 * Toggle flips each. `subIds`/`subSubIds` may be NULL for kinds that do
 * not use them. */
int USDGENPOMADE_API Pomade_SelectSet(PomadeModelContext *ctx,
                                    unsigned int kind, const int *ids,
                                    const int *subIds, const int *subSubIds,
                                    int n);
int USDGENPOMADE_API Pomade_SelectAdd(PomadeModelContext *ctx,
                                    unsigned int kind, const int *ids,
                                    const int *subIds, const int *subSubIds,
                                    int n);
int USDGENPOMADE_API Pomade_SelectToggle(PomadeModelContext *ctx,
                                       unsigned int kind, const int *ids,
                                       const int *subIds,
                                       const int *subSubIds, int n);
/* Marquee: every candidate whose projection lands in the pixel rectangle,
 * over the same candidate sets and the same projection Pomade_Pick uses.
 * `mode` is one of POMADE_SELECT_*; an empty band in SET mode clears the
 * kinds it was asked for. */
int USDGENPOMADE_API Pomade_SelectRect(PomadeModelContext *ctx,
                                     const float *viewProj, int w, int h,
                                     float x0, float y0, float x1, float y1,
                                     unsigned int kindMask, int mode);
/* Lasso: the same, against a closed polygon of `n` (x, y) pixel pairs
 * (>= 3; the last point joins the first). */
int USDGENPOMADE_API Pomade_SelectPolygon(PomadeModelContext *ctx,
                                        const float *viewProj, int w, int h,
                                        const float *xy, int n,
                                        unsigned int kindMask, int mode);
/* The one hovered item. `kind` 0 clears it. */
int USDGENPOMADE_API Pomade_SetHover(PomadeModelContext *ctx, unsigned int kind,
                                   int id, int subId, int subSubId);
int USDGENPOMADE_API Pomade_GetHover(PomadeModelContext const *ctx,
                                   unsigned int *outKind, int *outId,
                                   int *outSubId, int *outSubSubId);
/* Read one kind back, ascending by (id, subId, subSubId). Any output array
 * may be NULL; *outCount always receives the true count, so a NULL-array
 * call sizes the buffers. POMADE_ERROR when cap is short for the arrays
 * that were passed. */
int USDGENPOMADE_API Pomade_ReadSelection(PomadeModelContext const *ctx,
                                        unsigned int kind, int *outIds,
                                        int *outSubIds, int *outSubSubIds,
                                        int cap, int *outCount);
/* How many items of the kinds in `kindMask` are selected (0 = all). */
int USDGENPOMADE_API Pomade_GetSelectionCount(PomadeModelContext const *ctx,
                                            unsigned int kindMask);
/* World bounds of the selection — where a gizmo goes, what Frame Selected
 * frames. POMADE_ERROR (outputs untouched) when nothing selected has a
 * position. */
int USDGENPOMADE_API Pomade_GetSelectionBounds(PomadeModelContext const *ctx,
                                             float *outMin, float *outMax);
/* Pick one pixel and report the SELECTION ITEM it names, translating the
 * candidate ordinals Pomade_Pick reports (a graph node's array slot) into
 * the stable ids the selection stores. This is what a press and a hover
 * sample call; the item then goes to Pomade_SelectSet or Pomade_SetHover.
 * *outHit is 1 on a hit and 0 on a miss (outputs untouched). */
int USDGENPOMADE_API Pomade_PickItem(PomadeModelContext *ctx,
                                   const float *viewProj, int w, int h,
                                   float x, float y, float radiusPx,
                                   unsigned int kindMask, int *outHit,
                                   unsigned int *outKind, int *outId,
                                   int *outSubId, int *outSubSubId);

/* The gizmo the viewport draws (plan/18 §2.4). `kind` is 0 none,
 * 1 translate, 2 ringTRS, 3 nodeTranslate, 4 rotate, 5 scale; `frame` is
 * the u, v, w axes, 3 floats each; `activeHandle` is -1 or a stable handle
 * id (0/1/2 axis, 3 legacy ring, 4 centre, 5/6/7 planar, 8 view, 9 free).
 * Setting the same record twice
 * dirties nothing.  Pomade_SetGizmo allows every handle; Pomade_SetGizmoEx
 * also takes `allowedMask`, bit (1 << handleId) set for each handle the
 * headless fallback may draw (0xFFFFFFFF = all), so it hides exactly what
 * the tool's own overlay hides (GZ-06). */
int USDGENPOMADE_API Pomade_SetGizmo(PomadeModelContext *ctx, int kind,
                                   const float *origin, const float *frame,
                                   float sizeWorld, int activeHandle);
int USDGENPOMADE_API Pomade_SetGizmoEx(PomadeModelContext *ctx, int kind,
                                     const float *origin, const float *frame,
                                     float sizeWorld, int activeHandle,
                                     unsigned int allowedMask);
int USDGENPOMADE_API Pomade_GetGizmo(PomadeModelContext const *ctx,
                                   int *outKind, float *outOrigin,
                                   float *outFrame, float *outSizeWorld,
                                   int *outActiveHandle);
int USDGENPOMADE_API Pomade_GetGizmoAllowedMask(PomadeModelContext const *ctx,
                                              unsigned int *outMask);
/* The sculpt brush ring. A radius <= 0 clears it. */
int USDGENPOMADE_API Pomade_SetBrushRing(PomadeModelContext *ctx,
                                       const float *center,
                                       const float *normal,
                                       float radiusWorld);
int USDGENPOMADE_API Pomade_GetBrushRing(PomadeModelContext const *ctx,
                                       int *outActive, float *outCenter,
                                       float *outNormal, float *outRadius);

/* The gesture bracket (plan/18 §3.2). Begin takes ONE undo snapshot and
 * suppresses the rest, so a 200-sample drag is one undo step instead of
 * 200; End seals it and bumps the version once; Cancel restores the
 * press-time base bit-exactly. A nested Begin is an error. Cancel writes
 * the dirty bits the caller must publish (the stage never saw the drag,
 * but the viewport did). */
int USDGENPOMADE_API Pomade_BeginGesture(PomadeModelContext *ctx,
                                       const char *label);
int USDGENPOMADE_API Pomade_EndGesture(PomadeModelContext *ctx);
int USDGENPOMADE_API Pomade_CancelGesture(PomadeModelContext *ctx,
                                        unsigned int *outDirty);
int USDGENPOMADE_API Pomade_GetGestureDepth(PomadeModelContext const *ctx);

/* Redo (plan/18 §2.5): the states undo left, cleared by any new mutation.
 * Like Pomade_Undo it writes the dirty bits to publish (NULL to ignore). */
int USDGENPOMADE_API Pomade_Redo(PomadeModelContext *ctx,
                               unsigned int *outDirty);
int USDGENPOMADE_API Pomade_GetRedoDepth(PomadeModelContext const *ctx);
/* The label of one history step, NUL-terminated, for the Edit strip.
 * `depth` 0 is the step Ctrl+Z would undo; NEGATIVE depths address the
 * redo stack (-1 is the step Ctrl+Y would redo). POMADE_ERROR when no such
 * step exists or the buffer is too small. */
int USDGENPOMADE_API Pomade_GetUndoLabel(PomadeModelContext const *ctx,
                                       int depth, char *out, int cap);

USDGENPOMADE_API const char *Pomade_GetLastError(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* USDGEN_POMADE_API_C_ABI_H */
