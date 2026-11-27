/* usdGenPomade/pomadeApiStage.h — the stage-contract half of the C ABI (V0b).
 *
 * Same rules as pomadeApi.h: extern "C", control and scalars only, 0 =
 * success and non-zero = error, nothing throws across the boundary. The
 * message for the calling thread comes from Pomade_StageGetLastError(); this
 * translation unit keeps its own thread-local because pomadeApi.cpp's is
 * file-private.
 *
 * What lives here (plan/18 §7):
 *   G3  Pomade_Hydrate — open a committed groom into a model.
 *   G2  the per-tube forms of every §5.2/§5.3 operation; the tube-0
 *       spellings in pomadeApi.h are these with tubeId 0.
 *   G4  Pomade_BakeEnqueueLevels — an enqueue that carries the hierarchy, so
 *       the level channels of the region map are more than zeros.
 */
#ifndef USDGEN_POMADE_API_STAGE_C_ABI_H
#define USDGEN_POMADE_API_STAGE_C_ABI_H

#include "usdGenPomade/api.h"
#include "usdGenPomade/pomadeApi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The message for the last failed call on this thread (never null). */
USDGENPOMADE_API const char *Pomade_StageGetLastError(void);

/* Hydrate `ctx` from a committed groom (plan/17 §2.5). `layerOrStage` names
 * either an open SdfLayer identifier (an anonymous live layer included) or a
 * file path to open; `groomPath` is the absolute path of the
 * UsdGenPomadeGroom prim. The whole hierarchy comes back: L1 restored, deeper
 * levels re-derived from the stored (tubeId, seed) subdivision plus deltas,
 * imports re-imported, and stored guides that no tube claims turned into
 * locked tubes. Out-params may be null. */
int USDGENPOMADE_API Pomade_Hydrate(PomadeModelContext *ctx,
                                  const char *layerOrStage,
                                  const char *groomPath, int *outTubeCount,
                                  int *outGuideCount, int *outImportedCount);

/* Pomade_Hydrate over a stage composed from several open layers, so the tool
 * hydrates exactly what usdview shows (SS-03). `layerIdentifiers[0]` is the
 * root layer; with two identifiers the second is the session layer; with
 * more, identifiers 1..n-1 are the session layer's sublayers, strongest
 * first. Each identifier is found in the layer registry first (anonymous
 * layers included) and opened from disk otherwise. */
int USDGENPOMADE_API Pomade_HydrateFromLayers(
    PomadeModelContext *ctx, const char *const *layerIdentifiers,
    int layerCount, const char *groomPath, int *outTubeCount,
    int *outGuideCount, int *outImportedCount);

/* Per-tube §5.2 center operations. tubeId 0 is the primary tube; an id that
 * names a derived child refuses the two layout-changing ones (its CV count
 * belongs to the parent's subdivision). */
int USDGENPOMADE_API Pomade_InsertTubeCenterCV(PomadeModelContext *ctx,
                                             int tubeId, int atIndex);
int USDGENPOMADE_API Pomade_DeleteTubeCenterCV(PomadeModelContext *ctx,
                                             int tubeId, int index);
int USDGENPOMADE_API Pomade_SetTubeLengthFor(PomadeModelContext *ctx, int tubeId,
                                           float length);
int USDGENPOMADE_API Pomade_MatchTubeSurface(PomadeModelContext *ctx, int tubeId);
int USDGENPOMADE_API Pomade_SnapTubeRootToScalp(PomadeModelContext *ctx,
                                              int tubeId);
int USDGENPOMADE_API Pomade_RelaxTubeCenter(PomadeModelContext *ctx, int tubeId,
                                          float strength, int iterations);

/* Per-tube §5.2 section operations. */
int USDGENPOMADE_API Pomade_MoveTubeSectionRing(PomadeModelContext *ctx,
                                              int tubeId, int ring, float du,
                                              float dv);
int USDGENPOMADE_API Pomade_ScaleTubeSectionRing(PomadeModelContext *ctx,
                                               int tubeId, int ring,
                                               float scale);
int USDGENPOMADE_API Pomade_TwistTubeSectionRing(PomadeModelContext *ctx,
                                               int tubeId, int ring,
                                               float radians);
int USDGENPOMADE_API Pomade_MoveTubeSectionCV(PomadeModelContext *ctx, int tubeId,
                                            int ring, int slot, float du,
                                            float dv);
int USDGENPOMADE_API Pomade_AddTubeSectionRing(PomadeModelContext *ctx,
                                             int tubeId, float t);
int USDGENPOMADE_API Pomade_RemoveTubeSectionRing(PomadeModelContext *ctx,
                                                int tubeId, int ring);
int USDGENPOMADE_API Pomade_CopyTubeSectionRing(PomadeModelContext *ctx,
                                              int tubeId, int src, int dst);

/* Per-tube §5.3 fill params. `profile`/`profileCount` are the flattened
 * (position, value) pairs; pass null/0 for a uniform length. */
int USDGENPOMADE_API Pomade_SetTubeFillParams(PomadeModelContext *ctx, int tubeId,
                                            float density, int cvCount,
                                            int seed, float edgeBias,
                                            const float *profile,
                                            int profileCount);
int USDGENPOMADE_API Pomade_GetTubeFillParams(PomadeModelContext *ctx, int tubeId,
                                            float *outDensity, int *outCvCount,
                                            int *outSeed, float *outEdgeBias,
                                            int *outProfileCount);
/* 1 while the tube has subdivided children (its own fill is suspended),
 * 0 otherwise, negative on a bad argument. */
int USDGENPOMADE_API Pomade_IsTubeFillSuspended(PomadeModelContext const *ctx,
                                              int tubeId);

/* Hierarchy reads the stage contract needs: the parent and the index within
 * that parent's subdivision (-1 at L1 / for an on-the-fly parent). */
int USDGENPOMADE_API Pomade_GetTubeParent(PomadeModelContext const *ctx,
                                        int tubeId, int *outParent,
                                        int *outChildIndex);
/* 1 when the tube is an on-the-fly parent the artist kept, else 0. */
int USDGENPOMADE_API Pomade_IsTubePersistent(PomadeModelContext const *ctx,
                                           int tubeId);
/* SL-03 whole-tube Delete: remove `n` tubes, each with its subtree, as one
 * undo step (inside a gesture bracket the bracket's snapshot covers it).
 * POMADE_ERROR, model untouched, for n <= 0, an unknown id, tube 0 or any
 * other L1 root -- an L1 tube belongs to its graph region, which would
 * only rebuild it; Pomade_StageGetLastError names the refused tube. */
int USDGENPOMADE_API Pomade_RemoveTubes(PomadeModelContext *ctx, const int *ids,
                                      int n);

/* Enqueue a bake that carries the hierarchy, so channel k of the map holds
 * the level-(k+1) tube id (plan/17 §4.5). Same coalescing and worker as
 * Pomade_BakeEnqueue; `levelCount` follows Pomade_BakeSetOptions. */
int USDGENPOMADE_API Pomade_BakeEnqueueLevels(PomadeBakeContext *bake);

/* V4: where one section ring SITS (plan/18 §3.6 TubeLoop).
 *
 * Pomade_GetTubeSection hands out the ring in its own chart — (u, v) pairs,
 * a scale and a twist — which is the space Pomade_MoveTubeSectionRing and
 * Pomade_MoveTubeSectionCV take their deltas in. A gizmo drag arrives in
 * WORLD units, so the viewport needs the chart's placement to convert one
 * into the other, and it must be the SAME placement the K5 tessellation and
 * the K11 section-CV candidates use or the ring would not follow the
 * cursor. That placement is the K4 frame at the ring's t, and this is the
 * only way to read it from outside the model.
 *
 * `outOrigin` takes the ring's centroid (the point K11 picks a ring by) and
 * `outFrame` the u, v, w axes as 3 floats each: u and v span the ring plane
 * (the frame's normal and binormal) and w is the center tangent. Scale and
 * twist are handed back with them so one call carries everything the
 * conversion needs:
 *
 *   world delta D  ->  a = D.u, b = D.v
 *                  ->  du = ( a cos(twist) + b sin(twist)) / scale
 *                      dv = (-a sin(twist) + b cos(twist)) / scale
 *
 * Any out-param may be null. POMADE_ERROR for an unknown tube, a ring index
 * out of range, or a center column the K4 pass refuses. */
int USDGENPOMADE_API Pomade_GetTubeSectionFrame(PomadeModelContext *ctx,
                                              int tubeId, int ring,
                                              float *outOrigin,
                                              float *outFrame,
                                              float *outScale,
                                              float *outTwist);

/* V5: the shaped sculpt stroke (plan/18 section 3.6 SculptLoop, section 7
 * G13). The old Pomade_SculptStroke takes per-CV deltas the CALLER weighted,
 * which is why comb, twist, the falloff and the mirror lived in Python and
 * no two callers had to agree. This one takes the STROKE and shapes it:
 *
 *   viewProj/w/h/x/y/radiusPx   the screen falloff -- the same row-major
 *                               projection and top-left physical pixels
 *                               Pomade_Pick takes, and the brush ring the
 *                               artist sees;
 *   deltaWorld[3]               the drag (grab) or the push direction
 *                               (comb); ignored by smooth;
 *   amount                      the brush scalar at full weight: comb push
 *                               distance, smooth strength in [0, 1],
 *                               lengthen fraction (negative shortens),
 *                               twist radians;
 *   tCenter/tRadius             the falloff along the curve; tRadius <= 0
 *                               means no t bound;
 *   preserveLength              the K6 rule, ignored by lengthen/shorten
 *                               (those SET the length);
 *   mirrorX                     also stroke the tube symmetric about
 *                               x = 0, in mirrored space, so the far side
 *                               gets the same stroke rather than a negated
 *                               dx on this one.
 *
 * *outTouched (optional) receives the number of CVs that moved; 0 is a
 * valid answer (the brush was over nothing) and bumps no version. */
int USDGENPOMADE_API Pomade_SculptStrokeShaped(PomadeModelContext *ctx,
                                             int tubeId, const char *brush,
                                             const float *viewProj, int w,
                                             int h, float x, float y,
                                             float radiusPx,
                                             const float *deltaWorld,
                                             float amount, float tCenter,
                                             float tRadius,
                                             int preserveLength, int mirrorX,
                                             int *outTouched);

/* V5: subdivide along a DRAWN edge (plan/18 section 3.6 HierarchyLoop;
 * audit section 5.4 row 267 -- the edge line coefficients were hard-coded
 * to (1, 0, 0), so "the split follows the drawn edge" was unreachable).
 * `worldA`/`worldB` are the two ends of the stroke the artist drew across
 * the root region, in WORLD space; they are projected into the tube's root
 * frame here, since that chart is the model's own and no caller should
 * have to rebuild it. Always two children (edge mode). */
int USDGENPOMADE_API Pomade_SubdivideTubeEdge(PomadeModelContext *ctx,
                                            int tubeId, const float *worldA,
                                            const float *worldB, int seed,
                                            int *outIds, int outCap,
                                            int *outCount);

/* V5: the per-level draw mode (plan/18 section 3.7, the fallback ladder's
 * fourth step). Pomade_SetLevelDisplay's visible/x-ray pair cannot say
 * "centers only", and hiding the level would take the center curves with
 * it, so this is the third flag: the level keeps its center curves and CV
 * dots and drops the tube mesh, the rings and the guides. Viewport state,
 * like the other two: it re-publishes, never re-tessellates, and is never
 * authored to USD. */
int USDGENPOMADE_API Pomade_SetLevelDrawMode(PomadeModelContext *ctx, int level,
                                           int visible, int xray,
                                           int centersOnly);
int USDGENPOMADE_API Pomade_GetLevelDrawMode(PomadeModelContext const *ctx,
                                           int level, int *outVisible,
                                           int *outXray, int *outCentersOnly);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* USDGEN_POMADE_API_STAGE_C_ABI_H */
