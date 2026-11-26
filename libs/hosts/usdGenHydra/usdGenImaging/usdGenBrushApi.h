/* usdGenImaging/usdGenBrushApi.h — the C ABI behind the usdview attribute brush.
 *
 * The brush tool (plugin/usdGenTools/python/usdGenTools/brush*.py) used to do
 * everything per element in Python: the pick, the dab kernel, the footprint
 * spill onto neighbouring faces, the corner extraction and the preview
 * colours, each touching O(faces) Python objects per mouse move. This header
 * moves that hot path into C++ behind opaque handles, in the style of
 * usdGenToolsApi.h / pomadeApi.h, and adds the viewport overlay the preview
 * draws through (attributePreviewSceneIndex.h), so a move writes nothing to
 * the stage.
 *
 * Handles:
 *   mesh   — a world-space quad-mesh snapshot plus a uniform-grid face index
 *            (pick, footprint, edge lengths, vertex adjacency). Read-only
 *            after creation; one per gesture or per bind. An optional face
 *            mask (MeshCreateMasked: a face UsdGeomSubset's parent-mesh face
 *            ids, plan/02-schema.md §2.20) makes only those faces paintable.
 *   stroke — the press-time base (4 corners per face, bilinearly upsampled to
 *            a res x res texel grid per face), the working grid, and the
 *            recorded PRIMARY dabs. Footprint expansion happens inside
 *            StrokeDab through the mesh, so Commit (base + every recorded dab
 *            replayed through the same kernel) reproduces the working grid
 *            exactly. A stroke keeps its mesh alive.
 *
 * Conventions (shared with brushPick.py / brushAuthor.py): quads only;
 * face-local (u, v) with corners v0..v3 at (0,0),(1,0),(1,1),(0,1); corner
 * arrays are laid out [face][corner][channel]; floats are float32, points
 * double. Every entry point is extern "C", never throws, and returns 0 or a
 * negative value on error unless stated otherwise.
 *
 * Smooth: the persisted data is 4 corners per face, so a face-local texel
 * stencil is a no-op on it (a bilinear patch is its own 3x3 mean). Smooth
 * therefore works on corners across faces: every face-corner the footprint
 * reaches is blended, by the dab weight, toward the mean of its mesh vertex
 * (all face-corners sharing it) and that vertex's edge neighbours; affected
 * faces are then re-upsampled from their corners.
 *
 * Face mask: face ids stay parent-mesh ids and every array stays whole-mesh
 * (4 corners per parent face), so a masked mesh changes only what a stroke
 * may write. The pick still intersects every face -- a masked face in front
 * occludes -- and reports a miss when the nearest hit is masked out; the
 * footprint never spills onto a masked face; the smoother's vertex means
 * and edge neighbours come from paintable faces only; StrokeDab rejects a
 * primary on a masked face. Masked faces' working corners therefore always
 * equal the base.
 */
#ifndef USDGEN_IMAGING_BRUSH_API_H
#define USDGEN_IMAGING_BRUSH_API_H

#include "usdGenImaging/api.h"
#include <stddef.h>

#ifndef USDGENIMAGING_API
#define USDGENIMAGING_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only, nonblocking playback publication snapshot. Returns required bytes
 * including NUL; insufficient capacity writes nothing. Returns zero on an
 * exception. selectorJson contains renderer and top-level groom roots. */
USDGENIMAGING_API size_t usdGenImaging_copy_playback_status_json(
    const char* selectorJson, double requestedFrame, char* buffer, size_t capacity);

#define USDGEN_BRUSH_API_VERSION 1

#define USDGEN_BRUSH_MODE_SET 0
#define USDGEN_BRUSH_MODE_ADD 1
#define USDGEN_BRUSH_MODE_SMOOTH 2
#define USDGEN_BRUSH_MODE_ERASE 3

#define USDGEN_BRUSH_FALLOFF_CONSTANT 0
#define USDGEN_BRUSH_FALLOFF_LINEAR 1
#define USDGEN_BRUSH_FALLOFF_SMOOTH 2

#define USDGEN_BRUSH_COLORMAP_HEAT 0
#define USDGEN_BRUSH_COLORMAP_GRAY 1

USDGENIMAGING_API int UsdGenBrush_ApiVersion(void);

/* ---- mesh --------------------------------------------------------------- */

/* worldPoints: numPoints*3 doubles; faceVertexIndices: numFaces*4 ints.
 * NULL for bad input (no faces, an out-of-range index, non-finite points). */
USDGENIMAGING_API void *UsdGenBrush_MeshCreate(const double *worldPoints, int numPoints,
                                               const int *faceVertexIndices, int numFaces);
/* MeshCreate plus a face mask: only the maskCount face ids in maskFaces
 * (parent-mesh ids, any order, duplicates allowed) are paintable; see "Face
 * mask" above. maskFaces NULL is MeshCreate (every face paintable); a
 * non-NULL maskFaces with maskCount 0 masks every face out. NULL for a
 * negative maskCount or an id outside [0, numFaces). */
USDGENIMAGING_API void *UsdGenBrush_MeshCreateMasked(const double *worldPoints, int numPoints,
                                                     const int *faceVertexIndices, int numFaces,
                                                     const int *maskFaces, int maskCount);
USDGENIMAGING_API void UsdGenBrush_MeshDestroy(void *mesh);
USDGENIMAGING_API int UsdGenBrush_MeshFaceCount(void *mesh);
/* 1 when the face is paintable (always, on an unmasked mesh), 0 when the
 * mask leaves it out, <0 for a bad mesh or face. */
USDGENIMAGING_API int UsdGenBrush_MeshFaceInMask(void *mesh, int face);
/* Nearest quad under the ray (both windings). 1 hit, 0 miss (including a
 * nearest hit on a masked face), <0 error. */
USDGENIMAGING_API int UsdGenBrush_MeshPick(void *mesh, const double *rayOrigin3,
                                           const double *rayDir3, int *outFace,
                                           float *outU, float *outV, double *outPoint3);
/* Faces within worldRadius of point3 (NULL: the bilinear point of (face, u,
 * v)); the primary face first with its exact (u, v), the rest with their
 * closest-point (u, v). Masked faces never appear past the primary.
 * outRadiusUV[i] = worldRadius / longestEdge(face i).
 * Writes at most cap entries and returns the TOTAL footprint size (like
 * StrokeTakeTouched), so a return > cap means truncated; <0 on error. */
USDGENIMAGING_API int UsdGenBrush_MeshFootprint(void *mesh, int face, float u, float v,
                                                const double *point3, float worldRadius,
                                                int *outFaces, float *outU, float *outV,
                                                float *outRadiusUV, int cap);
/* Longest world edge of the face, 0 when unusable. */
USDGENIMAGING_API float UsdGenBrush_MeshFaceEdgeLen(void *mesh, int face);
/* Texels per face edge: 32, halved while numFaces*res*res > budgetTexels
 * (budgetTexels <= 0 means 4,000,000), never below 4 -- so one of 32, 16, 8,
 * 4. Face size does not enter (only four corners per face persist, so texels
 * only buy in-stroke detail); the median mean-edge length is reported in
 * infoBuf (may be NULL) for information. -1 on a bad mesh. */
USDGENIMAGING_API int UsdGenBrush_MeshSuggestResolution(void *mesh, int budgetTexels,
                                                        char *infoBuf, int infoLen);

/* ---- stroke ------------------------------------------------------------- */

/* resolution: power of two in [1, 256]; channels 1 or 3. baseCorners is
 * numFaces*4*channels floats (NULL fills with defaultValue). mesh may be
 * NULL (no footprint spill, face-local smooth); otherwise its face count
 * must equal numFaces. */
USDGENIMAGING_API void *UsdGenBrush_StrokeCreate(void *mesh, int numFaces, int resolution,
                                                 int channels, const float *baseCorners,
                                                 float defaultValue);
USDGENIMAGING_API void UsdGenBrush_StrokeDestroy(void *stroke);
/* Record one primary dab (isMove: interpolate from the previous primary on
 * the same face, stamps spaced spacing*radius apart) and apply it and its
 * footprint spill to the working grid. Returns the number of stamps applied
 * (primaries times footprint faces), <0 on a rejected dab (nothing recorded;
 * -11: the face is outside the mesh's face mask). */
USDGENIMAGING_API int UsdGenBrush_StrokeDab(void *stroke, int face, float u, float v,
                                            float radiusUV, float hardness, float strength,
                                            float value, int channel, int mode, int falloff,
                                            int isMove, float spacing);
/* Faces whose working values changed since the last take, ascending. Returns
 * the total pending count; writes min(count, cap) and clears only when the
 * whole set fit. */
USDGENIMAGING_API int UsdGenBrush_StrokeTakeTouched(void *stroke, int *outFaces, int cap);
USDGENIMAGING_API int UsdGenBrush_StrokeDabCount(void *stroke);
USDGENIMAGING_API int UsdGenBrush_StrokeFaceCount(void *stroke);
USDGENIMAGING_API int UsdGenBrush_StrokeChannels(void *stroke);
/* Working corners (numFaces*4*channels). Returns the float count. */
USDGENIMAGING_API int UsdGenBrush_StrokeWorkingCorners(void *stroke, float *out);
/* Base + every recorded dab, recomputed from scratch. Returns the count. */
USDGENIMAGING_API int UsdGenBrush_StrokeCommitCorners(void *stroke, float *out);
/* Forget every dab and reset the working grid to the base. */
USDGENIMAGING_API void UsdGenBrush_StrokeAbort(void *stroke);
/* Working corners of `channel` through the colour map over [lo, hi]
 * (brushPreview.py's heat/gray ramps). outRGB: numFaces*4*3. */
USDGENIMAGING_API int UsdGenBrush_StrokePreviewColors(void *stroke, int channel, int colorMap,
                                                      float lo, float hi, float *outRGB);

/* ---- viewport overlay --------------------------------------------------- */

/* Overlay primvars/displayColor on the mesh at primPath with faceVarying
 * colours (count RGB triples, count == numFaces*4) in every live
 * attribute-preview scene index. The first Set on a path dirties
 * primvars/displayColor, later Sets only its primvarValue. 1 on success. */
USDGENIMAGING_API int UsdGenBrush_PreviewSet(const char *primPath, const float *rgbFaceVarying,
                                             int count);
/* Remove the overlay (upstream shows through again). 1 when one was removed. */
USDGENIMAGING_API int UsdGenBrush_PreviewClear(const char *primPath);
/* Remove every overlay (stage replace, palette close). The count removed. */
USDGENIMAGING_API int UsdGenBrush_PreviewClearAll(void);
/* Number of attribute-preview scene indices currently in a render chain. */
USDGENIMAGING_API int UsdGenBrush_PreviewIndexCount(void);

/* ---- groom observation --------------------------------------------------- */

/* Process-wide groom scene-index totals: cooks issued and generations
 * published. A live-groom drag advances both; -1 on error. */
USDGENIMAGING_API long long UsdGenBrush_GroomCookCount(void);
USDGENIMAGING_API long long UsdGenBrush_GroomPublishCount(void);

/* Published strands of the groom rooted at `groomPath` (a description or
 * groom root) across live groom scene indices: curve count and summed
 * control-polygon length over its synthetic tiles. 1 when a tile is
 * published, 0 when none is, -1 on error. Either out pointer may be NULL. */
USDGENIMAGING_API int UsdGenBrush_GroomCurveStats(const char *groomPath,
                                                  long long *curves,
                                                  double *totalLength);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* USDGEN_IMAGING_BRUSH_API_H */
