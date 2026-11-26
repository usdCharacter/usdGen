/* usdGenImaging/cApi.h — USDGEN_IMAGING_C_API. Control and scalars only; no array
   ever crosses per element. Every entry point is extern "C", returns int status
   (0 = success, non-zero = error) except the two const char * accessors, and
   never throws across the boundary (ADR §9.4 R31).
   UsdGenImaging_GetLastError() returns the message for the calling thread.

   Contract C4 — declaration source of truth: plan/08-tools.md §C4 listing (18 entry
   points: the seventeen implemented-scope ADR §9.4 R31 names plus GetLastError;
   SetMaskVisualisation went with UsdGenMaskAPI). Signatures here
   are copied verbatim from that listing; if this header and 06-imaging.md §3.8
   ever diverge, 08-tools.md governs.
   IMPLEMENTATION STATUS (plan/13 D3): declared, NOT implemented — no cApi.cpp
   exists yet (M5 scope). Every entry point must return USDGEN_NOT_IMPLEMENTED
   until its C++ target lands; link the ABI only when that is true. */
#ifndef USDGEN_IMAGING_C_API_H
#define USDGEN_IMAGING_C_API_H

#include "usdGenImaging/api.h"

/* Visibility: the imaging lib follows the PXR convention (USDGENIMAGING_API
   from the build-macros header pulled in via usdGenImaging/api.h). The M0
   placeholder api.h defines no macro, so fall back to default visibility —
   correct for the Linux shared-object build (whole-archive default export). */
#ifndef USDGENIMAGING_API
#define USDGENIMAGING_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes (08-tools.md C4: 0 = success, non-zero = error). */
#define USDGEN_OK                0
#define USDGEN_ERROR             1  /* bad args / engine error (see GetLastError)   */
#define USDGEN_NOT_IMPLEMENTED   2  /* no C++ target on this build yet — honest fail */

int         USDGENIMAGING_API UsdGenImaging_Activate(long long stageCacheId, const char *groomRootPath,
                                   double frame);
int         USDGENIMAGING_API UsdGenImaging_Deactivate(void);
int         USDGENIMAGING_API UsdGenImaging_SetTime(double frame);
int         USDGENIMAGING_API UsdGenImaging_Commit(void);   /* R32 trigger (a): after SetTime and after  */
                                            /* live-override changes ONLY. Never for a   */
                                            /* stage edit — trigger (c) owns those (2.4) */
int         USDGENIMAGING_API UsdGenImaging_SetContext(const char *context); /* "interactive" | "render" */
long long   USDGENIMAGING_API UsdGenImaging_GetGeneration(void);             /* publication handshake    */
long long   USDGENIMAGING_API UsdGenImaging_GetTopologyGeneration(const char *descriptionPath);
                     /* bumps ONLY on a topology change (R31): a recapture, a new        */
                     /* curve set, a density or ceiling change. Never on a deform         */
                     /* frame and never on a live-override republish (3.4, 6.3)           */

int         USDGENIMAGING_API UsdGenImaging_BeginLiveOverride(const char *primPath);
int         USDGENIMAGING_API UsdGenImaging_SetLiveOverrideIndexed(const char *primPath, const int *cvIndices,
                                                 int count, const float *xyz);
int         USDGENIMAGING_API UsdGenImaging_ClearLiveOverride(const char *primPath);

int         USDGENIMAGING_API UsdGenImaging_PickCV(const char *primPath, const float viewProj[16],
                                 int w, int h, float x, float y, float radiusPx,
                                 int *outCurve, int *outCv, float *outDistPx);
int         USDGENIMAGING_API UsdGenImaging_Footprint(const char *primPath, const float viewProj[16],
                                    int w, int h, float x, float y, float radiusPx,
                                    int *outIdx, int maxOut, int *outCount);
int         USDGENIMAGING_API UsdGenImaging_ClosestSurfacePoint(const char *surfacePath, const float p[3],
                                              int *outFace, float outUV[2], float outP[3]);
int         USDGENIMAGING_API UsdGenImaging_BuildMirrorMap(const char *descriptionPath, int axis);

int         USDGENIMAGING_API UsdGenImaging_SetInteractiveLOD(const char *descriptionPath, int maxCurves);
int         USDGENIMAGING_API UsdGenImaging_ReloadMaps(void);
const char *USDGENIMAGING_API UsdGenImaging_GetStatsJson(void);
const char *USDGENIMAGING_API UsdGenImaging_GetLastError(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* USDGEN_IMAGING_C_API_H */
