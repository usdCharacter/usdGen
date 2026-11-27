// usdGenPomade — CUDA kernel launches (plan/17 §4, P0–P2).
//
// Header-only contract, Qt-free, no USD: the .cu TU implements the launches,
// the CPU twins live beside them (pomadeTessellate.h, pomadeScalp.cpp,
// pomadeGraph.cpp, pomadeRegion.cpp). Compiled only when
// USDGEN_POMADE_HAS_CUDA; every call site guards on it, and the T0/T1 tests
// skip the parity checks on GPU-less builds (plan/17 §3.4 CPU fallback).
#ifndef USDGEN_POMADE_KERNELS_H
#define USDGEN_POMADE_KERNELS_H

#include "usdGenPomade/api.h"
#include "usdGenPomade/pomadeCheck.h"
#include "usdGenPomade/pomadeHierarchy.h"
#include "usdGenPomade/pomadeTessellate.h"
#include "usdGenPomade/pomadeTube.h"

#include <cstddef>
#include <string>
#include <vector>

#ifdef USDGEN_POMADE_HAS_CUDA
#include <cuda_runtime.h>
#else
// Spelling-only fallback so call sites compile on GPU-less builds. No launch
// is ever issued there (every caller checks USDGEN_POMADE_HAS_CUDA first).
typedef void *cudaStream_t;
#endif

namespace usdGenPomade {

// Tessellate the test tube on `stream`: one thread per vertex, writing
// interleaved positions (3 floats) and normals (3 floats). Returns false on
// launch failure with a NUL-terminated message in errBuf (may be null).
USDGENPOMADE_API bool PomadeLaunchTessellate(PomadeTubeShape const &shape,
                           float const *deviceCenterX,
                           float const *deviceCenterY,
                           float const *deviceCenterZ,
                           float *devicePositions,
                           float *deviceNormals,
                           cudaStream_t stream,
                           char *errBuf,
                           size_t errBufLen);

// -- K1: LBVH raycast batch -----------------------------------------------
//
// Traverses the device LBVH (uploaded PomadeScalpBvhNode bytes, same layout)
// for `rayCount` rays. One thread per ray; the nearest hit wins. The device
// writes face ids, distances, positions and geometric normals; face-local
// (u, v) is derived on the host from (face, p) via PomadeFaceCoordinate, so
// no second convention exists. CPU twin: PomadeRaycastCpu (same traversal
// order and triangle tests).
struct PomadeDeviceHit {
    int faceId = -1;  // -1 = miss
    float t = 0.0f;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float nx = 0.0f, ny = 1.0f, nz = 0.0f;
};

// Plain-C mirror of PomadeScalpBvhNode (pomadeScalp.h): 6 floats + 4 ints.
// The caller uploads PomadeScalpBvh::nodes verbatim (verified by
// PomadeBvhNodeBytes(), which both sides assert).
struct PomadeDeviceBvhNode {
    float minX, minY, minZ, maxX, maxY, maxZ;
    int left, right, faceBegin, faceEnd;
};

USDGENPOMADE_API bool PomadeLaunchRaycastBatch(float const *devicePoints,
                             int const *deviceFaceCounts,
                             int const *deviceFaceIndices,
                             int const *deviceFaceOffsets,
                             float const *deviceFaceNormals,
                             PomadeDeviceBvhNode const *deviceNodes,
                             int const *deviceOrder,
                             int faceCount,
                             int nodeCount,
                             int root,
                             float const *deviceOrigins,  // 3 * rayCount
                             float const *deviceDirs,     // 3 * rayCount
                             PomadeDeviceHit *deviceHits,  // rayCount
                             int rayCount,
                             cudaStream_t stream,
                             char *errBuf,
                             size_t errBufLen);

// -- K2: edge trace --------------------------------------------------------
//
// One thread per chord sample: lerp(a, b) projected to the surface by
// brute-force closest point over all faces (test-scale meshes; the CPU twin
// uses the same projection with an AABB pre-filter). Ends are pinned
// exactly, as the CPU twin does. CPU twin: PomadeTraceEdgeCpu.
USDGENPOMADE_API bool PomadeLaunchEdgeTrace(float const *devicePoints,
                          int const *deviceFaceCounts,
                          int const *deviceFaceIndices,
                          int const *deviceFaceOffsets,
                          int faceCount,
                          float const a[3],
                          float const b[3],
                          float *deviceOutPos,  // 3 * sampleCount
                          int *deviceOutFaces,  // sampleCount
                          int sampleCount,
                          cudaStream_t stream,
                          char *errBuf,
                          size_t errBufLen);

// -- K3: region classification ----------------------------------------------
//
// One thread per query point: the same projected ray-cast inside-test over
// the same flattened loops as PomadeClassifyPointCpu (no flood fill on
// either side; the bake constrains by connectivity on the host). Writes the
// lowest claiming interp id, or -1.
USDGENPOMADE_API bool PomadeLaunchClassifyPoints(float const *devicePositions,  // 3 * pointCount
                               int pointCount,
                               float const *deviceLoopPts,  // 3 * total
                               int const *deviceLoopBegin,  // regionCount
                               int const *deviceLoopCount,  // regionCount
                               float const *devicePlaneN,   // 3 * regionCount
                               float const *devicePlaneP,   // 3 * regionCount
                               float const *deviceBasisU,   // 3 * regionCount
                               float const *deviceBasisV,   // 3 * regionCount
                               int const *deviceInterpIds,  // regionCount
                               int regionCount,
                               int *deviceOutIds,  // pointCount
                               cudaStream_t stream,
                               char *errBuf,
                               size_t errBufLen);

// -- K4: rotation-minimising center frames -----------------------------------
//
// The RMF propagation is inherently sequential, so one thread computes all
// `nCv` frames through the shared PomadeReflectNormal spelling; the lane
// exists to keep the data device-resident for K5/K9, not for parallelism.
// CPU twin: PomadeCenterFramesCpu (bit-exact: arithmetic only).
USDGENPOMADE_API bool PomadeLaunchCenterFrames(float const *deviceCenterX,
                             float const *deviceCenterY,
                             float const *deviceCenterZ, int nCv,
                             PomadeFrame *deviceFrames, cudaStream_t stream,
                             char *errBuf, size_t errBufLen);

// -- K5: Hermite tube tessellation ------------------------------------------
//
// One thread per vertex over the ((nSec - 1) * segmentsPerSpan + 1) x
// ringVerts grid, ring-major. Sections arrive as device arrays (t, packed
// u/v rows of ringVerts, scale, twist); frames come from K4. CPU twin:
// PomadeTessellateCpu (tolerance: twist trigonometry).
USDGENPOMADE_API bool PomadeLaunchTubeTessellate(
    float const *deviceCenterX, float const *deviceCenterY,
    float const *deviceCenterZ, int nCv, PomadeFrame const *deviceFrames,
    float const *deviceSectionT, float const *deviceSectionU,
    float const *deviceSectionV, float const *deviceSectionScale,
    float const *deviceSectionTwist, int nSec, int ringVerts,
    int segmentsPerSpan, float *devicePositions, float *deviceNormals,
    float *deviceRingT, cudaStream_t stream, char *errBuf, size_t errBufLen,
    float const *deviceRootOffsets = nullptr);

// Plain-C mirror of PomadeGuideRoot for the K8/K9 device lane.
struct PomadeDeviceRoot {
    int faceId = -1;
    float u = 0.0f, v = 0.0f;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float ru = 0.0f, rv = 0.0f;
};

// -- K8: blue-noise root sampling --------------------------------------------
//
// Dart throwing is sequential (each keep tests the kept set), so one
// thread runs the identical hash-order loop; frozen roots are already in
// `deviceRoots[0, frozen)` (uploaded by the caller) and seed the stream.
// The mesh form writes the sampled count to `deviceOutCount` (1 int).
// CPU twins: PomadeRootSampleDiscCpu / PomadeRootSampleMeshCpu (bit-exact:
// the same hash draws in the same order; sqrt/cos/sin take tolerance).
USDGENPOMADE_API bool PomadeLaunchRootSampleDisc(int tubeId, int seed, float radius,
                               float centerX, float centerY, float centerZ,
                               PomadeFrame const &rootFrame, int count,
                               PomadeDeviceRoot *deviceRoots, int frozen,
                               cudaStream_t stream, char *errBuf,
                               size_t errBufLen);
USDGENPOMADE_API bool PomadeLaunchRootSampleMesh(
    float const *devicePoints, int const *deviceFaceCounts,
    int const *deviceFaceIndices, int const *deviceFaceOffsets, int faceCount,
    int const *deviceRegionFaces, int regionFaceCount, float density,
    int tubeId, int seed, float rootCenterX, float rootCenterY,
    float rootCenterZ, PomadeFrame const &rootFrame, float rootRadius,
    PomadeDeviceRoot *deviceRoots, int maxRoots, int frozen,
    int *deviceOutCount, cudaStream_t stream, char *errBuf, size_t errBufLen);

// -- K9: guide fill -----------------------------------------------------------
//
// One thread per guide over guideCount x cvCount CVs. Each guide CV carries
// a host-built triangle/material binding into the complete K5 section
// payload, so terminal and interior rings retain authored U/V, scale/twist.
// CPU twin: PomadeGuideFillCpu (tolerance: Hermite/trigonometry).
USDGENPOMADE_API bool PomadeLaunchGuideFill(
    float const *deviceCenterX, float const *deviceCenterY,
    float const *deviceCenterZ, int nCv, PomadeFrame const *deviceFrames,
    float const *deviceSectionT, float const *deviceSectionU,
    float const *deviceSectionV, float const *deviceSectionScale,
    float const *deviceSectionTwist, int nSec, int ringVerts,
    PomadeDeviceRoot const *deviceRoots,
    PomadeGuideMaterialBinding const *deviceBindings, int guideCount,
    float edgeBias, float const *deviceProfilePairs, int profilePairCount,
    float *deviceOut, float *deviceLengths, int cvCount, cudaStream_t stream,
    char *errBuf, size_t errBufLen);

// -- K10: arc-length resample + root frames ------------------------------------
//
// One thread per guide (input CV count per guide <= 64, uniform or ragged
// via offsets). Root directions are 9 floats (tube root N/B/T). CPU twin:
// PomadeGuideResampleCpu (tolerance: sqrt/division chains).
USDGENPOMADE_API bool PomadeLaunchGuideResample(float const *devicePoints,
                              int const *deviceOffsets,
                              int const *deviceCounts, int guideCount,
                              int cvCount, float const rootDirs[9],
                              float *deviceOut, int *deviceOutCounts,
                              double *deviceOutFrames, cudaStream_t stream,
                              char *errBuf, size_t errBufLen);

// -- K11: screen-space pick reduction -------------------------------------------
//
// Two-stage reduction over one kind's candidate positions (3 floats each):
// per-block best, then a single-block finalize into `deviceBest` (scalar
// readback: the host copies one struct). The caller runs one reduction
// per kind present in the mask and keeps the pixel-nearest (depth breaks
// ties), which is exactly PomadePickCpu's rule. ViewProj uses the
// gpu/picking.cu convention (row-major, row-vector, top-left origin).
struct PomadeDevicePickBest {
    int index = -1;  // -1 = miss
    float distPx = 0.0f;
    float depth = 0.0f;
    int hit = 0;
};

USDGENPOMADE_API bool PomadeLaunchPickReduce(float const *devicePositions, int candidateCount,
                           float const viewProj[16], int w, int h, float x,
                           float y, float radiusPx,
                           PomadeDevicePickBest *deviceBest,
                           PomadeDevicePickBest *deviceScratch,
                           cudaStream_t stream, char *errBuf,
                           size_t errBufLen);

// -- K11b: marquee / lasso mask (plan/18 §2.3) ----------------------------------
//
// A rubber band selects a SET, not a winner, so there is nothing to reduce:
// one thread per candidate writes 0/1 into `deviceMask` and the host
// compacts. `devicePolygonXY` null (with polygonCount 0) selects the
// rectangle (x0, y0)-(x1, y1); otherwise the polygon of that many (x, y)
// pairs is tested and the rectangle arguments are ignored. Both tests spell
// through the PomadePointInRect / PomadePointInPolygon inlines the CPU twin
// uses, so the two lanes agree candidate for candidate.
USDGENPOMADE_API bool PomadeLaunchSelectMask(float const *devicePositions, int candidateCount,
                           float const viewProj[16], int w, int h, float x0,
                           float y0, float x1, float y1,
                           float const *devicePolygonXY, int polygonCount,
                           unsigned char *deviceMask, cudaStream_t stream,
                           char *errBuf, size_t errBufLen);

// -- K12/K13 check kernels (plan/17 §4.1, P4) -----------------------------------
//
// One thread per CV (K13) / per ordered pair + per tube (K12). Both spell
// through the pomadeCheck.h HD inlines, so the CPU twins match
// bit-exactly (TN-6). Charts come from PomadeRootChartCpu (same derivation
// the twin uses); the pair kernel re-derives AABBs on the fly with the
// twin's sequential scan order.
struct PomadeDeviceRootChart {
    int vertCount = 0;  // used ring verts in [3, 32]
    float center[3];
    float nrm[3];
    float bin[3];
    float verts[96];  // world xyz, 32 max (kPomadeCheckRingMax * 3)
};

USDGENPOMADE_API bool PomadeLaunchSmoothnessScores(float const *deviceCenterX,
                                 float const *deviceCenterY,
                                 float const *deviceCenterZ, int nCv,
                                 float *deviceScores, cudaStream_t stream,
                                 char *errBuf, size_t errBufLen);

// devicePairHits/deviceBroadHits take tubeCount^2 ints (slot i * n + j is
// the ordered pair, set only for i < j); deviceFlags takes tubeCount.
// Related pairs (PomadePairRelated over the id arrays) report 0/0.
USDGENPOMADE_API bool PomadeLaunchTubeIntersect(PomadeDeviceRootChart const *deviceCharts,
                              int const *deviceTubeIds,
                              int const *deviceParentIds, int tubeCount,
                              int *devicePairHits, int *deviceBroadHits,
                              int *deviceFlags, cudaStream_t stream,
                              char *errBuf, size_t errBufLen);

// -- K6/K7/K14 hierarchy device lanes (plan/17 §4.1) ---------------------------
//
// These three kernels consume whole tubes, and the tube store has no device
// mirror (the model keeps tubes on the host: PomadeModel::_tubes), so unlike
// the K4/K5/K11 lanes above they are host-facing. Each entry point packs the
// batch into flat device arrays, launches, and unpacks the result; the
// caller hands and receives the same PomadeTubeDesc the CPU twin uses. A
// batch is one launch, so the per-move cost is one round trip for the whole
// edited subtree rather than one per tube.
//
// Parity (TN-6): the device math spells the twins in pomadeHierarchy.cpp with
// the same operation order and the same float/double split, so the results
// are bit-identical wherever the path is arithmetic only. The three
// tolerance sites are the ring placement and section interpolation
// (cos/sin of twist), the mean-radius and arc-length accumulations (sqrt is
// correctly rounded on both sides, so these are in fact exact), and K7's
// angular sort key (atan2). Two DISTINCT cloud points whose atan2 and
// squared radius both round to the same doubles would be ordered by
// std::sort and by the device merge sort independently; identical points
// are deduplicated either way, so the difference is unobservable on any
// ring the refit accepts.
//
// Caps: the lanes use fixed per-block storage instead of a device heap, so
// a tube outside these bounds is refused (the entry point returns false and
// the caller runs the CPU twin). They cover every tube the model can build:
// PomadeTubeDesc documents ringVerts in [3, 32] and K14 emits at most
// kMaxRingVerts = 32.
constexpr int kPomadeDeviceMaxCv = 64;        // center CVs per tube
constexpr int kPomadeDeviceMaxSec = 64;       // cross-sections per tube
constexpr int kPomadeDeviceMaxRing = 32;      // ring CVs per section
constexpr int kPomadeDeviceMaxChildren = 8;   // K14 count, K7 group size

// K14 subdivide, batched over parents. Element i of `children` is what
// PomadeSubdivideTubeCpu(parents[i], parentFrames[i], params[i], ...) writes,
// child identities included. `parentFrames[i]` holds the K4 frames at
// parents[i]'s center CVs. One block per parent, one lane per child.
USDGENPOMADE_API bool PomadeSubdivideTubesDevice(
    PomadeTubeDesc const *parents,
    std::vector<PomadeFrame> const *parentFrames,
    PomadeSubdivideDesc const *params, int parentCount,
    std::vector<std::vector<PomadeTubeDesc>> *children, cudaStream_t stream,
    std::string *err);

// K7 parent average, batched over groups. Element i of `parentsOut` is what
// PomadeParentAverageCpu(groups[i], ...) writes, aggregate identity included.
// Two launches: centers plus parent frames (one block per group), then the
// per-section union refit (one block per group and section).
USDGENPOMADE_API bool PomadeParentAverageDevice(std::vector<PomadeTubeDesc> const *groups,
                              int groupCount,
                              std::vector<PomadeTubeDesc> *parentsOut,
                              cudaStream_t stream, std::string *err);

// One K6 work item: the arguments of PomadeHierarchicalSculptApplyCpu for a
// single child. The pointers are borrowed for the call only.
struct PomadeSculptApplyItem {
    PomadeTubeDesc const *derivedNew = nullptr;
    PomadeTubeDesc const *oldActual = nullptr;
    PomadeTubeDesc const *oldDerived = nullptr;
    PomadeShapeDeltas const *oldStored = nullptr;
    bool lockChildren = false;
    bool preserveLength = true;
};

// K6 hierarchical sculpt (the apply half), batched over children. Element i
// of the outputs is what PomadeHierarchicalSculptApplyCpu(items[i], ...)
// writes. The derivation half is K14 above, which production already runs
// once per parent; this lane is the per-child tail that follows it. One
// thread per item.
USDGENPOMADE_API bool PomadeHierarchicalSculptApplyDevice(
    PomadeSculptApplyItem const *items, int itemCount,
    std::vector<PomadeTubeDesc> *outActual,
    std::vector<PomadeShapeDeltas> *outStored, cudaStream_t stream,
    std::string *err);

// -- Host-facing wrappers for the K12/K13 production lanes --------------------
//
// The launches above take device pointers; these two carry the CPU twins'
// signatures, so a production call site is a one-line choice between the
// twin and the lane. Both kernels are bit-exact against their twins
// (proven in testUsdGenPomadeKernels), which is what makes them safe to run
// in production: a check that flags a tube flags it identically whichever
// lane ran. K8/K9/K10 deliberately get no wrapper. Those lanes agree with
// their twins only to a tolerance (sqrt/cos/sin in the dart stream), while
// the committer and hydrate compare generated guides bit for bit
// (pomadeModel.h, PomadeGenerateGuides), so a mixed lane would break the
// hydrate equality the P1 commit test asserts.
USDGENPOMADE_API bool PomadeSmoothnessScoreDevice(float const *cx, float const *cy,
                                float const *cz, int nCv, float *out,
                                cudaStream_t stream, std::string *err);

USDGENPOMADE_API bool PomadeTubeIntersectDevice(PomadeTubeDesc const *tubes, int tubeCount,
                              int *outFlags, cudaStream_t stream,
                              std::string *err);

}  // namespace usdGenPomade

#endif  // USDGEN_POMADE_KERNELS_H
