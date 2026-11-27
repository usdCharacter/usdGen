// usdGenPomade — P3 tube + fill model and CPU twins (plan/17 §4.1 K4/K5/K8–K11).
//
// Qt-free, no USD, no CUDA types: the .cu TU and the T0 tests share the
// USDGEN_POMADE_HD inline spellings below, and the CPU twins in pomadeTube.cpp
// implement the same operation order so TN-6 parity is bit-exact on
// arithmetic-only paths (transcendentals take tolerance).
//
// Where the Pomade publications are silent (the fill kernel, the sculpt
// formulation), the math here is our own reconstruction and is documented
// as such, per plan/17 §4.1 K6/K9.
//
// Conventions:
//   * A tube is a center curve (CVs) plus planar cross-sections in the
//     center's rotation-minimising frames (K4). Ring CV count is uniform
//     per tube (8 default, up to 32).
//   * Section CVs live in the section plane as (u, v) pairs; `scale`
//     multiplies them and `twist` (radians) rotates them before placement.
//   * Guide roots carry (faceId, u, v) on the scalp plus a world position;
//     K9 derives the normalised root-plane coordinate from the root frame.
//   * The length profile is flattened (position, value) pairs over the root
//     radial fraction (0 = center, 1 = edge); empty means uniform length.
//     Evaluation is piecewise-linear with clamped ends (the P3 subset of
//     the usdGenMath R11 ramp family; no public header may include
//     usdGenMath/ per N-7, so the evaluator lives here).
//   * Hash draws follow the usdGenMath/hash.h SplitMix64 spelling with a
//     Pomade-local salt, so the streams never couple to engine looks.
#ifndef USDGEN_POMADE_TUBE_H
#define USDGEN_POMADE_TUBE_H

#include "usdGenPomade/api.h"
#include <cmath>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define USDGEN_POMADE_HD __host__ __device__
#else
#define USDGEN_POMADE_HD
#endif

namespace usdGenPomade {

struct PomadeScalpMesh;
struct PomadeRegionLoops;

// Salt for the Pomade root/fill draw stream ("TonC").
constexpr uint32_t kPomadeSaltRoot = 0x546F6E43u;

// SplitMix64 finalizer over the salted key (the usdGenMath/hash.h spelling).
USDGEN_POMADE_HD inline uint64_t PomadeHash64(uint64_t key, uint32_t salt)
{
    uint64_t z =
        (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) +
        0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

USDGEN_POMADE_HD inline uint32_t PomadeHash32(uint64_t key, uint32_t salt)
{
    return uint32_t(PomadeHash64(key, salt) >> 32);
}

// Uniform float in [0,1) from the top 24 mantissa bits.
USDGEN_POMADE_HD inline float PomadeHash01(uint64_t key, uint32_t salt)
{
    return float(PomadeHash32(key, salt) >> 8) * 0x1.0p-24f;
}

// Piecewise-linear ramp over flattened (position, value) pairs with clamped
// ends. Empty pairs evaluate to 1. A single pair evaluates to its value.
USDGEN_POMADE_HD inline float PomadeEvalLengthProfile(float const *pairs,
                                                   int pairCount, float u)
{
    if (!pairs || pairCount <= 0) {
        return 1.0f;
    }
    if (pairCount == 1 || u <= pairs[0]) {
        return pairs[1];
    }
    for (int i = 1; i < pairCount; ++i) {
        float const p1 = pairs[size_t(i) * 2 + 0];
        float const v1 = pairs[size_t(i) * 2 + 1];
        if (u <= p1) {
            float const p0 = pairs[size_t(i - 1) * 2 + 0];
            float const v0 = pairs[size_t(i - 1) * 2 + 1];
            if (!(p1 > p0)) {
                return v1;
            }
            float const f = (u - p0) / (p1 - p0);
            return v0 + (v1 - v0) * f;
        }
    }
    return pairs[size_t(pairCount - 1) * 2 + 1];
}

// Soft-selection weight along the center parameter t in [0,1]: 1 at
// `center`, smooth to 0 at |t - center| == radius, 0 beyond. Mirrors
// pomadeMath.softSelection; the C++ gizmo path uses this spelling so the
// two cannot drift (compared in testUsdGenPomadeTubes).
USDGEN_POMADE_HD inline float PomadeSoftWeight(float t, float center,
                                             float radius)
{
    if (!(radius > 0.0f)) {
        return t == center ? 1.0f : 0.0f;
    }
    float w = 1.0f - fabsf((t - center) / radius);
    w = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
    return w * w * (3.0f - 2.0f * w);
}

// One rotation-minimising frame: tangent, normal, binormal (orthonormal).
struct PomadeFrame {
    float tx = 0.0f, ty = 1.0f, tz = 0.0f;
    float nx = 1.0f, ny = 0.0f, nz = 0.0f;
    float bx = 0.0f, by = 0.0f, bz = 1.0f;
};

// A proper world-space rigid transform. `rotation` is row-major and maps a
// column point as p' = R*p + translation.  Tube transport rejects reflection,
// scale and non-finite input rather than silently distorting groom geometry.
struct PomadeRigidTransform {
    float rotation[9] = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};
    float translation[3] = {0.0f, 0.0f, 0.0f};
};

// One cross-section: parameter t in [0,1], ring CVs in the section plane,
// a uniform scale and a twist (radians) applied before placement.
struct PomadeTubeSection {
    float t = 0.0f;
    std::vector<float> u;  // ringVerts entries
    std::vector<float> v;  // ringVerts entries
    float scale = 1.0f;
    float twist = 0.0f;
};

// A persistent K14 material binding for an L2 boundary sample.  It says that
// `childSlot` in this tube's `section` is the physical descendant of
// `parentSlot` in the parent section.  K14 writes one only when clipping
// actually retained the original parent corner; cut/intersection samples are
// deliberately absent.  The three integers are serialized as one int[]
// triple so old grooms can simply have an empty binding set.
struct PomadeParentBoundaryBinding {
    int section = -1;
    int parentSlot = -1;
    int childSlot = -1;
};

// The authored shape of one tube (plan/17 §2.1 Tube, P3 single-tube form).
struct PomadeTubeDesc {
    std::vector<float> centerX;  // center CVs, size nCv >= 2
    std::vector<float> centerY;
    std::vector<float> centerZ;
    std::vector<PomadeTubeSection> sections;  // size nSec >= 2, t ascending
    int ringVerts = 8;  // uniform per tube, in [3, 32]
    int tubeId = 0;
    int regionId = 0;
    int level = 1;
    // Region-rooted tubes pin only their first frame to the support plane.
    // K4 still derives every later frame from the authored center column, so
    // bending CV 1 changes the shaft without peeling the root ring off its
    // region.  Ordinary and legacy tubes leave this false and use pure K4.
    bool rootFramePinned = false;
    PomadeFrame rootFrame;
    // Derived from the bound growth surface, never serialized. The planar
    // authoring chart is retained, while its displayed root boundary follows
    // the scalp in 3D. Recomputed after hydrate, edits and hierarchy splits.
    std::vector<float> rootSurfaceOffsets;  // ringVerts world-space XYZ deltas
    // Proper local-to-world K4 material-frame reference.  Identity preserves
    // the historic world-axis bootstrap bit-for-bit.  A rigid region move
    // composes its rotation on the left, so K4's local chart and every
    // interpolated section retain their world-space shape.
    std::array<float, 9> frameReference = {{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f}};
    // P4 hierarchy links (plan/17 §2.3): the parent this tube subdivided
    // from and the index within that subdivision. -1 = no parent (L1 root
    // or un-subdivided tube).
    int parentTubeId = -1;
    int childIndex = -1;
    // K14-owned links to inherited parent boundary corners.  These are
    // material identity, not a nearest-point relationship: K7 uses them to
    // conform only the L1 edge held by an edited L2, leaving internal cuts
    // and unrelated parent slots untouched.
    std::vector<PomadeParentBoundaryBinding> inheritedBoundaryBindings;
};

// The first/last constructed rows are holding edges close to the end rings.
// They do not add controls or alter the exact authored-row stride.
USDGEN_POMADE_HD inline float PomadeDisplaySpanFraction(
    int row, int segments, int span, int spanCount)
{
    float const f = float(row) / float(segments);
    if (segments < 3) return f;
    if (span == 0 && row == 1) return 0.25f / float(segments);
    if (span == spanCount - 1 && row == segments - 1)
        return 1.0f - 0.25f / float(segments);
    return f;
}

USDGEN_POMADE_HD inline float PomadeRootAttachmentWeight(float t, float end)
{
    if (!(end > 0.0f) || t >= end) return 0.0f;
    float const f = t > 0.0f ? t / end : 0.0f;
    return 1.0f - f * f * (3.0f - 2.0f * f);
}

// Fill parameters (plan/17 §2.1 FillParams, P3 form with edgeBias).
enum class PomadeGuideSampler : uint8_t {
    Legacy = 0,
    RegionV3 = 1,
};

struct PomadeFillDesc {
    float density = 100.0f;  // expected guides per unit root area
    int cvCount = 8;         // CVs per guide, in [2, 64]
    int seed = 0;
    // Radial remap exponent control in [-1, 1]: r' = r^(1 - 0.5*bias), so
    // +1 pushes roots toward the tube wall, -1 pulls them to the center.
    float edgeBias = 0.0f;
    std::vector<float> lengthProfile;  // flattened (pos, value) pairs
    PomadeGuideSampler sampler = PomadeGuideSampler::RegionV3;
};

// One sampled guide root (K8 output, K9 input).
struct PomadeGuideRoot {
    int faceId = -1;  // -1 = disc root (no scalp)
    float u = 0.0f;
    float v = 0.0f;
    float px = 0.0f, py = 0.0f, pz = 0.0f;  // world position
    float ru = 0.0f, rv = 0.0f;  // normalised root-plane coordinate
};

// A transient K9 material binding. It names one deterministic triangle in
// the actual placed section-0 polygon and its barycentric point. Root caches
// remain ABI-compatible world roots; a fill rebuilds this binding from their
// current world positions, so frozen roots and old committed grooms need no
// migration. The same bytes upload to the optional K9 CUDA lane.
struct PomadeGuideMaterialBinding {
    int slot0 = -1, slot1 = -1, slot2 = -1;
    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
    // The shape-relative root radius after edge bias. It drives only the
    // length profile; the three weights carry the complete K5 footprint.
    float edgeRadius = 0.0f;
    // Physical root minus its un-biased section-0 cage point. K9 fades this
    // correction by (1 - t) so frozen/off-chart roots stay attached without
    // changing a full-length terminal ring.
    float rootDx = 0.0f, rootDy = 0.0f, rootDz = 0.0f;
};

// One ancestor partition a descendant root must satisfy. `childCenters` is
// ordered by childIndex; the sample survives only when its owning cell is
// `childIndex`. A chain is walked root-to-leaf so subface support and K14
// hierarchy ownership use one deterministic rule in preview and commit.
struct PomadeRootOwnershipCell {
    float rootCenter[3] = {0.0f, 0.0f, 0.0f};
    PomadeFrame frame;
    std::vector<float> childCenters;
    int childIndex = -1;
};

#ifndef __CUDA_ARCH__

// -- K4: rotation-minimising center frames (double reflection) -------------
USDGENPOMADE_API bool PomadeCenterFramesCpu(float const *cx, float const *cy, float const *cz,
                          int nCv, std::vector<PomadeFrame> *frames,
                          std::string *err);

// Descriptor-aware frame path. Identity frameReference preserves raw K4;
// a transported descriptor evaluates K4 in its material frame then maps it
// to world. A region-rooted descriptor still replaces frame zero with its
// stored support-plane frame. All display, guide, pick and commit paths that
// start from a PomadeTubeDesc must use this rather than raw K4 directly.
USDGENPOMADE_API bool PomadeTubeFramesCpu(PomadeTubeDesc const &tube,
                        std::vector<PomadeFrame> *frames, std::string *err);

// Transport one authored tube with a proper rigid world transform.  The
// center column and a pinned root frame are transformed directly. K4's
// local frame reference composes with the transform, so all section UV,
// scale and twist data stays authored and the complete K5 shape is rigid.
// It preserves topology and supports `out == &source`.
USDGENPOMADE_API bool PomadeRigidTransformTubeCpu(PomadeTubeDesc const &source,
                                PomadeRigidTransform const &transform,
                                PomadeTubeDesc *out, std::string *err);

// Evaluate the center curve (uniform Catmull-Rom, clamped ends) and its
// frame (normalised-lerp of the K4 frames) at t in [0,1].
USDGENPOMADE_API bool PomadeSampleCenterCpu(PomadeTubeDesc const &tube,
                          std::vector<PomadeFrame> const &frames, float t,
                          float *px, float *py, float *pz, PomadeFrame *frame,
                          std::string *err);

// Evaluate one exact K5 cross-section at t. The output is ringVerts world
// positions in authored slot order, including Hermite U/V, scale and twist.
// Guide material sampling and sparse output rails share this authoring chart.
// Display tessellation additionally applies surface attachment and end holding
// rows; those corrections do not change the stored chart or guide bindings.
USDGENPOMADE_API bool PomadeSampleTubeRingCpu(PomadeTubeDesc const &tube,
                            std::vector<PomadeFrame> const &frames, float t,
                            std::vector<float> *positions,
                            std::string *err);

// Deterministic nondegenerate ear triangulation in authored slot order. It
// retains every boundary slot, including collinear edge splits, so sparse
// rails and K9 material binds see the same section topology.
USDGENPOMADE_API bool PomadeTriangulateSectionSlotsCpu(
    PomadeTubeSection const &section,
    std::vector<std::array<int, 3>> *triangles, std::string *err);

// The editable/displayed location of a center CV.  It evaluates the actual
// section polygon at that CV's center parameter and returns its signed-area
// centroid in world space.  The authored center column remains the source of
// truth for K4/K6; consumers use this handle only for display, picking and
// manipulator placement so a clipped or asymmetric tube is controlled from
// inside its visible core rather than from an outer cage edge.
USDGENPOMADE_API bool PomadeCenterHandlePointCpu(PomadeTubeDesc const &tube, int centerCV,
                               float *px, float *py, float *pz,
                               std::string *err);
// Batch callers reuse frames from this exact descriptor instead of rebuilding
// the complete transported frame column for every center handle.
USDGENPOMADE_API bool PomadeCenterHandlePointCpu(PomadeTubeDesc const &tube,
                               std::vector<PomadeFrame> const &frames,
                               int centerCV, float *px, float *py, float *pz,
                               std::string *err);

// Mean ring radius of one section (used to normalise root coordinates).
USDGENPOMADE_API float PomadeSectionMeanRadius(PomadeTubeSection const &section);

// -- K5: Hermite tube tessellation ------------------------------------------
// Output grid: ((nSec - 1) * segmentsPerSpan + 1) rings x ringVerts verts,
// ring-major. Sections interpolate along t by cubic Hermite of the section
// CV vectors (central differences inside, one-sided at the ends — the
// PomadeEvalCenter rule); scale and twist take the same path as scalars.
USDGENPOMADE_API bool PomadeTessellateCpu(PomadeTubeDesc const &tube,
                        std::vector<PomadeFrame> const &frames,
                        int segmentsPerSpan, std::vector<float> *positions,
                        std::vector<float> *normals,
                        std::vector<float> *ringT, std::string *err);
USDGENPOMADE_API int PomadeTessellatedRingCount(PomadeTubeDesc const &tube, int segmentsPerSpan);

// -- K8: blue-noise root sampling -------------------------------------------
// Deterministic per (tubeId, seed). Disc roots scatter in the unit disc and
// are scaled by `radius` into the root section plane (center + frame0).
// Mesh roots scatter per face with area-weighted counts. `frozen` roots are
// kept verbatim (the Fill-mode "freeze roots" path); only the tail is
// (re-)sampled, continuing the same dart-throwing stream.
USDGENPOMADE_API bool PomadeRootSampleDiscCpu(int tubeId, int seed, float radius,
                            float const center[3], PomadeFrame const &frame,
                            int count, std::vector<PomadeGuideRoot> *roots,
                            int frozen, std::string *err);
USDGENPOMADE_API bool PomadeRootSampleMeshCpu(float const *points, int const *faceCounts,
                            int const *faceIndices, int const *faceOffsets,
                            int faceCount, int const *regionFaces,
                            int regionFaceCount, float density, int tubeId,
                            int seed, float const rootCenter[3],
                            PomadeFrame const &rootFrame, float rootRadius,
                            std::vector<PomadeGuideRoot> *roots, int frozen,
                            std::string *err);
// Exact graph-region form of K8.  Unlike the legacy face-list form above,
// this tests every emitted root against the closed graph polygon, so two
// disjoint regions on one coarse mesh face keep separate root support.  The
// source region id (not its interpolation id) selects the loop.
USDGENPOMADE_API bool PomadeRootSampleRegionMeshCpu(
    PomadeScalpMesh const &scalp, PomadeRegionLoops const &loops,
    int sourceRegionId, float density, int tubeId, int seed,
    float const rootCenter[3], PomadeFrame const &rootFrame, float rootRadius,
    std::vector<PomadeGuideRoot> *roots, int frozen, std::string *err);
USDGENPOMADE_API void PomadeFilterGuideRootsByOwnershipCells(
    std::vector<PomadeRootOwnershipCell> const &cells,
    std::vector<PomadeGuideRoot> *roots);

// -- K9: guide fill ----------------------------------------------------------
// A root binds to a deterministic triangle of the placed section-0 polygon.
// Its barycentric material point rides through the complete K5 section
// interpolation, including each CV, scale and twist; edgeBias remaps that
// point relative to the polygon's material anchor. Output is guideCount x
// fill.cvCount x 3, guide-major.
USDGENPOMADE_API bool PomadeBuildGuideMaterialBindingsCpu(
    PomadeTubeDesc const &tube, std::vector<PomadeFrame> const &frames,
    std::vector<PomadeGuideRoot> const &roots, PomadeFillDesc const &fill,
    std::vector<PomadeGuideMaterialBinding> *bindings, std::string *err);
USDGENPOMADE_API bool PomadeGuideFillCpu(PomadeTubeDesc const &tube,
                       std::vector<PomadeFrame> const &frames,
                       std::vector<PomadeGuideRoot> const &roots,
                       PomadeFillDesc const &fill, std::vector<float> *points,
                       std::vector<float> *lengthScales, std::string *err);

// -- K10: arc-length resample + root frames ----------------------------------
// Resamples every guide to `cvCount` CVs; emits one UsdGenCurveAPI root
// frame per guide (16 doubles, row-major: rows 0/1/2 are the tube root
// N/B/T directions, row 3 is the root position).
USDGENPOMADE_API bool PomadeGuideResampleCpu(float const *points, int const *counts,
                           int guideCount, int cvCount,
                           float const rootDirs[9], std::vector<float> *out,
                           std::vector<int> *outCounts,
                           std::vector<double> *outFrames, std::string *err);

#endif  // !__CUDA_ARCH__

// -- K11: screen-space pick with a kind mask ---------------------------------
// Reuses the gpu/picking.cu convention (row-major USD viewProj, row-vector,
// top-left pixel origin). Among candidates within `radiusPx`, the nearest
// in pixels wins; depth (NDC z) breaks ties toward the viewer, which is
// the documented stand-in for last-frame occlusion until the HdxPickTask
// fallback (plan/17 §4.6) is needed.
// The nine kinds of plan/17 §4.6, in one enum: it is both the pick mask and
// the selection kind (plan/18 §2.3), so a hit and the item it selects are
// spelled the same. Eight are screen-pickable; a hierarchy LEVEL has no
// screen candidate of its own (it is chosen from the breadcrumb, and
// selecting it means every tube at that level), so PomadePick_All — what a
// press asks for — stops at the eight, and PomadeSelect_All is the selection
// spelling that includes the level.
enum PomadePickKind : uint32_t {
    PomadePick_TubeVert = 1u << 0,
    PomadePick_CenterCV = 1u << 1,
    PomadePick_SectionCV = 1u << 2,
    PomadePick_GraphNode = 1u << 3,
    PomadePick_Guide = 1u << 4,
    PomadePick_GraphEdge = 1u << 5,
    PomadePick_Region = 1u << 6,
    PomadePick_SectionRing = 1u << 7,
    PomadePick_Level = 1u << 8,
    PomadePick_All = 0xFFu,
    PomadeSelect_All = 0x1FFu
};

struct PomadePickHit {
    bool hit = false;
    uint32_t kind = 0;  // one PomadePickKind bit
    int index = -1;     // candidate index within its kind
    int subIndex = -1;  // CV within a guide, slot within a ring, else -1
    float distPx = 0.0f;
    float depth = 0.0f;  // NDC z in [-1, 1]
};

// Project one point; returns false behind the camera or at w == 0.
USDGEN_POMADE_HD inline bool PomadeProjectPoint(float const p[3],
                                             float const viewProj[16], int w,
                                             int h, float *px, float *py,
                                             float *ndcZ)
{
    float const x = p[0], y = p[1], z = p[2];
    float const cx =
        x * viewProj[0] + y * viewProj[4] + z * viewProj[8] + viewProj[12];
    float const cy =
        x * viewProj[1] + y * viewProj[5] + z * viewProj[9] + viewProj[13];
    float const cz =
        x * viewProj[2] + y * viewProj[6] + z * viewProj[10] + viewProj[14];
    float const cw =
        x * viewProj[3] + y * viewProj[7] + z * viewProj[11] + viewProj[15];
    if (cw <= 0.0f) {
        return false;
    }
    float const inv = 1.0f / cw;
    float const nx = cx * inv, ny = cy * inv;
    *px = (nx * 0.5f + 0.5f) * float(w);
    *py = (1.0f - (ny * 0.5f + 0.5f)) * float(h);
    *ndcZ = cz * inv;
    return true;
}

// Marquee test: the box is normalised, so a drag in any direction selects
// the same candidates, and the edges are inclusive (a candidate exactly on
// the border is in).
USDGEN_POMADE_HD inline bool PomadePointInRect(float px, float py, float x0,
                                             float y0, float x1, float y1)
{
    float const lo0 = x0 < x1 ? x0 : x1;
    float const hi0 = x0 < x1 ? x1 : x0;
    float const lo1 = y0 < y1 ? y0 : y1;
    float const hi1 = y0 < y1 ? y1 : y0;
    return px >= lo0 && px <= hi0 && py >= lo1 && py <= hi1;
}

// Lasso test: crossing number over a closed polygon (`xy` holds n (x, y)
// pairs; the last vertex joins the first implicitly). The half-open rule on
// the y span counts a vertex exactly once, so a candidate on a horizontal
// run of the lasso is not double-counted. Same spelling on both lanes, so
// the CPU twin and the device mask agree candidate for candidate.
USDGEN_POMADE_HD inline bool PomadePointInPolygon(float px, float py,
                                                float const *xy, int n)
{
    if (!xy || n < 3) {
        return false;
    }
    bool inside = false;
    int j = n - 1;
    for (int i = 0; i < n; j = i++) {
        float const xi = xy[i * 2 + 0], yi = xy[i * 2 + 1];
        float const xj = xy[j * 2 + 0], yj = xy[j * 2 + 1];
        if ((yi > py) != (yj > py)) {
            float const t = (py - yi) / (yj - yi);
            if (px < xi + t * (xj - xi)) {
                inside = !inside;
            }
        }
    }
    return inside;
}

#ifndef __CUDA_ARCH__

// One candidate set per pickable kind; null data means the kind is absent.
// Guide candidates are per-CV (guideCount * guideCv points); the hit
// reports the guide in `index` and the CV in `subIndex`.
struct PomadePickSets {
    float const *tubeVerts = nullptr;  // 3 * tubeVertCount
    int tubeVertCount = 0;
    float const *centerCVs = nullptr;  // 3 * centerCVCount
    int centerCVCount = 0;
    float const *sectionCVs = nullptr;  // 3 * sectionCVCount (ring-major)
    int sectionCVCount = 0;
    int sectionRingVerts = 0;  // > 0 splits subIndex into (ring, slot)
    float const *graphNodes = nullptr;  // 3 * graphNodeCount
    int graphNodeCount = 0;
    float const *guideCVs = nullptr;  // 3 * guideCount * guideCvCount
    int guideCount = 0;
    int guideCvCount = 0;
    // The three candidate sets V1 adds (plan/18 §2.3). An edge is many
    // candidates — every point of its K2 polyline, with the edge id
    // alongside — so a click anywhere along it hits; a region and a
    // section ring are one representative point each (their centroid),
    // which is also the right rule for a marquee: the ring is caught when
    // its centre is in the box, not when one of its verts clips a corner.
    float const *graphEdgeCVs = nullptr;  // 3 * graphEdgeCVCount
    int const *graphEdgeIds = nullptr;    // graphEdgeCVCount (edge id/point)
    int graphEdgeCVCount = 0;
    float const *regionCenters = nullptr;  // 3 * regionCount
    int const *regionIds = nullptr;        // regionCount
    int regionCount = 0;
    float const *ringCenters = nullptr;  // 3 * ringCount (tube 0's rings)
    int ringCount = 0;
};

// One candidate the marquee/lasso caught, in the same (kind, index,
// subIndex) spelling PomadePickHit reports, so the point pick and the region
// pick hand their caller identical records and can never disagree about
// what a candidate means.
struct PomadePickCandidate {
    uint32_t kind = 0;
    int index = -1;
    int subIndex = -1;
};

USDGENPOMADE_API PomadePickHit PomadePickCpu(PomadePickSets const &sets, uint32_t kindMask,
                          float const viewProj[16], int w, int h, float x,
                          float y, float radiusPx);

#endif  // !__CUDA_ARCH__

// -- Shared CPU/GPU vector spelling (single source for TN-6 parity) --------
// The CPU twins and the CUDA kernels below both spell through these
// inlines, so arithmetic-only paths are bit-identical by construction
// (the T0 parity test still proves it; transcendentals take tolerance).
USDGEN_POMADE_HD inline void PomadeSub3(float const a[3], float const b[3],
                                      float out[3])
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

USDGEN_POMADE_HD inline float PomadeDot3(float const a[3], float const b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

USDGEN_POMADE_HD inline void PomadeCross3(float const a[3], float const b[3],
                                        float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

USDGEN_POMADE_HD inline float PomadeLen3(float const a[3])
{
    return sqrtf(PomadeDot3(a, a));
}

USDGEN_POMADE_HD inline float PomadeHermite(float a, float b, float m0,
                                          float m1, float u)
{
    float const u2 = u * u, u3 = u2 * u;
    return (2.0f * u3 - 3.0f * u2 + 1.0f) * a +
           (u3 - 2.0f * u2 + u) * m0 + (-2.0f * u3 + 3.0f * u2) * b +
           (u3 - u2) * m1;
}

// Catmull-Rom tangent at CV i (clamped ends), normalised. Coincident
// neighbours fall back to +Y.
USDGEN_POMADE_HD inline void PomadeCenterTangent(float const *cx,
                                               float const *cy,
                                               float const *cz, int n, int i,
                                               float out[3])
{
    int const i0 = i > 0 ? i - 1 : i;
    int const i1 = i + 1 < n ? i + 1 : i;
    float d[3] = {cx[i1] - cx[i0], cy[i1] - cy[i0], cz[i1] - cz[i0]};
    float const len = PomadeLen3(d);
    if (len > 1e-12f) {
        out[0] = d[0] / len;
        out[1] = d[1] / len;
        out[2] = d[2] / len;
        return;
    }
    out[0] = 0.0f;
    out[1] = 1.0f;
    out[2] = 0.0f;
}

USDGEN_POMADE_HD inline void PomadePerp3(float const t[3], float out[3])
{
    float ax = fabsf(t[0]), ay = fabsf(t[1]), az = fabsf(t[2]);
    float h[3] = {1.0f, 0.0f, 0.0f};
    if (ay <= ax && ay <= az) {
        h[0] = 0.0f;
        h[1] = 1.0f;
        h[2] = 0.0f;
    } else if (az <= ax && az <= ay) {
        h[0] = 0.0f;
        h[1] = 0.0f;
        h[2] = 1.0f;
    }
    PomadeCross3(t, h, out);
    float const len = PomadeLen3(out);
    if (len > 1e-12f) {
        out[0] /= len;
        out[1] /= len;
        out[2] /= len;
    } else {
        out[0] = 1.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
    }
}

// One double-reflection propagation step (Wang et al. 2008): reflect the
// normal in the chord bisector, then in the tangent bisector. Coincident
// points keep the incoming normal; parallel tangents skip the second
// reflection.
USDGEN_POMADE_HD inline void PomadeReflectNormal(float const p0[3],
                                               float const p1[3],
                                               float const tPrev[3],
                                               float const tCur[3], float n[3])
{
    float v1[3];
    PomadeSub3(p1, p0, v1);
    float const c1 = PomadeDot3(v1, v1);
    if (!(c1 > 1e-24f)) {
        return;
    }
    float const s1 = 2.0f / c1;
    float const d1n = PomadeDot3(v1, n);
    float r[3] = {n[0] - v1[0] * s1 * d1n, n[1] - v1[1] * s1 * d1n,
                  n[2] - v1[2] * s1 * d1n};
    float const d1t = PomadeDot3(v1, tPrev);
    float tt[3] = {tPrev[0] - v1[0] * s1 * d1t,
                   tPrev[1] - v1[1] * s1 * d1t,
                   tPrev[2] - v1[2] * s1 * d1t};
    float v2[3];
    PomadeSub3(tCur, tt, v2);
    float const c2 = PomadeDot3(v2, v2);
    if (c2 > 1e-24f) {
        float const s2 = 2.0f / c2;
        float const d = PomadeDot3(v2, r);
        n[0] = r[0] - v2[0] * s2 * d;
        n[1] = r[1] - v2[1] * s2 * d;
        n[2] = r[2] - v2[2] * s2 * d;
    } else {
        n[0] = r[0];
        n[1] = r[1];
        n[2] = r[2];
    }
    float const nl = PomadeLen3(n);
    if (nl > 1e-12f) {
        n[0] /= nl;
        n[1] /= nl;
        n[2] /= nl;
    } else {
        float t[3] = {tCur[0], tCur[1], tCur[2]};
        PomadePerp3(t, n);
    }
}

// Uniform Catmull-Rom center evaluation in Hermite form: central
// differences inside, one-sided differences at the ends (so straight
// lines reproduce exactly in every span, ends included).
USDGEN_POMADE_HD inline void PomadeEvalCenter(float const *cx, float const *cy,
                                            float const *cz, int n, float t,
                                            float out[3])
{
    float const tc = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    float const seg = tc * float(n - 1);
    int i1 = int(seg);
    if (i1 >= n - 1) {
        i1 = n - 2;
    }
    int const i0 = i1 > 0 ? i1 - 1 : i1;
    int const i2 = i1 + 1;
    int const i3 = i1 + 2 < n ? i1 + 2 : n - 1;
    float const f = seg - float(i1);
    bool const first = (i1 == 0);
    bool const last = (i2 == n - 1);
    float const m0x = first ? cx[i2] - cx[i1] : 0.5f * (cx[i2] - cx[i0]);
    float const m1x = last ? cx[i2] - cx[i1] : 0.5f * (cx[i3] - cx[i1]);
    float const m0y = first ? cy[i2] - cy[i1] : 0.5f * (cy[i2] - cy[i0]);
    float const m1y = last ? cy[i2] - cy[i1] : 0.5f * (cy[i3] - cy[i1]);
    float const m0z = first ? cz[i2] - cz[i1] : 0.5f * (cz[i2] - cz[i0]);
    float const m1z = last ? cz[i2] - cz[i1] : 0.5f * (cz[i3] - cz[i1]);
    out[0] = PomadeHermite(cx[i1], cx[i2], m0x, m1x, f);
    out[1] = PomadeHermite(cy[i1], cy[i2], m0y, m1y, f);
    out[2] = PomadeHermite(cz[i1], cz[i2], m0z, m1z, f);
}

// Normalised-lerp of the bracketing K4 frames at t in [0,1],
// re-orthogonalised (B = T x N). The bracket matches PomadeEvalCenter.
USDGEN_POMADE_HD inline void PomadeNlerpFrame(PomadeFrame const *frames, int n,
                                            float t, PomadeFrame *out)
{
    float const tc = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    float const seg = tc * float(n - 1);
    int i1 = int(seg);
    if (i1 >= n - 1) {
        i1 = n - 2;
    }
    int const i2 = i1 + 1;
    float const f = seg - float(i1);
    PomadeFrame const &a = frames[i1];
    PomadeFrame const &b = frames[i2];
    float tt[3] = {a.tx + (b.tx - a.tx) * f, a.ty + (b.ty - a.ty) * f,
                   a.tz + (b.tz - a.tz) * f};
    float nn[3] = {a.nx + (b.nx - a.nx) * f, a.ny + (b.ny - a.ny) * f,
                   a.nz + (b.nz - a.nz) * f};
    float tl = PomadeLen3(tt);
    if (!(tl > 1e-12f)) {
        tt[0] = a.tx;
        tt[1] = a.ty;
        tt[2] = a.tz;
        tl = PomadeLen3(tt);
    }
    tt[0] /= tl;
    tt[1] /= tl;
    tt[2] /= tl;
    float const dn = PomadeDot3(nn, tt);
    nn[0] -= dn * tt[0];
    nn[1] -= dn * tt[1];
    nn[2] -= dn * tt[2];
    float nl = PomadeLen3(nn);
    if (!(nl > 1e-12f)) {
        PomadePerp3(tt, nn);
        nl = 1.0f;
    }
    nn[0] /= nl;
    nn[1] /= nl;
    nn[2] /= nl;
    float bb[3];
    PomadeCross3(tt, nn, bb);
    out->tx = tt[0];
    out->ty = tt[1];
    out->tz = tt[2];
    out->nx = nn[0];
    out->ny = nn[1];
    out->nz = nn[2];
    out->bx = bb[0];
    out->by = bb[1];
    out->bz = bb[2];
}

#undef USDGEN_POMADE_HD

}  // namespace usdGenPomade

#endif  // USDGEN_POMADE_TUBE_H
