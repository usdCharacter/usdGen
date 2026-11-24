// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — reference guide blending for UsdGenGuideInterpolate's
// nearest / inverseDistance / gaussian weightings (the imported-groom path).
//
// Everything here reproduces the reference groomer's root-strand stage
// operation for operation, in single precision and in the reference's
// evaluation order (the float sums are written out so the compiler cannot
// reassociate them), so an imported groom cooks to the same strands:
//
//   * guide tables: every eligible guide (>= 2 points, length > 1e-6) is
//     resampled to ng = clamp(cvCount, 8, 1000) samples uniform in arc
//     length; per guide the initial direction, the parting query at its
//     root, and (bendFollow) rotation-minimising frames from Catmull-Rom
//     tangents of the raw guide;
//   * gaussian: the PCA plane of the guide roots, a Delaunay triangulation
//     of the projected roots, sigma^2 = (max(1e-6, 0.5 * mean edge))^2;
//   * per strand: pre-assigned guide (root on a guide root), kernel weights
//     over all guides, else the 3 nearest guides by parting-scaled distance
//     (barycentric / inverse distance / single guide), the strand built at
//     cvCount points, and the parting bend (rotation about the root).
//
// Strands are plain xyz arrays (3 * cvCount floats). Indices returned for a
// guide are TABLE indices (eligible guides); Guides::curve maps them to the
// guide set's curve index.
#ifndef USDGEN_OPS_GUIDE_BLEND_H
#define USDGEN_OPS_GUIDE_BLEND_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace usdGen {
namespace guideBlend {

/// Single-precision vector with the reference's component order.
struct V3
{
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

enum class Weighting { Nearest, InverseDistance, Gaussian };

/// Parting curves (polylines) and the parting controls.
struct Parting
{
    std::vector<V3> points;           // all curves, concatenated
    std::vector<uint32_t> offsets;    // curves + 1
    std::vector<float> lengths;       // polyline length per curve
    bool active = false;              // amount > 1e-4, angle > 1e-4, curves exist
    float amount = 0.0f;              // A = clamp(amount, 0, 1)
    float radius = 0.0f;              // R = clamp(radius, 0, 1): fraction of the curve length
    float tanAngle = 0.0f;            // tan(clamp(angle, 0, 89) deg)
    float distance = 1.0f;            // D = clamp(distance, 0, 1)

    /// Fills offsets/lengths from points + per-curve counts and the
    /// controls from the authored values (angle in degrees).
    void Set(std::vector<V3> pts, std::vector<int> const &counts, float amount_, float angle_,
             float radius_, float distance_);
    size_t CurveCount() const { return offsets.empty() ? 0 : offsets.size() - 1; }
};

/// The parting query at a point with a surface normal.
struct PartInfo
{
    bool valid = false;
    V3 closest;                       // closest point on the nearest parting curve
    V3 dir{0.0f, 1.0f, 0.0f};         // that segment's direction
    V3 away;                          // unit direction away from the curve, on the point's side
    float dist2 = 3.4028235e+38f;
    float curveLength = 0.0f;
    float weight = 0.0f;              // smoothstep(1 - d / (R Lc)) * A
    float offLine = 0.0f;             // 1 when the point is off the curve's line
    int curve = -1;
    float side = 0.0f;                // +1 / -1 / 0 (within 1e-5)
};

PartInfo QueryParting(Parting const &parting, V3 const &x, V3 const &normal);

/// Parting key: curve * 2 + (side <= 0), -1 for no parting / on the line.
int PartKey(PartInfo const &info);

/// The guide tables built once per capture.
struct Guides
{
    std::vector<uint32_t> curve;      // table index -> guide set curve index
    std::vector<uint32_t> rawOffset;  // table index + 1 -> raw point ranges
    std::vector<V3> raw;              // raw guide points
    std::vector<V3> root;             // guide roots (first point)
    std::vector<V3> rootNormal;       // guide root normals
    std::vector<float> length;        // polyline length
    int ng = 8;                       // samples per resampled guide
    std::vector<V3> table;            // count * ng resampled samples
    std::vector<V3> initialDir;       // per guide initial direction (parting bend)
    std::vector<PartInfo> part;       // per guide parting query (parting active only)
    bool frames = false;              // bendFollow: rotation-minimising frames
    std::vector<V3> frameT, frameN, frameB;   // count * ng
    // Gaussian kernel tables.
    bool plane = false;
    V3 origin, axisU, axisV;
    std::vector<float> planar;        // 2 * count projected roots
    float sigma2 = 0.0f;

    size_t Count() const { return root.size(); }
};

/// Eligible guides from a curve set: `points` (xyz per CV), `spans`
/// (curves + 1), and per curve root normals (`normals`, one per curve).
/// cvCount is the strand point count.
void BuildGuides(float const *px, float const *py, float const *pz,
                 std::vector<uint32_t> const &spans,
                 std::vector<V3> const &normals, int cvCount,
                 Weighting weighting, bool bendFollow, Parting const &parting,
                 Guides *out);

/// Per strand settings.
struct StrandSettings
{
    Weighting weighting = Weighting::Nearest;
    float tightness = 3.0f;
    float defaultLength = 0.01f;
    int cvCount = 8;
    bool bendFollow = false;
};

/// What the strand stage reports per strand.
struct StrandResult
{
    int primary = -1;                 // dominant guide (table index), -1 none
    int nearest = -1;                 // parting-aware nearest guide (table index)
    int partId = -1;                  // parting key (weight > 0.2) or -1
    int index[3] = {-1, -1, -1};      // the heaviest blend guides (table index)
    float weight[3] = {0.0f, 0.0f, 0.0f};
    float length = 0.0f;              // strand length (0 when dropped)
    bool dropped = false;             // length <= 1e-5: zero-length strand
};

/// Scratch reused across strands by one thread.
struct Scratch
{
    std::vector<float> weights;
};

/// Builds one strand rooted at `root` with surface normal `normal` into
/// `out` (3 * cvCount floats). Deterministic and thread-safe on const
/// tables.
void BuildStrand(Guides const &guides, Parting const &parting, StrandSettings const &settings,
                 V3 const &root, V3 const &normal, float *out, StrandResult *result,
                 Scratch *scratch);

/// The parting-aware nearest guide of a root (nearest-guide rule of the
/// clump stage): -1 without guides.
int NearestGuide(Guides const &guides, Parting const &parting, PartInfo const &rootPart,
                 V3 const &root);

// --- exposed for tests ------------------------------------------------------

/// Point at arc length s on a raw polyline (linear; extrapolates past the tip).
V3 SampleRaw(V3 const *points, size_t count, float s);
/// Polyline length, sequential float sum.
float PolylineLength(V3 const *points, size_t count);
/// 2-D Delaunay triangulation (Bowyer-Watson), triangles as index triples.
std::vector<int> Delaunay2D(std::vector<float> const &xy);

}  // namespace guideBlend
}  // namespace usdGen

#endif  // USDGEN_OPS_GUIDE_BLEND_H
