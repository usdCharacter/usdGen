// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — strand math shared by the whole-strand stylers (Clump, Braid,
// Twist, Length, Trim, Frizz, TieClump, Collide relax, CardSource) and the
// scatter / guide-blend generators.
//
// Everything here is deterministic, allocation-light, and independent of
// thread order, so a capture that calls these helpers from a parallel loop
// produces the same bytes on every run. Strands are plain xyz polylines:
// `float const *p` points at 3*n floats (x0 y0 z0 x1 ...).
#ifndef USDGEN_OPS_STRAND_UTIL_H
#define USDGEN_OPS_STRAND_UTIL_H

#include "usdGen/graphDesc.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace strand {

// ---------------------------------------------------------------------------
// Scalars, hashes and random numbers
// ---------------------------------------------------------------------------

inline float Clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

/// smoothstep on a value already mapped to [0,1]: x^2 (3 - 2x).
inline float SmoothStep01(float x)
{
    x = Clamp01(x);
    return (3.0f - 2.0f * x) * x * x;  // reference evaluation order
}

/// smoothstep((x - e0) / (e1 - e0)); 0 or 1 when the edge span is degenerate.
inline float SmoothStep(float e0, float e1, float x)
{
    float const span = e1 - e0;
    if (!(std::fabs(span) > 1e-12f)) return x < e0 ? 0.0f : 1.0f;
    return SmoothStep01((x - e0) / span);
}

/// splitmix64 finaliser (Steele et al.): z += golden; two xor-shift-multiply
/// rounds; final xor-shift.
inline uint64_t SplitMix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// 53-bit uniform double in [0,1) of a 64-bit value, narrowed to float the
/// same way the reference implementation narrows it.
inline float U01(uint64_t z)
{
    return static_cast<float>(static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0));
}

/// U01(SplitMix64(x)).
inline float Hash01(uint64_t x) { return U01(SplitMix64(x)); }

/// uniform01 from a std::mt19937 draw: 24 high bits.
inline float Mt01(std::mt19937 &rng)
{
    return static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f);
}

/// Uniform integer in [0, bound] from std::mt19937 with the rejection rule
/// of the MSVC standard library's `_Rng_from_urng` (used by std::shuffle):
/// draws x until it is not in the biased tail, returns x % (bound + 1).
/// Portable re-statement so every platform shuffles identically.
inline uint32_t MsvcUniformIndex(std::mt19937 &rng, uint32_t bound)
{
    uint64_t const range = uint64_t(bound) + 1u;
    if (range >= 0x100000000ull) return rng();
    uint32_t const r = static_cast<uint32_t>(range);
    uint32_t const limit = 0xFFFFFFFFu / r;
    uint32_t const rem = 0xFFFFFFFFu % r;
    uint32_t x;
    do {
        x = rng();
    } while (x / r >= limit && rem != r - 1u);
    return x % r;
}

/// Forward Fisher-Yates with MsvcUniformIndex: for i = 1..n-1 draw
/// k in [0, i] and swap a[i], a[k] when k != i (std::shuffle on MSVC).
template <class T>
void MsvcShuffle(std::vector<T> *a, std::mt19937 &rng)
{
    if (!a || a->size() < 2) return;
    for (size_t i = 1; i < a->size(); ++i) {
        uint32_t const k = MsvcUniformIndex(rng, static_cast<uint32_t>(i));
        if (k != i) std::swap((*a)[i], (*a)[k]);
    }
}

// ---------------------------------------------------------------------------
// Root / mid / tip profile
// ---------------------------------------------------------------------------

/// Quadratic Bezier profile through the control points (0, root),
/// (midShift, mid), (1, tip), evaluated at x = t in [0,1]: solve
/// x(u) = 2*midShift*u*(1-u) + u^2 = t for u, return y(u). `mid` is a
/// control value: the curve passes through it only when root, mid and tip
/// are collinear. midShift is clamped to [shiftMin, shiftMax]
/// (0.1..0.9 for clump/twist/thickness; frizz passes 0..1).
inline float BezierProfile(float t, float root, float mid, float midShift, float tip,
                           float shiftMin = 0.1f, float shiftMax = 0.9f)
{
    float const m = std::min(std::max(midShift, shiftMin), shiftMax);
    t = Clamp01(t);
    float const b = 2.0f * m;
    float const a = 1.0f - 2.0f * m;
    float u;
    if (std::fabs(a) <= 1e-6f) {
        u = b > 1e-6f ? Clamp01(t / b) : t;
    } else {
        float const d = std::sqrt(std::max(0.0f, b * b + 4.0f * a * t));
        float const u1 = (d - b) / (2.0f * a);
        float const u2 = (-b - d) / (2.0f * a);
        u = (u1 >= 0.0f && u1 <= 1.0f) ? u1 : ((u2 >= 0.0f && u2 <= 1.0f) ? u2 : Clamp01(u1));
    }
    float const v = 1.0f - u;
    return v * v * root + 2.0f * v * u * mid + u * u * tip;
}

// ---------------------------------------------------------------------------
// Value noise
// ---------------------------------------------------------------------------

/// 1-D value noise in [-1, 1]: lattice values 2*Hash01(j ^ key) - 1 at the
/// integers, blended with smoothstep weights; `sharpness` in [0,1] moves the
/// weight toward linear (1 = straight runs with sharp corners).
inline float ValueNoise1(float x, uint64_t key, float sharpness = 0.0f)
{
    float const fi = std::floor(x);
    float const f = x - fi;
    float w = (3.0f - 2.0f * f) * f * f;
    if (sharpness > 0.0f) w = (1.0f - sharpness) * w + sharpness * f;
    int64_t const i = static_cast<int64_t>(fi);
    float const a = 2.0f * Hash01(static_cast<uint64_t>(i) ^ key) - 1.0f;
    float const b = 2.0f * Hash01(static_cast<uint64_t>(i + 1) ^ key) - 1.0f;
    return a * (1.0f - w) + b * w;
}

// ---------------------------------------------------------------------------
// Vectors and polylines
// ---------------------------------------------------------------------------

inline GfVec3f Load(float const *p, size_t i) { return GfVec3f(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
inline void Store(float *p, size_t i, GfVec3f const &v)
{
    p[3 * i] = v[0]; p[3 * i + 1] = v[1]; p[3 * i + 2] = v[2];
}

inline GfVec3f Normalize(GfVec3f const &v, GfVec3f const &fallback, float eps = 1e-6f)
{
    float const l = v.GetLength();
    return l > eps ? v / l : fallback;
}

/// Rodrigues rotation of v about unit axis k by angle a (radians).
inline GfVec3f Rotate(GfVec3f const &v, GfVec3f const &k, float a)
{
    float const c = std::cos(a), s = std::sin(a);
    return v * c + GfCross(k, v) * s + k * (GfDot(k, v) * (1.0f - c));
}

/// Cumulative arc length: out[0] = 0, out[i] = sum of segment lengths to i.
/// Returns the total length.
float ArcLengths(float const *p, size_t n, std::vector<float> *out);

/// Point at arc length s along the polyline (linear between points). Past
/// the tip the last segment's direction is extrapolated by (s - L); before
/// the root the root is returned. A zero-length polyline returns the root.
GfVec3f SampleArc(float const *p, size_t n, std::vector<float> const &cum, float s);

/// Resample the polyline to m points uniformly in arc length over [s0, s1]
/// (s1 may exceed the length: extrapolates straight). dst has 3*m floats.
void ResampleArc(float const *p, size_t n, std::vector<float> const &cum,
                 float s0, float s1, size_t m, float *dst);

/// Rotation-minimising frames by the projection method: T[i] given (unit),
/// N0 = normalize(T0 x up) with up = +Y, or +X when |T0.y| >= 0.95;
/// N[i] = normalize(N[i-1] - (N[i-1].T[i]) T[i]); B[i] = T[i] x N[i].
void TransportFrames(std::vector<GfVec3f> const &T, std::vector<GfVec3f> *N,
                     std::vector<GfVec3f> *B);

/// Central-difference unit tangents of a polyline (forward/backward at the
/// ends); degenerate segments reuse the previous tangent, default +Y.
void PolylineTangents(float const *p, size_t n, std::vector<GfVec3f> *T);

/// Moves a strand so its root lands at root + delta while far CVs stay put:
/// p0 += delta; p_k += delta * (1 - smoothstep(s_k / F)) for k >= 1 until
/// s_k >= F, with s_k the ORIGINAL arc length and F = min(4|delta|, L/2).
void ResnapRoot(float *p, size_t n, GfVec3f const &delta);

// ---------------------------------------------------------------------------
// Surface: fan triangles and closest-point queries
// ---------------------------------------------------------------------------

/// One fan triangle of a polygon mesh: polygon `face`, corners
/// (f0, fk, fk+1) as POINT indices and as face-vertex (corner) indices.
struct FanTriangle
{
    int face = -1;
    int v[3] = {0, 0, 0};       // point indices
    int corner[3] = {0, 0, 0};  // face-vertex indices (faceVarying primvars)
};

/// Fan-triangulates every polygon in face order: polygon f with k corners
/// gives triangles (c0, cj, cj+1), j = 1..k-2. When `faces` is non-null only
/// those faces are emitted (sorted ascending, e.g. GeomSubset faces).
void FanTriangulate(VtIntArray const &counts, VtIntArray const &indices,
                    std::vector<FanTriangle> *out, VtIntArray const *faces = nullptr);

struct ClosestHit
{
    bool hit = false;
    int triangle = -1;          // index into the triangle list
    GfVec3f bary{1.0f, 0.0f, 0.0f};
    GfVec3f point{0.0f};
    float distanceSq = 0.0f;
};

/// Bounding-volume hierarchy over fan triangles for exact closest-point
/// queries (Ericson, "Real-Time Collision Detection" 5.1.5, double
/// precision). Deterministic: ties keep the lowest triangle index.
/// Also carries smooth per-corner normals and per-corner UVs so a hit can be
/// turned into a surface frame and a texture coordinate.
class TriangleBvh
{
public:
    /// `cornerNormals` / `cornerUvs`: 3 entries per triangle, or empty
    /// (normals then default to area-weighted vertex normals, UVs to 0).
    void Build(VtVec3fArray const &points, std::vector<FanTriangle> const &triangles,
               std::vector<GfVec3f> cornerNormals = {},
               std::vector<GfVec2f> cornerUvs = {});
    bool Empty() const { return _nodes.empty(); }
    size_t TriangleCount() const { return _tris.size(); }
    /// Closest point within sqrt(maxDistanceSq) (infinite by default).
    ClosestHit Closest(GfVec3f const &p, float maxDistanceSq = 3.0e38f) const;
    /// Unit geometric normal of a triangle (right-handed, (b-a)x(c-a)).
    GfVec3f FaceNormal(int triangle) const;
    /// Barycentric blend of the triangle's corner normals, normalised
    /// (face normal fallback).
    GfVec3f Normal(int triangle, GfVec3f const &bary) const;
    /// Barycentric blend of the triangle's corner UVs.
    GfVec2f Uv(int triangle, GfVec3f const &bary) const;
    /// Position from barycentrics.
    GfVec3f Position(int triangle, GfVec3f const &bary) const;
    std::vector<FanTriangle> const &Triangles() const { return _tris; }
    VtVec3fArray const &Points() const { return _points; }
private:
    struct Node { float lo[3], hi[3]; int left = -1, right = -1, first = 0, count = 0; };
    int BuildRange(int first, int count, std::vector<GfVec3f> const &centroids);
    std::vector<Node> _nodes;
    std::vector<int> _order;
    std::vector<FanTriangle> _tris;
    std::vector<GfVec3f> _normals;   // 3 per triangle
    std::vector<GfVec2f> _uvs;       // 3 per triangle, or empty
    VtVec3fArray _points;
};

/// Builds the closest-point structure for a surface's REST (Default-time)
/// or CURRENT points, honouring a GeomSubset's face list. Corner normals come
/// from the surface's rest normals when `rest` and they are vertex or
/// faceVarying, else area-weighted vertex normals of the chosen points.
/// Corner UVs come from `surface.uv` when it is vertex- (points) or
/// faceVarying- (face-vertex) sized.
void BuildSurfaceBvh(UsdGenSurfaceDesc const &surface, bool rest, TriangleBvh *bvh);

/// Samples a mask map at a point on the surface the BVH was built from:
///   UsdGenPaintMap -> barycentric blend of its faceVarying `paintValues` at
///                     the triangle's corners (face-vertex indices);
///   UsdGenImageMap -> nearest texel at the triangle's interpolated UV (USD
///                     convention, v = 0 is the bottom row): Rec.601 luma
///                     0.299 R + 0.587 G + 0.114 B for 3+ channels, else
///                     channel 0.
/// Returns `fallback` when the map has no usable payload. Clamped to [0,1].
float SampleMaskMap(UsdGenMapDesc const &map, TriangleBvh const &bvh, int triangle,
                    GfVec3f const &bary, float fallback = 1.0f);

/// The map descriptors a node binds through `relationship` (e.g.
/// "usdGen:densityMap"), resolved against the graph description, in
/// authored order. Untyped legacy bindings (empty relationship) never match.
std::vector<UsdGenMapDesc const *> BoundMaps(UsdGenGraphDesc const *desc,
                                             UsdGenMapBindingDesc const *bindings,
                                             uint32_t bindingCount,
                                             TfToken const &relationship);

}  // namespace strand
}  // namespace usdGen

#endif  // USDGEN_OPS_STRAND_UTIL_H
