#include "usdGen/surfaceRootBindings.h"
#include "usdGenSurface/surfaceInput.h"

#include "usdGen/taskScheduler.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

constexpr double kAreaEpsilon = 1.0e-24;
constexpr double kSolveTolerance = 1.0e-10;
constexpr double kEncodingReconstructionTolerance = 1.0e-5;
constexpr size_t kNearestFaceCount = 8;
constexpr size_t kParallelRootThreshold = 128;
constexpr size_t kParallelRootGrain = 32;

// body(i) for every i in [0, n) through UsdGenParallelFor (process
// scheduler; safe to nest and to call from several threads at once).
template <class F>
void _ParallelEach(size_t n, F &&body)
{
    using Body = std::remove_reference_t<F>;
    UsdGenParallelFor(n, [](size_t i, void *p) { (*static_cast<Body *>(p))(i); },
                           const_cast<void *>(static_cast<void const *>(&body)));
}

bool Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

bool Finite(double value) { return std::isfinite(value); }
bool Finite(GfVec3f const &value)
{
    return Finite(value[0]) && Finite(value[1]) && Finite(value[2]);
}
bool Finite(GfVec3d const &value)
{
    return Finite(value[0]) && Finite(value[1]) && Finite(value[2]);
}
bool Finite(GfMatrix4d const &value)
{
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!Finite(value[row][column])) return false;
    return true;
}
bool Affine(GfMatrix4d const &value)
{
    return Finite(value) && value[0][3] == 0.0 && value[1][3] == 0.0 &&
           value[2][3] == 0.0 && value[3][3] == 1.0;
}

GfVec3d ToDouble(GfVec3f const &value)
{
    return {value[0], value[1], value[2]};
}
double Dot(GfVec3d const &a, GfVec3d const &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
GfVec3d Cross(GfVec3d const &a, GfVec3d const &b)
{
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

bool TransformPoint(GfMatrix4d const &matrix, GfVec3d const &point,
                    GfVec3d *out)
{
    // Gf matrices use row-vector affine convention, matching the row-frame
    // convention in surfaceRootFrames and the post-flattened C3 descriptors.
    GfVec3d const result(
        point[0] * matrix[0][0] + point[1] * matrix[1][0] +
            point[2] * matrix[2][0] + matrix[3][0],
        point[0] * matrix[0][1] + point[1] * matrix[1][1] +
            point[2] * matrix[2][1] + matrix[3][1],
        point[0] * matrix[0][2] + point[1] * matrix[1][2] +
            point[2] * matrix[2][2] + matrix[3][2]);
    if (!Finite(result) || !std::isfinite(static_cast<float>(result[0])) ||
        !std::isfinite(static_cast<float>(result[1])) ||
        !std::isfinite(static_cast<float>(result[2]))) return false;
    *out = result;
    return true;
}

struct CentroidCloud {
    std::vector<GfVec3f> points;

    size_t kdtree_get_point_count() const { return points.size(); }
    float kdtree_get_pt(size_t index, size_t dimension) const
    {
        return points[index][dimension];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};

using CentroidIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, CentroidCloud>, CentroidCloud, 3, size_t>;

struct ClosestTriangle {
    GfVec3d point;
    // Barycentric weights for triangle vertices one and two.  Vertex zero's
    // weight is `1-u-v`, which is the C3 triangle/rootUV convention.
    double u = 0.0;
    double v = 0.0;
    double distance2 = std::numeric_limits<double>::infinity();
};

bool ClosestOnTriangle(GfVec3d const &query, GfVec3d const &a,
                       GfVec3d const &b, GfVec3d const &c,
                       ClosestTriangle *out)
{
    GfVec3d const ab = b - a;
    GfVec3d const ac = c - a;
    if (Dot(Cross(ab, ac), Cross(ab, ac)) <= kAreaEpsilon) return false;

    GfVec3d const ap = query - a;
    double const d1 = Dot(ab, ap);
    double const d2 = Dot(ac, ap);
    ClosestTriangle result;
    if (d1 <= 0.0 && d2 <= 0.0) {
        result.point = a;
    } else {
        GfVec3d const bp = query - b;
        double const d3 = Dot(ab, bp);
        double const d4 = Dot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) {
            result.point = b; result.u = 1.0;
        } else {
            double const vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                result.u = d1 / (d1 - d3);
                result.point = a + ab * result.u;
            } else {
                GfVec3d const cp = query - c;
                double const d5 = Dot(ab, cp);
                double const d6 = Dot(ac, cp);
                if (d6 >= 0.0 && d5 <= d6) {
                    result.point = c; result.v = 1.0;
                } else {
                    double const vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        result.v = d2 / (d2 - d6);
                        result.point = a + ac * result.v;
                    } else {
                        double const va = d3 * d6 - d5 * d4;
                        if (va <= 0.0 && (d4 - d3) >= 0.0 &&
                            (d5 - d6) >= 0.0) {
                            double const w = (d4 - d3) /
                                ((d4 - d3) + (d5 - d6));
                            result.u = 1.0 - w;
                            result.v = w;
                            result.point = b + (c - b) * w;
                        } else {
                            double const denominator = va + vb + vc;
                            if (!Finite(denominator) || denominator == 0.0) return false;
                            result.u = vb / denominator;
                            result.v = vc / denominator;
                            result.point = a + ab * result.u + ac * result.v;
                        }
                    }
                }
            }
        }
    }
    result.distance2 = Dot(result.point - query, result.point - query);
    if (!Finite(result.point) || !Finite(result.distance2)) return false;
    *out = result;
    return true;
}

struct QuadCandidate {
    double u = 0.0;
    double v = 0.0;
    GfVec3d point;
    double distance2 = std::numeric_limits<double>::infinity();
};

GfVec3d QuadPoint(std::array<GfVec3d, 4> const &p, double u, double v)
{
    double const a = (1.0 - u) * (1.0 - v);
    double const b = u * (1.0 - v);
    double const c = u * v;
    double const d = (1.0 - u) * v;
    return p[0] * a + p[1] * b + p[2] * c + p[3] * d;
}
GfVec3d QuadDu(std::array<GfVec3d, 4> const &p, double v)
{
    return (p[1] - p[0]) * (1.0 - v) + (p[2] - p[3]) * v;
}
GfVec3d QuadDv(std::array<GfVec3d, 4> const &p, double u)
{
    return (p[3] - p[0]) * (1.0 - u) + (p[2] - p[1]) * u;
}

bool QuadKkt(double u, double v, double gu, double gv)
{
    auto ok = [](double coordinate, double gradient) {
        if (coordinate <= kSolveTolerance) return gradient >= -kSolveTolerance;
        if (coordinate >= 1.0 - kSolveTolerance) return gradient <= kSolveTolerance;
        return std::abs(gradient) <= kSolveTolerance;
    };
    return ok(u, gu) && ok(v, gv);
}

bool RefineQuadSeed(std::array<GfVec3d, 4> const &p, GfVec3d const &query,
                    double initialU, double initialV, QuadCandidate *out)
{
    double u = std::clamp(initialU, 0.0, 1.0);
    double v = std::clamp(initialV, 0.0, 1.0);
    for (int iteration = 0; iteration != 32; ++iteration) {
        GfVec3d const point = QuadPoint(p, u, v);
        GfVec3d const residual = point - query;
        GfVec3d const du = QuadDu(p, v);
        GfVec3d const dv = QuadDv(p, u);
        double const gu = Dot(residual, du);
        double const gv = Dot(residual, dv);
        double const distance2 = Dot(residual, residual);
        if (!Finite(point) || !Finite(gu) || !Finite(gv) || !Finite(distance2)) return false;
        if (QuadKkt(u, v, gu, gv)) {
            out->u = u; out->v = v; out->point = point; out->distance2 = distance2;
            return true;
        }

        double const h00 = Dot(du, du);
        double const h01 = Dot(du, dv);
        double const h11 = Dot(dv, dv);
        double const determinant = h00 * h11 - h01 * h01;
        double stepU = 0.0, stepV = 0.0;
        if (Finite(determinant) && determinant > kAreaEpsilon) {
            stepU = (h11 * gu - h01 * gv) / determinant;
            stepV = (h00 * gv - h01 * gu) / determinant;
        } else {
            // Degenerate normal equations cannot certify an interior solve.
            // A bounded gradient attempt may still reach a valid edge KKT
            // solution; if it does not, the caller reports this root unresolved.
            double const scale = std::max({h00, h11, 1.0});
            stepU = gu / scale;
            stepV = gv / scale;
        }
        if (!Finite(stepU) || !Finite(stepV)) return false;

        bool advanced = false;
        double scale = 1.0;
        for (int lineSearch = 0; lineSearch != 12; ++lineSearch) {
            double const nextU = std::clamp(u - scale * stepU, 0.0, 1.0);
            double const nextV = std::clamp(v - scale * stepV, 0.0, 1.0);
            GfVec3d const nextPoint = QuadPoint(p, nextU, nextV);
            double const nextDistance2 = Dot(nextPoint - query, nextPoint - query);
            double const directional = gu * (nextU - u) + gv * (nextV - v);
            if (Finite(nextDistance2) &&
                nextDistance2 <= distance2 + 1.0e-4 * directional) {
                advanced = true;
                if (std::abs(nextU - u) + std::abs(nextV - v) <= kSolveTolerance) {
                    u = nextU; v = nextV;
                    break;
                }
                u = nextU; v = nextV;
                break;
            }
            scale *= 0.5;
        }
        if (!advanced) return false;
    }
    GfVec3d const point = QuadPoint(p, u, v);
    GfVec3d const residual = point - query;
    if (!Finite(point)) return false;
    if (QuadKkt(u, v, Dot(residual, QuadDu(p, v)), Dot(residual, QuadDv(p, u)))) {
        out->u = u; out->v = v; out->point = point;
        out->distance2 = Dot(residual, residual);
        return Finite(out->distance2);
    }
    return false;
}

bool ClosestOnQuad(std::array<GfVec3d, 4> const &p, GfVec3d const &query,
                   QuadCandidate *out)
{
    // A collapsed quad has neither a reliable bilinear parameterisation nor a
    // deterministic tangent frame. Do not manufacture an edge binding.
    GfVec3d const n0 = Cross(p[1] - p[0], p[2] - p[0]);
    GfVec3d const n1 = Cross(p[2] - p[0], p[3] - p[0]);
    if (Dot(n0, n0) + Dot(n1, n1) <= kAreaEpsilon) return false;

    // Nine grid starts followed by the historical two split-triangle starts.
    // Fixed storage matters in the 100k/1M root path: a viable quad no longer
    // allocates a tiny vector before every bounded solve.
    std::array<std::pair<double, double>, 11> seeds;
    size_t seedCount = 0;
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            seeds[seedCount++] = {0.5 * column, 0.5 * row};

    // The two fan triangles are only initial guesses. Their coordinates are
    // converted to the quad's bilinear domain and never emitted directly.
    ClosestTriangle triangle;
    if (ClosestOnTriangle(query, p[0], p[1], p[2], &triangle))
        seeds[seedCount++] = {triangle.u + triangle.v, triangle.v};
    if (ClosestOnTriangle(query, p[0], p[2], p[3], &triangle))
        seeds[seedCount++] = {triangle.u, triangle.u + triangle.v};

    bool found = false;
    QuadCandidate best;
    for (size_t i = 0; i != seedCount; ++i) {
        auto const &seed = seeds[i];
        QuadCandidate candidate;
        if (!RefineQuadSeed(p, query, seed.first, seed.second, &candidate)) continue;
        if (!found || candidate.distance2 < best.distance2 ||
            (candidate.distance2 == best.distance2 &&
             (candidate.u < best.u ||
              (candidate.u == best.u && candidate.v < best.v)))) {
            best = candidate;
            found = true;
        }
    }
    if (found) *out = best;
    return found;
}

bool QuadAabbStrictlyWorse(std::array<GfVec3d, 4> const &p,
                           GfVec3d const &query, double bestDistance2)
{
    if (!Finite(bestDistance2)) return false;
    GfVec3d lower = p[0], upper = p[0];
    double scale = std::max({1.0, std::abs(query[0]), std::abs(query[1]), std::abs(query[2])});
    for (GfVec3d const &point : p) {
        for (int axis = 0; axis != 3; ++axis) {
            lower[axis] = std::min(lower[axis], point[axis]);
            upper[axis] = std::max(upper[axis], point[axis]);
            scale = std::max(scale, std::abs(point[axis]));
        }
    }
    // Bilerp(u,v) is in the control-point convex hull and thus in this box.
    // Widen it generously before computing the lower bound: subtract/rounding
    // cannot turn an exact tie into a prune, and only strictly worse bounds
    // are rejected below.
    double const pad = 1024.0 * std::numeric_limits<double>::epsilon() * scale;
    double distance2 = 0.0;
    for (int axis = 0; axis != 3; ++axis) {
        double const lo = std::nextafter(lower[axis] - pad,
                                         -std::numeric_limits<double>::infinity());
        double const hi = std::nextafter(upper[axis] + pad,
                                         std::numeric_limits<double>::infinity());
        double delta = 0.0;
        if (query[axis] < lo) delta = lo - query[axis];
        else if (query[axis] > hi) delta = query[axis] - hi;
        distance2 += delta * delta;
    }
    return Finite(distance2) && distance2 > bestDistance2;
}

bool Better(double distance2, size_t face, int fan, double bestDistance2,
            size_t bestFace, int bestFan)
{
    return distance2 < bestDistance2 ||
           (distance2 == bestDistance2 &&
            (face < bestFace || (face == bestFace && fan < bestFan)));
}

bool EncodeTriangleUv(int faceCount, int fan, double u, double v, GfVec2f *out)
{
    if (!Finite(u) || !Finite(v)) return false;
    float encodedU = static_cast<float>(std::clamp(u, 0.0, 1.0));
    float encodedV = static_cast<float>(std::clamp(v, 0.0, 1.0));
    if (!std::isfinite(encodedU) || !std::isfinite(encodedV)) return false;
    // The frame/binding consumers validate in float. Repair only round-up at
    // the barycentric boundary, never a genuine out-of-domain projection.
    while (encodedU + encodedV > 1.0f)
        encodedU = std::nextafter(encodedU, 0.0f);
    if (faceCount == 3) {
        *out = GfVec2f(encodedU, encodedV);
        return true;
    }
    // `u==1,v==0` is the only upper integer boundary a valid triangle can
    // have.  Packed directly it names the next fan (and its vertex zero), not
    // this fan's vertex one. Prefer the exact equivalent prior-fan vertex;
    // fan zero has no equivalent patch, so retain an immediately-below-one
    // approximation that the caller must prove reconstructs within 1e-5.
    int encodedFan = fan;
    if (encodedU == 1.0f && encodedV == 0.0f) {
        if (fan > 0) {
            encodedFan = fan - 1;
            encodedU = 0.0f;
            encodedV = 1.0f;
        } else {
            encodedU = std::nextafter(1.0f, 0.0f);
        }
    }
    float const packed = static_cast<float>(encodedFan) + encodedU;
    if (!std::isfinite(packed) || std::floor(packed) != float(encodedFan) ||
        double(packed) >= double(faceCount - 2)) return false;
    // Above 2^24 a float cannot carry an n-gon fan fraction faithfully. Do
    // not bind to a different fan patch merely because the packed coordinate
    // rounded; leave that root explicitly unresolved instead.
    float const decodedU = packed - std::floor(packed);
    if ((encodedU > 0.0f && decodedU == 0.0f) ||
        decodedU + encodedV > 1.0f) return false;
    *out = GfVec2f(packed, encodedV);
    return true;
}

bool DecodeTriangleBinding(VtVec3fArray const &restPoints,
                           VtIntArray const &faceVertexCounts,
                           VtIntArray const &faceVertexIndices,
                           std::vector<uint32_t> const &faceOffsets,
                           size_t face, GfVec2f const &uv, GfVec3d *out)
{
    int const count = faceVertexCounts[face];
    uint32_t const begin = faceOffsets[face];
    if (!std::isfinite(uv[0]) || !std::isfinite(uv[1]) || uv[0] < 0.0f || uv[1] < 0.0f)
        return false;
    int fan = 0;
    float localU = uv[0];
    if (count == 3) {
        if (localU + uv[1] > 1.0f) return false;
    } else {
        if (double(uv[0]) >= double(count - 2)) return false;
        fan = static_cast<int>(std::floor(uv[0]));
        localU = uv[0] - float(fan);
        if (fan < 0 || fan >= count - 2 || localU + uv[1] > 1.0f) return false;
    }
    auto pointAt = [&](uint32_t corner) {
        return ToDouble(restPoints[static_cast<size_t>(
            faceVertexIndices[begin + corner])]);
    };
    uint32_t const one = count == 3 ? 1u : static_cast<uint32_t>(fan + 1);
    uint32_t const two = count == 3 ? 2u : static_cast<uint32_t>(fan + 2);
    *out = pointAt(0) * (1.0 - localU - uv[1]) + pointAt(one) * localU +
        pointAt(two) * uv[1];
    return Finite(*out);
}

} // namespace

struct UsdGenRestSurfaceBindingCache::Impl
{
    UsdGenNodeKey key;
    UsdGenSurfaceId id = 0;
    bool restFromCurrentPoints = false;
    GfMatrix4d worldMatrix{1.0}, worldInverse{1.0};
    VtVec3fArray restPoints;
    VtIntArray faceVertexCounts, faceVertexIndices;
    std::vector<uint32_t> faceOffsets;
    CentroidCloud centroids;
    std::unique_ptr<CentroidIndex> index;
    size_t bytesOwned = 0;
};

UsdGenRestSurfaceBindingCache::UsdGenRestSurfaceBindingCache(
    std::shared_ptr<const Impl> impl)
    : _impl(std::move(impl))
{
}

UsdGenRestSurfaceBindingCache::~UsdGenRestSurfaceBindingCache() = default;

std::shared_ptr<const UsdGenRestSurfaceBindingCache>
UsdGenRestSurfaceBindingCache::Create(UsdGenSurfaceInputView const &surface,
                                      std::string *error)
{
    if (error) error->clear();
    if (surface.restFromCurrentPoints || surface.restPoints.empty()) {
        Fail("automatic root binding requires authored/default-time surface rest points", error);
        return {};
    }
    if (!Affine(surface.worldMatrix)) {
        Fail("automatic root binding requires a finite affine surface transform", error);
        return {};
    }
    double determinant = 0.0;
    GfMatrix4d const inverse = surface.worldMatrix.GetInverse(&determinant);
    if (!Finite(determinant) || std::abs(determinant) <= std::numeric_limits<double>::epsilon()) {
        Fail("automatic root binding requires an invertible surface transform", error);
        return {};
    }
    for (GfVec3f const &point : surface.restPoints) {
        if (!Finite(point)) {
            Fail("surface rest points contain non-finite values", error);
            return {};
        }
    }

    if (surface.faceVertexCounts.size() > size_t(std::numeric_limits<int>::max()))
    {
        Fail("surface parent-face cardinality exceeds int rootPrim range", error);
        return {};
    }
    size_t indexCount = 0;
    for (int count : surface.faceVertexCounts) {
        if (count < 3 || indexCount > std::numeric_limits<size_t>::max() - size_t(count) ||
            indexCount > size_t(std::numeric_limits<uint32_t>::max()) - size_t(count)) {
            Fail("surface faceVertexCounts has an invalid face", error);
            return {};
        }
        indexCount += size_t(count);
    }
    if (surface.faceVertexIndices.size() != indexCount) {
        Fail("surface faceVertexIndices cardinality disagrees with faceVertexCounts", error);
        return {};
    }
    for (int vertex : surface.faceVertexIndices) {
        if (vertex < 0 || size_t(vertex) >= surface.restPoints.size()) {
            Fail("surface faceVertexIndices contains an out-of-range vertex", error);
            return {};
        }
    }

    try {
        auto impl = std::make_shared<Impl>();
        impl->key = surface.key;
        impl->id = surface.id;
        impl->restFromCurrentPoints = surface.restFromCurrentPoints;
        impl->worldMatrix = surface.worldMatrix;
        impl->worldInverse = inverse;
        // A copy: the cache outlives the input record it was built from.
        impl->restPoints.assign(surface.restPoints.begin(), surface.restPoints.end());
        impl->faceVertexCounts.assign(surface.faceVertexCounts.begin(), surface.faceVertexCounts.end());
        impl->faceVertexIndices.assign(surface.faceVertexIndices.begin(), surface.faceVertexIndices.end());
        impl->faceOffsets.assign(surface.faceVertexCounts.size() + 1, 0);
        for (size_t face = 0; face != surface.faceVertexCounts.size(); ++face)
            impl->faceOffsets[face + 1] = impl->faceOffsets[face] +
                static_cast<uint32_t>(surface.faceVertexCounts[face]);

        impl->centroids.points.reserve(surface.faceVertexCounts.size());
        for (size_t face = 0; face != surface.faceVertexCounts.size(); ++face) {
            GfVec3d centroid(0.0);
            uint32_t const begin = impl->faceOffsets[face];
            uint32_t const count = static_cast<uint32_t>(surface.faceVertexCounts[face]);
            for (uint32_t corner = 0; corner != count; ++corner)
                centroid += ToDouble(surface.restPoints[
                    static_cast<size_t>(surface.faceVertexIndices[begin + corner])]);
            centroid /= double(count);
            if (!Finite(centroid) ||
                !std::isfinite(static_cast<float>(centroid[0])) ||
                !std::isfinite(static_cast<float>(centroid[1])) ||
                !std::isfinite(static_cast<float>(centroid[2]))) {
                Fail("surface face centroids are not finite float values", error);
                return {};
            }
            impl->centroids.points.emplace_back(static_cast<float>(centroid[0]),
                                                static_cast<float>(centroid[1]),
                                                static_cast<float>(centroid[2]));
        }
        if (!impl->centroids.points.empty())
            impl->index = std::make_unique<CentroidIndex>(
                3, impl->centroids, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        impl->bytesOwned = sizeof(Impl) +
            impl->restPoints.size() * sizeof(GfVec3f) +
            impl->faceVertexCounts.size() * sizeof(int) +
            impl->faceVertexIndices.size() * sizeof(int) +
            impl->faceOffsets.capacity() * sizeof(uint32_t) +
            impl->centroids.points.capacity() * sizeof(GfVec3f);
        if (impl->index) impl->bytesOwned += impl->index->usedMemory(*impl->index);
        return std::shared_ptr<const UsdGenRestSurfaceBindingCache>(
            new UsdGenRestSurfaceBindingCache(std::move(impl)));
    } catch (...) {
        Fail("automatic root binding allocation or kd-tree construction failed", error);
        return {};
    }
}

namespace {

template <class T>
bool SameValues(UsdGenSpan<const T> a, VtArray<T> const &b)
{
    return a.size() == b.size() &&
        (a.cdata() == b.cdata() || std::equal(a.begin(), a.end(), b.cdata()));
}

}  // namespace

bool UsdGenRestSurfaceBindingCache::Matches(UsdGenSurfaceInputView const &surface) const
{
    return _impl && surface.key == _impl->key && surface.id == _impl->id &&
        surface.restFromCurrentPoints == _impl->restFromCurrentPoints &&
        surface.worldMatrix == _impl->worldMatrix &&
        SameValues(surface.restPoints, _impl->restPoints) &&
        SameValues(surface.faceVertexCounts, _impl->faceVertexCounts) &&
        SameValues(surface.faceVertexIndices, _impl->faceVertexIndices);
}

size_t UsdGenRestSurfaceBindingCache::BytesOwned() const noexcept
{
    return _impl ? _impl->bytesOwned : 0;
}

bool UsdGenRestSurfaceBindingCache::Bind(
    VtVec3fArray const &sourceRestRoots, GfMatrix4d const &sourceWorldMatrix,
    UsdGenSurfaceRootBindingResult *out, std::string *error) const
{
    if (error) error->clear();
    if (!out) return Fail("root-binding output is null", error);
    if (!_impl) return Fail("root-binding cache is empty", error);
    if (!Affine(sourceWorldMatrix))
        return Fail("automatic root binding requires a finite affine source transform", error);
    if (sourceRestRoots.size() > std::numeric_limits<uint32_t>::max())
        return Fail("root-binding curve cardinality exceeds uint32", error);
    for (GfVec3f const &root : sourceRestRoots)
        if (!Finite(root)) return Fail("source rest roots contain non-finite values", error);

    try {
        UsdGenSurfaceRootBindingResult candidate;
        size_t const roots = sourceRestRoots.size();
        candidate.rootPrim.assign(roots, -1);
        candidate.rootUV.assign(roots, GfVec2f(0.0f));
        candidate.valid.assign(roots, 0);
        candidate.unresolved.assign(roots, 0);
        if (roots == 0) {
            *out = std::move(candidate);
            return true;
        }
        if (_impl->centroids.points.empty())
            return Fail("automatic root binding requires a surface with at least one face", error);

        size_t const k = std::min(kNearestFaceCount, _impl->centroids.points.size());
        // Detach every output plane before the parallel region. Workers access only
        // these stable raw ranges; they never touch a VtArray object or share
        // a result counter/refcount. The final unresolved reduction and COW
        // publication happen after the parallel region joins.
        int *const rootPrim = candidate.rootPrim.data();
        GfVec2f *const rootUV = candidate.rootUV.data();
        uint8_t *const valid = candidate.valid.data();
        uint8_t *const unresolved = candidate.unresolved.data();
        std::atomic<bool> transformFailure{false};
        auto bindOne = [&](size_t rootIndex) {
            if (transformFailure.load(std::memory_order_relaxed)) return;
            std::array<size_t, kNearestFaceCount> nearestFaces{};
            std::array<float, kNearestFaceCount> nearestDistances{};
            GfVec3d worldRoot, localRoot;
            if (!TransformPoint(sourceWorldMatrix, ToDouble(sourceRestRoots[rootIndex]), &worldRoot) ||
                !TransformPoint(_impl->worldInverse, worldRoot, &localRoot)) {
                transformFailure.store(true, std::memory_order_relaxed);
                return;
            }
            float const query[3] = {static_cast<float>(localRoot[0]),
                                    static_cast<float>(localRoot[1]),
                                    static_cast<float>(localRoot[2])};
            size_t const found = _impl->index->knnSearch(
                query, k, nearestFaces.data(), nearestDistances.data());
            if (found == 0) {
                unresolved[rootIndex] = 1;
                return;
            }
            std::array<size_t, kNearestFaceCount> faces = nearestFaces;
            auto const sortedEnd = faces.begin() + found;
            auto faceLess = [&](size_t a, size_t b) {
                GfVec3f const &ca = _impl->centroids.points[a];
                GfVec3f const &cb = _impl->centroids.points[b];
                double const da = (double(ca[0]) - localRoot[0]) * (double(ca[0]) - localRoot[0]) +
                                  (double(ca[1]) - localRoot[1]) * (double(ca[1]) - localRoot[1]) +
                                  (double(ca[2]) - localRoot[2]) * (double(ca[2]) - localRoot[2]);
                double const db = (double(cb[0]) - localRoot[0]) * (double(cb[0]) - localRoot[0]) +
                                  (double(cb[1]) - localRoot[1]) * (double(cb[1]) - localRoot[1]) +
                                  (double(cb[2]) - localRoot[2]) * (double(cb[2]) - localRoot[2]);
                return da < db || (da == db && a < b);
            };
            // At most eight candidates: a bounded insertion sort avoids
            // std::sort's larger small-range threshold and keeps all accesses
            // visibly within the fixed scratch array.
            for (size_t i = 1; i < found; ++i) {
                size_t const face = faces[i];
                size_t j = i;
                while (j && faceLess(face, faces[j - 1])) {
                    faces[j] = faces[j - 1];
                    --j;
                }
                faces[j] = face;
            }
            auto const facesEnd = std::unique(faces.begin(), sortedEnd);

            bool hasBinding = false;
            double bestDistance2 = std::numeric_limits<double>::infinity();
            size_t bestFace = std::numeric_limits<size_t>::max();
            int bestFan = std::numeric_limits<int>::max();
            GfVec2f bestUv(0.0f);
            GfVec3d bestPoint(0.0);
            double bestTriangleU = 0.0, bestTriangleV = 0.0;
            bool bestIsTriangle = false;
            for (auto it = faces.begin(); it != facesEnd; ++it) {
                size_t const face = *it;
                uint32_t const begin = _impl->faceOffsets[face];
                int const count = _impl->faceVertexCounts[face];
                auto pointAt = [&](uint32_t corner) {
                    return ToDouble(_impl->restPoints[static_cast<size_t>(
                        _impl->faceVertexIndices[begin + corner])]);
                };
                if (count == 4) {
                    std::array<GfVec3d, 4> quad = {pointAt(0), pointAt(1), pointAt(2), pointAt(3)};
                    if (hasBinding && QuadAabbStrictlyWorse(quad, localRoot, bestDistance2))
                        continue;
                    QuadCandidate closest;
                    if (!ClosestOnQuad(quad, localRoot, &closest)) continue;
                    GfVec2f const uv(static_cast<float>(closest.u), static_cast<float>(closest.v));
                    if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) continue;
                    if (!hasBinding || Better(closest.distance2, face, 0, bestDistance2, bestFace, bestFan)) {
                        hasBinding = true; bestDistance2 = closest.distance2;
                        bestFace = face; bestFan = 0; bestUv = uv;
                        bestPoint = closest.point; bestIsTriangle = false;
                    }
                    continue;
                }
                int const fanCount = count == 3 ? 1 : count - 2;
                for (int fan = 0; fan != fanCount; ++fan) {
                    uint32_t const one = count == 3 ? 1u : static_cast<uint32_t>(fan + 1);
                    uint32_t const two = count == 3 ? 2u : static_cast<uint32_t>(fan + 2);
                    ClosestTriangle closest;
                    if (!ClosestOnTriangle(localRoot, pointAt(0), pointAt(one), pointAt(two), &closest))
                        continue;
                    if (!hasBinding || Better(closest.distance2, face, fan,
                                              bestDistance2, bestFace, bestFan)) {
                        hasBinding = true; bestDistance2 = closest.distance2;
                        bestFace = face; bestFan = fan;
                        bestPoint = closest.point;
                        bestTriangleU = closest.u; bestTriangleV = closest.v;
                        bestIsTriangle = true;
                    }
                }
            }
            if (!hasBinding) {
                unresolved[rootIndex] = 1;
                return;
            }
            if (bestIsTriangle) {
                int const count = _impl->faceVertexCounts[bestFace];
                GfVec2f encoded;
                GfVec3d reconstructed;
                // Encoding is tested only after stable geometric selection.
                // Therefore a nonrepresentable nearest n-gon boundary does
                // not silently fall through to a farther face/fan candidate.
                if (!EncodeTriangleUv(count, bestFan, bestTriangleU, bestTriangleV, &encoded) ||
                    !DecodeTriangleBinding(_impl->restPoints, _impl->faceVertexCounts,
                                           _impl->faceVertexIndices, _impl->faceOffsets,
                                           bestFace, encoded, &reconstructed) ||
                    Dot(reconstructed - bestPoint, reconstructed - bestPoint) >
                        kEncodingReconstructionTolerance * kEncodingReconstructionTolerance) {
                    unresolved[rootIndex] = 1;
                    return;
                }
                bestUv = encoded;
            }
            rootPrim[rootIndex] = static_cast<int>(bestFace);
            rootUV[rootIndex] = bestUv;
            valid[rootIndex] = 1;
        };
        if (roots < kParallelRootThreshold) {
            for (size_t rootIndex = 0; rootIndex != roots; ++rootIndex) bindOne(rootIndex);
        } else {
            size_t const blocks = (roots + kParallelRootGrain - 1) / kParallelRootGrain;
            _ParallelEach(blocks, [&](size_t block) {
                size_t const end = std::min(roots, (block + 1) * kParallelRootGrain);
                for (size_t rootIndex = block * kParallelRootGrain; rootIndex != end; ++rootIndex)
                    bindOne(rootIndex);
            });
        }
        if (transformFailure.load(std::memory_order_relaxed))
            return Fail("source rest root cannot be transformed into finite surface-local space", error);
        for (size_t rootIndex = 0; rootIndex != roots; ++rootIndex)
            candidate.unresolvedCount += unresolved[rootIndex] != 0;
        *out = std::move(candidate);
        return true;
    } catch (...) {
        return Fail("automatic root binding allocation or kd-tree construction failed", error);
    }
}

bool UsdGenBuildRestSurfaceRootBindings(
    UsdGenSurfaceInputView const &surface, VtVec3fArray const &sourceRestRoots,
    GfMatrix4d const &sourceWorldMatrix, UsdGenSurfaceRootBindingResult *out,
    std::string *error)
{
    if (error) error->clear();
    if (!out) return Fail("root-binding output is null", error);
    auto cache = UsdGenRestSurfaceBindingCache::Create(surface, error);
    return cache && cache->Bind(sourceRestRoots, sourceWorldMatrix, out, error);
}

} // namespace usdGen
