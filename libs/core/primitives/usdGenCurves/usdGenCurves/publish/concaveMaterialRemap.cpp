#include "usdGen/concaveMaterialRemap.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

bool Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

bool Finite(GfVec2f const &point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]);
}

double Cross(GfVec2f const &a, GfVec2f const &b, GfVec2f const &c)
{
    return (double(b[0]) - double(a[0])) * (double(c[1]) - double(a[1])) -
           (double(b[1]) - double(a[1])) * (double(c[0]) - double(a[0]));
}

double PolygonArea2(std::vector<GfVec2f> const &points)
{
    double area = 0.0;
    for (size_t i = 0; i != points.size(); ++i) {
        GfVec2f const &a = points[i];
        GfVec2f const &b = points[(i + 1) % points.size()];
        area += double(a[0]) * double(b[1]) - double(a[1]) * double(b[0]);
    }
    return area;
}

double Scale2(std::vector<GfVec2f> const &points)
{
    double minX = points.front()[0], maxX = minX;
    double minY = points.front()[1], maxY = minY;
    for (GfVec2f const &point : points) {
        minX = std::min(minX, double(point[0])); maxX = std::max(maxX, double(point[0]));
        minY = std::min(minY, double(point[1])); maxY = std::max(maxY, double(point[1]));
    }
    double const dx = maxX - minX, dy = maxY - minY;
    return dx * dx + dy * dy;
}

bool OnSegment(GfVec2f const &a, GfVec2f const &b, GfVec2f const &p,
               double crossEpsilon, double coordinateEpsilon)
{
    return std::abs(Cross(a, b, p)) <= crossEpsilon &&
        double(p[0]) >= std::min(double(a[0]), double(b[0])) - coordinateEpsilon &&
        double(p[0]) <= std::max(double(a[0]), double(b[0])) + coordinateEpsilon &&
        double(p[1]) >= std::min(double(a[1]), double(b[1])) - coordinateEpsilon &&
        double(p[1]) <= std::max(double(a[1]), double(b[1])) + coordinateEpsilon;
}

bool SegmentsIntersect(GfVec2f const &a, GfVec2f const &b,
                       GfVec2f const &c, GfVec2f const &d, double crossEpsilon,
                       double coordinateEpsilon)
{
    double const abC = Cross(a, b, c), abD = Cross(a, b, d);
    double const cdA = Cross(c, d, a), cdB = Cross(c, d, b);
    if (((abC > crossEpsilon && abD < -crossEpsilon) ||
         (abC < -crossEpsilon && abD > crossEpsilon)) &&
        ((cdA > crossEpsilon && cdB < -crossEpsilon) ||
         (cdA < -crossEpsilon && cdB > crossEpsilon)))
        return true;
    return OnSegment(a, b, c, crossEpsilon, coordinateEpsilon) ||
           OnSegment(a, b, d, crossEpsilon, coordinateEpsilon) ||
           OnSegment(c, d, a, crossEpsilon, coordinateEpsilon) ||
           OnSegment(c, d, b, crossEpsilon, coordinateEpsilon);
}

bool HasSelfIntersection(std::vector<GfVec2f> const &points, double crossEpsilon,
                         double coordinateEpsilon)
{
    size_t const count = points.size();
    for (size_t a = 0; a != count; ++a) {
        size_t const b = (a + 1) % count;
        for (size_t c = a + 1; c != count; ++c) {
            size_t const d = (c + 1) % count;
            if (a == c || a == d || b == c || b == d) continue;
            if (SegmentsIntersect(points[a], points[b], points[c], points[d], crossEpsilon,
                                  coordinateEpsilon))
                return true;
        }
    }
    return false;
}

bool Barycentric(GfVec2f const &point, GfVec2f const &a, GfVec2f const &b,
                 GfVec2f const &c, double determinantEpsilon,
                 double weightEpsilon, std::array<float, 3> *weights)
{
    double const determinant = Cross(a, b, c);
    if (!std::isfinite(determinant) || std::abs(determinant) <= determinantEpsilon) return false;
    double const w1 = Cross(a, point, c) / determinant;
    double const w2 = Cross(a, b, point) / determinant;
    double const w0 = 1.0 - w1 - w2;
    if (!std::isfinite(w0) || !std::isfinite(w1) || !std::isfinite(w2) ||
        w0 < -weightEpsilon || w1 < -weightEpsilon || w2 < -weightEpsilon ||
        w0 > 1.0 + weightEpsilon || w1 > 1.0 + weightEpsilon ||
        w2 > 1.0 + weightEpsilon)
        return false;
    std::array<float, 3> result{{float(std::max(0.0, std::min(1.0, w0))),
                                  float(std::max(0.0, std::min(1.0, w1))),
                                  float(std::max(0.0, std::min(1.0, w2)))}};
    float const sum = result[0] + result[1] + result[2];
    if (!(sum > 0.0f) || !std::isfinite(sum)) return false;
    for (float &weight : result) weight /= sum;
    *weights = result;
    return true;
}

std::vector<GfVec2f> CanonicalPolygon(size_t count)
{
    std::vector<GfVec2f> polygon;
    polygon.reserve(count);
    double const step = 2.0 * std::acos(-1.0) / double(count);
    for (size_t i = 0; i != count; ++i) {
        double const angle = step * double(i);
        polygon.emplace_back(float(std::cos(angle)), float(std::sin(angle)));
    }
    return polygon;
}

bool EarTriangulate(std::vector<GfVec2f> const &polygon, double epsilon,
                    double coordinateEpsilon,
                    std::vector<std::array<uint32_t, 3>> *triangles)
{
    double const area2 = PolygonArea2(polygon);
    if (std::abs(area2) <= epsilon ||
        HasSelfIntersection(polygon, epsilon, coordinateEpsilon)) return false;
    double const winding = area2 > 0.0 ? 1.0 : -1.0;
    std::vector<uint32_t> active(polygon.size());
    for (size_t i = 0; i != active.size(); ++i) active[i] = uint32_t(i);
    triangles->clear(); triangles->reserve(polygon.size() - 2);
    while (active.size() > 3) {
        bool removed = false;
        for (size_t at = 0; at != active.size(); ++at) {
            uint32_t const previous = active[(at + active.size() - 1) % active.size()];
            uint32_t const current = active[at];
            uint32_t const next = active[(at + 1) % active.size()];
            double const cross = winding * Cross(polygon[previous], polygon[current], polygon[next]);
            if (!(cross > epsilon)) continue; // do not discard collinear slots.
            bool contains = false;
            for (uint32_t other : active) {
                if (other == previous || other == current || other == next) continue;
                std::array<float, 3> weights;
                if (Barycentric(polygon[other], polygon[previous], polygon[current], polygon[next],
                                epsilon, 2.0e-6, &weights)) {
                    contains = true; break;
                }
            }
            if (contains) continue;
            // Never choose an ear that leaves the final three original slots
            // collinear.  This preserves authored collinear boundary slots
            // instead of silently dropping them from a material triangulation.
            if (active.size() == 4) {
                std::array<uint32_t, 3> remaining{};
                size_t write = 0;
                for (uint32_t slot : active)
                    if (slot != current) remaining[write++] = slot;
                if (std::abs(Cross(polygon[remaining[0]], polygon[remaining[1]],
                                   polygon[remaining[2]])) <= epsilon)
                    continue;
            }
            triangles->push_back({{previous, current, next}});
            active.erase(active.begin() + std::ptrdiff_t(at));
            removed = true;
            break;
        }
        if (!removed) return false;
    }
    double const cross = winding * Cross(polygon[active[0]], polygon[active[1]], polygon[active[2]]);
    if (!(cross > epsilon)) return false;
    triangles->push_back({{active[0], active[1], active[2]}});
    return true;
}

} // namespace

bool UsdGenTriangulateConcaveMaterialSlots(
    std::vector<GfVec2f> const &actual, std::vector<std::array<int, 3>> *triangles,
    std::string *error)
{
    if (error) error->clear();
    if (!triangles) return Fail("concave material triangulation has null output", error);
    if (actual.size() < 3) return Fail("concave material triangulation has invalid polygon", error);
    for (GfVec2f const &point : actual)
        if (!Finite(point)) return Fail("concave material triangulation has non-finite polygon", error);
    double const scale2 = Scale2(actual);
    if (!(scale2 > 0.0) || !std::isfinite(scale2))
        return Fail("concave material triangulation target polygon is degenerate", error);
    // The source positions are float but all geometric predicates are double.
    // A percentage-of-bounds cutoff discards valid thin K14 material cells;
    // use only double arithmetic roundoff relative to this polygon's extent.
    double const machine = 128.0 * std::numeric_limits<double>::epsilon();
    double const areaEpsilon = machine * scale2;
    double const coordinateEpsilon = machine * std::sqrt(scale2);
    std::vector<std::array<uint32_t, 3>> indices;
    if (!EarTriangulate(actual, areaEpsilon, coordinateEpsilon, &indices))
        return Fail("concave material triangulation target polygon is degenerate or self-intersecting",
                    error);
    std::vector<std::array<int, 3>> candidate;
    candidate.reserve(indices.size());
    for (std::array<uint32_t, 3> const &triangle : indices)
        candidate.push_back({{int(triangle[0]), int(triangle[1]), int(triangle[2])}});
    *triangles = std::move(candidate);
    return true;
}

bool UsdGenLocateConcaveMaterialPoint(
    size_t slotCount, std::vector<std::array<int, 3>> const &triangles,
    GfVec2f const &canonicalPoint, std::array<int, 3> *slots,
    GfVec3f *weights, std::string *error)
{
    if (error) error->clear();
    if (!slots || !weights) return Fail("concave material lookup has null output", error);
    if (slotCount < 3 || triangles.size() != slotCount - 2 || !Finite(canonicalPoint))
        return Fail("concave material lookup has invalid topology or point", error);
    std::vector<GfVec2f> const canonical = CanonicalPolygon(slotCount);
    constexpr double epsilon = 2.0e-6;
    for (std::array<int, 3> const &triangle : triangles) {
        if (triangle[0] < 0 || triangle[1] < 0 || triangle[2] < 0 ||
            size_t(triangle[0]) >= slotCount || size_t(triangle[1]) >= slotCount ||
            size_t(triangle[2]) >= slotCount || triangle[0] == triangle[1] ||
            triangle[0] == triangle[2] || triangle[1] == triangle[2])
            return Fail("concave material lookup has invalid triangle slots", error);
        std::array<float, 3> bary;
        if (!Barycentric(canonicalPoint, canonical[size_t(triangle[0])],
                         canonical[size_t(triangle[1])], canonical[size_t(triangle[2])],
                         epsilon, epsilon, &bary))
            continue;
        *slots = triangle;
        *weights = GfVec3f(bary[0], bary[1], bary[2]);
        return true;
    }
    return Fail("concave material lookup point lies outside canonical polygon", error);
}

bool UsdGenRemapConcaveMaterialPoint(
    std::vector<GfVec2f> const &actual, GfVec2f const &canonicalPoint,
    std::array<int, 3> *slots, GfVec3f *weights,
    std::string *error)
{
    if (error) error->clear();
    if (!slots || !weights) return Fail("concave material remap has null output", error);
    if (actual.size() < 3 || !Finite(canonicalPoint))
        return Fail("concave material remap has invalid polygon or point", error);
    std::vector<std::array<int, 3>> triangles;
    if (!UsdGenTriangulateConcaveMaterialSlots(actual, &triangles, error)) return false;
    return UsdGenLocateConcaveMaterialPoint(actual.size(), triangles, canonicalPoint,
                                            slots, weights, error);
}

} // namespace usdGen
