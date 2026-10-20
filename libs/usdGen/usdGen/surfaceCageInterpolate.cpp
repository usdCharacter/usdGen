#include "usdGen/surfaceCageInterpolate.h"

#include "usdGen/concaveMaterialRemap.h"

#include "usdGenMath/usdGenMath/hash.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

constexpr float kEpsilon = 1.0e-7f;
constexpr size_t kMaterialRingCacheLimit = 256;

bool Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

bool Finite(float value) { return std::isfinite(value); }
bool Finite(GfVec2f const &value) { return Finite(value[0]) && Finite(value[1]); }
bool Finite(GfVec3f const &value) {
    return Finite(value[0]) && Finite(value[1]) && Finite(value[2]);
}

float TriangleArea(GfVec3f const &a, GfVec3f const &b, GfVec3f const &c)
{
    return 0.5f * GfCross(b - a, c - a).GetLength();
}

struct FaceData
{
    int face = -1;
    std::vector<int> vertices;
    std::vector<float> fanAreas;
    float area = 0.0f;
};

struct OwnerData
{
    UsdGenSurfaceCageOwner const *owner = nullptr;
    std::vector<uint32_t> triangles;
    std::vector<uint32_t> guides; // immutable writer ring-slot order
    int profileBegin = 0, profileEnd = 0;
};

bool ValidateGuide(UsdGenSurfaceCageGuide const &guide, size_t index,
                   std::map<int, OwnerData> const &owners, std::string *error)
{
    if (owners.find(guide.ownerTubeId) == owners.end())
        return Fail("surface-cage guide " + std::to_string(index) +
                    " has no declared owner", error);
    size_t const count = guide.points.size();
    if (count < 2 || guide.normalizedT.size() != count ||
        (!guide.rest.empty() && guide.rest.size() != count) ||
        (!guide.widths.empty() && guide.widths.size() != count))
        return Fail("surface-cage guide " + std::to_string(index) +
                    " has inconsistent rail cardinality", error);
    float prior = -1.0f;
    for (size_t point = 0; point != count; ++point) {
        if (!Finite(guide.points[point]) ||
            (!guide.rest.empty() && !Finite(guide.rest[point])) ||
            (!guide.widths.empty() &&
             (!Finite(guide.widths[point]) || guide.widths[point] < 0.0f)) ||
            !Finite(guide.normalizedT[point]) ||
            guide.normalizedT[point] < 0.0f || guide.normalizedT[point] > 1.0f ||
            (point != 0 && !(guide.normalizedT[point] > prior)))
            return Fail("surface-cage guide " + std::to_string(index) +
                        " has invalid normalized rail samples", error);
        prior = guide.normalizedT[point];
    }
    if (std::abs(guide.normalizedT.front()) > kEpsilon ||
        std::abs(guide.normalizedT.back() - 1.0f) > kEpsilon)
        return Fail("surface-cage guide " + std::to_string(index) +
                    " must span normalized t from zero to one", error);
    return true;
}

bool ValidateInput(UsdGenSurfaceCageInput const &input,
                   std::vector<FaceData> *faces,
                   std::map<int, OwnerData> *owners,
                   std::vector<int> *ptexFirstFaceIds,
                   std::string *error)
{
    if (!faces || !owners || !ptexFirstFaceIds)
        return Fail("surface-cage validation has null scratch output", error);
    if (!Finite(input.densityMultiplier) || input.densityMultiplier < 0.0f)
        return Fail("surface-cage density multiplier must be finite and non-negative", error);
    if (!input.ownerMap.texture || input.ownerMap.channel != 0 ||
        input.ownerMap.options.firstChannel != 0 ||
        input.ownerMap.options.channelCount < 1 ||
        input.ownerMap.options.filter != "nearest")
        return Fail("surface-cage owner map must use categorical nearest channel zero", error);
    if (input.ownerMap.texture->SampleChannels() < 1)
        return Fail("surface-cage owner map has no sampled channel zero", error);
    if (input.surface.restFromCurrentPoints || input.surface.restPoints.empty())
        return Fail("surface-cage requires authored/default-time surface rest points", error);
    if (input.surface.faceVertexCounts.empty() ||
        input.surface.faceVertexCounts.size() > std::numeric_limits<int>::max())
        return Fail("surface-cage requires nonempty valid surface topology", error);
    if (!input.surface.uv.empty() &&
        input.surface.uv.size() != input.surface.restPoints.size())
        return Fail("surface-cage surface UV cardinality differs from rest positions", error);
    for (GfVec3f const &point : input.surface.restPoints)
        if (!Finite(point)) return Fail("surface-cage rest points are non-finite", error);
    for (GfVec2f const &uv : input.surface.uv)
        if (!Finite(uv)) return Fail("surface-cage surface UVs are non-finite", error);

    size_t corners = 0;
    faces->clear();
    faces->reserve(input.surface.faceVertexCounts.size());
    for (size_t face = 0; face != input.surface.faceVertexCounts.size(); ++face) {
        int const count = input.surface.faceVertexCounts[face];
        if (count < 3 || corners > input.surface.faceVertexIndices.size() ||
            corners > size_t(std::numeric_limits<int>::max()) - size_t(count) ||
            size_t(count) > input.surface.faceVertexIndices.size() - corners)
            return Fail("surface-cage surface has invalid face topology", error);
        FaceData data;
        data.face = static_cast<int>(face);
        data.vertices.reserve(size_t(count));
        for (int corner = 0; corner != count; ++corner) {
            int const vertex = input.surface.faceVertexIndices[corners + size_t(corner)];
            if (vertex < 0 || size_t(vertex) >= input.surface.restPoints.size())
                return Fail("surface-cage surface has an out-of-range face vertex", error);
            data.vertices.push_back(vertex);
        }
        GfVec3f const &root = input.surface.restPoints[size_t(data.vertices.front())];
        for (size_t triangle = 1; triangle + 1 < data.vertices.size(); ++triangle) {
            float const area = TriangleArea(root,
                input.surface.restPoints[size_t(data.vertices[triangle])],
                input.surface.restPoints[size_t(data.vertices[triangle + 1])]);
            if (!Finite(area)) return Fail("surface-cage face area is non-finite", error);
            data.fanAreas.push_back(area);
            data.area += area;
        }
        if (!(data.area > 0.0f))
            return Fail("surface-cage surface has a degenerate rest face", error);
        faces->push_back(std::move(data));
        corners += size_t(count);
    }
    if (corners != input.surface.faceVertexIndices.size())
        return Fail("surface-cage surface face-index cardinality disagrees with counts", error);

    owners->clear();
    if (input.owners.empty() ||
        input.ownerLengthProfileOffsets.size() != input.owners.size() + 1 ||
        input.ownerLengthProfileOffsets.empty() ||
        input.ownerLengthProfileOffsets.front() != 0 ||
        input.ownerLengthProfileOffsets.back() != int(input.ownerLengthProfile.size()))
        return Fail("surface-cage owner length profiles have invalid offsets", error);
    for (size_t ownerIndex = 0; ownerIndex != input.owners.size(); ++ownerIndex) {
        UsdGenSurfaceCageOwner const &owner = input.owners[ownerIndex];
        int const begin = input.ownerLengthProfileOffsets[ownerIndex];
        int const end = input.ownerLengthProfileOffsets[ownerIndex + 1];
        if (owner.ownerTubeId < 0 || owner.ownerTubeId >= 16777216 ||
            !Finite(owner.density) || owner.density < 0.0f ||
            owner.cvCount < 2 || owner.cvCount > int(kUsdGenMaxCvsPerCurve) ||
            !Finite(owner.defaultWidth) || owner.defaultWidth < 0.0f ||
            !Finite(owner.edgeBias) || owner.edgeBias < -1.0f || owner.edgeBias > 1.0f ||
            !Finite(owner.chartCentroid) ||
            !Finite(owner.chartMeanRadius) || !(owner.chartMeanRadius > 0.0f) ||
            begin < 0 || end <= begin || size_t(end) > input.ownerLengthProfile.size() ||
            owners->find(owner.ownerTubeId) != owners->end())
            return Fail("surface-cage owner " + std::to_string(ownerIndex) +
                        " has invalid controls", error);
        float lastX = -1.0f;
        for (int profile = begin; profile != end; ++profile) {
            GfVec2f const pair = input.ownerLengthProfile[size_t(profile)];
            if (!Finite(pair) || pair[0] < 0.0f || pair[0] > 1.0f ||
                !(pair[0] > lastX) || pair[1] < 0.0f || pair[1] > 1.0f)
                return Fail("surface-cage owner length profile is invalid", error);
            lastX = pair[0];
        }
        OwnerData data;
        data.owner = &owner; data.profileBegin = begin; data.profileEnd = end;
        owners->emplace(owner.ownerTubeId, std::move(data));
    }
    std::set<int> closedGuideOwners;
    int lastGuideOwner = -1;
    bool haveLastGuideOwner = false;
    for (size_t guide = 0; guide != input.guides.size(); ++guide) {
        if (!ValidateGuide(input.guides[guide], guide, *owners, error))
            return false;
        int const ownerId = input.guides[guide].ownerTubeId;
        if (haveLastGuideOwner && ownerId != lastGuideOwner) {
            closedGuideOwners.insert(lastGuideOwner);
            if (closedGuideOwners.find(ownerId) != closedGuideOwners.end())
                return Fail("surface-cage owner rails must be contiguous in immutable slot order", error);
        }
        owners->find(ownerId)->second.guides.push_back(uint32_t(guide));
        lastGuideOwner = ownerId;
        haveLastGuideOwner = true;
    }

    std::set<uint32_t> referencedGuides;
    for (size_t triangle = 0; triangle != input.triangles.size(); ++triangle) {
        UsdGenSurfaceCageTriangle const &binding = input.triangles[triangle];
        auto const owner = owners->find(binding.ownerTubeId);
        if (owner == owners->end() || binding.guides[0] == binding.guides[1] ||
            binding.guides[0] == binding.guides[2] || binding.guides[1] == binding.guides[2])
            return Fail("surface-cage triangle " + std::to_string(triangle) +
                        " has invalid ownership or repeated rails", error);
        for (size_t rail = 0; rail != 3; ++rail) {
            uint32_t const index = binding.guides[rail];
            if (index >= input.guides.size() ||
                input.guides[index].ownerTubeId != binding.ownerTubeId ||
                !Finite(binding.rootChart[rail]))
                return Fail("surface-cage triangle " + std::to_string(triangle) +
                            " has an invalid rail binding", error);
        }
        auto rootPoint = [&](uint32_t guide) -> GfVec3f const & {
            UsdGenSurfaceCageGuide const &rail = input.guides[guide];
            return rail.rest.empty() ? rail.points.front() : rail.rest.front();
        };
        GfVec3f const a = rootPoint(binding.guides[0]);
        GfVec3f const b = rootPoint(binding.guides[1]);
        GfVec3f const c = rootPoint(binding.guides[2]);
        GfVec3d const ab(double(b[0]) - double(a[0]), double(b[1]) - double(a[1]),
                          double(b[2]) - double(a[2]));
        GfVec3d const ac(double(c[0]) - double(a[0]), double(c[1]) - double(a[1]),
                          double(c[2]) - double(a[2]));
        double const d00 = GfDot(ab, ab);
        double const d01 = GfDot(ab, ac);
        double const d11 = GfDot(ac, ac);
        double const determinant = d00 * d11 - d01 * d01;
        double const tolerance = 128.0 * std::numeric_limits<double>::epsilon() * d00 * d11;
        if (!std::isfinite(determinant) || !(d00 > 0.0) || !(d11 > 0.0) ||
            determinant <= tolerance) {
            std::ostringstream details;
            details << std::setprecision(17) << "surface-cage triangle " << triangle
                    << " has a zero-area root binding: a=(" << a[0] << ',' << a[1] << ',' << a[2]
                    << ") b=(" << b[0] << ',' << b[1] << ',' << b[2] << ") c=(" << c[0] << ','
                    << c[1] << ',' << c[2] << ") d00=" << d00 << " d11=" << d11
                    << " det=" << determinant << " tol=" << tolerance;
            return Fail(details.str(), error);
        }
        owner->second.triangles.push_back(static_cast<uint32_t>(triangle));
        referencedGuides.insert(binding.guides[0]);
        referencedGuides.insert(binding.guides[1]);
        referencedGuides.insert(binding.guides[2]);
    }

    for (auto const &entry : *owners) {
        if (entry.second.triangles.empty() || entry.second.guides.size() < 3)
            return Fail("surface-cage owner has no root triangles", error);
        for (uint32_t guide : entry.second.guides)
            if (referencedGuides.find(guide) == referencedGuides.end())
                return Fail("surface-cage owner has a rail outside its material topology", error);
    }

    *ptexFirstFaceIds = input.ownerMap.firstFaceIds;
    if (input.ownerMap.texture->IsTriangleMesh()) {
        for (FaceData const &face : *faces)
            if (face.vertices.size() != 3)
                return Fail("triangle Ptex owner map requires a triangle rest surface", error);
        ptexFirstFaceIds->clear();
    } else if (ptexFirstFaceIds->empty()) {
        int total = -1;
        *ptexFirstFaceIds = UsdGenPtexFirstFaceIds(input.surface.faceVertexCounts.cdata(),
                                                    input.surface.faceVertexCounts.size(), &total);
        if (total < 0 || ptexFirstFaceIds->size() != faces->size())
            return Fail("surface-cage cannot derive Ptex face mapping", error);
    } else if (ptexFirstFaceIds->size() != faces->size()) {
        return Fail("surface-cage Ptex face mapping differs from surface topology", error);
    }
    return true;
}

GfVec3f SampleRail(VtVec3fArray const &values, VtFloatArray const &t, float sample)
{
    sample = std::max(0.0f, std::min(1.0f, sample));
    auto const upper = std::upper_bound(t.cbegin(), t.cend(), sample);
    size_t right = size_t(upper - t.cbegin());
    if (right == 0) return values.front();
    if (right >= t.size()) return values.back();
    size_t const left = right - 1;
    float const alpha = (sample - t[left]) / (t[right] - t[left]);
    return values[left] * (1.0f - alpha) + values[right] * alpha;
}

float SampleRail(VtFloatArray const &values, VtFloatArray const &t, float sample)
{
    sample = std::max(0.0f, std::min(1.0f, sample));
    auto const upper = std::upper_bound(t.cbegin(), t.cend(), sample);
    size_t right = size_t(upper - t.cbegin());
    if (right == 0) return values.front();
    if (right >= t.size()) return values.back();
    size_t const left = right - 1;
    float const alpha = (sample - t[left]) / (t[right] - t[left]);
    return values[left] * (1.0f - alpha) + values[right] * alpha;
}

float ProfileLength(UsdGenSurfaceCageInput const &input, OwnerData const &owner,
                    GfVec2f const &chart)
{
    UsdGenSurfaceCageOwner const &controls = *owner.owner;
    float const radius = std::max(0.0f, std::min(1.0f,
        (chart - controls.chartCentroid).GetLength() / controls.chartMeanRadius));
    float const biased = std::pow(radius, 1.0f - 0.5f * controls.edgeBias);
    int const begin = owner.profileBegin, end = owner.profileEnd;
    if (biased <= input.ownerLengthProfile[size_t(begin)][0])
        return input.ownerLengthProfile[size_t(begin)][1];
    if (biased >= input.ownerLengthProfile[size_t(end - 1)][0])
        return input.ownerLengthProfile[size_t(end - 1)][1];
    for (int index = begin + 1; index != end; ++index) {
        GfVec2f const left = input.ownerLengthProfile[size_t(index - 1)];
        GfVec2f const right = input.ownerLengthProfile[size_t(index)];
        if (biased <= right[0]) {
            float const alpha = (biased - left[0]) / (right[0] - left[0]);
            return left[1] * (1.0f - alpha) + right[1] * alpha;
        }
    }
    return input.ownerLengthProfile[size_t(end - 1)][1];
}

std::array<float, 3> BiasedWeights(std::array<float, 3> weights,
                                   float edgeBias, float progress)
{
    float const minimum = std::min(weights[0], std::min(weights[1], weights[2]));
    float const rho = std::max(0.0f, 1.0f - 3.0f * minimum);
    std::array<float, 3> biased{{1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f}};
    if (rho > kEpsilon) {
        float const ratio = std::pow(rho, 1.0f - 0.5f * edgeBias) / rho;
        for (size_t i = 0; i != 3; ++i)
            biased[i] += (weights[i] - 1.0f / 3.0f) * ratio;
    }
    for (size_t i = 0; i != 3; ++i)
        weights[i] = weights[i] * (1.0f - progress) + biased[i] * progress;
    return weights;
}

bool ProjectRingToNewellPlane(std::vector<GfVec3f> const &ring,
                              std::vector<GfVec2f> *projected)
{
    if (!projected || ring.size() < 3) return false;
    GfVec3d normal(0.0, 0.0, 0.0);
    double minX = ring.front()[0], maxX = minX;
    double minY = ring.front()[1], maxY = minY;
    double minZ = ring.front()[2], maxZ = minZ;
    for (size_t i = 0; i != ring.size(); ++i) {
        GfVec3f const &a = ring[i];
        GfVec3f const &b = ring[(i + 1) % ring.size()];
        if (!Finite(a)) return false;
        normal[0] += double(a[1] - b[1]) * double(a[2] + b[2]);
        normal[1] += double(a[2] - b[2]) * double(a[0] + b[0]);
        normal[2] += double(a[0] - b[0]) * double(a[1] + b[1]);
        minX = std::min(minX, double(a[0])); maxX = std::max(maxX, double(a[0]));
        minY = std::min(minY, double(a[1])); maxY = std::max(maxY, double(a[1]));
        minZ = std::min(minZ, double(a[2])); maxZ = std::max(maxZ, double(a[2]));
    }
    double const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    double const scale2 = dx * dx + dy * dy + dz * dz;
    double const normal2 = GfDot(normal, normal);
    if (!(scale2 > 0.0) || !std::isfinite(scale2) || !std::isfinite(normal2) ||
        normal2 <= double(std::numeric_limits<float>::epsilon()) * scale2 * scale2)
        return false;
    normal /= std::sqrt(normal2);
    GfVec3d axis(1.0, 0.0, 0.0);
    if (std::abs(normal[1]) <= std::abs(normal[0]) &&
        std::abs(normal[1]) <= std::abs(normal[2])) axis = GfVec3d(0.0, 1.0, 0.0);
    else if (std::abs(normal[2]) <= std::abs(normal[0]) &&
             std::abs(normal[2]) <= std::abs(normal[1])) axis = GfVec3d(0.0, 0.0, 1.0);
    GfVec3d const u = GfCross(axis, normal).GetNormalized();
    GfVec3d const v = GfCross(normal, u);
    GfVec3d const origin(ring.front()[0], ring.front()[1], ring.front()[2]);
    projected->clear(); projected->reserve(ring.size());
    for (GfVec3f const &point : ring) {
        GfVec3d const relative = GfVec3d(point) - origin;
        projected->emplace_back(float(GfDot(relative, u)), float(GfDot(relative, v)));
    }
    return true;
}

GfVec2f CanonicalPoint(OwnerData const &owner, UsdGenSurfaceCageTriangle const &binding,
                       std::array<float, 3> const &weights)
{
    size_t const count = owner.guides.size();
    uint32_t const first = owner.guides.front();
    GfVec2f point(0.0f);
    double const step = 2.0 * std::acos(-1.0) / double(count);
    for (size_t i = 0; i != 3; ++i) {
        uint32_t const slot = binding.guides[i] - first;
        double const angle = step * double(slot);
        point += GfVec2f(float(std::cos(angle)), float(std::sin(angle))) * weights[i];
    }
    return point;
}

template <class T>
T BlendRing(std::array<int, 3> const &slots, GfVec3f const &weights,
            std::vector<T> const &ring)
{
    return ring[size_t(slots[0])] * weights[0] + ring[size_t(slots[1])] * weights[1] +
           ring[size_t(slots[2])] * weights[2];
}

struct MaterialRingSample
{
    std::vector<GfVec3f> points, rest;
    std::vector<float> widths;
    std::vector<std::array<int, 3>> triangles;
    bool allWidths = false;
    bool collapsed = false;
};

double RingExtent2(std::vector<GfVec3f> const &ring)
{
    double largest = 0.0;
    for (size_t a = 0; a != ring.size(); ++a)
        for (size_t b = a + 1; b != ring.size(); ++b) {
            GfVec3d const delta(double(ring[a][0]) - double(ring[b][0]),
                                double(ring[a][1]) - double(ring[b][1]),
                                double(ring[a][2]) - double(ring[b][2]));
            largest = std::max(largest, GfDot(delta, delta));
        }
    return largest;
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float cache key size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool BuildMaterialRingSample(UsdGenSurfaceCageInput const &input, OwnerData const &owner,
                             float railT, MaterialRingSample *sample, std::string *error)
{
    if (!sample) return Fail("surface-cage has null material sample output", error);
    MaterialRingSample candidate;
    candidate.points.reserve(owner.guides.size()); candidate.rest.reserve(owner.guides.size());
    candidate.widths.reserve(owner.guides.size());
    candidate.allWidths = true;
    for (uint32_t guideIndex : owner.guides) {
        UsdGenSurfaceCageGuide const &guide = input.guides[guideIndex];
        candidate.points.push_back(SampleRail(guide.points, guide.normalizedT, railT));
        candidate.rest.push_back(SampleRail(guide.rest.empty() ? guide.points : guide.rest,
                                            guide.normalizedT, railT));
        candidate.allWidths = candidate.allWidths && !guide.widths.empty();
        candidate.widths.push_back(guide.widths.empty() ? owner.owner->defaultWidth :
                                   SampleRail(guide.widths, guide.normalizedT, railT));
    }
    std::vector<GfVec3f> rootRing;
    rootRing.reserve(owner.guides.size());
    for (uint32_t guideIndex : owner.guides)
        rootRing.push_back(input.guides[guideIndex].points.front());
    double const rootExtent2 = RingExtent2(rootRing);
    double const currentExtent2 = RingExtent2(candidate.points);
    if (!(rootExtent2 > 0.0) || !std::isfinite(rootExtent2) ||
        !std::isfinite(currentExtent2))
        return Fail("surface-cage target material ring has invalid extent", error);
    if (currentExtent2 <= double(std::numeric_limits<float>::epsilon()) * rootExtent2) {
        candidate.collapsed = true;
        *sample = std::move(candidate);
        return true;
    }
    std::vector<GfVec2f> projected;
    if (!ProjectRingToNewellPlane(candidate.points, &projected))
        return Fail("surface-cage target material ring is degenerate", error);
    std::string triangulateError;
    if (!UsdGenTriangulateConcaveMaterialSlots(projected, &candidate.triangles, &triangulateError))
        return Fail("surface-cage target material ring is invalid: " + triangulateError, error);
    *sample = std::move(candidate);
    return true;
}

bool MaterialRingPoint(UsdGenSurfaceCageInput const &input, OwnerData const &owner,
                       UsdGenSurfaceCageTriangle const &binding,
                       std::array<float, 3> const &rootWeights, float railT,
                       std::map<uint32_t, MaterialRingSample> *cache,
                       GfVec3f *point, GfVec3f *rest, float *width,
                       std::string *error)
{
    if (!cache || !point || !rest || !width)
        return Fail("surface-cage has null material output", error);
    uint32_t const key = FloatBits(railT);
    auto found = cache->find(key);
    MaterialRingSample uncached;
    MaterialRingSample const *sample = nullptr;
    if (found == cache->end()) {
        if (!BuildMaterialRingSample(input, owner, railT, &uncached, error)) return false;
        if (cache->size() < kMaterialRingCacheLimit)
            found = cache->emplace(key, std::move(uncached)).first;
        else
            sample = &uncached;
    }
    if (!sample) sample = &found->second;
    if (sample->collapsed) {
        *point = sample->points.front();
        *rest = sample->rest.front();
        *width = sample->allWidths ? sample->widths.front() : owner.owner->defaultWidth;
        return Finite(*point) && Finite(*rest) && Finite(*width) && *width >= 0.0f;
    }
    std::array<int, 3> slots;
    GfVec3f weights;
    std::string remapError;
    if (!UsdGenLocateConcaveMaterialPoint(owner.guides.size(), sample->triangles,
                                          CanonicalPoint(owner, binding, rootWeights),
                                          &slots, &weights, &remapError))
        return Fail("surface-cage target material ring is invalid: " + remapError, error);
    *point = BlendRing(slots, weights, sample->points);
    *rest = BlendRing(slots, weights, sample->rest);
    *width = sample->allWidths ? BlendRing(slots, weights, sample->widths) :
        owner.owner->defaultWidth;
    return Finite(*point) && Finite(*rest) && Finite(*width) && *width >= 0.0f;
}

bool ContainingTriangle(UsdGenSurfaceCageInput const &input,
                        OwnerData const &owner, GfVec3f const &root,
                        uint32_t *triangleOut,
                        std::array<float, 3> *weightsOut)
{
    if (!triangleOut || !weightsOut) return false;
    // Sparse roots lie on the captured rest scalp.  The triangle chart is
    // geometric, so compute its projected barycentrics in that same local
    // rest space.  A material owner has no nearest-triangle fallback: only a
    // closed containing cell may shape this emitted strand.
    constexpr float tolerance = 2.0e-5f;
    for (uint32_t triangleIndex : owner.triangles) {
        UsdGenSurfaceCageTriangle const &binding = input.triangles[triangleIndex];
        auto rootPoint = [&](uint32_t guide) -> GfVec3f const & {
            UsdGenSurfaceCageGuide const &rail = input.guides[guide];
            return rail.rest.empty() ? rail.points.front() : rail.rest.front();
        };
        GfVec3f const a = rootPoint(binding.guides[0]);
        GfVec3f const b = rootPoint(binding.guides[1]);
        GfVec3f const c = rootPoint(binding.guides[2]);
        GfVec3d const v0(double(b[0]) - double(a[0]), double(b[1]) - double(a[1]),
                          double(b[2]) - double(a[2]));
        GfVec3d const v1(double(c[0]) - double(a[0]), double(c[1]) - double(a[1]),
                          double(c[2]) - double(a[2]));
        GfVec3d const v2(double(root[0]) - double(a[0]), double(root[1]) - double(a[1]),
                          double(root[2]) - double(a[2]));
        double const d00 = GfDot(v0, v0), d01 = GfDot(v0, v1);
        double const d11 = GfDot(v1, v1), d20 = GfDot(v2, v0);
        double const d21 = GfDot(v2, v1);
        double const determinant = d00 * d11 - d01 * d01;
        double const determinantTolerance =
            128.0 * std::numeric_limits<double>::epsilon() * d00 * d11;
        if (!(d00 > 0.0) || !(d11 > 0.0) || !std::isfinite(determinant) ||
            determinant <= determinantTolerance) continue;
        float const w1 = float((d11 * d20 - d01 * d21) / determinant);
        float const w2 = float((d00 * d21 - d01 * d20) / determinant);
        float const w0 = 1.0f - w1 - w2;
        if (!Finite(w0) || !Finite(w1) || !Finite(w2) ||
            w0 < -tolerance || w1 < -tolerance || w2 < -tolerance ||
            w0 > 1.0f + tolerance || w1 > 1.0f + tolerance || w2 > 1.0f + tolerance)
            continue;
        *triangleOut = triangleIndex;
        *weightsOut = {{std::max(0.0f, std::min(1.0f, w0)),
                        std::max(0.0f, std::min(1.0f, w1)),
                        std::max(0.0f, std::min(1.0f, w2))}};
        // Renormalise a boundary tolerance clamp before using it in every
        // rail/frame expression.
        float const sum = (*weightsOut)[0] + (*weightsOut)[1] + (*weightsOut)[2];
        for (float &weight : *weightsOut) weight /= sum;
        return true;
    }
    return false;
}

template <class T>
T Blend(std::array<float, 3> const &weights, T const &a, T const &b, T const &c)
{
    return a * weights[0] + b * weights[1] + c * weights[2];
}

uint64_t CurveId(int ownerTubeId, int seed, int face, uint32_t ordinal)
{
    uint64_t const base = UsdGenCurveId(seed, uint32_t(face), ordinal);
    return UsdGenHash64(base, uint64_t(uint32_t(ownerTubeId)), kSaltScatter);
}

} // namespace

bool UsdGenInterpolateSurfaceCage(UsdGenSurfaceCageInput const &input,
                                  UsdGenSurfaceCageResult *out,
                                  std::string *error)
{
    if (error) error->clear();
    if (!out) return Fail("surface-cage output is null", error);
    std::vector<FaceData> faces;
    std::map<int, OwnerData> owners;
    std::vector<int> firstPtexFaceIds;
    if (!ValidateInput(input, &faces, &owners, &firstPtexFaceIds, error)) return false;

    UsdGenSurfaceCageResult candidate;
    try {
        std::unique_ptr<UsdGenPtexTexture::Sampler> sampler = input.ownerMap.texture->MakeSampler();
        if (!sampler) return Fail("surface-cage cannot allocate an owner-map sampler", error);
        std::vector<float> sample(size_t(input.ownerMap.texture->SampleChannels()), 0.0f);
        std::set<uint64_t> emittedIds;
        std::vector<int> faceOffsets(faces.size() + 1, 0);
        for (size_t face = 0; face != faces.size(); ++face)
            faceOffsets[face + 1] = faceOffsets[face] + int(faces[face].vertices.size());
        std::vector<int> selectedFaces;
        if (!UsdGenSurfaceRestricted(input.surface)) {
            selectedFaces.resize(faces.size());
            std::iota(selectedFaces.begin(), selectedFaces.end(), 0);
        } else {
            selectedFaces.assign(input.surface.subsetFaces.cbegin(), input.surface.subsetFaces.cend());
        }

        for (auto const &entry : owners) {
            OwnerData const &ownerData = entry.second;
            UsdGenSurfaceCageOwner const &owner = *ownerData.owner;
            if (ownerData.triangles.empty() || !(owner.density > 0.0f) ||
                !(input.densityMultiplier > 0.0f))
                continue;
            // Many roots share the same profile station (the usual flat
            // profile case). Reuse the expensive projected ring/ear topology
            // by exact sampled-t bits; canonical lookup remains per root. A
            // bounded cache avoids retaining one ring per dense strand when a
            // varying profile produces unique stations.
            std::map<uint32_t, MaterialRingSample> materialRings;
            for (int faceValue : selectedFaces) {
                if (faceValue < 0 || size_t(faceValue) >= faces.size())
                    return Fail("surface-cage subset contains an out-of-range face", error);
                FaceData const &face = faces[size_t(faceValue)];
                double const expected = double(owner.density) *
                    double(input.densityMultiplier) * double(face.area);
                if (!std::isfinite(expected) || expected > double(std::numeric_limits<uint32_t>::max()))
                    return Fail("surface-cage owner emits an invalid root count", error);
                uint64_t const faceKey = UsdGenHash64(
                    UsdGenHash64(uint64_t(uint32_t(owner.seed)),
                                  uint64_t(uint32_t(owner.ownerTubeId)), kSaltScatter),
                    uint64_t(uint32_t(faceValue)), kSaltScatter);
                uint32_t const count = uint32_t(std::floor(expected)) +
                    (UsdGenHash01(faceKey, kSaltScatter) < float(expected - std::floor(expected)) ? 1u : 0u);
                for (uint32_t ordinal = 0; ordinal != count; ++ordinal) {
                    uint64_t const id = CurveId(owner.ownerTubeId, owner.seed, faceValue, ordinal);
                    float const pick = UsdGenDraw01(owner.seed, id, kSaltScatterBary);
                    float const u1 = UsdGenDraw01(owner.seed, id, kSaltScatterBary + 1u);
                    float const u2 = UsdGenDraw01(owner.seed, id, kSaltScatterBary + 2u);
                    size_t fan = face.fanAreas.size() - 1;
                    double cumulative = 0.0, target = double(pick) * double(face.area);
                    for (size_t index = 0; index != face.fanAreas.size(); ++index) {
                        cumulative += face.fanAreas[index];
                        if (target < cumulative) { fan = index; break; }
                    }
                    float const r = std::sqrt(u1);
                    std::array<float, 3> const surfaceWeights{{1.0f - r, u2 * r,
                                                                r * (1.0f - u2)}};
                    int const ia = face.vertices[0];
                    int const ib = face.vertices[fan + 1];
                    int const ic = face.vertices[fan + 2];
                    GfVec3f const root = Blend(surfaceWeights, input.surface.restPoints[size_t(ia)],
                                                input.surface.restPoints[size_t(ib)],
                                                input.surface.restPoints[size_t(ic)]);
                    int ptexFace = -1;
                    float ptexU = 0.0f, ptexV = 0.0f;
                    if (!UsdGenPtexFaceCoordinate(
                            reinterpret_cast<float const *>(input.surface.restPoints.cdata()),
                            input.surface.restPoints.size(), input.surface.faceVertexCounts.cdata(),
                            input.surface.faceVertexIndices.cdata(), faceOffsets.data(), faces.size(),
                            firstPtexFaceIds.empty() ? nullptr : firstPtexFaceIds.data(),
                            input.ownerMap.texture->IsTriangleMesh(), faceValue,
                            root[0], root[1], root[2], &ptexFace, &ptexU, &ptexV) ||
                        !sampler->Sample(ptexFace, ptexU, ptexV, sample.data()) ||
                        sample[0] != float(owner.ownerTubeId + 1))
                        continue; // categorical zero/other-owner/outside all reject exactly.

                    uint32_t triangleIndex = 0;
                    std::array<float, 3> rootWeights;
                    if (!ContainingTriangle(input, ownerData, root, &triangleIndex,
                                            &rootWeights))
                        continue;
                    if (!emittedIds.insert(id).second)
                        return Fail("surface-cage emitted a duplicate stable curve id", error);
                    UsdGenSurfaceCageTriangle const &binding = input.triangles[triangleIndex];
                    GfVec2f const chart = Blend(rootWeights, binding.rootChart[0],
                                                 binding.rootChart[1], binding.rootChart[2]);
                    float const profileLength = ProfileLength(input, ownerData, chart);
                    if (!Finite(profileLength))
                        return Fail("surface-cage length profile is non-finite", error);
                    float const length = std::max(0.0f, std::min(1.0f, profileLength));

                    if (candidate.curveVertexCounts.size() >=
                            size_t(std::numeric_limits<uint32_t>::max()) ||
                        candidate.points.size() >
                            size_t(std::numeric_limits<uint32_t>::max()) - owner.cvCount)
                        return Fail("surface-cage output cardinality exceeds uint32", error);
                    size_t const base = candidate.points.size();
                    candidate.curveVertexCounts.push_back(int(owner.cvCount));
                    candidate.curveId.push_back(id);
                    candidate.rootPrim.push_back(faceValue);
                    candidate.rootUV.push_back(input.surface.uv.empty()
                        ? GfVec2f(ptexU, ptexV)
                        : Blend(surfaceWeights, input.surface.uv[size_t(ia)],
                                input.surface.uv[size_t(ib)], input.surface.uv[size_t(ic)]));
                    candidate.ownerTubeId.push_back(owner.ownerTubeId);
                    for (uint32_t cv = 0; cv != owner.cvCount; ++cv) {
                        float const progress = owner.cvCount == 1 ? 0.0f :
                            float(cv) / float(owner.cvCount - 1);
                        float const railT = progress * length;
                        std::array<float, 3> const weights =
                            BiasedWeights(rootWeights, owner.edgeBias, railT);
                        UsdGenSurfaceCageGuide const &a = input.guides[binding.guides[0]];
                        UsdGenSurfaceCageGuide const &b = input.guides[binding.guides[1]];
                        UsdGenSurfaceCageGuide const &c = input.guides[binding.guides[2]];
                        GfVec3f point, rest;
                        float width = owner.defaultWidth;
                        if (!MaterialRingPoint(input, ownerData, binding, weights, railT,
                                               &materialRings,
                                               &point, &rest, &width, error))
                            return false;
                        GfVec3f const rootPoint = Blend(rootWeights, a.points.front(),
                                                         b.points.front(), c.points.front());
                        GfVec3f const rootRest = Blend(rootWeights,
                            (a.rest.empty() ? a.points : a.rest).front(),
                            (b.rest.empty() ? b.points : b.rest).front(),
                            (c.rest.empty() ? c.points : c.rest).front());
                        float const anchor = 1.0f - railT;
                        candidate.points.push_back(cv == 0 ? root :
                            point + (root - rootPoint) * anchor);
                        candidate.rest.push_back(cv == 0 ? root :
                            rest + (root - rootRest) * anchor);
                        candidate.widths.push_back(width);
                        candidate.hairT.push_back(progress);
                    }
                    // The physical root correction fades in rail parameter,
                    // so c0 is exact and an unshortened tip remains the full
                    // cage ring.  This is shared with Fill's CPU/CUDA path.
                    if (candidate.points[base] != root || candidate.rest[base] != root)
                        return Fail("surface-cage failed to anchor an emitted root", error);
                }
            }
        }
    } catch (...) {
        return Fail("surface-cage interpolation allocation failed", error);
    }
    *out = std::move(candidate);
    return true;
}

} // namespace usdGen
