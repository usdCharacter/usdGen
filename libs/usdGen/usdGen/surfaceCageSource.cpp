#include "usdGen/surfaceCageSource.h"

#include "usdGen/maps/ptexMap.h"
#include "usdGen/surfaceCageInterpolate.h"

#include "pxr/base/gf/vec3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
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
bool Finite(float value) { return std::isfinite(value); }
bool Finite(GfVec3f const &value) {
    return Finite(value[0]) && Finite(value[1]) && Finite(value[2]);
}
bool Finite(GfMatrix4d const &matrix) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(matrix[row][column])) return false;
    return true;
}

UsdGenParamValue const *FindParam(std::vector<UsdGenParamValue> const &params,
                                  char const *name)
{
    TfToken const key(name);
    UsdGenParamValue const *found = nullptr;
    for (UsdGenParamValue const &param : params) {
        if (param.name != key) continue;
        if (found) return nullptr; // duplicate means malformed, never pick one.
        found = &param;
    }
    return found;
}

bool Number(VtValue const &value, double *out)
{
    if (!out) return false;
    if (value.IsHolding<float>()) { *out = value.UncheckedGet<float>(); return true; }
    if (value.IsHolding<double>()) { *out = value.UncheckedGet<double>(); return true; }
    if (value.IsHolding<int>()) { *out = value.UncheckedGet<int>(); return true; }
    return false;
}

bool Token(UsdGenMapDesc const &map, char const *name, std::string *out)
{
    UsdGenParamValue const *param = FindParam(map.params, name);
    if (!param) return false;
    if (param->value.IsHolding<TfToken>()) {
        *out = param->value.UncheckedGet<TfToken>().GetString(); return true;
    }
    if (param->value.IsHolding<std::string>()) {
        *out = param->value.UncheckedGet<std::string>(); return true;
    }
    return false;
}

bool MapNumber(UsdGenMapDesc const &map, char const *name, double fallback,
               double *out)
{
    UsdGenParamValue const *param = FindParam(map.params, name);
    if (!param) { *out = fallback; return true; }
    return Number(param->value, out) && std::isfinite(*out);
}

struct Prepared
{
    UsdGenSurfaceCageInput input;
    std::map<int, int> ownerRegion, ownerHierarchy;
    float densityMultiplier = 1.0f;
    uint64_t boundCurves = 0, boundCvs = 0;
};

bool GetUniformIntPlane(UsdGenCurveSetDesc const &sparse, char const *name,
                        bool required, VtIntArray *values, std::string *error)
{
    UsdGenAuthoredPlaneDesc const *found = nullptr;
    for (UsdGenAuthoredPlaneDesc const &plane : sparse.authoredPlanes) {
        if (plane.name != TfToken(name)) continue;
        if (found)
            return Fail(std::string("surface-cage sparse curves have duplicate ") + name +
                        " planes", error);
        found = &plane;
    }
    if (!found) {
        if (!required) { values->clear(); return true; }
        return Fail(std::string("surface-cage sparse curves require a uniform int ") + name +
                    " plane", error);
    }
    if (!found || found->type != UsdGenAuthoredPlaneType::Int32 ||
        found->domain != UsdGenAuthoredPlaneDomain::Primitive ||
        found->arity != 1 || found->intValues.size() != sparse.curveVertexCounts.size())
        return Fail(std::string("surface-cage sparse curves require a uniform int ") + name +
                    " plane", error);
    *values = found->intValues;
    return true;
}

bool TransformPoint(GfMatrix4d const &matrix, GfVec3f const &source,
                    GfVec3f *out)
{
    if (!out || !Finite(source)) return false;
    GfVec3d const value = matrix.Transform(GfVec3d(source));
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
        !std::isfinite(float(value[0])) || !std::isfinite(float(value[1])) ||
        !std::isfinite(float(value[2]))) return false;
    *out = GfVec3f(float(value[0]), float(value[1]), float(value[2]));
    return true;
}

bool Prepare(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
             UsdGenCurveSetDesc const &sparse, bool openMap,
             Prepared *prepared, std::string *error)
{
    if (!prepared) return Fail("surface-cage bridge has null prepared output", error);
    if (!sparse.surfaceCage || sparse.restFromCurrentPoints || sparse.points.empty() ||
        sparse.rest.size() != sparse.points.size())
        return Fail("surface-cage requires captured sparse points and authored rest", error);
    if (node.surfaces.size() != 1)
        return Fail("surface-cage requires exactly one bound surface", error);
    auto const surface = std::find_if(desc.surfaces.begin(), desc.surfaces.end(),
        [&](UsdGenSurfaceDesc const &candidate) { return candidate.path == node.surfaces.front(); });
    if (surface == desc.surfaces.end())
        return Fail("surface-cage cannot resolve its bound rest surface", error);
    if (surface->restFromCurrentPoints || surface->restPoints.empty())
        return Fail("surface-cage requires authored/default-time surface rest", error);
    if (!std::isfinite(desc.defaultWidth) || desc.defaultWidth < 0.0f)
        return Fail("surface-cage description default width is invalid", error);
    if (!Finite(sparse.worldMatrix) || !Finite(surface->worldMatrix))
        return Fail("surface-cage source or surface transform is non-finite", error);

    std::vector<UsdGenMapBindingDesc> const &bindings = node.mapBindings;
    if (bindings.size() != 1 || bindings.front().relationship != TfToken("usdGen:regionMap") ||
        bindings.front().map.IsEmpty())
        return Fail("surface-cage requires exactly one usdGen:regionMap binding", error);
    if (!node.maps.empty() && (node.maps.size() != 1 || node.maps.front() != bindings.front().map))
        return Fail("surface-cage regionMap and legacy map paths disagree", error);
    auto const map = std::find_if(desc.maps.begin(), desc.maps.end(),
        [&](UsdGenMapDesc const &candidate) { return candidate.path == bindings.front().map; });
    if (map == desc.maps.end() || map->type != TfToken("UsdGenPtexMap") ||
        map->resolvedAssetPath.empty())
        return Fail("surface-cage regionMap must resolve to a UsdGenPtexMap asset", error);
    UsdGenParamValue const *expect = FindParam(node.params, "expectMapGeneration");
    UsdGenParamValue const *authoredMapGeneration =
        FindParam(map->params, "map:textureGeneration");
    if (!expect || !expect->value.IsHolding<uint64_t>() || !authoredMapGeneration ||
        !authoredMapGeneration->value.IsHolding<uint64_t>() ||
        expect->value.UncheckedGet<uint64_t>() !=
            authoredMapGeneration->value.UncheckedGet<uint64_t>())
        return Fail("surface-cage expectMapGeneration does not match authored map:textureGeneration", error);
    UsdGenParamValue const *multiplier = FindParam(node.params, "densityMultiplier");
    float densityMultiplier = 1.0f;
    if (multiplier) {
        if (!multiplier->value.IsHolding<float>())
            return Fail("surface-cage densityMultiplier must be a float", error);
        densityMultiplier = multiplier->value.UncheckedGet<float>();
    }
    if (!Finite(densityMultiplier) || !(densityMultiplier > 0.0f))
        return Fail("surface-cage densityMultiplier must be finite and positive", error);
    UsdGenParamValue const *channel = FindParam(node.params, "regionMapChannel");
    if (channel && (!channel->value.IsHolding<int>() ||
                    channel->value.UncheckedGet<int>() != 0))
        return Fail("surface-cage regionMapChannel must be categorical channel zero", error);

    UsdGenPtexMapOptions options;
    double number = 0.0;
    std::string token;
    if (!Token(*map, "map:filter", &token) || token != "nearest" ||
        !MapNumber(*map, "map:firstChannel", 0.0, &number) || number != 0.0 ||
        !MapNumber(*map, "map:channelCount", 1.0, &number) || number < 1.0 ||
        number > double(std::numeric_limits<int>::max()) || std::floor(number) != number)
        return Fail("surface-cage regionMap must use nearest channel zero", error);
    options.filter = "nearest"; options.firstChannel = 0;
    options.channelCount = int(number);
    if (!Token(*map, "map:borderMode", &options.borderMode)) options.borderMode = "clamp";
    if (!MapNumber(*map, "map:blur", 0.0, &number) || number != 0.0)
        return Fail("surface-cage regionMap blur must be zero", error);
    options.blur = 0.0f;

    VtIntArray tubeIds, regionIds, hierarchyLevels;
    if (!GetUniformIntPlane(sparse, "tubeId", true, &tubeIds, error) ||
        !GetUniformIntPlane(sparse, "regionId", false, &regionIds, error) ||
        !GetUniformIntPlane(sparse, "hierarchyLevel", false, &hierarchyLevels, error))
        return false;
    std::map<int, int> ownerRegion, ownerHierarchy;
    for (size_t curve = 0; curve != tubeIds.size(); ++curve) {
        auto checkOwnerValue = [&](VtIntArray const &values, std::map<int, int> *perOwner,
                                   char const *name) -> bool {
            if (values.empty()) return true;
            auto const found = perOwner->find(tubeIds[curve]);
            if (found != perOwner->end() && found->second != values[curve])
                return Fail(std::string("surface-cage owner has inconsistent ") + name +
                            " metadata", error);
            (*perOwner)[tubeIds[curve]] = values[curve];
            return true;
        };
        if (!checkOwnerValue(regionIds, &ownerRegion, "regionId") ||
            !checkOwnerValue(hierarchyLevels, &ownerHierarchy, "hierarchyLevel"))
            return false;
    }
    UsdGenSurfaceCagePayload const &payload = *sparse.surfaceCage;
    size_t const owners = payload.ownerIds.size();
    if (owners == 0 || payload.ownerDensities.size() != owners ||
        payload.ownerSeeds.size() != owners || payload.ownerCvCounts.size() != owners ||
        payload.ownerEdgeBias.size() != owners || payload.ownerChartCentroids.size() != owners ||
        payload.ownerChartMeanRadii.size() != owners ||
        payload.ownerLengthProfileOffsets.size() != owners + 1 ||
        payload.ownerLengthProfileOffsets.empty() ||
        payload.ownerLengthProfileOffsets.front() != 0 ||
        payload.ownerLengthProfileOffsets.back() != int(payload.ownerLengthProfile.size()) ||
        payload.normalizedT.size() != sparse.points.size() ||
        payload.triangleOwnerIndices.size() != payload.triangles.size() ||
        payload.triangleRootCharts.size() != payload.triangles.size() * 3)
        return Fail("surface-cage payload cardinalities are inconsistent", error);

    Prepared candidate;
    candidate.densityMultiplier = densityMultiplier;
    candidate.input.surface = *surface;
    // Fill measures density and finds cage bindings in world space.  Make the
    // pure helper use that same metric, including under a non-uniform scalp
    // transform, and return an explicitly world-space transient descriptor.
    candidate.input.surface.worldMatrix = GfMatrix4d(1.0);
    size_t surfaceCorners = 0;
    for (size_t face = 0; face != surface->faceVertexCounts.size(); ++face) {
        int const count = surface->faceVertexCounts[face];
        if (count < 3 || surfaceCorners > surface->faceVertexIndices.size() ||
            size_t(count) > surface->faceVertexIndices.size() - surfaceCorners)
            return Fail("surface-cage rest surface has invalid face topology", error);
        for (int corner = 0; corner != count; ++corner) {
            int const vertex = surface->faceVertexIndices[surfaceCorners + size_t(corner)];
            if (vertex < 0 || size_t(vertex) >= surface->restPoints.size())
                return Fail("surface-cage rest surface has an invalid face vertex", error);
        }
        surfaceCorners += size_t(count);
    }
    if (surfaceCorners != surface->faceVertexIndices.size())
        return Fail("surface-cage rest surface index cardinality is invalid", error);
    candidate.input.surface.restPoints.clear();
    candidate.input.surface.restPoints.reserve(surface->restPoints.size());
    for (GfVec3f const &point : surface->restPoints) {
        GfVec3f world;
        if (!TransformPoint(surface->worldMatrix, point, &world))
            return Fail("surface-cage rest surface has non-finite points", error);
        candidate.input.surface.restPoints.push_back(world);
    }
    candidate.input.densityMultiplier = densityMultiplier;
    candidate.input.ownerMap.options = options;
    candidate.input.ownerMap.channel = 0;
    candidate.input.ownerLengthProfileOffsets = payload.ownerLengthProfileOffsets;
    candidate.input.ownerLengthProfile = payload.ownerLengthProfile;
    candidate.input.owners.reserve(owners);
    for (size_t index = 0; index != owners; ++index) {
        UsdGenSurfaceCageOwner owner;
        owner.ownerTubeId = payload.ownerIds[index];
        owner.density = payload.ownerDensities[index]; owner.seed = payload.ownerSeeds[index];
        if (payload.ownerCvCounts[index] < 0) return Fail("surface-cage owner cvCount is negative", error);
        owner.cvCount = uint32_t(payload.ownerCvCounts[index]);
        owner.edgeBias = payload.ownerEdgeBias[index];
        owner.chartCentroid = payload.ownerChartCentroids[index];
        owner.chartMeanRadius = payload.ownerChartMeanRadii[index];
        owner.defaultWidth = desc.defaultWidth;
        candidate.input.owners.push_back(owner);
    }
    for (UsdGenSurfaceCageOwner const &owner : candidate.input.owners) {
        if ((!regionIds.empty() && ownerRegion.find(owner.ownerTubeId) == ownerRegion.end()) ||
            (!hierarchyLevels.empty() &&
             ownerHierarchy.find(owner.ownerTubeId) == ownerHierarchy.end()))
            return Fail("surface-cage owner has no matching sparse ownership metadata", error);
    }

    size_t pointOffset = 0;
    if (!sparse.curveId.empty() && sparse.curveId.size() != sparse.curveVertexCounts.size())
        return Fail("surface-cage sparse curveId cardinality is invalid", error);
    candidate.input.guides.reserve(sparse.curveVertexCounts.size());
    for (size_t guideIndex = 0; guideIndex != sparse.curveVertexCounts.size(); ++guideIndex) {
        int const count = sparse.curveVertexCounts[guideIndex];
        if (count < 2 || pointOffset > sparse.points.size() ||
            size_t(count) > sparse.points.size() - pointOffset)
            return Fail("surface-cage sparse rails have invalid topology", error);
        UsdGenSurfaceCageGuide guide;
        guide.ownerTubeId = tubeIds[guideIndex];
        guide.curveId = guideIndex < sparse.curveId.size() ? sparse.curveId[guideIndex] : uint64_t(guideIndex);
        guide.points.reserve(size_t(count)); guide.rest.reserve(size_t(count));
        guide.normalizedT.reserve(size_t(count));
        if (!sparse.widths.empty()) {
            if (sparse.widths.size() != sparse.points.size())
                return Fail("surface-cage sparse width cardinality is invalid", error);
            guide.widths.reserve(size_t(count));
        }
        for (int cv = 0; cv != count; ++cv) {
            GfVec3f point, rest;
            if (!TransformPoint(sparse.worldMatrix, sparse.points[pointOffset + size_t(cv)], &point) ||
                !TransformPoint(sparse.worldMatrix, sparse.rest[pointOffset + size_t(cv)], &rest))
                return Fail("surface-cage sparse rail transform is non-finite", error);
            guide.points.push_back(point); guide.rest.push_back(rest);
            guide.normalizedT.push_back(payload.normalizedT[pointOffset + size_t(cv)]);
            if (!sparse.widths.empty()) guide.widths.push_back(sparse.widths[pointOffset + size_t(cv)]);
        }
        pointOffset += size_t(count);
        candidate.input.guides.push_back(std::move(guide));
    }
    if (pointOffset != sparse.points.size())
        return Fail("surface-cage sparse point cardinality disagrees with topology", error);

    candidate.input.triangles.reserve(payload.triangles.size());
    for (size_t index = 0; index != payload.triangles.size(); ++index) {
        int const ownerIndex = payload.triangleOwnerIndices[index];
        GfVec3i const triangle = payload.triangles[index];
        if (ownerIndex < 0 || size_t(ownerIndex) >= owners || triangle[0] < 0 ||
            triangle[1] < 0 || triangle[2] < 0)
            return Fail("surface-cage triangle has an invalid owner or rail index", error);
        UsdGenSurfaceCageTriangle binding;
        binding.ownerTubeId = candidate.input.owners[size_t(ownerIndex)].ownerTubeId;
        binding.guides = {{uint32_t(triangle[0]), uint32_t(triangle[1]), uint32_t(triangle[2])}};
        for (size_t chart = 0; chart != 3; ++chart)
            binding.rootChart[chart] = payload.triangleRootCharts[index * 3 + chart];
        candidate.input.triangles.push_back(binding);
    }

    // CUDA admission needs an upper bound before the Ptex clipping pass.  It
    // is intentionally conservative: every face could belong to every owner.
    // Cache the world-space offsets and areas once; the previous owner loop
    // rebuilt each prefix and made this O(owners * faces^2).
    std::vector<double> faceAreas(surface->faceVertexCounts.size(), 0.0);
    for (size_t face = 0, offset = 0; face != surface->faceVertexCounts.size(); ++face) {
        int const count = surface->faceVertexCounts[face];
        auto point = [&](int corner) -> GfVec3f const & {
            int const vertex = candidate.input.surface.faceVertexIndices[
                offset + size_t(corner)];
            return candidate.input.surface.restPoints[size_t(vertex)];
        };
        for (int corner = 1; corner + 1 < count; ++corner)
            faceAreas[face] += 0.5 * double(GfCross(point(corner) - point(0),
                                                     point(corner + 1) - point(0)).GetLength());
        if (!std::isfinite(faceAreas[face]))
            return Fail("surface-cage rest surface area is invalid", error);
        offset += size_t(count);
    }
    for (UsdGenSurfaceCageOwner const &owner : candidate.input.owners) {
        if (owner.cvCount < 2 || owner.cvCount > int(kUsdGenMaxCvsPerCurve) || !Finite(owner.density) ||
            !(owner.density >= 0.0f))
            return Fail("surface-cage owner controls are invalid", error);
        for (int face = 0; face != int(surface->faceVertexCounts.size()); ++face) {
            int const count = surface->faceVertexCounts[size_t(face)];
            if (count < 3) return Fail("surface-cage rest surface topology is invalid", error);
            // Bounds do not consult Ptex, so owner clipping can only lower it.
            double const expected = double(owner.density) * double(densityMultiplier) *
                faceAreas[size_t(face)];
            if (!std::isfinite(expected) || expected > double(std::numeric_limits<uint32_t>::max()))
                return Fail("surface-cage dense bound is invalid", error);
            uint64_t const countBound = uint64_t(std::ceil(expected));
            if (candidate.boundCurves > uint64_t(std::numeric_limits<uint32_t>::max()) - countBound ||
                countBound > (uint64_t(std::numeric_limits<uint32_t>::max()) - candidate.boundCvs) /
                    uint64_t(owner.cvCount))
                return Fail("surface-cage dense bound exceeds uint32 output capacity", error);
            candidate.boundCurves += countBound;
            candidate.boundCvs += countBound * uint64_t(owner.cvCount);
        }
    }
    if (openMap) {
        std::string mapError;
        candidate.input.ownerMap.texture = UsdGenPtexTexture::Open(map->resolvedAssetPath, options, &mapError);
        if (!candidate.input.ownerMap.texture)
            return Fail("surface-cage cannot open categorical regionMap: " + mapError, error);
    }
    // Generated curves are keyed by leaf tube ID, so later dense ownership
    // planes can be reconstructed after Ptex clipping without copying a
    // sparse primitive array with the wrong cardinality.
    candidate.ownerRegion = std::move(ownerRegion);
    candidate.ownerHierarchy = std::move(ownerHierarchy);
    *prepared = std::move(candidate);
    return true;
}

} // namespace

bool UsdGenEstimateSurfaceCageCurveSet(
    UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
    UsdGenCurveSetDesc const &sparse, uint64_t *curves, uint64_t *cvs,
    std::string *error)
{
    if (error) error->clear();
    if (!curves || !cvs) return Fail("surface-cage estimate has null output", error);
    Prepared prepared;
    if (!Prepare(desc, node, sparse, false, &prepared, error)) return false;
    *curves = prepared.boundCurves; *cvs = prepared.boundCvs;
    return true;
}

bool UsdGenBuildSurfaceCageCurveSet(
    UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
    UsdGenCurveSetDesc const &sparse, UsdGenCurveSetDesc *out,
    std::string *error)
{
    if (error) error->clear();
    if (!out) return Fail("surface-cage bridge has null output", error);
    Prepared prepared;
    if (!Prepare(desc, node, sparse, true, &prepared, error)) return false;
    UsdGenSurfaceCageResult dense;
    if (!UsdGenInterpolateSurfaceCage(prepared.input, &dense, error)) return false;
    UsdGenCurveSetDesc candidate;
    candidate.path = sparse.path;
    candidate.role = UsdGenRole::Curves;
    candidate.curveRole = TfToken("hair");
    candidate.curveVertexCounts = std::move(dense.curveVertexCounts);
    candidate.points = std::move(dense.points);
    candidate.rest = std::move(dense.rest);
    candidate.restFromCurrentPoints = false;
    candidate.widths = std::move(dense.widths);
    candidate.type = TfToken("cubic"); candidate.basis = TfToken("bspline");
    candidate.wrap = TfToken("pinned"); candidate.widthsInterpolation = TfToken("vertex");
    candidate.skinPrim = std::move(dense.rootPrim);
    candidate.skinPrimUv = std::move(dense.rootUV);
    candidate.curveId = std::move(dense.curveId);
    candidate.worldMatrix = GfMatrix4d(1.0);
    candidate.curveGeneration = sparse.curveGeneration;
    auto appendOwnerPlane = [&](char const *name, std::map<int, int> const &perOwner) -> bool {
        if (perOwner.empty()) return true;
        UsdGenAuthoredPlaneDesc plane;
        plane.name = TfToken(name); plane.type = UsdGenAuthoredPlaneType::Int32;
        plane.domain = UsdGenAuthoredPlaneDomain::Primitive; plane.arity = 1;
        plane.intValues.reserve(dense.ownerTubeId.size());
        for (int ownerTubeId : dense.ownerTubeId) {
            auto const value = perOwner.find(ownerTubeId);
            if (value == perOwner.end())
                return Fail(std::string("surface-cage generated owner lacks ") + name +
                            " metadata", error);
            plane.intValues.push_back(value->second);
        }
        candidate.authoredPlanes.push_back(std::move(plane));
        return true;
    };
    UsdGenAuthoredPlaneDesc ownership;
    ownership.name = TfToken("tubeId"); ownership.type = UsdGenAuthoredPlaneType::Int32;
    ownership.domain = UsdGenAuthoredPlaneDomain::Primitive; ownership.arity = 1;
    ownership.intValues = dense.ownerTubeId;
    candidate.authoredPlanes.push_back(std::move(ownership));
    if (!appendOwnerPlane("regionId", prepared.ownerRegion) ||
        !appendOwnerPlane("hierarchyLevel", prepared.ownerHierarchy))
        return false;
    *out = std::move(candidate);
    return true;
}

} // namespace usdGen
