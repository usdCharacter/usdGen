// Pure runtime contract for sparse surface-cage C3 interpolation.
#include "utils/pxrInterop.h"
#include "usdGen/surfaceCageInterpolate.h"
#include "usdGenUsd/lower.h"
#include "usdGen/surfaceCageSource.h"
#include "usdGen/concaveMaterialRemap.h"

#include <Ptexture.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int gFailures = 0;
void Check(bool ok, char const *what)
{
    if (ok) std::printf("ok:   %s\n", what);
    else { ++gFailures; std::printf("FAIL: %s\n", what); }
}

bool Near(float a, float b, float tolerance = 1.0e-5f)
{
    return std::abs(a - b) <= tolerance;
}

bool Near(usdGen::UsdGenVec3f const &a, usdGen::UsdGenVec3f const &b, float tolerance = 1.0e-5f)
{
    return Near(a[0], b[0], tolerance) && Near(a[1], b[1], tolerance) &&
           Near(a[2], b[2], tolerance);
}

bool WriteOwnerMap(std::string const &path, int ptexFaces, float ownerValue)
{
    Ptex::String error;
    PtexWriter *writer = PtexWriter::open(path.c_str(), Ptex::mt_quad, Ptex::dt_float,
                                           1, -1, ptexFaces, error, false);
    if (!writer) {
        std::printf("cannot create owner map: %s\n", error.c_str());
        return false;
    }
    Ptex::Res const resolution(int8_t(0), int8_t(0));
    int adjacentFaces[4] = {-1, -1, -1, -1};
    int adjacentEdges[4] = {0, 0, 0, 0};
    bool ok = true;
    for (int face = 0; face != ptexFaces; ++face)
        ok = writer->writeConstantFace(face, Ptex::FaceInfo(resolution, adjacentFaces,
                                                             adjacentEdges), &ownerValue) && ok;
    ok = writer->close(error) && ok;
    writer->release();
    if (!ok) std::printf("cannot write owner map: %s\n", error.c_str());
    return ok;
}

UsdGenSurfaceDesc Surface(bool ngon, float scale = 1.0f)
{
    UsdGenSurfaceDesc surface;
    if (!ngon) {
        surface.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(scale, 0, 0),
                                           usdGen::UsdGenVec3f(scale, scale, 0), usdGen::UsdGenVec3f(0, scale, 0)};
        surface.faceVertexCounts = VtIntArray{4};
        surface.faceVertexIndices = VtIntArray{0, 1, 2, 3};
    } else {
        surface.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0.50f * scale, 0.00f, 0),
                                           usdGen::UsdGenVec3f(1.00f * scale, 0.35f * scale, 0),
                                           usdGen::UsdGenVec3f(0.80f * scale, 1.00f * scale, 0),
                                           usdGen::UsdGenVec3f(0.20f * scale, 1.00f * scale, 0),
                                           usdGen::UsdGenVec3f(0.00f, 0.35f * scale, 0)};
        surface.faceVertexCounts = VtIntArray{5};
        surface.faceVertexIndices = VtIntArray{0, 1, 2, 3, 4};
    }
    return surface;
}

UsdGenSurfaceCageInput Input(UsdGenSurfaceDesc surface,
                              std::shared_ptr<const UsdGenPtexTexture> texture,
                              float rootScale = 1.0f)
{
    UsdGenSurfaceCageInput input;
    input.surface = surface;
    input.densityMultiplier = 1.0f;
    input.ownerMap.texture = std::move(texture);
    input.ownerMap.options.filter = "nearest";
    input.ownerMap.options.firstChannel = 0;
    input.ownerMap.options.channelCount = 1;
    input.ownerMap.channel = 0;

    UsdGenSurfaceCageOwner owner;
    owner.ownerTubeId = 41;
    owner.density = 30.0f / (rootScale * rootScale);
    owner.seed = 73;
    owner.cvCount = 4;
    owner.defaultWidth = 0.2f;
    owner.edgeBias = 0.0f;
    owner.chartCentroid = usdGen::UsdGenVec2f(0.0f, 0.0f);
    owner.chartMeanRadius = rootScale;
    input.owners.push_back(owner);
    input.ownerLengthProfileOffsets = VtIntArray{0, 2};
    // Constant half-length proves sampling follows authored normalized t,
    // not an arc-length reconstruction.
    input.ownerLengthProfile = usdGen::UsdGenVec2fArray{usdGen::UsdGenVec2f(0.0f, 0.5f),
                                             usdGen::UsdGenVec2f(1.0f, 0.5f)};
    for (int guide = 0; guide != 3; ++guide) {
        UsdGenSurfaceCageGuide rail;
        rail.ownerTubeId = 41;
        rail.curveId = uint64_t(100 + guide);
        // Keep the authored rail in the same units as its rest binding.  The
        // scaled case is specifically checking geometric barycentrics, not a
        // mismatch between a centimetre-sized scalp and metre-sized rails.
        float const x = guide == 1 ? rootScale : 0.0f;
        // Rails deliberately begin above the scalp.  A valid runtime output
        // must replace c0 with the sampled rest-surface root, then fade the
        // correction so the terminal shape remains the authored trajectory.
        float const rootX = guide == 1 ? rootScale : 0.0f;
        float const rootY = guide == 2 ? rootScale : 0.0f;
        // Keep every sampled target ring nondegenerate. The prior fixture's
        // third rail matched rail zero above c0, which was harmless to a
        // three-rail blend but invalid for the new whole-ring containment
        // contract. A Z-only separation preserves the y/t assertion below.
        float const zOffset = guide == 2 ? rootScale : 0.0f;
        rail.points = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(rootX, rootY, 2.0f),
                                    usdGen::UsdGenVec3f(x, 4.0f, 1.0f + zOffset),
                                    usdGen::UsdGenVec3f(x, 5.0f, zOffset)};
        rail.rest = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(rootX, rootY, 0.0f),
                                  usdGen::UsdGenVec3f(x, 4.0f, 1.0f + zOffset),
                                  usdGen::UsdGenVec3f(x, 5.0f, zOffset)};
        rail.normalizedT = VtFloatArray{0.0f, 0.25f, 1.0f};
        rail.widths = VtFloatArray{0.1f, 0.2f, 0.3f};
        input.guides.push_back(std::move(rail));
    }
    UsdGenSurfaceCageTriangle triangle;
    triangle.ownerTubeId = 41;
    triangle.guides = {{0, 1, 2}};
    triangle.rootChart = {{usdGen::UsdGenVec2f(-rootScale, 0), usdGen::UsdGenVec2f(rootScale, 0),
                           usdGen::UsdGenVec2f(0, rootScale)}};
    input.triangles.push_back(triangle);
    return input;
}

bool InsidePolygon(usdGen::UsdGenVec2f const &point, std::vector<usdGen::UsdGenVec2f> const &polygon)
{
    bool inside = false;
    for (size_t i = 0, previous = polygon.size() - 1; i != polygon.size(); previous = i++) {
        usdGen::UsdGenVec2f const &a = polygon[previous];
        usdGen::UsdGenVec2f const &b = polygon[i];
        float const cross = (b[0] - a[0]) * (point[1] - a[1]) -
                            (b[1] - a[1]) * (point[0] - a[0]);
        float const dot = (point[0] - a[0]) * (point[0] - b[0]) +
                          (point[1] - a[1]) * (point[1] - b[1]);
        if (std::abs(cross) <= 1.0e-5f && dot <= 1.0e-5f) return true;
        bool const crosses = (a[1] > point[1]) != (b[1] > point[1]);
        if (crosses && point[0] < (b[0] - a[0]) * (point[1] - a[1]) /
                                     (b[1] - a[1]) + a[0])
            inside = !inside;
    }
    return inside;
}

UsdGenSurfaceCageInput ConcaveInput(UsdGenSurfaceDesc surface,
                                    std::shared_ptr<const UsdGenPtexTexture> texture)
{
    UsdGenSurfaceCageInput input;
    input.surface = surface;
    input.ownerMap.texture = std::move(texture);
    input.ownerMap.options.filter = "nearest";
    input.ownerMap.options.firstChannel = 0;
    input.ownerMap.options.channelCount = 1;
    input.ownerMap.channel = 0;
    input.densityMultiplier = 1.0f;
    input.owners.push_back({41, 90.0f, 37, 3, 0.1f, 0.0f, usdGen::UsdGenVec2f(0.5f), 1.0f});
    input.ownerLengthProfileOffsets = VtIntArray{0, 2};
    input.ownerLengthProfile = usdGen::UsdGenVec2fArray{usdGen::UsdGenVec2f(0.0f, 1.0f), usdGen::UsdGenVec2f(1.0f, 1.0f)};
    std::array<usdGen::UsdGenVec2f, 4> const root{{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 0),
                                         usdGen::UsdGenVec2f(1, 1), usdGen::UsdGenVec2f(0, 1)}};
    std::array<usdGen::UsdGenVec2f, 4> const terminal{{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 0),
                                             usdGen::UsdGenVec2f(.2f, .2f), usdGen::UsdGenVec2f(0, 1)}};
    for (int slot = 0; slot != 4; ++slot) {
        UsdGenSurfaceCageGuide guide;
        guide.ownerTubeId = 41; guide.curveId = uint64_t(200 + slot);
        guide.points = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(root[size_t(slot)][0], root[size_t(slot)][1], 0),
                                    usdGen::UsdGenVec3f(terminal[size_t(slot)][0], terminal[size_t(slot)][1], 1)};
        guide.rest = guide.points;
        guide.normalizedT = VtFloatArray{0.0f, 1.0f};
        guide.widths = VtFloatArray{0.1f, 0.1f};
        input.guides.push_back(std::move(guide));
    }
    input.triangles = {
        {41, {{0, 1, 2}}, {{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 0), usdGen::UsdGenVec2f(1, 1)}}},
        {41, {{0, 2, 3}}, {{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 1), usdGen::UsdGenVec2f(0, 1)}}}};
    return input;
}

void CheckConcaveContainment(std::shared_ptr<const UsdGenPtexTexture> const &texture)
{
    UsdGenSurfaceCageInput input = ConcaveInput(Surface(false), texture);
    UsdGenSurfaceCageResult output;
    std::string error;
    bool const ok = UsdGenInterpolateSurfaceCage(input, &output, &error);
    std::vector<usdGen::UsdGenVec2f> const concave{{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(1, 0),
                                         usdGen::UsdGenVec2f(.2f, .2f), usdGen::UsdGenVec2f(0, 1)}};
    bool terminalsInside = ok && !output.curveVertexCounts.empty();
    for (size_t base = 0, curve = 0; terminalsInside && curve != output.curveVertexCounts.size(); ++curve) {
        usdGen::UsdGenVec3f const &tip = output.points[base + size_t(output.curveVertexCounts[curve] - 1)];
        terminalsInside = Near(tip[2], 1.0f) && InsidePolygon(usdGen::UsdGenVec2f(tip[0], tip[1]), concave);
        base += size_t(output.curveVertexCounts[curve]);
    }
    Check(terminalsInside,
          ("concave terminal cage keeps every runtime point inside its material ring: " + error).c_str());

    std::array<int, 3> slots;
    usdGen::UsdGenVec3f weights;
    bool const direct = UsdGenRemapConcaveMaterialPoint(
        concave, usdGen::UsdGenVec2f(0.4f, 0.4f), &slots, &weights, &error);
    usdGen::UsdGenVec2f const remapped = direct
        ? concave[size_t(slots[0])] * weights[0] + concave[size_t(slots[1])] * weights[1] +
              concave[size_t(slots[2])] * weights[2]
        : usdGen::UsdGenVec2f(0.0f);
    Check(direct && InsidePolygon(remapped, concave),
          ("concave canonical transfer contains the root-square 0.4,0.4 witness: " + error).c_str());

    // Thin K14 material cells remain valid when their double-precision area
    // is nonzero. This used to be rejected by a percentage-of-bounds cutoff.
    std::vector<usdGen::UsdGenVec2f> const thin{{usdGen::UsdGenVec2f(0, 0), usdGen::UsdGenVec2f(.01f, 0),
                                     usdGen::UsdGenVec2f(.000001f, .000000001f)}};
    std::vector<std::array<int, 3>> thinTriangles;
    bool const thinOk = UsdGenTriangulateConcaveMaterialSlots(thin, &thinTriangles, &error);
    Check(thinOk && thinTriangles.size() == 1,
          ("thin nonzero material triangle is not lost to a scale cutoff: " + error).c_str());
}

void CheckWorldMetricBridge(std::string const &mapFile)
{
    // A non-uniform scalp transform must affect world-space density and leave
    // the transient in world space.  This catches a tempting but incorrect
    // surface-local bridge implementation that disagrees with Fill.
    UsdGenSurfaceCageInput const input = Input(Surface(false), nullptr);
    usdGen::UsdGenMatrix4d world(1.0);
    world[0][0] = 3.0; world[1][1] = 2.0;
    world[3][0] = 7.0; world[3][1] = -1.0;

    UsdGenGraphDesc desc;
    desc.defaultWidth = 0.2f;
    UsdGenSurfaceDesc surface;
    static_cast<UsdGenSurfaceData &>(surface) = input.surface;
    surface.path = SdfPath("/Scalp");
    surface.worldMatrix = world;
    desc.surfaces.push_back(surface);
    UsdGenMapDesc map;
    map.path = SdfPath("/RegionMap");
    map.type = TfToken("UsdGenPtexMap");
    map.resolvedAssetPath = mapFile;
    map.params = {
        {TfToken("map:filter"), VtValue(TfToken("nearest")), false},
        {TfToken("map:firstChannel"), VtValue(0), false},
        {TfToken("map:channelCount"), VtValue(1), false},
        {TfToken("map:blur"), VtValue(0.0f), false},
        {TfToken("map:textureGeneration"), VtValue(uint64_t(91)), false}};
    desc.maps.push_back(map);

    UsdGenNodeDesc node;
    node.path = SdfPath("/Source");
    node.surfaces = {surface.path};
    node.maps = {map.path};
    node.mapBindings = {{map.path, TfToken("usdGen:regionMap")}};
    node.params = {{TfToken("expectMapGeneration"), VtValue(uint64_t(91)), false},
                   {TfToken("densityMultiplier"), VtValue(1.0f), false},
                   {TfToken("regionMapChannel"), VtValue(0), false}};

    UsdGenCurveSetDesc sparse;
    sparse.path = SdfPath("/Sparse");
    sparse.role = UsdGenLane::Reference;
    sparse.curveRole = TfToken("guide");
    sparse.worldMatrix = world;
    auto payload = std::make_shared<UsdGenSurfaceCagePayload>();
    payload->ownerIds = VtIntArray{41};
    payload->ownerDensities = VtFloatArray{30.0f};
    payload->ownerSeeds = VtIntArray{73};
    payload->ownerCvCounts = VtIntArray{4};
    payload->ownerEdgeBias = VtFloatArray{0.0f};
    payload->ownerChartCentroids = usdGen::UsdGenVec2fArray{usdGen::UsdGenVec2f(0.0f)};
    payload->ownerChartMeanRadii = VtFloatArray{1.0f};
    payload->ownerLengthProfileOffsets = VtIntArray{0, 2};
    payload->ownerLengthProfile = usdGen::UsdGenVec2fArray{usdGen::UsdGenVec2f(0.0f, 0.5f), usdGen::UsdGenVec2f(1.0f, 0.5f)};
    payload->triangles = usdGen::UsdGenVec3iArray{usdGen::UsdGenVec3i(0, 1, 2)};
    payload->triangleOwnerIndices = VtIntArray{0};
    payload->triangleRootCharts = {usdGen::UsdGenVec2f(-1.0f, 0.0f), usdGen::UsdGenVec2f(1.0f, 0.0f),
                                   usdGen::UsdGenVec2f(0.0f, 1.0f)};
    for (UsdGenSurfaceCageGuide const &guide : input.guides) {
        sparse.curveVertexCounts.push_back(int(guide.points.size()));
        sparse.points.insert(sparse.points.end(), guide.points.begin(), guide.points.end());
        sparse.rest.insert(sparse.rest.end(), guide.rest.begin(), guide.rest.end());
        payload->normalizedT.insert(payload->normalizedT.end(), guide.normalizedT.begin(),
                                    guide.normalizedT.end());
    }
    UsdGenAuthoredPlaneDesc tubeIds;
    tubeIds.name = TfToken("tubeId"); tubeIds.type = UsdGenAuthoredPlaneType::Int32;
    tubeIds.domain = UsdGenAuthoredPlaneDomain::Primitive; tubeIds.arity = 1;
    tubeIds.intValues = VtIntArray{41, 41, 41};
    sparse.authoredPlanes.push_back(std::move(tubeIds));
    UsdGenAuthoredPlaneDesc regionIds;
    regionIds.name = TfToken("regionId"); regionIds.type = UsdGenAuthoredPlaneType::Int32;
    regionIds.domain = UsdGenAuthoredPlaneDomain::Primitive; regionIds.arity = 1;
    regionIds.intValues = VtIntArray{8, 8, 8};
    sparse.authoredPlanes.push_back(std::move(regionIds));
    UsdGenAuthoredPlaneDesc hierarchy;
    hierarchy.name = TfToken("hierarchyLevel"); hierarchy.type = UsdGenAuthoredPlaneType::Int32;
    hierarchy.domain = UsdGenAuthoredPlaneDomain::Primitive; hierarchy.arity = 1;
    hierarchy.intValues = VtIntArray{2, 2, 2};
    sparse.authoredPlanes.push_back(std::move(hierarchy));
    sparse.surfaceCage = std::move(payload);

    uint64_t curves = 0, cvs = 0;
    std::string error;
    bool const estimated = UsdGenEstimateSurfaceCageCurveSet(usdGenUsd::Lower(desc), usdGenUsd::Lower(node), usdGenUsd::Lower(sparse),
                                                               &curves, &cvs, &error);
    Check(estimated && curves == 180 && cvs == 720,
          ("bridge measures transformed scalp area in world space: " + error).c_str());
    UsdGenCurveSetData dense;
    bool const built = UsdGenBuildSurfaceCageCurveSet(usdGenUsd::Lower(desc), usdGenUsd::Lower(node), usdGenUsd::Lower(sparse), &dense, &error);
    bool worldRoots = built && !dense.points.empty() && dense.worldMatrix == usdGen::UsdGenMatrix4d(1.0);
    for (size_t begin = 0, curve = 0; worldRoots && curve != dense.curveVertexCounts.size(); ++curve) {
        usdGen::UsdGenVec3f const &root = dense.points[begin];
        worldRoots = root[0] >= 7.0f && root[0] <= 10.0f &&
                     root[1] >= -1.0f && root[1] <= 1.0f && Near(root[2], 0.0f);
        begin += size_t(dense.curveVertexCounts[curve]);
    }
    Check(worldRoots, ("bridge emits world-space roots for transformed scalp: " + error).c_str());
    auto hasOwnerPlane = [&](char const *name, int expected) {
        auto const plane = std::find_if(dense.authoredPlanes.begin(), dense.authoredPlanes.end(),
            [&](UsdGenAuthoredPlaneDesc const &candidate) { return candidate.name == TfToken(name); });
        return plane != dense.authoredPlanes.end() &&
            plane->type == UsdGenAuthoredPlaneType::Int32 &&
            plane->domain == UsdGenAuthoredPlaneDomain::Primitive && plane->arity == 1 &&
            plane->intValues.size() == dense.curveVertexCounts.size() &&
            std::all_of(plane->intValues.begin(), plane->intValues.end(),
                        [&](int value) { return value == expected; });
    };
    Check(built && hasOwnerPlane("regionId", 8) && hasOwnerPlane("hierarchyLevel", 2),
          ("bridge preserves owner-constant region and hierarchy metadata: " + error).c_str());
}

void CheckOutput(char const *label, UsdGenSurfaceCageInput const &input,
                 UsdGenSurfaceCageResult const &out)
{
    Check(!out.curveVertexCounts.empty(), (std::string(label) + " scatters owner roots").c_str());
    Check(out.curveVertexCounts.size() == out.curveId.size() &&
              out.curveId.size() == out.rootPrim.size() &&
              out.rootPrim.size() == out.rootUV.size() &&
              out.rootUV.size() == out.ownerTubeId.size(),
          (std::string(label) + " has complete uniform root metadata").c_str());
    bool allOwned = true, anchored = true, commonCount = true, normalisedHairT = true;
    bool triangleBaryFollowsRoot = true;
    size_t base = 0;
    for (size_t curve = 0; curve != out.curveVertexCounts.size(); ++curve) {
        int const count = out.curveVertexCounts[curve];
        allOwned = allOwned && out.ownerTubeId[curve] == 41;
        commonCount = commonCount && count == 4;
        anchored = anchored && base < out.points.size() && base < out.rest.size() &&
            Near(out.points[base], out.rest[base]) && Near(out.points[base][2], 0.0f);
        for (int cv = 0; cv != count; ++cv)
            normalisedHairT = normalisedHairT &&
                Near(out.hairT[base + size_t(cv)], float(cv) / float(count - 1));
        // The three root rails form (0,0), (scale,0), (0,scale), and their x
        // rail values remain those coordinates.  With zero edge bias, the final
        // generated x must therefore equal the containing triangle's root x;
        // a random triangle/barycentric assignment cannot satisfy this.
        triangleBaryFollowsRoot = triangleBaryFollowsRoot &&
            Near(out.points[base + size_t(count - 1)][0], out.points[base][0], 3.0e-4f);
        base += size_t(count);
    }
    Check(allOwned, (std::string(label) + " rejects cross-owner interpolation").c_str());
    Check(anchored, (std::string(label) + " anchors each C3 root to rest scalp").c_str());
    Check(commonCount && normalisedHairT,
          (std::string(label) + " emits normalized common-CV transient C3").c_str());
    Check(triangleBaryFollowsRoot,
          (std::string(label) + " uses the sampled root's containing triangle barycentrics").c_str());
    // t=.5 lies between sparse t=.25 (y=4) and t=1 (y=5), so the output
    // must be 4 1/3.  An arc-length resample would not produce this value.
    Check(base >= 4 && Near(out.points[3][1], 4.3333335f, 2.0e-4f),
          (std::string(label) + " samples sparse rails by normalized t, not arc length").c_str());
}

void RunCase(char const *label, bool ngon, float scale = 1.0f)
{
    std::filesystem::path const path = std::filesystem::temp_directory_path() /
        (std::string("usdgen_surface_cage_") + (ngon ? "ngon" : "quad") + ".ptx");
    int const ptexFaces = ngon ? 5 : 1;
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    if (!WriteOwnerMap(path.string(), ptexFaces, 42.0f)) {
        Check(false, "writes categorical owner map");
        return;
    }
    UsdGenPtexMapOptions options;
    options.filter = "nearest";
    options.firstChannel = 0;
    options.channelCount = 1;
    std::string error;
    std::shared_ptr<const UsdGenPtexTexture> texture =
        UsdGenPtexTexture::Open(path.string(), options, &error);
    Check(texture != nullptr, (std::string(label) + " opens categorical owner map").c_str());
    if (!texture) return;
    UsdGenSurfaceCageInput input = Input(Surface(ngon, scale), texture, scale);
    UsdGenSurfaceCageResult output;
    bool const ok = UsdGenInterpolateSurfaceCage(input, &output, &error);
    Check(ok, (std::string(label) + " accepts matching Ptex owner roots: " + error).c_str());
    if (ok) CheckOutput(label, input, output);
    if (!ngon && scale == 1.0f) {
        CheckWorldMetricBridge(path.string());
        CheckConcaveContainment(texture);
    }

    // A categorical map that does not name this leaf is a successful empty
    // result, never a nearest-owner fallback.
    input.ownerMap.texture.reset();
    texture.reset();
    UsdGenPtexTexture::PurgeCache();
    std::filesystem::remove(path, ignored);
    if (WriteOwnerMap(path.string(), ptexFaces, 0.0f)) {
        texture = UsdGenPtexTexture::Open(path.string(), options, &error);
        input.ownerMap.texture = texture;
        UsdGenSurfaceCageResult rejected;
        bool const rejectedOk = texture && UsdGenInterpolateSurfaceCage(input, &rejected, &error);
        Check(rejectedOk && rejected.curveVertexCounts.empty(),
              (std::string(label) + " rejects empty owner map without fallback").c_str());
    }
    std::filesystem::remove(path, ignored);
}

} // namespace

int main()
{
    RunCase("quad", false);
    RunCase("ngon", true);
    RunCase("small quad", false, 1.0e-2f);
    if (gFailures != 0) {
        std::printf("%d surface-cage interpolation FAILURES\n", gFailures);
        return 1;
    }
    std::printf("surface-cage interpolation contract holds\n");
    return 0;
}
