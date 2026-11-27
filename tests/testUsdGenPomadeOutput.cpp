// testUsdGenPomadeOutput -- T1 Output commit acceptance.
//
// Output is commit-only.  It authors an immutable sparse C3 cage plus its
// categorical owner map, while CurveSource expands the dense K8/K9/K10-like
// result at cook time.  Preview state must not affect either authored asset;
// runtime density must not cause Pomade to rebake it.

#include "usdGen/compiler.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/surfaceRootFrames.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"

#include "usdGenPomade/pomadeCommit.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeScalp.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/listOp.h"
#include "pxr/usd/sdf/primSpec.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdGeom/tokens.h"
#include "pxr/usd/usdGeom/xform.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include "utils/pxrInterop.h"

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
using namespace usdGenPomade;
using namespace usdGenImaging;

namespace {

int failures = 0;
void Check(bool ok, std::string const &what)
{
    if (ok) std::printf("ok:   %s\n", what.c_str());
    else { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

bool WaitCommitted(PomadeCommitter &committer, SdfLayerHandle const &live,
                   uint64_t want, int timeoutMs = 5000)
{
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
    do {
        committer.SwapIfIdle(live, /*gestureActive=*/false);
        if (committer.CommittedVersion() >= want) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

struct Grid {
    std::vector<float> points;
    std::vector<int> counts;
    std::vector<int> indices;
};

Grid MakeGrid(int n)
{
    Grid grid;
    for (int x = 0; x <= n; ++x) {
        for (int z = 0; z <= n; ++z) {
            grid.points.insert(grid.points.end(), {float(x), 0.0f, float(z)});
        }
    }
    auto point = [n](int x, int z) { return x * (n + 1) + z; };
    for (int x = 0; x < n; ++x) {
        for (int z = 0; z < n; ++z) {
            grid.counts.push_back(4);
            grid.indices.insert(grid.indices.end(), {
                point(x, z), point(x, z + 1), point(x + 1, z + 1), point(x + 1, z)});
        }
    }
    return grid;
}

PomadeHit Hit(PomadeScalpMesh const &mesh, int n, float x, float z)
{
    PomadeHit hit;
    hit.hit = true;
    int const ix = std::min(std::max(int(std::floor(x)), 0), n - 1);
    int const iz = std::min(std::max(int(std::floor(z)), 0), n - 1);
    hit.faceId = ix * n + iz;
    hit.u = z - float(iz);
    hit.v = x - float(ix);
    if (!PomadeFacePosition(mesh, hit.faceId, hit.u, hit.v,
                           &hit.px, &hit.py, &hit.pz)) {
        hit.hit = false;
        return hit;
    }
    hit.nx = 0.0f; hit.ny = 1.0f; hit.nz = 0.0f;
    return hit;
}

void AddTwoAdjacentRegions(PomadeModel *model, PomadeScalpMesh const &mesh, int n)
{
    float const middle = float(n) * 0.5f;
    float const hi = float(n) - 1.0f;
    int const a0 = model->GraphAddNode(Hit(mesh, n, 0.0f, 1.0f));
    int const a1 = model->GraphAddNode(Hit(mesh, n, middle, 1.0f));
    int const a2 = model->GraphAddNode(Hit(mesh, n, middle, hi));
    int const a3 = model->GraphAddNode(Hit(mesh, n, 0.0f, hi));
    int const b1 = model->GraphAddNode(Hit(mesh, n, float(n), 1.0f));
    int const b2 = model->GraphAddNode(Hit(mesh, n, float(n), hi));
    model->GraphConnect(a0, a1); model->GraphConnect(a1, a2);
    model->GraphConnect(a2, a3); model->GraphConnect(a3, a0);
    model->GraphConnect(a1, b1); model->GraphConnect(b1, b2);
    model->GraphConnect(b2, a2);
}

SdfPath OutputCurvesPath(PomadeCommitPaths const &paths)
{
    return paths.groomPath.AppendChild(TfToken("OutputCurves"));
}

SdfPath OutputDescriptionPath(PomadeCommitPaths const &paths)
{
    return paths.groomPath.AppendChild(TfToken("Output"));
}

SdfPath OutputRegionMapPath(PomadeCommitPaths const &paths)
{
    return paths.groomPath.AppendChild(TfToken("OutputRegionMap"));
}

struct Cooked {
    UsdGenCurveBuffer buffer;
    bool ok = false;
    std::string diagnostic;
};

Cooked CookOutput(UsdStagePtr const &stage, SdfPath const &description)
{
    Cooked result;
    UsdGenGraphDescBuildOptions options;
    UsdGenGraphDesc const desc = BuildGraphDescFromStage(stage, description, options);
    if (!desc.validationErrors.empty()) {
        result.diagnostic = desc.validationErrors.front();
        return result;
    }
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        result.diagnostic = compiled.errors.empty() ? "compile failed" : compiled.errors.front();
        return result;
    }
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Spec();
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    if (run.diagnostics.HasErrors()) {
        result.diagnostic = "scheduler failed";
        for (std::string const &error : run.diagnostics.errors)
            result.diagnostic += ": " + error;
        return result;
    }
    UsdGenNodeId const terminal = usdGenUsd::NodeIdForPath(graph, desc.terminal);
    if (terminal == kUsdGenInvalidNode) {
        result.diagnostic = "missing terminal";
        return result;
    }
    result.buffer = NodeBuffer(graph, terminal);
    result.ok = true;
    return result;
}

UsdStageRefPtr CommitStage(PomadeModel const &model, PomadeCommitPaths const &paths,
                           std::string *err)
{
    PomadeSnapshot snapshot = PomadeSnapshotFromModel(model);
    // The direct helper takes a snapshot rather than a committer session.  A
    // real PomadeCommitter stamps this link as it snapshots its bound stage;
    // give the standalone helper the same authored scalp identity so hydrate
    // can reconstruct the graph after the layer is exported.
    if (snapshot.scalpPath.IsEmpty()) snapshot.scalpPath = paths.scalpPath;
    if (snapshot.scalpMeshPath.IsEmpty()) snapshot.scalpMeshPath = paths.scalpMeshPath;
    SdfLayerRefPtr layer;
    if (!PomadeBuildCommitLayer(snapshot, paths, &layer, err) || !layer) {
        return nullptr;
    }
    UsdStageRefPtr stage = UsdStage::Open(layer);
    // A commit owns world-space Output points.  Put the groom below a real
    // inherited transform: both OutputCurves and Output reset their xform
    // stack, so this parent cannot move or double-transform the engine input.
    SdfPath const groomParent = paths.groomPath.GetParentPath();
    if (stage && groomParent.IsPrimPath() &&
        groomParent != SdfPath::AbsoluteRootPath()) {
        UsdGeomXform parent = UsdGeomXform::Define(stage, groomParent);
        parent.AddTranslateOp().Set(GfVec3d(13.0, -5.0, 7.0));
    }
    // PomadeBuildCommitLayer deliberately authors only an over for its scalp
    // link.  This standalone commit-layer test supplies the backing mesh so
    // the real stage builder can resolve Output's CurveSource surface.  A
    // face GeomSubset link (plan/02 §2.20) gets its parent mesh plus the
    // subset naming the model's faces.
    if (stage && snapshot.hasScalp && !paths.scalpPath.IsEmpty()) {
        UsdGeomMesh mesh = UsdGeomMesh::Define(stage, paths.ScalpMeshPath());
        usdGen::UsdGenVec3fArray points(snapshot.scalp.points.size() / 3);
        for (size_t i = 0; i < points.size(); ++i) {
            points[i] = usdGen::UsdGenVec3f(snapshot.scalp.points[i * 3],
                                snapshot.scalp.points[i * 3 + 1],
                                snapshot.scalp.points[i * 3 + 2]);
        }
        mesh.GetPointsAttr().Set(ToHost(points));
        mesh.GetFaceVertexCountsAttr().Set(
            VtIntArray(snapshot.scalp.faceVertexCounts.begin(),
                       snapshot.scalp.faceVertexCounts.end()));
        mesh.GetFaceVertexIndicesAttr().Set(
            VtIntArray(snapshot.scalp.faceVertexIndices.begin(),
                       snapshot.scalp.faceVertexIndices.end()));
        if (paths.ScalpMeshPath() != paths.scalpPath) {
            UsdGeomSubset subset = UsdGeomSubset::Define(stage, paths.scalpPath);
            subset.GetElementTypeAttr().Set(UsdGeomTokens->face);
            subset.GetIndicesAttr().Set(VtIntArray(
                snapshot.scalp.activeFaces.begin(), snapshot.scalp.activeFaces.end()));
        }
    }
    return stage;
}

UsdGenPlane const *FindCurvePlane(UsdGenCurveBuffer const &buffer, char const *name)
{
    TfToken const wanted(name);
    for (UsdGenPlane const &plane : buffer.extraCurve)
        if (plane.name == wanted) return &plane;
    return nullptr;
}

bool CurveSpan(UsdGenCurveBuffer const &buffer, int curve, int *begin, int *end)
{
    if (curve < 0 || size_t(curve) >= buffer.totalCurves) return false;
    if (!buffer.cvOffsets.empty()) {
        if (buffer.cvOffsets.size() != size_t(buffer.totalCurves) + 1) return false;
        *begin = buffer.cvOffsets[size_t(curve)];
        *end = buffer.cvOffsets[size_t(curve + 1)];
    } else {
        if (buffer.totalCurves == 0 || buffer.totalCvs % buffer.totalCurves) return false;
        int const width = int(buffer.totalCvs / buffer.totalCurves);
        *begin = curve * width;
        *end = *begin + width;
    }
    return *begin >= 0 && *end >= *begin && size_t(*end) <= buffer.totalCvs;
}

// OutputCurves is a committed C3 cage, not a serialized copy of the dense
// runtime stream. Capture every topology/owner array which CurveSource uses
// so a direct runtime density edit cannot silently trigger a Pomade rebake.
struct CagePayload {
    VtVec3fArray points;
    VtIntArray counts;
    std::vector<VtValue> fields;
};

bool ReadCagePayload(UsdPrim const &curves, CagePayload *out)
{
    if (!out || !curves ||
        !curves.GetAttribute(TfToken("points")).Get(&out->points) ||
        !curves.GetAttribute(TfToken("curveVertexCounts")).Get(&out->counts) ||
        out->points.empty() || out->counts.empty()) return false;
    static char const *const names[] = {
        "usdGen:surfaceCage:ownerIds",
        "usdGen:surfaceCage:ownerDensities",
        "usdGen:surfaceCage:ownerSeeds",
        "usdGen:surfaceCage:ownerCvCounts",
        "usdGen:surfaceCage:ownerEdgeBias",
        "usdGen:surfaceCage:ownerChartCentroids",
        "usdGen:surfaceCage:ownerChartMeanRadii",
        "usdGen:surfaceCage:ownerLengthProfileOffsets",
        "usdGen:surfaceCage:ownerLengthProfile",
        "usdGen:surfaceCage:triangles",
        "usdGen:surfaceCage:triangleOwnerIndices",
        "usdGen:surfaceCage:triangleRootCharts",
        "usdGen:surfaceCage:normalizedT"};
    out->fields.clear();
    out->fields.reserve(sizeof(names) / sizeof(names[0]));
    for (char const *name : names) {
        VtValue value;
        UsdAttribute const attr = curves.GetAttribute(TfToken(name));
        if (!attr || !attr.Get(&value) || value.IsEmpty()) return false;
        out->fields.push_back(std::move(value));
    }
    return true;
}

bool SameCagePayload(CagePayload const &a, CagePayload const &b)
{
    return a.points == b.points && a.counts == b.counts && a.fields == b.fields;
}

struct OutputMapPayload {
    SdfAssetPath asset;
    uint64_t generation = 0;
};

bool ReadOutputMapPayload(UsdPrim const &map, OutputMapPayload *out)
{
    if (!out || !map) return false;
    bool owned = false;
    return map.GetAttribute(TfToken("usdGen:pomade:outputOwned")).Get(&owned) &&
        owned &&
        map.GetAttribute(TfToken("usdGen:map:file")).Get(&out->asset) &&
        !out->asset.GetAssetPath().empty() &&
        map.GetAttribute(TfToken("usdGen:map:textureGeneration")).Get(
            &out->generation) &&
        out->generation != 0;
}

bool ExistingOutputMapAsset(OutputMapPayload const &map)
{
    std::error_code error;
    std::filesystem::path path(map.asset.GetResolvedPath());
    if (path.empty()) path = map.asset.GetAssetPath();
    return !path.empty() && std::filesystem::is_regular_file(path, error);
}

bool SurfaceCageSourceContract(UsdPrim const &source, SdfPath const &mapPath,
                               uint64_t generation)
{
    if (!source || generation == 0) return false;
    TfToken mode;
    int channel = -1;
    uint64_t expected = 0;
    SdfPathVector targets;
    return source.GetAttribute(TfToken("usdGen:interpolationMode")).Get(&mode) &&
        mode == TfToken("surfaceCage") &&
        source.GetAttribute(TfToken("usdGen:regionMapChannel")).Get(&channel) &&
        channel == 0 &&
        source.GetAttribute(TfToken("usdGen:expectMapGeneration")).Get(&expected) &&
        expected == generation &&
        source.GetRelationship(TfToken("usdGen:regionMap")).GetTargets(&targets) &&
        targets == SdfPathVector{mapPath};
}

std::string CageTriangleDiagnostics(UsdPrim const &curves)
{
    VtVec3iArray triangles;
    VtVec2fArray charts;
    VtIntArray triangleOwners, owners;
    if (!curves.GetAttribute(TfToken("usdGen:surfaceCage:triangles")).Get(
            &triangles) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:triangleRootCharts")).Get(
            &charts) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:triangleOwnerIndices")).Get(
            &triangleOwners) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:ownerIds")).Get(&owners) ||
        charts.size() != triangles.size() * 3 ||
        triangleOwners.size() != triangles.size()) {
        return " [cage diagnostic unavailable]";
    }
    for (size_t index = 0; index < triangles.size(); ++index) {
        usdGen::UsdGenVec2f const a = charts[index * 3], b = charts[index * 3 + 1],
                      c = charts[index * 3 + 2];
        float const area2 = (b[0] - a[0]) * (c[1] - a[1]) -
                            (b[1] - a[1]) * (c[0] - a[0]);
        if (std::abs(area2) > 1.0e-7f) continue;
        int const ownerIndex = triangleOwners[index];
        int const owner = ownerIndex >= 0 && size_t(ownerIndex) < owners.size()
            ? owners[size_t(ownerIndex)] : -1;
        char detail[512];
        std::snprintf(detail, sizeof(detail),
                      " [degenerate triangle=%zu slots=(%d,%d,%d) ownerIndex=%d owner=%d "
                      "charts=((%.7g,%.7g),(%.7g,%.7g),(%.7g,%.7g)) area2=%.7g]",
                      index, triangles[index][0], triangles[index][1], triangles[index][2],
                      ownerIndex, owner, a[0], a[1], b[0], b[1], c[0], c[1], area2);
        return detail;
    }
    return " [no degenerate triangle charts]";
}

// K10 serializes its root frame row-major.  C3 names row 0 rootT and row 2
// rootN; the former is the scalp tangent/radial normal while the latter is
// the tube shaft/scalp normal.  Both direction and origin must survive the
// conversion, even though the middle axis is Gram-Schmidt repaired.
bool OutputFrameContract(UsdStagePtr const &stage, SdfPath const &description)
{
    UsdGenGraphDescBuildOptions options;
    UsdGenGraphDesc const desc = BuildGraphDescFromStage(stage, description, options);
    if (!desc.validationErrors.empty()) return false;
    size_t curveCount = 0;
    for (UsdGenCurveSetDesc const &set : desc.curveSets) {
        if (set.rootFrame.size() != set.curveVertexCounts.size() ||
            set.curveId.size() != set.curveVertexCounts.size()) return false;
        for (size_t curve = 0; curve < set.rootFrame.size(); ++curve) {
            usdGen::UsdGenMatrix4d const &frame = set.rootFrame[curve];
            if (!UsdGenValidateAuthoredRootFrame(frame)) return false;
            ++curveCount;
        }
    }
    return curveCount > 0;
}

bool OwnersMatchKnownLeaves(PomadeSnapshotGuides const &guides,
                            UsdGenCurveBuffer const &buffer)
{
    struct Expected { char const *name; std::vector<int> const *values; };
    Expected const fields[] = {{"tubeId", &guides.tubeIds},
                               {"regionId", &guides.regionIds},
                               {"hierarchyLevel", &guides.levels}};
    if (guides.tubeIds.size() != guides.regionIds.size() ||
        guides.tubeIds.size() != guides.levels.size() ||
        buffer.curveId.size() != buffer.totalCurves) return false;
    std::map<int, std::pair<int, int>> owners;
    for (size_t i = 0; i < guides.tubeIds.size(); ++i) {
        std::pair<int, int> const value = {guides.regionIds[i], guides.levels[i]};
        auto const found = owners.emplace(guides.tubeIds[i], value);
        if (!found.second && found.first->second != value) return false;
    }
    UsdGenPlane const *tubeIds = FindCurvePlane(buffer, "tubeId");
    UsdGenPlane const *regionIds = FindCurvePlane(buffer, "regionId");
    UsdGenPlane const *levels = FindCurvePlane(buffer, "hierarchyLevel");
    if (!tubeIds || !regionIds || !levels ||
        tubeIds->i.size() != buffer.curveId.size() ||
        regionIds->i.size() != buffer.curveId.size() ||
        levels->i.size() != buffer.curveId.size()) return false;
    for (Expected const &expected : fields) {
        UsdGenPlane const *plane = FindCurvePlane(buffer, expected.name);
        if (!plane || plane->interpolation != TfToken("uniform") ||
            plane->type != TfToken("int") || plane->arity != 1 ||
            plane->i.size() != buffer.curveId.size()) return false;
    }
    for (size_t i = 0; i < tubeIds->i.size(); ++i) {
        auto const found = owners.find(tubeIds->i[i]);
        if (found == owners.end() || regionIds->i[i] != found->second.first ||
            levels->i[i] != found->second.second) return false;
    }
    return true;
}

bool HasOnlyLeafOwners(PomadeSnapshotGuides const &guides,
                       UsdGenCurveBuffer const &buffer)
{
    UsdGenPlane const *owners = FindCurvePlane(buffer, "tubeId");
    if (!owners || owners->i.size() != buffer.curveId.size()) return false;
    std::vector<int> allowed = guides.tubeIds;
    std::sort(allowed.begin(), allowed.end());
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return std::all_of(owners->i.begin(), owners->i.end(), [&](int id) {
        return std::binary_search(allowed.begin(), allowed.end(), id);
    });
}

bool SameCurveForId(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b,
                    uint64_t id)
{
    auto find = [id](UsdGenCurveBuffer const &buffer) {
        for (size_t i = 0; i < buffer.curveId.size(); ++i)
            if (buffer.curveId[i] == id) return int(i);
        return -1;
    };
    int const ai = find(a), bi = find(b);
    if (ai < 0 || bi < 0) return false;
    auto span = [](UsdGenCurveBuffer const &buffer, int curve, int *begin, int *end) {
        if (!buffer.cvOffsets.empty()) {
            *begin = buffer.cvOffsets[size_t(curve)];
            *end = buffer.cvOffsets[size_t(curve + 1)];
        } else {
            int const width = int(buffer.totalCvs / buffer.totalCurves);
            *begin = curve * width; *end = *begin + width;
        }
    };
    int ab = 0, ae = 0, bb = 0, be = 0;
    span(a, ai, &ab, &ae); span(b, bi, &bb, &be);
    if (ae - ab != be - bb) return false;
    for (int i = 0; i < ae - ab; ++i)
        if (a.px[size_t(ab + i)] != b.px[size_t(bb + i)] ||
            a.py[size_t(ab + i)] != b.py[size_t(bb + i)] ||
            a.pz[size_t(ab + i)] != b.pz[size_t(bb + i)]) return false;
    return true;
}

// A direct USD edit of the cage is deliberately stronger than a count or
// bounds check.  CurveSource must retain each sampled scalp root while the
// terminal rail position follows the authored affine cage edit exactly.
bool CageAffineTerminalResponse(UsdGenCurveBuffer const &before,
                                 UsdGenCurveBuffer const &after,
                                 std::string *diagnostic = nullptr)
{
    auto fail = [&](std::string const &why) {
        if (diagnostic) *diagnostic = why;
        return false;
    };
    if (before.curveId.empty() || before.curveId.size() != after.curveId.size()) {
        return fail("curve-id cardinality changed");
    }
    auto find = [](UsdGenCurveBuffer const &buffer, uint64_t id) {
        for (size_t i = 0; i < buffer.curveId.size(); ++i)
            if (buffer.curveId[i] == id) return int(i);
        return -1;
    };
    for (size_t i = 0; i < before.curveId.size(); ++i) {
        int const other = find(after, before.curveId[i]);
        int begin = 0, end = 0, changedBegin = 0, changedEnd = 0;
        if (other < 0 || !CurveSpan(before, int(i), &begin, &end) ||
            !CurveSpan(after, other, &changedBegin, &changedEnd) ||
            end - begin < 2 || changedEnd - changedBegin != end - begin) {
            return fail("stable curve id or span changed");
        }
        auto almostEqual = [](float a, float b) {
            return std::abs(a - b) <= 2.0e-5f;
        };
        // c0 is categorically sampled from the rest surface and must not
        // follow a rail-only edit.
        if (!almostEqual(before.px[size_t(begin)], after.px[size_t(changedBegin)]) ||
            !almostEqual(before.py[size_t(begin)], after.py[size_t(changedBegin)]) ||
            !almostEqual(before.pz[size_t(begin)], after.pz[size_t(changedBegin)])) {
            char message[256];
            std::snprintf(message, sizeof(message),
                          "root moved id=%llu old=(%.7g,%.7g,%.7g) new=(%.7g,%.7g,%.7g)",
                          static_cast<unsigned long long>(before.curveId[i]),
                          before.px[size_t(begin)], before.py[size_t(begin)],
                          before.pz[size_t(begin)], after.px[size_t(changedBegin)],
                          after.py[size_t(changedBegin)], after.pz[size_t(changedBegin)]);
            return fail(message);
        }
        int const tip = end - 1;
        int const changedTip = changedEnd - 1;
        if (!almostEqual(after.px[size_t(changedTip)],
                         1.25f * before.px[size_t(tip)] + 0.03125f) ||
            !almostEqual(after.py[size_t(changedTip)], before.py[size_t(tip)]) ||
            !almostEqual(after.pz[size_t(changedTip)],
                         0.75f * before.pz[size_t(tip)] - 0.0175f)) {
            char message[320];
            std::snprintf(message, sizeof(message),
                          "tip affine mismatch id=%llu old=(%.7g,%.7g,%.7g) new=(%.7g,%.7g,%.7g) "
                          "want=(%.7g,%.7g,%.7g)",
                          static_cast<unsigned long long>(before.curveId[i]),
                          before.px[size_t(tip)], before.py[size_t(tip)], before.pz[size_t(tip)],
                          after.px[size_t(changedTip)], after.py[size_t(changedTip)],
                          after.pz[size_t(changedTip)],
                          1.25f * before.px[size_t(tip)] + 0.03125f,
                          before.py[size_t(tip)],
                          0.75f * before.pz[size_t(tip)] - 0.0175f);
            return fail(message);
        }
    }
    return true;
}

bool BarycentricXZ(usdGen::UsdGenVec3f const &p, usdGen::UsdGenVec3f const &a, usdGen::UsdGenVec3f const &b,
                   usdGen::UsdGenVec3f const &c, std::array<float, 3> *out)
{
    float const abx = b[0] - a[0], abz = b[2] - a[2];
    float const acx = c[0] - a[0], acz = c[2] - a[2];
    float const apx = p[0] - a[0], apz = p[2] - a[2];
    float const determinant = abx * acz - abz * acx;
    if (std::abs(determinant) < 1.0e-7f || !out) return false;
    float const v = (apx * acz - apz * acx) / determinant;
    float const w = (abx * apz - abz * apx) / determinant;
    *out = {{1.0f - v - w, v, w}};
    return true;
}

usdGen::UsdGenVec3f SampleCageRail(usdGen::UsdGenVec3fArray const &points, VtFloatArray const &stations,
                        int begin, int end, float t)
{
    if (t <= stations[size_t(begin)]) return points[size_t(begin)];
    if (t >= stations[size_t(end - 1)]) return points[size_t(end - 1)];
    for (int at = begin + 1; at < end; ++at) {
        if (t > stations[size_t(at)]) continue;
        float const span = stations[size_t(at)] - stations[size_t(at - 1)];
        float const f = span > 0.0f ? (t - stations[size_t(at - 1)]) / span : 0.0f;
        return points[size_t(at - 1)] * (1.0f - f) + points[size_t(at)] * f;
    }
    return points[size_t(end - 1)];
}

std::array<float, 3> CageBiasedWeights(std::array<float, 3> weights,
                                        float edgeBias, float progress)
{
    float const minimum = std::min(weights[0], std::min(weights[1], weights[2]));
    float const rho = std::max(0.0f, 1.0f - 3.0f * minimum);
    std::array<float, 3> biased{{1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f}};
    if (rho > 1.0e-6f) {
        float const ratio = std::pow(rho, 1.0f - 0.5f * edgeBias) / rho;
        for (size_t i = 0; i != 3; ++i)
            biased[i] += (weights[i] - 1.0f / 3.0f) * ratio;
    }
    for (size_t i = 0; i != 3; ++i)
        weights[i] = weights[i] * (1.0f - progress) + biased[i] * progress;
    return weights;
}

// The irregular .31 station below makes authored normalizedT observable at
// a dense intermediate CV.  Recomputing the complete local trajectory from
// the captured cage catches both a uniform-index station bug and an
// unrelated-triangle/weight choice that a terminal-only assertion misses.
bool CageTrajectoryFollowsPhysicalRoot(UsdPrim const &curves,
                                       UsdGenCurveBuffer const &output)
{
    VtVec3fArray rails;
    VtIntArray counts, railOwners, ownerIds, triangleOwners;
    VtFloatArray stations, ownerBias;
    VtVec3iArray triangles;
    if (!curves.GetAttribute(TfToken("points")).Get(&rails) ||
        !curves.GetAttribute(TfToken("curveVertexCounts")).Get(&counts) ||
        !curves.GetAttribute(TfToken("primvars:tubeId")).Get(&railOwners) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:normalizedT")).Get(
            &stations) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:ownerIds")).Get(
            &ownerIds) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:ownerEdgeBias")).Get(
            &ownerBias) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:triangles")).Get(
            &triangles) ||
        !curves.GetAttribute(TfToken("usdGen:surfaceCage:triangleOwnerIndices")).Get(
            &triangleOwners) ||
        counts.size() != railOwners.size() || rails.size() != stations.size() ||
        ownerIds.size() != ownerBias.size() || triangles.size() != triangleOwners.size()) {
        return false;
    }
    std::vector<int> offsets(counts.size() + 1, 0);
    for (size_t rail = 0; rail < counts.size(); ++rail) {
        if (counts[rail] < 2 || offsets[rail] > int(rails.size()) - counts[rail]) {
            return false;
        }
        offsets[rail + 1] = offsets[rail] + counts[rail];
    }
    UsdGenPlane const *owners = FindCurvePlane(output, "tubeId");
    if (!owners || owners->i.size() != output.curveId.size()) return false;
    size_t resolved = 0;
    for (size_t curve = 0; curve < output.curveId.size(); ++curve) {
        int begin = 0, end = 0;
        if (!CurveSpan(output, int(curve), &begin, &end) || end - begin < 3) {
            return false;
        }
        usdGen::UsdGenVec3f const root(output.px[size_t(begin)], output.py[size_t(begin)],
                           output.pz[size_t(begin)]);
        int const owner = owners->i[curve];
        bool found = false;
        for (size_t index = 0; index < triangles.size(); ++index) {
            usdGen::UsdGenVec3i const triangle = triangles[index];
            int const ia = triangle[0], ib = triangle[1], ic = triangle[2];
            int const ownerIndex = triangleOwners[index];
            if (ownerIndex < 0 || size_t(ownerIndex) >= ownerIds.size() ||
                ownerIds[size_t(ownerIndex)] != owner || ia < 0 || ib < 0 || ic < 0 ||
                size_t(ia) >= counts.size() || size_t(ib) >= counts.size() ||
                size_t(ic) >= counts.size() || railOwners[size_t(ia)] != owner ||
                railOwners[size_t(ib)] != owner || railOwners[size_t(ic)] != owner) {
                continue;
            }
            std::array<float, 3> rootWeights;
            if (!BarycentricXZ(root, rails[size_t(offsets[size_t(ia)])],
                               rails[size_t(offsets[size_t(ib)])],
                               rails[size_t(offsets[size_t(ic)])], &rootWeights) ||
                rootWeights[0] < -2.0e-4f || rootWeights[1] < -2.0e-4f ||
                rootWeights[2] < -2.0e-4f) {
                continue;
            }
            usdGen::UsdGenVec3f const rootPoint =
                rails[size_t(offsets[size_t(ia)])] * rootWeights[0] +
                rails[size_t(offsets[size_t(ib)])] * rootWeights[1] +
                rails[size_t(offsets[size_t(ic)])] * rootWeights[2];
            for (int cv = 0; cv < end - begin; ++cv) {
                float const progress = float(cv) / float(end - begin - 1);
                std::array<float, 3> const weights =
                    CageBiasedWeights(rootWeights, ownerBias[size_t(ownerIndex)], progress);
                usdGen::UsdGenVec3f const local =
                    SampleCageRail(ToEngine(rails), stations, offsets[size_t(ia)],
                                   offsets[size_t(ia + 1)], progress) * weights[0] +
                    SampleCageRail(ToEngine(rails), stations, offsets[size_t(ib)],
                                   offsets[size_t(ib + 1)], progress) * weights[1] +
                    SampleCageRail(ToEngine(rails), stations, offsets[size_t(ic)],
                                   offsets[size_t(ic + 1)], progress) * weights[2];
                usdGen::UsdGenVec3f const expected = cv == 0 ? root :
                    local + (root - rootPoint) * (1.0f - progress);
                int const at = begin + cv;
                usdGen::UsdGenVec3f const actual(output.px[size_t(at)], output.py[size_t(at)],
                                     output.pz[size_t(at)]);
                if ((actual - expected).GetLength() > 5.0e-4f) return false;
            }
            found = true;
            ++resolved;
            break;
        }
        if (!found) return false;
    }
    return resolved >= 2;
}

} // namespace

// plan/02 §2.20: Output on a face GeomSubset scalp. The subset drops faces 5
// and 6 (x in [1, 2], z in [1, 3]), which lie inside region A. Output's
// usdGen:surface names the subset, the rest binding sits on the parent Mesh,
// and every cooked root lands on a subset face named by its parent-mesh id;
// the same groom bound whole does grow roots there, so the check has teeth.
void CheckSubsetOutput(Grid const &grid, int n)
{
    std::vector<int> const subsetFaces = {0, 1, 2, 3, 4, 7, 8, 9,
                                          10, 11, 12, 13, 14, 15};
    auto build = [&](PomadeModel *model, std::vector<int> const &active) {
        if (!model->BindScalp(grid.points, grid.counts, grid.indices, active) ||
            !model->GetScalp()) {
            return false;
        }
        AddTwoAdjacentRegions(model, *model->GetScalp(), n);
        PomadeModel::FillParams fill;
        fill.density = 12.0f;
        fill.cvCount = 6;
        fill.seed = 19;
        PomadeModel::OutputSettings output;
        output.enabled = true;
        output.densityMultiplier = 4.0f;
        output.width = 0.02f;
        return model->Rasterise() && model->BuildTubeFromRegion(0, 5, 8, 2.0f) &&
               model->BuildTubeFromRegion(1, 5, 8, 2.0f) &&
               model->L1TubeIds().size() == 2 &&
               model->SetTubeFillParams(model->L1TubeIds()[0], fill) &&
               model->SetTubeFillParams(model->L1TubeIds()[1], fill) &&
               model->SetOutputSettings(output);
    };
    auto rootFaces = [](Cooked const &cooked) {
        std::vector<int> faces(cooked.buffer.rootPrim.begin(),
                               cooked.buffer.rootPrim.end());
        std::sort(faces.begin(), faces.end());
        faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        return faces;
    };
    auto touchesDropped = [](std::vector<int> const &faces) {
        return std::find(faces.begin(), faces.end(), 5) != faces.end() ||
               std::find(faces.begin(), faces.end(), 6) != faces.end();
    };

    PomadeModel whole;
    PomadeModel subset;
    Check(build(&whole, {}) && build(&subset, subsetFaces),
          "Output subset: the same two-region groom binds whole and through a "
          "face subset");
    PomadeCommitPaths wholePaths;
    wholePaths.groomPath = SdfPath("/World/PomadeWholeOutput");
    wholePaths.scalpPath = SdfPath("/Scalp");
    PomadeCommitPaths subsetPaths;
    subsetPaths.groomPath = SdfPath("/World/PomadeSubsetOutput");
    subsetPaths.scalpPath = SdfPath("/Scalp/patch");
    subsetPaths.scalpMeshPath = SdfPath("/Scalp");
    std::string error;
    UsdStageRefPtr wholeStage = CommitStage(whole, wholePaths, &error);
    Cooked const wholeCook = wholeStage
        ? CookOutput(wholeStage, OutputDescriptionPath(wholePaths)) : Cooked{};
    std::vector<int> const wholeFaces = rootFaces(wholeCook);
    Check(wholeCook.ok && wholeCook.buffer.totalCurves > 0 &&
          touchesDropped(wholeFaces),
          "Output subset: bound whole, the groom roots hairs on faces 5/6: " +
          error + wholeCook.diagnostic);

    error.clear();
    UsdStageRefPtr stage = CommitStage(subset, subsetPaths, &error);
    Check(bool(stage), "Output subset: builds the subset commit layer: " + error);
    if (!stage) return;
    SdfPathVector surface;
    stage->GetPrimAtPath(OutputDescriptionPath(subsetPaths))
        .GetRelationship(TfToken("usdGen:surface")).GetTargets(&surface);
    Check(surface == SdfPathVector{subsetPaths.scalpPath},
          "Output subset: Output's usdGen:surface names the subset");
    TfTokenVector applied;
    if (SdfPrimSpecHandle const over =
            stage->GetRootLayer()->GetPrimAtPath(subsetPaths.ScalpMeshPath())) {
        VtValue const value = over->GetInfo(TfToken("apiSchemas"));
        if (value.IsHolding<SdfTokenListOp>()) {
            value.UncheckedGet<SdfTokenListOp>().ApplyOperations(&applied);
        }
    }
    VtIntArray regions;
    Check(std::find(applied.begin(), applied.end(), TfToken("UsdGenRestAPI")) !=
              applied.end() &&
          stage->GetPrimAtPath(subsetPaths.ScalpMeshPath())
              .GetAttribute(TfToken("primvars:usdGen:pomadeRegion")).Get(&regions) &&
          regions.size() == grid.counts.size() && regions[5] == -1 &&
          regions[6] == -1,
          "Output subset: RestAPI and the parent-sized live primvar land on the "
          "parent mesh");
    Cooked const cooked = CookOutput(stage, OutputDescriptionPath(subsetPaths));
    std::vector<int> const faces = rootFaces(cooked);
    bool inSubset = !faces.empty();
    for (int face : faces) {
        inSubset = inSubset && std::binary_search(subsetFaces.begin(),
                                                  subsetFaces.end(), face);
    }
    Check(cooked.ok && cooked.buffer.totalCurves > 0 && inSubset &&
          !touchesDropped(faces),
          "Output subset: every cooked root lands on a subset face, by its "
          "parent-mesh id: " + cooked.diagnostic);
}

int main()
{
    int const n = 4;
    Grid const grid = MakeGrid(n);
    PomadeModel model;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "Output: binds the two-region scalp");
    std::shared_ptr<PomadeScalpMesh const> const scalp = model.GetScalp();
    if (!scalp) return 1;
    AddTwoAdjacentRegions(&model, *scalp, n);
    Check(model.Rasterise() && model.GetRegionLoops().regionIds.size() == 2,
          "Output: extracts two adjacent subface regions");
    Check(model.BuildTubeFromRegion(0, 5, 8, 2.0f) &&
          model.BuildTubeFromRegion(1, 5, 8, 2.0f),
          "Output: creates one L1 root per region");
    std::vector<int> const roots = model.L1TubeIds();
    Check(roots.size() == 2, "Output: owns two independent roots");
    if (roots.size() != 2) return 1;

    PomadeModel::FillParams fill;
    fill.density = 12.0f;
    fill.cvCount = 11;  // includes a dense sample across the authored t=.31 knot
    fill.seed = 19;
    fill.edgeBias = 0.35f;
    fill.lengthProfile = {0.0f, 1.0f, 1.0f, 1.0f};
    Check(model.SetTubeFillParams(roots[0], fill) &&
          model.SetTubeFillParams(roots[1], fill),
          "Output: seeds deterministic full Fill on both roots");
    Check(model.AddTubeSectionRing(roots[1], 0.31f),
          "Output: adds an irregular authored station before child derivation");
    std::vector<int> aChildren, bChildren, aNested, bNested;
    Check(model.SubdivideTube(roots[0], 2, "kmeans", 7, &aChildren) &&
          model.SubdivideTube(roots[1], 2, "kmeans", 11, &bChildren) &&
          aChildren.size() == 2 && bChildren.size() == 2,
          "Output: subdivides both region roots");
    Check(model.SubdivideTube(aChildren[0], 2, "kmeans", 17, &aNested) &&
          model.SubdivideTube(bChildren[0], 2, "kmeans", 23, &bNested) &&
          aNested.size() == 2 && bNested.size() == 2,
          "Output: creates independent nested leaves");
    int irregularRing = -1;
    for (int ring = 0; ring < model.GetTubeSectionCount(bNested[0]); ++ring) {
        PomadeTubeSection section;
        if (model.GetTubeSection(bNested[0], ring, &section) &&
            std::abs(section.t - 0.31f) < 1.0e-6f) {
            irregularRing = ring;
            break;
        }
    }
    int const asymmetricLastRing = model.GetTubeSectionCount(bNested[0]) - 1;
    Check(irregularRing > 0 && asymmetricLastRing > 0 &&
          model.MoveTubeCenterCV(aNested[0], 2, 0.10f, 0.0f, 0.03f) &&
          model.MoveTubeSectionCV(bNested[0], irregularRing, 2, 0.035f, -0.05f) &&
          model.MoveTubeSectionCV(bNested[0], asymmetricLastRing, 1,
                                  -0.06f, 0.04f) &&
          model.ScaleTubeSectionRing(bNested[0], asymmetricLastRing, 1.19f) &&
          model.TwistTubeSectionRing(bNested[0], asymmetricLastRing, 0.23f),
          "Output: sculpts an asymmetric final ring in one adjacent nested leaf");

    PomadeModel::OutputSettings output;
    output.enabled = true;
    output.densityMultiplier = 1.0f;
    output.width = 0.0375f;
    Check(model.SetOutputSettings(output), "Output: enables full-fill output");

    PomadeCommitPaths paths;
    paths.groomPath = SdfPath("/World/PomadeOutput");
    // This is deliberately unrelated to Output: it verifies Output owns its
    // own description rather than mutating a legacy description link.
    paths.descriptionPath = SdfPath("/Legacy/Description");
    paths.scalpPath = SdfPath("/Scalp");
    std::string error;
    UsdStageRefPtr stage = CommitStage(model, paths, &error);
    Check(bool(stage), "Output: builds a committed layer: " + error);
    if (!stage) return 1;

    SdfPath const curvesPath = OutputCurvesPath(paths);
    SdfPath const mapPath = OutputRegionMapPath(paths);
    SdfPath const descPath = OutputDescriptionPath(paths);
    UsdPrim const curves = stage->GetPrimAtPath(curvesPath);
    UsdPrim const outputMap = stage->GetPrimAtPath(mapPath);
    UsdPrim const outputDesc = stage->GetPrimAtPath(descPath);
    Check(bool(curves) && curves.GetTypeName() == TfToken("BasisCurves") &&
          bool(outputMap) && outputMap.GetTypeName() == TfToken("UsdGenPtexMap") &&
          bool(outputDesc) && outputDesc.GetTypeName() == TfToken("UsdGenDescription"),
          "Output: authors atomic OutputCurves, OutputRegionMap and Output triple");
    Check(!stage->GetPrimAtPath(paths.descriptionPath),
          "Output: leaves an unrelated legacy description path untouched");

    UsdPrim const sourceOp = stage->GetPrimAtPath(descPath.AppendChild(
        TfToken("Ops")).AppendChild(TfToken("source")));
    OutputMapPayload initialMap;
    SdfPathVector sourceCurves;
    float sourceDensity = 0.0f;
    Check(ReadOutputMapPayload(outputMap, &initialMap) &&
          ExistingOutputMapAsset(initialMap) &&
          SurfaceCageSourceContract(sourceOp, mapPath, initialMap.generation) &&
          sourceOp.GetRelationship(TfToken("usdGen:curves")).GetTargets(
              &sourceCurves) &&
          sourceCurves == SdfPathVector{curvesPath} &&
          sourceOp.GetAttribute(TfToken("usdGen:densityMultiplier")).Get(
              &sourceDensity) &&
          std::abs(sourceDensity - 1.0f) <= 1e-6f,
          "Output: current owned PTex map is on disk and bound by the surface cage source");

    PomadeSnapshotGuides const leafOwners =
        PomadeGuidesFromSnapshot(PomadeSnapshotFromModel(model));
    CagePayload initialCage;
    Check(ReadCagePayload(curves, &initialCage) &&
          OutputFrameContract(stage, descPath),
          "Output: commits a populated C3 surface cage with valid root frames");
    Cooked first = CookOutput(stage, descPath);
    Check(first.ok, "Output: CurveSource -> Width compiles and executes: " +
          first.diagnostic + (first.ok ? "" : CageTriangleDiagnostics(curves)));
    if (first.ok) {
        Check(OwnersMatchKnownLeaves(leafOwners, first.buffer) &&
                  HasOnlyLeafOwners(leafOwners, first.buffer),
              "Output: engine preserves leaf tube, region and hierarchy owners without crossings");
        Check(CageTrajectoryFollowsPhysicalRoot(curves, first.buffer),
              "Output: each sampled root follows its local asymmetric cage triangle through every CV");
        bool widthOk = first.buffer.width.size() == first.buffer.totalCvs;
        for (float value : first.buffer.width)
            widthOk = widthOk && std::abs(value - output.width) <= 1e-6f;
        Check(widthOk, "Output: downstream Width owns the configured width");
    }

    // Interactive state must never choose the committed Output source.
    Check(model.SetFocusLevel(3) && model.SetPreviewFraction(0.125f) &&
          model.RefillGuides(0.125f) && model.ClearGeneratedCurves(),
          "Output: changes display level, preview fraction and clears guides");
    error.clear();
    UsdStageRefPtr hiddenStage = CommitStage(model, paths, &error);
    Cooked hidden = hiddenStage ? CookOutput(hiddenStage, descPath) : Cooked{};
    Check(hidden.ok && hidden.buffer.totalCurves == first.buffer.totalCurves &&
          hidden.buffer.totalCvs == first.buffer.totalCvs,
          "Output: cleared or preview guides never change committed full Fill");

    // Density is an ordinary CurveSource runtime input.  Editing it directly
    // must recook transient hair while leaving every Pomade-authored cage/map
    // payload byte-identical; committing a new dense USD source would defeat
    // this contract.
    uint64_t const modelVersionBeforeDensity = model.GetVersion();
    size_t const guideCountBeforeDensity =
        PomadeGuidesFromSnapshot(PomadeSnapshotFromModel(model)).counts.size();
    OutputMapPayload const mapBeforeDensity = initialMap;
    CagePayload cageBeforeDensity;
    UsdAttribute const density = sourceOp.GetAttribute(
        TfToken("usdGen:densityMultiplier"));
    bool const directDensity = ReadCagePayload(curves, &cageBeforeDensity) && density &&
        density.Set(2.0f);
    Cooked dense = directDensity ? CookOutput(stage, descPath) : Cooked{};
    CagePayload cageAfterDensity;
    Check(directDensity && dense.ok &&
          dense.buffer.totalCurves > first.buffer.totalCurves &&
          ReadCagePayload(curves, &cageAfterDensity) &&
          ReadOutputMapPayload(outputMap, &initialMap) &&
          initialMap.asset == mapBeforeDensity.asset &&
          initialMap.generation == mapBeforeDensity.generation &&
          ExistingOutputMapAsset(initialMap) &&
          SurfaceCageSourceContract(sourceOp, mapPath, initialMap.generation) &&
          SameCagePayload(initialCage, cageBeforeDensity) &&
          SameCagePayload(cageBeforeDensity, cageAfterDensity) &&
          model.GetVersion() == modelVersionBeforeDensity &&
          PomadeGuidesFromSnapshot(PomadeSnapshotFromModel(model)).counts.size() ==
              guideCountBeforeDensity,
          "Output: direct runtime density grows transient hair without rebaking cage or guides");

    // The sparse source itself remains a live USD input.  An affine cage
    // edit must recook the dense stream from the same sampled roots: c0 is
    // still the scalp point while each terminal point follows the authored
    // non-uniform x/z transform.  This catches a runtime path that scatters
    // the right count but ignores the local cage trajectory.
    VtVec3fArray affineRails;
    VtIntArray affineRailCounts;
    bool const readRails =
        curves.GetAttribute(TfToken("points")).Get(&affineRails) &&
        curves.GetAttribute(TfToken("curveVertexCounts")).Get(&affineRailCounts);
    // Keep every sampled scalp root fixed.  Moving the entire rail cage would
    // legitimately change its coverage and therefore the dense curve ID set;
    // changing only each terminal station isolates runtime trajectory recook.
    size_t affineOffset = 0;
    bool validAffineRails = readRails;
    for (int const count : affineRailCounts) {
        if (count < 2 || affineOffset + size_t(count) > affineRails.size()) {
            validAffineRails = false;
            break;
        }
        GfVec3f &point = affineRails[affineOffset + size_t(count - 1)];
        point[0] = 1.25f * point[0] + 0.03125f;
        point[2] = 0.75f * point[2] - 0.0175f;
        affineOffset += size_t(count);
    }
    validAffineRails = validAffineRails && affineOffset == affineRails.size();
    bool const setRails = validAffineRails &&
        curves.GetAttribute(TfToken("points")).Set(affineRails);
    Cooked affine = setRails ? CookOutput(stage, descPath) : Cooked{};
    std::string affineDiagnostic;
    bool affineOk = setRails && affine.ok && dense.ok;
    if (affineOk) {
        affineOk = CageAffineTerminalResponse(dense.buffer, affine.buffer,
                                               &affineDiagnostic);
    }
    Check(affineOk,
          "Output: direct affine cage rails recook locally while sampled roots stay fixed: " +
          (affine.ok ? affineDiagnostic : affine.diagnostic));

    // Output consists of several coherent prims.  Exercise the committer's
    // incremental transfer path, rather than only the one-shot layer build,
    // so a partial swap cannot expose half a source graph.
    SdfLayerRefPtr live = SdfLayer::CreateAnonymous("pomade-output-live");
    UsdStageRefPtr swapStage = UsdStage::CreateInMemory("pomade-output-swap");
    swapStage->GetSessionLayer()->InsertSubLayerPath(live->GetIdentifier(), 0);
    {
        UsdGeomMesh mesh = UsdGeomMesh::Define(swapStage, paths.scalpPath);
        usdGen::UsdGenVec3fArray points(grid.points.size() / 3);
        for (size_t i = 0; i < points.size(); ++i) {
            points[i] = usdGen::UsdGenVec3f(grid.points[i * 3], grid.points[i * 3 + 1],
                                grid.points[i * 3 + 2]);
        }
        mesh.GetPointsAttr().Set(ToHost(points));
        mesh.GetFaceVertexCountsAttr().Set(
            VtIntArray(grid.counts.begin(), grid.counts.end()));
        mesh.GetFaceVertexIndicesAttr().Set(
            VtIntArray(grid.indices.begin(), grid.indices.end()));
    }
    PomadeCommitter partial(&model, paths);
    partial.SetSwapBudgetMs(0.0);
    partial.Enqueue(swapStage);
    Check(WaitCommitted(partial, live, model.GetVersion()) &&
          partial.PartialMode() &&
          bool(swapStage->GetPrimAtPath(curvesPath)) &&
          bool(swapStage->GetPrimAtPath(mapPath)) &&
          bool(swapStage->GetPrimAtPath(descPath)),
          "Output: forced partial commit swaps the complete Output triple");

    // Return to the baseline density before checking regeneration: a density
    // change legitimately changes the local-root stream, whereas this next
    // assertion isolates one nested tube edit.
    output.densityMultiplier = 1.0f;
    Check(model.SetOutputSettings(output),
          "Output: restores model density before the regeneration check");
    error.clear();
    UsdStageRefPtr baselineStage = CommitStage(model, paths, &error);
    Cooked baseline = baselineStage ? CookOutput(baselineStage, descPath) : Cooked{};
    Check(baseline.ok && baseline.buffer.totalCurves == first.buffer.totalCurves &&
          baseline.buffer.totalCvs == first.buffer.totalCvs,
          "Output: rebuilt baseline density reproduces the original runtime cardinality");

    // An edit regenerates only its own shape; a known other-region curve
    // remains byte-identical by stable id even though the whole output cooks.
    uint64_t stableOther = 0;
    UsdGenPlane const *baselineOwners = FindCurvePlane(baseline.buffer, "tubeId");
    if (baselineOwners) {
        for (size_t i = 0; i < baselineOwners->i.size(); ++i) {
            if (baselineOwners->i[i] == bNested[1]) {
                stableOther = baseline.buffer.curveId[i];
                break;
            }
        }
    }
    Check(stableOther != 0 && model.MoveTubeSectionCV(aNested[0], 2, 2, 0.08f, -0.03f),
          "Output: edits one nested leaf after the first output build");
    error.clear();
    UsdStageRefPtr editedStage = CommitStage(model, paths, &error);
    Cooked edited = editedStage ? CookOutput(editedStage, descPath) : Cooked{};
    Check(edited.ok && OwnersMatchKnownLeaves(leafOwners, edited.buffer) &&
          SameCurveForId(baseline.buffer, edited.buffer, stableOther),
          "Output: regenerates edited geometry while an unrelated region owner stays exact");

    // Persistence takes the actual Save groom route.  Its portable map copy
    // must leave an independently reopened description able to cook without
    // a Pomade model or the versioned temporary source map.
    std::error_code removeError;
    std::filesystem::path const saveDir = std::filesystem::temp_directory_path() /
        "usdGen-pomade-output-roundtrip";
    std::filesystem::remove_all(saveDir, removeError);
    std::filesystem::create_directories(saveDir, removeError);
    std::filesystem::path const saved = saveDir / "groom.usdc";
    std::string saveError;
    Check(editedStage && ReadOutputMapPayload(
              editedStage->GetPrimAtPath(mapPath), &initialMap) &&
          PomadeSaveGroomAndMaps(editedStage, editedStage->GetRootLayer(),
                                saved.string(), "", paths, &saveError),
          "Output: Save groom relocates the owned output map: " + saveError);
    UsdStageRefPtr reopened = UsdStage::Open(saved.string());
    UsdPrim const reopenedMap = reopened ? reopened->GetPrimAtPath(mapPath) : UsdPrim();
    OutputMapPayload portableMap;
    SdfAssetPath portableAsset;
    std::string const portablePrefix = "./" + saved.stem().string() +
        ".output-outputRegionMap.";
    std::string const portableName = reopenedMap.GetAttribute(
        TfToken("usdGen:map:file")).Get(&portableAsset)
        ? portableAsset.GetAssetPath() : std::string();
    bool const portableNameOk = portableName.rfind(portablePrefix, 0) == 0 &&
        portableName.size() > portablePrefix.size() + 4 &&
        portableName.compare(portableName.size() - 4, 4, ".ptx") == 0;
    bool const portableMapOk = reopened && ReadOutputMapPayload(reopenedMap, &portableMap) &&
        portableNameOk &&
        ExistingOutputMapAsset(portableMap);
    Check(portableNameOk && ExistingOutputMapAsset(portableMap),
          "Output: a groom-local content-addressed owner map lands beside the saved groom");
    Cooked reopenedCook = portableMapOk ? CookOutput(reopened, descPath) : Cooked{};
    Check(portableMapOk && reopenedCook.ok &&
          reopenedCook.buffer.totalCurves == edited.buffer.totalCurves &&
          reopenedCook.buffer.totalCvs == edited.buffer.totalCvs,
          "Output: standalone reopened groom cooks from its relocated categorical map");
    PomadeModel hydrated;
    PomadeHydrateResult const hydrate = reopened
        ? PomadeHydrateModel(reopened, paths.groomPath, &hydrated)
        : PomadeHydrateResult{};
    PomadeModel::OutputSettings const restored = hydrated.GetOutputSettings();
    Check(hydrate.ok && restored.enabled &&
          restored.densityMultiplier == output.densityMultiplier &&
          restored.width == output.width,
          "Output: save/reopen hydrates enabled, multiplier and width: " +
              hydrate.diagnostic);
    std::filesystem::remove_all(saveDir, removeError);

    // A deliberately empty Fill remains a valid, executable Output stack.
    // An enabled CurveSource must carry an empty curve set, never malformed
    // topology that makes the compiler reject a legitimate density-zero groom.
    PomadeModel empty;
    PomadeModel::FillParams none;
    none.density = 0.0f;
    none.cvCount = 6;
    PomadeModel::OutputSettings emptyOutput;
    emptyOutput.enabled = true;
    emptyOutput.densityMultiplier = 1.0f;
    emptyOutput.width = 0.02f;
    Check(empty.BindScalp(grid.points, grid.counts, grid.indices) &&
          bool(empty.GetScalp()), "Output: binds a scalp for zero-density Fill");
    if (std::shared_ptr<PomadeScalpMesh const> emptyScalp = empty.GetScalp()) {
        AddTwoAdjacentRegions(&empty, *emptyScalp, n);
    }
    Check(empty.Rasterise() && empty.BuildTubeFromRegion(0, 5, 8, 2.0f) &&
          empty.SetTubeFillParams(0, none) && empty.SetOutputSettings(emptyOutput),
          "Output: accepts enabled zero-density Fill");
    PomadeCommitPaths emptyPaths;
    emptyPaths.groomPath = SdfPath("/World/PomadeEmptyOutput");
    emptyPaths.scalpPath = SdfPath("/Scalp");
    error.clear();
    UsdStageRefPtr emptyStage = CommitStage(empty, emptyPaths, &error);
    Cooked emptyCook = emptyStage
        ? CookOutput(emptyStage, OutputDescriptionPath(emptyPaths)) : Cooked{};
    Check(emptyCook.ok && emptyCook.buffer.totalCurves == 0 &&
          emptyCook.buffer.totalCvs == 0,
          "Output: enabled zero-density Fill cooks an empty valid source");

    CheckSubsetOutput(grid, n);

    std::printf("testUsdGenPomadeOutput: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
