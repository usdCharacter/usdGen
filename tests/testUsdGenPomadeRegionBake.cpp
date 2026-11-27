// testUsdGenPomadeRegionBake — T1: the versioned region-map bake (plan/17
// §3.1a, P2 exit).
//
// Proven here:
//   * PomadeBakePtex writes a readable multi-face .ptx (tmp -> rename);
//   * sampling that .ptx through the real ptex() expression — evaluated by
//     UsdGenCpuParameters for a GuideInterpolate node, exactly as the
//     engine does — yields, per strand, the GuideInterpolate region key of
//     the per-face live primvar of its root (opUtil::RegionKey, the same
//     function guideInterpolate.cpp groups by);
//   * the worker coalesces rapid enqueues to one bake, cancels a superseded
//     bake (no file for it), bakes incrementally (an unchanged graph
//     re-classifies zero faces), and sweeps stale versions;
//   * the one-attribute swap advances usdGen:map:file; a stale file
//     (superseded, deleted, or tmp) is never referenced by the live layer;
//   * the §3.1 build authors the graph, the live primvar, the current map
//     file and the RegionExpr; hydrate round-trips the graph bit-exactly;
//   * SaveGroomAndMaps copies regionMap.ptx beside the saved groom and
//     repoints the file layer, leaving the live versioned reference;
//   * the Clump wiring offer fires only for unconnected usdGen:clump:map.

#include "utils/pxrInterop.h"
#include "usdGenPomade/pomadeApi.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenPomade/pomadeApiStage.h"
#include "usdGenPomade/pomadeBake.h"
#include "usdGenPomade/pomadeCommit.h"
#include "usdGenPomade/pomadeGraph.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeRegion.h"
#include "usdGenPomade/pomadeScalp.h"

#include "usdGenExpr/cpuParameters.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/ops/opUtil.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/mesh.h"

// Ptexture.h drags in <Windows.h>; it goes after the pxr
// headers so their std::min/std::max are already parsed.
#include <Ptexture.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
    std::fflush(stdout);
}

struct Grid {
    std::vector<float> points;
    std::vector<int> counts;
    std::vector<int> indices;
};

Grid MakeGrid(int n = 4)
{
    Grid grid;
    for (int ix = 0; ix <= n; ++ix) {
        for (int iz = 0; iz <= n; ++iz) {
            grid.points.push_back(float(ix));
            grid.points.push_back(0.0f);
            grid.points.push_back(float(iz));
        }
    }
    auto pid = [&](int ix, int iz) { return ix * (n + 1) + iz; };
    for (int ix = 0; ix < n; ++ix) {
        for (int iz = 0; iz < n; ++iz) {
            grid.counts.push_back(4);
            grid.indices.push_back(pid(ix, iz));
            grid.indices.push_back(pid(ix, iz + 1));
            grid.indices.push_back(pid(ix + 1, iz + 1));
            grid.indices.push_back(pid(ix + 1, iz));
        }
    }
    return grid;
}

usdGenPomade::PomadeHit Locate(usdGenPomade::PomadeScalpMesh const &mesh, int n,
                             float x, float z)
{
    using namespace usdGenPomade;
    int ix = std::min(std::max(int(std::floor(x)), 0), n - 1);
    int iz = std::min(std::max(int(std::floor(z)), 0), n - 1);
    PomadeHit hit;
    hit.hit = true;
    hit.faceId = ix * n + iz;
    hit.u = z - float(iz);
    hit.v = x - float(ix);
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (!PomadeFacePosition(mesh, hit.faceId, hit.u, hit.v, &px, &py, &pz)) {
        hit.hit = false;
        return hit;
    }
    hit.px = px;
    hit.py = py;
    hit.pz = pz;
    hit.nx = 0.0f;
    hit.ny = 1.0f;
    hit.nz = 0.0f;
    return hit;
}

// Two adjoining rects on the n-grid: A is x in [0, n/2], B is x in
// [n/2, n], both z in [1, 3]. Faces with iz in {1, 2} are claimed.
void BuildAdjoining(usdGenPomade::PomadeModel *model,
                    usdGenPomade::PomadeScalpMesh const &mesh, int n)
{
    float const mid = float(n) / 2.0f;
    float const farX = float(n);  // `far` is a windows.h macro
    int a0 = model->GraphAddNode(Locate(mesh, n, 0.0f, 1.0f));
    int a1 = model->GraphAddNode(Locate(mesh, n, mid, 1.0f));
    int a2 = model->GraphAddNode(Locate(mesh, n, mid, 3.0f));
    int a3 = model->GraphAddNode(Locate(mesh, n, 0.0f, 3.0f));
    model->GraphConnect(a0, a1);
    model->GraphConnect(a1, a2);
    model->GraphConnect(a2, a3);
    model->GraphConnect(a3, a0);
    int b1 = model->GraphAddNode(Locate(mesh, n, farX, 1.0f));
    int b2 = model->GraphAddNode(Locate(mesh, n, farX, 3.0f));
    model->GraphConnect(a1, b1);
    model->GraphConnect(b1, b2);
    model->GraphConnect(b2, a2);
    model->GraphConnect(a2, a1);  // refused: B closes over A's edge
}

usdGen::UsdGenCurveBuffer StrandsAtCentroids(Grid const &grid)
{
    // One 2-CV strand per face, rooted at the face centroid.
    int const faces = int(grid.counts.size());
    usdGen::UsdGenCurveBuffer b;
    b.totalCurves = uint32_t(faces);
    b.totalCvs = uint32_t(faces * 2);
    b.px.resize(faces * 2);
    b.py.resize(faces * 2);
    b.pz.resize(faces * 2);
    b.curveId.resize(faces);
    b.rootPrim.resize(faces);
    int corner = 0;
    for (int f = 0; f < faces; ++f) {
        float cx = 0.0f, cz = 0.0f;
        for (int k = 0; k < 4; ++k) {
            int const v = grid.indices[size_t(corner + k)];
            cx += grid.points[size_t(v) * 3 + 0];
            cz += grid.points[size_t(v) * 3 + 2];
        }
        corner += 4;
        b.px[size_t(f) * 2 + 0] = cx / 4.0f;
        b.py[size_t(f) * 2 + 0] = 0.0f;
        b.pz[size_t(f) * 2 + 0] = cz / 4.0f;
        b.px[size_t(f) * 2 + 1] = cx / 4.0f;
        b.py[size_t(f) * 2 + 1] = 1.0f;
        b.pz[size_t(f) * 2 + 1] = cz / 4.0f;
        b.curveId[size_t(f)] = uint64_t(f + 1);
        b.rootPrim[size_t(f)] = f;
    }
    return b;
}

void CheckBakeAndExpression()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "bake: the scalp binds");
    BuildAdjoining(&model, *model.GetScalp(), 4);
    Check(model.Rasterise(), "bake: K3 runs");
    PomadeModel::GraphSnapshot snap = model.SnapshotGraph();
    Check(snap.regionLoops.size() == 2 && snap.faceRegions.size() == 16,
          "bake: two regions rasterise over 16 faces");

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeRegionBake";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    std::string const v1 =
        (dir / PomadeBakeFileName("regionMap", 1)).string();
    PomadeBakeInput input;
    input.scalp = model.GetScalp();
    input.graph = model.GetGraph();
    input.outDir = dir.string();
    std::vector<std::vector<float>> cache;
    PomadeBakeStats stats;
    std::string err;
    Check(PomadeBakePtex(input, v1, nullptr, &cache, &stats, &err),
          "bake: the synchronous bake writes v1: " + err);
    Check(fs::is_regular_file(v1, ec) && stats.ptexFaces == 16 &&
              stats.channels == 1,
          "bake: v1 holds 16 ptex faces, 1 channel");
    Check(!fs::is_regular_file(v1 + ".tmp", ec),
          "bake: no tmp file survives the rename");

    // Sample v1 through the real ptex() expression on a GuideInterpolate.
    usdGen::UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Desc");
    usdGen::UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Scalp");
    surface.faceVertexCounts =
        VtIntArray(grid.counts.data(), grid.counts.data() + grid.counts.size());
    surface.faceVertexIndices = VtIntArray(grid.indices.data(),
                                           grid.indices.data() +
                                               grid.indices.size());
    surface.restPoints.reserve(grid.points.size() / 3);
    for (size_t i = 0; i < grid.points.size() / 3; ++i) {
        surface.restPoints.push_back(
            GfVec3f(grid.points[i * 3], grid.points[i * 3 + 1],
                    grid.points[i * 3 + 2]));
    }
    desc.surfaces.push_back(surface);
    usdGen::UsdGenMapDesc map;
    map.path = SdfPath("/Groom/RegionMap");
    map.type = TfToken("UsdGenPtexMap");
    map.resolvedAssetPath = v1;
    map.params.push_back(
        {TfToken("map:filter"), VtValue(TfToken("nearest")), false});
    // Mirrors the authored RegionMap prim: equal components disable the
    // [0, 1] clamp, so uncovered (-1) strands survive the read.
    map.params.push_back({TfToken("map:clamp"), VtValue(GfVec2f(0, 0)),
                          false});
    desc.maps.push_back(map);
    usdGen::UsdGenExpressionDesc expression;
    expression.path = SdfPath("/Groom/Desc/Expressions/region");
    expression.source = "ptex(\"regionMap\")";
    usdGen::UsdGenExpressionInputDesc einput;
    einput.name = TfToken("regionMap");
    einput.targets = {map.path};
    einput.maps = {map.path};
    expression.inputs.push_back(einput);
    usdGen::UsdGenExpressionOutputDesc eresult;
    eresult.name = TfToken("result");
    eresult.nativeType = TfToken("float");
    eresult.shape.scalar = usdGen::expr::ScalarType::Float32;
    expression.outputs.push_back(eresult);
    desc.expressions.push_back(expression);
    usdGen::UsdGenNodeDesc node;
    node.path = SdfPath("/Groom/Desc/Ops/interp");
    node.type = TfToken("UsdGenGuideInterpolate");
    node.surfaces = {surface.path};
    usdGen::UsdGenExpressionBinding binding;
    binding.expression = expression.path;
    binding.output = TfToken();
    binding.nativeType = TfToken("float");
    binding.destination = TfToken("usdGen:region");
    binding.destinationShape.scalar = usdGen::expr::ScalarType::Float32;
    binding.domain = usdGen::expr::Domain::Primitive;
    binding.literal = VtValue(0.0f);
    node.expressionBindings.push_back(binding);
    desc.nodes.push_back(node);

    usdGen::UsdGenCurveBuffer const strands = StrandsAtCentroids(grid);
    usdGen::UsdGenCpuParameters parameters;
    bool changed = false;
    std::vector<std::string> errors;
    bool const ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), usdGen::UsdGenCurveSetView::Of(strands), 0, 0, 0,
                                        &changed, &errors);
    Check(ok, "bake: the ptex() expression evaluates" +
                      (errors.empty() ? "" : ": " + errors[0]));
    auto const *value = parameters.Find(TfToken("region"));
    bool perStrand = value && value->values.size() == 16;
    for (size_t s = 0; s < 16; ++s) {
        // GuideInterpolate groups by RegionKey (guideInterpolate.cpp); the
        // baked id must equal the live primvar of the strand's root face.
        int const expect = snap.faceRegions[s];
        double const got = value ? value->values[s] : 999.0;
        bool const strandOk =
            perStrand &&
            usdGen::opUtil::RegionKey(got) ==
                usdGen::opUtil::RegionKey(double(expect));
        if (!strandOk) {
            std::printf("debug strand %d: got %g expect %d\n", int(s), got,
                        expect);
        }
        perStrand = perStrand && strandOk;
    }
    Check(perStrand,
          "bake: ptex() per strand equals the primvar of its root face");
    Check(parameters.TakeWarnings().empty(), "bake: the read warns nothing");
    fs::remove_all(dir, ec);
}

bool WaitBakes(usdGenPomade::PomadeBakeWorker const &worker, size_t count,
               int timeoutMs = 30000)
{
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
    while (worker.BakeCount() < count) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

void CheckWorker()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    model.BindScalp(grid.points, grid.counts, grid.indices);
    BuildAdjoining(&model, *model.GetScalp(), 4);

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeBakeWorker";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    auto inputFor = [&] {
        PomadeBakeInput input;
        input.scalp = model.GetScalp();
        input.graph = model.GetGraph();
        input.outDir = dir.string();
        return input;
    };
    // Every version below comes from the model (synthetic versions would
    // collide with later model edits and desynchronise the waits).
    PomadeBakeWorker worker;
    // Coalescing: three model edits while paused bake once, the latest.
    worker.PauseWorker(true);
    model.GraphMoveNode(0, Locate(*model.GetScalp(), 4, 0.05f, 1.0f));
    worker.Enqueue(model.GetMapVersion(), inputFor());
    model.GraphMoveNode(0, Locate(*model.GetScalp(), 4, 0.10f, 1.0f));
    worker.Enqueue(model.GetMapVersion(), inputFor());
    model.GraphMoveNode(0, Locate(*model.GetScalp(), 4, 0.15f, 1.0f));
    uint64_t const vC = model.GetMapVersion();
    worker.Enqueue(vC, inputFor());
    worker.PauseWorker(false);
    Check(worker.WaitCompleted(vC), "worker: the latest version lands");
    Check(WaitBakes(worker, 1) && worker.BakeCount() == 1 &&
              worker.CompletedVersion() == vC,
          "worker: three enqueues coalesce to one bake");
    uint64_t done = 0;
    std::string path;
    Check(worker.TakeCompleted(&done, &path) && done == vC &&
              fs::is_regular_file(path, ec),
          "worker: TakeCompleted reports the versioned file");
    Check(!worker.TakeCompleted(&done, &path),
          "worker: completions report once");
    // Incremental: an unchanged graph re-classifies zero faces.
    worker.Enqueue(vC, inputFor());
    Check(WaitBakes(worker, 2), "worker: the repeat bake lands");
    Check(worker.LastBakedFaces() == 0,
          "worker: an unchanged graph re-classifies zero faces");
    Check(worker.TakeCompleted(&done, &path) && done == vC,
          "worker: the repeat completion reports");
    // A graph edit can move a boundary within a face without changing that
    // face's centroid label.  The cache therefore re-classifies every face
    // whenever its classifier inputs change.
    Check(model.GraphMoveNode(1, Locate(*model.GetScalp(), 4, 2.5f, 1.5f)),
          "worker: the nudge moves the shared corner");
    uint64_t const vD = model.GetMapVersion();
    worker.Enqueue(vD, inputFor());
    bool const landed = worker.WaitCompleted(vD);
    if (!landed) {
        std::printf("debug nudge: bakes=%d cancels=%d completed=%llu "
                    "diag=%s\n",
                    int(worker.BakeCount()), int(worker.CancelCount()),
                    (unsigned long long)worker.CompletedVersion(),
                    worker.TakeDiagnostic().c_str());
    }
    Check(landed, "worker: the edited bake lands");
    Check(worker.LastBakedFaces() == grid.counts.size(),
          "worker: a graph edit invalidates every cached texel");
    // Supersede: a version enqueued during a long bake cancels it (or the
    // wake coalesces first — either way exactly one bake finishes and the
    // superseded version leaves no file).
    Grid const big = MakeGrid(320);
    PomadeModel bigModel;
    bigModel.BindScalp(big.points, big.counts, big.indices);
    BuildAdjoining(&bigModel, *bigModel.GetScalp(), 320);
    PomadeBakeInput bigInput;
    bigInput.scalp = bigModel.GetScalp();
    bigInput.graph = bigModel.GetGraph();
    bigInput.outDir = dir.string();
    bigInput.resOverride = 0;  // 1 texel/face: the rasterise dominates
    size_t const bakesBefore = worker.BakeCount();
    size_t const cancelsBefore = worker.CancelCount();
    worker.PauseWorker(true);
    worker.Enqueue(9001, bigInput);
    worker.PauseWorker(false);
    worker.Enqueue(9002, bigInput);
    Check(worker.WaitCompleted(9002, 60000), "worker: the slow bake lands");
    size_t const bakes = worker.BakeCount() - bakesBefore;
    size_t const cancels = worker.CancelCount() - cancelsBefore;
    Check(bakes == 1 && (cancels == 0 || cancels == 1),
          "worker: supersede finishes one bake (cancelled or coalesced)");
    Check(!fs::is_regular_file(dir / PomadeBakeFileName("regionMap", 9001),
                               ec) &&
              fs::is_regular_file(dir / PomadeBakeFileName("regionMap", 9002),
                                  ec),
          "worker: the superseded version leaves no file");
    // Sweep: swapping the latest deletes older versions from the disk.
    worker.NoteSwapped(9002,
                       (dir / PomadeBakeFileName("regionMap", 9002)).string());
    worker.Enqueue(9003, bigInput);  // a bake runs the sweep at its end
    Check(worker.WaitCompleted(9003, 60000), "worker: the sweep bake lands");
    // The sweep runs on the worker right after the completion flag: poll.
    auto swept = [&] {
        for (auto const &entry : fs::directory_iterator(dir, ec)) {
            std::string const name = entry.path().filename().string();
            if (name.find("regionMap.v") == 0 &&
                name != "regionMap.v9003.ptx" &&
                name != "regionMap.v9002.ptx") {
                return false;
            }
        }
        return true;
    };
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!swept() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Check(swept(), "worker: stale versions are swept once unreferenced");
    bool noTmp = true;
    for (auto const &entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() == ".tmp") {
            noTmp = false;
        }
    }
    Check(noTmp, "worker: no tmp file ever survives");
    fs::remove_all(dir, ec);
}

// A closed region can sit entirely inside a coarse quad.  Its corners and
// centroid then all classify as uncovered, so neither the old collapse rule
// nor the old centroid-id cache key saw it.  Keep two disjoint loops here to
// prove the texel map preserves both ids and a same-face boundary move
// invalidates its cached texels.
void CheckSubfaceRegionBake()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(1);
    PomadeModel model;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "subface: the single quad binds");
    auto addLoop = [&](float x0, float z0, float x1, float z1) {
        int const a = model.GraphAddNode(Locate(*model.GetScalp(), 1, x0, z0));
        int const b = model.GraphAddNode(Locate(*model.GetScalp(), 1, x1, z0));
        int const c = model.GraphAddNode(Locate(*model.GetScalp(), 1, x1, z1));
        int const d = model.GraphAddNode(Locate(*model.GetScalp(), 1, x0, z1));
        Check(a >= 0 && b >= 0 && c >= 0 && d >= 0 &&
                  model.GraphConnect(a, b) >= 0 &&
                  model.GraphConnect(b, c) >= 0 &&
                  model.GraphConnect(c, d) >= 0 &&
                  model.GraphConnect(d, a) >= 0,
              "subface: a small closed loop connects");
        return a;
    };
    int const moveNode = addLoop(0.08f, 0.08f, 0.30f, 0.30f);
    addLoop(0.68f, 0.68f, 0.90f, 0.90f);
    Check(model.GetGraph().RegionCount() == 2,
          "subface: two disjoint loops extract");

    PomadeRegionLoops loops;
    std::string err;
    Check(PomadeFlattenLoops(model.GetGraph(), &loops, &err),
          "subface: loops flatten: " + err);
    float const centroid[3] = {0.5f, 0.0f, 0.5f};
    Check(PomadeClassifyPointCpu(loops, centroid) == -1,
          "subface: neither small region contains the face centroid");
    PomadeRegionMaps maps;
    Check(PomadeRasteriseRegionsCpu(*model.GetScalp(), model.GetGraph(), &maps,
                                   &err),
          "subface: coarse rasterise completes: " + err);
    std::vector<int> const autoRes =
        PomadeFaceResLog2(*model.GetScalp(), maps, loops);
    Check(autoRes.size() == 1 && autoRes[0] == 6,
          "subface: auto bake keeps a 64x64 boundary face");

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeSubfaceRegionBake";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    auto inputFor = [&] {
        PomadeBakeInput input;
        input.scalp = model.GetScalp();
        input.graph = model.GetGraph();
        input.outDir = dir.string();
        input.resOverride = 6;
        return input;
    };
    auto readFace = [](std::string const &path, std::vector<float> *out) {
        Ptex::String error;
        PtexPtr<PtexTexture> tex(
            PtexTexture::open(path.c_str(), error, /*premultiply=*/false));
        if (!tex || tex->numFaces() != 1 || tex->numChannels() != 1) {
            return false;
        }
        Ptex::Res const res = tex->getFaceInfo(0).res;
        out->resize(size_t(res.size()));
        tex->getData(0, out->data(), 0);
        return true;
    };
    PomadeBakeWorker worker;
    uint64_t const first = model.GetMapVersion();
    worker.Enqueue(first, inputFor());
    std::string firstPath;
    uint64_t done = 0;
    Check(worker.WaitCompleted(first) &&
              worker.TakeCompleted(&done, &firstPath) && done == first,
          "subface: fixed-resolution first bake completes");
    std::vector<float> before;
    Check(readFace(firstPath, &before), "subface: the first face reads");
    bool sawA = false, sawB = false, sawOutside = false;
    for (float value : before) {
        sawA = sawA || value == 0.0f;
        sawB = sawB || value == 1.0f;
        sawOutside = sawOutside || value == -1.0f;
    }
    Check(sawA && sawB && sawOutside,
          "subface: two polygons retain distinct texel ids");

    Check(model.GraphMoveNode(
              moveNode, Locate(*model.GetScalp(), 1, 0.18f, 0.08f)),
          "subface: a boundary moves within the same face");
    uint64_t const second = model.GetMapVersion();
    worker.Enqueue(second, inputFor());
    std::string secondPath;
    Check(worker.WaitCompleted(second) &&
              worker.TakeCompleted(&done, &secondPath) && done == second,
          "subface: fixed-resolution moved bake completes");
    std::vector<float> after;
    Check(readFace(secondPath, &after) && before.size() == after.size(),
          "subface: the moved face reads");
    bool changed = false;
    for (size_t i = 0; i < before.size() && i < after.size(); ++i) {
        changed = changed || before[i] != after[i];
    }
    Check(worker.LastBakedFaces() == 1 && changed,
          "subface: a same-centroid boundary move refreshes cached texels");
    fs::remove_all(dir, ec);
}

void CheckSwapAndCommit()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    model.BuildTestTube();
    model.BindScalp(grid.points, grid.counts, grid.indices);
    BuildAdjoining(&model, *model.GetScalp(), 4);
    model.NoteBakedMapFile(7, "regionMap.v7.ptx");

    PomadeCommitPaths paths;
    paths.groomPath = SdfPath("/PomadeGroom");
    paths.descriptionPath = SdfPath("/Groom/Hair");
    paths.scalpPath = SdfPath("/Scalp");
    PomadeSnapshot snapshot = PomadeSnapshotFromModel(model);
    snapshot.scalpPath = paths.scalpPath;
    snapshot.createInterpOp = true;
    snapshot.setInterpGuides = true;
    snapshot.setInterpRegion = true;
    snapshot.interpOpPath = paths.InterpOpPath();
    SdfLayerRefPtr built;
    std::string err;
    Check(PomadeBuildCommitLayer(snapshot, paths, &built, &err),
          "commit: the layer builds with a graph: " + err);
    // The scalp lives on the in-memory stage root; the built layer rides
    // beneath as a session sublayer (the P1 MakeStage pattern). Composed
    // reads below see the overs exactly as usdview would.
    UsdStageRefPtr stage = UsdStage::CreateInMemory("pomadeBake");
    stage->DefinePrim(SdfPath("/Groom/Hair"), TfToken("UsdGenDescription"));
    stage->DefinePrim(SdfPath("/Groom/Hair/Ops"), TfToken("Scope"));
    UsdGeomMesh scalpMesh =
        UsdGeomMesh(stage->DefinePrim(paths.scalpPath, TfToken("Mesh")));
    VtVec3fArray scalpPts;
    for (size_t i = 0; i < grid.points.size() / 3; ++i) {
        scalpPts.push_back(GfVec3f(grid.points[i * 3], grid.points[i * 3 + 1],
                                   grid.points[i * 3 + 2]));
    }
    scalpMesh.GetPointsAttr().Set(scalpPts);
    scalpMesh.GetFaceVertexCountsAttr().Set(VtIntArray(
        grid.counts.data(), grid.counts.data() + grid.counts.size()));
    scalpMesh.GetFaceVertexIndicesAttr().Set(VtIntArray(
        grid.indices.data(), grid.indices.data() + grid.indices.size()));
    stage->GetSessionLayer()->InsertSubLayerPath(built->GetIdentifier(), 0);

    // The graph, the primvar, the map file and the expression.
    VtIntArray nodeFaceIds;
    stage->GetPrimAtPath(paths.ScalpGraphPath())
        .GetAttribute(TfToken("usdGen:pomade:nodeFaceIds"))
        .Get(&nodeFaceIds);
    Check(nodeFaceIds.size() == 6, "commit: six graph nodes author");
    VtIntArray primvar;
    stage->GetPrimAtPath(paths.scalpPath)
        .GetAttribute(TfToken("primvars:usdGen:pomadeRegion"))
        .Get(&primvar);
    bool primvarOk = primvar.size() == 16;
    for (size_t f = 0; primvarOk && f < 16; ++f) {
        primvarOk = primvar[f] == snapshot.graph.faceRegions[f];
    }
    Check(primvarOk, "commit: the live primvar equals the snapshot map");
    SdfAssetPath mapFile;
    stage->GetPrimAtPath(paths.RegionMapPath())
        .GetAttribute(TfToken("usdGen:map:file"))
        .Get(&mapFile);
    Check(mapFile.GetAssetPath() == "regionMap.v7.ptx",
          "commit: §3.1 re-authors the current baked file (never wipes it)");
    Check(bool(stage->GetPrimAtPath(paths.RegionMapPath())) &&
              bool(stage->GetPrimAtPath(paths.RegionExprPath())),
          "commit: RegionMap + RegionExpr author");
    Check(!bool(stage->GetPrimAtPath(paths.LevelMapPath(2))),
          "commit: P2 bakes one level (no RegionMapL2)");
    SdfPathVector targets;
    stage->GetPrimAtPath(paths.InterpOpPath())
        .GetRelationship(TfToken("usdGen:guides"))
        .GetTargets(&targets);
    SdfPathVector region;
    stage->GetPrimAtPath(paths.InterpOpPath())
        .GetAttribute(TfToken("usdGen:region"))
        .GetConnections(&region);
    Check(targets.size() == 1 && targets[0] == paths.GuidesPath() &&
              region.size() == 1 && region[0] == paths.RegionExprPath(),
          "commit: the P1 fill-in survives the P2 build");

    // Hydrate round-trips the graph bit-exactly.
    PomadeModel hydrated;
    PomadeHydrateResult hr = PomadeHydrateModel(stage, paths.groomPath,
                                             &hydrated);
    Check(hr.ok && hr.graphRoundTrip && hr.guidesBitEqual &&
              hr.graphNodeCount == 6 && hr.graphRegionCount == 2,
          "hydrate: loops + primvar + guides round-trip bit-exactly: " +
              hr.diagnostic);

    // The one-attribute swap advances the map file, and only it.
    std::string swapErr;
    Check(!PomadeBakeSwapMapFile(SdfLayerHandle(), paths.RegionMapPath(),
                                "regionMap.v8.ptx", &swapErr),
          "swap: a null live layer fails honestly");
    Check(!PomadeBakeSwapMapFile(built, paths.RegionMapPath(), "",
                                &swapErr),
          "swap: an empty file fails honestly");
    Check(PomadeBakeSwapMapFile(built, paths.RegionMapPath(),
                               "regionMap.v8.ptx", &swapErr),
          "swap: the one-attribute author lands: " + swapErr);
    SdfAssetPath swapped =
        built->GetAttributeAtPath(paths.RegionMapPath().AppendProperty(
            TfToken("usdGen:map:file")))->GetDefaultValue().Get<SdfAssetPath>();
    Check(swapped.GetAssetPath() == "regionMap.v8.ptx",
          "swap: the live layer points at v8");
    // A stale file is never referenced: v7 is gone, the layer names v8.
    std::string exported;
    Check(built->ExportToString(&exported), "swap: the layer exports");
    Check(exported.find("regionMap.v8.ptx") != std::string::npos &&
              exported.find("regionMap.v7.ptx") == std::string::npos &&
              exported.find(".tmp") == std::string::npos,
          "swap: the live layer names only the latest versioned file");
}

void CheckSaveAndClump()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    model.BuildTestTube();
    model.BindScalp(grid.points, grid.counts, grid.indices);
    BuildAdjoining(&model, *model.GetScalp(), 4);
    PomadeCommitPaths paths;
    paths.groomPath = SdfPath("/PomadeGroom");
    paths.scalpPath = SdfPath("/Scalp");
    // A live layer with a baked map + a Clump op with an unconnected map.
    SdfLayerRefPtr live = SdfLayer::CreateAnonymous("usdGenPomade-live");
    PomadeSnapshot snapshot = PomadeSnapshotFromModel(model);
    snapshot.scalpPath = paths.scalpPath;
    SdfLayerRefPtr built;
    std::string err;
    PomadeBuildCommitLayer(snapshot, paths, &built, &err);
    live->TransferContent(built);
    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeSaveMaps";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    std::string const v9 =
        (dir / PomadeBakeFileName("regionMap", 9)).string();
    {
        PomadeBakeInput input;
        input.scalp = model.GetScalp();
        input.graph = model.GetGraph();
        input.outDir = dir.string();
        std::vector<std::vector<float>> cache;
        PomadeBakePtex(input, v9, nullptr, &cache, nullptr, &err);
    }
    PomadeBakeSwapMapFile(live, paths.RegionMapPath(), v9, &err);
    UsdStageRefPtr stage = UsdStage::CreateInMemory("pomadeSave");
    stage->GetSessionLayer()->InsertSubLayerPath(live->GetIdentifier(), 0);
    std::string const filePath = (dir / "groom.usdc").string();
    Check(PomadeSaveGroomAndMaps(stage, live, filePath, v9, paths, &err),
          "save: the groom saves with its map: " + err);
    Check(fs::is_regular_file(dir / "regionMap.ptx", ec),
          "save: regionMap.ptx lands beside the saved groom");
    SdfLayerRefPtr file = SdfLayer::FindOrOpen(filePath);
    SdfAssetPath saved =
        file->GetAttributeAtPath(paths.RegionMapPath().AppendProperty(
            TfToken("usdGen:map:file")))->GetDefaultValue().Get<SdfAssetPath>();
    // V0b: the asset path is RELATIVE to the saved layer, so the groom
    // stays portable when its folder moves (plan/18 §7 G5).
    Check(saved.GetAssetPath() == "./regionMap.ptx",
          "save: the file layer repoints at the copied map, relative");
    SdfAssetPath stillLive =
        live->GetAttributeAtPath(paths.RegionMapPath().AppendProperty(
            TfToken("usdGen:map:file")))->GetDefaultValue().Get<SdfAssetPath>();
    Check(stillLive.GetAssetPath() == v9,
          "save: the live layer keeps its versioned reference");
    // The Clump offer fires only for unconnected maps.
    UsdStageRefPtr clumpStage = UsdStage::CreateInMemory();
    UsdPrim const clump =
        clumpStage->DefinePrim(SdfPath("/Groom/Hair/Desc/Ops/clump"),
                               TfToken("UsdGenClump"));
    paths.descriptionPath = SdfPath("/Groom/Hair/Desc");
    PomadeClumpOffer offer =
        PomadePlanClumpFill(clumpStage, paths, /*deepestLevel*/ 1);
    Check(!offer.opPath.IsEmpty() &&
              offer.opPath == SdfPath("/Groom/Hair/Desc/Ops/clump") &&
              offer.exprPath == paths.RegionExprPath(),
          "clump: an unconnected map is offered the RegionExpr");
    UsdAttribute mapAttr = clump.CreateAttribute(TfToken("usdGen:clump:map"),
                                                 SdfValueTypeNames->Token);
    mapAttr.AddConnection(SdfPath("/Groom/OtherMap"));
    offer = PomadePlanClumpFill(clumpStage, paths, 1);
    Check(offer.opPath.IsEmpty(),
          "clump: a connected map is never offered (never rewritten)");
    fs::remove_all(dir, ec);
}

// V0b (plan/18 §7 G4): channel k of the map is the level-(k+1) tube id, so a
// Clump wired to level 2 through firstChannel = 1 reads real ids for every
// strand instead of the zeros P2 wrote.
void CheckLevelChannels()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "levels: the scalp binds");
    BuildAdjoining(&model, *model.GetScalp(), 4);
    Check(model.Rasterise(), "levels: K3 runs");
    PomadeModel::GraphSnapshot const snap = model.SnapshotGraph();
    Check(model.BuildTubeFromRegion(0, 4, 8, 3.0f),
          "levels: an L1 tube grows from region 0");
    std::vector<int> kids;
    Check(model.SubdivideTube(0, 2, "kmeans", 4, &kids) && kids.size() == 2,
          "levels: the L1 tube subdivides into two L2 tubes");

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeRegionBakeLevels";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    std::string const file =
        (dir / PomadeBakeFileName("regionMap", 1)).string();
    PomadeBakeInput input;
    input.scalp = model.GetScalp();
    input.graph = model.GetGraph();
    input.outDir = dir.string();
    input.levelCount = 2;
    PomadeCollectBakeTubes(model, &input.tubes);
    Check(input.tubes.size() == 3,
          "levels: the bake input carries the L1 tube and both children");
    std::vector<std::vector<float>> cache;
    PomadeBakeStats stats;
    std::string err;
    Check(PomadeBakePtex(input, file, nullptr, &cache, &stats, &err),
          "levels: the two-channel bake writes: " + err);
    Check(stats.channels == 2, "levels: the file holds two channels");

    // Read channel 1 through the real ptex() expression, exactly as a Clump
    // wired to level 2 would (firstChannel = 1).
    usdGen::UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Desc");
    usdGen::UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Scalp");
    surface.faceVertexCounts =
        VtIntArray(grid.counts.data(), grid.counts.data() + grid.counts.size());
    surface.faceVertexIndices = VtIntArray(
        grid.indices.data(), grid.indices.data() + grid.indices.size());
    for (size_t i = 0; i < grid.points.size() / 3; ++i) {
        surface.restPoints.push_back(
            GfVec3f(grid.points[i * 3], grid.points[i * 3 + 1],
                    grid.points[i * 3 + 2]));
    }
    desc.surfaces.push_back(surface);
    usdGen::UsdGenMapDesc map;
    map.path = SdfPath("/Groom/RegionMapL2");
    map.type = TfToken("UsdGenPtexMap");
    map.resolvedAssetPath = file;
    map.params.push_back(
        {TfToken("map:filter"), VtValue(TfToken("nearest")), false});
    map.params.push_back({TfToken("map:clamp"), VtValue(GfVec2f(0, 0)),
                          false});
    map.params.push_back({TfToken("map:firstChannel"), VtValue(1), false});
    map.params.push_back({TfToken("map:channelCount"), VtValue(1), false});
    desc.maps.push_back(map);
    usdGen::UsdGenExpressionDesc expression;
    expression.path = SdfPath("/Groom/Desc/Expressions/regionL2");
    expression.source = "ptex(\"regionMap\")";
    usdGen::UsdGenExpressionInputDesc einput;
    einput.name = TfToken("regionMap");
    einput.targets = {map.path};
    einput.maps = {map.path};
    expression.inputs.push_back(einput);
    usdGen::UsdGenExpressionOutputDesc eresult;
    eresult.name = TfToken("result");
    eresult.nativeType = TfToken("float");
    eresult.shape.scalar = usdGen::expr::ScalarType::Float32;
    expression.outputs.push_back(eresult);
    desc.expressions.push_back(expression);
    usdGen::UsdGenNodeDesc node;
    node.path = SdfPath("/Groom/Desc/Ops/clump");
    node.type = TfToken("UsdGenGuideInterpolate");
    node.surfaces = {surface.path};
    usdGen::UsdGenExpressionBinding binding;
    binding.expression = expression.path;
    binding.output = TfToken();
    binding.nativeType = TfToken("float");
    binding.destination = TfToken("usdGen:region");
    binding.destinationShape.scalar = usdGen::expr::ScalarType::Float32;
    binding.domain = usdGen::expr::Domain::Primitive;
    binding.literal = VtValue(0.0f);
    node.expressionBindings.push_back(binding);
    desc.nodes.push_back(node);

    usdGen::UsdGenCurveBuffer const strands = StrandsAtCentroids(grid);
    usdGen::UsdGenCpuParameters parameters;
    bool changed = false;
    std::vector<std::string> errors;
    bool const ok = parameters.Evaluate(usdGenUsd::Lower(desc), usdGenUsd::Lower(desc.nodes[0]), usdGen::UsdGenCurveSetView::Of(strands), 0, 0, 0,
                                        &changed, &errors);
    Check(ok, "levels: the level-2 ptex() expression evaluates" +
                  (errors.empty() ? "" : ": " + errors[0]));
    auto const *value = parameters.Find(TfToken("region"));
    Check(value && value->values.size() == 16,
          "levels: one value per strand");
    // Every strand rooted in the tube's region reads one of the two L2 ids;
    // strands outside it read 0 (no tube at that level there). Both cells
    // have to appear, or the partition would be a no-op.
    int const interp = model.GetGraph().InterpId(0);
    bool perStrand = bool(value);
    bool sawFirst = false, sawSecond = false, sawOutsideZero = false;
    for (size_t s = 0; value && s < value->values.size(); ++s) {
        int const got = int(std::lround(value->values[s]));
        bool const inRegion = snap.faceRegions[s] == interp;
        if (inRegion) {
            bool const claimed = got == kids[0] || got == kids[1];
            if (!claimed) {
                std::printf("debug level strand %d: got %d, want %d or %d\n",
                            int(s), got, kids[0], kids[1]);
            }
            perStrand = perStrand && claimed;
            sawFirst = sawFirst || got == kids[0];
            sawSecond = sawSecond || got == kids[1];
        } else {
            perStrand = perStrand && got == 0;
            sawOutsideZero = sawOutsideZero || got == 0;
        }
    }
    Check(perStrand,
          "levels: every strand in the region reads a level-2 tube id");
    Check(sawFirst && sawSecond,
          "levels: both child cells claim texels");
    Check(sawOutsideZero,
          "levels: a level with no tube there reads 0");
    fs::remove_all(dir, ec);
}

// V0b (plan/18 §7 G5): NoteSwapped wakes the worker for a sweep. Without it
// the file the PREVIOUS swap pinned survives until the next bake, which may
// never come.
void CheckSweepAfterSwap()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "sweep: the scalp binds");
    BuildAdjoining(&model, *model.GetScalp(), 4);
    Check(model.Rasterise(), "sweep: K3 runs");

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeRegionBakeSweep";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    PomadeBakeWorker worker;
    auto enqueue = [&](uint64_t version) {
        PomadeBakeInput input;
        input.scalp = model.GetScalp();
        input.graph = model.GetGraph();
        input.outDir = dir.string();
        worker.Enqueue(version, std::move(input));
    };
    enqueue(1);
    Check(worker.WaitCompleted(1), "sweep: v1 bakes");
    uint64_t version = 0;
    std::string path1;
    Check(worker.TakeCompleted(&version, &path1) && version == 1,
          "sweep: v1 completes");
    worker.NoteSwapped(1, path1);
    // A trivial graph touch so v2 is a different version with the same
    // content; the incremental cache keeps it cheap.
    enqueue(2);
    Check(worker.WaitCompleted(2), "sweep: v2 bakes");
    std::string path2;
    Check(worker.TakeCompleted(&version, &path2) && version == 2,
          "sweep: v2 completes");
    Check(fs::is_regular_file(path1, ec),
          "sweep: v1 survives while the live layer still references it");
    size_t const sweeps = worker.SweepCount();
    worker.NoteSwapped(2, path2);
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    while (worker.SweepCount() == sweeps &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(worker.SweepCount() > sweeps,
          "sweep: NoteSwapped wakes the worker for a sweep");
    auto const gone =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    while (fs::is_regular_file(path1, ec) &&
           std::chrono::steady_clock::now() < gone) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(!fs::is_regular_file(path1, ec),
          "sweep: the file the previous swap pinned is deleted");
    Check(fs::is_regular_file(path2, ec), "sweep: the swapped file stays");

    // The same loop survives a throw (plan/17 §3.4).
    size_t const throwsBefore = worker.WorkerThrowCount();
    worker.ThrowOnNextBakesForTest(1);
    enqueue(3);
    auto const tdl =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    while (worker.WorkerThrowCount() == throwsBefore &&
           std::chrono::steady_clock::now() < tdl) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(worker.WorkerThrowCount() == throwsBefore + 1,
          "sweep: the bake worker catches a throw");
    Check(!worker.TakeDiagnostic().empty(),
          "sweep: the throw reaches the status path");
    enqueue(4);
    Check(worker.WaitCompleted(4),
          "sweep: the bake worker survives the throw");
    fs::remove_all(dir, ec);
}

// V7: the Output panel's texel resolution override reaches the file.
//
// The override is the one bake option an artist sets by hand, so what it
// has to be worth is a .ptx whose faces really carry that resolution --
// not a value that stops at the bake context. The auto plan is read back
// first, because an override that happened to match it would prove
// nothing.
void CheckTexelResolutionOverride()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    PomadeModel model;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices),
          "texel override: the scalp binds");
    BuildAdjoining(&model, *model.GetScalp(), 4);
    Check(model.Rasterise(), "texel override: K3 runs");

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeBakeTexelOverride";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    // The per-face (ulog2, vlog2) of every face in a baked file.
    auto faceRes = [](std::string const &path,
                      std::vector<std::pair<int, int>> *out) {
        out->clear();
        Ptex::String error;
        PtexPtr<PtexTexture> tex(PtexTexture::open(path.c_str(), error,
                                                   /*premultiply=*/false));
        if (!tex) {
            return false;
        }
        for (int f = 0; f < tex->numFaces(); ++f) {
            Ptex::Res const res = tex->getFaceInfo(f).res;
            out->push_back({int(res.ulog2), int(res.vlog2)});
        }
        return true;
    };

    auto bakeAt = [&](int resOverride, uint64_t version,
                      std::vector<std::pair<int, int>> *res) {
        PomadeBakeInput input;
        input.scalp = model.GetScalp();
        input.graph = model.GetGraph();
        input.outDir = dir.string();
        input.resOverride = resOverride;
        std::string const path =
            (dir / PomadeBakeFileName("regionMap", version)).string();
        std::vector<std::vector<float>> cache;
        std::string err;
        if (!PomadeBakePtex(input, path, nullptr, &cache, nullptr, &err)) {
            Check(false, "texel override: bake failed: " + err);
            return false;
        }
        return faceRes(path, res);
    };

    std::vector<std::pair<int, int>> autoRes;
    Check(bakeAt(-1, 1, &autoRes) && !autoRes.empty(),
          "texel override: the auto plan bakes and reads back");
    bool autoUniformAt5 = true;
    for (auto const &r : autoRes) {
        autoUniformAt5 &= (r.first == 5 && r.second == 5);
    }
    Check(!autoUniformAt5,
          "texel override: the auto plan is not already a flat 32x32, so "
          "an override to it would be visible");

    std::vector<std::pair<int, int>> forced;
    Check(bakeAt(5, 2, &forced) && forced.size() == autoRes.size(),
          "texel override: the forced bake writes the same face count");
    bool allForced = !forced.empty();
    for (auto const &r : forced) {
        allForced &= (r.first == 5 && r.second == 5);
    }
    Check(allForced,
          "texel override: every ptex face is 32x32 when the override "
          "asks for 2^5");

    // And a second value, so the file follows the knob rather than one
    // hard-coded resolution.
    std::vector<std::pair<int, int>> forced3;
    Check(bakeAt(3, 3, &forced3) && forced3.size() == autoRes.size(),
          "texel override: 2^3 bakes too");
    bool allForced3 = !forced3.empty();
    for (auto const &r : forced3) {
        allForced3 &= (r.first == 3 && r.second == 3);
    }
    Check(allForced3, "texel override: and every face is 8x8");

    fs::remove_all(dir, ec);
}

// V0b: the C ABI that carries the hierarchy into the bake.
void CheckBakeLevelsAbi()
{
    Grid const grid = MakeGrid(4);
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK, "levels ABI: the model creates");
    Check(Pomade_BindScalp(ctx, grid.points.data(),
                          int(grid.points.size()), grid.counts.data(),
                          int(grid.counts.size()), grid.indices.data(),
                          int(grid.indices.size())) == POMADE_OK,
          "levels ABI: the scalp binds");
    Check(Pomade_BakeEnqueueLevels(nullptr) == POMADE_ERROR,
          "levels ABI: a null bake context is refused");
    PomadeBakeContext *bake = nullptr;
    std::error_code ec;
    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeRegionBakeAbi";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    Check(Pomade_BakeCreate(ctx, dir.string().c_str(), "regionMap", &bake) ==
              POMADE_OK,
          "levels ABI: the bake context creates");
    Check(Pomade_BakeSetOptions(bake, -1, 2) == POMADE_OK,
          "levels ABI: two channels are requested");
    Check(Pomade_Rasterise(ctx) == POMADE_OK, "levels ABI: K3 runs");
    Check(Pomade_BakeEnqueueLevels(bake) == POMADE_OK,
          "levels ABI: the enqueue with the hierarchy succeeds");
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
    while (Pomade_BakeCompletedVersion(bake) < Pomade_GetMapVersion(ctx) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(Pomade_BakeCompletedVersion(bake) == Pomade_GetMapVersion(ctx),
          "levels ABI: the bake completes");
    Check(Pomade_BakeDestroy(bake) == POMADE_OK,
          "levels ABI: the bake context destroys");
    Check(Pomade_Destroy(ctx) == POMADE_OK, "levels ABI: the model destroys");
    fs::remove_all(dir, ec);
}

// plan/02 §2.20: a face GeomSubset scalp still bakes every PARENT face
// (ptex ids are parent-derived and never renumbered), but a face the subset
// leaves out is never claimed: one texel of -1 even inside a region, on the
// synchronous CPU bake and on the worker (whose GPU lane classifies when a
// device exists). The same graph bound whole claims those faces.
void CheckSubsetBake()
{
    using namespace usdGenPomade;
    Grid const grid = MakeGrid(4);
    // Faces 5 and 6 (x in [1, 2], z in [1, 3]) lie inside region A.
    std::vector<int> const subsetFaces = {0, 1, 2, 3, 4, 7, 8, 9,
                                          10, 11, 12, 13, 14, 15};
    PomadeModel model;
    PomadeModel whole;
    Check(model.BindScalp(grid.points, grid.counts, grid.indices,
                          subsetFaces) &&
              whole.BindScalp(grid.points, grid.counts, grid.indices),
          "subset bake: the scalp binds through a face subset and whole");
    BuildAdjoining(&model, *model.GetScalp(), 4);
    BuildAdjoining(&whole, *whole.GetScalp(), 4);
    Check(model.Rasterise() && whole.Rasterise(), "subset bake: K3 runs");

    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeSubsetRegionBake";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    auto inputFor = [&](PomadeModel const &from) {
        PomadeBakeInput input;
        input.scalp = from.GetScalp();
        input.graph = from.GetGraph();
        input.outDir = dir.string();
        return input;
    };
    auto readFaces = [](std::string const &path,
                        std::vector<std::vector<float>> *out) {
        out->clear();
        Ptex::String error;
        PtexPtr<PtexTexture> tex(
            PtexTexture::open(path.c_str(), error, /*premultiply=*/false));
        if (!tex || tex->numChannels() != 1) {
            return false;
        }
        for (int f = 0; f < tex->numFaces(); ++f) {
            Ptex::Res const res = tex->getFaceInfo(f).res;
            out->emplace_back(size_t(res.size()), 0.0f);
            tex->getData(f, out->back().data(), 0);
        }
        return true;
    };
    auto claims = [](std::vector<float> const &texels, float id) {
        return std::find(texels.begin(), texels.end(), id) != texels.end();
    };
    auto leftOutUnclaimed = [](std::vector<std::vector<float>> const &faces) {
        bool ok = faces.size() == 16;
        for (int f : {5, 6}) {
            ok = ok && faces[size_t(f)].size() == 1 &&
                 faces[size_t(f)][0] == -1.0f;
        }
        return ok;
    };

    std::vector<std::vector<float>> texels;
    std::string const wholePath =
        (dir / PomadeBakeFileName("wholeMap", 1)).string();
    std::vector<std::vector<float>> cache;
    std::string err;
    Check(PomadeBakePtex(inputFor(whole), wholePath, nullptr, &cache, nullptr,
                        &err) &&
              readFaces(wholePath, &texels) && texels.size() == 16 &&
              claims(texels[5], 0.0f) && claims(texels[6], 0.0f),
          "subset bake: bound whole, region A claims faces 5 and 6: " + err);

    std::string const syncPath =
        (dir / PomadeBakeFileName("regionMap", 1)).string();
    cache.clear();
    Check(PomadeBakePtex(inputFor(model), syncPath, nullptr, &cache, nullptr,
                        &err) &&
              readFaces(syncPath, &texels) && leftOutUnclaimed(texels) &&
              claims(texels[1], 0.0f) && claims(texels[9], 1.0f),
          "subset bake: the CPU bake keeps 16 parent faces and bakes the "
          "left-out ones as one unclaimed texel: " + err);

    PomadeBakeWorker worker;
    uint64_t const version = model.GetMapVersion();
    worker.Enqueue(version, inputFor(model));
    std::string workerPath;
    uint64_t done = 0;
    Check(worker.WaitCompleted(version) &&
              worker.TakeCompleted(&done, &workerPath) && done == version &&
              readFaces(workerPath, &texels) && leftOutUnclaimed(texels) &&
              claims(texels[1], 0.0f) && claims(texels[9], 1.0f),
          "subset bake: the worker (GPU lane when present) agrees");
    fs::remove_all(dir, ec);
}

}  // namespace

int
main()
{
    CheckBakeAndExpression();
    CheckWorker();
    CheckSubfaceRegionBake();
    CheckSwapAndCommit();
    CheckSaveAndClump();
    CheckLevelChannels();
    CheckSweepAfterSwap();
    CheckTexelResolutionOverride();
    CheckBakeLevelsAbi();
    CheckSubsetBake();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
