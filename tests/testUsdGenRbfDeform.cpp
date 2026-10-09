// UsdGenDeform on the CPU lane: hair deformed by animated curves through a
// cubic RBF (plan/examples/rbf-deformation.md, examples/rbf-guides-plane.usda).
//
// rbf::CubicField, the math:
//   * the rest pose is the identity field;
//   * the field passes through every sample;
//   * any affine motion of the samples is reproduced everywhere;
//   * coplanar, coincident or too few samples are refused;
//   * SelectSamples drops duplicates and keeps a deterministic, spread subset.
// The example, cooked through the engine (stage builder):
//   * frame 1 (the drivers' rest pose) leaves the styled strands untouched;
//   * frame 20 bends them the way their nearest driver bends, roots locked;
//   * translating every driver translates every CV with lockRoots off;
//   * usdGen:mask 0 is a passthrough; a missing or planar driver set is a
//     diagnostic, not a silent identity.
// And through the groom scene index: moving the stage frame re-cooks the
// published strands. Stepping a frame in usdview's order (stage time, then
// the scene globals' current frame) cooks once and tells Hydra only what
// moved: no universal dirty, the tiles' points and occlusion but not their
// widths or colours, the drivers with the stage's own locators; the scene
// globals' frame alone cooks nothing (the example reads no $frame).
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/deform.h"
#include "usdGen/ops/rbfField.h"
#include "usdGen/ops/curveWrap.h"
#include "usdGen/scheduler.h"
#include "usdGen/workerPool.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/range3f.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/imaging/hdsi/sceneGlobalsSceneIndex.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/relationship.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
    else std::printf("ok: %s\n", what.c_str());
}

constexpr char const *kScene = USDGEN_TEST_SOURCE_DIR "/examples/rbf-guides-plane.usda";
SdfPath const kDescription("/World/Groom/Hair");
SdfPath const kDeform("/World/Groom/Hair/Ops/deform");
SdfPath const kDeformInput("/World/Groom/Hair/Ops/width");   // the operator before deform
SdfPath const kDrivers("/World/Drivers");

double Hash01(uint64_t k)
{
    k += 0x9e3779b97f4a7c15ULL;
    k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ULL;
    k = (k ^ (k >> 27)) * 0x94d049bb133111ebULL;
    return double((k ^ (k >> 31)) >> 11) / double(1ull << 53);
}

std::vector<GfVec3d> Cloud(size_t n, uint64_t seed)
{
    std::vector<GfVec3d> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = GfVec3d(Hash01(seed + i * 3), Hash01(seed + i * 3 + 1),
                         Hash01(seed + i * 3 + 2)) * 2.0 - GfVec3d(1.0);
    return out;
}

// --- the field -----------------------------------------------------------------

void CheckField()
{
    std::vector<GfVec3d> const rest = Cloud(40, 7);
    std::vector<GfVec3d> const queries = Cloud(200, 1001);
    rbf::CubicField field;
    std::string error;
    bool const bound = field.Bind(rest, &error);
    Check(bound, "a 3D sample cloud binds " + error);
    Check(field.Solve(rest, &error), "the rest pose solves");
    double identity = 0.0;
    for (GfVec3d const &q : queries) identity = std::max(identity, field.Displacement(q).GetLength());
    Check(identity < 1e-10, "the rest pose is the identity field (" + std::to_string(identity) + ")");

    // Arbitrary displacements: the field passes through every sample.
    std::vector<GfVec3d> moved(rest.size());
    for (size_t i = 0; i < rest.size(); ++i)
        moved[i] = rest[i] + GfVec3d(0.2 * std::sin(3.0 * rest[i][1]), 0.1 * rest[i][0] * rest[i][2],
                                     -0.15 * std::cos(2.0 * rest[i][0]));
    Check(field.Solve(moved, &error), "an arbitrary pose solves");
    double through = 0.0;
    for (size_t i = 0; i < rest.size(); ++i)
        through = std::max(through, (rest[i] + field.Displacement(rest[i]) - moved[i]).GetLength());
    Check(through < 1e-9, "the field passes through every sample (" + std::to_string(through) + ")");

    // Affine motion: rotation, non-uniform scale and translation.
    GfMatrix4d affine(1.0);
    affine.SetRotate(GfRotation(GfVec3d(0.3, 1.0, -0.2), 35.0));
    GfMatrix4d scale(1.0);
    scale.SetScale(GfVec3d(1.2, 0.9, 1.1));
    affine = scale * affine;
    affine.SetTranslateOnly(GfVec3d(0.4, -0.25, 0.1));
    for (size_t i = 0; i < rest.size(); ++i) moved[i] = affine.Transform(rest[i]);
    Check(field.Solve(moved, &error), "an affine pose solves");
    double reproduced = 0.0;
    for (GfVec3d const &q : queries)
        reproduced = std::max(reproduced, (q + field.Displacement(q) - affine.Transform(q)).GetLength());
    Check(reproduced < 1e-8, "affine motion of the samples is reproduced everywhere (" +
                                 std::to_string(reproduced) + ")");

    // Refusals.
    rbf::CubicField bad;
    std::vector<GfVec3d> planar = rest;
    for (GfVec3d &p : planar) p[1] = 0.3 * p[0] - 0.2 * p[2];   // a tilted plane
    bool refused = !bad.Bind(planar, &error) && error.find("span 3D") != std::string::npos;
    Check(refused, "coplanar samples are refused: " + error);
    refused = !bad.Bind({rest[0], rest[1], rest[2]}, &error) &&
        error.find("four") != std::string::npos;
    Check(refused, "three samples are refused: " + error);
    std::vector<GfVec3d> doubled = rest;
    doubled.push_back(rest[5]);
    refused = !bad.Bind(doubled, &error) && error.find("coincide") != std::string::npos;
    Check(refused, "coincident samples are refused: " + error);
    Check(!bad.Bound() && !bad.Solve(rest, &error), "a refused field does not solve");

    // Sample selection.
    std::vector<GfVec3d> points = Cloud(100, 42);
    points.push_back(points[3]);
    std::vector<size_t> const all = rbf::SelectSamples(points, 1000, 1e-9);
    Check(all.size() == 100 && std::find(all.begin(), all.end(), size_t(100)) == all.end(),
          "an exact duplicate is dropped");
    std::vector<size_t> const some = rbf::SelectSamples(points, 12, 1e-9);
    Check(some.size() == 12 && some == rbf::SelectSamples(points, 12, 1e-9) &&
          std::find(some.begin(), some.end(), size_t(0)) != some.end(),
          "a budget keeps a deterministic subset starting from the first point");
    double spread = 1e30;
    for (size_t a = 0; a < some.size(); ++a)
        for (size_t b = a + 1; b < some.size(); ++b)
            spread = std::min(spread, (points[some[a]] - points[some[b]]).GetLength());
    Check(spread > 0.4, "farthest-point samples are spread out (" + std::to_string(spread) + ")");
}

// DisplaceBatch is bitwise Displacement, through every cascade width (8,
// 4, tail singles), past the single-query path's 256-sample stack row,
// through the grouped-strand counts the deform strands loop issues (up to
// 300, covering the 256-query full group and its tail shapes), and for
// the unbound field.
void CheckBatchBitwise()
{
    for (size_t samples : {size_t(12), size_t(100), size_t(300)}) {
        std::vector<GfVec3d> const rest = Cloud(samples, 7 + samples);
        std::vector<GfVec3d> moved(rest.size());
        for (size_t i = 0; i < rest.size(); ++i)
            moved[i] = rest[i] + GfVec3d(0.2 * std::sin(3.0 * rest[i][1]),
                                         0.1 * rest[i][0] * rest[i][2],
                                         -0.15 * std::cos(2.0 * rest[i][0]));
        rbf::CubicField field;
        std::string error;
        if (!field.Bind(rest, &error) || !field.Solve(moved, &error)) {
            Check(false, "batch fixture binds (" + std::to_string(samples) + " samples)");
            continue;
        }
        std::vector<GfVec3d> const queries = Cloud(300, 1001);
        size_t worst = 0;
        for (size_t count = 0; count <= queries.size(); ++count) {
            std::vector<GfVec3d> batched(count), single(count);
            field.DisplaceBatch(queries.data(), batched.data(), count);
            for (size_t t = 0; t < count; ++t) single[t] = field.Displacement(queries[t]);
            if (count && !worst && std::memcmp(batched.data(), single.data(),
                                                count * sizeof(GfVec3d)) != 0)
                worst = count;
        }
        Check(worst == 0, "DisplaceBatch is bitwise Displacement (" +
                               std::to_string(samples) + " samples" +
                               (worst ? ", first diff at count " + std::to_string(worst) : "") + ")");
    }
    rbf::CubicField unbound;
    std::vector<GfVec3d> zeros(10, GfVec3d(1.0)), expect(10, GfVec3d(0.0));
    unbound.DisplaceBatch(zeros.data(), zeros.data(), zeros.size());
    Check(std::memcmp(zeros.data(), expect.data(), zeros.size() * sizeof(GfVec3d)) == 0,
          "an unbound field batches zeros");
}

// The NEON block agrees with the scalar blocks bitwise, through block
// boundaries (4k, 4k+1..3 tails, unaligned starts) and sample counts on
// both sides of the single-query path's 256-sample stack row. Off AArch64
// both paths are the scalar blocks and the test passes trivially.
void CheckBatchPathsAgree()
{
    for (size_t samples : {size_t(12), size_t(100), size_t(300)}) {
        std::vector<GfVec3d> const rest = Cloud(samples, 7 + samples);
        std::vector<GfVec3d> moved(rest.size());
        for (size_t i = 0; i < rest.size(); ++i)
            moved[i] = rest[i] + GfVec3d(0.2 * std::sin(3.0 * rest[i][1]),
                                         0.1 * rest[i][0] * rest[i][2],
                                         -0.15 * std::cos(2.0 * rest[i][0]));
        rbf::CubicField field;
        std::string error;
        if (!field.Bind(rest, &error) || !field.Solve(moved, &error)) {
            Check(false, "path-agreement fixture binds (" + std::to_string(samples) +
                             " samples)");
            continue;
        }
        // One extra query lets every count also run one triple over,
        // flipping the 16-byte alignment the vld3/vst3 pair sees.
        std::vector<GfVec3d> const queries = Cloud(302, 1001);
        size_t worst = 0;
        for (size_t count = 0; count <= 300; ++count) {
            for (size_t off = 0; off <= 1; ++off) {
                std::vector<GfVec3d> neon(count), scalar(count);
                rbf::TestForceScalarDisplace(false);
                field.DisplaceBatch(queries.data() + off, neon.data(), count);
                rbf::TestForceScalarDisplace(true);
                field.DisplaceBatch(queries.data() + off, scalar.data(), count);
                rbf::TestForceScalarDisplace(false);
                if (count && !worst &&
                    std::memcmp(neon.data(), scalar.data(), count * sizeof(GfVec3d)) != 0)
                    worst = count * 2 + off;
            }
        }
        Check(worst == 0, "NEON and scalar DisplaceBatch agree bitwise (" +
                               std::to_string(samples) + " samples" +
                               (worst ? ", first diff at " + std::to_string(worst) : "") + ")");
    }
}

// DisplaceBatchPlanar is bitwise Displacement, through every cascade width
// (8, 4, tail singles), on both the NEON and the forced-scalar path, and
// for the unbound field. The planar queries round-trip through float, so
// the singles compare against the same float triples the batch converts.
void CheckBatchPlanarBitwise()
{
    for (size_t samples : {size_t(12), size_t(100), size_t(300)}) {
        std::vector<GfVec3d> const rest = Cloud(samples, 7 + samples);
        std::vector<GfVec3d> moved(rest.size());
        for (size_t i = 0; i < rest.size(); ++i)
            moved[i] = rest[i] + GfVec3d(0.2 * std::sin(3.0 * rest[i][1]),
                                         0.1 * rest[i][0] * rest[i][2],
                                         -0.15 * std::cos(2.0 * rest[i][0]));
        rbf::CubicField field;
        std::string error;
        if (!field.Bind(rest, &error) || !field.Solve(moved, &error)) {
            Check(false, "planar batch fixture binds (" + std::to_string(samples) +
                             " samples)");
            continue;
        }
        std::vector<GfVec3d> const cloud = Cloud(300, 1001);
        std::vector<float> qx(300), qy(300), qz(300);
        for (size_t i = 0; i < 300; ++i) {
            qx[i] = float(cloud[i][0]);
            qy[i] = float(cloud[i][1]);
            qz[i] = float(cloud[i][2]);
        }
        size_t worst = 0;
        for (int force = 0; force <= 1 && !worst; ++force) {
            rbf::TestForceScalarDisplace(force != 0);
            for (size_t count = 0; count <= 300; ++count) {
                std::vector<double> dx(count), dy(count), dz(count);
                field.DisplaceBatchPlanar(qx.data(), qy.data(), qz.data(), dx.data(),
                                          dy.data(), dz.data(), count);
                for (size_t t = 0; t < count; ++t) {
                    GfVec3d const s =
                        field.Displacement(GfVec3d(qx[t], qy[t], qz[t]));
                    if (std::memcmp(&dx[t], &s[0], sizeof(double)) != 0 ||
                        std::memcmp(&dy[t], &s[1], sizeof(double)) != 0 ||
                        std::memcmp(&dz[t], &s[2], sizeof(double)) != 0) {
                        worst = count * 2 + size_t(force) + 1;
                        break;
                    }
                }
                if (worst) break;
            }
        }
        rbf::TestForceScalarDisplace(false);
        Check(worst == 0, "DisplaceBatchPlanar is bitwise Displacement (" +
                               std::to_string(samples) + " samples" +
                               (worst ? ", first diff at " + std::to_string(worst) : "") +
                               ")");
    }
    rbf::CubicField unbound;
    std::vector<float> qx(10, 1.0f), qy(10, 2.0f), qz(10, 3.0f);
    std::vector<double> dx(10, 1.0), dy(10, 1.0), dz(10, 1.0);
    unbound.DisplaceBatchPlanar(qx.data(), qy.data(), qz.data(), dx.data(), dy.data(),
                               dz.data(), dx.size());
    std::vector<double> expect(10, 0.0);
    Check(std::memcmp(dx.data(), expect.data(), dx.size() * sizeof(double)) == 0 &&
              std::memcmp(dy.data(), expect.data(), dy.size() * sizeof(double)) == 0 &&
              std::memcmp(dz.data(), expect.data(), dz.size() * sizeof(double)) == 0,
          "an unbound field batches planar zeros");
}

// Large batches split over the internal pool (16k queries and up, in
// 1k-query slices): counts around the threshold and the slice edges
// stay bitwise Displacement, on both entries and both scalar-seam
// positions.
void CheckBatchThreaded()
{
    size_t const samples = 100;
    std::vector<GfVec3d> const rest = Cloud(samples, 7 + samples);
    std::vector<GfVec3d> moved(rest.size());
    for (size_t i = 0; i < rest.size(); ++i)
        moved[i] = rest[i] + GfVec3d(0.2 * std::sin(3.0 * rest[i][1]),
                                     0.1 * rest[i][0] * rest[i][2],
                                     -0.15 * std::cos(2.0 * rest[i][0]));
    rbf::CubicField field;
    std::string error;
    if (!field.Bind(rest, &error) || !field.Solve(moved, &error)) {
        Check(false, "threaded batch fixture binds");
        return;
    }
    size_t const n = 20000;
    std::vector<GfVec3d> const cloud = Cloud(n, 1001);
    std::vector<float> qx(n), qy(n), qz(n);
    for (size_t i = 0; i < n; ++i) {
        qx[i] = float(cloud[i][0]);
        qy[i] = float(cloud[i][1]);
        qz[i] = float(cloud[i][2]);
    }
    size_t worst = 0;
    for (size_t count : {size_t(16383), size_t(16384), size_t(16385),
                         size_t(17407), size_t(17408), size_t(17409),
                         size_t(19999), size_t(20000)}) {
        for (int force = 0; force <= 1 && !worst; ++force) {
            rbf::TestForceScalarDisplace(force != 0);
            std::vector<GfVec3d> batched(count), single(count);
            field.DisplaceBatch(cloud.data(), batched.data(), count);
            for (size_t t = 0; t < count; ++t)
                single[t] = field.Displacement(cloud[t]);
            if (std::memcmp(batched.data(), single.data(),
                            count * sizeof(GfVec3d)) != 0) {
                worst = count * 4 + size_t(force) + 1;
                break;
            }
            std::vector<double> dx(count), dy(count), dz(count);
            field.DisplaceBatchPlanar(qx.data(), qy.data(), qz.data(),
                                      dx.data(), dy.data(), dz.data(), count);
            for (size_t t = 0; t < count; ++t) {
                GfVec3d const s =
                    field.Displacement(GfVec3d(qx[t], qy[t], qz[t]));
                if (std::memcmp(&dx[t], &s[0], sizeof(double)) != 0 ||
                    std::memcmp(&dy[t], &s[1], sizeof(double)) != 0 ||
                    std::memcmp(&dz[t], &s[2], sizeof(double)) != 0) {
                    worst = count * 4 + size_t(force) + 3;
                    break;
                }
            }
        }
        if (worst) break;
    }
    rbf::TestForceScalarDisplace(false);
    Check(worst == 0, "threaded batches are bitwise Displacement" +
                           (worst ? " (first diff at " + std::to_string(worst) + ")" : ""));
}

// The pool discipline holds: concurrent large batches from many threads
// serialize and stay bitwise, and large batches issued from another
// pool's workers block-join the displace pool without deadlocking.
void CheckBatchConcurrent()
{
    size_t const samples = 100;
    std::vector<GfVec3d> const rest = Cloud(samples, 7 + samples);
    std::vector<GfVec3d> moved(rest.size());
    for (size_t i = 0; i < rest.size(); ++i)
        moved[i] = rest[i] + GfVec3d(0.2 * std::sin(3.0 * rest[i][1]),
                                     0.1 * rest[i][0] * rest[i][2],
                                     -0.15 * std::cos(2.0 * rest[i][0]));
    rbf::CubicField field;
    std::string error;
    if (!field.Bind(rest, &error) || !field.Solve(moved, &error)) {
        Check(false, "concurrent batch fixture binds");
        return;
    }
    size_t const n = 20000;
    std::vector<GfVec3d> const cloud = Cloud(n, 1001);
    std::vector<GfVec3d> expect(n);
    for (size_t t = 0; t < n; ++t) expect[t] = field.Displacement(cloud[t]);
    std::atomic<size_t> bad{0};
    auto run = [&]() {
        for (int k = 0; k < 4; ++k) {
            std::vector<GfVec3d> got(n);
            field.DisplaceBatch(cloud.data(), got.data(), n);
            if (std::memcmp(got.data(), expect.data(), n * sizeof(GfVec3d)) != 0)
                bad.fetch_add(1, std::memory_order_relaxed);
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back(run);
    for (std::thread &t : threads) t.join();
    Check(bad.load() == 0, "concurrent large batches stay bitwise");
    struct Nested {
        rbf::CubicField const *field;
        GfVec3d const *qs;
        GfVec3d *ds0, *ds1;
        size_t count;
    };
    std::vector<GfVec3d> nestedOut0(n), nestedOut1(n);
    Nested nested{&field, cloud.data(), nestedOut0.data(), nestedOut1.data(), n};
    UsdGenWorkerPool outer(2);
    outer.ParallelFor(2,
                      [](size_t i, void *p) {
                          Nested const &job = *static_cast<Nested const *>(p);
                          job.field->DisplaceBatch(job.qs, i == 0 ? job.ds0 : job.ds1,
                                                  job.count);
                      },
                      &nested);
    Check(std::memcmp(nestedOut0.data(), expect.data(), n * sizeof(GfVec3d)) == 0 &&
              std::memcmp(nestedOut1.data(), expect.data(), n * sizeof(GfVec3d)) == 0,
          "batches nested in pool workers stay bitwise");
}

// --- the example, through the engine ---------------------------------------------

struct Cooked {
    bool ok = false;
    std::vector<std::string> errors;
    UsdGenCurveBuffer input, output;   // the deform node's input and output
    uint32_t cvs = 0;
};

Cooked Cook(UsdStageRefPtr const &stage, double time,
            SdfPath const &description = kDescription,
            SdfPath const &deform = kDeform,
            SdfPath const &input = kDeformInput)
{
    Cooked out;
    usdGenImaging::UsdGenGraphDescBuildOptions options;
    options.time = time;
    UsdGenGraphDesc const desc = usdGenImaging::BuildGraphDescFromStage(stage, description, options);
    out.errors = desc.validationErrors;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        out.errors.insert(out.errors.end(), compiled.errors.begin(), compiled.errors.end());
        return out;
    }
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext context;
    context.time = time;
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    out.errors.insert(out.errors.end(), run.diagnostics.errors.begin(), run.diagnostics.errors.end());
    if (run.diagnostics.HasErrors()) return out;
    out.input = graph.Node(graph.NodeIdForPath(input)).buffer;
    out.output = graph.Node(graph.NodeIdForPath(deform)).buffer;
    out.ok = out.output.totalCurves > 0 && out.output.totalCvs == out.input.totalCvs &&
        out.output.totalCvs % out.output.totalCurves == 0;
    out.cvs = out.ok ? out.output.totalCvs / out.output.totalCurves : 0;
    return out;
}

GfVec3f P(UsdGenCurveBuffer const &b, size_t cv) { return GfVec3f(b.px[cv], b.py[cv], b.pz[cv]); }

double MaxShift(Cooked const &c, bool rootsOnly)
{
    double shift = 0.0;
    for (size_t cv = 0; cv < c.output.totalCvs; ++cv) {
        if (rootsOnly && cv % c.cvs != 0) continue;
        shift = std::max(shift, double((P(c.output, cv) - P(c.input, cv)).GetLength()));
    }
    return shift;
}

std::string Errors(Cooked const &c)
{
    std::string text;
    for (auto const &e : c.errors) text += "\n  " + e;
    return text;
}

void CheckExample()
{
    UsdStageRefPtr const stage = UsdStage::Open(kScene);
    if (!stage) { Check(false, std::string("cannot open ") + kScene); return; }

    Cooked const rest = Cook(stage, 1.0);
    Check(rest.ok, "the example cooks at frame 1" + Errors(rest));
    if (!rest.ok) return;
    std::printf("  %u strands, %u CVs each\n", rest.output.totalCurves, rest.cvs);
    Check(MaxShift(rest, false) < 1e-5, "frame 1 (the drivers' rest pose) leaves the strands untouched");

    Cooked const posed = Cook(stage, 20.0);
    Check(posed.ok, "the example cooks at frame 20" + Errors(posed));
    if (!posed.ok) return;
    Check(MaxShift(posed, true) < 1e-5, "usdGen:lockRoots keeps every root in place");
    double tips = 0.0;
    for (size_t c = 0; c < posed.output.totalCurves; ++c) {
        size_t const tip = (c + 1) * posed.cvs - 1;
        tips += (P(posed.output, tip) - P(posed.input, tip)).GetLength();
    }
    tips /= double(posed.output.totalCurves);
    std::printf("  mean tip displacement at frame 20: %.3f\n", tips);
    Check(tips > 0.05, "frame 20 bends the strands");

    // Each strand bends the way the driver rooted nearest it bends.
    UsdGeomBasisCurves drivers(stage->GetPrimAtPath(kDrivers));
    VtIntArray counts;
    VtVec3fArray restPoints, posePoints;
    drivers.GetCurveVertexCountsAttr().Get(&counts);
    drivers.GetPointsAttr().Get(&restPoints, UsdTimeCode::Default());
    drivers.GetPointsAttr().Get(&posePoints, UsdTimeCode(20.0));
    std::vector<GfVec3f> roots, tipMotion;
    size_t first = 0;
    for (int n : counts) {
        roots.push_back(restPoints[first]);
        tipMotion.push_back(posePoints[first + n - 1] - restPoints[first + n - 1]);
        first += size_t(n);
    }
    size_t agree = 0, considered = 0;
    for (size_t c = 0; c < posed.output.totalCurves; ++c) {
        GfVec3f const root = P(posed.input, c * posed.cvs);
        size_t nearest = 0;
        for (size_t g = 1; g < roots.size(); ++g)
            if ((roots[g] - root).GetLengthSq() < (roots[nearest] - root).GetLengthSq()) nearest = g;
        if ((roots[nearest] - root).GetLength() > 0.15 || tipMotion[nearest].GetLength() < 0.05) continue;
        size_t const tip = (c + 1) * posed.cvs - 1;
        GfVec3f const motion = P(posed.output, tip) - P(posed.input, tip);
        ++considered;
        if (GfDot(motion, tipMotion[nearest]) > 0.0f) ++agree;
    }
    std::printf("  %zu of %zu strands near a moving driver follow it\n", agree, considered);
    Check(considered > 50 && agree == considered, "strands near a driver bend the way it bends");

    // A rigid translation of every driver moves every CV by it (lockRoots off).
    VtVec3fArray shifted = restPoints;
    GfVec3f const offset(0.1f, 0.05f, -0.03f);
    for (GfVec3f &p : shifted) p += offset;
    drivers.GetPointsAttr().Set(shifted, UsdTimeCode(100.0));
    UsdAttribute lock = stage->GetAttributeAtPath(kDeform.AppendProperty(TfToken("usdGen:lockRoots")));
    lock.Set(false);
    Cooked const moved = Cook(stage, 100.0);
    Check(moved.ok, "the translated pose cooks" + Errors(moved));
    double error = 0.0;
    for (size_t cv = 0; moved.ok && cv < moved.output.totalCvs; ++cv)
        error = std::max(error, double((P(moved.output, cv) - P(moved.input, cv) - offset).GetLength()));
    Check(moved.ok && error < 2e-4, "translating the drivers translates the whole groom (" +
                                        std::to_string(error) + ")");
    lock.Set(true);

    // The envelope.
    UsdAttribute mask = stage->GetAttributeAtPath(kDeform.AppendProperty(TfToken("usdGen:mask")));
    Check(mask && mask.Set(0.0f), "author usdGen:mask = 0");
    Cooked const muted = Cook(stage, 20.0);
    Check(muted.ok && MaxShift(muted, false) < 1e-6, "mask 0 leaves the strands untouched");
    mask.Set(1.0f);

    // Refusals are diagnostics, not a silent identity.
    UsdRelationship guides = stage->GetPrimAtPath(kDeform).GetRelationship(TfToken("usdGen:guides"));
    guides.ClearTargets(true);
    Cooked const unguided = Cook(stage, 20.0);
    bool mentions = false;
    for (auto const &e : unguided.errors)
        mentions |= e.find("span 3D") != std::string::npos || e.find("rest") != std::string::npos;
    Check(!unguided.ok && mentions, "without guides the planar/rest-less surface is refused" + Errors(unguided));

    UsdGeomBasisCurves flat = UsdGeomBasisCurves::Define(stage, SdfPath("/World/FlatDrivers"));
    flat.GetPrim().AddAppliedSchema(TfToken("UsdGenCurveAPI"));
    flat.GetCurveVertexCountsAttr().Set(VtIntArray{2, 2, 2});
    flat.GetPointsAttr().Set(VtVec3fArray{GfVec3f(-1, 0, -1), GfVec3f(1, 0, -1), GfVec3f(-1, 0, 1),
                                          GfVec3f(1, 0, 1), GfVec3f(0, 0, 0), GfVec3f(0.5f, 0, 0.2f)});
    guides.SetTargets({SdfPath("/World/FlatDrivers")});
    Cooked const planar = Cook(stage, 20.0);
    mentions = false;
    for (auto const &e : planar.errors) mentions |= e.find("span 3D") != std::string::npos;
    Check(!planar.ok && mentions, "drivers in one plane are refused" + Errors(planar));
    guides.SetTargets({kDrivers});
}

void CheckSurfaceExample(bool outside = false)
{
    std::string const scene = std::string(USDGEN_TEST_SOURCE_DIR) + "/examples/motion/" +
        (outside ? "felt_sphere_groom_outside_xform.usda" : "felt_sphere_animated.usda");
    auto stage = UsdStage::Open(scene);
    Check(bool(stage), "animated sphere example opens");
    if (!stage) return;
    SdfPath const description(outside ? "/World/Groom/Hair" : "/World/Motion/Groom/Hair");
    SdfPath const deform = description.AppendPath(SdfPath("Ops/surfaceAnimate"));
    SdfPath const input = description.AppendPath(SdfPath("Ops/width"));
    auto cook = [&](double time) { return Cook(stage, time, description, deform, input); };
    Cooked const rest = cook(1);
    Check(rest.ok && (outside || MaxShift(rest, false) < 1e-6), "surface RBF cooks the rest groom" + Errors(rest));
    if (!rest.ok) return;
    for (int frame : {13, 26, 38, 51, 75, 100}) {
        Cooked const pose = cook(frame);
        bool stable = pose.ok && pose.input.totalCvs == rest.input.totalCvs;
        double error = 0;
        double const stretch = std::exp(.48 * std::sin(4 * 3.141592653589793 * (frame-1)/99.0));
        double const wide = 1 / std::sqrt(stretch);
        for (size_t cv = 0; stable && cv < pose.input.totalCvs; ++cv) {
            stable = P(pose.input,cv) == P(rest.input,cv);
            GfVec3f const p = P(rest.input,cv);
            GfVec3f expected(float(p[0]*wide),float(p[1]*wide),float(p[2]*stretch));
            if (outside) expected += GfVec3f(float(.55*std::sin(2*3.141592653589793*(frame-1)/99.0)),0,.5f);
            error = std::max(error,double((P(pose.output,cv)-expected).GetLength()));
        }
        Check(stable, "Scatter/Grow/style remain rest-local at frame " + std::to_string(frame) + Errors(pose));
        Check(pose.ok && error < 2e-6, "surface RBF follows squash/stretch without double parent motion at frame " +
              std::to_string(frame) + " error=" + std::to_string(error));
    }
    // Opening at a posed frame must still use Default-time rest, not first pull.
    auto posedFirst = UsdStage::Open(scene);
    Cooked const first = Cook(posedFirst,38,description,deform,input);
    bool same = first.ok && first.input.totalCvs == rest.input.totalCvs;
    for (size_t cv=0; same && cv<first.input.totalCvs; ++cv)
        same = P(first.input,cv) == P(rest.input,cv);
    Check(same,"opening on frame 38 generates from Default, not the animated input");
    // A time-sampled surface with no default must fail, not bind to first pull.
    auto prim=stage->GetPrimAtPath(SdfPath("/World/Motion/Sphere"));
    prim.GetAttribute(TfToken("points")).ClearAtTime(UsdTimeCode::Default());
    prim.RemoveProperty(TfToken("primvars:rest"));
    Cooked const missing = cook(38);
    Check(!missing.ok,"surface RBF refuses missing Default-time rest data");
}

// Finite drivers, but the field overflows float at a far CV: the capture
// must fail with the non-finite diagnostic, not publish infinities.
void CheckNonFiniteCapture()
{
    UsdGenGraphDesc desc;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/surface");
    int const N = 5;
    surface.restPoints = VtVec3fArray(N * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            float const x = float(i) * 0.1f, y = float(j) * 0.1f;
            surface.restPoints[j * N + i] =
                GfVec3f(x, y, 2.0f * std::sin(x * 0.3f) * std::cos(y * 0.3f));
        }
    surface.points = surface.restPoints;
    surface.points[0] = surface.restPoints[0] + GfVec3f(1e10f, 0.0f, 0.0f);
    desc.surfaces.push_back(surface);

    UsdGenCurveBuffer upstream;
    upstream.totalCurves = 1;
    upstream.totalCvs = 2;
    upstream.px = VtFloatArray(2);
    upstream.py = VtFloatArray(2);
    upstream.pz = VtFloatArray(2);
    upstream.px[1] = 1e20f;  // the cubic kernel overflows float here

    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.surface = 0;
    UsdGenDeformOp op;
    auto capture = op.CreateCapture();
    UsdGenDiagnostics diagnostics;
    bool const ok = op.Capture(ctx, upstream, capture.get(), &diagnostics);
    bool const refused = !ok && !diagnostics.errors.empty() &&
        diagnostics.errors[0].find("non-finite") != std::string::npos;
    Check(refused, "a deformation that overflows float is refused, not published");
}

// The surface-driven capture digest hashes only the posed drivers the
// capture reads: rest decides the farthest-point selection, so a move
// outside the chosen set must neither re-capture nor change the output,
// while a move at a chosen driver must re-capture to new output.
UsdGenGraphDesc MakeSmallSurfaceDeformDesc()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.terminal = SdfPath("/groom/deform");
    desc.time = 0.0;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    int const NX = 15, NY = 15;
    s.restPoints = VtVec3fArray((NX + 1) * (NY + 1));
    s.uv = VtVec2fArray((NX + 1) * (NY + 1));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            int const k = j * (NX + 1) + i;
            float const x = float(i) * 0.1f, y = float(j) * 0.1f;
            s.restPoints[k] = GfVec3f(x, y, 2.0f * std::sin(x * 0.3f) * std::cos(y * 0.3f));
            s.uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
        }
    s.points = s.restPoints;
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            int const a = j * (NX + 1) + i, o = (j * NX + i) * 4;
            s.faceVertexIndices[o] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    desc.surfaces.push_back(s);
    auto addNode = [&](std::string const &name, TfToken type, std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = type;
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (type == TfToken("UsdGenScatter") || type == TfToken("UsdGenDeform"))
            n.surfaces.push_back(SdfPath("/groom/surface"));
        desc.nodes.push_back(std::move(n));
    };
    addNode("scatter", TfToken("UsdGenScatter"), "", 42);
    addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
    addNode("deform", TfToken("UsdGenDeform"), "grow", 44);
    auto setp = [&](std::string const &name, TfToken p, VtValue v) {
        for (auto &n : desc.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{p, v, false});
    };
    setp("scatter", TfToken("density"), VtValue(25.0));
    setp("grow", TfToken("segments"), VtValue(2));
    setp("grow", TfToken("length"), VtValue(1.0));
    setp("deform", TfToken("rbfSamples"), VtValue(100));
    setp("deform", TfToken("lockRoots"), VtValue(true));
    return desc;
}

void CheckDeformChosenDigest()
{
    UsdGenGraphDesc desc = MakeSmallSurfaceDeformDesc();

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "the digest fixture compiles");
    if (!compiled.ok) return;
    UsdGenNodeId const deformId = graph.NodeIdForPath(SdfPath("/groom/deform"));
    auto *deformOp = static_cast<UsdGenDeformOp *>(graph.Node(deformId).op.get());
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    uint64_t gen = 0;
    auto runPose = [&]() -> bool {
        UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
        ctx.desc = &graph.Desc();
        UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
        return rr.ok && !run.diagnostics.HasErrors();
    };
    auto outputBits = [&]() {
        UsdGenCurveBuffer const &b = graph.Node(deformId).buffer;
        std::vector<float> bits;
        bits.reserve(b.px.size() + b.py.size() + b.pz.size() + 2);
        bits.push_back(float(b.totalCurves));
        bits.push_back(float(b.totalCvs));
        bits.insert(bits.end(), b.px.begin(), b.px.end());
        bits.insert(bits.end(), b.py.begin(), b.py.end());
        bits.insert(bits.end(), b.pz.begin(), b.pz.end());
        return bits;
    };

    Check(runPose(), "the digest fixture cooks its rest pose");
    if (graph.Node(deformId).buffer.totalCurves == 0) {
        Check(false, "the digest fixture grows curves");
        return;
    }
    std::vector<float> const restBits = outputBits();
    std::vector<size_t> const selection = deformOp->SurfaceSelectionForTesting();
    bool selectionSane = selection.size() == 100;
    for (size_t k : selection) selectionSane &= k < desc.surfaces[0].restPoints.size();
    Check(selectionSane, "the rest pose settles a 100-driver selection");
    Check(deformOp->ChosenDigestHitsForTesting() == 0,
          "the first digest hashes the whole posed surface (no cache yet)");
    if (!selectionSane) return;
    std::vector<char> chosenMark(desc.surfaces[0].restPoints.size(), 0);
    for (size_t k : selection) chosenMark[k] = 1;

    // Half the unchosen drivers jump: the digest must not move, so the
    // output is the rest output bit for bit. Rest is read through a const
    // reference: desc is non-const, so a direct restPoints[v] would take
    // the mutating subscript and detach the shared rest buffer, and the
    // selection cache (keyed on that buffer's identity) would miss.
    VtVec3fArray const &rest = desc.surfaces[0].restPoints;
    int moved = 0;
    for (size_t v = 0; v < chosenMark.size() && moved < 50; ++v) {
        if (chosenMark[v]) continue;
        desc.surfaces[0].points[v] = rest[v] + GfVec3f(50.0f, 0.0f, 0.0f);
        ++moved;
    }
    Check(moved == 50, "the fixture has unchosen drivers to move");
    Check(runPose(), "the unchosen-move pose cooks");
    Check(outputBits() == restBits, "moving only unchosen drivers leaves the output bitwise alone");
    Check(deformOp->ChosenDigestHitsForTesting() == 1,
          "the unchosen-move digest hashed only the chosen drivers");
    Check(deformOp->SurfaceSelectionForTesting() == selection,
          "posed-only motion keeps the rest-decided selection");

    // One chosen driver jumps: the digest must move and re-capture to
    // output that differs.
    desc.surfaces[0].points = desc.surfaces[0].restPoints;
    desc.surfaces[0].points[selection[0]] = rest[selection[0]] + GfVec3f(50.0f, 0.0f, 0.0f);
    Check(runPose(), "the chosen-move pose cooks");
    Check(outputBits() != restBits, "moving a chosen driver re-captures to new output");
    Check(deformOp->ChosenDigestHitsForTesting() == 2,
          "the chosen-move digest hashed only the chosen drivers");
    Check(deformOp->SurfaceSelectionForTesting() == selection,
          "the re-capture reuses the rest-decided selection");
}

void CheckDeformEvaluateViewShapes()
{
    // Deform Evaluate must be view-shape transparent: one real capture,
    // re-evaluated through a whole-groom uniform view, a ragged view over
    // identical spans, and an offset chunk, agrees with the cooked
    // terminal bitwise on every path (full-mask copy and partial-mask
    // blend). No goldens: every comparison is within this run, so the
    // test holds on any platform. This pins the contract a future
    // uniform fast path must satisfy.
    UsdGenGraphDesc desc = MakeSmallSurfaceDeformDesc();
    // Smooth bend so the capture displaces nearly every CV.
    VtVec3fArray const &rest = desc.surfaces[0].restPoints;
    desc.surfaces[0].points.resize(rest.size());
    for (size_t i = 0; i < rest.size(); ++i) {
        GfVec3f const r = rest[i];
        float const bend =
            float(0.6 * std::sin(0.05 * r[0] + 0.35) * std::cos(0.04 * r[1] - 0.175));
        desc.surfaces[0].points[i] = GfVec3f(r[0], r[1], r[2] + bend);
    }
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "the view-shape fixture compiles");
    UsdGenNodeId const deformId = graph.NodeIdForPath(SdfPath("/groom/deform"));
    auto *op = static_cast<UsdGenDeformOp *>(graph.Node(deformId).op.get());
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    uint64_t gen = 0;
    UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
    ctx.desc = &graph.Desc();
    UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
    Check(rr.ok && !run.diagnostics.HasErrors(), "the view-shape pose cooks");
    UsdGenCurveBuffer const &term = graph.Node(deformId).buffer;
    UsdGenCurveBuffer const &up = graph.Node(graph.Node(deformId).input).buffer;
    if (term.totalCurves < 2 || term.totalCvs == 0 || up.px.size() != term.totalCvs ||
        term.totalCvs % term.totalCurves != 0 || !term.cvOffsets.empty()) {
        Check(false, "the view-shape fixture grows a uniform groom");
        return;
    }
    // Vacuity guard: the pose must actually move points.
    bool moved = false;
    for (size_t i = 0; i < term.totalCvs && !moved; ++i)
        moved = term.px[i] != up.px[i] || term.py[i] != up.py[i] || term.pz[i] != up.pz[i];
    Check(moved, "the view-shape pose displaces points");
    if (!moved) return;
    size_t const curves = term.totalCurves, perCurve = term.totalCvs / curves;
    auto same = [](std::vector<float> const &a, std::vector<float> const &b) {
        return a.size() == b.size() &&
            std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    };
    UsdGenChunkDesc fullDesc;
    fullDesc.firstCurve = 0;
    fullDesc.curveCount = uint32_t(curves);
    fullDesc.firstCv = 0;
    fullDesc.cvCount = uint32_t(perCurve);
    auto makeView = [&](UsdGenChunkDesc *d, float *px, float *py, float *pz,
                        size_t inBase, uint32_t nCurves, uint32_t cvCount,
                        int const *cvOffsets) {
        UsdGenChunkView v{};
        v.desc = d;
        v.px = px;
        v.py = py;
        v.pz = pz;
        v.inPx = up.px.cdata() + inBase;
        v.inPy = up.py.cdata() + inBase;
        v.inPz = up.pz.cdata() + inBase;
        v.curveCount = nCurves;
        v.cvCount = cvCount;
        v.cvOffsets = cvOffsets;
        return v;
    };
    UsdGenEvalContext evalCtx; // params null: default mask 1.0
    std::vector<float> uniPx(term.totalCvs), uniPy(term.totalCvs), uniPz(term.totalCvs);
    UsdGenChunkView uniView = makeView(&fullDesc, uniPx.data(), uniPy.data(), uniPz.data(),
                                       0, uint32_t(curves), uint32_t(perCurve), nullptr);
    op->Evaluate(evalCtx, *graph.Node(deformId).capture, &uniView);
    auto plane = [&](VtFloatArray const &p) {
        return std::vector<float>(p.begin(), p.end());
    };
    Check(same(uniPx, plane(term.px)) && same(uniPy, plane(term.py)) &&
              same(uniPz, plane(term.pz)),
          "a whole-groom uniform re-evaluate matches the terminal bitwise");
    // Ragged view, identical spans: the ragged path must agree bitwise.
    std::vector<int> offsets(curves + 1);
    for (size_t c = 0; c <= curves; ++c) offsets[c] = int(c * perCurve);
    UsdGenChunkDesc raggedDesc = fullDesc;
    raggedDesc.cvCount = 0;
    std::vector<float> ragPx(term.totalCvs), ragPy(term.totalCvs), ragPz(term.totalCvs);
    UsdGenChunkView ragView = makeView(&raggedDesc, ragPx.data(), ragPy.data(), ragPz.data(),
                                       0, uint32_t(curves), 0, offsets.data());
    op->Evaluate(evalCtx, *graph.Node(deformId).capture, &ragView);
    Check(same(uniPx, ragPx) && same(uniPy, ragPy) && same(uniPz, ragPz),
          "ragged and uniform views agree bitwise at full mask");
    // Offset chunk (second half of the curves): the firstCv math.
    size_t const half = curves / 2;
    UsdGenChunkDesc tailDesc;
    tailDesc.firstCurve = uint32_t(half);
    tailDesc.curveCount = uint32_t(curves - half);
    tailDesc.firstCv = uint32_t(half * perCurve);
    tailDesc.cvCount = uint32_t(perCurve);
    size_t const tailCvs = (curves - half) * perCurve;
    std::vector<float> tailPx(tailCvs), tailPy(tailCvs), tailPz(tailCvs);
    UsdGenChunkView tailView = makeView(&tailDesc, tailPx.data(), tailPy.data(), tailPz.data(),
                                        half * perCurve, uint32_t(curves - half),
                                        uint32_t(perCurve), nullptr);
    op->Evaluate(evalCtx, *graph.Node(deformId).capture, &tailView);
    bool tailOk = true;
    for (size_t i = 0; i < tailCvs && tailOk; ++i) {
        size_t const o = half * perCurve + i;
        tailOk = tailPx[i] == uniPx[o] && tailPy[i] == uniPy[o] && tailPz[i] == uniPz[o];
    }
    Check(tailOk, "an offset chunk matches its slice of the full evaluate");
    // Partial uniform mask: ragged and uniform agree on the blend path too.
    UsdGenNodeDesc maskNode;
    maskNode.params.push_back(UsdGenParamValue{TfToken("mask"), VtValue(0.5), false});
    UsdGenParamView maskView;
    maskView.node = &maskNode;
    UsdGenEvalContext maskCtx;
    maskCtx.params = &maskView;
    std::vector<float> blendUniPx(term.totalCvs), blendUniPy(term.totalCvs),
        blendUniPz(term.totalCvs);
    UsdGenChunkView blendUniView =
        makeView(&fullDesc, blendUniPx.data(), blendUniPy.data(), blendUniPz.data(), 0,
                 uint32_t(curves), uint32_t(perCurve), nullptr);
    op->Evaluate(maskCtx, *graph.Node(deformId).capture, &blendUniView);
    std::vector<float> blendRagPx(term.totalCvs), blendRagPy(term.totalCvs),
        blendRagPz(term.totalCvs);
    UsdGenChunkView blendRagView =
        makeView(&raggedDesc, blendRagPx.data(), blendRagPy.data(), blendRagPz.data(), 0,
                 uint32_t(curves), 0, offsets.data());
    op->Evaluate(maskCtx, *graph.Node(deformId).capture, &blendRagView);
    Check(same(blendUniPx, blendRagPx) && same(blendUniPy, blendRagPy) &&
              same(blendUniPz, blendRagPz),
          "ragged and uniform views agree bitwise at partial mask");
}

void CheckCurveWrapField()
{
    curveWrap::Field field;
    std::vector<GfVec3d> rest={{0,0,0},{0,0,.25},{0,0,.5},{0,0,.75},{0,0,1}};
    std::string error;
    Check(field.Bind(rest,rest,&error),"straight centerline binds without a 3D RBF cage");
    auto queries=Cloud(100,456);
    double identity=0;
    for (auto const &q:queries) identity=std::max(identity,(field.Map(q)-q).GetLength());
    Check(identity<1e-12,"curveWrap rest pose is identity, including endpoint offsets");
    auto pose=rest;
    GfRotation rotation(GfVec3d(0,1,0),65);
    GfVec3d shift(.2,.1,-.3);
    for (auto &p:pose) p=rotation.TransformDir(p)+shift;
    Check(field.Bind(rest,pose,&error),"single centerline rotation and translation bind");
    double rigid=0;
    for (auto const &q:queries) rigid=std::max(rigid,(field.Map(q)-rotation.TransformDir(q)-shift).GetLength());
    Check(rigid<1e-10,"curveWrap preserves the full cross-section under rigid bending");
    for (size_t i=0;i<pose.size();++i) {
        double const a=double(i)*.3;
        pose[i]=GfVec3d(1-std::cos(a),0,std::sin(a));
    }
    Check(field.Bind(rest,pose,&error),"a planar curved pose binds");
    double radiusError=0;
    for (size_t i=0;i+1<rest.size();++i) for (int j=0;j<12;++j) {
        double const a=j*2*3.141592653589793/12;
        auto const center=(rest[i]+rest[i+1])*.5;
        auto const q=center+GfVec3d(.05*std::cos(a),.05*std::sin(a),0);
        radiusError=std::max(radiusError,std::abs((field.Map(q)-(pose[i]+pose[i+1])*.5).GetLength()-.05));
    }
    Check(radiusError<1e-10,"curved centerline keeps braid cross-section radius");
    auto bad=rest;bad[1]=bad[0];
    Check(!field.Bind(rest,bad,&error),"collapsed driver segment is diagnosed");
    bad=rest;bad.pop_back();
    Check(!field.Bind(rest,bad,&error),"changing driver topology is diagnosed");
}

void CheckBraidExamples()
{
    auto guide = UsdStage::Open(USDGEN_TEST_SOURCE_DIR "/examples/motion/braid_animated_guides.usda");
    auto both = UsdStage::Open(USDGEN_TEST_SOURCE_DIR "/examples/motion/braid_animated_guides_and_surface.usda");
    Check(bool(guide) && bool(both), "both guide-grown braid examples open");
    if (!guide || !both) return;
    SdfPath const description("/World/Motion/Groom/Hair");
    auto path = [&](char const *name) { return description.AppendPath(SdfPath(std::string("Ops/")+name)); };
    auto const rest = Cook(guide,1,description,path("guideAnimate"),path("width"));
    Check(rest.ok && rest.output.totalCurves == 381 && rest.cvs == 64,
          "fixed guide roots grow 381 strands, 64 CVs each" + Errors(rest));
    if (!rest.ok) return;
    Check(MaxShift(rest,false)<1e-6,"guide rest pose preserves the grown braid");
    for (int frame : {13,26,38,51,75,100}) {
        auto const a = Cook(guide,frame,description,path("guideAnimate"),path("width"));
        auto const b = Cook(both,frame,description,path("surfaceAnimate"),path("width"));
        bool stable = a.ok && b.ok && a.input.totalCvs==rest.input.totalCvs &&
            b.output.totalCvs==a.output.totalCvs;
        double error=0;
        double const stretch=std::exp(.48*std::sin(4*3.141592653589793*(frame-1)/99.0));
        double const wide=1/std::sqrt(stretch);
        for (size_t cv=0; stable && cv<a.input.totalCvs; ++cv) {
            stable = P(a.input,cv)==P(rest.input,cv) && P(b.input,cv)==P(rest.input,cv);
            auto const p=P(a.output,cv);
            GfVec3f const expected(float(p[0]*wide),float(p[1]*wide),float(p[2]*stretch));
            error=std::max(error,double((P(b.output,cv)-expected).GetLength()));
        }
        Check(stable,"braid growth stays at Default at frame "+std::to_string(frame)+Errors(a)+Errors(b));
        Check(a.ok && MaxShift(a,true)<1e-6,"guide bend locks braid roots");
        Check(stable && error<3e-6,"guide and surface deformation compose once ("+std::to_string(error)+")");
        if (frame==26) Check(MaxShift(a,false)>.15,"animated guides visibly bend the braid");
    }
}

void CheckSingleCenterExample()
{
    auto stage=UsdStage::Open(USDGEN_TEST_SOURCE_DIR "/examples/motion/braid_simulated_center_curve.usda");
    Check(bool(stage),"single simulated center example opens");
    if (!stage) return;
    SdfPath const desc("/World/Motion/Groom/Hair");
    auto const deform=desc.AppendPath(SdfPath("Ops/guideAnimate"));
    auto const input=desc.AppendPath(SdfPath("Ops/width"));
    auto rest=Cook(stage,1,desc,deform,input);
    Check(rest.ok && rest.output.totalCurves==381 && MaxShift(rest,false)<1e-6,
          "one straight center drives 381 rest-grown braid strands"+Errors(rest));
    if (!rest.ok) return;
    for (int frame:{18,37,65,100,1}) {
        auto pose=Cook(stage,frame,desc,deform,input);
        bool stable=pose.ok && pose.input.totalCvs==rest.input.totalCvs;
        for (size_t cv=0;stable && cv<pose.input.totalCvs;++cv)
            stable=P(pose.input,cv)==P(rest.input,cv);
        Check(stable,"simulated-center playback keeps rest growth fixed at "+std::to_string(frame)+Errors(pose));
        Check(pose.ok && MaxShift(pose,true)<1e-6,"pinned center preserves attached braid roots");
        if (frame==37) Check(MaxShift(pose,false)>.02,"simulation visibly moves the braid");
    }
    auto center=stage->GetPrimAtPath(SdfPath("/World/Motion/SimulatedCenter"));
    VtIntArray counts;
    center.GetAttribute(TfToken("curveVertexCounts")).Get(&counts);
    Check(counts==VtIntArray{31},"exactly one 31-CV simulation driver is authored");
    center.GetAttribute(TfToken("curveVertexCounts")).Set(VtIntArray{15,16});
    auto invalid=Cook(stage,37,desc,deform,input);
    Check(!invalid.ok,"curveWrap refuses multiple driver curves");
}

void CheckRegionExamples()
{
    for (bool dense:{false,true}) {
        std::string const scene=std::string(USDGEN_TEST_SOURCE_DIR)+"/examples/motion/"+
            (dense?"two_braids_ptex_regions.usda":"two_curves_ptex_regions.usda");
        auto stage=UsdStage::Open(scene);
        Check(bool(stage),"Ptex multi-center example opens");if (!stage) continue;
        SdfPath const desc("/World/Motion/Groom/Hair");
        auto const deform=desc.AppendPath(SdfPath("Ops/regionAnimate"));
        auto const input=desc.AppendPath(SdfPath("Ops/width"));
        auto cook=[&](int frame) {return Cook(stage,frame,desc,deform,input);};
        auto rest=cook(1);
        Check(rest.ok && rest.output.totalCurves==(dense?762u:2u),
              "one description grows the expected number of region strands"+Errors(rest));
        if (!rest.ok) continue;
        auto driver=UsdGeomBasisCurves(stage->GetPrimAtPath(SdfPath("/World/Motion/Drivers")));
        VtVec3fArray driverRest;VtIntArray counts;
        driver.GetPointsAttr().Get(&driverRest,UsdTimeCode::Default());
        driver.GetCurveVertexCountsAttr().Get(&counts);
        Check(counts==VtIntArray({31,31}),"two center curves drive a single description");
        auto verify=[&](int frame,bool swapped) {
            auto pose=cook(frame);
            VtVec3fArray driverNow;driver.GetPointsAttr().Get(&driverNow,UsdTimeCode(frame));
            curveWrap::Field fields[2];std::string error;
            for (int g=0;g<2;++g) {
                std::vector<GfVec3d> r,n;
                for (int i=g*31;i<(g+1)*31;++i) {r.emplace_back(driverRest[i]);n.emplace_back(driverNow[i]);}
                Check(fields[g].Bind(r,n,&error),"region driver field binds");
            }
            bool stable=pose.ok && pose.input.totalCvs==rest.input.totalCvs;double maxError=0;
            for (size_t c=0;stable && c<pose.input.totalCurves;++c) {
                int g=P(rest.input,c*rest.cvs)[0]<0 ? 0:1;if (swapped) g=1-g;
                for (size_t cv=c*rest.cvs;stable && cv<(c+1)*rest.cvs;++cv) {
                    stable=P(pose.input,cv)==P(rest.input,cv);
                    maxError=std::max(maxError,(GfVec3d(P(pose.output,cv))-fields[g].Map(GfVec3d(P(rest.input,cv)))).GetLength());
                }
            }
            Check(stable && maxError<2e-6,(swapped?"swapping Ptex texels rebinds across space":"Ptex regions follow only their assigned center")+
                  std::string(" error=")+std::to_string(maxError)+Errors(pose));
        };
        verify(37,false);verify(100,false);
        // Moving just driver 1 cannot affect region 0, even though they share a
        // description, operator, surface and one two-curve driver prim.
        VtVec3fArray changed=driverRest;
        for (size_t i=31;i<changed.size();++i) changed[i]+=GfVec3f(.1f,.07f,-.02f);
        driver.GetPointsAttr().Set(changed,UsdTimeCode(55));verify(55,false);
        auto map=stage->GetPrimAtPath(desc.AppendPath(SdfPath("Maps/Regions")));
        map.GetAttribute(TfToken("usdGen:map:file")).Set(SdfAssetPath(
            USDGEN_TEST_SOURCE_DIR "/examples/motion/maps/two_regions_swapped.ptx"));
        verify(37,true);
        // Reject missing driver IDs instead of silently choosing a nearby driver.
        auto regionIds=stage->GetPrimAtPath(deform).GetAttribute(TfToken("usdGen:guideRegions"));
        regionIds.Set(VtIntArray{7,8});
        auto missing=cook(37);
        Check(!missing.ok && Errors(missing).find("has no center curve")!=std::string::npos,
              "unmapped Ptex IDs are diagnosed"+Errors(missing));
        regionIds.Set(VtIntArray{0,1});
        map.GetAttribute(TfToken("usdGen:map:filter")).Set(TfToken("bilinear"));
        auto filtered=cook(37);
        Check(!filtered.ok,"regionMap refuses filtered/blended category IDs");
    }
}

// --- the example, through the groom scene index -----------------------------------

std::vector<GfVec3f> PublishedPoints(HdSceneIndexBaseRefPtr const &groom,
                                  SdfPath const &description = kDescription)
{
    std::vector<GfVec3f> points;
    SdfPath const render = description.AppendChild(TfToken("__usdGenRender"));
    SdfPathVector tiles = groom->GetChildPrimPaths(render);
    std::sort(tiles.begin(), tiles.end());
    for (SdfPath const &tile : tiles) {
        // Tiles only. The render scope also carries the scalp-shadow cap,
        // whose points are a tessellation of the emitting surface and whose
        // count changes with the hair over it.
        if (tile.GetName().rfind("tile_", 0) != 0) continue;
        auto sampled = HdSampledDataSource::Cast(HdContainerDataSource::Get(
            groom->GetPrim(tile).dataSource,
            HdDataSourceLocator(TfToken("primvars"), TfToken("points"), TfToken("primvarValue"))));
        if (!sampled) continue;
        VtValue const value = sampled->GetValue(0.0f);
        if (!value.IsHolding<VtVec3fArray>()) continue;
        for (GfVec3f const &p : value.UncheckedGet<VtVec3fArray>()) points.push_back(p);
    }
    return points;
}

void CheckOutsidePublication()
{
    auto stage=UsdStage::Open(USDGEN_TEST_SOURCE_DIR "/examples/motion/felt_sphere_groom_outside_xform.usda");
    if (!stage) { Check(false,"outside groom stage opens for Hydra"); return; }
    SdfPath const description("/World/Groom/Hair");
    // An independently transformed groom must cancel its own world transform
    // during deformation, then receive it once from Hydra publication.
    auto container=stage->GetPrimAtPath(SdfPath("/World/Groom"));
    container.SetTypeName(TfToken("Xform"));
    GfMatrix4d transform(1.0);
    transform.SetRotate(GfRotation(GfVec3d(0,1,0),23));
    transform.SetTranslateOnly(GfVec3d(.12,.05,-.1));
    container.CreateAttribute(TfToken("xformOp:transform"),SdfValueTypeNames->Matrix4d).Set(transform);
    container.CreateAttribute(TfToken("xformOpOrder"),SdfValueTypeNames->TokenArray).Set(VtTokenArray{TfToken("xformOp:transform")});
    UsdImagingCreateSceneIndicesInfo info; info.stage=stage;
    auto indices=UsdImagingCreateSceneIndices(info);
    auto groom=UsdGenGroomSceneIndex::New(indices.finalSceneIndex);
    auto *owner=dynamic_cast<UsdGenGroomSceneIndex *>(groom.operator->());
    for (int frame : {38,1,26,100}) {
        indices.stageSceneIndex->SetTime(UsdTimeCode(frame));
        indices.stageSceneIndex->ApplyPendingUpdates(); owner->Synchronize();
        auto actual=PublishedPoints(groom,description);
        auto rest=Cook(stage,frame,description,description.AppendPath(SdfPath("Ops/surfaceAnimate")),
                       description.AppendPath(SdfPath("Ops/width")));
        double error=0;
        double const stretch=std::exp(.48*std::sin(4*3.141592653589793*(frame-1)/99.0));
        double const wide=1/std::sqrt(stretch);
        GfVec3d offset(.55*std::sin(2*3.141592653589793*(frame-1)/99.0),0,.5);
        bool valid=rest.ok && actual.size()==rest.input.totalCvs;
        for (size_t cv=0; valid && cv<actual.size(); ++cv) {
            auto p=P(rest.input,cv);
            GfVec3d expected(p[0]*wide,p[1]*wide,p[2]*stretch);
            error=std::max(error,(transform.Transform(GfVec3d(actual[cv]))-expected-offset).GetLength());
        }
        Check(valid && error<3e-6,"separately transformed groom publishes correct world points at frame "+
              std::to_string(frame)+" error="+std::to_string(error));
    }
}

void CheckScene()
{
    UsdStageRefPtr const stage = UsdStage::Open(kScene, UsdStage::LoadAll);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(info);
    indices.stageSceneIndex->SetTime(UsdTimeCode(1.0));
    HdSceneIndexBaseRefPtr groom = UsdGenGroomSceneIndex::New(indices.finalSceneIndex);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(groom.operator->());
    Check(owner != nullptr, "the example opens in a groom scene index");
    if (!owner) return;
    owner->Synchronize();
    std::vector<GfVec3f> const atRest = PublishedPoints(groom);
    Check(!atRest.empty(), "frame 1 publishes strands through Hydra");

    indices.stageSceneIndex->SetTime(UsdTimeCode(20.0));
    indices.stageSceneIndex->ApplyPendingUpdates();
    owner->Synchronize();
    std::vector<GfVec3f> const posed = PublishedPoints(groom);
    double moved = 0.0;
    for (size_t i = 0; i < posed.size() && i < atRest.size(); ++i)
        moved = std::max(moved, double((posed[i] - atRest[i]).GetLength()));
    Check(posed.size() == atRest.size() && moved > 0.05,
          "changing the frame re-cooks the strands with the drivers (" + std::to_string(moved) + ")");

    indices.stageSceneIndex->SetTime(UsdTimeCode(1.0));
    indices.stageSceneIndex->ApplyPendingUpdates();
    owner->Synchronize();
    std::vector<GfVec3f> const back = PublishedPoints(groom);
    double drift = 0.0;
    for (size_t i = 0; i < back.size() && i < atRest.size(); ++i)
        drift = std::max(drift, double((back[i] - atRest[i]).GetLength()));
    Check(back.size() == atRest.size() && drift < 1e-6, "scrubbing back to frame 1 restores the rest groom");
}

// --- playback notices ----------------------------------------------------------

class DirtyLog : public HdSceneIndexObserver
{
public:
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &entries) override
    {
        added += entries.size();
    }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &entries) override
    {
        removed += entries.size();
    }
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &entries) override
    {
        dirtied.insert(dirtied.end(), entries.begin(), entries.end());
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
    void Clear()
    {
        added = removed = 0;
        dirtied.clear();
    }

    size_t added = 0, removed = 0;
    DirtiedPrimEntries dirtied;
};

bool Universal(HdDataSourceLocatorSet const &locators)
{
    for (HdDataSourceLocator const &locator : locators)
        if (locator.IsEmpty()) return true;
    return false;
}

std::string Describe(HdDataSourceLocatorSet const &locators)
{
    std::string text;
    for (HdDataSourceLocator const &locator : locators)
        text += " " + (locator.IsEmpty() ? std::string("<universal>") : locator.GetString());
    return text;
}

void CheckPlaybackNotices()
{
    UsdStageRefPtr const stage = UsdStage::Open(kScene, UsdStage::LoadAll);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(info);
    HdsiSceneGlobalsSceneIndexRefPtr const globals =
        HdsiSceneGlobalsSceneIndex::New(indices.finalSceneIndex);
    indices.stageSceneIndex->SetTime(UsdTimeCode(10.0));
    globals->SetCurrentFrame(10.0);
    HdSceneIndexBaseRefPtr groom = UsdGenGroomSceneIndex::New(globals);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(groom.operator->());
    if (!owner) { Check(false, "the example opens under the scene globals"); return; }
    owner->Synchronize();

    SdfPath const render = kDescription.AppendChild(TfToken("__usdGenRender"));
    std::vector<SdfPath> tiles;
    for (SdfPath const &child : groom->GetChildPrimPaths(render))
        if (groom->GetPrim(child).primType == TfToken("basisCurves")) tiles.push_back(child);
    Check(!tiles.empty(), "frame 10 publishes tiles");

    DirtyLog log;
    groom->AddObserver(HdSceneIndexObserverPtr(&log));

    // One frame step, the way usdview takes it.
    uint64_t const cooks = UsdGenImagingTestHook::groomCookCount(*groom);
    indices.stageSceneIndex->SetTime(UsdTimeCode(11.0));
    globals->SetCurrentFrame(11.0);
    indices.stageSceneIndex->ApplyPendingUpdates();
    owner->Synchronize();
    uint64_t const stepCooks = UsdGenImagingTestHook::groomCookCount(*groom) - cooks;
    Check(stepCooks == 1, "a frame step cooks once (" + std::to_string(stepCooks) + ")");
    Check(log.added == 0 && log.removed == 0, "a frame step adds and removes nothing");

    TfToken const primvars("primvars");
    HdDataSourceLocator const points(primvars, TfToken("points"));
    // `normals` is the scalp cap's: it follows the emitting surface, which the
    // RBF drivers deform, so it moves whenever the cap's points do.
    std::vector<TfToken> const moving = {TfToken("points"), TfToken("furTauP"),
                                         TfToken("furTauN"), TfToken("normals")};
    std::string universal, unexpected;
    size_t tilesWithPoints = 0;
    bool driverPoints = false, driverUniversal = false;
    for (HdSceneIndexObserver::DirtiedPrimEntry const &entry : log.dirtied) {
        HdDataSourceLocatorSet const &locators = entry.dirtyLocators;
        if (Universal(locators)) universal += " " + entry.primPath.GetString();
        if (entry.primPath == kDrivers) {
            driverPoints |= locators.Intersects(points);
            driverUniversal |= Universal(locators);
            continue;
        }
        bool const tile = std::find(tiles.begin(), tiles.end(), entry.primPath) != tiles.end();
        // The scalp-shadow cap is rebaked with the groom, so it moves every
        // frame too. It is held to the same discipline as a tile: precise
        // locators over the primvars that actually change, never universal.
        bool const cap = entry.primPath.GetName() == "scalpShadow";
        if (!tile && !cap) {
            if (entry.primPath.HasPrefix(render))
                unexpected += " " + entry.primPath.GetString() + ":" + Describe(locators);
            continue;
        }
        if (tile && locators.Intersects(points)) ++tilesWithPoints;
        for (HdDataSourceLocator const &locator : locators) {
            if (!locator.HasPrefix(HdDataSourceLocator(primvars))) continue;
            bool const allowed = locator.GetElementCount() >= 2 &&
                std::find(moving.begin(), moving.end(), locator.GetElement(1)) != moving.end();
            if (!allowed) unexpected += " " + entry.primPath.GetName() + ":" + locator.GetString();
        }
    }
    Check(universal.empty(), "a frame step sends no universal dirty (" + universal + " )");
    Check(tilesWithPoints == tiles.size(), "every tile dirties its points (" +
                                               std::to_string(tilesWithPoints) + " of " +
                                               std::to_string(tiles.size()) + ")");
    Check(unexpected.empty(),
          "tiles dirty only points and occlusion; the scope and material stay clean (" +
              unexpected + " )");
    Check(driverPoints && !driverUniversal,
          "the drivers' dirty reaches Hydra with the stage's locators");

    // The scene globals' frame alone.
    log.Clear();
    uint64_t const before = UsdGenImagingTestHook::groomCookCount(*groom);
    globals->SetCurrentFrame(12.0);
    owner->Synchronize();
    uint64_t const globalsCooks = UsdGenImagingTestHook::groomCookCount(*groom) - before;
    bool tileDirtied = false, rootOnlyGlobals = true;
    for (HdSceneIndexObserver::DirtiedPrimEntry const &entry : log.dirtied) {
        tileDirtied |= std::find(tiles.begin(), tiles.end(), entry.primPath) != tiles.end();
        if (entry.primPath.IsAbsoluteRootPath())
            for (HdDataSourceLocator const &locator : entry.dirtyLocators)
                rootOnlyGlobals &= locator.HasPrefix(HdDataSourceLocator(TfToken("sceneGlobals")));
    }
    Check(globalsCooks == 0 && !tileDirtied,
          "the scene globals' frame alone cooks nothing (" + std::to_string(globalsCooks) + ")");
    Check(rootOnlyGlobals, "the root dirty keeps the scene globals' locator");

    groom->RemoveObserver(HdSceneIndexObserverPtr(&log));
}

} // namespace

void CheckTilesMatchSequential(UsdGenGraph const &graph, std::string const &what)
{
    // Every dirtied tile must publish bitwise the sequential extent and
    // counts over the terminal points. Untouched tiles keep their previous
    // extent and are skipped.
    UsdGenCompiledNode const &tn = graph.Node(graph.TerminalNodeId());
    UsdGenCurveBuffer const &term = graph.Output();
    bool const ragged = !term.cvOffsets.empty();
    float const *tpx = term.px.empty() ? nullptr : term.px.cdata();
    float const *tpy = term.py.empty() ? nullptr : term.py.cdata();
    float const *tpz = term.pz.empty() ? nullptr : term.pz.cdata();
    size_t const nPx = term.px.size();
    size_t compared = 0, extentsOk = 0, countsOk = 0;
    for (UsdGenTileView const &tv : graph.Tiles()) {
        // Untouched tiles keep their previous extent; only a tile the
        // interleave rewrote (pointsDirty) must match the reference.
        if (!tv.pointsDirty) continue;
        if (tn.chunks.size() < size_t(tv.firstChunk) + tv.chunkCount) continue;
        GfRange3f ref;
        uint64_t liveCurves = 0, liveCvs = 0;
        for (uint32_t i = 0; i < tv.chunkCount; ++i) {
            UsdGenChunkDesc const &cd = tn.chunks[tv.firstChunk + i];
            if (ragged && cd.cvCount == 0) {
                for (uint32_t c = 0; c < cd.liveCount; ++c) {
                    size_t const g = size_t(cd.firstCurve) + c;
                    if (g + 1 >= term.cvOffsets.size()) break;
                    uint32_t const p0 = uint32_t(term.cvOffsets[g]);
                    uint32_t const len = uint32_t(term.cvOffsets[g + 1]) - p0;
                    ++liveCurves;
                    liveCvs += len;
                    if (!tpx || !tpy || !tpz) continue;
                    for (uint32_t v = 0; v < len; ++v) {
                        size_t const p = size_t(p0) + v;
                        if (p >= nPx) break;
                        ref.ExtendBy(GfVec3f(tpx[p], tpy[p], tpz[p]));
                    }
                }
                continue;
            }
            liveCurves += cd.liveCount;
            liveCvs += uint64_t(cd.liveCount) * cd.cvCount;
            if (!tpx || !tpy || !tpz) continue;
            for (uint32_t c = 0; c < cd.liveCount; ++c) {
                size_t const o = size_t(cd.firstCv) + size_t(c) * cd.cvCount;
                for (uint32_t v = 0; v < cd.cvCount; ++v) {
                    size_t const p = o + v;
                    if (p >= nPx) break;
                    ref.ExtendBy(GfVec3f(tpx[p], tpy[p], tpz[p]));
                }
            }
        }
        ++compared;
        extentsOk += std::memcmp(&tv.extent.GetMin(), &ref.GetMin(), sizeof(GfVec3f)) == 0 &&
                std::memcmp(&tv.extent.GetMax(), &ref.GetMax(), sizeof(GfVec3f)) == 0;
        countsOk += tv.totalLiveCurves == liveCurves && tv.totalLiveCvs == liveCvs;
    }
    Check(compared > 0, what + ": the pose dirties tiles");
    Check(compared > 0 && extentsOk == compared && countsOk == compared,
          what + ": tile extents and counts are bitwise the sequential pass (" +
              std::to_string(compared) + " tiles)");
}

void CheckFusedTileExtents()
{
    // The tile interleave fuses per-chunk extents recorded by the deform
    // (the deepest points-writer here is the terminal deform) instead of
    // re-reading every point. Two commits on one graph: the recompiled
    // pose re-captures the deform, and every dirtied tile must publish
    // bitwise the sequential extent and counts over the terminal points.
    UsdStageRefPtr const stage = UsdStage::Open(kScene);
    if (!stage) { Check(false, "fused extents: cannot open the scene"); return; }
    auto build = [&](double time) {
        usdGenImaging::UsdGenGraphDescBuildOptions options;
        options.time = time;
        return usdGenImaging::BuildGraphDescFromStage(stage, kDescription, options);
    };
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenGraphDesc desc = build(1.0);
    if (!compiler.Compile(desc, &graph).ok) {
        Check(false, "fused extents: the rest pose compiles"); return;
    }
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext context;
    context.time = 1.0;
    uint64_t gen = 0;
    if (scheduler.Run(graph, context, ++gen).diagnostics.HasErrors()) {
        Check(false, "fused extents: the rest pose cooks"); return;
    }
    desc = build(20.0);
    context.time = 20.0;
    if (!compiler.Recompile(desc, &graph).ok) {
        Check(false, "fused extents: the pose recompiles"); return;
    }
    UsdGenRunResult const run = scheduler.Run(graph, context, ++gen);
    if (run.diagnostics.HasErrors()) {
        Check(false, "fused extents: the pose cooks"); return;
    }
    bool swept = false;
    for (auto const &st : run.nodeStats)
        if (graph.Node(st.id).type == TfToken("UsdGenDeform") && st.chunksEvaluated > 0)
            swept = true;
    Check(swept, "fused extents: the pose produces the deform outputs");
    if (!swept) return;
    // Engagement: this scene's deform has a uniform-1 mask, so the pose
    // capture must have written direct with recorded extents; otherwise
    // the comparison below exercises the point pass, not the fused path.
    int deforms = 0;
    bool direct = true, recorded = true;
    for (size_t i = 0; i < size_t(graph.NodeCount()); ++i) {
        UsdGenCompiledNode const &n = graph.Node(UsdGenNodeId(i));
        if (n.type != TfToken("UsdGenDeform") || !n.capture) continue;
        ++deforms;
        direct = direct && n.capture->WroteDirectOutput();
        recorded = recorded && n.capture->RecordedChunkExtents();
    }
    Check(deforms > 0, "fused extents: the scene has a deform");
    Check(direct, "fused extents: the pose capture wrote direct");
    Check(recorded, "fused extents: the pose capture recorded chunk extents");
    CheckTilesMatchSequential(graph, "fused extents");
}

// Capture-direct through a chained pair: the upstream deform goes direct
// without recording (slots belong to the deepest writer alone) while the
// terminal deform records, and both publish bitwise-sequential tiles.
void CheckCaptureDirectChained()
{
    UsdGenGraphDesc desc = MakeSmallSurfaceDeformDesc();
    UsdGenNodeDesc second;
    second.path = SdfPath("/groom/deform2");
    second.type = TfToken("UsdGenDeform");
    second.enabled = true;
    second.seed = 45;
    second.inputs.push_back(SdfPath("/groom/deform"));
    second.surfaces.push_back(SdfPath("/groom/surface"));
    desc.nodes.push_back(second);
    for (auto &n : desc.nodes)
        if (n.path == SdfPath("/groom/deform2")) {
            n.params.push_back(UsdGenParamValue{TfToken("rbfSamples"), VtValue(100), false});
            n.params.push_back(UsdGenParamValue{TfToken("lockRoots"), VtValue(true), false});
        }
    desc.terminal = SdfPath("/groom/deform2");
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "chained direct: the fixture compiles");
    UsdGenNodeId const d1 = graph.NodeIdForPath(SdfPath("/groom/deform"));
    UsdGenNodeId const d2 = graph.NodeIdForPath(SdfPath("/groom/deform2"));
    if (d1 == kUsdGenInvalidNode || d2 == kUsdGenInvalidNode) {
        Check(false, "chained direct: both deforms compile");
        return;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    uint64_t gen = 0;
    auto runPose = [&]() -> bool {
        UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
        ctx.desc = &graph.Desc();
        UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
        return rr.ok && !run.diagnostics.HasErrors();
    };
    auto bend = [&](double t) {
        VtVec3fArray const &rest = desc.surfaces[0].restPoints;
        desc.surfaces[0].points.resize(rest.size());
        for (size_t i = 0; i < rest.size(); ++i) {
            GfVec3f const r = rest[i];
            float const b = float(0.6 * std::sin(0.05 * r[0] + t) *
                                   std::cos(0.04 * r[1] - 0.5 * t));
            desc.surfaces[0].points[i] = GfVec3f(r[0], r[1], r[2] + b);
        }
    };
    bend(0.0);
    Check(runPose(), "chained direct: the rest pose cooks");
    bend(1.0);
    Check(runPose(), "chained direct: a bent pose cooks");
    UsdGenCompiledNode const &n1 = graph.Node(d1);
    UsdGenCompiledNode const &n2 = graph.Node(d2);
    bool const d1direct = n1.capture && n1.capture->WroteDirectOutput();
    bool const d1recorded = n1.capture && n1.capture->RecordedChunkExtents();
    bool const d2direct = n2.capture && n2.capture->WroteDirectOutput();
    bool const d2recorded = n2.capture && n2.capture->RecordedChunkExtents();
    Check(d1direct && !d1recorded,
          "chained direct: the upstream deform writes direct without recording");
    Check(d2direct && d2recorded,
          "chained direct: the terminal deform writes direct and records");
    CheckTilesMatchSequential(graph, "chained direct");
}

// A mask edit reuses the direct capture without re-capturing: the sweep
// must blend from the capture's deformed values (not the input), bitwise
// the float lerp of the direct outputs.
void CheckCaptureDirectMaskReuse()
{
    UsdGenGraphDesc desc = MakeSmallSurfaceDeformDesc();
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "mask reuse: the fixture compiles");
    UsdGenNodeId const deformId = graph.NodeIdForPath(SdfPath("/groom/deform"));
    UsdGenNodeId const growId = graph.Node(deformId).input;
    auto *op = static_cast<UsdGenDeformOp *>(graph.Node(deformId).op.get());
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    uint64_t gen = 0;
    auto runPose = [&]() -> bool {
        UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
        ctx.desc = &graph.Desc();
        UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
        return rr.ok && !run.diagnostics.HasErrors();
    };
    Check(runPose(), "mask reuse: the rest pose cooks");
    // Bend the drivers so the direct cook differs from the rest cook.
    VtVec3fArray const &rest = desc.surfaces[0].restPoints;
    desc.surfaces[0].points.resize(rest.size());
    for (size_t i = 0; i < rest.size(); ++i) {
        GfVec3f const r = rest[i];
        float const b = float(0.6 * std::sin(0.05 * r[0] + 0.35) *
                               std::cos(0.04 * r[1] - 0.175));
        desc.surfaces[0].points[i] = GfVec3f(r[0], r[1], r[2] + b);
    }
    Check(runPose(), "mask reuse: a bent pose cooks");
    UsdGenCapture const *directCap = graph.Node(deformId).capture.get();
    UsdGenEpoch const directEpoch = graph.Node(deformId).captureEpoch;
    Check(directCap && directCap->WroteDirectOutput(),
          "mask reuse: the bent cook goes direct");
    if (!directCap || !directCap->WroteDirectOutput()) return;
    UsdGenCurveBuffer const &up = graph.Node(growId).buffer;
    std::vector<float> const beforeX(up.px.begin(), up.px.end());
    std::vector<float> const beforeY(up.py.begin(), up.py.end());
    std::vector<float> const beforeZ(up.pz.begin(), up.pz.end());
    UsdGenCurveBuffer const &term = graph.Node(deformId).buffer;
    std::vector<float> const directX(term.px.begin(), term.px.end());
    std::vector<float> const directY(term.py.begin(), term.py.end());
    std::vector<float> const directZ(term.pz.begin(), term.pz.end());
    // Vacuity guard: the pose must actually move points, or the checks
    // below compare against their own input and prove nothing.
    Check(directX != beforeX || directY != beforeY || directZ != beforeZ,
          "mask reuse: the pose displaces points");
    // The direct capture carries its deformed values for any later sweep
    // without re-capture (a value-only edit reuses the capture): a whole
    // range hand re-evaluate must reproduce its outputs bitwise.
    {
        size_t const curves = term.totalCurves, total = term.totalCvs;
        bool uniform = curves > 0 && total % curves == 0;
        Check(uniform, "mask reuse: the fixture is uniform");
        if (uniform) {
            uint32_t const perCurve = uint32_t(total / curves);
            UsdGenChunkDesc fullDesc;
            fullDesc.firstCurve = 0;
            fullDesc.curveCount = uint32_t(curves);
            fullDesc.firstCv = 0;
            fullDesc.cvCount = perCurve;
            std::vector<float> hx(total), hy(total), hz(total);
            UsdGenChunkView v{};
            v.desc = &fullDesc;
            v.px = hx.data();
            v.py = hy.data();
            v.pz = hz.data();
            v.inPx = up.px.cdata();
            v.inPy = up.py.cdata();
            v.inPz = up.pz.cdata();
            v.curveCount = uint32_t(curves);
            v.cvCount = perCurve;
            UsdGenEvalContext evalCtx; // params null: default mask 1.0
            op->Evaluate(evalCtx, *directCap, &v);
            Check(hx == directX && hy == directY && hz == directZ,
                  "mask reuse: the direct capture carries its deformed values");
        }
    }
    // A mask edit changes the desc, so the compiler re-captures through
    // the result path (direct declined); the sweep blends bitwise the
    // float lerp of the deformed values.
    for (auto &n : desc.nodes)
        if (n.path == SdfPath("/groom/deform"))
            n.params.push_back(UsdGenParamValue{TfToken("mask"), VtValue(0.5), false});
    Check(runPose(), "mask reuse: the mask edit cooks");
    Check(graph.Node(deformId).captureEpoch != directEpoch,
          "mask reuse: the mask edit re-captures");
    Check(!graph.Node(deformId).capture->WroteDirectOutput(),
          "mask reuse: the mask edit declines direct output");
    UsdGenCurveBuffer const &out = graph.Node(deformId).buffer;
    bool blendOk = out.px.size() == directX.size();
    for (size_t i = 0; blendOk && i < directX.size(); ++i) {
        float const e0 = beforeX[i] + (directX[i] - beforeX[i]) * 0.5f;
        float const e1 = beforeY[i] + (directY[i] - beforeY[i]) * 0.5f;
        float const e2 = beforeZ[i] + (directZ[i] - beforeZ[i]) * 0.5f;
        blendOk = blendOk && out.px[i] == e0 && out.py[i] == e1 && out.pz[i] == e2;
    }
    Check(blendOk, "mask reuse: the mask edit blends bitwise the float lerp");
}

// A failing direct capture restores the input planes: the failed run must
// not publish a mix of old and new values.
void CheckCaptureDirectFailureRestore()
{
    UsdGenGraphDesc desc = MakeSmallSurfaceDeformDesc();
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "failure restore: the fixture compiles");
    UsdGenNodeId const deformId = graph.NodeIdForPath(SdfPath("/groom/deform"));
    UsdGenNodeId const growId = graph.Node(deformId).input;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    uint64_t gen = 0;
    auto runPose = [&]() -> UsdGenRunResult {
        UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
        ctx.desc = &graph.Desc();
        UsdGenRunResult run = scheduler.Run(graph, ctx, ++gen);
        if (!rr.ok) run.diagnostics.Error("recompile failed");
        return run;
    };
    Check(!runPose().diagnostics.HasErrors(), "failure restore: the rest pose cooks");
    // Bend the drivers so the second cook goes direct.
    VtVec3fArray const &rest = desc.surfaces[0].restPoints;
    desc.surfaces[0].points.resize(rest.size());
    for (size_t i = 0; i < rest.size(); ++i) {
        GfVec3f const r = rest[i];
        float const b = float(0.6 * std::sin(0.05 * r[0] + 1.0) *
                               std::cos(0.04 * r[1] - 0.5));
        desc.surfaces[0].points[i] = GfVec3f(r[0], r[1], r[2] + b);
    }
    Check(!runPose().diagnostics.HasErrors(), "failure restore: a bent pose cooks");
    UsdGenCapture const *cap = graph.Node(deformId).capture.get();
    Check(cap && cap->WroteDirectOutput(),
          "failure restore: the bent pose goes direct");
    // Astronomical grow length: the cubic kernel overflows float at the
    // far CVs, so the direct capture must refuse.
    for (auto &n : desc.nodes)
        if (n.path == SdfPath("/groom/grow"))
            for (auto &pv : n.params)
                if (pv.name == TfToken("length")) pv.value = VtValue(1e20);
    UsdGenRunResult const failed = runPose();
    bool refused = failed.diagnostics.HasErrors();
    bool nonFinite = false;
    for (auto const &e : failed.diagnostics.errors)
        nonFinite = nonFinite || e.find("non-finite") != std::string::npos;
    Check(refused && nonFinite,
          "failure restore: the overflowing pose is refused, not published");
    UsdGenCurveBuffer const &out = graph.Node(deformId).buffer;
    UsdGenCurveBuffer const &up = graph.Node(growId).buffer;
    bool restored = out.px.size() == up.px.size();
    restored = restored && std::memcmp(out.px.cdata(), up.px.cdata(),
                                       out.px.size() * sizeof(float)) == 0;
    restored = restored && std::memcmp(out.py.cdata(), up.py.cdata(),
                                       up.py.size() * sizeof(float)) == 0;
    restored = restored && std::memcmp(out.pz.cdata(), up.pz.cdata(),
                                       out.pz.size() * sizeof(float)) == 0;
    Check(restored, "failure restore: the failed run restores the input planes");
}

int main()
{
    usdGenRegisterM1Operators();
    CheckField();
    CheckBatchBitwise();
    CheckBatchPathsAgree();
    CheckBatchPlanarBitwise();
    CheckBatchThreaded();
    CheckBatchConcurrent();
    CheckDeformChosenDigest();
    CheckDeformEvaluateViewShapes();
    CheckFusedTileExtents();
    CheckCaptureDirectChained();
    CheckCaptureDirectMaskReuse();
    CheckCaptureDirectFailureRestore();
    CheckCurveWrapField();
    CheckExample();
    CheckSurfaceExample();
    CheckSurfaceExample(true);
    CheckNonFiniteCapture();
    CheckBraidExamples();
    CheckSingleCenterExample();
    CheckRegionExamples();
    CheckOutsidePublication();
    CheckScene();
    CheckPlaybackNotices();
    UsdGenGroomSceneIndex::DrainRetired();
    std::printf("testUsdGenRbfDeform: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
