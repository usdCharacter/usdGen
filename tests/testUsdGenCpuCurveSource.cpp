#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/imagePayload.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"
#ifdef USDGEN_TEST_CUDA_SOURCE_PARITY
#include "usdGen/cudaExecution.h"
#include "usdGen/gpu/generation.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static UsdGenGraphDesc Desc() {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom");
    d.defaultWidth = .125f;
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Hair");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {3, 2};
    curves.rest = {{10,0,0}, {11,0,0}, {12,0,0}, {20,0,0}, {21,0,0}};
    curves.points = curves.rest;
    for (auto& p : curves.points) p[0] += 100;
    curves.widths = {.1f, .2f, .3f, .4f, .5f};
    curves.curveId = {40, 7};
    curves.skinPrim = {2, 1};
    curves.skinPrimUv = {{.2f,.3f}, {.4f,.5f}};
    curves.rootFrame = {GfMatrix4d(1.0), GfMatrix4d(1.0)};
    UsdGenAuthoredPlaneDesc cv;
    cv.name = TfToken("sourceValue");
    cv.type = UsdGenAuthoredPlaneType::Float32;
    cv.domain = UsdGenAuthoredPlaneDomain::Point;
    cv.arity = 1;
    cv.floatValues = {0,1,2,3,4};
    curves.authoredPlanes.push_back(cv);
    UsdGenAuthoredPlaneDesc uniform;
    uniform.name = TfToken("sourceGroup");
    uniform.type = UsdGenAuthoredPlaneType::Int32;
    uniform.domain = UsdGenAuthoredPlaneDomain::Primitive;
    uniform.arity = 1;
    uniform.intValues = {40,7};
    curves.authoredPlanes.push_back(uniform);
    d.curveSets.push_back(curves);
    UsdGenNodeDesc node;
    node.path = SdfPath("/Source");
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {curves.path};
    node.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                   {TfToken("useRest"), VtValue(true), false}};
    d.nodes.push_back(node);
    d.terminal = node.path;
    return d;
}

#ifdef USDGEN_TEST_CUDA_SOURCE_PARITY
template<class T> static bool Read(gpu::DeviceView<const T> input,
                                  std::vector<T>* output) {
    output->resize(input.size);
    return input.size == 0 || cudaMemcpy(output->data(), input.data,
        input.size * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
}

static bool CudaParity(UsdGenGraphDesc d, UsdGenCurveBuffer const& cpu) {
    d.executionBackend = UsdGenExecutionBackend::Cuda;
    UsdGenDiagnostics diagnostics;
    auto plan = CompileCudaGraph(d, &diagnostics);
    if (!plan) {
        for (auto const& e : diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
        return false;
    }
    auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    if (!workspace) return false;
    auto generation = ExecuteCudaGraph(*plan, *workspace, 1, 1, &diagnostics);
    if (!generation) {
        for (auto const& e : diagnostics.errors) std::fprintf(stderr,"%s\n",e.c_str());
        return false;
    }
    auto lease = gpu::AcquireGeometry(generation, nullptr);
    if (!lease) return false;
    std::vector<float3> points, rest;
    std::vector<float> widths, hairT;
    std::vector<uint64_t> ids;
    std::vector<int32_t> rootPrim;
    std::vector<float2> rootUV;
    auto const& geometry = lease.Geometry();
    if (!Read(geometry.points,&points) || !Read(geometry.restPoints,&rest) ||
        !Read(geometry.widths,&widths) ||
        !Read(lease.HairT(),&hairT) || !Read(geometry.stableIds,&ids) ||
        !Read(lease.RootPrim(),&rootPrim) || !Read(lease.RootUV(),&rootUV) ||
        rootPrim.size() != cpu.rootPrim.size() || rootUV.size() != cpu.rootUV.size() ||
        points.size() != cpu.totalCvs || rest.size() != cpu.rest.size() ||
        widths.size() != cpu.width.size() ||
        hairT.size() != cpu.hairT.size() || ids.size() != cpu.curveId.size()) return false;
    for (size_t i=0; i<ids.size(); ++i) if (ids[i] != cpu.curveId[i]) return false;
    for (size_t i=0; i<rootPrim.size(); ++i)
        if (rootPrim[i] != cpu.rootPrim[i] ||
            std::fabs(rootUV[i].x-cpu.rootUV[i][0]) > 1.e-5f ||
            std::fabs(rootUV[i].y-cpu.rootUV[i][1]) > 1.e-5f) return false;
    for (size_t i=0; i<points.size(); ++i) {
        if (std::fabs(points[i].x-cpu.px[i]) > 1e-5f ||
            std::fabs(points[i].y-cpu.py[i]) > 1e-5f ||
            std::fabs(points[i].z-cpu.pz[i]) > 1e-5f ||
            std::fabs(rest[i].x-cpu.rest[i][0]) > 1e-5f ||
            std::fabs(rest[i].y-cpu.rest[i][1]) > 1e-5f ||
            std::fabs(rest[i].z-cpu.rest[i][2]) > 1e-5f ||
            std::fabs(widths[i]-cpu.width[i]) > 1e-5f ||
            std::fabs(hairT[i]-cpu.hairT[i]) > 1e-5f) return false;
    }
    for (auto const* planes : {&cpu.extraCv, &cpu.extraCurve}) {
        for (auto const& plane : *planes) {
            auto channel = gpu::AcquireNamedChannel(generation, plane.name.GetString(), nullptr);
            if (!channel) return false;
            auto bytes = channel.Bytes();
            if (plane.type == TfToken("float")) {
                std::vector<float> values(plane.f.size());
                if (bytes.size != values.size()*sizeof(float) ||
                    (bytes.size && cudaMemcpy(values.data(),bytes.data,bytes.size,
                        cudaMemcpyDeviceToHost)!=cudaSuccess)) return false;
                for (size_t i=0;i<values.size();++i)
                    if (std::fabs(values[i]-plane.f[i])>1e-5f) return false;
            } else {
                std::vector<int> values(plane.i.size());
                if (bytes.size != values.size()*sizeof(int) ||
                    (bytes.size && cudaMemcpy(values.data(),bytes.data,bytes.size,
                        cudaMemcpyDeviceToHost)!=cudaSuccess)) return false;
                for (size_t i=0;i<values.size();++i)
                    if (values[i]!=plane.i[i]) return false;
            }
        }
    }
    return true;
}
#endif

static bool Run(UsdGenGraphDesc const& d, UsdGenCurveBuffer* result,
                bool printErrors = true) {
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = compiler.Compile(d, &graph);
    if (!compiled.ok) return false;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    auto run = scheduler.Run(graph, ctx, 1);
    if (printErrors) for (auto const& e : run.diagnostics.errors)
        std::fprintf(stderr, "%s\n", e.c_str());
    if (run.diagnostics.HasErrors()) return false;
    *result = graph.Output();
#ifdef USDGEN_TEST_CUDA_SOURCE_PARITY
    if (!CudaParity(d,*result)) return false;
#endif
    return true;
}

int main() {
#ifdef USDGEN_TEST_CUDA_SOURCE_PARITY
    int deviceCount=0;
    if (cudaGetDeviceCount(&deviceCount)!=cudaSuccess || deviceCount==0) return 77;
#endif
    usdGenRegisterM1Operators();
    auto d = Desc();
    UsdGenCurveBuffer out;
    CHECK(Run(d, &out));
    CHECK(out.totalCurves == 2 && out.totalCvs == 5);
    CHECK(out.curveId == VtArray<uint64_t>({7,40}));
    CHECK(out.cvOffsets == VtIntArray({0,2,5}));
    CHECK(out.px == VtFloatArray({120,121,110,111,112}));
    CHECK(out.rest == VtVec3fArray({GfVec3f(20,0,0),GfVec3f(21,0,0),
        GfVec3f(10,0,0),GfVec3f(11,0,0),GfVec3f(12,0,0)}));
    CHECK(out.width == VtFloatArray({.4f,.5f,.1f,.2f,.3f}));
    CHECK(out.hairT == VtFloatArray({0,1,0,.5f,1}));
    CHECK(out.rootPrim == VtIntArray({1,2}));
    CHECK(out.rootUV == VtVec2fArray({GfVec2f(.4f,.5f), GfVec2f(.2f,.3f)}));
    CHECK(out.rootT == VtVec3fArray(2,GfVec3f(1,0,0)));
    CHECK(out.rootB == VtVec3fArray(2,GfVec3f(0,1,0)));
    CHECK(out.rootN == VtVec3fArray(2,GfVec3f(0,0,1)));
    CHECK(out.extraCv.size() == 1 && out.extraCv[0].f == VtFloatArray({3,4,0,1,2}));
    CHECK(out.extraCurve.size() == 1 && out.extraCurve[0].i == VtIntArray({7,40}));
    CHECK(d.curveSets[0].curveId == VtArray<uint64_t>({40,7}));
    CHECK(d.curveSets[0].widths == VtFloatArray({.1f,.2f,.3f,.4f,.5f}));
    // Ordinary C3 CurveSource remains the default path.  surfaceCage is an
    // explicit expansion mode and must not require cage payload, a bound map,
    // or a bound surface when absent (or when authored explicitly as none).
    {
        auto ordinary = d;
        ordinary.nodes[0].params.push_back(
            {TfToken("interpolationMode"), VtValue(TfToken("none")), false});
        UsdGenCurveBuffer ordinaryOut;
        CHECK(Run(ordinary, &ordinaryOut));
        CHECK(ordinaryOut.px == out.px && ordinaryOut.rest == out.rest &&
              ordinaryOut.width == out.width && ordinaryOut.curveId == out.curveId &&
              ordinaryOut.rootPrim == out.rootPrim && ordinaryOut.rootUV == out.rootUV);
    }
    {
        auto unbound = d;
        unbound.curveSets[0].rootFrame.clear();
        unbound.curveSets[0].skinPrim.clear();
        unbound.curveSets[0].skinPrimUv.clear();
        UsdGenCurveBuffer raw;
        CHECK(Run(unbound,&raw) && raw.px == out.px && raw.rootPrim.empty() &&
              raw.rootUV.empty() && raw.rootT.empty() && raw.rootN.empty());
    }

    {
        auto derived = d;
        auto& hair = derived.curveSets[0];
        hair.rootFrame.clear();
        hair.skinPrim = {0,1};
        hair.skinPrimUv = {{.2f,.3f},{.4f,.5f}};
        UsdGenSurfaceDesc scalp;
        scalp.path = SdfPath("/Scalp");
        scalp.restPoints = {{0,0,0},{1,0,0},{0,1,0},
                            {2,0,0},{3,0,0},{2,1,0}};
        scalp.points = scalp.restPoints;
        scalp.faceVertexCounts = {3,3};
        scalp.faceVertexIndices = {0,1,2,3,4,5};
        derived.surfaces = {scalp};
        derived.nodes[0].surfaces = {scalp.path};
        UsdGenCurveBuffer framed;
        auto automatic = derived;
        automatic.nodes[0].params[0].value = VtValue(TfToken("onError"));
        automatic.curveSets[0].skinPrim.clear();
        automatic.curveSets[0].skinPrimUv.clear();
        CHECK(Run(automatic,&framed) && framed.totalCurves == 2 &&
              framed.rootPrim.size() == 2 && framed.rootUV.size() == 2);
        CHECK(automatic.curveSets[0].skinPrim.empty() &&
              automatic.curveSets[0].skinPrimUv.empty() &&
              automatic.curveSets[0].rootFrame.empty());
        CHECK(Run(derived,&framed));
        CHECK(framed.curveId == out.curveId && framed.px == out.px &&
              framed.rootT == out.rootT && framed.rootN == out.rootN);
        {
            UsdGenCompiler frameCompiler;
            UsdGenGraph frameGraph;
            CHECK(frameCompiler.Compile(derived,&frameGraph).ok);
            UsdGenScheduler frameScheduler(2);
            UsdGenEvalContext frameCtx; frameCtx.desc = &frameGraph.Desc();
            CHECK(!frameScheduler.Run(frameGraph,frameCtx,1).diagnostics.HasErrors());
            auto const frozen = frameGraph.Output();
            auto posed = derived;
            posed.surfaces[0].points[0] += GfVec3f(0,0,2);
            posed.surfaces[0].surfaceGeneration = 91;
            posed.surfaces[0].worldMatrix.SetTranslate(GfVec3d(.4,0,0));
            posed.curveSets[0].worldMatrix = posed.surfaces[0].worldMatrix;
            posed.surfaces[0].samples.push_back({38,posed.surfaces[0].points});
            CHECK(frameCompiler.Recompile(posed,&frameGraph).ok);
            frameCtx.desc = &frameGraph.Desc();
            CHECK(!frameScheduler.Run(frameGraph,frameCtx,2).diagnostics.HasErrors());
            // Rest root capture survives changing posed points/sample times;
            // reusing its COW storage proves generation was not rerun.
            CHECK(frameGraph.Output().px.cdata() == frozen.px.cdata());
            auto priorNormals = frameGraph.Output().rootN;
            auto recaptured = derived;
            recaptured.surfaces[0].restNormalDomain = UsdGenSurfaceNormalDomain::Constant;
            recaptured.surfaces[0].restNormals = {{0,1,1}};
            // Deliberately no surfaceGeneration change: actual payload edits
            // must invalidate the source capture independently of that hint.
            CHECK(frameCompiler.Recompile(recaptured,&frameGraph).ok);
            frameCtx.desc = &frameGraph.Desc();
            CHECK(!frameScheduler.Run(frameGraph,frameCtx,3).diagnostics.HasErrors());
            CHECK(frameGraph.Output().rootN != priorNormals &&
                  priorNormals == VtVec3fArray(2,GfVec3f(0,0,1)));
        }
        auto old = framed;
        // Drop the authored second curve (stable ID 7) while preserving all
        // data belonging to the first curve and the previous COW generation.
        derived.surfaces[0].restPoints[4] = derived.surfaces[0].restPoints[3];
        CHECK(Run(derived,&framed));
        CHECK(framed.totalCurves == 1 && framed.totalCvs == 3 &&
              framed.curveId == VtArray<uint64_t>({40}) &&
              framed.px == VtFloatArray({110,111,112}) &&
              framed.width == VtFloatArray({.1f,.2f,.3f}) &&
              framed.extraCv[0].f == VtFloatArray({0,1,2}) &&
              framed.extraCurve[0].i == VtIntArray({40}));
        CHECK(old.totalCurves == 2 && old.curveId == VtArray<uint64_t>({7,40}) &&
              old.px == out.px && hair.rootFrame.empty());
        hair.curveId.clear();
        CHECK(Run(derived,&framed) && framed.curveId == VtArray<uint64_t>({0}));
        derived.surfaces[0].restPoints[1] = derived.surfaces[0].restPoints[0];
        CHECK(Run(derived,&framed) && framed.totalCurves == 0 && framed.totalCvs == 0);
        derived.surfaces[0] = scalp;
        // Authored frames bypass surface frame construction entirely. A
        // binding still names a valid face, but no surface rest payload is
        // required to consume the already-authored frame.
        auto explicitFrames = derived;
        explicitFrames.curveSets[0].rootFrame.assign(2,GfMatrix4d(1.0));
        explicitFrames.surfaces[0].restPoints.clear();
        CHECK(Run(explicitFrames,&framed) && framed.totalCurves == 2);
        derived.surfaces[0].faceVertexCounts = {5};
        derived.surfaces[0].faceVertexIndices = {0,1,4,5,2};
        hair.skinPrim = {0,0};
        hair.skinPrimUv = {{1.2f,.3f},{2.2f,.3f}};
        CHECK(Run(derived,&framed) && framed.totalCurves == 2);
    }

    auto styled = d;
    UsdGenNodeDesc width;
    width.path = SdfPath("/Width"); width.type = TfToken("UsdGenWidth");
    width.inputs = {d.terminal};
    width.params = {{TfToken("width"), VtValue(.25f), false}};
    styled.nodes.push_back(width); styled.terminal = width.path;
    auto sourceRest = out.rest;
    CHECK(Run(styled,&out) && out.width == VtFloatArray(5,.25f) && out.rest == sourceRest);
    {
        UsdGenCompiler compiler;
        UsdGenGraph graph;
        CHECK(compiler.Compile(styled,&graph).ok);
        UsdGenScheduler scheduler(2);
        UsdGenEvalContext ctx; ctx.desc = &graph.Desc();
        CHECK(!scheduler.Run(graph,ctx,1).diagnostics.HasErrors());
        auto const& source = graph.Node(graph.NodeIdForPath(d.terminal)).buffer;
        CHECK(graph.Output().rest.cdata() == source.rest.cdata());
        CHECK(graph.Output().width.cdata() != source.width.cdata());
    }
    // CPU topology prerequisite: ragged input roots use their actual offsets,
    // and the generated rest geometry follows rest roots, not posed roots.
    {
        auto grown = d;
        UsdGenNodeDesc grow;
        grow.path = SdfPath("/Grow"); grow.type = TfToken("UsdGenGrow");
        grow.inputs = {d.terminal};
        grow.params = {{TfToken("segments"),VtValue(4),false},
                       {TfToken("length"),VtValue(1.0f),false}};
        grown.nodes.push_back(grow); grown.terminal = grow.path;
        UsdGenCompiler compiler;
        UsdGenGraph graph;
        CHECK(compiler.Compile(grown,&graph).ok);
        UsdGenScheduler scheduler(2);
        UsdGenEvalContext ctx; ctx.desc = &graph.Desc();
        auto run = scheduler.Run(graph,ctx,1);
        for (auto const& error : run.diagnostics.errors) std::fprintf(stderr,"%s\n",error.c_str());
        CHECK(!run.diagnostics.HasErrors());
        auto const& source = graph.Node(graph.NodeIdForPath(d.terminal)).buffer;
        auto const& output = graph.Output();
        CHECK(output.totalCvs == 8 && output.rest.size() == 8);
        CHECK(output.px[0] == 120 && output.px[4] == 110);
        CHECK(output.pz[3] == 1 && output.pz[7] == 1);
        CHECK(output.rest[0] == GfVec3f(20,0,0) && output.rest[3] == GfVec3f(20,0,1));
        CHECK(output.rest[4] == GfVec3f(10,0,0) && output.rest[7] == GfVec3f(10,0,1));
        CHECK(output.rest.cdata() != source.rest.cdata());
        CHECK(source.rest == sourceRest);
    }

    auto resampled = d;
    resampled.nodes[0].params.push_back({TfToken("resampleTo"),VtValue(4),false});
    CHECK(Run(resampled, &out));
    CHECK(out.totalCvs == 8 && out.cvOffsets.empty());
    CHECK(out.curveId == VtArray<uint64_t>({7,40}));
    CHECK(out.px.front() == 120 && out.px[3] == 121 && out.px[4] == 110 && out.px[7] == 112);
    CHECK(out.rest.front()[0] == 20 && out.rest[3][0] == 21 && out.rest[7][0] == 12);
    CHECK(std::fabs(out.width[1] - (.4f + .1f/3)) < 1e-6f);
    CHECK(std::fabs(out.extraCv[0].f[1] - (3.f + 1.f/3)) < 1e-6f);
    CHECK(out.hairT[0] == 0 && out.hairT[3] == 1 && out.hairT[4] == 0 && out.hairT[7] == 1);

    auto fallback = d;
    fallback.curveSets[0].widths.clear();
    CHECK(Run(fallback, &out) && out.width == VtFloatArray(5,.125f));
    fallback.curveSets[0].widths = {.75f};
    fallback.curveSets[0].widthsInterpolation = TfToken("constant");
    CHECK(Run(fallback, &out) && out.width == VtFloatArray(5,.75f));
    fallback = d;
    fallback.curveSets[0].curveId.clear();
    CHECK(Run(fallback, &out));
    CHECK(out.curveId == VtArray<uint64_t>({0,1}) && out.px.front() == 110);
    auto current = d;
    current.nodes[0].params[1].value = VtValue(false);
    CHECK(Run(current, &out) && out.px == VtFloatArray({120,121,110,111,112}));
    current.curveSets[0].rest.clear();
    CHECK(Run(current, &out) && out.rest.size() == out.totalCvs);
    CHECK(out.rest[0] == GfVec3f(120,0,0) && out.rest[2] == GfVec3f(110,0,0));
    auto empty = d;
    auto& e = empty.curveSets[0];
    e.curveVertexCounts.clear(); e.points.clear(); e.rest.clear();
    e.widths.clear(); e.curveId.clear(); e.skinPrim.clear();
    e.skinPrimUv.clear(); e.rootFrame.clear(); e.authoredPlanes.clear();
    CHECK(Run(empty, &out) && out.totalCurves == 0 && out.totalCvs == 0);
    CHECK(out.px.empty() && out.width.empty() && out.curveId.empty());
    auto exact = d;
    auto& x = exact.curveSets[0];
    x.curveVertexCounts = {14};
    x.points = VtVec3fArray(14,GfVec3f(0)); x.points[8] = GfVec3f(1000000,0,0);
    x.rest = x.points; x.widths = VtFloatArray(14,.25f);
    x.curveId = {9}; x.skinPrim = {0}; x.skinPrimUv = {{0,0}};
    x.rootFrame = {GfMatrix4d(1.0)}; x.authoredPlanes.clear();
    CHECK(Run(exact,&out) && out.cvOffsets.empty());
    CHECK(out.px[7] == 0 && out.px[8] == 1000000 && out.rest == x.rest);

    // Recompile without a generation bump: capture identity must include the
    // authored payload, and old published arrays remain immutable COW data.
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    CHECK(compiler.Compile(d, &graph).ok);
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    CHECK(!scheduler.Run(graph, ctx, 1).diagnostics.HasErrors());
    auto old = graph.Output();
    d.curveSets[0].rest[3][0] = 30;
    d.curveSets[0].points[3][0] = 130;
    d.curveSets[0].widths[3] = .9f;
    CHECK(compiler.Recompile(d, &graph).ok);
    ctx.desc = &graph.Desc();
    CHECK(!scheduler.Run(graph, ctx, 2).diagnostics.HasErrors());
    CHECK(graph.Output().px.front() == 130 && graph.Output().width.front() == .9f);
    CHECK(graph.Output().rest.front()[0] == 30);
    CHECK(old.px.front() == 120 && old.width.front() == .4f && old.rest.front()[0] == 20);
    CHECK(old.px.cdata() != graph.Output().px.cdata());
    CHECK(old.width.cdata() != graph.Output().width.cdata());
    CHECK(old.rest.cdata() != graph.Output().rest.cdata());

    // Failures must not overwrite a caller's last-good source value.
    auto good = graph.Output();
    d.curveSets[0].curveId = {7,7};
    CHECK(compiler.Recompile(d, &graph).ok);
    ctx.desc = &graph.Desc();
    CHECK(scheduler.Run(graph,ctx,3).diagnostics.HasErrors());
    CHECK(graph.Output().px == good.px && graph.Output().width == good.width &&
          graph.Output().rest == good.rest);
    for (int variant = 0; variant != 10; ++variant) {
        auto bad = Desc();
        auto& c = bad.curveSets[0];
        if (variant == 0) c.points.pop_back();
        if (variant == 1) c.widths[0] = -1;
        if (variant == 2) c.widths[0] = std::numeric_limits<float>::quiet_NaN();
        if (variant == 3) c.curveVertexCounts[0] = 1;
        // Linear polylines are accepted now; a cubic bezier basis is not.
        if (variant == 4) c.basis = TfToken("bezier");
        if (variant == 5) c.widthsInterpolation = TfToken("varying");
        if (variant == 6) bad.nodes[0].curves = {SdfPath("/Missing")};
        if (variant == 7) bad.nodes[0].curves.push_back(SdfPath("/Other"));
        if (variant == 8) {
            bad.nodes[0].params[1].value = VtValue(false);
            c.rest.pop_back();
        }
        if (variant == 9) {
            bad.nodes[0].params[1].value = VtValue(false);
            c.rest[0][0] = std::numeric_limits<float>::infinity();
        }
        CHECK(!Run(bad, &out, false));
    }
    std::puts("testUsdGenCpuCurveSource: PASS");
    return 0;
}
