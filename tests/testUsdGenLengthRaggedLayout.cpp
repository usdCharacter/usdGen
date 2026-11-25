#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace usdGen;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (false)

namespace {

UsdGenGraphDesc MakeDescriptor(double factor, bool empty = false) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/LengthRagged");
    desc.executionBackend = UsdGenExecutionBackend::CpuReference;

    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/LengthRagged/curves");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2, 4, 3};
    curves.curveId = {100, 200, 300};
    curves.widths = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    curves.points = {{100, 0, 0}, {101, 1, 0},
                     {200, 0, 0}, {201, 1, 0}, {202, 2, 0}, {203, 3, 0},
                     {300, 0, 0}, {301, 1, 0}, {302, 2, 0}};
    curves.rest = {{100, 0, 10}, {101, 1, 10},
                   {200, 0, 10}, {201, 1, 10}, {202, 2, 10}, {203, 3, 10},
                   {300, 0, 10}, {301, 1, 10}, {302, 2, 10}};
    if (empty) { curves.points.clear(); curves.rest.clear(); curves.curveVertexCounts.clear(); curves.curveId.clear(); curves.widths.clear(); }
    curves.widthsInterpolation = TfToken("vertex");
    curves.curveGeneration = 17;
    desc.curveSets.push_back(curves);

    UsdGenNodeDesc source;
    source.path = SdfPath("/LengthRagged/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                     {TfToken("useRest"), VtValue(false), false}};
    UsdGenNodeDesc length;
    length.path = SdfPath("/LengthRagged/length");
    length.type = TfToken("UsdGenLength");
    length.inputs = {source.path};
    length.params = {{TfToken("length:value"), VtValue(factor), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/LengthRagged/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {length.path};
    width.params = {{TfToken("width"), VtValue(.5f), false}};
    desc.nodes = {source, length, width};
    desc.terminal = width.path;
    return desc;
}

int Run(double factor, bool empty = false) {
    auto desc = MakeDescriptor(factor, empty);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = compiler.Compile(desc, &graph);
    CHECK(compiled.ok);
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    auto run = scheduler.Run(graph, context, 1);
    CHECK(!run.diagnostics.HasErrors());
    UsdGenCurveBuffer const buffer = NodeBuffer(graph, graph.NodeIdForPath(desc.terminal));
    if (empty) {
        UsdGenCurveBuffer const source = NodeBuffer(graph, graph.NodeIdForPath(desc.nodes.front().path));
        std::fprintf(stderr, "empty Length: curves=%u cvs=%u offsets=%zu sourceOffsets=%zu\n",
            buffer.totalCurves, buffer.totalCvs, buffer.cvOffsets.size(), source.cvOffsets.size());
        CHECK(buffer.totalCurves == 0 && buffer.totalCvs == 0);
        CHECK(buffer.cvOffsets == source.cvOffsets && buffer.curveId == source.curveId);
        CHECK(buffer.rest.empty() && buffer.width.empty() && buffer.hairT.empty());
        return 0;
    }
    CHECK(buffer.totalCvs == 9 && buffer.cvOffsets.size() == 4);
    CHECK(buffer.cvOffsets[0] == 0 && buffer.cvOffsets[1] == 2 &&
          buffer.cvOffsets[2] == 6 && buffer.cvOffsets[3] == 9);
    CHECK(buffer.curveId.size() == 3 && buffer.curveId[0] == 100 &&
          buffer.curveId[1] == 200 && buffer.curveId[2] == 300);
    if (factor == 1.0) {
        CHECK(buffer.px[1] == 101 && buffer.py[1] == 1 &&
              buffer.px[5] == 203 && buffer.py[5] == 3);
    }
    if (factor == 0.0) {
        for (size_t curve = 0; curve < 3; ++curve)
            for (int i = buffer.cvOffsets[curve]; i < buffer.cvOffsets[curve + 1]; ++i)
                CHECK(buffer.px[i] == float(100 * (curve + 1)) && buffer.py[i] == 0 && buffer.pz[i] == 0);
    }
    CHECK(buffer.rest == desc.curveSets[0].rest);
    return 0;
}

} // namespace

int main() {
    auto desc = MakeDescriptor(1.25);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = compiler.Compile(desc, &graph);
    CHECK(compiled.ok);
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context; context.desc = &graph.Desc();
    auto run = scheduler.Run(graph, context, 1);
    CHECK(!run.diagnostics.HasErrors());
    UsdGenCurveBuffer const buffer = NodeBuffer(graph, graph.NodeIdForPath(desc.terminal));
    CHECK(buffer.totalCvs == 9 && buffer.cvOffsets.size() == 4 &&
          buffer.cvOffsets[0] == 0 && buffer.cvOffsets[1] == 2 &&
          buffer.cvOffsets[2] == 6 && buffer.cvOffsets[3] == 9);
    CHECK(buffer.curveId.size() == 3 && buffer.curveId[0] == 100 &&
          buffer.curveId[1] == 200 && buffer.curveId[2] == 300);
    std::vector<GfVec3f> expected = {{100, 0, 0}, {101.25f, 1.25f, 0},
        {200, 0, 0}, {201.25f, 1.25f, 0}, {202.5f, 2.5f, 0}, {203.75f, 3.75f, 0},
        {300, 0, 0}, {301.25f, 1.25f, 0}, {302.5f, 2.5f, 0}};
    for (size_t i = 0; i != expected.size(); ++i)
        CHECK(buffer.px[i] == expected[i][0] && buffer.py[i] == expected[i][1] && buffer.pz[i] == expected[i][2]);
    CHECK(buffer.rest == desc.curveSets[0].rest);
    CHECK(Run(0.0) == 0 && Run(1.0) == 0 && Run(1.25, true) == 0);
    std::puts("UsdGen Length ragged layout: PASS");
    return 0;
}
