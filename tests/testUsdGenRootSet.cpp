// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenRootSet: the RootSet primitive and typed geometry ports (6.2).
//   - the type registers with a one-level layout, statics in slot order
//   - Scatter declares a RootSet output; Grow and GuideInterpolate accept it
//   - the compiler rejects an edge whose type the consumer does not take;
//     RootSet converts explicitly into ops that take the default primitive
//   - UsdGenRootSetView reads Scatter's cooked roots field for field
//   - view -> RootSet -> view is bitwise; the partition hook chunks roots

#include "nodeSets.h"
#include "usdGenSurface/types/rootSet.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/typeRegistry.h"
#include "usdGenUsd/lower.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "utils/pxrInterop.h"

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T>
bool Same(UsdGenSpan<const T> a, UsdGenSpan<const T> b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool SameRoots(UsdGenRootSetView const &a, UsdGenRootSetView const &b)
{
    return a.totalCurves == b.totalCurves && Same(a.px, b.px) && Same(a.py, b.py) && Same(a.pz, b.pz) &&
           Same(a.curveId, b.curveId) && Same(a.rootPrim, b.rootPrim) && Same(a.rootUV, b.rootUV) &&
           Same(a.rootT, b.rootT) && Same(a.rootN, b.rootN) && Same(a.rootB, b.rootB);
}

SdfPath const kRoot("/rootSet");

UsdGenNodeDesc Node(char const *name, char const *type, int seed)
{
    UsdGenNodeDesc n;
    n.path = kRoot.AppendChild(TfToken(name));
    n.type = TfToken(type);
    n.enabled = true;
    n.seed = seed;
    return n;
}

/// Scatter on a 2x2 quad grid, then 'consumer' reading the scatter (Grow
/// when null), then 'tail' reading the consumer when given.
UsdGenGraphDesc MakeDesc(char const *consumer, char const *tail = nullptr,
                         char const *source = "UsdGenScatter")
{
    UsdGenGraphDesc d;
    d.description = kRoot;
    d.curveBasis = TfToken("bspline");
    d.xformMatrix = GfMatrix4d(1.0);
    UsdGenSurfaceDesc surf;
    surf.path = kRoot.AppendChild(TfToken("surface"));
    surf.id = 0;
    surf.faceVertexCounts = VtIntArray(4, 4);
    surf.faceVertexIndices = VtIntArray{0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7};
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x) surf.restPoints.push_back(GfVec3f(float(x), float(y), 0.0f));
    surf.points = surf.restPoints;
    d.surfaces.push_back(surf);
    UsdGenNodeDesc scatter = Node("scatter", source, 3);
    scatter.surfaces.push_back(surf.path);
    scatter.params = {{TfToken("density"), VtValue(40.0f), false}};
    d.nodes.push_back(scatter);
    UsdGenNodeDesc read = Node("read", consumer ? consumer : "UsdGenGrow", 4);
    read.inputs = {scatter.path};
    d.nodes.push_back(read);
    d.terminal = read.path;
    if (tail) {
        UsdGenNodeDesc last = Node("tail", tail, 5);
        last.inputs = {read.path};
        d.nodes.push_back(last);
        d.terminal = last.path;
    }
    return d;
}

void TestType()
{
    UsdGenTypeId const id = UsdGenRootSetType();
    UsdGenElementLayout const &layout = UsdGenRootSetLayout();
    Check(id != UsdGenTypeId::Invalid, "RootSet registers");
    Check(UsdGenTypeRegistry::Get().Find(TfToken("UsdGenRootSet")) == id, "type found by name");
    Check(UsdGenTypeRegistry::Get().Def(id).layout == &layout, "type def points at the layout");
    Check(layout.primitive == id && layout.levels == 1 && layout.channels.size() == kUsdGenRootStaticSlots,
          "one level, nine static channels");
    bool levels = true;
    for (UsdGenChannelDesc const &d : layout.channels) levels = levels && d.level == 1;
    Check(levels, "every static channel is per root");
    Check(layout.channels[kUsdGenRootPx].role == UsdGenChannelRole::Position &&
              layout.channels[kUsdGenRootPz].component == 2 &&
              layout.channels[kUsdGenRootId].scalar == UsdGenScalar::U64 &&
              layout.channels[kUsdGenRootUV].arity == 2 &&
              layout.channels[kUsdGenRootN].role == UsdGenChannelRole::Frame &&
              layout.channels[kUsdGenRootN].component == 1,
          "static channels in slot order");
}

void TestPorts()
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    auto record = [&](char const *name) -> UsdGenOpRecord const & {
        return ops.Record(ops.Find(TfToken(name)));
    };
    TfToken const roots("UsdGenRootSet");
    Check(record("UsdGenScatter").OutputType() == roots, "Scatter outputs a RootSet");
    Check(record("UsdGenGrow").OutputType().IsEmpty(), "Grow outputs the default primitive");
    Check(record("UsdGenGrow").AcceptsInput(roots) && record("UsdGenGrow").AcceptsInput(TfToken()),
          "Grow accepts roots and curves");
    Check(record("UsdGenGuideInterpolate").AcceptsInput(roots) &&
              record("UsdGenGuideInterpolate").AcceptsInput(TfToken()),
          "GuideInterpolate accepts roots and curves");
    Check(!record("UsdGenNoise").AcceptsInput(roots) && record("UsdGenNoise").AcceptsInput(TfToken()),
          "Noise accepts curves only");
}

void TestEdges()
{
    UsdGenCompiler compiler;
    {
        UsdGenGraph graph;
        Check(usdGenUsd::Compile(compiler, MakeDesc(nullptr, "UsdGenNoise"), &graph).ok,
              "Scatter -> Grow -> Noise compiles");
    }
    {
        UsdGenGraph graph;
        UsdGenCompileResult const r = usdGenUsd::Compile(compiler, MakeDesc("UsdGenGuideInterpolate"), &graph);
        bool typed = false;
        for (std::string const &e : r.errors) typed = typed || e.find("UsdGenRootSet") != std::string::npos;
        Check(!typed, "Scatter -> GuideInterpolate is not a type error");
    }
    // Roots read as one-CV curves (examples/guide-interpolate-plane.usda).
    for (char const *consumer : {"UsdGenNoise", "UsdGenWidth"}) {
        UsdGenGraph graph;
        Check(usdGenUsd::Compile(compiler, MakeDesc(consumer), &graph).ok,
              (std::string("Scatter -> ") + consumer + " compiles through the RootSet conversion").c_str());
    }
    // A type with no conversion is rejected with a typed status.
    for (char const *consumer : {"UsdGenNoise", "UsdGenGrow"}) {
        UsdGenGraph graph;
        UsdGenCompileResult const r =
            usdGenUsd::Compile(compiler, MakeDesc(consumer, nullptr, "UsdGenTestPointSource"), &graph);
        bool named = false;
        for (std::string const &e : r.errors)
            named = named || (e.find("UsdGenTestPoints") != std::string::npos &&
                              e.find("cannot take input") != std::string::npos &&
                              e.find("scatter") != std::string::npos);
        Check(!r.ok && named, (std::string("TestPoints -> ") + consumer + " is rejected").c_str());
    }
}

/// A primitive type without a conversion and a source producing it (a clone
/// of Scatter's def), registered before the registries freeze.
void RegisterTestPoints(UsdGenRegistrar &r)
{
    UsdGenTypeDef type;
    type.name = "UsdGenTestPoints";
    r.RegisterType(type);
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpDef d = ops.Record(ops.Find(TfToken("UsdGenScatter"))).Def();
    d.name = "UsdGenTestPointSource";
    d.outputType = "UsdGenTestPoints";
    r.RegisterOp(d);
}

void TestCookedRoots()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenGraphDesc const desc = MakeDesc(nullptr);
    Check(usdGenUsd::Compile(compiler, desc, &graph).ok, "Scatter -> Grow compiles");
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Spec();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(), "Scatter -> Grow cooks");
    UsdGenNodeId const scatterId = usdGenUsd::NodeIdForPath(graph, desc.nodes[0].path);
    UsdGenElementSet const &out = graph.NodeOutput(scatterId);
    UsdGenRootSetView const roots = UsdGenRootSetView::Of(out);
    UsdGenCurveSetView const curves = UsdGenCurveSetView::Of(out);
    Check(roots.valid && roots.totalCurves > 0 && roots.totalCurves == curves.totalCurves,
          "the scatter output reads as roots");
    Check(Same(roots.px, curves.px) && Same(roots.py, curves.py) && Same(roots.pz, curves.pz) &&
              Same(roots.curveId, curves.curveId) && Same(roots.rootPrim, curves.rootPrim) &&
              Same(roots.rootUV, curves.rootUV) && Same(roots.rootT, curves.rootT) &&
              Same(roots.rootN, curves.rootN) && Same(roots.rootB, curves.rootB),
          "root view fields equal the curve view's");
    Check(roots.topologyVersion == curves.topologyVersion && roots.valueVersion == curves.valueVersion,
          "versions carried");
    UsdGenRootSetView const grown = UsdGenRootSetView::Of(graph.Output());
    Check(!grown.valid && grown.totalCurves == 0, "grown curves are not roots");

    UsdGenRef<const UsdGenElementSet> set = UsdGenRootSetFromView(roots, 7, 9);
    Check(bool(set), "view -> RootSet");
    if (!set) return;
    Check(&set->Channels().Layout() == &UsdGenRootSetLayout() && set->Count(1) == roots.totalCurves &&
              set->TopologyVersion() == 7 && set->ValueVersion() == 9,
          "RootSet layout, counts and versions");
    UsdGenRootSetView const again = UsdGenRootSetView::Of(*set);
    Check(again.valid && SameRoots(roots, again), "RootSet -> view is bitwise");
    Check(!UsdGenCurveSetView::Of(*set).totalCurves, "a RootSet is not a curve set");

    UsdGenPartitionParams params;
    params.chunkSize = 4;
    uint32_t const n = roots.totalCurves;
    uint32_t const expected = (n + 3) / 4;
    std::vector<UsdGenChunkRange> chunks(expected);
    uint32_t const got = UsdGenRootSetLayout().partition(
        n, set->View(), params, UsdGenSpan<UsdGenChunkRange>(chunks.data(), chunks.size()));
    bool tiled = got == expected;
    uint32_t covered = 0;
    for (uint32_t c = 0; tiled && c < got; ++c) {
        tiled = chunks[c].first[0] == covered && chunks[c].count[0] == std::min(4u, n - covered) &&
                chunks[c].live == chunks[c].count[0] && chunks[c].count[1] == 0;
        covered += chunks[c].count[0];
    }
    Check(tiled && covered == n, "partition chunks every root once");
    std::vector<UsdGenChunkGroup> groups(got);
    uint32_t const nGroups = UsdGenRootSetLayout().groups(
        got, params, UsdGenSpan<UsdGenChunkGroup>(groups.data(), groups.size()));
    uint32_t chunkSum = 0;
    for (uint32_t g = 0; g < nGroups; ++g) chunkSum += groups[g].chunkCount;
    Check(nGroups >= 1 && chunkSum == got, "groups cover every chunk");
}

}  // namespace

int main()
{
    std::string error;
    UsdGenEngineEnsureInitialized(&error);
    Check(UsdGenRegisterStatic(&RegisterTestPoints), "test types register before the first compile");
    TestType();
    TestPorts();
    TestEdges();
    TestCookedRoots();
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenRootSet: ok\n");
    return 0;
}
