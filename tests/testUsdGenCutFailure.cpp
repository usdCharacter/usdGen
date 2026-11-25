// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Transactional CPU publication coverage for fallible vertex remappers.

#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/scheduler.h"
#include "usdGenCurves/types/curveSet.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

std::atomic<bool> g_failEnabled{false};
std::atomic<uint32_t> g_evaluations{0};
int g_failures = 0;

void Check(bool condition, char const *message)
{
    std::printf("%s: %s\n", condition ? "ok" : "FAIL", message);
    if (!condition) ++g_failures;
}

class FallibleRemap final : public UsdGenOp
{
public:
    explicit FallibleRemap(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    uint32_t PlanesTouched() const override { return kPlanePoints | kPlaneWidths; }
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override
    {
        return {0x4355544641494cULL, 1};
    }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override
    {
        return true;
    }
    void Evaluate(UsdGenEvalContext const &ctx, UsdGenCapture const &,
                  UsdGenChunkView const &chunk) const override
    {
        UsdGenCurveChunk const curve(chunk);
        UsdGenCurveChunk const *view = &curve;
        uint32_t count = view->curveCount * view->cvCount;
        if (view->cvCount == 0 && view->cvOffsets)
            count = uint32_t(view->cvOffsets[view->curveCount] -
                             view->cvOffsets[0]);
        for (uint32_t cv = 0; cv < count; ++cv) {
            view->px[cv] = view->inPx[cv];
            view->py[cv] = view->inPy[cv];
            view->pz[cv] = view->inPz[cv];
            if (view->width && view->inWidth)
                view->width[cv] = view->inWidth[cv] + 1.0f;
            for (uint32_t plane=0;plane<view->extraCvCount;++plane) {
                auto &field=view->extraCv[plane];
                size_t const index=(view->desc ? view->desc->firstCv : 0)+cv;
                // Whole-column bases: the node's plane and its input's.
                if (field.type==TfToken("float") && view->extraCvF[plane] &&
                    view->inExtraCvF[plane])
                    view->extraCvF[plane][index] = view->inExtraCvF[plane][index] + 1.0f;
            }
        }

        // Fail only after another chunk has completed its writes. The fixed
        // two-chunk fixture makes the failure independent of authored values
        // while still exercising a partially evaluated candidate buffer.
        uint32_t const call = g_evaluations.fetch_add(1, std::memory_order_acq_rel);
        if (g_failEnabled.load(std::memory_order_acquire) && call >= 1 &&
            ctx.failureCode) {
            uint32_t expected = 0;
            ctx.failureCode->compare_exchange_strong(
                expected, 77, std::memory_order_acq_rel);
        }
    }
};

UsdGenCurveSetDesc MakeCurves()
{
    constexpr int curveCount = 600;
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/cutFailure/curves");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts.assign(curveCount, 2);
    curves.points.reserve(curveCount * 2);
    curves.rest.reserve(curveCount * 2);
    curves.curveId.reserve(curveCount);
    curves.skinPrim.assign(curveCount, 0);
    curves.skinPrimUv.assign(curveCount, GfVec2f(0.0f));
    curves.rootFrame.assign(curveCount, GfMatrix4d(1.0));
    for (int curve = 0; curve < curveCount; ++curve) {
        float const x = float(curve) * 0.01f;
        curves.points.push_back(GfVec3f(x, 0, 0));
        curves.points.push_back(GfVec3f(x, 0, 1));
        curves.curveId.push_back(uint64_t(curve + 1));
    }
    curves.rest = curves.points;
    UsdGenAuthoredPlaneDesc field;
    field.name=TfToken("failureVertex");
    field.type=UsdGenAuthoredPlaneType::Float32;
    field.domain=UsdGenAuthoredPlaneDomain::Point;
    field.floatValues.assign(curves.points.size(),0.125f);
    curves.authoredPlanes.push_back(std::move(field));
    return curves;
}

UsdGenNodeDesc Source(char const *name)
{
    UsdGenNodeDesc node;
    node.path = SdfPath(std::string("/cutFailure/") + name);
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {SdfPath("/cutFailure/curves")};
    node.params.push_back({TfToken("resampleTo"), VtValue(2), false});
    return node;
}

UsdGenNodeDesc Width(char const *name, SdfPath const &input, float value)
{
    UsdGenNodeDesc node;
    node.path = SdfPath(std::string("/cutFailure/") + name);
    node.type = TfToken("UsdGenWidth");
    node.inputs = {input};
    node.params.push_back({TfToken("width"), VtValue(value), false});
    return node;
}

UsdGenGraphDesc MakeGraph(bool unequalDepth)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/cutFailure");
    desc.terminal = SdfPath("/cutFailure/terminal");
    desc.curveSets.push_back(MakeCurves());

    UsdGenNodeDesc terminalSource = Source("terminalSource");
    UsdGenNodeDesc terminal = Width("terminal", terminalSource.path, 0.25f);
    UsdGenNodeDesc failureSource = Source("failureSource");
    UsdGenNodeDesc failure;
    failure.path = SdfPath("/cutFailure/failure");
    failure.type = TfToken("UsdGenTestFallibleRemap");
    failure.inputs = {failureSource.path};

    desc.nodes = {terminal, failure, terminalSource, failureSource};
    if (unequalDepth) {
        UsdGenNodeDesc delay = Width("delay", failureSource.path, 0.5f);
        desc.nodes[1].inputs = {delay.path};
        desc.nodes.push_back(std::move(delay));
    }
    return desc;
}

struct OutputSnapshot
{
    std::vector<float> px, py, pz, width, vertex;
};

OutputSnapshot Snapshot(UsdGenCurveBuffer const &buffer)
{
    OutputSnapshot result{{buffer.px.begin(), buffer.px.end()},
            {buffer.py.begin(), buffer.py.end()},
            {buffer.pz.begin(), buffer.pz.end()},
            {buffer.width.begin(), buffer.width.end()}, {}};
    for (auto const &field:buffer.extraCv)
        if (field.name==TfToken("failureVertex"))
            result.vertex.assign(field.f.begin(),field.f.end());
    return result;
}

bool Same(OutputSnapshot const &a, OutputSnapshot const &b)
{
    return a.px == b.px && a.py == b.py && a.pz == b.pz &&
           a.width == b.width && a.vertex == b.vertex;
}

bool Compile(UsdGenGraphDesc const &desc, UsdGenGraph *graph)
{
    UsdGenCompiler compiler;
    UsdGenCompileResult const result = compiler.Compile(desc, graph);
    if (!result.ok)
        for (std::string const &error : result.errors)
            std::printf("compile error: %s\n", error.c_str());
    return result.ok;
}

OutputSnapshot FreshOutput(UsdGenGraphDesc const &desc)
{
    UsdGenGraph graph;
    if (!Compile(desc, &graph)) return {};
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.time = 7.0;
    context.desc = &graph.Desc();
    g_failEnabled.store(false, std::memory_order_release);
    g_evaluations.store(0, std::memory_order_release);
    UsdGenRunResult const run = scheduler.Run(graph, context, 2);
    Check(!run.diagnostics.HasErrors(), "fresh parity graph succeeds");
    return Snapshot(OutputBuffer(graph));
}

void TestTransactionalFailure(bool unequalDepth)
{
    char const *shape = unequalDepth ? "unequal-depth" : "equal-depth";
    UsdGenGraphDesc desc = MakeGraph(unequalDepth);
    UsdGenGraph graph;
    Check(Compile(desc, &graph),
          unequalDepth ? "unequal-depth graph compiles" : "equal-depth graph compiles");
    if (graph.NodeCount() == 0) return;

    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.time = 7.0;
    context.desc = &graph.Desc();
    int callbacks = 0;
    OutputSnapshot published;
    auto completed = [&](UsdGenChunkGroup const &, UsdGenElementSet const &set) {
        ++callbacks;
        published = Snapshot(UsdGenCurveBufferFromSet(set));
    };

    g_failEnabled.store(false, std::memory_order_release);
    g_evaluations.store(0, std::memory_order_release);
    UsdGenRunResult run = scheduler.Run(graph, context, 1, completed);
    Check(!run.diagnostics.HasErrors(),
          unequalDepth ? "unequal-depth baseline succeeds" : "equal-depth baseline succeeds");
    Check(callbacks > 0,
          unequalDepth ? "unequal-depth baseline publishes" : "equal-depth baseline publishes");
    OutputSnapshot const baseline = published;
    Check(baseline.vertex.size()==desc.curveSets.front().points.size() &&
          std::all_of(baseline.vertex.begin(),baseline.vertex.end(),
                      [](float value) { return value==0.125f; }),
          "sibling remap writes preserve published vertex values");

    // Update the terminal through the ordinary value-edit/recompile path.
    // The failing run must not expose this 0.75 candidate through the
    // publication callback, even when the terminal finishes first.
    auto terminal = std::find_if(desc.nodes.begin(), desc.nodes.end(),
        [&](UsdGenNodeDesc const &node) { return node.path == desc.terminal; });
    Check(terminal != desc.nodes.end() && !terminal->params.empty(),
          unequalDepth ? "unequal-depth terminal value is editable"
                       : "equal-depth terminal value is editable");
    if (terminal == desc.nodes.end() || terminal->params.empty()) return;
    terminal->params.front().value = VtValue(0.75f);
    UsdGenCompiler compiler;
    UsdGenCompileResult const recompiled = compiler.Recompile(desc, &graph);
    Check(recompiled.ok,
          unequalDepth ? "unequal-depth value edit recompiles"
                       : "equal-depth value edit recompiles");
    if (!recompiled.ok) return;
    context.desc = &graph.Desc();

    graph.MarkNode(graph.NodeIdForPath(SdfPath("/cutFailure/failure")),
                   UsdGenDirtyParameter);
    callbacks = 0;
    g_evaluations.store(0, std::memory_order_release);
    g_failEnabled.store(true, std::memory_order_release);
    run = scheduler.Run(graph, context, 2, completed);
    Check(run.diagnostics.HasErrors(),
          unequalDepth ? "unequal-depth candidate fails" : "equal-depth candidate fails");
    Check(callbacks == 0,
          unequalDepth ? "unequal-depth failure publishes no tiles"
                       : "equal-depth failure publishes no tiles");
    Check(Same(baseline, published),
          unequalDepth ? "unequal-depth failure preserves prior publication"
                       : "equal-depth failure preserves prior publication");

    callbacks = 0;
    g_evaluations.store(0, std::memory_order_release);
    g_failEnabled.store(false, std::memory_order_release);
    run = scheduler.Run(graph, context, 2, completed);
    Check(!run.diagnostics.HasErrors(),
          unequalDepth ? "unequal-depth same-time retry succeeds"
                       : "equal-depth same-time retry succeeds");
    Check(callbacks > 0,
          unequalDepth ? "unequal-depth retry publishes after success"
                       : "equal-depth retry publishes after success");
    Check(!Same(published, baseline),
          unequalDepth ? "unequal-depth retry publishes edited terminal value"
                       : "equal-depth retry publishes edited terminal value");
    Check(Same(published, FreshOutput(desc)),
          unequalDepth ? "unequal-depth retry matches fresh cook"
                       : "equal-depth retry matches fresh cook");
    std::printf("completed %s transactional case\n", shape);
}

bool gRegisteredFallible = false;

void RegisterFallibleRemap(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenTestFallibleRemap";
    d.displayName = "TestFallibleRemap";
    d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
        return std::make_unique<FallibleRemap>(rec);
    };
    d.traits = kUsdGenOpRemapsVertexPlanes;
    gRegisteredFallible = r.RegisterOp(d) != UsdGenOpId::Invalid;
}

} // namespace

int main()
{
    UsdGenRegisterStatic(&RegisterFallibleRemap);
    Check(gRegisteredFallible, "registers fallible remapper");
    TestTransactionalFailure(false);
    TestTransactionalFailure(true);
    std::printf(g_failures ? "testUsdGenCutFailure: FAILED (%d)\n"
                           : "testUsdGenCutFailure: PASS\n",
                g_failures);
    return g_failures ? 1 : 0;
}
