// Backend-neutral CPU execution graph coverage for unary operator dataflow.
//
// A shared immutable Source value may feed several unary consumers.  Each
// consumer CoW-materializes only the planes it writes and retains read-only
// aliases for untouched planes, so evaluation order and authored node order
// cannot leak one sibling's writes into the other or into the terminal branch.

#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "bufferOpCalls.h"
#include "usdGenCurves/types/curveCapture.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/ops/grow.h"
#include "usdGenOpFixture.h"
#include "usdGen/scheduler.h"
#include "usdGenCurves/types/curveSet.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

int g_failures = 0;

class ExtraCvWriter final : public UsdGenOp
{
public:
    explicit ExtraCvWriter(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {7, 1}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override { return true; }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView const &chunk) const override
    {
        UsdGenCurveChunk const curve(chunk);
        UsdGenCurveChunk const *view = &curve;
        if (!view->outF || !view->inF || !view->outF[0] || !view->inF[0]) return;
        uint32_t count = view->curveCount * view->cvCount;
        if (view->cvCount == 0 && view->cvOffsets)
            count = uint32_t(view->cvOffsets[view->curveCount] - view->cvOffsets[0]);
        for (uint32_t i = 0; i < count; ++i) view->outF[0][i] = view->inF[0][i] + 10.0f;
    }
};

class ExtraCurveWriter final : public UsdGenOp
{
public:
    explicit ExtraCurveWriter(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {7, 2}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override { return true; }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView const &chunk) const override
    {
        UsdGenCurveChunk const curve(chunk);
        UsdGenCurveChunk const *view = &curve;
        if (!view->outI || !view->inI || !view->outI[0] || !view->inI[0]) return;
        for (uint32_t i = 0; i < view->curveCount; ++i)
            view->outI[0][i] = view->inI[0][i] + 20;
    }
};

struct HeldEvalGate
{
    std::atomic<int> entered{0};
    std::atomic<bool> release{false};
};

HeldEvalGate *g_heldEvalGate = nullptr;

class HeldWidthWriter final : public UsdGenOp
{
public:
    explicit HeldWidthWriter(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {9, 1}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &,
                 UsdGenCapture *, UsdGenDiagnostics *) override { return true; }
    uint32_t PlanesTouched() const override { return kPlaneWidths; }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView const &chunk) const override
    {
        UsdGenCurveChunk const curve(chunk);
        UsdGenCurveChunk const *view = &curve;
        HeldEvalGate *gate = g_heldEvalGate;
        if (gate) {
            gate->entered.fetch_add(1, std::memory_order_release);
            while (!gate->release.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
        if (!view || !view->width) return;
        uint32_t count = view->curveCount * view->cvCount;
        if (view->cvCount == 0 && view->cvOffsets)
            count = uint32_t(view->cvOffsets[view->curveCount] - view->cvOffsets[0]);
        for (uint32_t i = 0; i < count; ++i) view->width[i] = 2.0f;
    }
};

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

UsdGenNodeDesc SourceNode()
{
    UsdGenNodeDesc node;
    node.path = SdfPath("/fanout/source");
    node.type = TfToken("UsdGenCurveSource");
    node.surfaces = {SdfPath("/fanout/curves")};
    node.params.push_back({TfToken("resampleTo"), VtValue(4), false});
    return node;
}

UsdGenNodeDesc WidthNode(std::string const &name, float width)
{
    UsdGenNodeDesc node;
    node.path = SdfPath("/fanout/" + name);
    node.type = TfToken("UsdGenWidth");
    node.inputs = {SdfPath("/fanout/source")};
    node.params.push_back({TfToken("width"), VtValue(width), false});
    return node;
}

UsdGenGraphDesc MakeFanout(std::string const &terminalName,
                           std::string const &siblingName,
                           std::vector<int> const &authoredOrder)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/fanout");
    desc.terminal = SdfPath("/fanout/" + terminalName);

    UsdGenSurfaceDesc curves;
    curves.path = SdfPath("/fanout/curves");
    curves.id = 0;
    curves.faceVertexCounts = VtIntArray{4, 4};
    curves.faceVertexIndices = VtIntArray{0, 1, 2, 3, 4, 5, 6, 7};
    curves.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(0, 0, 1),
        GfVec3f(0, 0, 2), GfVec3f(0, 0, 3),
        GfVec3f(1, 0, 0), GfVec3f(1, 0, 1),
        GfVec3f(1, 0, 2), GfVec3f(1, 0, 3)};
    curves.points = curves.restPoints;
    desc.surfaces.push_back(std::move(curves));

    std::vector<UsdGenNodeDesc> nodes;
    nodes.push_back(SourceNode());
    nodes.push_back(WidthNode(terminalName, 0.25f));
    nodes.push_back(WidthNode(siblingName, 0.75f));
    for (int i : authoredOrder) desc.nodes.push_back(nodes.at(size_t(i)));
    return desc;
}

bool GeometryArity(char const *type, size_t *out)
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    UsdGenOpId const id = registry.Find(TfToken(type));
    if (id == UsdGenOpId::Invalid) return false;
    *out = registry.Record(id).Def().geometryInputArity;
    return true;
}

void RegisterFanoutTestOps(UsdGenRegistrar &r)
{
    static UsdGenPlaneDecl const kCv[] = {
        {"cvPayload", UsdGenPlaneRate::Vertex, UsdGenPlaneScalar::Float, 1}};
    static UsdGenPlaneDecl const kCurve[] = {
        {"curvePayload", UsdGenPlaneRate::Uniform, UsdGenPlaneScalar::Int, 1}};
    {
        UsdGenOpDef d;
        d.name = "UsdGenTestExtraCvWriter";
        d.displayName = "TestExtraCvWriter";
        d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
            return std::make_unique<ExtraCvWriter>(rec);
        };
        d.inputPlanes = UsdGenSpan(kCv);
        d.outputPlanes = UsdGenSpan(kCv);
        r.RegisterOp(d);
    }
    {
        UsdGenOpDef d;
        d.name = "UsdGenTestExtraCurveWriter";
        d.displayName = "TestExtraCurveWriter";
        d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
            return std::make_unique<ExtraCurveWriter>(rec);
        };
        d.inputPlanes = UsdGenSpan(kCurve);
        d.outputPlanes = UsdGenSpan(kCurve);
        r.RegisterOp(d);
    }
    {
        UsdGenOpDef d;
        d.name = "UsdGenTestHeldWidthWriter";
        d.displayName = "TestHeldWidthWriter";
        d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
            return std::make_unique<HeldWidthWriter>(rec);
        };
        r.RegisterOp(d);
    }
}

struct FanoutResult
{
    std::vector<float> terminal;
    std::vector<float> sibling;
    std::vector<float> source;
    bool branchStorageIsolated = false;
    bool unchangedPlanesShared = false;
    bool siblingRan = false;
};

FanoutResult RunFanout(UsdGenGraphDesc const &desc,
                       std::string const &siblingName)
{
    FanoutResult result;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "fan-out graph compiles");
    if (!compiled.ok) {
        for (auto const &error : compiled.errors)
            std::printf("  compile error: %s\n", error.c_str());
        return result;
    }

    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    Check(!run.diagnostics.HasErrors(), "fan-out graph executes without diagnostics");
    if (run.diagnostics.HasErrors()) {
        for (auto const &error : run.diagnostics.errors)
            std::printf("  run error: %s\n", error.c_str());
        return result;
    }

    auto const terminalView = OutputView(graph);
    UsdGenCurveBuffer const terminal = OutputBuffer(graph);
    auto const siblingView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/" + siblingName)));
    UsdGenCurveBuffer const sibling = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/" + siblingName)));
    auto const sourceView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/source")));
    UsdGenCurveBuffer const source = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/source")));
    result.terminal.assign(terminal.width.begin(), terminal.width.end());
    result.sibling.assign(sibling.width.begin(), sibling.width.end());
    result.source.assign(source.width.begin(), source.width.end());
    result.branchStorageIsolated =
        terminalView.width.cdata() != siblingView.width.cdata() &&
        terminalView.width.cdata() != sourceView.width.cdata() &&
        siblingView.width.cdata() != sourceView.width.cdata();
    // Width is the sole written plane. Points and generated hairT must retain
    // the source allocation in both branches; this catches a whole-buffer
    // clone masquerading as copy-on-write.
    result.unchangedPlanesShared =
        !source.px.empty() && !source.py.empty() && !source.pz.empty() &&
        !source.hairT.empty() &&
        terminalView.px.cdata() == sourceView.px.cdata() &&
        terminalView.py.cdata() == sourceView.py.cdata() &&
        terminalView.pz.cdata() == sourceView.pz.cdata() &&
        terminalView.hairT.cdata() == sourceView.hairT.cdata() &&
        siblingView.px.cdata() == sourceView.px.cdata() &&
        siblingView.py.cdata() == sourceView.py.cdata() &&
        siblingView.pz.cdata() == sourceView.pz.cdata() &&
        siblingView.hairT.cdata() == sourceView.hairT.cdata();
    result.siblingRan = std::all_of(result.sibling.begin(), result.sibling.end(),
                                    [](float value) { return value == 0.75f; });
    return result;
}

bool AllEqual(std::vector<float> const &values, float expected)
{
    return !values.empty() &&
        std::all_of(values.begin(), values.end(),
                    [expected](float value) { return value == expected; });
}

void InstallExtraPlanes(UsdGenCurveBuffer *buffer)
{
    UsdGenPlane cv;
    cv.name = TfToken("cvPayload");
    cv.interpolation = TfToken("vertex");
    cv.type = TfToken("float");
    cv.f.resize(buffer->totalCvs);
    for (size_t i = 0; i < cv.f.size(); ++i) cv.f[i] = float(i);
    UsdGenPlane curve;
    curve.name = TfToken("curvePayload");
    curve.interpolation = TfToken("uniform");
    curve.type = TfToken("int");
    curve.i.resize(buffer->totalCurves);
    for (size_t i = 0; i < curve.i.size(); ++i) curve.i[i] = int(i);
    buffer->extraCv = {std::move(cv)};
    buffer->extraCurve = {std::move(curve)};
}

UsdGenCurveBuffer MakeTopologyPlaneSource()
{
    UsdGenCurveBuffer source;
    source.totalCurves = 3;
    source.totalCvs = 9;
    source.cvOffsets = VtIntArray{0, 2, 5, 9};

    UsdGenPlane cvFloat;
    cvFloat.name = TfToken("cvFloat");
    cvFloat.interpolation = TfToken("vertex");
    cvFloat.type = TfToken("float");
    cvFloat.arity = 2;
    cvFloat.f.resize(source.totalCvs * cvFloat.arity);
    UsdGenPlane cvIndex;
    cvIndex.name = TfToken("cvIndex");
    cvIndex.interpolation = TfToken("vertex");
    cvIndex.type = TfToken("int");
    cvIndex.i.resize(source.totalCvs);
    for (uint32_t i = 0; i != source.totalCvs; ++i) {
        cvFloat.f[i * 2] = float(i);
        cvFloat.f[i * 2 + 1] = 100.0f + float(i);
        cvIndex.i[i] = int(i);
    }

    UsdGenPlane constant;
    constant.name = TfToken("constantPayload");
    constant.interpolation = TfToken("constant");
    constant.type = TfToken("float");
    constant.arity = 2;
    constant.f = VtFloatArray{7.0f, 11.0f};
    UsdGenPlane curve;
    curve.name = TfToken("curvePayload");
    curve.interpolation = TfToken("uniform");
    curve.type = TfToken("int");
    curve.arity = 3;
    curve.i = VtIntArray{10, 11, 12, 20, 21, 22, 30, 31, 32};
    source.extraCv = {std::move(cvFloat), std::move(cvIndex)};
    source.extraCurve = {std::move(constant), std::move(curve)};
    return source;
}

UsdGenPlane MakeStalePlane()
{
    UsdGenPlane stale;
    stale.name = TfToken("stale");
    stale.interpolation = TfToken("vertex");
    stale.type = TfToken("float");
    stale.f = VtFloatArray{99.0f};
    return stale;
}

void TestTopologyExtraPlaneTransforms()
{
    UsdGenCurveBuffer const source = MakeTopologyPlaneSource();
    UsdGenCurveBuffer resampled;
    resampled.totalCurves = 3;
    resampled.totalCvs = 12;  // uniform four-CV target
    resampled.extraCv = {MakeStalePlane()};
    std::string error;
    Check(UsdGenResampleExtraPlanes(source, &resampled, &error),
          "topology resample transforms named extra planes");
    Check(resampled.extraCv.size() == 2 && resampled.extraCurve.size() == 2 &&
              resampled.extraCv[0].name == TfToken("cvFloat") &&
              resampled.extraCv[0].interpolation == TfToken("vertex") &&
              resampled.extraCv[0].type == TfToken("float") &&
              resampled.extraCv[0].arity == 2 &&
              resampled.extraCv[1].name == TfToken("cvIndex") &&
              resampled.extraCv[1].type == TfToken("int") &&
              resampled.extraCurve[0].interpolation == TfToken("constant") &&
              resampled.extraCurve[1].interpolation == TfToken("uniform") &&
              resampled.extraCv[0].f.cdata() != source.extraCv[0].f.cdata() &&
              resampled.extraCv[1].i.cdata() != source.extraCv[1].i.cdata() &&
              resampled.extraCurve[0].f.cdata() != source.extraCurve[0].f.cdata() &&
              resampled.extraCurve[1].i.cdata() != source.extraCurve[1].i.cdata(),
          "resample preserves plane domain/type/arity in private COW outputs");
    Check(resampled.extraCv[0].f.size() == 24 &&
              resampled.extraCv[0].f[2] == 1.0f / 3.0f &&
              resampled.extraCv[0].f[3] == 100.0f + 1.0f / 3.0f &&
              resampled.extraCv[1].i == VtIntArray{0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8} &&
              resampled.extraCurve[0].f == VtFloatArray{7.0f, 11.0f} &&
              resampled.extraCurve[1].i == source.extraCurve[1].i,
          "resample linearly maps float vertices, nearest-maps int vertices, and preserves curve planes");
    resampled.extraCv[0].f[0] = -5.0f;
    Check(source.extraCv[0].f[0] == 0.0f,
          "resample output writes cannot mutate immutable fan-out source planes");

    UsdGenCurveBuffer compacted;
    compacted.totalCurves = 2;
    compacted.totalCvs = 6;
    compacted.cvOffsets = VtIntArray{0, 2, 6};
    compacted.extraCv = {MakeStalePlane()};
    Check(UsdGenCompactExtraPlanes(source, {0, 2}, &compacted, &error),
          "topology compaction transforms named extra planes");
    Check(compacted.extraCv.size() == 2 && compacted.extraCurve.size() == 2 &&
              compacted.extraCv[0].f == VtFloatArray{
                  0.0f, 100.0f, 1.0f, 101.0f,
                  5.0f, 105.0f, 6.0f, 106.0f,
                  7.0f, 107.0f, 8.0f, 108.0f} &&
              compacted.extraCv[1].i == VtIntArray{0, 1, 5, 6, 7, 8} &&
              compacted.extraCurve[0].f == VtFloatArray{7.0f, 11.0f} &&
              compacted.extraCurve[1].i == VtIntArray{10, 11, 12, 30, 31, 32} &&
              compacted.extraCv[0].f.cdata() != source.extraCv[0].f.cdata() &&
              compacted.extraCurve[1].i.cdata() != source.extraCurve[1].i.cdata(),
          "compaction follows stable survivor CV/curve mapping with private storage");
    compacted.extraCurve[1].i[0] = -1;
    Check(source.extraCurve[1].i[0] == 10,
          "compaction output writes cannot mutate immutable fan-out source planes");

    UsdGenCurveBuffer invalidTarget;
    invalidTarget.totalCurves = 3;
    invalidTarget.totalCvs = 12;
    invalidTarget.extraCv = {MakeStalePlane()};
    UsdGenCurveBuffer malformed = source;
    malformed.extraCv[0].interpolation = TfToken("uniform");
    Check(!UsdGenResampleExtraPlanes(malformed, &invalidTarget, &error) &&
              invalidTarget.extraCv.size() == 1 &&
              invalidTarget.extraCv[0].name == TfToken("stale"),
          "unsupported topology-plane interpolation fails closed without stale-output replacement");
    malformed = source;
    malformed.extraCv[0].type = TfToken("half");
    Check(!UsdGenResampleExtraPlanes(malformed, &invalidTarget, &error) &&
              invalidTarget.extraCv.size() == 1 &&
              invalidTarget.extraCv[0].name == TfToken("stale"),
          "unsupported topology-plane type fails closed without coercion");
}

void InstallGrowExtraPlanes(UsdGenCurveBuffer *buffer)
{
    InstallExtraPlanes(buffer);
    UsdGenPlane index;
    index.name = TfToken("cvIndex");
    index.interpolation = TfToken("vertex");
    index.type = TfToken("int");
    index.i.resize(buffer->totalCvs);
    for (size_t i = 0; i != index.i.size(); ++i) index.i[i] = int(i);
    buffer->extraCv.insert(buffer->extraCv.begin(), std::move(index));
    UsdGenPlane constant;
    constant.name = TfToken("constantPayload");
    constant.interpolation = TfToken("constant");
    constant.type = TfToken("float");
    constant.f = VtFloatArray{3.5f};
    buffer->extraCurve.insert(buffer->extraCurve.begin(), std::move(constant));
}

void TestGrowNamedPlaneTopologyOwnership()
{
    UsdGenGraphDesc desc = MakeFanout("unused", "alsoUnused", {0, 1, 2});
    desc.terminal = SdfPath("/fanout/grow");
    UsdGenNodeDesc source = SourceNode();
    UsdGenNodeDesc grow;
    grow.path = desc.terminal;
    grow.type = TfToken("UsdGenGrow");
    grow.inputs = {source.path};
    grow.params.push_back({TfToken("segments"), VtValue(3), false});
    grow.params.push_back({TfToken("length"), VtValue(1.0), false});
    desc.nodes = {source, grow};

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok,
          "Grow named-plane topology fixture compiles");
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(),
          "Grow named-plane topology fixture establishes source output");

    UsdGenNodeId const sourceId = graph.NodeIdForPath(source.path);
    UsdGenNodeId const growId = graph.NodeIdForPath(grow.path);
    {
        UsdGenCurveBuffer planes = NodeBuffer(graph, sourceId);
        InstallGrowExtraPlanes(&planes);
        Check(SetNodeBuffer(graph, sourceId, planes), "Grow named-plane fixture installs");
    }
    graph.MarkNode(growId, UsdGenDirtyCapture);
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "Grow captures transformed named planes without diagnostics");

    auto const inputView = NodeView(graph, sourceId);
    UsdGenCurveBuffer const input = NodeBuffer(graph, sourceId);
    auto const outputView = NodeView(graph, growId);
    UsdGenCurveBuffer const output = NodeBuffer(graph, growId);
    Check(input.totalCvs == 8 && output.totalCvs == 6 &&
              output.extraCv.size() == 2 && output.extraCurve.size() == 2 &&
              output.extraCv[0].name == TfToken("cvIndex") &&
              output.extraCv[0].i == VtIntArray{0, 2, 3, 4, 6, 7} &&
              output.extraCv[1].name == TfToken("cvPayload") &&
              output.extraCv[1].f == VtFloatArray{0.0f, 1.5f, 3.0f,
                                                   4.0f, 5.5f, 7.0f} &&
              output.extraCurve[0].name == TfToken("constantPayload") &&
              output.extraCurve[0].f == VtFloatArray{3.5f} &&
              output.extraCurve[1].name == TfToken("curvePayload") &&
              output.extraCurve[1].i == VtIntArray{0, 1},
          "Grow resamples vertex and copies uniform/constant named planes");
    Check(outputView.ExtraCv(0).i.cdata() != inputView.ExtraCv(0).i.cdata() &&
              outputView.ExtraCv(1).f.cdata() != inputView.ExtraCv(1).f.cdata() &&
              outputView.ExtraCurve(0).f.cdata() != inputView.ExtraCurve(0).f.cdata() &&
              outputView.ExtraCurve(1).i.cdata() != inputView.ExtraCurve(1).i.cdata(),
          "Grow owns private COW named-plane outputs after topology change");
    const_cast<float *>(NodeView(graph, growId).ExtraCv(1).f.data())[0] = -9.0f;
    Check(NodeView(graph, sourceId).ExtraCv(1).f[0] == 0.0f,
          "Grow named-plane output mutation cannot alter the upstream COW owner");

    // A value-only upstream width update must re-capture Grow's transformed
    // width owner even though the root/CV topology did not move.
    {
        UsdGenCurveBuffer widths = NodeBuffer(graph, sourceId);
        widths.width = VtFloatArray{
            0.0f, 0.25f, 0.75f, 1.0f,
            1.0f, 0.75f, 0.25f, 0.0f};
        widths.valueVersion += 1;
        Check(SetNodeBuffer(graph, sourceId, widths), "upstream width edit installs");
    }
    graph.MarkNode(growId, UsdGenDirtyParameter);
    Check(!scheduler.Run(graph, context, 3).diagnostics.HasErrors(),
          "Grow re-captures its width owner after an upstream value edit");
    auto const widthRefreshedView = NodeView(graph, growId);
    UsdGenCurveBuffer const widthRefreshed = NodeBuffer(graph, growId);
    Check(widthRefreshed.width == VtFloatArray{
              0.0f, 0.5f, 1.0f, 1.0f, 0.5f, 0.0f} &&
              widthRefreshedView.width.cdata() != NodeView(graph, sourceId).width.cdata(),
          "Grow resamples an upstream width plane into a private COW owner");
    const_cast<float *>(NodeView(graph, growId).width.data())[1] = -7.0f;
    Check(NodeView(graph, sourceId).width[1] == 0.25f,
          "Grow width output writes cannot mutate the upstream width owner");
}

void TestGrowWidthFallbackCapture()
{
    UsdGenGraphDesc desc;
    desc.defaultWidth = 0.025f;
    UsdGenNodeDesc node;
    node.path = SdfPath("/growWidth/grow");
    node.type = TfToken("UsdGenGrow");
    node.params = {
        {TfToken("segments"), VtValue(3), false},
        {TfToken("length"), VtValue(1.0), false}};
    usdGenTest::OpView paramsView(usdGenTest::OpRecord("UsdGenGrow"), &node);
    UsdGenParamView const &params = paramsView.View();
    UsdGenGrowOp grow(usdGenTest::OpRecord("UsdGenGrow"));
    UsdGenDiagnostics diagnostics;
    Check(grow.Bind(params, &diagnostics),
          "Grow width fallback fixture binds");

    UsdGenCurveBuffer roots;
    roots.totalCurves = 2;
    roots.totalCvs = 2;
    roots.curveId = VtArray<uint64_t>{11, 12};
    roots.px = roots.py = roots.pz = VtFloatArray(2, 0.0f);
    UsdGenCaptureContext context;
    context.desc = &desc;
    context.params = &params;

    auto fallbackCapture = grow.CreateCapture();
    Check(grow.Capture(context, UsdGenBufferInputs(roots), fallbackCapture.get(), &diagnostics) &&
              UsdGenCaptureBuffer(*fallbackCapture).width == VtFloatArray(6, 0.025f),
          "Grow materializes desc.defaultWidth for roots without widths");

    roots.totalCvs = 8;
    roots.px = roots.py = roots.pz = VtFloatArray(8, 0.0f);
    roots.width = VtFloatArray{
        0.0f, 0.25f, 0.75f, 1.0f,
        1.0f, 0.75f, 0.25f, 0.0f};
    auto inheritedCapture = grow.CreateCapture();
    Check(grow.Capture(context, UsdGenBufferInputs(roots), inheritedCapture.get(), &diagnostics),
          "Grow captures an inherited width plane");
    UsdGenCurveBuffer &grown = UsdGenCaptureMutableBuffer(*inheritedCapture);
    Check(grown.width == VtFloatArray{0.0f, 0.5f, 1.0f, 1.0f, 0.5f, 0.0f} &&
              grown.width.cdata() != roots.width.cdata(),
          "Grow resamples inherited widths to a fresh topology owner");
    grown.width[0] = -3.0f;
    Check(roots.width[0] == 0.0f,
          "Grow's captured width owner remains COW-isolated from its input");
}

void TestGrowAngularLift()
{
    UsdGenGrowOp grow(usdGenTest::OpRecord("UsdGenGrow"));
    UsdGenNodeDesc node;
    node.path = SdfPath("/growLift/grow");
    node.type = TfToken("UsdGenGrow");
    node.params = {
        {TfToken("segments"), VtValue(3), false},
        {TfToken("length"), VtValue(1.0), false},
        {TfToken("direction"), VtValue(TfToken("surfaceNormal")), false},
        {TfToken("lift"), VtValue(90.0), false}};
    usdGenTest::OpView paramsView(usdGenTest::OpRecord("UsdGenGrow"), &node);
    UsdGenParamView const &params = paramsView.View();
    UsdGenDiagnostics diagnostics;
    Check(grow.Bind(params, &diagnostics),
          "Grow accepts lift in degrees in the documented range");

    UsdGenCurveBuffer roots;
    roots.totalCurves = 1;
    roots.totalCvs = 2;
    roots.curveId = VtArray<uint64_t>{7};
    roots.px = {10.0f, 10.0f};
    roots.py = {20.0f, 20.0f};
    roots.pz = {30.0f, 31.0f};
    roots.rest = {GfVec3f(0, 0, 0), GfVec3f(0, 0, 1)};
    roots.rootT = {GfVec3f(0, 1, 0)};
    roots.rootB = {GfVec3f(0, 0, 1)};
    roots.rootN = {GfVec3f(1, 0, 0)};
    UsdGenCaptureContext context;
    context.params = &params;
    context.desc = nullptr;
    auto capture = grow.CreateCapture();
    Check(grow.Capture(context, UsdGenBufferInputs(roots), capture.get(), &diagnostics),
          "Grow captures angular lift without mutating input");
    auto const &lifted = UsdGenCaptureBuffer(*capture).rest;
    Check(lifted.size() == 3 && lifted[0] == GfVec3f(0, 0, 0) &&
              std::fabs(lifted[1][0]) < 1.e-5f &&
              std::fabs(lifted[1][1] - 0.5f) < 1.e-5f &&
              std::fabs(lifted[1][2]) < 1.e-5f &&
              std::fabs(lifted[2][0]) < 1.e-5f &&
              std::fabs(lifted[2][1] - 1.0f) < 1.e-5f &&
              std::fabs(lifted[2][2]) < 1.e-5f &&
              roots.rest[1] == GfVec3f(0, 0, 1),
          "Grow rotates surfaceNormal about root-frame B");

    node.params.back().value = VtValue(-90.0);
    usdGenTest::OpView negativeView(usdGenTest::OpRecord("UsdGenGrow"), &node);
    UsdGenParamView const &negativeParams = negativeView.View();
    context.params = &negativeParams;
    auto negativeCapture = grow.CreateCapture();
    Check(grow.Capture(context, UsdGenBufferInputs(roots), negativeCapture.get(), &diagnostics),
          "Grow captures negative angular lift");
    auto const &negative = UsdGenCaptureBuffer(*negativeCapture).rest;
    Check(negative.size() == 3 && std::fabs(negative[1][1] + 0.5f) < 1.e-5f &&
              std::fabs(negative[2][1] + 1.0f) < 1.e-5f &&
              std::fabs(negative[2][0]) < 1.e-5f &&
              std::fabs(negative[2][2]) < 1.e-5f,
          "Grow rotates in the opposite direction for negative lift");

    // Exercise the value phase separately: posed roots are read from the
    // current input planes, while the capture above used immutable rest roots.
    node.params.back().value = VtValue(90.0);
    usdGenTest::OpView evaluateView(usdGenTest::OpRecord("UsdGenGrow"), &node);
    UsdGenParamView const &evaluateParams = evaluateView.View();
    context.params = &evaluateParams;
    auto evaluateCapture = grow.CreateCapture();
    Check(grow.Capture(context, UsdGenBufferInputs(roots), evaluateCapture.get(), &diagnostics),
          "Grow prepares angular value evaluation");
    UsdGenChunkDesc chunk;
    chunk.curveCount = 1;
    chunk.liveCount = 1;
    chunk.cvCount = 3;
    float px[3] = {}, py[3] = {}, pz[3] = {}, hairT[3] = {};
    UsdGenTestChunk view;
    view.desc = &chunk;
    view.px = px; view.py = py; view.pz = pz; view.hairT = hairT;
    view.inPx = roots.px.cdata(); view.inPy = roots.py.cdata();
    view.inPz = roots.pz.cdata(); view.curveId = roots.curveId.cdata();
    view.rootT = roots.rootT.cdata(); view.rootB = roots.rootB.cdata();
    view.rootN = roots.rootN.cdata(); view.curveCount = 1; view.cvCount = 3;
    view.inCvCount = 2;
    UsdGenEvalContext evalContext;
    evalContext.params = &evaluateParams;
    grow.Evaluate(evalContext, *evaluateCapture, view.View());
    Check(std::fabs(px[0] - 10.0f) < 1.e-5f &&
              std::fabs(px[1] - 10.0f) < 1.e-5f &&
              std::fabs(px[2] - 10.0f) < 1.e-5f &&
              std::fabs(py[2] - 21.0f) < 1.e-5f &&
              std::fabs(pz[2] - 30.0f) < 1.e-5f &&
              hairT[0] == 0.0f && hairT[2] == 1.0f,
          "Grow Evaluate rotates posed points about root-frame B");

    UsdGenCaptureContext digestContext;
    digestContext.params = &evaluateParams;
    auto const liftedDigest = grow.CaptureDigest(digestContext);
    auto changed = node;
    changed.params.back().value = VtValue(-90.0);
    digestContext.params = nullptr;
    usdGenTest::OpView changedView(usdGenTest::OpRecord("UsdGenGrow"), &changed);
    UsdGenParamView const &changedParams = changedView.View();
    digestContext.params = &changedParams;
    Check(liftedDigest[1] != grow.CaptureDigest(digestContext)[1],
          "Grow capture digest includes angular lift");
    changed.params.back().value = VtValue(90.0);
    changed.params.push_back({TfToken("directionVector"),
                              VtValue(GfVec3f(1, 0, 0)), false});
    changedView.Set(&changed);   // the parameter vector reallocated
    digestContext.params = &changedParams;
    auto const vectorDigest = grow.CaptureDigest(digestContext);
    Check(vectorDigest[1] != liftedDigest[1],
          "Grow capture digest includes directionVector");

    auto invalid = node;
    invalid.params.back().value = VtValue(90.1);
    usdGenTest::OpView invalidView(usdGenTest::OpRecord("UsdGenGrow"), &invalid);
    UsdGenParamView const &invalidParams = invalidView.View();
    diagnostics = {};
    Check(!grow.Bind(invalidParams, &diagnostics) && diagnostics.HasErrors(),
          "Grow rejects lift outside [-90,90] before capture");
    invalid.params.back().value =
        VtValue(std::numeric_limits<double>::infinity());
    diagnostics = {};
    Check(!grow.Bind(invalidParams, &diagnostics) && diagnostics.HasErrors(),
          "Grow rejects non-finite lift before capture");
}

UsdGenGraphDesc MakeScatterValidationDesc(double density)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scatterValidation");
    desc.terminal = SdfPath("/scatterValidation/scatter");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/scatterValidation/surface");
    surface.id = 0;
    surface.faceVertexCounts = VtIntArray{3};
    surface.faceVertexIndices = VtIntArray{0, 1, 2};
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
    surface.points = surface.restPoints;
    desc.surfaces = {surface};
    UsdGenNodeDesc scatter;
    scatter.path = desc.terminal;
    scatter.type = TfToken("UsdGenScatter");
    scatter.surfaces = {surface.path};
    scatter.params.push_back({TfToken("density"), VtValue(density), false});
    desc.nodes = {scatter};
    return desc;
}

void TestGrowAzimuth()
{
    UsdGenGrowOp grow(usdGenTest::OpRecord("UsdGenGrow"));
    UsdGenNodeDesc node;
    node.params = {{TfToken("segments"), VtValue(2), false},
                   {TfToken("length"), VtValue(1.0f), false},
                   {TfToken("lift"), VtValue(42.0f), false},
                   {TfToken("azimuth"), VtValue(0.0f), false},
                   {TfToken("azimuthRandom"), VtValue(0.0f), false}};
    usdGenTest::OpView paramsView(usdGenTest::OpRecord("UsdGenGrow"), &node);
    UsdGenParamView &params = paramsView.View();
    UsdGenCaptureContext context; context.params = &params; context.seed = 21;
    UsdGenDiagnostics diag;
    UsdGenCurveBuffer roots;
    constexpr uint32_t count = 2048;
    roots.totalCurves = roots.totalCvs = count;
    roots.px.assign(count, 0); roots.py.assign(count, 0); roots.pz.assign(count, 0);
    roots.curveId.resize(count);
    roots.rootT.assign(count, GfVec3f(1,0,0));
    roots.rootB.assign(count, GfVec3f(0,1,0));
    roots.rootN.assign(count, GfVec3f(0,0,1));
    for (uint32_t c=0;c<count;++c) roots.curveId[c] = c+100;
    auto capture = [&]() {
        paramsView.Set(&node);   // the node's parameters change between captures
        auto cap = grow.CreateCapture();
        Check(grow.Capture(context, UsdGenBufferInputs(roots), cap.get(), &diag), "Grow azimuth capture succeeds");
        return cap;
    };
    auto zero = capture();
    // Explicit zero must retain the exact existing direction path.
    node.params.erase(node.params.begin()+3,node.params.end());
    auto absent = capture();
    Check(UsdGenCaptureBuffer(*zero).rest == UsdGenCaptureBuffer(*absent).rest,
          "Grow zero azimuth controls preserve existing points bit-for-bit");
    node.params.push_back({TfToken("azimuth"), VtValue(90.0f), false});
    node.params.push_back({TfToken("azimuthRandom"), VtValue(0.0f), false});
    auto fixed = capture();
    auto const endpoint = UsdGenCaptureBuffer(*fixed).rest[1];
    Check(std::fabs(endpoint[0]) < 1e-5f &&
          std::fabs(endpoint[1]-std::sin(42.f*3.14159265358979323846f/180.f)) < 1e-5f &&
          std::fabs(endpoint[2]-UsdGenCaptureBuffer(*zero).rest[1][2]) < 1e-5f,
          "Grow azimuth turns the leaned direction around N, preserving inclination");
    node.params[3].value = VtValue(0.0f);
    node.params[4].value = VtValue(1.0f);
    auto random = capture();
    auto repeat = capture();
    Check(UsdGenCaptureBuffer(*random).rest == UsdGenCaptureBuffer(*repeat).rest, "Grow azimuth repeatability");
    std::swap(roots.curveId[0], roots.curveId[1]);
    auto reordered = capture();
    Check(UsdGenCaptureBuffer(*reordered).rest[1] == UsdGenCaptureBuffer(*random).rest[3] &&
          UsdGenCaptureBuffer(*reordered).rest[3] == UsdGenCaptureBuffer(*random).rest[1],
          "Grow random azimuth follows stable IDs when strand order changes");
    std::swap(roots.curveId[0], roots.curveId[1]);
    GfVec3f mean(0);
    int quadrants[4] = {};
    bool shape = true;
    for(uint32_t c=0;c<count;++c) {
        auto const v = UsdGenCaptureBuffer(*random).rest[c*2+1];
        mean += v;
        ++quadrants[(v[0]<0 ? 1:0)+(v[1]<0 ? 2:0)];
        shape &= std::fabs(v.GetLength()-1.f)<1e-5f &&
                 std::fabs(v[2]-UsdGenCaptureBuffer(*zero).rest[1][2])<1e-5f &&
                 UsdGenCaptureBuffer(*random).rest[c*2] == GfVec3f(0);
    }
    mean /= float(count);
    Check(shape && std::fabs(mean[0])<.04f && std::fabs(mean[1])<.04f &&
          *std::min_element(quadrants,quadrants+4)>400,
          "Random azimuth covers the full circle without changing roots, lengths or lean");
    // Regression for the puppet's visible quad pattern: four neighbouring
    // parameter charts turn T/B by 90 degrees, while N stays smooth.
    // No chart should receive one common azimuth or a preferred direction.
    roots.rootPrim.resize(count);
    GfVec3f const tangents[4] = {GfVec3f(1,0,0),GfVec3f(0,1,0),GfVec3f(-1,0,0),GfVec3f(0,-1,0)};
    GfVec3f const binormals[4] = {GfVec3f(0,1,0),GfVec3f(-1,0,0),GfVec3f(0,-1,0),GfVec3f(1,0,0)};
    for(uint32_t c=0;c<count;++c) {
        uint32_t const face=c/(count/4);
        roots.rootPrim[c]=int(face);
        roots.rootT[c]=tangents[face]; roots.rootB[c]=binormals[face];
    }
    auto charts=capture();
    bool isotropicCharts=true;
    for(uint32_t face=0;face<4;++face) {
        GfVec3f chartMean(0);
        int bins[4]={};
        for(uint32_t c=face*(count/4);c<(face+1)*(count/4);++c) {
            GfVec3f const v=UsdGenCaptureBuffer(*charts).rest[c*2+1];
            chartMean+=v;
            ++bins[(v[0]<0 ? 1:0)+(v[1]<0 ? 2:0)];
        }
        chartMean/=float(count/4);
        isotropicCharts &= std::fabs(chartMean[0])<.09f && std::fabs(chartMean[1])<.09f &&
            *std::min_element(bins,bins+4)>75;
    }
    Check(isotropicCharts, "Full random azimuth has no shared face direction across rotated quad charts");
    for(uint32_t c=0;c<count;++c) roots.rootPrim[c]+=73;
    auto renamedFaces=capture();
    Check(UsdGenCaptureBuffer(*renamedFaces).rest == UsdGenCaptureBuffer(*charts).rest,
          "Grow direction does not use face ID as a random input");
    roots.rootPrim.clear();
    roots.rootT.assign(count,GfVec3f(1,0,0)); roots.rootB.assign(count,GfVec3f(0,1,0));
    // Evaluate as two nonzero-offset chunks; captured randomness must not
    // depend on how the scheduler partitions the strands.
    bool equal = true;
    for(uint32_t begin : {0u, count/2}) {
        UsdGenChunkDesc chunk; chunk.firstCurve=begin; chunk.curveCount=count/2;
        chunk.liveCount=count/2; chunk.cvCount=2;
        std::vector<float> x(count),y(count),z(count);
        UsdGenTestChunk view; view.desc=&chunk;
        view.curveCount=count/2; view.cvCount=2; view.inCvCount=1;
        view.inPx=roots.px.cdata()+begin; view.inPy=roots.py.cdata()+begin; view.inPz=roots.pz.cdata()+begin;
        view.rootN=roots.rootN.cdata()+begin; view.rootB=roots.rootB.cdata()+begin;
        view.px=x.data(); view.py=y.data(); view.pz=z.data();
        UsdGenEvalContext eval; eval.params=&params;
        grow.Evaluate(eval, *random, view.View());
        for(uint32_t i=0;i<count;++i)
            equal &= GfVec3f(x[i],y[i],z[i]) == UsdGenCaptureBuffer(*random).rest[begin*2+i];
    }
    Check(equal, "Grow posed/rest azimuth agrees across chunk boundaries");
    auto const digest = grow.CaptureDigest(context);
    context.seed = 22;
    auto reseeded = capture();
    Check(UsdGenCaptureBuffer(*reseeded).rest != UsdGenCaptureBuffer(*random).rest,
          "Grow seed changes azimuth without a shared global RNG");
    context.seed = 21;
    node.params[4].value = VtValue(.5f);
    Check(grow.CaptureDigest(context)[0] != digest[0], "Grow azimuthRandom invalidates capture");
    node.params[4].value = VtValue(1.0f);
    node.params[3].value = VtValue(15.0f);
    Check(grow.CaptureDigest(context)[0] != digest[0], "Grow azimuth invalidates capture");
    // Exercise actual primitive-domain expression evaluation, not just
    // connection admission. Each root gets its own angle; changing the
    // connected value must reach Capture and validate per strand.
    node.path=SdfPath("/groom/grow"); node.type=TfToken("UsdGenGrow");
    UsdGenGraphDesc desc; desc.description=SdfPath("/groom");
    expr::ValueShape const scalarShape{expr::ScalarType::Float32,1,1,1,1,false};
    desc.expressions.push_back({SdfPath("/groom/azimuth"),"90 * ($index % 2)",
        {{TfToken("result"),TfToken("float"),scalarShape}}});
    node.expressionBindings.push_back({SdfPath("/groom/azimuth"),TfToken("result"),TfToken("float"),
        TfToken("azimuth"),scalarShape,expr::Domain::Primitive,VtValue(0.f)});
    node.params[3].value=VtValue(0.f); node.params[4].value=VtValue(0.f);
    UsdGenCpuParameters expressions; bool changed=false;
    std::vector<std::string> expressionErrors;
    Check(expressions.Evaluate(desc,node,UsdGenCurveSetView::Of(roots),0,0,21,&changed,&expressionErrors),
          "Grow primitive azimuth expression evaluates");
    params.expressions=&expressions;
    auto connected=capture();
    Check(UsdGenCaptureBuffer(*connected).rest[1] == UsdGenCaptureBuffer(*zero).rest[1] &&
          (UsdGenCaptureBuffer(*connected).rest[3]-UsdGenCaptureBuffer(*fixed).rest[1]).GetLength()<1e-5f,
          "Grow consumes a distinct connected azimuth at each root");
    node.expressionBindings[0].destination=TfToken("azimuthRandom");
    desc.expressions[0].source="$index % 2";
    Check(expressions.Evaluate(desc,node,UsdGenCurveSetView::Of(roots),0,0,21,&changed,&expressionErrors),
          "Grow primitive azimuthRandom expression evaluates");
    auto connectedRandom=capture();
    Check(UsdGenCaptureBuffer(*connectedRandom).rest[1] == UsdGenCaptureBuffer(*zero).rest[1] &&
          UsdGenCaptureBuffer(*connectedRandom).rest[3] == UsdGenCaptureBuffer(*random).rest[3],
          "Grow consumes connected azimuthRandom independently per root");
    desc.expressions[0].source="1.1";
    Check(expressions.Evaluate(desc,node,UsdGenCurveSetView::Of(roots),0,0,21,&changed,&expressionErrors),
          "Out-of-range expression is evaluated before Grow validates it");
    auto invalidConnected=grow.CreateCapture();
    Check(!grow.Capture(context,UsdGenBufferInputs(roots),invalidConnected.get(),&diag),
          "Grow rejects out-of-range connected azimuthRandom");
    params.expressions=nullptr;
    for (int index : {3,4}) for(float invalid : {-361.f, 361.f, std::numeric_limits<float>::quiet_NaN()}) {
        node.params[3].value=VtValue(0.f); node.params[4].value=VtValue(0.f);
        node.params[index].value=VtValue(invalid); diag={};
        auto rejected=grow.CreateCapture();
        Check(!grow.Bind(params,&diag) && !grow.Capture(context,UsdGenBufferInputs(roots),rejected.get(),&diag),
              "Grow rejects invalid azimuth controls in Bind and Capture");
    }
}

void TestScatterGrowDefaultWidthThroughScheduler()
{
    UsdGenGraphDesc desc = MakeScatterValidationDesc(300.0);
    desc.defaultWidth = 0.025f;
    SdfPath const scatterPath = desc.terminal;
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/scatterValidation/grow");
    grow.type = TfToken("UsdGenGrow");
    grow.inputs = {scatterPath};
    grow.params = {
        {TfToken("segments"), VtValue(5), false},
        {TfToken("length"), VtValue(1.0), false}};
    desc.nodes.push_back(grow);
    desc.terminal = grow.path;

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok,
          "Scatter->Grow default-width scheduler fixture compiles");
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    Check(!run.diagnostics.HasErrors(),
          "Scatter->Grow default-width scheduler fixture executes");
    UsdGenCurveBuffer const scatter = NodeBuffer(graph, graph.NodeIdForPath(scatterPath));
    UsdGenCurveBuffer const output = OutputBuffer(graph);
    Check(scatter.width.empty() && output.totalCvs > 0 &&
              output.width == VtFloatArray(output.totalCvs, desc.defaultWidth),
          "Scatter preserves absent widths so Grow publishes desc.defaultWidth");
}

void TestScatterDensityValidation()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto hasDensityError = [](UsdGenDiagnostics const &diagnostics) {
        return std::any_of(diagnostics.errors.begin(), diagnostics.errors.end(),
            [](std::string const &error) {
                return error.find("density must be finite and >= 0") !=
                    std::string::npos;
            });
    };
    auto negative = compiler.Compile(MakeScatterValidationDesc(-1.0), &graph);
    Check(negative.ok, "negative Scatter density reaches capture validation");
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    auto negativeRun = scheduler.Run(graph, context, 1);
    Check(negativeRun.diagnostics.HasErrors() && hasDensityError(negativeRun.diagnostics),
          "Scatter rejects negative density before root allocation");
    auto nonfinite = compiler.Compile(MakeScatterValidationDesc(
        std::numeric_limits<double>::quiet_NaN()), &graph);
    Check(nonfinite.ok, "non-finite Scatter density reaches capture validation");
    context.desc = &graph.Desc();
    auto nonfiniteRun = scheduler.Run(graph, context, 1);
    Check(nonfiniteRun.diagnostics.HasErrors() && hasDensityError(nonfiniteRun.diagnostics),
          "Scatter rejects non-finite density before root allocation");

    auto overflow = compiler.Compile(MakeScatterValidationDesc(1.0e20), &graph);
    Check(overflow.ok, "Scatter cardinality-overflow fixture compiles before capture");
    context.desc = &graph.Desc();
    auto const run = scheduler.Run(graph, context, 1);
    Check(run.diagnostics.HasErrors() && !run.diagnostics.errors.empty() &&
              run.diagnostics.errors.front().find("root count exceeds uint32 cardinality") !=
                  std::string::npos,
          "Scatter rejects uint32 root cardinality overflow before allocation");
}

void TestAuthoredSourcePlanes()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/authoredPlanes");
    desc.terminal = SdfPath("/authoredPlanes/writer");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/authoredPlanes/surface");
    // C3 skinprim addresses parent mesh faces, which must be actual polygon
    // faces (at least three vertices).  This fixture only needs the binding
    // provenance; retain two 2-CV source curves independently of the mesh.
    surface.faceVertexCounts = VtIntArray{3, 3};
    surface.faceVertexIndices = VtIntArray{0, 1, 2, 1, 3, 2};
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(0, 0, 1),
        GfVec3f(1, 0, 0), GfVec3f(1, 0, 1)};
    surface.points = surface.restPoints;
    desc.surfaces.push_back(surface);

    UsdGenCurveSetDesc sourceCurves;
    sourceCurves.path = SdfPath("/authoredPlanes/curves");
    sourceCurves.role = UsdGenRole::Curves;
    sourceCurves.curveRole = TfToken("hair");
    sourceCurves.curveVertexCounts = VtIntArray{2, 2};
    sourceCurves.points = surface.restPoints;
    sourceCurves.rest = sourceCurves.points;
    sourceCurves.curveId = {0, 1};
    sourceCurves.skinPrim = {0, 1};
    sourceCurves.skinPrimUv = {{0, 0}, {0, 0}};
    sourceCurves.rootFrame = {GfMatrix4d(1.0), GfMatrix4d(1.0)};
    UsdGenAuthoredPlaneDesc point;
    point.name = TfToken("cvPayload");
    point.type = UsdGenAuthoredPlaneType::Float32;
    point.domain = UsdGenAuthoredPlaneDomain::Point;
    point.arity = 2;
    point.floatValues = VtFloatArray{0, 10, 1, 11, 2, 12, 3, 13};
    sourceCurves.authoredPlanes.push_back(point);
    UsdGenAuthoredPlaneDesc primitive;
    primitive.name = TfToken("curvePayload");
    primitive.type = UsdGenAuthoredPlaneType::Int32;
    primitive.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primitive.intValues = VtIntArray{7, 9};
    sourceCurves.authoredPlanes.push_back(primitive);
    UsdGenAuthoredPlaneDesc groom;
    groom.name = TfToken("groomPayload");
    groom.type = UsdGenAuthoredPlaneType::Float32;
    groom.domain = UsdGenAuthoredPlaneDomain::Groom;
    groom.floatValues = VtFloatArray{0.5f};
    sourceCurves.authoredPlanes.push_back(groom);
    desc.curveSets.push_back(sourceCurves);

    UsdGenNodeDesc source = SourceNode();
    source.path = SdfPath("/authoredPlanes/source");
    source.surfaces = {surface.path};
    source.curves = {sourceCurves.path};
    UsdGenNodeDesc writer;
    writer.path = desc.terminal;
    writer.type = TfToken("UsdGenWidth");
    writer.inputs = {source.path};
    writer.params.push_back({TfToken("width"), VtValue(0.2f), false});
    desc.nodes = {writer, source};

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "authored named-plane source fixture compiles");
    if (!compiled.ok) return;
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, context, 1);
    Check(!run.diagnostics.HasErrors(), "authored named-plane source executes on CPU");
    if (run.diagnostics.HasErrors()) return;
    auto const sourceBufferView = NodeView(graph, 
        graph.NodeIdForPath(source.path));
    UsdGenCurveBuffer const sourceBuffer = NodeBuffer(graph, 
        graph.NodeIdForPath(source.path));
    UsdGenNodeId writerId = graph.NodeIdForPath(writer.path);
    auto const writerBufferView = NodeView(graph, writerId);
    UsdGenCurveBuffer const writerBuffer = NodeBuffer(graph, writerId);
    VtFloatArray const expectedPoint = VtFloatArray{
        0, 10, 1.0f / 3.0f, 10.0f + 1.0f / 3.0f,
        2.0f / 3.0f, 10.0f + 2.0f / 3.0f, 1, 11,
        2, 12, 2.0f + 1.0f / 3.0f, 12.0f + 1.0f / 3.0f,
        2.0f + 2.0f / 3.0f, 12.0f + 2.0f / 3.0f, 3, 13};
    bool const sourcePlanesMaterialized =
        sourceBuffer.extraCv.size() == 1 && sourceBuffer.extraCurve.size() == 2 &&
        sourceBuffer.extraCv[0].name == TfToken("cvPayload") &&
        sourceBuffer.extraCv[0].arity == 2 && sourceBuffer.extraCv[0].f == expectedPoint &&
        sourceBuffer.extraCurve[0].name == TfToken("curvePayload") &&
        sourceBuffer.extraCurve[0].i == primitive.intValues &&
        sourceBuffer.extraCurve[1].name == TfToken("groomPayload") &&
        sourceBuffer.extraCurve[1].interpolation == TfToken("constant");
    Check(sourcePlanesMaterialized,
          "source materializes point, primitive, and groom authored planes");
    if (!sourcePlanesMaterialized) return;
    Check(sourceBufferView.ExtraCv(0).f.cdata() !=
              graph.Desc().curveSets[0].authoredPlanes[0].floatValues.cdata(),
          "resampled source plane materializes private COW storage");
    Check(writerBuffer.extraCv.size() == 1 && writerBufferView.ExtraCv(0).f.cdata() ==
              sourceBufferView.ExtraCv(0).f.cdata(),
          "unchanged downstream plane shares immutable source COW data");

    // Recompile may retain CurveSource's op/capture when relationship topology
    // is unchanged. Its capture digest must still invalidate stale authored
    // COW aliases when a payload changes or the authored planes disappear.
    desc.curveSets[0].authoredPlanes[1].intValues[0] = 42;
    compiled = compiler.Recompile(desc, &graph);
    Check(compiled.ok, "authored plane value edit recompiles incrementally");
    context.desc = &graph.Desc();
    run = scheduler.Run(graph, context, 2);
    auto const editedView = NodeView(graph, 
        graph.NodeIdForPath(source.path));
    UsdGenCurveBuffer const edited = NodeBuffer(graph, 
        graph.NodeIdForPath(source.path));
    Check(!run.diagnostics.HasErrors() && edited.extraCurve.size() == 2 &&
              edited.extraCurve[0].i[0] == 42,
          "incremental capture refreshes changed authored COW payload");

    desc.curveSets[0].authoredPlanes.clear();
    compiled = compiler.Recompile(desc, &graph);
    Check(compiled.ok, "authored plane removal recompiles incrementally");
    context.desc = &graph.Desc();
    run = scheduler.Run(graph, context, 3);
    auto const removedView = NodeView(graph, 
        graph.NodeIdForPath(source.path));
    UsdGenCurveBuffer const removed = NodeBuffer(graph, 
        graph.NodeIdForPath(source.path));
    Check(!run.diagnostics.HasErrors() && removed.extraCv.empty() &&
              removed.extraCurve.empty(),
          "incremental capture removes stale authored COW planes");
}

void TestAuthoredPlaneValidation()
{
    UsdGenGraphDesc desc = MakeFanout("terminal", "sibling", {0, 1, 2});
    desc.curveSets.clear();
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/fanout/curves");
    curves.role = UsdGenRole::Curves;
    curves.curveVertexCounts = VtIntArray{4, 4};
    curves.points.resize(8);
    UsdGenAuthoredPlaneDesc invalid;
    invalid.name = TfToken("bad");
    invalid.type = UsdGenAuthoredPlaneType::Float32;
    invalid.domain = UsdGenAuthoredPlaneDomain::Point;
    invalid.arity = 17;
    curves.authoredPlanes.push_back(invalid);
    desc.curveSets.push_back(curves);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult result = compiler.Compile(desc, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("arity outside [1,16]") != std::string::npos,
          "invalid authored plane arity fails closed during compile");
    UsdGenGraph retained;
    Check(compiler.Compile(MakeFanout("terminal", "sibling", {0, 1, 2}),
                           &retained).ok,
          "valid graph seeds transactional authored-plane validation fixture");
    result = compiler.Recompile(desc, &retained);
    Check(!result.ok && retained.NodeCount() == 3,
          "invalid authored plane recompile preserves prior graph transactionally");

    UsdGenGraphDesc duplicate = desc;
    duplicate.curveSets[0].authoredPlanes.clear();
    UsdGenAuthoredPlaneDesc valid;
    valid.name = TfToken("duplicate");
    valid.floatValues = VtFloatArray(8, 1.0f);
    duplicate.curveSets[0].authoredPlanes = {valid, valid};
    result = compiler.Compile(duplicate, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("empty or duplicate authored plane") != std::string::npos,
          "duplicate authored plane names fail closed during compile");

    UsdGenGraphDesc wrongCardinality = desc;
    wrongCardinality.curveSets[0].authoredPlanes.clear();
    UsdGenAuthoredPlaneDesc wrongType;
    wrongType.name = TfToken("wrongType");
    wrongType.type = UsdGenAuthoredPlaneType::Int32;
    wrongType.domain = UsdGenAuthoredPlaneDomain::Primitive;
    wrongType.intValues = VtIntArray{1};
    wrongCardinality.curveSets[0].authoredPlanes.push_back(wrongType);
    result = compiler.Compile(wrongCardinality, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("invalid type or cardinality") != std::string::npos,
          "authored plane type/cardinality mismatch fails closed during compile");

    UsdGenGraphDesc reserved = desc;
    reserved.curveSets[0].authoredPlanes.clear();
    UsdGenAuthoredPlaneDesc collision;
    collision.name = TfToken("widths");
    collision.floatValues = VtFloatArray(8, 1.0f);
    reserved.curveSets[0].authoredPlanes.push_back(collision);
    result = compiler.Compile(reserved, &graph);
    Check(!result.ok && !result.errors.empty() &&
              result.errors.front().find("reserved C3 channel") != std::string::npos,
          "authored plane cannot shadow a standard C3 channel");
}

UsdGenGraphDesc MakeExtraPlaneFanout()
{
    UsdGenGraphDesc desc = MakeFanout("unused", "alsoUnused", {0, 1, 2});
    desc.terminal = SdfPath("/fanout/cvWriter");
    UsdGenNodeDesc cv;
    cv.path = desc.terminal;
    cv.type = TfToken("UsdGenTestExtraCvWriter");
    cv.inputs = {SdfPath("/fanout/source")};
    UsdGenNodeDesc curve;
    curve.path = SdfPath("/fanout/curveWriter");
    curve.type = TfToken("UsdGenTestExtraCurveWriter");
    curve.inputs = {SdfPath("/fanout/source")};
    desc.nodes = {curve, SourceNode(), cv};
    return desc;
}

void TestExtraPlaneCowAndTopologyRefresh()
{
    UsdGenGraphDesc desc = MakeExtraPlaneFanout();
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "extra-plane fan-out compiles");
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(),
          "extra-plane fan-out establishes source topology");

    auto sourceId = graph.NodeIdForPath(SdfPath("/fanout/source"));
    auto cvId = graph.NodeIdForPath(SdfPath("/fanout/cvWriter"));
    auto curveId = graph.NodeIdForPath(SdfPath("/fanout/curveWriter"));
    {
        UsdGenCurveBuffer planes = NodeBuffer(graph, sourceId);
        InstallExtraPlanes(&planes);
        Check(SetNodeBuffer(graph, sourceId, planes), "extra-plane fixture installs");
    }
    graph.MarkNode(cvId, UsdGenDirtyParameter);
    graph.MarkNode(curveId, UsdGenDirtyParameter);
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "extra-plane fan-out evaluates declared slots");

    auto const sourceView = NodeView(graph, sourceId);
    UsdGenCurveBuffer const source = NodeBuffer(graph, sourceId);
    auto const cvView = NodeView(graph, cvId);
    UsdGenCurveBuffer const cv = NodeBuffer(graph, cvId);
    auto const curveView = NodeView(graph, curveId);
    UsdGenCurveBuffer const curve = NodeBuffer(graph, curveId);
    Check(cvView.ExtraCv(0).f.cdata() != sourceView.ExtraCv(0).f.cdata() &&
              curveView.ExtraCurve(0).i.cdata() != sourceView.ExtraCurve(0).i.cdata(),
          "declared OutputPrimvars privately materialize their VtArray storage");
    Check(cvView.ExtraCurve(0).i.cdata() == sourceView.ExtraCurve(0).i.cdata() &&
              curveView.ExtraCv(0).f.cdata() == sourceView.ExtraCv(0).f.cdata(),
          "fan-out branches retain aliases for untouched extra CV and curve planes");
    Check(cv.extraCv[0].f[3] == source.extraCv[0].f[3] + 10.0f &&
              curve.extraCurve[0].i[1] == source.extraCurve[0].i[1] + 20 &&
              source.extraCv[0].f[3] == 3.0f && source.extraCurve[0].i[1] == 1,
          "extra-plane input/output slots isolate writes from their shared source");

    // Change the source's authored resample topology while retaining the
    // compiled nodes.  This exercises refresh of old extra-plane aliases and
    // private output arrays through the actual incremental Recompile path.
    for (auto& node : desc.nodes)
        if (node.path == SdfPath("/fanout/source"))
            node.params.front().value = VtValue(6);
    Check(compiler.Recompile(desc, &graph).ok,
          "extra-plane fan-out recompiles changed source topology");
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 3).diagnostics.HasErrors(),
          "extra-plane fan-out refreshes changed source topology");
    sourceId = graph.NodeIdForPath(SdfPath("/fanout/source"));
    cvId = graph.NodeIdForPath(SdfPath("/fanout/cvWriter"));
    curveId = graph.NodeIdForPath(SdfPath("/fanout/curveWriter"));
    {
        UsdGenCurveBuffer planes = NodeBuffer(graph, sourceId);
        InstallExtraPlanes(&planes);
        Check(SetNodeBuffer(graph, sourceId, planes), "extra-plane fixture installs");
    }
    graph.MarkNode(cvId, UsdGenDirtyParameter);
    graph.MarkNode(curveId, UsdGenDirtyParameter);
    Check(!scheduler.Run(graph, context, 4).diagnostics.HasErrors(),
          "extra-plane fan-out evaluates refreshed aliases");

    auto const newSourceView = NodeView(graph, sourceId);
    UsdGenCurveBuffer const newSource = NodeBuffer(graph, sourceId);
    auto const newCvView = NodeView(graph, cvId);
    UsdGenCurveBuffer const newCv = NodeBuffer(graph, cvId);
    auto const newCurveView = NodeView(graph, curveId);
    UsdGenCurveBuffer const newCurve = NodeBuffer(graph, curveId);
    Check(newSource.totalCurves == 2 && newSource.totalCvs == 12 &&
              newCv.extraCv[0].f.size() == 12 && newCurve.extraCurve[0].i.size() == 2,
          "extra-plane output storage is resized for the refreshed topology");
    Check(newCvView.ExtraCurve(0).i.cdata() == newSourceView.ExtraCurve(0).i.cdata() &&
              newCurveView.ExtraCv(0).f.cdata() == newSourceView.ExtraCv(0).f.cdata() &&
              newCvView.ExtraCv(0).f.cdata() != newSourceView.ExtraCv(0).f.cdata() &&
              newCurveView.ExtraCurve(0).i.cdata() != newSourceView.ExtraCurve(0).i.cdata(),
          "topology refresh replaces stale extra-plane aliases while preserving COW isolation");
}

void TestFanoutDeterminism()
{
    // The first graph evaluates the terminal branch before its sibling by
    // lexical tie-break.  The second puts the sibling first and also reverses
    // authored order.  Terminal selection, not sibling scheduling, determines
    // the published value.
    FanoutResult const terminalFirst = RunFanout(
        MakeFanout("aTerminal", "zSibling", {2, 0, 1}), "zSibling");
    FanoutResult const siblingFirst = RunFanout(
        MakeFanout("zTerminal", "aSibling", {1, 0, 2}), "aSibling");

    Check(AllEqual(terminalFirst.terminal, 0.25f),
          "terminal-first graph publishes only terminal branch widths");
    Check(AllEqual(siblingFirst.terminal, 0.25f),
          "sibling-first graph publishes only terminal branch widths");
    Check(terminalFirst.terminal == siblingFirst.terminal,
          "terminal output is deterministic across authored/sibling order");
    Check(terminalFirst.siblingRan && siblingFirst.siblingRan,
          "non-terminal sibling executes in both topological orders");
    Check(AllEqual(terminalFirst.source, UsdGenGraphDesc{}.defaultWidth) &&
              AllEqual(siblingFirst.source, UsdGenGraphDesc{}.defaultWidth),
          "both fan-out executions preserve the shared source value");
    Check(terminalFirst.branchStorageIsolated &&
              siblingFirst.branchStorageIsolated,
          "fan-out consumers own isolated writable width storage");
    Check(terminalFirst.unchangedPlanesShared &&
              siblingFirst.unchangedPlanesShared,
          "fan-out consumers CoW-share every unchanged CV plane");
}

void TestCowAliasesRefreshAfterTopologyRevision()
{
    UsdGenGraphDesc desc = MakeFanout(
        "terminal", "sibling", {0, 1, 2});
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok,
          "COW refresh fixture compiles its initial topology");
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(),
          "COW refresh fixture evaluates its initial topology");

    // Add a third authored curve while retaining both Width nodes. Recompile
    // deliberately exercises persistent node buffers and their old aliases.
    auto &surface = desc.surfaces.front();
    surface.faceVertexCounts = VtIntArray{4, 4, 4};
    surface.faceVertexIndices = VtIntArray{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    surface.restPoints.push_back(GfVec3f(2, 0, 0));
    surface.restPoints.push_back(GfVec3f(2, 0, 1));
    surface.restPoints.push_back(GfVec3f(2, 0, 2));
    surface.restPoints.push_back(GfVec3f(2, 0, 3));
    surface.points = surface.restPoints;
    Check(compiler.Compile(desc, &graph).ok,
          "COW refresh fixture recompiles a changed source topology");
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "COW refresh fixture evaluates the changed source topology");

    auto const sourceView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/source")));
    UsdGenCurveBuffer const source = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/source")));
    auto const terminalView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/terminal")));
    UsdGenCurveBuffer const terminal = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/terminal")));
    auto const siblingView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/sibling")));
    UsdGenCurveBuffer const sibling = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/sibling")));
    Check(source.totalCurves == 3 && terminal.totalCurves == 3 &&
              sibling.totalCurves == 3 && source.totalCvs == 12 &&
              terminal.totalCvs == 12 && sibling.totalCvs == 12,
          "reused COW branches refresh inherited topology totals");
    Check(terminalView.curveId.cdata() == sourceView.curveId.cdata() &&
              siblingView.curveId.cdata() == sourceView.curveId.cdata() &&
              terminalView.rootUV.cdata() == sourceView.rootUV.cdata() &&
              siblingView.rootUV.cdata() == sourceView.rootUV.cdata(),
          "reused COW branches replace stale per-curve aliases");
    Check(terminalView.px.cdata() == sourceView.px.cdata() &&
              siblingView.px.cdata() == sourceView.px.cdata() &&
              terminalView.width.cdata() != sourceView.width.cdata() &&
              siblingView.width.cdata() != sourceView.width.cdata() &&
              terminalView.width.cdata() != siblingView.width.cdata(),
          "topology revision still shares reads and materializes writes");
}

void TestUnaryFanInRejected()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/fanin");
    desc.terminal = SdfPath("/fanin/width");

    UsdGenNodeDesc sourceA;
    sourceA.path = SdfPath("/fanin/sourceA");
    sourceA.type = TfToken("UsdGenCurveSource");
    sourceA.surfaces = {SdfPath("/fanin/curves")};
    UsdGenNodeDesc sourceB = sourceA;
    sourceB.path = SdfPath("/fanin/sourceB");

    UsdGenNodeDesc width;
    width.path = desc.terminal;
    width.type = TfToken("UsdGenWidth");
    width.inputs = {sourceA.path, sourceB.path};
    desc.nodes = {sourceB, width, sourceA};

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenGraphDesc retained;
    retained.description = SdfPath("/retained");
    retained.terminal = SdfPath("/retained/source");
    UsdGenNodeDesc retainedSource;
    retainedSource.path = retained.terminal;
    retainedSource.type = TfToken("UsdGenCurveSource");
    retained.nodes = {retainedSource};
    Check(compiler.Compile(retained, &graph).ok,
          "valid destination graph is seeded before rejection");
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    Check(!compiled.ok, "multi-input use of unary Width is rejected at compile");
    Check(compiled.errors.size() == 1,
          "unary fan-in rejection emits one deterministic diagnostic");
    if (!compiled.errors.empty()) {
        Check(compiled.errors.front() ==
                  "UsdGenCompiler: operator '/fanin/width' (type 'UsdGenWidth') "
                  "requires exactly 1 geometry input; found 2",
              "unary fan-in diagnostic names operator, type, contract and actual arity");
    }
    Check(graph.NodeCount() == 1 &&
              graph.NodeIdForPath(retained.terminal) != UsdGenGraph::InvalidNode &&
              graph.Desc().description == retained.description,
          "failed fan-in compile leaves the prior destination graph untouched");

    UsdGenGraphDesc missingInput;
    missingInput.description = SdfPath("/missing");
    missingInput.terminal = SdfPath("/missing/width");
    UsdGenNodeDesc missingWidth;
    missingWidth.path = missingInput.terminal;
    missingWidth.type = TfToken("UsdGenWidth");
    missingInput.nodes = {missingWidth};
    UsdGenGraph missingGraph;
    UsdGenCompileResult const missing = compiler.Compile(missingInput, &missingGraph);
    Check(!missing.ok && missing.errors.size() == 1 &&
              missing.errors.front() ==
                  "UsdGenCompiler: operator '/missing/width' (type 'UsdGenWidth') "
                  "requires exactly 1 geometry input; found 0",
          "unary Width without an input is rejected by the same contract");

    UsdGenGraphDesc sourceWithInput;
    sourceWithInput.description = SdfPath("/sourceInput");
    sourceWithInput.terminal = SdfPath("/sourceInput/second");
    UsdGenNodeDesc firstSource;
    firstSource.path = SdfPath("/sourceInput/first");
    firstSource.type = TfToken("UsdGenCurveSource");
    UsdGenNodeDesc secondSource = firstSource;
    secondSource.path = sourceWithInput.terminal;
    secondSource.inputs = {firstSource.path};
    sourceWithInput.nodes = {secondSource, firstSource};
    UsdGenGraph sourceGraph;
    UsdGenCompileResult const sourceResult =
        compiler.Compile(sourceWithInput, &sourceGraph);
    Check(!sourceResult.ok && sourceResult.errors.size() == 1 &&
              sourceResult.errors.front() ==
                  "UsdGenCompiler: operator '/sourceInput/second' "
                  "(type 'UsdGenCurveSource') requires exactly 0 geometry "
                  "inputs; found 1",
          "source operator with an input is rejected by its zero-input contract");

    // Keep the unary contract diagnostic exact for every currently shipped
    // unary geometry consumer. Dependency lowering must not turn these
    // malformed declarations into an implicit merge/fan-in.
    for (char const* type : {"UsdGenLength", "UsdGenDeform"}) {
        UsdGenGraphDesc malformed;
        malformed.description = SdfPath(std::string("/fanin/") + type);
        malformed.terminal = malformed.description.AppendChild(TfToken("operator"));
        UsdGenNodeDesc left = firstSource;
        left.path = malformed.description.AppendChild(TfToken("left"));
        UsdGenNodeDesc right = firstSource;
        right.path = malformed.description.AppendChild(TfToken("right"));
        UsdGenNodeDesc consumer;
        consumer.path = malformed.terminal;
        consumer.type = TfToken(type);
        consumer.inputs = {left.path, right.path};
        malformed.nodes = {consumer, left, right};
        UsdGenGraph rejected;
        UsdGenCompileResult const result = compiler.Compile(malformed, &rejected);
        std::string const expected =
            "UsdGenCompiler: operator '" + consumer.path.GetString() +
            "' (type '" + type +
            "') requires exactly 1 geometry input; found 2";
        Check(!result.ok && result.errors.size() == 1 &&
                  result.errors.front() == expected,
              std::string(type) +
                  " multi-geometry-input fan-in remains an exact unary rejection");
    }
}

void TestOrderedWidthBlend()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/widthBlend");
    desc.terminal = SdfPath("/widthBlend/merge");

    UsdGenSurfaceDesc curves;
    curves.path = SdfPath("/widthBlend/curves");
    curves.id = 0;
    curves.faceVertexCounts = VtIntArray{4, 4};
    curves.faceVertexIndices = VtIntArray{0, 1, 2, 3, 4, 5, 6, 7};
    curves.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(0, 0, 1),
        GfVec3f(0, 0, 2), GfVec3f(0, 0, 3),
        GfVec3f(1, 0, 0), GfVec3f(1, 0, 1),
        GfVec3f(1, 0, 2), GfVec3f(1, 0, 3)};
    curves.points = curves.restPoints;
    desc.surfaces.push_back(std::move(curves));

    UsdGenNodeDesc source;
    source.path = SdfPath("/widthBlend/source");
    source.type = TfToken("UsdGenCurveSource");
    source.surfaces = {SdfPath("/widthBlend/curves")};
    source.params.push_back({TfToken("resampleTo"), VtValue(4), false});
    UsdGenNodeDesc left;
    left.path = SdfPath("/widthBlend/left");
    left.type = TfToken("UsdGenWidth");
    left.inputs = {source.path};
    left.params.push_back({TfToken("width"), VtValue(0.25f), false});
    UsdGenNodeDesc right = left;
    right.path = SdfPath("/widthBlend/right");
    right.params[0].value = VtValue(0.75f);
    UsdGenNodeDesc merge;
    merge.path = desc.terminal;
    merge.type = TfToken("UsdGenWidthBlend");
    merge.inputs = {left.path, right.path};
    merge.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
    // Deliberately authored in a non-topological order.  The compiler must
    // retain merge.inputs' left/right correspondence after Kahn sorting.
    desc.nodes = {merge, right, source, left};

    size_t arity = 0;
    Check(GeometryArity("UsdGenWidthBlend", &arity) && arity == 2,
          "registry exposes WidthBlend ordered binary contract");

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "ordered WidthBlend graph compiles");
    if (!compiled.ok) return;
    UsdGenNodeId const mergeId = graph.NodeIdForPath(desc.terminal);
    UsdGenCompiledNode const &compiledMerge = graph.Node(mergeId);
    Check(compiledMerge.inputs.size() == 2 &&
              compiledMerge.inputs[0] == graph.NodeIdForPath(left.path) &&
              compiledMerge.inputs[1] == graph.NodeIdForPath(right.path),
          "WidthBlend preserves authored left/right input order after Kahn sort");

    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    Check(!run.diagnostics.HasErrors(),
          "ordered WidthBlend executes without diagnostics");
    if (run.diagnostics.HasErrors()) return;
    auto const leftBufferView = NodeView(graph, 
        graph.NodeIdForPath(left.path));
    UsdGenCurveBuffer const leftBuffer = NodeBuffer(graph, 
        graph.NodeIdForPath(left.path));
    auto const rightBufferView = NodeView(graph, 
        graph.NodeIdForPath(right.path));
    UsdGenCurveBuffer const rightBuffer = NodeBuffer(graph, 
        graph.NodeIdForPath(right.path));
    auto const mergedView = OutputView(graph);
    UsdGenCurveBuffer const merged = OutputBuffer(graph);
    bool values = merged.width.size() == 8;
    if (values)
        for (float value : merged.width)
            values = values && std::fabs(value - 0.375f) < 1e-6f;
    Check(values, "WidthBlend applies blend interpolation exactly once");
    Check(mergedView.width.cdata() != leftBufferView.width.cdata() &&
              mergedView.width.cdata() != rightBufferView.width.cdata() &&
              mergedView.px.cdata() == leftBufferView.px.cdata() &&
              mergedView.px.cdata() != mergedView.width.cdata(),
          "WidthBlend fresh width storage is CoW-isolated while non-width data aliases left");
    bool inputsPreserved = leftBuffer.width.size() == 8 &&
                           rightBuffer.width.size() == 8;
    if (inputsPreserved) {
        for (size_t i = 0; i != 8; ++i)
            inputsPreserved = inputsPreserved &&
                std::fabs(leftBuffer.width[i] - .25f) < 1e-6f &&
                std::fabs(rightBuffer.width[i] - .75f) < 1e-6f;
    }
    Check(inputsPreserved,
          "WidthBlend leaves both immutable predecessor width planes unchanged");

    // Swap the authored operands and verify that order is observable, not a
    // canonicalized set of dependencies.
    desc.nodes[0].inputs = {right.path, left.path};
    UsdGenGraph swapped;
    UsdGenCompileResult const swappedCompile = compiler.Compile(desc, &swapped);
    Check(swappedCompile.ok, "swapped WidthBlend graph compiles");
    if (!swappedCompile.ok) return;
    context.desc = &swapped.Desc();
    UsdGenRunResult const swappedRun = scheduler.Run(swapped, context, 2);
    Check(!swappedRun.diagnostics.HasErrors(),
          "swapped WidthBlend executes without diagnostics");
    if (!swappedRun.diagnostics.HasErrors()) {
        bool swappedValue = OutputBuffer(swapped).width.size() == 8;
        if (swappedValue)
            for (float value : OutputBuffer(swapped).width)
                swappedValue = swappedValue && std::fabs(value - 0.625f) < 1e-6f;
        Check(swappedValue, "WidthBlend authored operand order changes only merged widths");
    }

    // The descriptor-only operator is deliberately narrow.  Muting it or
    // attaching generic operator payload would otherwise create CPU/CUDA
    // divergence, while a duplicate predecessor would only disguise a unary
    // operation as fan-in.
    auto expectRejected = [&](UsdGenGraphDesc malformed, char const *message) {
        UsdGenGraph rejected;
        UsdGenCompileResult const result = compiler.Compile(malformed, &rejected);
        Check(!result.ok, message);
    };
    UsdGenGraphDesc disabled = desc;
    disabled.nodes[0].enabled = false;
    expectRejected(std::move(disabled), "WidthBlend rejects enabled=false");
    UsdGenGraphDesc duplicate = desc;
    duplicate.nodes[0].inputs = {left.path, left.path};
    expectRejected(std::move(duplicate), "WidthBlend rejects duplicate operands");
    UsdGenGraphDesc auxiliary = desc;
    auxiliary.nodes[0].params.push_back(
        {TfToken("unexpected"), VtValue(1.0f), false});
    expectRejected(std::move(auxiliary), "WidthBlend rejects auxiliary parameters");
    UsdGenGraphDesc invalidWeight = desc;
    invalidWeight.nodes[0].params[0].value = VtValue(1.25f);
    expectRejected(std::move(invalidWeight),
                   "WidthBlend rejects an out-of-range widthBlend:weight");
}

void TestDependencyReadyHeldFanout()
{
    UsdGenGraphDesc desc = MakeFanout("heldLeft", "heldRight", {2, 0, 1});
    for (UsdGenNodeDesc &node : desc.nodes) {
        if (node.path == SdfPath("/fanout/heldLeft") ||
            node.path == SdfPath("/fanout/heldRight")) {
            node.type = TfToken("UsdGenTestHeldWidthWriter");
        }
    }

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok,
          "held dependency-ready fan-out graph compiles");
    if (graph.NodeCount() == 0) return;

    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    HeldEvalGate gate;
    g_heldEvalGate = &gate;
    UsdGenRunResult run;
    std::atomic<bool> finished{false};
    std::thread runner([&] {
        run = scheduler.Run(graph, context, 1);
        finished.store(true, std::memory_order_release);
    });

    auto const deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(1000);
    while (gate.entered.load(std::memory_order_acquire) < 2 &&
           !finished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    bool const bothEntered = gate.entered.load(std::memory_order_acquire) >= 2;
    gate.release.store(true, std::memory_order_release);
    runner.join();
    g_heldEvalGate = nullptr;

    Check(bothEntered,
          "dependency-ready fan-out enters both independent held nodes concurrently");
    Check(finished.load(std::memory_order_acquire) &&
              !run.diagnostics.HasErrors(),
          "dependency-ready held fan-out completes without diagnostics");
    if (run.diagnostics.HasErrors()) return;

    auto const sourceView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/source")));
    UsdGenCurveBuffer const source = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/source")));
    auto const leftView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/heldLeft")));
    UsdGenCurveBuffer const left = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/heldLeft")));
    auto const rightView = NodeView(graph, graph.NodeIdForPath(
        SdfPath("/fanout/heldRight")));
    UsdGenCurveBuffer const right = NodeBuffer(graph, graph.NodeIdForPath(
        SdfPath("/fanout/heldRight")));
    Check(!left.width.empty() && !right.width.empty() &&
              leftView.width.cdata() != rightView.width.cdata() &&
              leftView.width.cdata() != sourceView.width.cdata() &&
              rightView.width.cdata() != sourceView.width.cdata(),
          "held fan-out branches materialize isolated writable COW widths");
    Check(!source.px.empty() && leftView.px.cdata() == sourceView.px.cdata() &&
              rightView.px.cdata() == sourceView.px.cdata(),
          "held fan-out branches retain immutable non-width aliases");
}

} // namespace

int main()
{
    size_t arity = 99;
    Check(GeometryArity("UsdGenCurveSource", &arity) && arity == 0,
          "registry exposes CurveSource zero-input contract");
    Check(GeometryArity("UsdGenWidth", &arity) && arity == 1,
          "registry exposes Width unary contract");
    Check(GeometryArity("UsdGenGrow", &arity) && arity == 1,
          "topology-owning Grow remains a unary geometry consumer");
    Check(!GeometryArity("UsdGenMissingInputContract", &arity),
          "registry rejects an unknown input contract");
    Check(UsdGenRegisterStatic(&RegisterFanoutTestOps),
          "registers the fan-out test writers");
    for (char const *name : {"UsdGenTestExtraCvWriter", "UsdGenTestExtraCurveWriter",
                             "UsdGenTestHeldWidthWriter"})
        Check(UsdGenOpRegistry::Get().Find(TfToken(name)) != UsdGenOpId::Invalid,
              std::string("registered ") + name);
    TestFanoutDeterminism();
    TestCowAliasesRefreshAfterTopologyRevision();
    TestExtraPlaneCowAndTopologyRefresh();
    TestTopologyExtraPlaneTransforms();
    TestGrowNamedPlaneTopologyOwnership();
    TestGrowWidthFallbackCapture();
    TestGrowAngularLift();
    TestGrowAzimuth();
    TestScatterGrowDefaultWidthThroughScheduler();
    TestScatterDensityValidation();
    TestAuthoredSourcePlanes();
    TestAuthoredPlaneValidation();
    TestUnaryFanInRejected();
    TestOrderedWidthBlend();
    TestDependencyReadyHeldFanout();
    std::printf(g_failures ? "testUsdGenCpuFanout: FAILED (%d)\n"
                           : "testUsdGenCpuFanout: PASS\n",
                g_failures);
    return g_failures ? 1 : 0;
}
