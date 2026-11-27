// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenOpBridge (5.3a, 5.3f): ops over buffer-backed set views.
//  - an op written against the Capture/Evaluate family cooks in a graph and
//    its Capture sees the input node's output set (5.4: no buffer behind it);
//  - a direct Evaluate over a test chunk view reproduces the cooked bytes;
//  - UsdGenOp has no default Capture/Evaluate (5.3f deleted the old family);
//  - UsdGenCurveBufferSetView (tests/curveBufferSetView.h) shows a buffer as
//    a set over the buffer's own arrays and rebinds without allocating.

#include "usdGen/compile/compiler.h"
#include "nodeSets.h"
#include "usdGen/compile/graph.h"
#include "usdGenUsd/lower.h"
#include "usdGen/exec/allocCounter.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/registrar.h"
#include "bufferOpCalls.h"
#include "curveBufferSetView.h"
#include "usdGenCurves/types/curveSet.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "utils/pxrInterop.h"

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

static_assert(std::is_abstract<UsdGenOp>::value, "an op must implement Capture and Evaluate");

char const *const kProbeType = "UsdGenBridgeProbe";

struct ProbeCapture final : UsdGenCapture
{
    uint32_t curves = 0, cvs = 0;
    bool oneInput = false;
    float const *inPx = nullptr;
    std::unique_ptr<UsdGenCapture> Clone() const override { return std::make_unique<ProbeCapture>(*this); }
};

/// Writes p = in + (1, 2, 3) over uniform chunks.
void Shift(UsdGenCurveChunk const &v)
{
    for (uint32_t c = 0; c < v.curveCount; ++c)
        for (uint32_t i = 0; i < v.cvCount; ++i) {
            size_t const o = v.Cv(c, i), u = size_t(c) * v.inCvCount + i;
            v.px[o] = v.inPx[u] + 1.0f;
            v.py[o] = v.inPy[u] + 2.0f;
            v.pz[o] = v.inPz[u] + 3.0f;
        }
}

class Probe final : public UsdGenOp
{
public:
    explicit Probe(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {7, 1}; }
    uint32_t PlanesTouched() const override { return kPlanePoints; }
    std::unique_ptr<UsdGenCapture> CreateCapture() const override
    {
        return std::make_unique<ProbeCapture>();
    }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &in, UsdGenCapture *out,
                 UsdGenDiagnostics *) override
    {
        UsdGenCurveSetView const up = UsdGenCurveSetView::Of(in.Primary());
        auto &cap = static_cast<ProbeCapture &>(*out);
        cap.curves = up.totalCurves;
        cap.cvs = up.totalCvs;
        cap.oneInput = in.count == 1;
        cap.inPx = up.px.data();
        return true;
    }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView const &chunk) const override
    {
        UsdGenCurveChunk const curve(chunk);
        Shift(curve);
    }
};

void RegisterProbes(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = kProbeType;
    d.displayName = kProbeType + 6;
    d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
        return std::make_unique<Probe>(rec);
    };
    r.RegisterOp(d);
}

UsdGenGraphDesc MakeDesc(char const *probeType)
{
    constexpr int N = 8;
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/probe");
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.restPoints = usdGen::UsdGenVec3fArray((N + 1) * (N + 1));
    s.uv = usdGen::UsdGenVec2fArray((N + 1) * (N + 1));
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i <= N; ++i) {
            int const k = j * (N + 1) + i;
            s.restPoints[k] = usdGen::UsdGenVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = usdGen::UsdGenVec2f(float(i) / N, float(j) / N);
        }
    s.faceVertexCounts = VtIntArray(N * N, 4);
    s.faceVertexIndices = VtIntArray(N * N * 4);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            int const a = j * (N + 1) + i, o = (j * N + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + N + 2;
            s.faceVertexIndices[o + 3] = a + N + 1;
        }
    d.surfaces.push_back(std::move(s));
    auto add = [&](char const *name, char const *type, char const *input) {
        UsdGenNodeDesc n;
        n.path = SdfPath(std::string("/groom/") + name);
        n.type = TfToken(type);
        if (*input) n.inputs.push_back(SdfPath(std::string("/groom/") + input));
        else n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    add("scatter", "UsdGenScatter", "");
    add("grow", "UsdGenGrow", "scatter");
    add("probe", probeType, "grow");
    return d;
}

struct Cooked
{
    UsdGenGraph graph;
    UsdGenRunResult result;
    bool compiled = false;
};

void TestProbe()
{
    UsdGenScheduler scheduler(4);
    Cooked cooked;
    UsdGenCompiler compiler;
    cooked.compiled = usdGenUsd::Compile(compiler, MakeDesc(kProbeType), &cooked.graph).ok;
    Check(cooked.compiled, std::string(kProbeType) + " compiles");
    if (!cooked.compiled) return;
    UsdGenEvalContext ctx;
    ctx.desc = &cooked.graph.Spec();
    cooked.result = scheduler.Run(cooked.graph, ctx, 1);
    Check(!cooked.result.diagnostics.HasErrors(), "the probe cooks");

    UsdGenNodeId const probe = cooked.graph.TerminalNodeId();
    auto const growView = NodeView(cooked.graph, cooked.graph.Node(probe).input);
    UsdGenCurveBuffer const grow = NodeBuffer(cooked.graph, cooked.graph.Node(probe).input);
    UsdGenCurveBuffer const a = OutputBuffer(cooked.graph);
    Check(a.totalCvs > 0 && a.totalCvs == grow.totalCvs, "the probe wrote every CV");
    bool shifted = a.px.size() == grow.px.size();
    for (size_t i = 0; shifted && i < a.px.size(); ++i)
        shifted = a.px[i] == grow.px[i] + 1.0f && a.py[i] == grow.py[i] + 2.0f && a.pz[i] == grow.pz[i] + 3.0f;
    Check(shifted, "the probe moved every CV by (1, 2, 3)");
    auto const *nc = dynamic_cast<ProbeCapture const *>(cooked.graph.Node(probe).capture.get());
    Check(nc && nc->curves == grow.totalCurves && nc->cvs == grow.totalCvs && nc->oneInput &&
              nc->inPx == growView.px.data(),
          "Capture saw the input node's output set");

    // Direct calls: Capture over UsdGenBufferInputs, Evaluate over a test chunk.
    UsdGenOp &op = *cooked.graph.Node(probe).op;
    UsdGenCaptureContext cctx;
    cctx.desc = &cooked.graph.Spec();
    UsdGenDiagnostics diag;
    std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
    bool const captured = op.Capture(cctx, UsdGenBufferInputs(grow), cap.get(), &diag);
    auto const &pc = static_cast<ProbeCapture const &>(*cap);
    Check(captured && pc.curves == grow.totalCurves && pc.cvs == grow.totalCvs && pc.oneInput &&
              pc.inPx == grow.px.cdata(),
          "a direct Capture reads a buffer through UsdGenBufferInputs");

    uint32_t const cvp = grow.totalCurves ? grow.totalCvs / grow.totalCurves : 0;
    std::vector<float> x(grow.totalCvs), y(grow.totalCvs), z(grow.totalCvs);
    UsdGenChunkDesc desc;
    desc.curveCount = desc.liveCount = grow.totalCurves;
    desc.cvCount = cvp;
    UsdGenTestChunk view;
    view.desc = &desc;
    view.px = x.data(); view.py = y.data(); view.pz = z.data();
    view.inPx = growView.px.cdata(); view.inPy = growView.py.cdata(); view.inPz = growView.pz.cdata();
    view.curveCount = grow.totalCurves;
    view.cvCount = view.inCvCount = cvp;
    UsdGenEvalContext ectx;
    op.Evaluate(ectx, *cap, view.View());
    bool same = x.size() == a.px.size();
    for (size_t i = 0; same && i < x.size(); ++i) same = x[i] == a.px[i] && y[i] == a.py[i] && z[i] == a.pz[i];
    Check(same, "a direct Evaluate over a test chunk reproduces the cooked bytes");
}

void TestBufferSetView()
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenCurveBuffer b;
    b.totalCurves = 2;
    b.totalCvs = 5;
    b.px = VtFloatArray(5, 1.0f); b.py = VtFloatArray(5, 2.0f); b.pz = VtFloatArray(5, 3.0f);
    b.rest = usdGen::UsdGenVec3fArray(5, usdGen::UsdGenVec3f(0.5f));
    b.curveId = VtArray<uint64_t>(2, 9u);
    b.cvOffsets = VtIntArray{0, 2, 5};
    UsdGenPlane part;
    part.name = TfToken("partId");
    part.interpolation = tk.uniform;
    part.type = tk.int_;
    part.i = VtIntArray(2, 4);
    b.extraCurve.push_back(part);
    UsdGenPlane w;
    w.name = TfToken("weight");
    w.interpolation = tk.vertex;
    w.type = tk.float_;
    w.f = VtFloatArray(5, 0.25f);
    b.extraCv.push_back(w);

    UsdGenCurveBufferSetView sv;
    sv.Bind(b);
    UsdGenElementSetView const &v = sv.View();
    UsdGenSlot const partSlot = v.channels ? v.channels->Find(part.name) : kUsdGenNoSlot;
    UsdGenSlot const wSlot = v.channels ? v.channels->Find(w.name) : kUsdGenNoSlot;
    Check(v.channels && v.count[1] == 2 && v.count[2] == 5 &&
              v.offsets[2] == reinterpret_cast<uint32_t const *>(b.cvOffsets.cdata()),
          "set view counts and offsets are the buffer's");
    Check(v.columns[kUsdGenCurvePx] == b.px.cdata() && v.values[kUsdGenCurveRest] == 15 &&
              v.columns[kUsdGenCurveRest] == b.rest.cdata() && !v.columns[kUsdGenCurveWidth] &&
              v.columns[kUsdGenCurveId] == b.curveId.cdata(),
          "static columns alias the buffer arrays; empty arrays are absent");
    Check(partSlot != kUsdGenNoSlot && wSlot != kUsdGenNoSlot &&
              v.columns[partSlot] == b.extraCurve[0].i.cdata() &&
              v.columns[wSlot] == b.extraCv[0].f.cdata() &&
              (*v.channels)[partSlot].level == 1 && (*v.channels)[wSlot].level == 2,
          "named planes are dynamic channels over the plane arrays");
    UsdGenCurveSetView const cs = UsdGenCurveSetView::Of(v);
    Check(cs.ExtraCurveCount() == 1 && cs.ExtraCvCount() == 1 &&
              cs.ExtraCurve(0).i.data() == b.extraCurve[0].i.cdata() &&
              cs.ExtraCv(0).f.data() == b.extraCv[0].f.cdata() &&
              cs.totalCvs == 5 && cs.px.data() == b.px.cdata(),
          "UsdGenCurveSetView reads a buffer-backed set as the buffer");

    UsdGenChannelTable const *const table = sv.Channels();
    b.px = VtFloatArray(5, 7.0f);   // new storage, same layout
    UsdGenEngineAllocCounting(true);
    sv.Bind(b);
    uint64_t const allocs = UsdGenEngineAllocCount();
    UsdGenEngineAllocCounting(false);
    Check(allocs == 0 && sv.Channels() == table && sv.View().columns[kUsdGenCurvePx] == b.px.cdata(),
          "a same-layout rebind keeps the table and allocates nothing");
    b.extraCv.clear();
    sv.Bind(b);
    Check(sv.Channels() != table && sv.View().channels->Find(w.name) == kUsdGenNoSlot,
          "a layout change rebuilds the table");
}

}  // namespace

int main()
{
    Check(UsdGenRegisterStatic(&RegisterProbes), "probes register before the first compile");
    TestProbe();
    TestBufferSetView();
    std::printf("testUsdGenOpBridge: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
