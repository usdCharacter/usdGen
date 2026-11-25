// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenOpBridge (5.3a): the old and new op families meet in the middle.
//  - an op written against the new Capture/Evaluate family cooks the same
//    bytes as its twin written against the old one;
//  - the old entry points reach a new-family op through the forwarding
//    defaults (views over the buffers, a wrapper chunk view);
//  - an op overriding neither family fails its capture or its cook;
//  - UsdGenCurveBufferSetView shows a buffer as a set over the buffer's own
//    arrays and rebinds without allocating.
// Every run here has USDGEN_BRIDGE_VERIFY on: the scheduler compares its
// generic chunk view with the old one per chunk and fails the cook on a
// difference.

#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGen/compile/graphDesc.h"
#include "usdGen/exec/allocCounter.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/registrar.h"
#include "usdGenCurves/types/curveBridge.h"
#include "usdGenCurves/types/curveSet.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/setenv.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

enum Family : intptr_t { kOld, kNew, kNeitherCapture, kNeitherEvaluate, kFamilies };
char const *const kTypes[kFamilies] = {"UsdGenBridgeOld", "UsdGenBridgeNew",
                                       "UsdGenBridgeNeitherCapture", "UsdGenBridgeNeitherEvaluate"};

struct ProbeCapture final : UsdGenCapture
{
    uint32_t curves = 0, cvs = 0;
    bool sawBuffer = false;
    std::unique_ptr<UsdGenCapture> Clone() const override { return std::make_unique<ProbeCapture>(*this); }
};

/// Writes p = in + (1, 2, 3) over uniform chunks.
template <class View>
void Shift(View const &v)
{
    for (uint32_t c = 0; c < v.curveCount; ++c)
        for (uint32_t i = 0; i < v.cvCount; ++i) {
            size_t const o = v.Cv(c, i), u = size_t(c) * v.inCvCount + i;
            v.px[o] = v.inPx[u] + 1.0f;
            v.py[o] = v.inPy[u] + 2.0f;
            v.pz[o] = v.inPz[u] + 3.0f;
        }
}

/// One op class; a family that does not "override" an entry point calls the
/// base default, which is what an op without the override gets.
class Probe final : public UsdGenOp
{
public:
    Probe(UsdGenOpRecord const &r, Family f) : UsdGenOp(r), _f(f) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {7, 1}; }
    uint32_t PlanesTouched() const override { return kPlanePoints; }
    std::unique_ptr<UsdGenCapture> CreateCapture() const override
    {
        return std::make_unique<ProbeCapture>();
    }
    bool Capture(UsdGenCaptureContext const &ctx, UsdGenOpInputs const &in, UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override
    {
        if (_f != kNew) return UsdGenOp::Capture(ctx, in, out, diag);
        UsdGenCurveSetView const up = UsdGenCurveSetView::Of(in.Primary());
        auto &cap = static_cast<ProbeCapture &>(*out);
        cap.curves = up.totalCurves;
        cap.cvs = up.totalCvs;
        cap.sawBuffer = up.LegacyBuffer() != nullptr && in.count == 1;
        return true;
    }
    bool Capture(UsdGenCaptureContext const &ctx, UsdGenCurveBuffer const &up, UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override
    {
        if (_f == kNew || _f == kNeitherCapture) return UsdGenOp::Capture(ctx, up, out, diag);
        auto &cap = static_cast<ProbeCapture &>(*out);
        cap.curves = up.totalCurves;
        cap.cvs = up.totalCvs;
        cap.sawBuffer = true;
        return true;
    }
    void Evaluate(UsdGenEvalContext const &ctx, UsdGenCapture const &cap,
                  UsdGenChunkView const &chunk) const override
    {
        if (_f != kNew) { UsdGenOp::Evaluate(ctx, cap, chunk); return; }
        UsdGenCurveChunk const curve(chunk);
        Shift(curve);
    }
    void Evaluate(UsdGenEvalContext const &ctx, UsdGenCapture const &cap,
                  UsdGenLegacyChunkView *view) const override
    {
        if (_f == kNew || _f == kNeitherEvaluate) { UsdGenOp::Evaluate(ctx, cap, view); return; }
        Shift(*view);
    }

private:
    Family _f;
};

void RegisterProbes(UsdGenRegistrar &r)
{
    for (intptr_t f = 0; f < kFamilies; ++f) {
        UsdGenOpDef d;
        d.name = kTypes[f];
        d.displayName = kTypes[f] + 6;
        d.create = [](UsdGenOpRecord const &rec, void *context) -> std::unique_ptr<UsdGenOp> {
            return std::make_unique<Probe>(rec, Family(reinterpret_cast<intptr_t>(context)));
        };
        d.createContext = reinterpret_cast<void *>(f);
        r.RegisterOp(d);
    }
}

UsdGenGraphDesc MakeDesc(char const *probeType)
{
    constexpr int N = 8;
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/probe");
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.restPoints = VtVec3fArray((N + 1) * (N + 1));
    s.uv = VtVec2fArray((N + 1) * (N + 1));
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i <= N; ++i) {
            int const k = j * (N + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / N, float(j) / N);
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

bool HasText(std::vector<std::string> const &messages, char const *text)
{
    for (std::string const &m : messages)
        if (m.find(text) != std::string::npos) return true;
    return false;
}

struct Cooked
{
    UsdGenGraph graph;
    UsdGenRunResult result;
    bool compiled = false;
};

void Cook(Family f, UsdGenScheduler &scheduler, Cooked *out)
{
    UsdGenCompiler compiler;
    out->compiled = compiler.Compile(MakeDesc(kTypes[f]), &out->graph).ok;
    if (!out->compiled) return;
    UsdGenEvalContext ctx;
    ctx.desc = &out->graph.Desc();
    out->result = scheduler.Run(out->graph, ctx, 1);
}

bool SameBits(VtFloatArray const &a, VtFloatArray const &b)
{
    return a.size() == b.size() && (a.empty() || !std::memcmp(a.cdata(), b.cdata(), a.size() * sizeof(float)));
}

void TestFamilies()
{
    UsdGenScheduler scheduler(4);
    Cooked cooked[kFamilies];
    for (intptr_t f = 0; f < kFamilies; ++f) Cook(Family(f), scheduler, &cooked[f]);
    for (intptr_t f = 0; f < kFamilies; ++f) Check(cooked[f].compiled, std::string(kTypes[f]) + " compiles");

    Cooked const &o = cooked[kOld], &n = cooked[kNew];
    Check(!o.result.diagnostics.HasErrors() && !n.result.diagnostics.HasErrors(),
          "both families cook (verify on)");
    UsdGenNodeId const probe = o.graph.TerminalNodeId();
    UsdGenCurveBuffer const &grow = o.graph.Node(o.graph.Node(probe).input).buffer;
    UsdGenCurveBuffer const &a = o.graph.Output(), &b = n.graph.Output();
    Check(a.totalCvs > 0 && a.totalCvs == grow.totalCvs, "the old-family probe wrote every CV");
    Check(SameBits(a.px, b.px) && SameBits(a.py, b.py) && SameBits(a.pz, b.pz),
          "new-family output is bitwise the old-family output");
    bool shifted = a.px.size() == grow.px.size();
    for (size_t i = 0; shifted && i < a.px.size(); ++i)
        shifted = a.px[i] == grow.px[i] + 1.0f && a.py[i] == grow.py[i] + 2.0f && a.pz[i] == grow.pz[i] + 3.0f;
    Check(shifted, "the probe moved every CV by (1, 2, 3)");
    auto const *nc = dynamic_cast<ProbeCapture const *>(n.graph.Node(n.graph.TerminalNodeId()).capture.get());
    Check(nc && nc->curves == grow.totalCurves && nc->cvs == grow.totalCvs && nc->sawBuffer,
          "the new Capture saw the input as a buffer-backed view");

    Check(cooked[kNeitherCapture].result.diagnostics.HasErrors() &&
              HasText(cooked[kNeitherCapture].result.diagnostics.errors, "neither Capture family"),
          "overriding neither Capture family fails the capture");
    Check(cooked[kNeitherEvaluate].result.diagnostics.HasErrors() &&
              HasText(cooked[kNeitherEvaluate].result.diagnostics.errors, "neither Evaluate family"),
          "overriding neither Evaluate family fails the cook");

    // Old entry points on the new-family op: the defaults forward.
    UsdGenOp &op = *n.graph.Node(n.graph.TerminalNodeId()).op;
    UsdGenCaptureContext cctx;
    cctx.desc = &n.graph.Desc();
    UsdGenDiagnostics diag;
    std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
    bool const captured = op.Capture(cctx, grow, cap.get(), &diag);
    auto const &pc = static_cast<ProbeCapture const &>(*cap);
    Check(captured && pc.curves == grow.totalCurves && pc.cvs == grow.totalCvs && pc.sawBuffer,
          "old Capture reaches the new family over a view of the buffer");

    uint32_t const cvp = grow.totalCurves ? grow.totalCvs / grow.totalCurves : 0;
    std::vector<float> x(grow.totalCvs), y(grow.totalCvs), z(grow.totalCvs);
    UsdGenChunkDesc desc;
    desc.curveCount = desc.liveCount = grow.totalCurves;
    desc.cvCount = cvp;
    UsdGenLegacyChunkView view{};
    view.desc = &desc;
    view.px = x.data(); view.py = y.data(); view.pz = z.data();
    view.inPx = grow.px.cdata(); view.inPy = grow.py.cdata(); view.inPz = grow.pz.cdata();
    view.curveCount = grow.totalCurves;
    view.cvCount = view.inCvCount = cvp;
    UsdGenEvalContext ectx;
    op.Evaluate(ectx, *cap, &view);
    bool same = x.size() == a.px.size();
    for (size_t i = 0; same && i < x.size(); ++i) same = x[i] == a.px[i] && y[i] == a.py[i] && z[i] == a.pz[i];
    Check(same, "old Evaluate reaches the new family through a wrapper view");
}

void TestBufferSetView()
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenCurveBuffer b;
    b.totalCurves = 2;
    b.totalCvs = 5;
    b.px = VtFloatArray(5, 1.0f); b.py = VtFloatArray(5, 2.0f); b.pz = VtFloatArray(5, 3.0f);
    b.rest = VtVec3fArray(5, GfVec3f(0.5f));
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
    Check(v.channels && v.count[1] == 2 && v.count[2] == 5 && v.legacy == &b &&
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
    Check(cs.LegacyBuffer() == &b && cs.ExtraCurveCount() == 1 && cs.ExtraCvCount() == 1 &&
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
    TfSetenv("USDGEN_BRIDGE_VERIFY", "1");
    Check(UsdGenRegisterStatic(&RegisterProbes), "probes register before the first compile");
    TestFamilies();
    TestBufferSetView();
    std::printf("testUsdGenOpBridge: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
