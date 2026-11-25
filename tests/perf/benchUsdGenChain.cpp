// benchUsdGenChain — timing + bit-identity gates E-1, E-6, E-7, E-8
// (plan/09 §5.1). Exit is nonzero on bitwise mismatch ALWAYS, and on timing
// FAIL only when USDGEN_GATE=1 (CI must not flake on absolute timings).
//   E-1 full-dirty commit, G3 @8 threads   : median <= 1.5 ms (9 runs)
//   E-6 chain-200 append Recompile @8      : median <= 0.2 ms, rebuilds == 1
//   E-7 E-1 scaling, 1 vs 8 threads        : median1 / median8 >= 3x
//   E-8 1-thread vs 8-thread terminal      : bitwise identical (ALWAYS hard)
// E-2 lives in benchUsdGenSparse (same 1%-chunk form).

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"
#include "usdGenCurves/types/curveSet.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

int g_timingFail = 0;
int g_hardFail = 0;

bool GateMode()
{
    char const *g = std::getenv("USDGEN_GATE");
    return g && std::string(g) == "1";
}

double Median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

void ReportTiming(char const *gate, double value, double limit, bool lowerBetter)
{
    bool const pass = lowerBetter ? value <= limit : value >= limit;
    std::printf("%s: %.3f (limit %.3f) -> %s%s\n", gate, value, limit,
                pass ? "PASS" : "FAIL", pass ? "" : (GateMode() ? " [hard]" : " [info]"));
    if (!pass) ++g_timingFail;
}

// G3 @100k (same shape as benchUsdGenSparse / 07 §2).
UsdGenGraphDesc MakeG3()
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    const int NX = 400, NY = 250;
    s.restPoints = VtVec3fArray((NX + 1) * (NY + 1));
    s.uv = VtVec2fArray((NX + 1) * (NY + 1));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            const int k = j * (NX + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
        }
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            const int a = j * (NX + 1) + i;
            const int o = (j * NX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    d.surfaces.push_back(std::move(s));

    auto addNode = [&](std::string const &name, TfToken type,
                       std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = type;
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (type == TfToken("UsdGenScatter"))
            n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    addNode("scatter", TfToken("UsdGenScatter"), "", 42);
    addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
    addNode("noise", TfToken("UsdGenNoise"), "grow", 44);
    addNode("length", TfToken("UsdGenLength"), "noise", 45);
    addNode("width", TfToken("UsdGenWidth"), "length", 46);

    auto setp = [&](std::string const &name, TfToken p, VtValue v) {
        for (auto &n : d.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{p, v, false});
    };
    setp("grow", TfToken("segments"), VtValue(8));
    setp("grow", TfToken("length"), VtValue(1.0));
    setp("noise", TfToken("noise:magnitude"), VtValue(0.05));
    setp("noise", TfToken("noise:frequency"), VtValue(3.0));
    setp("noise", TfToken("noise:octaves"), VtValue(2));
    setp("noise", TfToken("noise:correlation"), VtValue(0.5));
    setp("length", TfToken("length:mode"), VtValue(TfToken("scale")));
    setp("length", TfToken("length:value"), VtValue(1.2));
    setp("width", TfToken("width"), VtValue(0.02));
    return d;
}
// n-node chain (same form as testUsdGenGraph).
UsdGenGraphDesc MakeChain(int n)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/chain");
    d.time = 0.0;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/chain/surface");
    s.id = 0;
    s.restPoints = VtVec3fArray(4, GfVec3f(0.f, 0.f, 0.f));
    s.uv = VtVec2fArray(4, GfVec2f(0.f, 0.f));
    s.faceVertexCounts = VtIntArray(1, 4);
    s.faceVertexIndices = VtIntArray({0, 1, 2, 3});
    d.surfaces.push_back(std::move(s));
    for (int i = 0; i < n; ++i) {
        UsdGenNodeDesc nd;
        nd.path = SdfPath("/chain/n" + std::to_string(i));
        nd.type = i == 0 ? TfToken("UsdGenScatter") : TfToken("UsdGenGrow");
        nd.enabled = true;
        nd.seed = 2000 + i;
        if (i == 0) nd.surfaces.push_back(SdfPath("/chain/surface"));
        else nd.inputs.push_back(SdfPath("/chain/n" + std::to_string(i - 1)));
        d.nodes.push_back(std::move(nd));
    }
    d.terminal = d.nodes.back().path;
    return d;
}

// Force every chunk of every node fully value-dirty (the E-1 worst case).
void ForceFullDirty(UsdGenGraph &g)
{
    for (int i = 0; i < g.NodeCount(); ++i)
        for (UsdGenChunkDirtyBits &byte : g.Node(i).chunkDirty)
            byte = UsdGenDirtyParameter;
}

template <class A>
bool SameArray(A const &a, A const &b)
{
    if (a.size() != b.size()) return false;
    return a.empty() ||
           std::memcmp(&a[0], &b[0], a.size() * sizeof(typename A::value_type)) == 0;
}

}  // namespace

int main(int argc, char **argv)
{
   if (argc > 1 && std::string(argv[1]) == "--ragged") {
        std::printf("ragged: UNMEASURED (engine ops pending)\\n");
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--perop") {
        // Per-node capture/eval split of one full-dirty G3 commit @8 threads
        // (reads UsdGenRunResult::nodeStats; feeds the E-1 breakdown).
        UsdGenCompiler c;
        UsdGenGraph g;
        if (!c.Compile(MakeG3(), &g).ok) {
            std::printf("perop: compile failed\n");
            return 1;
        }
        UsdGenEvalContext ctx;
        ctx.desc = &g.Desc();
        UsdGenScheduler s8(8);
        uint64_t gen = 0;
        s8.Run(g, ctx, ++gen);
        for (int i = 0; i < 3; ++i) {
            ForceFullDirty(g);
            UsdGenRunResult const r = s8.Run(g, ctx, ++gen);
            for (UsdGenNodeRunStats const &st : r.nodeStats)
                std::printf("perop run=%d node=%u type=%s captureMs=%.3f evalMs=%.3f chunks=%llu\n",
                            i, st.id, g.Node(st.id).record->Name().GetText(),
                            st.captureMs, st.evalMs,
                            (unsigned long long)st.chunksEvaluated);
        }
        return 0;
    }

    UsdGenEvalContext ctx;

    // ---- E-1 + E-7: full-dirty re-commit, 8 vs 1 threads ----------------------
    {
        UsdGenCompiler c;
        UsdGenGraph g;
        UsdGenCompileResult const r = c.Compile(MakeG3(), &g);
        if (!r.ok) {
            std::printf("E-1: compile failed:");
            for (auto const &e : r.errors) std::printf(" %s", e.c_str());
            std::printf("\n");
            return 1;
        }
        ctx.desc = &g.Desc();
        UsdGenScheduler s8(8), s1(1);
        uint64_t gen = 0;
        s8.Run(g, ctx, ++gen);

        std::vector<double> t8, t1;
        for (int i = 0; i < 9; ++i) {
            ForceFullDirty(g);
            auto t0 = std::chrono::steady_clock::now();
            s8.Run(g, ctx, ++gen);
            t8.push_back(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
        }
        for (int i = 0; i < 9; ++i) {
            ForceFullDirty(g);
            auto t0 = std::chrono::steady_clock::now();
            s1.Run(g, ctx, ++gen);
            t1.push_back(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
        }
        ReportTiming("E-1 (G3 full-dirty commit, ms, 8 threads)", Median(t8), 1.5, true);
        ReportTiming("E-7 (1-thread / 8-thread scaling ratio)",
                     Median(t1) / Median(t8), 3.0, false);
    }

    // ---- E-6: 200-node chain, 1 node appended -> Recompile --------------------
    {
        UsdGenGraphDesc base = MakeChain(200);
        UsdGenGraphDesc grown = base;
        {
            UsdGenNodeDesc nd;
            nd.path = SdfPath("/chain/n200");
            nd.type = TfToken("UsdGenScatter");   // Clone() is scatter-only in M1
            nd.enabled = true;
            nd.seed = 2200;
            nd.surfaces.push_back(base.surfaces[0].path);
            grown.nodes.push_back(std::move(nd));
            grown.terminal = grown.nodes.back().path;
        }
        UsdGenCompiler c;
        UsdGenGraph g;
        UsdGenCompileResult const r = c.Compile(base, &g);
        if (!r.ok) {
            std::printf("E-6: compile failed:");
            for (auto const &e : r.errors) std::printf(" %s", e.c_str());
            std::printf("\n");
            return 1;
        }
        std::vector<double> ts;
        bool rebuiltOnce = true;
        for (int i = 0; i < 9; ++i) {
            c.Recompile(base, &g);   // back to 200 nodes (untimed half)
            auto t0 = std::chrono::steady_clock::now();
            UsdGenCompileResult const rr = c.Recompile(grown, &g);
            ts.push_back(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
            rebuiltOnce = rebuiltOnce && rr.ok && rr.rebuilt.size() == 1;
        }
        if (!rebuiltOnce) ++g_hardFail;
        std::printf("E-6 rebuild-exactly-1: %s\n", rebuiltOnce ? "PASS" : "FAIL");
        ReportTiming("E-6 (chain-200 append Recompile, ms)", Median(ts), 0.2, true);
    }

    // ---- E-8: bitwise 1-thread vs 8-thread ------------------------------------
    {
        UsdGenScheduler s8(8), s1(1);
        UsdGenGraph gA, gB;
        UsdGenCompiler cA, cB;
        if (!cA.Compile(MakeG3(), &gA).ok || !cB.Compile(MakeG3(), &gB).ok) {
            std::printf("E-8: compile failed\n");
            return 1;
        }
        UsdGenEvalContext ctxA;
        ctxA.desc = &gA.Desc();
        UsdGenEvalContext ctxB;
        ctxB.desc = &gB.Desc();
        UsdGenRunResult rA = s1.Run(gA, ctxA, 1);
        UsdGenRunResult rB = s8.Run(gB, ctxB, 1);
        UsdGenCurveBuffer const a = UsdGenCurveBufferFromSet(*rA.terminalOutput);
        UsdGenCurveBuffer const b = UsdGenCurveBufferFromSet(*rB.terminalOutput);
        bool same = a.totalCurves == b.totalCurves && a.totalCvs == b.totalCvs &&
                    SameArray(a.px, b.px) && SameArray(a.py, b.py) &&
                    SameArray(a.pz, b.pz) && SameArray(a.width, b.width) &&
                    SameArray(a.hairT, b.hairT) && SameArray(a.curveId, b.curveId) &&
                    SameArray(a.rootPrim, b.rootPrim) &&
                    rA.tiles.size() == rB.tiles.size();
        for (size_t i = 0; same && i < rA.tiles.size(); ++i)
            same = rA.tiles[i].totalLiveCurves == rB.tiles[i].totalLiveCurves &&
                   rA.tiles[i].totalLiveCvs == rB.tiles[i].totalLiveCvs;
        std::printf("E-8 (1 vs 8 threads bitwise): %s\n", same ? "PASS" : "FAIL");
        if (!same) ++g_hardFail;
    }

    int const hard = g_hardFail ? 1 : ((g_timingFail && GateMode()) ? 1 : 0);
    std::printf("hardFail=%d timingFail=%d gateMode=%d -> exit %d\n", g_hardFail,
                g_timingFail, GateMode() ? 1 : 0, hard);
    return hard;
}
