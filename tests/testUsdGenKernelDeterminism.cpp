// testUsdGenKernelDeterminism — gate E-8 (T0, plan 11 §2.2 / 09 §5.1):
//
//   Run the M1 5-op chain (Scatter random -> Grow -> Noise -> Length ->
//   Width) on the G3 fixture (100 k curves x 8 CV) at 1 thread and at
//   8 threads (each in its own private tbb::task_arena, I8) and
//   BITWISE-COMPARE every output plane of every node plus the chunk and
//   tile partition state. Any difference is a failure (usdGenMath is
//   built -ffp-contract=off; the kernels must carry that guarantee).
//
// Tier T0: engine core only (no Hydra, no stage — gate B-1).

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

/// G3: 100000 root points on a 1000x100 quad grid (100 k faces), grown to
/// 8 CVs -> 100 k curves x 8 CV = 800 k CVs through 5 operators.
UsdGenGraphDesc MakeG3()
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    const int NX = 1000, NY = 100;
    const int np = (NX + 1) * (NY + 1);
    s.restPoints = VtVec3fArray(np);
    s.uv = VtVec2fArray(np);
    {
        auto *rp = s.restPoints.data();
        auto *uv = s.uv.data();
        for (int j = 0; j <= NY; ++j)
            for (int i = 0; i <= NX; ++i) {
                const int k = j * (NX + 1) + i;
                rp[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
                uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
            }
    }
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    {
        auto *fvi = s.faceVertexIndices.data();
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                const int a = j * (NX + 1) + i;
                const int o = (j * NX + i) * 4;
                fvi[o + 0] = a;
                fvi[o + 1] = a + 1;
                fvi[o + 2] = a + NX + 2;
                fvi[o + 3] = a + NX + 1;
            }
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
        std::string const target = "/groom/" + name;
        for (auto &n : d.nodes)
            if (n.path == SdfPath(target))
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

// ---- bitwise comparison ----------------------------------------------------

bool g_allOk = true;
void Report(bool ok, std::string what)
{
    if (!ok) {
        g_allOk = false;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

template <class A>
bool SameArray(A const &a, A const &b, std::string const &what)
{
    bool ok = a.size() == b.size();
    if (ok)
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) { ok = false; break; }
    Report(ok, what + (ok ? "" : " (size " + std::to_string(a.size()) +
                                 " vs " + std::to_string(b.size()) + ")"));
    return ok;
}

bool SameFloatArray(VtFloatArray const &a, VtFloatArray const &b,
                    std::string const &what)
{
    bool ok = a.size() == b.size();
    if (ok) {
        auto const *pa = a.cdata();
        auto const *pb = b.cdata();
        for (size_t i = 0; i < a.size(); ++i) {
            if (pa[i] != pb[i]) { ok = false; break; }  // float equality, no NaN games
        }
    }
    Report(ok, what + (ok ? "" : " (size " + std::to_string(a.size()) +
                                 " vs " + std::to_string(b.size()) + ")"));
    return ok;
}

bool CompareGraphs(UsdGenGraph const &a, UsdGenGraph const &b)
{
    bool ok = true;
    ok = (a.NodeCount() == b.NodeCount()) && ok;
    Report(a.NodeCount() == b.NodeCount(),
           std::to_string(a.NodeCount()) + " == " + std::to_string(b.NodeCount()) +
           " nodes");
    ok = (a.TerminalNodeId() == b.TerminalNodeId()) && ok;
    ok = (a.NumTiles() == b.NumTiles()) && ok;
    Report(a.NumTiles() == b.NumTiles(),
           std::to_string(a.NumTiles()) + " == " + std::to_string(b.NumTiles()) +
           " tiles");

    for (int i = 0; i < a.NodeCount(); ++i) {
        UsdGenCompiledNode const &na = a.Node(i);
        UsdGenCompiledNode const &nb = b.Node(i);
        std::string p = "node " + std::to_string(i) + " (" + na.record->Name().GetString() + ") ";

        ok = (na.buffer.totalCurves == nb.buffer.totalCurves) && ok;
        ok = (na.buffer.totalCvs == nb.buffer.totalCvs) && ok;
        ok = (na.buffer.topologyVersion == nb.buffer.topologyVersion) && ok;
        ok = (na.buffer.valueVersion == nb.buffer.valueVersion) && ok;
        Report(na.buffer.totalCurves == nb.buffer.totalCurves &&
                   na.buffer.totalCvs == nb.buffer.totalCvs &&
                   na.buffer.topologyVersion == nb.buffer.topologyVersion &&
                   na.buffer.valueVersion == nb.buffer.valueVersion,
               p + "topology/valueVersion (" +
               std::to_string(na.buffer.totalCurves) + " cv, " +
               std::to_string(na.buffer.totalCvs) + " cvs, topo " +
               std::to_string(na.buffer.topologyVersion) + ", val " +
               std::to_string(na.buffer.valueVersion) + ")");

        ok = SameFloatArray(na.buffer.px, nb.buffer.px, p + "px") && ok;
        ok = SameFloatArray(na.buffer.py, nb.buffer.py, p + "py") && ok;
        ok = SameFloatArray(na.buffer.pz, nb.buffer.pz, p + "pz") && ok;
        ok = SameFloatArray(na.buffer.width, nb.buffer.width, p + "widths") && ok;
        ok = SameFloatArray(na.buffer.hairT, nb.buffer.hairT, p + "hairT") && ok;
        ok = SameArray(na.buffer.curveId, nb.buffer.curveId, p + "curveId") && ok;
        ok = SameArray(na.buffer.rootPrim, nb.buffer.rootPrim, p + "rootPrim") && ok;
        ok = SameArray(na.buffer.rootUV, nb.buffer.rootUV, p + "rootUV") && ok;

        // chunk partition
        bool chunkOk = na.chunks.size() == nb.chunks.size();
        if (chunkOk)
            for (size_t c = 0; c < na.chunks.size(); ++c) {
                auto const &ca = na.chunks[c];
                auto const &cb = nb.chunks[c];
                if (ca.firstCurve != cb.firstCurve || ca.curveCount != cb.curveCount ||
                    ca.firstCv != cb.firstCv || ca.cvCount != cb.cvCount ||
                    ca.liveCount != cb.liveCount || ca.tile != cb.tile) {
                    chunkOk = false;
                    break;
                }
            }
        Report(chunkOk, p + "chunks (curveVertexCounts/partition)");
        ok = chunkOk && ok;
    }

    // tiles: extents + live counts
    for (int t = 0; t < a.NumTiles(); ++t) {
        auto const &ta = a.Tiles()[t];
        auto const &tb = b.Tiles()[t];
        bool okT = ta.totalLiveCurves == tb.totalLiveCurves &&
                   ta.totalLiveCvs == tb.totalLiveCvs &&
                   ta.extent.GetMin() == tb.extent.GetMin() &&
                   ta.extent.GetMax() == tb.extent.GetMax();
        Report(okT, "tile " + std::to_string(t) + " extent/liveCounts");
        ok = okT && ok;
    }
    return ok;
}

}  // namespace
}  // namespace usdGen

using namespace usdGen;

int main()
{

    auto runAt = [](int threads) -> UsdGenGraph {
        UsdGenGraphDesc d = MakeG3();
        UsdGenCompiler compiler;
        UsdGenGraph graph;
        UsdGenCompileResult const r = compiler.Compile(d, &graph);
        if (!r.ok) {
            std::printf("FAIL: compile at %d threads: ", threads);
            for (auto const &e : r.errors) std::printf("%s ", e.c_str());
            std::printf("\n");
            std::exit(1);
        }
        UsdGenScheduler scheduler(threads);
        UsdGenEvalContext ctx;
        ctx.desc = &graph.Desc();
        ctx.time = d.time;
        UsdGenRunResult const res = scheduler.Run(graph, ctx, /*gen=*/1);
        if (res.diagnostics.HasErrors()) {
            std::printf("FAIL: run diagnostics at %d threads:\n", threads);
            for (auto const &e : res.diagnostics.errors) std::printf("  %s\n", e.c_str());
            std::exit(1);
        }
        std::printf("run @ %d threads: terminal %u curves x %u CVs, %u chunks, %d tiles\n",
                    threads, res.terminalOutput->totalCurves,
                    res.terminalOutput->totalCvs,
                    static_cast<unsigned>(graph.Chunks(graph.TerminalNodeId()).size()),
                    graph.NumTiles());
        return graph;
    };

    UsdGenGraph g1 = runAt(1);
    UsdGenGraph g8 = runAt(8);

    std::printf("--- bitwise comparison (1-thread vs 8-thread) ---\n");
    bool const ok = usdGen::CompareGraphs(g1, g8);
    std::printf(ok ? "testUsdGenKernelDeterminism: PASS (E-8: bitwise identical)\n"
                   : "testUsdGenKernelDeterminism: FAILED (E-8: outputs differ)\n");
    return ok ? 0 : 1;
}
