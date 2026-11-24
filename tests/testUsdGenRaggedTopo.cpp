// testUsdGenRaggedTopo — ragged (mixed CV-count) topology through the real
// pipeline (plan/03-execution-engine.md §1.3 ragged chunk contract, §3.2
// per-node buffers/value versions).
//
// Graph: UsdGenCurveSource (usdGen:resampleTo == 0 -> keep source counts, so
// the buffer is ragged: cvCount == 0 + cvOffsets) feeding ONE position-only
// reader (UsdGenDeform, the registered P1 ragged pair — it reads/writes px,py,
// pz over the chunk CV span and ignores uniform cvCount; the scheduler's
// SweepChunk hands it view.cvOffsets = buf.cvOffsets + firstCurve).
//
// Topology: 3 curves with vertex counts {2,6,4} => cvOffsets {0,2,8,12},
// totalCvs == 12, cvCount == 0 on every chunk.
//
// Asserted (engine-only, no Hydra, no stage — gate B-1 style):
//   1. Compile + first Run complete with no diagnostics errors (no crash).
//   2. Terminal buffer: totalCurves == 3, totalCvs == 12,
//      cvOffsets == {0,2,8,12} preserved through the pipeline (source ->
//      deform carries upstream cvOffsets verbatim, scheduler owns-guard).
//   3. Every deform chunk is ragged: cvCount == 0, firstCv/firstCurve match
//      the cvOffsets layout (Repartition rebuilds these per layout, §1.3).
//   4. Position planes are sized to 12 CVs on the output.
//   5. Re-Run with nothing dirty cooks ZERO chunks (value-digest skip,
//      §3.2/§5.4 step 4 — no re-sweep) and the output is bit-identical.
//
// Tier T0/T1 list: standalone main(), links usdGen. No CMake registration
// here (out of scope for this ticket).

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

// CurveSource::Capture (04 §2.4, resampleTo == 0) derives the ragged layout
// from the bound source's per-curve vertex counts: cvOffsets[c+1] =
// cvOffsets[c] + faceVertexCounts[c]. So a 3-curve ragged source is a
// surface desc with faceVertexCounts {2,6,4} and 12 points.
UsdGenGraphDesc MakeRaggedDesc()
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/ragged");
    d.terminal    = SdfPath("/ragged/deform");
    d.time        = 0.0;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/ragged/sourceCurves");
    s.id   = 0;
    s.faceVertexCounts = VtIntArray({2, 6, 4});   // 3 curves, ragged, 12 CVs
    s.faceVertexIndices = VtIntArray(12);
    s.restPoints = VtVec3fArray(12);
    s.points     = VtVec3fArray(12);
    for (int k = 0; k < 12; ++k) {
        s.faceVertexIndices[k] = k;
        s.restPoints[k] = GfVec3f(float(k) * 0.1f, float(k % 3) * 0.2f, 0.0f);
        s.points[k]     = s.restPoints[k];
    }
    d.surfaces.push_back(std::move(s));

    UsdGenNodeDesc src;
    src.path    = SdfPath("/ragged/src");
    src.type    = TfToken("UsdGenCurveSource");
    src.enabled = true;
    src.surfaces.push_back(SdfPath("/ragged/sourceCurves"));
    d.nodes.push_back(std::move(src));

    // Position-only reader: Deform touches px/py/pz, no CV-count arithmetic
    // on its own (ragged spans come from view.cvOffsets).
    UsdGenNodeDesc def;
    def.path    = SdfPath("/ragged/deform");
    def.type    = TfToken("UsdGenDeform");
    def.enabled = true;
    def.inputs.push_back(SdfPath("/ragged/src"));
    d.nodes.push_back(std::move(def));
    return d;
}

uint64_t TotalChunksEvaluated(UsdGenRunResult const &r)
{
    uint64_t n = 0;
    for (UsdGenNodeRunStats const &st : r.nodeStats) n += st.chunksEvaluated;
    return n;
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();

    UsdGenGraphDesc const desc = MakeRaggedDesc();
    UsdGenCompiler  compiler;
    UsdGenGraph     graph;
    UsdGenCompileResult const cr = compiler.Compile(desc, &graph);
    if (!cr.ok) {
        std::printf("compile failed:");
        for (auto const &e : cr.errors) std::printf(" %s", e.c_str());
        std::printf("\n");
        std::printf("RaggedTopo FAIL\n");
        return 1;
    }
    Check(graph.NodeCount() == 2, "2-node ragged graph compiled");

    UsdGenScheduler scheduler(8);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    ctx.time = desc.time;

    // ---- first eval: full run over the ragged topology -------------------
    UsdGenRunResult const run1 = scheduler.Run(graph, ctx, 1);
    if (run1.diagnostics.HasErrors()) {
        std::printf("run1 diagnostics:");
        for (auto const &e : run1.diagnostics.errors) std::printf(" %s", e.c_str());
        std::printf("\n");
    }
    Check(!run1.diagnostics.HasErrors(), "first eval ran without errors (no crash)");

    UsdGenCurveBuffer const *out = graph.Output().totalCvs ? &graph.Output() : run1.terminalOutput;
    Check(out != nullptr, "terminal output buffer published");
    if (!out) { std::printf("RaggedTopo FAIL\n"); return 1; }

    Check(out->totalCurves == 3, "terminal totalCurves == 3");
    Check(out->totalCvs == 12, "terminal totalCvs == 12");
    Check(out->cvOffsets.size() == 4 &&
          out->cvOffsets[0] == 0 && out->cvOffsets[1] == 2 &&
          out->cvOffsets[2] == 8 && out->cvOffsets[3] == 12,
          "cvOffsets {0,2,8,12} preserved through the pipeline");
    Check(out->px.size() == 12 && out->py.size() == 12 && out->pz.size() == 12,
          "position planes sized to 12 CVs");

    UsdGenNodeId const deform = graph.NodeIdForPath(SdfPath("/ragged/deform"));
    bool chunksRagged = !graph.Chunks(deform).empty();
    int  covered = 0;
    for (UsdGenChunkDesc const &cd : graph.Chunks(deform)) {
        if (cd.cvCount != 0) chunksRagged = false;
        covered += int(cd.curveCount);
    }
    Check(chunksRagged, "every deform chunk carries cvCount == 0 (ragged)");
    Check(covered == 3, "chunk curve spans cover all 3 curves");

    // ---- second eval: nothing dirty must skip the re-sweep entirely -------
    std::vector<float> px1(out->px.cbegin(), out->px.cend());
    std::vector<float> py1(out->py.cbegin(), out->py.cend());
    std::vector<float> pz1(out->pz.cbegin(), out->pz.cend());
    UsdGenRunResult const run2 = scheduler.Run(graph, ctx, 2);
    Check(TotalChunksEvaluated(run2) == 0,
          "clean re-run cooks zero chunks (no re-sweep, 03 §3.2/§5.4)");
    Check(!graph.AnyDirty(), "graph still clean after the second eval");
    Check(std::vector<float>(graph.Output().px.cbegin(), graph.Output().px.cend()) == px1 &&
          std::vector<float>(graph.Output().py.cbegin(), graph.Output().py.cend()) == py1 &&
          std::vector<float>(graph.Output().pz.cbegin(), graph.Output().pz.cend()) == pz1,
          "second eval output is idempotent (bit-identical positions, vs snapshots)");

    if (g_failures) std::printf("RaggedTopo FAIL\n");
    else            std::printf("RaggedTopo PASS\n");
    return g_failures ? 1 : 0;
}
