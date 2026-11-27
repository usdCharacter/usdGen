// testUsdGenCurveRagged — UsdGenCurveSourceOp resampleTo fork (plan
// 04-operators.md §2.4, resampleTo row) through the REAL pipeline.
//
// FIX-4 (review): the chunk-descriptor assertions need real chunks, so both
// modes are driven through UsdGenCompiler + UsdGenScheduler::Run with the
// graph harness using source -> identity Width, not a bare op.Capture() call.
// Deform is CUDA-only and cannot serve as a CPU pass-through fixture.
//
//   * resampleTo == 0 (default): source CV counts PRESERVED ragged —
//     cvOffsets = {0,2,8,12} (size nCurves+1, [n] == totalCvs == 12),
//     chunk cvCount == 0 on every Width chunk, CVs verbatim on the source
//     node's own buffer.
//   * resampleTo == 4: uniform — cvOffsets stays EMPTY, every Width chunk
//     carries cvCount == 4 == totalCvs/totalCurves, interiors interpolated.
//
// Tier T0: engine core only, no stage, no Hydra (gate B-1). Standalone
// main(): PASS/FAIL + exit code.

#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "nodeSets.h"
#include "usdGen/compile/graph.h"
#include "usdGenUsd/lower.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/compile/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <cstdio>
#include <limits>
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

// Source surface: 3 curves with counts [2, 6, 4] (totalCvs == 12); curve c
// runs along +x from (0,c,0) to (count-1,c,0). resampleTo > 0 is authored on
// the source node (prefix-stripped param name per C1, FIX-2).
UsdGenGraphDesc MakeRaggedCurveDesc(int resampleTo, float defaultWidth = 0.01f)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/curveRagged");
    d.terminal    = SdfPath("/curveRagged/width");
    d.time        = 0.0;
    d.defaultWidth = defaultWidth;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/curveRagged/sourceCurves");
    s.id   = 0;
    s.faceVertexCounts  = VtIntArray({2, 6, 4});
    s.faceVertexIndices = VtIntArray(12);
    s.restPoints        = usdGen::UsdGenVec3fArray(12);
    s.points            = usdGen::UsdGenVec3fArray(12);
    size_t k = 0;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < s.faceVertexCounts[c]; ++i, ++k) {
            s.faceVertexIndices[k] = int(k);
            s.restPoints[k] = usdGen::UsdGenVec3f(float(i), float(c), 0.0f);
            s.points[k]     = s.restPoints[k];
        }
    d.surfaces.push_back(std::move(s));

    UsdGenNodeDesc src;
    src.path    = SdfPath("/curveRagged/src");
    src.type    = TfToken("UsdGenCurveSource");
    src.enabled = true;
    src.surfaces.push_back(SdfPath("/curveRagged/sourceCurves"));
    if (resampleTo > 0) {
        UsdGenParamValue p;
        p.name  = TfToken("resampleTo");   // C1: usdGen: prefix stripped
        p.value = VtValue(resampleTo);
        src.params.push_back(p);
    }
    d.nodes.push_back(std::move(src));

    UsdGenNodeDesc def;
    def.path    = SdfPath("/curveRagged/width");
    def.type    = TfToken("UsdGenWidth");
    def.params = {{TfToken("width"), VtValue(1.0f), false},
                  {TfToken("replace"), VtValue(false), false}};
    def.enabled = true;
    def.inputs.push_back(SdfPath("/curveRagged/src"));
    d.nodes.push_back(std::move(def));
    return d;
}

// Compile + one Run; leaves graph/params alive for the caller's asserts.
struct Harness {
    UsdGenGraphDesc desc;
    UsdGenCompiler  compiler;
    UsdGenGraph     graph;

    explicit Harness(int resampleTo, float defaultWidth = 0.01f)
        : desc(MakeRaggedCurveDesc(resampleTo, defaultWidth))
    {
        UsdGenCompileResult const cr = usdGenUsd::Compile(compiler, desc, &graph);
        if (!cr.ok) {
            std::printf("compile failed:");
            for (auto const &e : cr.errors) std::printf(" %s", e.c_str());
            std::printf("\n");
            ++g_failures;
            return;
        }
        UsdGenScheduler scheduler(8);
        UsdGenEvalContext ctx;
        ctx.desc = &graph.Spec();
        ctx.time = desc.time;
        UsdGenRunResult const run = scheduler.Run(graph, ctx, 1);
        if (run.diagnostics.HasErrors()) {
            std::printf("run diagnostics:");
            for (auto const &e : run.diagnostics.errors) std::printf(" %s", e.c_str());
            std::printf("\n");
            ++g_failures;
        }
    }

    UsdGenCurveBuffer Terminal() const { return OutputBuffer(graph); }
    UsdGenCurveBuffer Source() const {
        return NodeBuffer(graph, usdGenUsd::NodeIdForPath(graph, SdfPath("/curveRagged/src")));
    }
    UsdGenRef<const UsdGenElementSet> SourceSet() const {
        return UsdGenRef<const UsdGenElementSet>(
            &graph.NodeOutput(usdGenUsd::NodeIdForPath(graph, SdfPath("/curveRagged/src"))));
    }
    UsdGenSpan<const UsdGenChunkRange> TerminalChunks() const {
        return graph.Chunks(usdGenUsd::NodeIdForPath(graph, desc.terminal));
    }
};

}  // namespace

int main()
{

    // ---- mode 1: resampleTo == 0 (default) — ragged through the pipeline --
    {
        Harness h(0);
        UsdGenCurveBuffer const &out = h.Terminal();
        Check(out.totalCurves == 3, "ragged: terminal totalCurves == 3");
        Check(out.totalCvs == 12, "ragged: terminal totalCvs == 12");
        VtIntArray want{0, 2, 8, 12};
        Check(h.Source().cvOffsets == want,
              "ragged: source cvOffsets == {0,2,8,12}");
        Check(h.Source().cvOffsets.size() == 4 &&
              h.Source().cvOffsets.back() == int(h.Source().totalCvs),
              "ragged: cvOffsets[n] == totalCvs");
        Check(out.cvOffsets == want,
              "ragged: cvOffsets preserved through Width");

        auto const chunks = h.TerminalChunks();
        Check(!chunks.empty(), "ragged: Width has real chunk descriptors");
        Check(std::all_of(chunks.begin(), chunks.end(),
                          [](UsdGenChunkRange const &cd) { return cd.stride[1] == 0; }),
              "ragged: every Width chunk cvCount == 0");
        int covered = 0;
        for (UsdGenChunkRange const &cd : chunks) covered += int(cd.count[0]);
        Check(covered == 3, "ragged: chunk curve spans cover all 3 curves");

        // Source-node buffer carries the CVs verbatim (ragged, counts [2,6,4]).
        UsdGenCurveBuffer const &src = h.Source();
        Check(src.px.size() == 12 && src.px[0] == 0.0f && src.px[1] == 1.0f &&
              src.px[2] == 0.0f && src.py[2] == 1.0f &&   // curve 1 verbatim
              src.px[7] == 5.0f && src.px[8] == 0.0f && src.py[8] == 2.0f,
              "ragged: CVs verbatim, source counts [2,6,4] preserved");
    }

    // ---- mode 2: resampleTo == 4 — uniform, cvOffsets stays empty ---------
    {
        Harness h(4);
        UsdGenCurveBuffer const &out = h.Terminal();
        UsdGenCurveBuffer const &src = h.Source();
        Check(src.cvOffsets.empty(), "uniform: source cvOffsets stays empty");
        Check(out.cvOffsets.empty(), "uniform: terminal cvOffsets stays empty");
        Check(src.totalCurves == 3 && src.totalCvs == 12,
              "uniform: source totalCvs == 3 * resampleTo == 12");
        Check(out.totalCurves == 3 && out.totalCvs == 12 &&
              out.totalCvs / out.totalCurves == 4,
              "uniform: terminal cvCount == totalCvs/totalCurves == 4");

        auto const chunks = h.TerminalChunks();
        Check(!chunks.empty() &&
              std::all_of(chunks.begin(), chunks.end(),
                          [](UsdGenChunkRange const &cd) { return cd.stride[1] == 4; }),
              "uniform: every Width chunk carries cvCount == 4");

        // curve 0 = 2 source CVs resampled to 4 along the polyline: endpoints
        // exact, interior strictly interpolated.
        Check(src.px.size() == 12 && src.px[0] == 0.0f && src.px[1] > 0.3f &&
              src.px[1] < 0.7f && src.px[3] == 1.0f && src.py[3] == 0.0f,
              "uniform: curve0 [0,1] resampled to 4, interiors interpolated");
    }

    // The CPU surface-source path has no C3 authored-width payload; its
    // missing-width contract is the description fallback.  Recompile through
    // the real graph preserves the old VtArray owner while the changed capture
    // receives a fresh private COW width plane.
    {
        Harness h(0, 0.25f);
        UsdGenCurveBuffer const &initial = h.Source();
        UsdGenRef<const UsdGenElementSet> const initialSet = h.SourceSet();
        UsdGenSpan<const float> const oldWidths = UsdGenCurveSetView::Of(*initialSet).width;
        Check(initial.width == VtFloatArray(12, 0.25f) &&
                  h.Terminal().width == VtFloatArray(12, 0.25f),
              "default width: surface source and terminal use description fallback");

        h.desc.defaultWidth = 0.5f;
        UsdGenCompileResult const recomp = usdGenUsd::Recompile(h.compiler, h.desc, &h.graph);
        Check(recomp.ok, "default width: changed fallback recompiles");
        UsdGenScheduler scheduler(8);
        UsdGenEvalContext ctx;
        ctx.desc = &h.graph.Spec();
        ctx.time = h.desc.time;
        UsdGenRunResult const run = scheduler.Run(h.graph, ctx, 2);
        Check(!run.diagnostics.HasErrors(),
              "default width: changed fallback recaptures cleanly");
        UsdGenCurveBuffer const &updated = h.Source();
        Check(VtFloatArray(oldWidths.begin(), oldWidths.end()) == VtFloatArray(12, 0.25f) &&
                  updated.width == VtFloatArray(12, 0.5f) &&
                  oldWidths.cdata() != UsdGenCurveSetView::Of(*h.SourceSet()).width.cdata(),
              "default width: recapture keeps immutable predecessor COW owner");

        for (float invalidWidth : {-1.0f,
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity()}) {
            h.desc.defaultWidth = invalidWidth;
            UsdGenCompileResult const invalid = usdGenUsd::Recompile(h.compiler, h.desc, &h.graph);
            Check(invalid.ok, "default width: invalid value reaches capture validation");
            ctx.desc = &h.graph.Spec();
            UsdGenRunResult const rejected = scheduler.Run(h.graph, ctx, 3);
            Check(rejected.diagnostics.HasErrors(),
                  "default width: non-finite/negative fallback is rejected");
            Check(h.Source().width == VtFloatArray(12, 0.5f),
                  "default width: rejected capture does not mutate prior COW buffer");
        }
    }

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
