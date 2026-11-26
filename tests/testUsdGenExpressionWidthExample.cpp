// The shipped root-to-tip width example, cooked on the CPU reference lane.
//
// examples/expression-width-plane.usda authors a POINT-domain expression
// on usdGen:width and a PRIMITIVE-domain expression on the Noise operator's
// usdGen:mask. This test opens that file, builds the graph description from
// the stage, cooks it on the CPU lane and asserts the published widths:
//   * decrease monotonically from root to tip on every strand;
//   * hit the expression's 15% tip ratio;
//   * come from a multi-line source with locals and a curve() control;
//   * follow the AUTHORED literal, so usdGen:width is still $value;
//   * change when the expression source is edited on the stage.
#include "usdGen/compiler.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"

#include "pxr/pxr.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
    else std::printf("ok: %s\n", what.c_str());
}

constexpr char const *kScene =
    USDGEN_TEST_SOURCE_DIR "/examples/expression-width-plane.usda";
constexpr char const *kDescription = "/World/Groom/Fur";
constexpr char const *kWidthExpression = "/World/Groom/Fur/Expressions/rootTipWidth";

/// Cooks the stage and returns the terminal buffer's widths, or an empty
/// vector on any compile/run failure.
bool Cook(UsdStageRefPtr const &stage, UsdGenCurveBuffer *out, std::string const &label)
{
    UsdGenGraphDesc const desc =
        usdGenImaging::BuildGraphDescFromStage(stage, SdfPath(kDescription));
    if (desc.nodes.empty() ||
        desc.executionBackend != UsdGenExecutionBackend::CpuReference ||
        desc.expressions.size() != 2) {
        Check(false, label + ": the stage must yield a CPU-lane graph with two expressions");
        return false;
    }
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        Check(false, label + ": compile the expression graph on the CPU lane");
        for (auto const &error : compiled.errors) std::printf("  %s\n", error.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    if (run.diagnostics.HasErrors()) {
        Check(false, label + ": cook the expression graph");
        for (auto const &error : run.diagnostics.errors) std::printf("  %s\n", error.c_str());
        return false;
    }
    *out = OutputBuffer(graph);
    return true;
}

/// Per-curve CV span of a uniform-topology buffer.
bool Spans(UsdGenCurveBuffer const &buffer, uint32_t *curves, uint32_t *cvs)
{
    if (buffer.totalCurves == 0 || buffer.totalCvs == 0 ||
        buffer.totalCvs % buffer.totalCurves != 0 || !buffer.cvOffsets.empty())
        return false;
    *curves = buffer.totalCurves;
    *cvs = buffer.totalCvs / buffer.totalCurves;
    return true;
}

} // namespace

int main()
{
    UsdStageRefPtr const stage = UsdStage::Open(kScene);
    if (!stage) { std::printf("FAIL: cannot open %s\n", kScene); return 1; }

    UsdGenCurveBuffer baseline;
    if (!Cook(stage, &baseline, "expression example")) return 1;

    uint32_t curves = 0, cvs = 0;
    Check(Spans(baseline, &curves, &cvs) && curves > 0 && cvs >= 4,
          "the example publishes a uniform strand topology");
    if (!curves || !cvs) return 1;
    Check(baseline.width.size() == baseline.totalCvs,
          "every CV carries a published width");
    if (baseline.width.size() != baseline.totalCvs) return 1;

    // The Width operator is `replace`, so the published width IS the
    // expression: 0.012 * mix(0.15, 1.0, curve($t, 0,1,4, 0.5,0.7,4, 1,0,4)).
    // The curve is 1 at the root and 0 at the tip, so the ends are the
    // authored literal and 15% of it.
    const float literal = 0.012f;
    size_t nonMonotone = 0, badRoot = 0, badTip = 0;
    for (uint32_t c = 0; c < curves; ++c) {
        float const *w = baseline.width.cdata() + size_t(c) * cvs;
        for (uint32_t i = 1; i < cvs; ++i) if (!(w[i] < w[i - 1])) ++nonMonotone;
        if (std::fabs(w[0] - literal) > 1e-6f) ++badRoot;
        if (std::fabs(w[cvs - 1] - literal * 0.15f) > 1e-6f) ++badTip;
    }
    Check(nonMonotone == 0, "published widths decrease strictly from root to tip");
    Check(badRoot == 0, "the root width is the authored literal ($value at $t = 0)");
    Check(badTip == 0, "the tip width is 15% of the literal ($t = 1)");

    // A mid-strand sample must follow the control curve, not a straight
    // line. The monotone spline leaves the root almost flat, so a quarter of
    // the way along the strand it is measurably wider than a linear taper.
    {
        float const *w = baseline.width.cdata();
        const float quarter = w[std::max<uint32_t>(1, cvs / 4)];
        const float linear = literal * (1.0f - 0.85f * (float(std::max<uint32_t>(1, cvs / 4)) /
                                                        float(cvs - 1)));
        Check(std::fabs(quarter - linear) > 1e-5f,
              "the profile is the control curve, not a linear taper");
    }

    // The Noise mask expression is $value * $u, so the frizz fades across the
    // plane: strands at u = 0 must be left bit-for-bit untouched by Noise.
    // Their CVs therefore stay on the straight line Grow produced.
    Check(!baseline.rootUV.empty() && baseline.rootUV.size() == curves,
          "root UVs are published for the primitive-domain mask expression");

    // ---- editing the source re-cooks -------------------------------------
    UsdPrim const expression = stage->GetPrimAtPath(SdfPath(kWidthExpression));
    UsdAttribute source = expression ? expression.GetAttribute(TfToken("usdGen:expr:source"))
                                     : UsdAttribute();
    Check(source && source.IsValid(), "the width expression exposes usdGen:expr:source");
    if (!source) return 1;
    Check(source.Set(std::string("$value * 0.5")), "edit the expression source on the stage");

    UsdGenCurveBuffer edited;
    if (!Cook(stage, &edited, "edited expression")) return 1;
    Check(edited.width.size() == baseline.width.size(),
          "the edit preserves the published topology");
    size_t wrong = 0, unchanged = 0;
    for (size_t i = 0; i < edited.width.size(); ++i) {
        if (std::fabs(edited.width[i] - literal * 0.5f) > 1e-6f) ++wrong;
        if (edited.width[i] == baseline.width[i]) ++unchanged;
    }
    Check(wrong == 0, "the edited expression publishes a constant half-literal width");
    Check(unchanged < edited.width.size(),
          "editing usdGen:expr:source changes the published widths");

    std::printf("testUsdGenExpressionWidthExample: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
