// usdGen:enabled through the stage-backed path (02 §6.2/§6.3).
//
// A disabled styler passes its upstream through unchanged; a disabled
// generator publishes an EMPTY curve set, so the description publishes no
// curves at all.  Toggling either back restores the previous points exactly.
// This test drives the same builder the viewport uses (BuildGraphDescFromStage)
// so an edit that never reaches the engine is caught here rather than in a
// render.
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"

#include "pxr/pxr.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using usdGen::UsdGenCompiler;
using usdGen::UsdGenCurveBuffer;
using usdGen::UsdGenEvalContext;
using usdGen::UsdGenGraph;
using usdGen::UsdGenGraphDesc;
using usdGen::UsdGenScheduler;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what); }
    else std::printf("ok:   %s\n", what);
}

char const *kScene = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Mesh "Plane" (
        prepend apiSchemas = ["UsdGenRestAPI"]
    )
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 3, 2, 1]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        point3f[] primvars:rest = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
    }

    def Xform "Groom"
    {
        def UsdGenDescription "Fur"
        {
            rel usdGen:surface = </World/Plane>
            float usdGen:width:default = 0.004

            def Scope "Ops"
            {
                def UsdGenNoise "noise"
                {
                    int usdGen:seed = 7
                    float usdGen:noise:magnitude = 0.05
                    float usdGen:noise:frequency = 2.5
                    int usdGen:noise:octaves = 2
                    float usdGen:preserveLength = 0
                }

                def UsdGenGrow "grow"
                {
                    int usdGen:seed = 19
                    int usdGen:segments = 8
                    float usdGen:length = 0.3
                    uniform token usdGen:direction = "vector"
                    vector3f usdGen:directionVector = (0.35, 1, 0.2)
                }

                def UsdGenScatter "scatter"
                {
                    int usdGen:seed = 41
                    float usdGen:density = 40
                }
            }
        }
    }
}
)usda";

// Cook the CURRENT stage state through the builder the viewport uses and
// return the terminal's published points.
bool Cook(UsdStageRefPtr const &stage, UsdGenCurveBuffer *out, uint64_t generation)
{
    UsdGenGraphDesc const desc = usdGenImaging::BuildGraphDescFromStage(
        stage, SdfPath("/World/Groom/Fur"));
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    usdGen::UsdGenCompileResult const compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        for (auto const &error : compiled.errors)
            std::printf("      compile: %s\n", error.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Spec();
    usdGen::UsdGenRunResult const run = scheduler.Run(graph, context, generation);
    if (run.diagnostics.HasErrors()) {
        for (auto const &error : run.diagnostics.errors)
            std::printf("      run: %s\n", error.c_str());
        return false;
    }
    *out = OutputBuffer(graph);
    return true;
}

void SetEnabled(UsdStageRefPtr const &stage, char const *opPath, bool value)
{
    UsdPrim prim = stage->GetPrimAtPath(SdfPath(opPath));
    prim.GetAttribute(TfToken("usdGen:enabled")).Set(value);
}

bool SamePoints(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return a.totalCurves == b.totalCurves && a.totalCvs == b.totalCvs &&
           a.px == b.px && a.py == b.py && a.pz == b.pz;
}

}  // namespace

int main()
{
    SdfLayerRefPtr layer = SdfLayer::CreateAnonymous(".usda");
    if (!layer->ImportFromString(kScene)) {
        std::printf("FAIL: could not import the toggle fixture\n");
        return 1;
    }
    UsdStageRefPtr stage = UsdStage::Open(layer);
    Check(static_cast<bool>(stage), "toggle fixture opens");
    if (!stage) return 1;

    UsdGenCurveBuffer full;
    Check(Cook(stage, &full, 1), "baseline groom cooks");
    Check(full.totalCurves > 0 && full.totalCvs == full.totalCurves * 8,
          "baseline publishes grown strands");

    // A disabled styler is a pass-through: the same strands, without the
    // noise displacement.
    SetEnabled(stage, "/World/Groom/Fur/Ops/noise", false);
    UsdGenCurveBuffer mutedNoise;
    Check(Cook(stage, &mutedNoise, 2), "disabled Noise cooks");
    Check(mutedNoise.totalCurves == full.totalCurves &&
              mutedNoise.totalCvs == full.totalCvs,
          "disabled Noise keeps the upstream topology");
    Check(!SamePoints(mutedNoise, full),
          "disabling Noise changes the published points");

    SetEnabled(stage, "/World/Groom/Fur/Ops/noise", true);
    UsdGenCurveBuffer restoredNoise;
    Check(Cook(stage, &restoredNoise, 3), "re-enabled Noise cooks");
    Check(SamePoints(restoredNoise, full),
          "re-enabling Noise restores the published points exactly");

    // A disabled generator publishes nothing: the description has no curves.
    SetEnabled(stage, "/World/Groom/Fur/Ops/scatter", false);
    UsdGenCurveBuffer mutedScatter;
    Check(Cook(stage, &mutedScatter, 4), "disabled Scatter cooks");
    Check(mutedScatter.totalCurves == 0 && mutedScatter.totalCvs == 0,
          "disabling the generator publishes an empty curve set");

    SetEnabled(stage, "/World/Groom/Fur/Ops/scatter", true);
    UsdGenCurveBuffer restoredScatter;
    Check(Cook(stage, &restoredScatter, 5), "re-enabled Scatter cooks");
    Check(SamePoints(restoredScatter, full),
          "re-enabling the generator restores the published points exactly");

    // Grow is the CV-count generator: disabling it must also empty the set
    // rather than leaving the roots behind as degenerate curves.
    SetEnabled(stage, "/World/Groom/Fur/Ops/grow", false);
    UsdGenCurveBuffer mutedGrow;
    Check(Cook(stage, &mutedGrow, 6), "disabled Grow cooks");
    Check(mutedGrow.totalCurves == 0 && mutedGrow.totalCvs == 0,
          "disabling Grow publishes an empty curve set");

    std::printf(g_failures ? "testUsdGenEnabledToggle: FAILED (%d)\n"
                           : "testUsdGenEnabledToggle: PASS\n",
                g_failures);
    return g_failures ? 1 : 0;
}
