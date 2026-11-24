// testUsdGenPaintLengthRecook.cpp -- T1: repainting ONLY a paint-map
// primvar recooks the groom that samples it through ptex().
//
// The fixture is what the usdview brush authors (brushAuthor.SetupDescription
// + BindMaskPreset("length") + EnsurePaintWiring("length")): Grow's
// usdGen:length is connected to <desc>/Expressions/lengthScale,
// `ptex("lengthPaint") * 0.25`, whose input:lengthPaint targets the
// UsdGenPaintMap reading primvars:usdGen:paint:length on the scalp. A second
// wired preset (width) proves the same path for the other ptex presets.
//
// User report: "when I paint length, it does not affect the groom until I
// paint something else". The test cooks, rewrites only the length primvar
// (every corner 1.0 -> 0.3), cooks again through the real UsdImaging chain
// and asserts the strands got shorter; then does the same for width.
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {
int failures = 0;

void Check(bool value, char const *message)
{
    if (!value) {
        ++failures;
        std::printf("FAIL: %s\n", message);
    } else {
        std::printf("ok:   %s\n", message);
    }
}

// Exported from a stage authored by brushAuthor (setup, bind length, wire
// length); the width, clump and curl wirings mirror EnsurePaintWiring for
// those presets (PAINT_WIRING in brushAuthor.py).
char const *kFixture = R"USDA(#usda 1.0
def Mesh "Scalp"
{
    int[] faceVertexCounts = [4, 4, 4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7]
    point3f[] points = [(-1, 0, -1), (0, 0, -1), (1, 0, -1), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 0, 1), (0, 0, 1), (1, 0, 1)]
    float[] primvars:usdGen:paint:length = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (
        interpolation = "faceVarying"
    )
    float[] primvars:usdGen:paint:width = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (
        interpolation = "faceVarying"
    )
    float[] primvars:usdGen:paint:clump = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (
        interpolation = "faceVarying"
    )
    float[] primvars:usdGen:paint:curl = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (
        interpolation = "faceVarying"
    )
}

def Scope "Groom"
{
    def UsdGenDescription "Description"
    {
        rel usdGen:surface = </Scalp>

        def Scope "Ops"
        {
            def UsdGenWidth "width"
            {
                bool usdGen:replace = 1
                float usdGen:width = 0.005 (
                    customData = {
                        dictionary usdGen = {
                            string evaluation = "primitive"
                        }
                    }
                )
                prepend float usdGen:width.connect = </Groom/Description/Expressions/widthScale>
                float2[] usdGen:width:knots = [(0, 1), (0.6, 0.8), (1, 0.15)]
            }

            def UsdGenCurl "curl"
            {
                float usdGen:frequency = 2
                float usdGen:radius = 0.02 (
                    customData = {
                        dictionary usdGen = {
                            string evaluation = "primitive"
                        }
                    }
                )
                prepend float usdGen:radius.connect = </Groom/Description/Expressions/curlRadiusScale>
                uniform int usdGen:seed = 13
            }

            def UsdGenClump "clump"
            {
                float usdGen:clump:amount = 0.8 (
                    customData = {
                        dictionary usdGen = {
                            string evaluation = "primitive"
                        }
                    }
                )
                prepend float usdGen:clump:amount.connect = </Groom/Description/Expressions/clumpAmountScale>
                float2[] usdGen:clump:profile:knots = [(0, 0.25), (1, 1)]
                uniform int usdGen:seed = 71
            }

            def UsdGenGrow "grow"
            {
                uniform token usdGen:direction = "surfaceNormal"
                float usdGen:length = 0.25 (
                    customData = {
                        dictionary usdGen = {
                            string evaluation = "primitive"
                        }
                    }
                )
                prepend float usdGen:length.connect = </Groom/Description/Expressions/lengthScale>
                float2 usdGen:lengthRandom = (0.7, 1.3)
                uniform int usdGen:seed = 19
                int usdGen:segments = 5
            }

            def UsdGenScatter "scatter"
            {
                float usdGen:density = 600
                uniform int usdGen:seed = 41
            }
        }

        def Scope "Maps"
        {
            def UsdGenPaintMap "lengthPaint"
            {
                float usdGen:map:default = 1
                token usdGen:paint:interpolation = "faceVarying"
                token usdGen:paint:primvar = "usdGen:paint:length"
                int usdGen:paint:resolution = 32
                token usdGen:paint:storage = "primvar"
                rel usdGen:paint:surface = </Scalp>
            }

            def UsdGenPaintMap "widthPaint"
            {
                float usdGen:map:default = 1
                token usdGen:paint:interpolation = "faceVarying"
                token usdGen:paint:primvar = "usdGen:paint:width"
                int usdGen:paint:resolution = 32
                token usdGen:paint:storage = "primvar"
                rel usdGen:paint:surface = </Scalp>
            }

            def UsdGenPaintMap "clumpPaint"
            {
                float usdGen:map:default = 1
                token usdGen:paint:interpolation = "faceVarying"
                token usdGen:paint:primvar = "usdGen:paint:clump"
                int usdGen:paint:resolution = 32
                token usdGen:paint:storage = "primvar"
                rel usdGen:paint:surface = </Scalp>
            }

            def UsdGenPaintMap "curlPaint"
            {
                float usdGen:map:default = 1
                token usdGen:paint:interpolation = "faceVarying"
                token usdGen:paint:primvar = "usdGen:paint:curl"
                int usdGen:paint:resolution = 32
                token usdGen:paint:storage = "primvar"
                rel usdGen:paint:surface = </Scalp>
            }
        }

        def Scope "Expressions"
        {
            def UsdGenExpression "lengthScale"
            {
                custom rel input:lengthPaint = </Groom/Description/Maps/lengthPaint>
                custom float outputs:result
                string usdGen:expr:source = 'ptex("lengthPaint") * 0.25'
            }

            def UsdGenExpression "widthScale"
            {
                custom rel input:widthPaint = </Groom/Description/Maps/widthPaint>
                custom float outputs:result
                string usdGen:expr:source = 'ptex("widthPaint") * 0.005'
            }

            def UsdGenExpression "clumpAmountScale"
            {
                custom rel input:clumpPaint = </Groom/Description/Maps/clumpPaint>
                custom float outputs:result
                string usdGen:expr:source = 'ptex("clumpPaint") * 0.8'
            }

            def UsdGenExpression "curlRadiusScale"
            {
                custom rel input:curlPaint = </Groom/Description/Maps/curlPaint>
                custom float outputs:result
                string usdGen:expr:source = 'ptex("curlPaint") * 0.02'
            }
        }
    }
}
)USDA";

struct Scene {
    UsdStageRefPtr stage;
    UsdImagingSceneIndices indices;
    HdSceneIndexBaseRefPtr groom;
    UsdGenGroomSceneIndex *owner = nullptr;
};

VtValue TileValue(Scene const &scene, SdfPath const &tile, TfToken const &primvar)
{
    HdSceneIndexPrim prim = scene.groom->GetPrim(tile);
    HdSampledDataSourceHandle sampled = HdSampledDataSource::Cast(
        HdContainerDataSource::Get(prim.dataSource,
            HdDataSourceLocator(TfToken("primvars"), primvar, TfToken("primvarValue"))));
    return sampled ? sampled->GetValue(0.0f) : VtValue();
}

struct Measure {
    size_t curves = 0;
    double totalLength = 0.0;  // summed polyline length over every strand
    double maxWidth = 0.0;
    std::vector<GfVec3f> points;  // every published CV, tile order
};

// Sums the control-polygon length of every published strand across all
// synthetic tiles; the control polygon scales with usdGen:length.
Measure MeasureGroom(Scene const &scene)
{
    Measure out;
    SdfPath const render("/Groom/Description/__usdGenRender");
    for (SdfPath const &tile : scene.groom->GetChildPrimPaths(render)) {
        HdSceneIndexPrim prim = scene.groom->GetPrim(tile);
        HdSampledDataSourceHandle countsDs = HdSampledDataSource::Cast(
            HdContainerDataSource::Get(prim.dataSource,
                HdDataSourceLocator(TfToken("basisCurves"), TfToken("topology"),
                                    TfToken("curveVertexCounts"))));
        VtValue counts = countsDs ? countsDs->GetValue(0.0f) : VtValue();
        VtValue points = TileValue(scene, tile, TfToken("points"));
        if (!counts.IsHolding<VtIntArray>() || !points.IsHolding<VtVec3fArray>()) continue;
        VtIntArray const &n = counts.UncheckedGet<VtIntArray>();
        VtVec3fArray const &p = points.UncheckedGet<VtVec3fArray>();
        out.points.insert(out.points.end(), p.begin(), p.end());
        size_t base = 0;
        for (int count : n) {
            if (base + size_t(count) > p.size()) break;
            for (int k = 1; k < count; ++k)
                out.totalLength += double((p[base + k] - p[base + k - 1]).GetLength());
            base += size_t(count);
            ++out.curves;
        }
        VtValue widths = TileValue(scene, tile, TfToken("widths"));
        if (widths.IsHolding<VtFloatArray>())
            for (float w : widths.UncheckedGet<VtFloatArray>())
                out.maxWidth = std::max(out.maxWidth, double(w));
    }
    return out;
}

// Largest CV displacement between two cooks of the same strand set.
double MaxMove(Measure const &a, Measure const &b)
{
    if (a.points.size() != b.points.size()) return -1.0;
    double m = 0.0;
    for (size_t i = 0; i < a.points.size(); ++i)
        m = std::max(m, double((a.points[i] - b.points[i]).GetLength()));
    return m;
}

void Repaint(Scene &scene, char const *primvar, float value)
{
    UsdAttribute attr = scene.stage->GetPrimAtPath(SdfPath("/Scalp"))
                            .GetAttribute(TfToken(primvar));
    Check(bool(attr) && attr.Set(VtFloatArray(16, value)),
          "repaint authors every corner of the paint primvar");
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
}
}  // namespace

int main()
{
    usdGen::usdGenRegisterM1Operators();
    Scene scene;
    scene.stage = UsdStage::CreateInMemory("paint-length-recook.usda");
    scene.stage->GetRootLayer()->ImportFromString(kFixture);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = scene.stage;
    scene.indices = UsdImagingCreateSceneIndices(info);
    scene.groom = UsdGenGroomSceneIndex::New(scene.indices.finalSceneIndex);
    scene.owner = dynamic_cast<UsdGenGroomSceneIndex *>(scene.groom.operator->());
    Check(scene.owner != nullptr, "concrete groom scene owner is available");
    if (!scene.owner) return 1;
    scene.owner->Synchronize();

    Measure const before = MeasureGroom(scene);
    std::printf("before: curves=%zu totalLength=%.6f maxWidth=%.6g\n",
                before.curves, before.totalLength, before.maxWidth);
    Check(before.curves > 0 && before.totalLength > 0.0,
          "the brush-wired description cooks strands");

    // Length: 1.0 -> 0.3 on every corner, nothing else touched.
    Repaint(scene, "primvars:usdGen:paint:length", 0.3f);
    Measure const shorter = MeasureGroom(scene);
    double const ratio = before.totalLength > 0.0
        ? shorter.totalLength / before.totalLength : 0.0;
    std::printf("after length paint 0.3: curves=%zu totalLength=%.6f ratio=%.4f\n",
                shorter.curves, shorter.totalLength, ratio);
    Check(shorter.curves == before.curves,
          "a length repaint keeps the strand count (density untouched)");
    Check(std::fabs(ratio - 0.3) < 0.02,
          "a length-only repaint recooks the groom at 0.3x strand length");

    // Width: the same ptex path through another wired preset.
    Repaint(scene, "primvars:usdGen:paint:width", 0.3f);
    Measure const thinner = MeasureGroom(scene);
    std::printf("after width paint 0.3: maxWidth=%.6g (was %.6g)\n",
                thinner.maxWidth, shorter.maxWidth);
    Check(shorter.maxWidth > 0.0 &&
              std::fabs(thinner.maxWidth / shorter.maxWidth - 0.3) < 0.02,
          "a width-only repaint recooks the groom at 0.3x strand width");

    // And back: a second length repaint on an already-recooked groom.
    Repaint(scene, "primvars:usdGen:paint:length", 1.0f);
    Measure const restored = MeasureGroom(scene);
    std::printf("after length paint 1.0: totalLength=%.6f\n", restored.totalLength);
    Check(std::fabs(restored.totalLength - before.totalLength) <=
              1e-4 * before.totalLength,
          "repainting length back to 1.0 restores the original strands");

    // Clump and curl: mid-chain ops like Grow, rebuilt by every recompile
    // (their params are connected), with a captured op downstream.
    Repaint(scene, "primvars:usdGen:paint:clump", 0.3f);
    Measure const unclumped = MeasureGroom(scene);
    double const clumpMove = MaxMove(restored, unclumped);
    std::printf("after clump paint 0.3: max CV move %.6g\n", clumpMove);
    Check(clumpMove > 1e-4, "a clump-only repaint moves the strands");

    Repaint(scene, "primvars:usdGen:paint:curl", 0.3f);
    Measure const uncurled = MeasureGroom(scene);
    double const curlMove = MaxMove(unclumped, uncurled);
    std::printf("after curl paint 0.3: max CV move %.6g\n", curlMove);
    Check(curlMove > 1e-4, "a curl-only repaint moves the strands");

    // Back to the original paint: every CV returns exactly, so no stale
    // capture survived any of the repaints.
    Repaint(scene, "primvars:usdGen:paint:clump", 1.0f);
    Repaint(scene, "primvars:usdGen:paint:curl", 1.0f);
    Measure const original = MeasureGroom(scene);
    double const backMove = MaxMove(restored, original);
    std::printf("after clump/curl paint 1.0: max CV move vs before %.6g\n", backMove);
    Check(backMove >= 0.0 && backMove < 1e-5,
          "repainting clump and curl back restores the strands");

    return failures ? 1 : 0;
}
