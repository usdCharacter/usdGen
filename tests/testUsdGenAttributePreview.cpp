// testUsdGenAttributePreview (T1, no GPU) — the live paint-preview overlay,
// the app -> Hydra leg of a brush stroke:
//
//   * SetMapPreview overlays primvars/displayColor on the brushed mesh (one
//     colour per map face, through the shared UsdGenPreviewColor maps) while
//     every other primvar and every other prim shows through unchanged;
//   * SetGroomPreview overlays one colour per strand root on the groom,
//     sampled exactly as the capture loop samples, with the no-value colour
//     for roots the map cannot sample;
//   * the first Set dirties primvars/displayColor (structural), a re-Set
//     dirties primvars/displayColor/primvarValue only, and ClearPreview
//     restores upstream (structural);
//   * invalid Sets fail closed and leave any previous preview in place.
#include "usdGenImaging/attributePreviewSceneIndex.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "usdGen/maps/attributeMap.h"
#include "usdGenHydraCurves/previewColor.h"

#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

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
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

bool Near(GfVec3f const &a, GfVec3f const &b, float tolerance = 1e-5f)
{
    return std::fabs(a[0] - b[0]) <= tolerance &&
           std::fabs(a[1] - b[1]) <= tolerance &&
           std::fabs(a[2] - b[2]) <= tolerance;
}

// The scalp is the bake test's two quads (plus an authored vertex
// displayColor the overlay must win over); the groom is two strands.
char const *kFixture = R"USDA(#usda 1.0
def Mesh "Scalp"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(-1,0,-1), (0,0,-1), (1,0,-1), (-1,0,1), (0,0,1), (1,0,1)]
    color3f[] primvars:displayColor = [(1,0,0)] (interpolation = "constant")
}
def BasisCurves "Groom"
{
    uniform token type = "linear"
    int[] curveVertexCounts = [2, 2]
    point3f[] points = [(0,0,0), (0,1,0), (1,0,0), (1,1,0)]
}
)USDA";

class DirtyRecorder final : public HdSceneIndexObserver {
public:
    std::vector<std::pair<SdfPath, HdDataSourceLocatorSet>> dirtied;
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &) override {}
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &) override {}
    void PrimsDirtied(HdSceneIndexBase const &,
                      DirtiedPrimEntries const &entries) override
    {
        for (auto const &entry : entries)
            dirtied.emplace_back(entry.primPath, entry.dirtyLocators);
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
    void Clear() { dirtied.clear(); }
    size_t CountFor(SdfPath const &path, HdDataSourceLocator const &locator) const
    {
        size_t n = 0;
        for (auto const &entry : dirtied)
            if (entry.first == path && entry.second.Intersects(locator)) ++n;
        return n;
    }
};

VtVec3fArray DisplayColors(HdSceneIndexBase &index, SdfPath const &path,
                           TfToken *interpolation = nullptr)
{
    HdSceneIndexPrim const prim = index.GetPrim(path);
    HdPrimvarSchema const pv(HdContainerDataSource::Cast(
        HdContainerDataSource::Get(prim.dataSource,
                                   HdDataSourceLocator(TfToken("primvars"),
                                                       TfToken("displayColor")))));
    if (interpolation) {
        *interpolation = TfToken();
        if (pv.IsDefined() && pv.GetInterpolation())
            *interpolation = pv.GetInterpolation()->GetTypedValue(0.0f);
    }
    if (!pv.IsDefined() || !pv.GetPrimvarValue()) return VtVec3fArray();
    VtValue const value = pv.GetPrimvarValue()->GetValue(0.0f);
    if (!value.IsHolding<VtVec3fArray>()) return VtVec3fArray();
    return value.UncheckedGet<VtVec3fArray>();
}

std::shared_ptr<UsdGenAttributeMap> TwoFaceMap()
{
    UsdGenAttributeMapSpec spec;
    spec.numFaces = 2;
    spec.resolution = 4;  // exact-mean path
    spec.channels = 1;
    std::string error;
    return UsdGenAttributeMap::Create(spec, &error);
}

}  // namespace

int main()
{
    usdGenTest::EnsureHydraPublishers();
    SdfPath const scalp("/Scalp");
    SdfPath const groom("/Groom");
    HdDataSourceLocator const structural(TfToken("primvars"),
                                         TfToken("displayColor"));
    HdDataSourceLocator const values(TfToken("primvars"), TfToken("displayColor"),
                                     TfToken("primvarValue"));

    UsdStageRefPtr stage = UsdStage::CreateInMemory("attribute-preview");
    stage->GetRootLayer()->ImportFromString(kFixture);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(info);
    TfRefPtr<usdGenImaging::UsdGenAttributePreviewSceneIndex> preview =
        usdGenImaging::UsdGenAttributePreviewSceneIndex::New(
            indices.finalSceneIndex);
    Check(preview != nullptr, "the preview index constructs");
    if (!preview) return 1;
    DirtyRecorder recorder;
    preview->AddObserver(TfCreateWeakPtr(&recorder));

    TfToken gray("gray");
    GfVec2f const unit(0.0f, 1.0f);

    // ---- no preview: upstream shows through --------------------------------
    {
        TfToken interp;
        VtVec3fArray const colors = DisplayColors(*preview, scalp, &interp);
        Check(colors.size() == 1 && Near(colors[0], GfVec3f(1.0f, 0.0f, 0.0f)),
              "without a preview the mesh keeps its authored displayColor");
        Check(DisplayColors(*preview, groom).empty(),
              "without a preview the groom has no displayColor");
        Check(!preview->HasPreview(scalp) && !preview->HasPreview(groom),
              "nothing previewed initially");
    }

    // ---- the map preview: one colour per face, overlay wins -----------------
    auto map = TwoFaceMap();
    map->Fill(0.0f);
    for (int t = 0; t < 4; ++t)
        for (int s = 0; s < 4; ++s) map->SetTexel(1, s, t, 0, 1.0f);
    recorder.Clear();
    Check(preview->SetMapPreview(scalp, map, gray, unit), "SetMapPreview accepts");
    Check(preview->HasPreview(scalp) && !preview->HasPreview(groom),
          "the preview is per target");
    {
        TfToken interp;
        VtVec3fArray const colors = DisplayColors(*preview, scalp, &interp);
        Check(colors.size() == 2 && Near(colors[0], GfVec3f(0.0f)) &&
                  Near(colors[1], GfVec3f(1.0f)),
              "face 0 reads black, face 1 white (gray 0..1)");
        Check(interp == TfToken("uniform"), "the map overlay is uniform");
        Check(recorder.CountFor(scalp, structural) == 1,
              "the first Set dirties primvars/displayColor once");
        Check(DisplayColors(*preview, groom).empty(),
              "the mesh preview leaves the groom alone");
    }

    // ---- a re-Set re-reads values only --------------------------------------
    map->Fill(0.5f);
    recorder.Clear();
    Check(preview->SetMapPreview(scalp, map, gray, unit), "re-Set accepts");
    {
        VtVec3fArray const colors = DisplayColors(*preview, scalp);
        // sRGB 0.5 is linear 0.214: the overlay shares the value preview maps.
        Check(colors.size() == 2 && Near(colors[0], GfVec3f(0.2140f), 1e-3f) &&
                  Near(colors[1], GfVec3f(0.2140f), 1e-3f),
              "the re-Set recolours both faces");
        Check(recorder.CountFor(scalp, values) == 1,
              "a re-Set dirties primvars/displayColor/primvarValue once");
    }

    // ---- invalid Sets fail closed -------------------------------------------
    {
        VtVec3fArray const before = DisplayColors(*preview, scalp);
        Check(!preview->SetMapPreview(scalp, nullptr, gray, unit),
              "a null map is rejected");
        Check(!preview->SetMapPreview(scalp, map, gray, unit, 1),
              "a channel past the end is rejected");
        Check(!preview->SetMapPreview(scalp, map, gray, unit, -1),
              "a negative channel is rejected");
        Check(DisplayColors(*preview, scalp) == before,
              "rejected Sets leave the preview in place");
        Check(!preview->SetGroomPreview(groom, map, {}, {}, gray, unit),
              "empty roots are rejected");
        Check(!preview->SetGroomPreview(groom, map, {0}, {GfVec2f(0.5f), GfVec2f(0.5f)},
                                        gray, unit),
              "mismatched roots are rejected");
        Check(!preview->HasPreview(groom), "rejected groom Sets preview nothing");
    }

    // ---- the groom preview: one colour per strand root ----------------------
    map->Fill(0.0f);
    for (int t = 0; t < 4; ++t)
        for (int s = 0; s < 4; ++s) map->SetTexel(1, s, t, 0, 1.0f);
    recorder.Clear();
    std::vector<int> const faces = {0, 1};
    std::vector<GfVec2f> const uv = {GfVec2f(0.5f), GfVec2f(0.5f)};
    Check(preview->SetGroomPreview(groom, map, faces, uv, gray, unit),
          "SetGroomPreview accepts");
    {
        TfToken interp;
        VtVec3fArray const colors = DisplayColors(*preview, groom, &interp);
        Check(colors.size() == 2 && Near(colors[0], GfVec3f(0.0f)) &&
                  Near(colors[1], GfVec3f(1.0f)),
              "each strand reads its root through the gray map");
        Check(interp == TfToken("uniform"), "the groom overlay is uniform");
        Check(recorder.CountFor(groom, structural) == 1,
              "the groom Set dirties primvars/displayColor once");
    }
    // A root the map cannot sample shows the no-value colour, not a failure.
    recorder.Clear();
    Check(preview->SetGroomPreview(groom, map, {0, 7}, uv, gray, unit),
          "an out-of-range root does not fail the Set");
    {
        VtVec3fArray const colors = DisplayColors(*preview, groom);
        Check(colors.size() == 2 && Near(colors[0], GfVec3f(0.0f)) &&
                  Near(colors[1], UsdGenPreviewMissingColor()),
              "the unreadable root shows the no-value colour");
    }

    // ---- faceVarying overlays (the brush ABI's shape) -------------------------
    {
        VtVec3fArray fv(8, GfVec3f(0.0f));
        fv[5] = GfVec3f(0.0f, 1.0f, 0.0f);
        recorder.Clear();
        Check(preview->SetFaceVaryingPreview(scalp, fv), "SetFaceVaryingPreview accepts");
        TfToken interp;
        VtVec3fArray const colors = DisplayColors(*preview, scalp, &interp);
        Check(colors.size() == 8 && interp == TfToken("faceVarying") &&
                  Near(colors[5], GfVec3f(0.0f, 1.0f, 0.0f)),
              "the mesh reads four colours per face, faceVarying");
        Check(recorder.CountFor(scalp, structural) == 1,
              "uniform -> faceVarying is a structural dirty");
        fv[5] = GfVec3f(1.0f);
        recorder.Clear();
        Check(preview->SetFaceVaryingPreview(scalp, fv), "a faceVarying re-Set accepts");
        Check(recorder.dirtied.size() == 1 && recorder.CountFor(scalp, values) == 1 &&
                  !recorder.dirtied[0].second.Contains(structural),
              "a same-size faceVarying re-Set dirties primvarValue only");
        Check(!preview->SetFaceVaryingPreview(scalp, VtVec3fArray()),
              "an empty faceVarying array is rejected");
        VtVec3fArray wrong(5, GfVec3f(1.0f));
        Check(preview->SetFaceVaryingPreview(scalp, wrong), "a mis-sized array is stored");
        Check(DisplayColors(*preview, scalp).size() == 1,
              "but withheld: the authored displayColor shows through");
        // Back to the per-face state the rest of the test expects.
        Check(preview->SetMapPreview(scalp, map, gray, unit), "uniform re-Set after faceVarying");
        Check(DisplayColors(*preview, scalp).size() == 2, "the uniform overlay is back");
        Check(!preview->HasPreview(SdfPath("/Nothing")) &&
                  usdGenImaging::UsdGenAttributePreviewRegistry::Get().EntryCount() == 0,
              "an unattached index leaves the global registry alone");
    }

    // ---- clearing restores upstream -----------------------------------------
    recorder.Clear();
    Check(preview->ClearPreview(scalp), "ClearPreview removes the map overlay");
    {
        VtVec3fArray const colors = DisplayColors(*preview, scalp);
        Check(colors.size() == 1 && Near(colors[0], GfVec3f(1.0f, 0.0f, 0.0f)),
              "clearing restores the authored displayColor");
        Check(recorder.CountFor(scalp, structural) == 1,
              "ClearPreview dirties primvars/displayColor once");
        Check(!preview->ClearPreview(scalp), "clearing twice reports false");
        Check(!preview->HasPreview(scalp) && preview->HasPreview(groom),
              "clearing one target keeps the other");
    }
    preview->ClearAllPreviews();
    Check(!preview->HasPreview(groom) && DisplayColors(*preview, groom).empty(),
          "ClearAllPreviews removes the groom overlay");

    preview->RemoveObserver(TfCreateWeakPtr(&recorder));

    // ---- an INDEXED upstream displayColor ----------------------------------
    // The overlay merges into the upstream primvar container; without blocks
    // the upstream indexedPrimvarValue + indices would survive and the
    // scene-delegate adapter (IsIndexed) would read them instead.
    {
        UsdStageRefPtr indexed = UsdStage::CreateInMemory("attribute-preview-indexed");
        indexed->GetRootLayer()->ImportFromString(R"USDA(#usda 1.0
def Mesh "Scalp"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(-1,0,-1), (0,0,-1), (1,0,-1), (-1,0,1), (0,0,1), (1,0,1)]
    color3f[] primvars:displayColor = [(1,0,0), (0,1,0)] (interpolation = "uniform")
    int[] primvars:displayColor:indices = [1, 0]
}
)USDA");
        UsdImagingCreateSceneIndicesInfo indexedInfo;
        indexedInfo.stage = indexed;
        UsdImagingSceneIndices indexedIndices = UsdImagingCreateSceneIndices(indexedInfo);
        auto indexedPreview = usdGenImaging::UsdGenAttributePreviewSceneIndex::New(
            indexedIndices.finalSceneIndex);
        auto primvarOf = [&](HdSceneIndexBase &index) {
            return HdPrimvarSchema(HdContainerDataSource::Cast(HdContainerDataSource::Get(
                index.GetPrim(scalp).dataSource,
                HdDataSourceLocator(TfToken("primvars"), TfToken("displayColor")))));
        };
        Check(primvarOf(*indexedIndices.finalSceneIndex).IsIndexed(),
              "the fixture's upstream displayColor is indexed");
        VtVec3fArray fv(8, GfVec3f(0.0f, 0.0f, 1.0f));
        Check(indexedPreview->SetFaceVaryingPreview(scalp, fv),
              "a faceVarying overlay over an indexed displayColor accepts");
        HdPrimvarSchema const pv = primvarOf(*indexedPreview);
        Check(!pv.IsIndexed() && !pv.GetIndices(),
              "the overlay blocks the upstream indices");
        HdSampledDataSourceHandle const adapterValue = pv.GetIndexedPrimvarValue();
        VtValue const value = adapterValue ? adapterValue->GetValue(0.0f) : VtValue();
        Check(value.IsHolding<VtVec3fArray>() &&
                  value.UncheckedGet<VtVec3fArray>().size() == 8 &&
                  Near(value.UncheckedGet<VtVec3fArray>()[0], GfVec3f(0.0f, 0.0f, 1.0f)),
              "so the adapter's indexed read gets the overlay colours");
        Check(indexedPreview->ClearPreview(scalp) &&
                  primvarOf(*indexedPreview).IsIndexed(),
              "clearing shows the indexed upstream again");
    }
    std::printf("testUsdGenAttributePreview: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
