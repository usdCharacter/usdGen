// usdGen:preview:* — the viewport value preview, through the groom scene index.
//
// examples/clump-ptex-plane.usda, with the preview authored in the session
// layer the way the SeExpr editor does it:
//   * a UsdGenPtexMap source colours every strand by the map at its root
//     ("ids": one colour per clump cell), and the tiles bind the synthetic
//     material_preview, which the scene index then serves;
//   * a UsdGenExpression source is evaluated over the published strands, per
//     strand or per CV;
//   * an operator attribute shows the values that operator was cooked with,
//     or its authored value when nothing is connected;
//   * editing a previewed expression that drives nothing still recolours;
//   * "flat" shading swaps the material; clearing the preview restores the
//     look and removes the preview material.

// The Hydra material schemas first: a header below pulls in <windows.h>,
// whose `interface` macro breaks materialNetworkSchema.h.
#include "utils/pxrInterop.h"
#include "pxr/imaging/hd/dataSource.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/materialNetworkSchema.h"
#include "pxr/imaging/hd/materialNodeSchema.h"
#include "pxr/imaging/hd/materialSchema.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"

#include "usdGen/registries/op.h"
#include "usdGenUsd/lower.h"
#include "usdGenUsd/publishPaths.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGenHydraCurves/previewColor.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"
#include "usdGenHydraCurves/usdGenTilePublisher.h"

#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/editContext.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/relationship.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
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

constexpr char const *kScene = USDGEN_TEST_SOURCE_DIR "/examples/clump-ptex-plane.usda";
SdfPath const kDescription("/World/Groom/Fur");
SdfPath const kRender("/World/Groom/Fur/__usdGenRender");
SdfPath const kRegionMap("/World/Groom/Fur/Maps/clumpRegions");
SdfPath const kTightnessMap("/World/Groom/Fur/Maps/clumpTightness");
SdfPath const kTightness("/World/Groom/Fur/Expressions/clumpTightness");
SdfPath const kClump("/World/Groom/Fur/Ops/clump");

/// Every primvar locator the groom scene index dirties on a tile.
class DirtyLog : public HdSceneIndexObserver
{
public:
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &) override {}
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &) override {}
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &entries) override
    {
        for (auto const &entry : entries)
            for (HdDataSourceLocator const &locator : entry.dirtyLocators)
                if (entry.primPath.GetName().rfind("tile_", 0) == 0 &&
                    locator.GetFirstElement() == TfToken("primvars"))
                    locators.insert(locator.GetString());
    }
    std::set<std::string> locators;
};

struct Scene {
    UsdStageRefPtr stage;
    UsdImagingSceneIndices indices;
    HdSceneIndexBaseRefPtr groom;
    UsdGenGroomSceneIndex *owner = nullptr;
    std::shared_ptr<DirtyLog> log = std::make_shared<DirtyLog>();

    void Settle()
    {
        indices.stageSceneIndex->ApplyPendingUpdates();
        owner->Synchronize();
    }
};

Scene MakeScene()
{
    Scene out;
    out.stage = UsdStage::Open(kScene, UsdStage::LoadAll);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = out.stage;
    out.indices = UsdImagingCreateSceneIndices(info);
    out.groom = UsdGenGroomSceneIndex::New(out.indices.finalSceneIndex);
    out.owner = dynamic_cast<UsdGenGroomSceneIndex *>(out.groom.operator->());
    out.groom->AddObserver(TfCreateWeakPtr(out.log.get()));
    return out;
}

/// What the tiles publish, concatenated in tile order.
struct Published {
    size_t curves = 0, points = 0;
    std::vector<GfVec3f> colors;
    std::vector<int> clumpIds;
    std::set<SdfPath> bindings;
    SdfPathVector children;
};

VtValue PrimvarValue(HdContainerDataSourceHandle const &prim, char const *name)
{
    auto sampled = HdSampledDataSource::Cast(HdContainerDataSource::Get(
        prim, HdDataSourceLocator(TfToken("primvars"), TfToken(name), TfToken("primvarValue"))));
    return sampled ? sampled->GetValue(0.0f) : VtValue();
}

Published Publish(Scene const &scene)
{
    Published out;
    out.children = scene.groom->GetChildPrimPaths(kRender);
    SdfPathVector tiles;
    for (SdfPath const &child : out.children)
        if (child.GetName().rfind("tile_", 0) == 0) tiles.push_back(child);
    std::sort(tiles.begin(), tiles.end());
    for (SdfPath const &tile : tiles) {
        HdSceneIndexPrim const prim = scene.groom->GetPrim(tile);
        VtValue const counts = HdSampledDataSource::Cast(HdContainerDataSource::Get(
            prim.dataSource, HdDataSourceLocator(TfToken("basisCurves"), TfToken("topology"),
                                                 TfToken("curveVertexCounts"))))->GetValue(0.0f);
        for (int n : counts.Get<VtIntArray>()) { ++out.curves; out.points += size_t(n); }
        VtValue const colors = PrimvarValue(prim.dataSource, "displayColor");
        if (colors.IsHolding<VtVec3fArray>())
            for (GfVec3f const &c : colors.UncheckedGet<VtVec3fArray>()) out.colors.push_back(c);
        VtValue const ids = PrimvarValue(prim.dataSource, "clumpId_0");
        if (ids.IsHolding<VtIntArray>())
            for (int id : ids.UncheckedGet<VtIntArray>()) out.clumpIds.push_back(id);
        // The publisher's own binding (the overlay only masks it at low
        // complexity).
        HdMaterialBindingsSchema const bindings = HdMaterialBindingsSchema::GetFromParent(
            HdContainerDataSource::Cast(prim.dataSource));
        if (HdPathDataSourceHandle const path = bindings.GetMaterialBinding().GetPath())
            out.bindings.insert(path->GetTypedValue(0.0f));
    }
    return out;
}

bool Has(SdfPathVector const &paths, SdfPath const &path)
{
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

TfToken SurfaceNode(HdSceneIndexPrim const &prim, TfToken const &renderContext)
{
    HdMaterialNetworkSchema const network =
        HdMaterialSchema::GetFromParent(prim.dataSource).GetMaterialNetwork(
            renderContext);
    if (!network) return TfToken();
    HdMaterialNodeSchema const node = network.GetNodes().Get(TfToken("surface"));
    HdTokenDataSourceHandle const id = node.GetNodeIdentifier();
    return id ? id->GetTypedValue(0.0f) : TfToken();
}

/// Authors the preview in the session layer, as the editor does.
void SetPreview(Scene &scene, SdfPath const &source, char const *colorMap,
                GfVec2f range = GfVec2f(0.0f, 1.0f), char const *evaluation = "primitive",
                char const *shading = "lit")
{
    UsdEditContext session(scene.stage, scene.stage->GetSessionLayer());
    UsdPrim const description = scene.stage->GetPrimAtPath(kDescription);
    description.GetRelationship(TfToken("usdGen:preview:source")).SetTargets({source});
    description.GetAttribute(TfToken("usdGen:preview:colorMap")).Set(TfToken(colorMap));
    description.GetAttribute(TfToken("usdGen:preview:range")).Set(range);
    description.GetAttribute(TfToken("usdGen:preview:evaluation")).Set(TfToken(evaluation));
    description.GetAttribute(TfToken("usdGen:preview:shading")).Set(TfToken(shading));
    scene.Settle();
}

bool Near(GfVec3f const &a, GfVec3f const &b, float tolerance = 1e-5f)
{
    return std::fabs(a[0] - b[0]) <= tolerance && std::fabs(a[1] - b[1]) <= tolerance &&
        std::fabs(a[2] - b[2]) <= tolerance;
}

size_t Distinct(std::vector<GfVec3f> const &colors)
{
    std::set<std::tuple<float, float, float>> seen;
    for (GfVec3f const &c : colors) seen.emplace(c[0], c[1], c[2]);
    return seen.size();
}

void CheckColorMaps()
{
    TfToken const gray("gray"), rgb("rgb"), ids("ids"), heat("heat");
    GfVec2f const unit(0.0f, 1.0f);
    double const zero = 0.0, one = 1.0, half = 0.5, below = -3.0;
    Check(Near(UsdGenPreviewColor(gray, unit, &zero, 1), GfVec3f(0.0f)), "gray maps the range start to black");
    Check(Near(UsdGenPreviewColor(gray, unit, &one, 1), GfVec3f(1.0f)), "gray maps the range end to white");
    GfVec3f const mid = UsdGenPreviewColor(gray, unit, &half, 1);
    Check(std::fabs(mid[0] - 0.2140f) < 1e-3f, "gray is perceptual: sRGB 0.5 is linear 0.214");
    Check(Near(UsdGenPreviewColor(gray, unit, &below, 1), GfVec3f(0.0f)), "values below the range clamp");
    double const colour[3] = {0.25, 0.5, 1.0};
    Check(Near(UsdGenPreviewColor(rgb, unit, colour, 3), GfVec3f(0.25f, 0.5f, 1.0f)),
          "rgb shows three components as the linear colour");
    double const nan = std::numeric_limits<double>::quiet_NaN();
    Check(Near(UsdGenPreviewColor(heat, unit, &nan, 1), GfVec3f(1.0f, 0.0f, 1.0f)),
          "a value that is not finite is magenta");
    double const a = 0.25, b = 0.25 + 1e-6, c = 0.5;
    Check(Near(UsdGenPreviewColor(ids, unit, &a, 1), UsdGenPreviewColor(ids, unit, &b, 1)),
          "ids: equal values share a colour");
    Check(!Near(UsdGenPreviewColor(ids, unit, &a, 1), UsdGenPreviewColor(ids, unit, &c, 1)),
          "ids: different values differ");
    Check(Distinct({UsdGenPreviewColor(heat, unit, &zero, 1), UsdGenPreviewColor(heat, unit, &half, 1),
                    UsdGenPreviewColor(heat, unit, &one, 1)}) == 3, "heat varies across the range");
    for (bool const flat : {false, true})
        Check(usdGenUsd::PrimPath(UsdGenPrimRef{usdGenUsd::KeyOf(kDescription),
                                  UsdGenPreviewMaterialRole(TfToken(flat ? "flat" : "lit")), 0},
                                  kDescription) ==
                  usdGenImaging::UsdGenTilePublisher::PreviewMaterialPath(kDescription, flat),
              std::string("engine and publisher agree on the ") + (flat ? "flat" : "lit") +
                  " preview material path");
}

void CheckBuilder()
{
    UsdStageRefPtr const stage = UsdStage::Open(kScene);
    UsdPrim const description = stage->GetPrimAtPath(kDescription);
    UsdGenGraphDesc before = usdGenImaging::BuildGraphDescFromStage(stage, kDescription);
    Check(!before.preview.Active(), "an unauthored preview is inactive");
    Check(before.preview.colorMap == TfToken("heat") && before.preview.shading == TfToken("lit") &&
          before.preview.evaluation == TfToken("primitive"),
          "the preview reads the schema fallbacks");
    description.GetRelationship(TfToken("usdGen:preview:source")).SetTargets({kTightnessMap});
    description.GetAttribute(TfToken("usdGen:preview:range")).Set(GfVec2f(0.2f, 0.8f));
    UsdGenGraphDesc after = usdGenImaging::BuildGraphDescFromStage(stage, kDescription);
    Check(after.preview.source == kTightnessMap && after.preview.range == GfVec2f(0.2f, 0.8f),
          "the stage builder reads usdGen:preview:source and range");
    Check(std::count_if(after.maps.begin(), after.maps.end(),
                        [](UsdGenMapDesc const &m) { return m.path == kTightnessMap; }) == 1,
          "a previewed map is in the map pool once");
}

void CheckScene()
{
    Scene scene = MakeScene();
    Check(scene.owner != nullptr, "the clump example opens in a groom scene index");
    if (!scene.owner) return;
    scene.owner->Synchronize();
    SdfPath const storm = usdGenImaging::UsdGenTilePublisher::MaterialPath(kDescription);
    SdfPath const lit = usdGenImaging::UsdGenTilePublisher::PreviewMaterialPath(kDescription, false);
    SdfPath const flat = usdGenImaging::UsdGenTilePublisher::PreviewMaterialPath(kDescription, true);

    Published const look = Publish(scene);
    Check(look.curves > 1000 && look.colors.size() == look.curves,
          "the look publishes one colour per strand (" + std::to_string(look.curves) + ")");
    Check(look.bindings == std::set<SdfPath>{storm}, "without a preview the tiles bind material_storm");
    Check(!Has(look.children, lit) && !scene.groom->GetPrim(lit).dataSource,
          "without a preview there is no preview material");

    // 1. The region map, one colour per cell.
    SetPreview(scene, kRegionMap, "ids");
    Published const regions = Publish(scene);
    Check(regions.bindings == std::set<SdfPath>{lit}, "a previewing groom binds material_preview");
    Check(Has(regions.children, lit), "material_preview is a child of the render scope");
    HdSceneIndexPrim const material = scene.groom->GetPrim(lit);
    Check(material.primType == TfToken("material") &&
          SurfaceNode(material, TfToken("glslfx")) == TfToken("UsdGenValuePreview"),
          "material_preview is a UsdGenValuePreview material in the glslfx context");
    Check(SurfaceNode(material, HdMaterialSchemaTokens->universalRenderContext) ==
              TfToken("UsdPreviewSurface"),
          "material_preview falls back to UsdPreviewSurface universally");
    Check(regions.curves == look.curves && regions.colors.size() == regions.curves,
          "the map preview is one colour per strand");
    size_t const cells = Distinct(regions.colors);
    std::printf("  %zu region colours\n", cells);
    Check(cells >= 40 && cells <= 120, "the region map shows one colour per clump cell");
    std::map<int, std::set<std::tuple<float, float, float>>> perClump;
    for (size_t s = 0; s < regions.clumpIds.size() && s < regions.colors.size(); ++s)
        if (regions.clumpIds[s] >= 0)
            perClump[regions.clumpIds[s]].emplace(regions.colors[s][0], regions.colors[s][1],
                                                  regions.colors[s][2]);
    size_t single = 0;
    for (auto const &clump : perClump) single += clump.second.size() == 1;
    Check(regions.clumpIds.size() == regions.curves && !perClump.empty() &&
          single == perClump.size(), "every map clump is one region colour");

    // 2. An expression prim, per strand, and the operator attribute it drives.
    GfVec2f const tightRange(0.3f, 0.95f);
    SetPreview(scene, kTightness, "heat", tightRange);
    Published const expression = Publish(scene);
    Check(expression.colors.size() == expression.curves, "a primitive expression preview is per strand");
    Check(Distinct(expression.colors) > 4, "the tightness expression varies over the groom");
    SetPreview(scene, kClump.AppendProperty(TfToken("usdGen:clump:amount")), "heat", tightRange);
    Published const attribute = Publish(scene);
    Check(attribute.colors.size() == expression.colors.size(),
          "the connected clump:amount previews per strand");
    size_t same = 0;
    for (size_t s = 0; s < attribute.colors.size() && s < expression.colors.size(); ++s)
        same += Near(attribute.colors[s], expression.colors[s], 1e-4f);
    Check(same == attribute.colors.size(),
          "clump:amount shows the values the clump was cooked with (" + std::to_string(same) + "/" +
              std::to_string(attribute.colors.size()) + ")");

    // 3. An authored, unconnected value.
    SetPreview(scene, kClump.AppendProperty(TfToken("usdGen:clump:size")), "gray");
    Published const literal = Publish(scene);
    double const size = 0.6;
    GfVec3f const expected = UsdGenPreviewColor(TfToken("gray"), GfVec2f(0, 1), &size, 1);
    Check(!literal.colors.empty() &&
          std::all_of(literal.colors.begin(), literal.colors.end(),
                      [&](GfVec3f const &c) { return Near(c, expected); }),
          "an unconnected attribute shows its authored value on every strand");

    // 4. A per-CV expression that drives nothing: root to tip, and an edit of
    //    its text recolours although no strand moves.
    {
        UsdEditContext session(scene.stage, scene.stage->GetSessionLayer());
        UsdPrim probe = scene.stage->DefinePrim(
            kDescription.AppendPath(SdfPath("Expressions/probe")), TfToken("UsdGenExpression"));
        probe.CreateAttribute(TfToken("usdGen:expr:source"), SdfValueTypeNames->String).Set(std::string("$t"));
        probe.CreateAttribute(TfToken("outputs:result"), SdfValueTypeNames->Float);
    }
    SdfPath const probe = kDescription.AppendPath(SdfPath("Expressions/probe"));
    scene.log->locators.clear();
    SetPreview(scene, probe, "gray", GfVec2f(0, 1), "point");
    // Not just a value dirty: Hydra must re-read the primvar's interpolation.
    Check(scene.log->locators.count("primvars") == 1 ||
          scene.log->locators.count("primvars/displayColor") == 1,
          "per-strand to per-CV colours dirty the tiles' primvars, so Hydra "
          "re-reads displayColor's interpolation");
    Published const alongT = Publish(scene);
    Check(alongT.colors.size() == alongT.points, "a point expression preview is per CV");
    if (alongT.colors.size() == alongT.points && alongT.points > 1) {
        Check(Near(alongT.colors.front(), GfVec3f(0.0f)), "the root of a strand reads $t = 0");
        size_t const perStrand = alongT.points / alongT.curves;
        Check(Near(alongT.colors[perStrand - 1], GfVec3f(1.0f), 1e-4f), "its tip reads $t = 1");
    }
    std::vector<GfVec3f> const beforeEdit = alongT.colors;
    {
        UsdEditContext session(scene.stage, scene.stage->GetSessionLayer());
        scene.stage->GetAttributeAtPath(probe.AppendProperty(TfToken("usdGen:expr:source")))
            .Set(std::string("1 - $t"));
    }
    scene.Settle();
    Published const edited = Publish(scene);
    Check(edited.colors.size() == beforeEdit.size() && !edited.colors.empty() &&
          Near(edited.colors.front(), GfVec3f(1.0f), 1e-4f),
          "editing the previewed expression recolours the strands");

    // 5. A source the preview cannot read: dark strands, still previewing.
    SetPreview(scene, SdfPath("/World/Skin"), "heat");
    Published const broken = Publish(scene);
    Check(!broken.colors.empty() &&
          std::all_of(broken.colors.begin(), broken.colors.end(),
                      [](GfVec3f const &c) { return Near(c, UsdGenPreviewMissingColor()); }),
          "an unreadable source shows the no-value colour");

    // 6. Flat shading swaps the material.
    SetPreview(scene, kRegionMap, "ids", GfVec2f(0, 1), "primitive", "flat");
    Published const flatShaded = Publish(scene);
    Check(flatShaded.bindings == std::set<SdfPath>{flat}, "flat shading binds material_preview_flat");
    Check(Has(flatShaded.children, flat) && !Has(flatShaded.children, lit) &&
          !scene.groom->GetPrim(lit).dataSource,
          "only the bound preview material exists");

    // 7. Clearing the preview restores the look.
    {
        UsdEditContext session(scene.stage, scene.stage->GetSessionLayer());
        scene.stage->GetPrimAtPath(kDescription)
            .GetRelationship(TfToken("usdGen:preview:source")).ClearTargets(true);
    }
    scene.Settle();
    Published const restored = Publish(scene);
    Check(restored.bindings == std::set<SdfPath>{storm}, "clearing the preview rebinds material_storm");
    Check(restored.colors == look.colors, "clearing the preview restores the look's colours");
    Check(!Has(restored.children, flat) && !scene.groom->GetPrim(flat).dataSource,
          "clearing the preview removes the preview material");
}

} // namespace

int main()
{
    usdGenTest::EnsureHydraPublishers();
    CheckColorMaps();
    CheckBuilder();
    CheckScene();
    UsdGenGroomSceneIndex::DrainRetired();
    std::printf("testUsdGenValuePreview: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
