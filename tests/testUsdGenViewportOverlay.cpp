// testUsdGenViewportOverlay (T1, no GPU) — what the synthetic tiles must
// inherit from their owning UsdGenDescription for the Storm viewport to
// behave like a native UsdGeomBasisCurves:
//
//   * displayStyle  — the usdview complexity slider is pushed as
//     HdsiLegacyDisplayStyleOverrideSceneIndex::SetRefineLevelFallback, an
//     UNDERLAY on the stage prims only. The synthetic tiles are not in that
//     index's input, so UsdGenGroomSceneIndex::GetPrim must overlay the
//     Description's displayStyle on top of the published value (upstream
//     first: the slider wins, the publication is the no-opinion fallback),
//     UNCLAMPED so a tile picks exactly the repr a native UsdGeomBasisCurves
//     would, and must forward the invalidation to every tile.
//   * selections    — UsdImagingSelectionSceneIndex stamps `selections` on
//     stage prims upstream of us; HdxSelectionTracker reads them off the
//     TERMINAL index, downstream of us. Republishing the Description's
//     container verbatim onto every tile is what makes the highlight appear,
//     and forwarding the dirty is what makes deselect work.
//   * material_storm — with no authored material binding the tiles used to
//     carry no materialBindings at all, so Storm fell back to flat
//     displayColor shading. A tile now binds the synthetic
//     <description>/__usdGenRender/material_storm, which must be a reachable
//     `material` prim with TWO networks: the `glslfx` one, whose surface
//     terminal is the Sdr id
//     UsdGenTilePublisher::DefaultMaterialIdentifier() (the host renderer-parity WS1: UsdGenHairStrandsTranslucent),
//     and the universal one, a UsdPreviewSurface reading displayColor
//     through a primvar reader, which is what hdMoonray (and every other
//     non-Storm delegate) falls back to (docs/moonray-fur.md).
//   * materialBindings — the binding is a FUNCTION of the inherited
//     displayStyle, which is what buys complexity parity with a native
//     UsdGeomBasisCurves. refineLevel 0 carries NO binding at all (the hair
//     shader reads inData.Neye, which Storm's WIRE repr does not declare, so
//     it would fail to COMPILE, and usdGen cannot vouch for someone else's
//     shader there either); refineLevel 1 hides the synthetic default only,
//     so Medium is Storm's flat displayColor shading exactly like a native
//     curve, while an AUTHORED binding wins from 1 up; refineLevel 2 and 3
//     are bound as published. The masking is an HdBlockDataSource, because an
//     empty container would merge with the publisher's under the overlay.
//     The invalidation carries the materialBindings locator too, or moving
//     the slider live in usdview re-reprs without rebinding.
//
// The upstream opinions are injected with a tiny filtering scene index
// spliced between the UsdImaging chain and the groom index, which is exactly
// where HdsiLegacyDisplayStyleOverrideSceneIndex and
// UsdImagingSelectionSceneIndex sit in UsdImagingGLEngine.

#include "utils/pxrInterop.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "usdGenHydraCurves/usdGenTilePublisher.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/imaging/hd/filteringSceneIndex.h"
#include "pxr/imaging/hd/legacyDisplayStyleSchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/materialConnectionSchema.h"
#include "pxr/imaging/hd/materialNetworkSchema.h"
#include "pxr/imaging/hd/materialNodeParameterSchema.h"
#include "pxr/imaging/hd/materialNodeSchema.h"
#include "pxr/imaging/hd/materialSchema.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/imaging/hd/selectionSchema.h"
#include "pxr/imaging/hd/selectionsSchema.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdShade/materialBindingAPI.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int failures = 0;

void Check(bool value, std::string const &message)
{
    if (!value) {
        ++failures;
        std::printf("FAIL: %s\n", message.c_str());
    } else {
        std::printf("ok:   %s\n", message.c_str());
    }
}

// The CPU reference groom of tests/testUsdGenScenePublication.cpp: an
// untyped Scope parent, so no CUDA backend schema fallback applies and the
// publication lane is the stock-Storm one.
char const *kFixture = R"USDA(#usda 1.0
def Mesh "Scalp" (prepend apiSchemas = ["UsdGenRestAPI"])
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)]
    point3f[] primvars:rest = [(-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)] (interpolation = "vertex")
    texCoord2f[] primvars:st = [(0,0), (1,0), (1,1), (0,1)] (interpolation = "vertex")
}
def Scope "Looks"
{
    def Material "Fur"
    {
        token outputs:surface.connect = </Looks/Fur/preview.outputs:surface>
        def Shader "preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            token outputs:surface
        }
    }
}
def Scope "Groom"
{
    def UsdGenDescription "hair"
    {
        def Scope "Ops"
        {
            def UsdGenWidth "width"
            {
                float usdGen:width = 0.02
                bool usdGen:replace = true
                uniform token usdGen:width:interpolation = "linear"
                float2[] usdGen:width:knots = [(0, 1), (1, 1)]
            }
            def UsdGenCurveSource "source"
            {
                rel usdGen:surface = </Scalp>
                rel usdGen:curves = </Groom/hair/curves>
                int usdGen:resampleTo = 4
                uniform token usdGen:rebind = "never"
            }
        }
        def BasisCurves "curves" (
            prepend apiSchemas = ["UsdGenCurveAPI"]
        )
        {
            uniform token type = "cubic"
            uniform token basis = "bspline"
            uniform token wrap = "pinned"
            int[] curveVertexCounts = [4]
            point3f[] points = [(0,0,0), (0,1,0), (0,2,0), (0,3,0)]
            uniform token primvars:usdGen:role = "hair"
            uint64[] primvars:usdGen:curveId = [1] (interpolation = "uniform")
            int[] primvars:skinprim = [0] (interpolation = "uniform")
            texCoord2f[] primvars:skinprimuv = [(0.5,0.5)] (interpolation = "uniform")
            matrix4d[] primvars:usdGen:rootFrame = [((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1))] (interpolation = "uniform")
        }
    }
}
)USDA";

// Stands in for HdsiLegacyDisplayStyleOverrideSceneIndex /
// UsdImagingSelectionSceneIndex: overlays one retained container onto one
// prim of the input and announces the change, nothing else.
class StampSceneIndex final : public HdSingleInputFilteringSceneIndexBase
{
public:
    static TfRefPtr<StampSceneIndex> New(HdSceneIndexBaseRefPtr const &input)
    {
        return TfCreateRefPtr(new StampSceneIndex(input));
    }

    HdSceneIndexPrim GetPrim(SdfPath const &path) const override
    {
        HdSceneIndexPrim prim = _GetInputSceneIndex()->GetPrim(path);
        if (_overlay && path == _target && prim.dataSource) {
            prim.dataSource =
                HdOverlayContainerDataSource::New(_overlay, prim.dataSource);
        }
        return prim;
    }

    SdfPathVector GetChildPrimPaths(SdfPath const &path) const override
    {
        return _GetInputSceneIndex()->GetChildPrimPaths(path);
    }

    void Stamp(SdfPath const &target,
               HdContainerDataSourceHandle const &overlay,
               HdDataSourceLocatorSet const &locators)
    {
        _target = target;
        _overlay = overlay;
        _SendPrimsDirtied({{target, locators}});
    }

protected:
    explicit StampSceneIndex(HdSceneIndexBaseRefPtr const &input)
        : HdSingleInputFilteringSceneIndexBase(input) {}

    void _PrimsAdded(
        HdSceneIndexBase const &,
        HdSceneIndexObserver::AddedPrimEntries const &entries) override
    { _SendPrimsAdded(entries); }
    void _PrimsRemoved(
        HdSceneIndexBase const &,
        HdSceneIndexObserver::RemovedPrimEntries const &entries) override
    { _SendPrimsRemoved(entries); }
    void _PrimsDirtied(
        HdSceneIndexBase const &,
        HdSceneIndexObserver::DirtiedPrimEntries const &entries) override
    { _SendPrimsDirtied(entries); }

private:
    SdfPath _target;
    HdContainerDataSourceHandle _overlay;
};

// Records every PrimsDirtied entry so the per-tile fan-out can be counted
// exactly (one entry per tile, no duplicates).
class DirtyRecorder final : public HdSceneIndexObserver
{
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
    // Entries that cover `locator`. The requirement is EXACTLY ONE per tile:
    // no tile may be left un-invalidated, and none may be notified twice.
    // The set may be wider than `locator` -- the same upstream dirty also
    // re-cooks the groom, and the republication's own resync entry covers
    // everything; the targeted displayStyle/selections locators are what the
    // tile gets when no republication intervenes.
    size_t CountFor(SdfPath const &path, HdDataSourceLocator const &locator) const
    {
        size_t n = 0;
        for (auto const &entry : dirtied)
            if (entry.first == path && entry.second.Intersects(locator)) ++n;
        return n;
    }
};

// The tile's all-purpose material, or an empty path when it carries none.
SdfPath TileMaterial(HdSceneIndexBase const &index, SdfPath const &tile)
{
    HdMaterialBindingsSchema bindings =
        HdMaterialBindingsSchema::GetFromParent(index.GetPrim(tile).dataSource);
    if (!bindings.IsDefined()) return SdfPath();
    if (HdPathDataSourceHandle const path =
            bindings.GetMaterialBinding().GetPath())
        return path->GetTypedValue(0);
    return SdfPath();
}

VtValue NodeParam(HdMaterialNodeSchema const &node, char const *name)
{
    HdMaterialNodeParameterSchema const param =
        node.GetParameters().Get(TfToken(name));
    HdSampledDataSourceHandle const sampled =
        param.IsDefined() ? param.GetValue() : nullptr;
    return sampled ? sampled->GetValue(0.0f) : VtValue();
}

int TileRefineLevel(HdSceneIndexBase const &index, SdfPath const &tile)
{
    HdSceneIndexPrim const prim = index.GetPrim(tile);
    HdLegacyDisplayStyleSchema style =
        HdLegacyDisplayStyleSchema::GetFromParent(prim.dataSource);
    if (!style.IsDefined() || !style.GetRefineLevel()) return -1;
    return style.GetRefineLevel()->GetTypedValue(0);
}

HdContainerDataSourceHandle DisplayStyleOverlay(int refineLevel)
{
    return HdRetainedContainerDataSource::New(
        HdLegacyDisplayStyleSchema::GetSchemaToken(),
        HdLegacyDisplayStyleSchema::Builder()
            .SetRefineLevel(
                HdRetainedTypedSampledDataSource<int>::New(refineLevel))
            .Build());
}

HdContainerDataSourceHandle SelectionsOverlay()
{
    HdDataSourceBaseHandle const selection =
        HdSelectionSchema::Builder()
            .SetFullySelected(
                HdRetainedTypedSampledDataSource<bool>::New(true))
            .Build();
    return HdRetainedContainerDataSource::New(
        HdSelectionsSchema::GetSchemaToken(),
        HdRetainedSmallVectorDataSource::New(1, &selection));
}

} // namespace

int main()
{
    usdGenTest::EnsureHydraPublishers();

    UsdStageRefPtr stage = UsdStage::CreateInMemory("viewport-overlay");
    stage->GetRootLayer()->ImportFromString(kFixture);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(info);
    TfRefPtr<StampSceneIndex> stamp =
        StampSceneIndex::New(indices.finalSceneIndex);
    HdSceneIndexBaseRefPtr groomBase = UsdGenGroomSceneIndex::New(stamp);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(groomBase.operator->());
    Check(owner != nullptr, "groom scene index is available");
    if (!owner) return 1;

    DirtyRecorder recorder;
    groomBase->AddObserver(TfCreateWeakPtr(&recorder));

    SdfPath const description("/Groom/hair");
    SdfPath const render = description.AppendChild(TfToken("__usdGenRender"));
    SdfPath const material =
        usdGenImaging::UsdGenTilePublisher::MaterialPath(description);

    owner->Synchronize();

    // A ray-tracing frontend shares the groom data but must not inherit
    // Storm's wire/ribbon material suppression at Low and Medium.
    HdSceneIndexBaseRefPtr traced = UsdGenGroomSceneIndex::New(stamp, 0, false, false);
    auto *tracedOwner = dynamic_cast<UsdGenGroomSceneIndex *>(traced.operator->());
    tracedOwner->Synchronize();

    SdfPathVector tiles;
    for (SdfPath const &child : groomBase->GetChildPrimPaths(render)) {
        if (child.GetName().rfind("tile_", 0) == 0) tiles.push_back(child);
    }
    Check(!tiles.empty(), "the groom publishes at least one synthetic tile");
    if (tiles.empty()) return 1;
    std::printf("published tiles: %zu\n", tiles.size());

    // ---- the synthetic default material ---------------------------------
    {
        SdfPathVector const children = groomBase->GetChildPrimPaths(render);
        Check(std::find(children.begin(), children.end(), material) !=
                  children.end(),
              "material_storm is reachable from GetChildPrimPaths(render)");
        HdSceneIndexPrim const prim = groomBase->GetPrim(material);
        Check(prim.primType == TfToken("material"),
              "material_storm has Hydra prim type 'material'");
        HdMaterialSchema const matSchema =
            HdMaterialSchema::GetFromParent(prim.dataSource);
        // The glslfx network is Storm's hair shading, unchanged: Storm
        // prefers its glslfx render context over the universal one.
        HdMaterialNetworkSchema const glslfx =
            matSchema.GetMaterialNetwork(TfToken("glslfx"));
        Check(glslfx.IsDefined(),
              "material_storm carries a glslfx-render-context network");
        HdMaterialNodeSchema const hair =
            glslfx.GetNodes().Get(TfToken("surface"));
        TfToken const &expected =
            ::usdGenImaging::UsdGenTilePublisher::DefaultMaterialIdentifier();
        Check(hair.IsDefined() && hair.GetNodeIdentifier() &&
                  hair.GetNodeIdentifier()->GetTypedValue(0) == expected,
              ("glslfx surface node binds the Sdr identifier " +
               expected.GetString()).c_str());
        HdMaterialConnectionSchema const hairTerminal =
            glslfx.GetTerminals().Get(TfToken("surface"));
        Check(hairTerminal.IsDefined() && hairTerminal.GetUpstreamNodePath() &&
                  hairTerminal.GetUpstreamNodePath()->GetTypedValue(0) ==
                      TfToken("surface"),
              "the glslfx surface terminal points at the surface node");
        // The fixture authors no look: default tip color flows through.
        VtValue const tipColor = NodeParam(hair, "tipColor");
        Check(tipColor.IsHolding<GfVec3f>() &&
                  tipColor.UncheckedGet<GfVec3f>() ==
                      GfVec3f(0.210f, 0.115f, 0.045f),
              "glslfx hair node carries the default tip color");
        // The universal network is the renderer-agnostic fallback
        // (docs/moonray-fur.md): hdMoonray reads the universal context
        // when no moonray context exists, and a UsdGenHairStrands node
        // there makes it look for an RDL DSO of that name. The preview
        // instead reads the per-curve displayColor the tiles bake.
        HdMaterialNetworkSchema const universal =
            matSchema.GetMaterialNetwork(
                HdMaterialSchemaTokens->universalRenderContext);
        Check(universal.IsDefined(),
              "material_storm carries a universal-render-context network");
        HdMaterialNodeSchema const preview =
            universal.GetNodes().Get(TfToken("surface"));
        Check(preview.IsDefined() && preview.GetNodeIdentifier() &&
                  preview.GetNodeIdentifier()->GetTypedValue(0) ==
                      TfToken("UsdPreviewSurface"),
              "universal surface node is a UsdPreviewSurface");
        HdMaterialConnectionSchema const previewTerminal =
            universal.GetTerminals().Get(TfToken("surface"));
        Check(previewTerminal.IsDefined() &&
                  previewTerminal.GetUpstreamNodePath() &&
                  previewTerminal.GetUpstreamNodePath()->GetTypedValue(0) ==
                      TfToken("surface"),
              "the universal surface terminal points at the surface node");
        HdMaterialNodeSchema const reader =
            universal.GetNodes().Get(TfToken("DisplayColor"));
        Check(reader.IsDefined() && reader.GetNodeIdentifier() &&
                  reader.GetNodeIdentifier()->GetTypedValue(0) ==
                      TfToken("UsdPrimvarReader_float3"),
              "universal network reads displayColor through a primvar reader");
        VtValue const varname = NodeParam(reader, "varname");
        Check(varname.IsHolding<TfToken>() &&
                  varname.UncheckedGet<TfToken>() == TfToken("displayColor"),
              "the reader reads varname displayColor");
        VtValue const fallback = NodeParam(reader, "fallback");
        Check(fallback.IsHolding<GfVec3f>() &&
                  fallback.UncheckedGet<GfVec3f>() ==
                      GfVec3f(0.035f, 0.018f, 0.008f),
              "the reader falls back to the default root color");
        HdMaterialConnectionVectorSchema const conns =
            preview.GetInputConnections().Get(TfToken("diffuseColor"));
        HdMaterialConnectionSchema const diffuseColor =
            conns.IsDefined() && conns.GetNumElements() > 0
                ? conns.GetElement(0)
                : HdMaterialConnectionSchema(HdContainerDataSourceHandle());
        Check(diffuseColor.IsDefined() && diffuseColor.GetUpstreamNodePath() &&
                  diffuseColor.GetUpstreamNodePath()->GetTypedValue(0) ==
                      TfToken("DisplayColor") &&
                  diffuseColor.GetUpstreamNodeOutputName() &&
                  diffuseColor.GetUpstreamNodeOutputName()->GetTypedValue(0) ==
                      TfToken("result"),
              "preview diffuseColor connects to the reader result");
        VtValue const roughness = NodeParam(preview, "roughness");
        Check(roughness.IsHolding<float>() &&
                  roughness.UncheckedGet<float>() == 0.6f,
              "preview roughness is the documented constant");

        bool bound = true;
        for (SdfPath const &tile : tiles)
            if (TileMaterial(*groomBase, tile) != material) bound = false;
        Check(bound,
              "every tile binds material_storm when nothing is authored");
    }

    // ---- refineLevel: no upstream opinion -> the published fallback ------
    {
        bool fallback = true;
        for (SdfPath const &tile : tiles)
            if (TileRefineLevel(*groomBase, tile) != 2) fallback = false;
        Check(fallback,
              "no host opinion -> tiles report the published refineLevel 2");
    }

    // ---- refineLevel: the slider reaches the tile UNCLAMPED, and the
    // default binding follows it ------------------------------------------
    // Low (0) and Medium (1) must look like an unbound native
    // UsdGeomBasisCurves: the repr is whatever the slider asked for and the
    // synthetic hair material is hidden, so Storm shades displayColor flat.
    // High (2) and Very High (3) keep the binding.
    for (auto const &[upstream, wantBound] :
         {std::pair{3, true}, std::pair{2, true}, std::pair{1, false},
          std::pair{0, false}}) {
        recorder.Clear();
        stamp->Stamp(description, DisplayStyleOverlay(upstream),
                     HdDataSourceLocatorSet{
                         HdLegacyDisplayStyleSchema::GetDefaultLocator()});
        owner->Synchronize();

        std::string const at = " at refineLevel " + std::to_string(upstream);
        tracedOwner->Synchronize();
        bool tracedCorrect = true;
        for (SdfPath const &tile : tiles)
            if (TileMaterial(*traced, tile) != material ||
                TileRefineLevel(*traced, tile) != upstream) tracedCorrect = false;
        Check(tracedCorrect, "ray tracing retains universal material and requested refinement" + at);
        bool followed = true, correct = true;
        for (SdfPath const &tile : tiles) {
            if (TileRefineLevel(*groomBase, tile) != upstream) followed = false;
            SdfPath const got = TileMaterial(*groomBase, tile);
            if (got != (wantBound ? material : SdfPath())) correct = false;
        }
        Check(followed,
              "upstream displayStyle/refineLevel " + std::to_string(upstream) +
                  " reaches every tile unclamped");
        Check(correct,
              wantBound ? "every tile still binds material_storm" + at
                        : "no tile carries any materialBindings" + at);

        for (auto const &[locator, label] :
             {std::pair{HdLegacyDisplayStyleSchema::GetDefaultLocator(),
                        "displayStyle"},
              std::pair{HdMaterialBindingsSchema::GetDefaultLocator(),
                        "materialBindings"}}) {
            bool once = true;
            for (SdfPath const &tile : tiles) {
                size_t const n = recorder.CountFor(tile, locator);
                if (n != 1) {
                    once = false;
                    std::printf("  %s: %zu %s dirty entries\n", tile.GetText(),
                                n, label);
                }
            }
            Check(once, std::string("exactly one ") + label +
                            " PrimsDirtied entry per tile" + at);
        }
    }

    // ---- an AUTHORED binding wins from refineLevel 1 up, and is hidden at
    // 0 with the rest (the wire repr is drawn unbound, whoever authored the
    // material) ------------------------------------------------------------
    {
        SdfPath const authored("/Looks/Fur");
        UsdShadeMaterialBindingAPI::Apply(stage->GetPrimAtPath(description))
            .Bind(UsdShadeMaterial(stage->GetPrimAtPath(authored)));
        indices.stageSceneIndex->ApplyPendingUpdates();

        for (auto const &[upstream, want] :
             {std::pair{3, true}, std::pair{2, true}, std::pair{1, true},
              std::pair{0, false}}) {
            stamp->Stamp(description, DisplayStyleOverlay(upstream),
                         HdDataSourceLocatorSet{
                             HdLegacyDisplayStyleSchema::GetDefaultLocator()});
            owner->Synchronize();

            tracedOwner->Synchronize();
            bool tracedCorrect = true;
            for (SdfPath const &tile : tiles)
                if (TileMaterial(*traced, tile) != authored ||
                    TileRefineLevel(*traced, tile) != upstream) tracedCorrect = false;
            Check(tracedCorrect, "ray tracing retains authored material at refineLevel " +
                  std::to_string(upstream));
            bool correct = true;
            for (SdfPath const &tile : tiles)
                if (TileMaterial(*groomBase, tile) !=
                    (want ? authored : SdfPath()))
                    correct = false;
            Check(correct,
                  std::string(want ? "the authored binding reaches every tile"
                                   : "the authored binding is hidden too") +
                      " at refineLevel " + std::to_string(upstream));
        }
    }

    // ---- selections: forwarded verbatim, and invalidated ------------------
    {
        recorder.Clear();
        HdContainerDataSourceHandle const overlay = SelectionsOverlay();
        stamp->Stamp(description, overlay,
                     HdDataSourceLocatorSet{
                         HdSelectionsSchema::GetDefaultLocator()});
        owner->Synchronize();

        HdDataSourceBaseHandle const authored =
            overlay->Get(HdSelectionsSchema::GetSchemaToken());
        bool verbatim = true, once = true;
        for (SdfPath const &tile : tiles) {
            HdSceneIndexPrim const prim = groomBase->GetPrim(tile);
            HdDataSourceBaseHandle const got =
                prim.dataSource
                    ? prim.dataSource->Get(HdSelectionsSchema::GetSchemaToken())
                    : nullptr;
            if (got != authored) verbatim = false;
            if (recorder.CountFor(
                    tile, HdSelectionsSchema::GetDefaultLocator()) != 1)
                once = false;
        }
        Check(verbatim,
              "the Description's selections container reaches every tile "
              "verbatim (same data source)");
        Check(once, "exactly one selections PrimsDirtied entry per tile");

        // Deselect: clearing the upstream opinion must reach the tiles too,
        // or the highlight sticks.
        recorder.Clear();
        stamp->Stamp(description, nullptr,
                     HdDataSourceLocatorSet{
                         HdSelectionsSchema::GetDefaultLocator()});
        owner->Synchronize();
        bool cleared = true, clearedOnce = true;
        for (SdfPath const &tile : tiles) {
            HdSceneIndexPrim const prim = groomBase->GetPrim(tile);
            if (prim.dataSource &&
                prim.dataSource->Get(HdSelectionsSchema::GetSchemaToken()))
                cleared = false;
            if (recorder.CountFor(
                    tile, HdSelectionsSchema::GetDefaultLocator()) != 1)
                clearedOnce = false;
        }
        Check(cleared, "deselect removes selections from every tile");
        Check(clearedOnce,
              "deselect emits exactly one selections PrimsDirtied per tile");
    }

    groomBase->RemoveObserver(TfCreateWeakPtr(&recorder));
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
