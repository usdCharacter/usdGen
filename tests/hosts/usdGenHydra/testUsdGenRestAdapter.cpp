// testUsdGenRestAdapter — gate SI-7 rest leg (contract §3.3(2)):
// serve a Mesh with UsdGenRestAPI applied, assert usdGen/rest/points reads
// Default() (not time-varying), plus faceVertexCounts/Indices + st leaves;
// assert refusal (nullptr + warn) on a GeomSubset (R15).
// Unchanged by the locator fix (fixed usdGen/rest/* locators).
#include "utils/pxrInterop.h"
#include "usdGenImaging/restApiSchemaAdapter.h"
#include "usdGenImaging/usdGenRestApiDataSource.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usdImaging/usdImaging/dataSourceStageGlobals.h"

#include <cstdio>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
    else     { std::printf("ok:   %s\n", what.c_str()); }
}

class _TestGlobals : public UsdImagingDataSourceStageGlobals {
public:
    UsdTimeCode GetTime() const override { return UsdTimeCode::Default(); }
    void FlagAsTimeVarying(const SdfPath &,
                           const HdDataSourceLocator &) const override {}
    void FlagAsAssetPathDependent(const SdfPath &) const override {}
};

VtValue LeafValue(HdDataSourceBaseHandle ds)
{
    if (HdSampledDataSourceHandle s = HdSampledDataSource::Cast(ds))
        return s->GetValue(0.0);
    return VtValue();
}

HdDataSourceBaseHandle GetAt(HdContainerDataSourceHandle const &root,
                             HdDataSourceLocator const &loc)
{
    HdDataSourceBaseHandle cur = root;
    for (size_t i = 0; i < loc.GetElementCount(); ++i) {
        HdContainerDataSourceHandle c = HdContainerDataSource::Cast(cur);
        if (!c) return HdDataSourceBaseHandle();
        cur = c->Get(loc.GetElement(i));
        if (!cur) return HdDataSourceBaseHandle();
    }
    return cur;
}

}  // namespace

int main()
{
    UsdStageRefPtr stage = UsdStage::CreateInMemory("si7rest");
    _TestGlobals globals;

    // Mesh with distinct Default vs time-sampled points and normals.
    SdfPath meshPath("/scalp");
    UsdGeomMesh mesh = UsdGeomMesh::Define(stage, meshPath);
    VtVec3fArray restPts(4), defPts(4);
    restPts[0] = GfVec3f(0, 0, 0); restPts[1] = GfVec3f(1, 0, 0);
    restPts[2] = GfVec3f(1, 1, 0); restPts[3] = GfVec3f(0, 1, 0);
    for (size_t i = 0; i < 4; ++i) defPts[i] = restPts[i] + GfVec3f(0, 0, 5);
    mesh.CreatePointsAttr(VtValue(restPts));
    mesh.GetPointsAttr().Set(VtValue(defPts), UsdTimeCode(24.0));
    VtIntArray fvc(1, 4);
    VtIntArray fvi(4);
    for (int i = 0; i < 4; ++i) fvi[i] = i;
    mesh.CreateFaceVertexCountsAttr(VtValue(fvc));
    mesh.CreateFaceVertexIndicesAttr(VtValue(fvi));
    VtVec3fArray restNormals(4), sampledNormals(4);
    for (size_t i = 0; i < 4; ++i) {
        restNormals[i] = GfVec3f(0, 0, 1);
        sampledNormals[i] = GfVec3f(0, 1, 0);
    }
    mesh.CreateNormalsAttr(VtValue(restNormals));
    mesh.SetNormalsInterpolation(TfToken("vertex"));
    mesh.GetNormalsAttr().Set(VtValue(sampledNormals), UsdTimeCode(24.0));
    UsdPrim meshPrim = stage->GetPrimAtPath(meshPath);
    meshPrim.AddAppliedSchema(TfToken("UsdGenRestAPI"));

    // GeomSubset of the same mesh (R15 refusal case).
    SdfPath subsetPath("/scalp/subset");
    UsdGeomSubset subset = UsdGeomSubset::Define(stage, subsetPath);
    subset.CreateFamilyNameAttr(VtValue(TfToken("materialBind")));
    subset.CreateIndicesAttr(VtValue(VtIntArray{0}));
    UsdPrim subsetPrim = stage->GetPrimAtPath(subsetPath);
    subsetPrim.AddAppliedSchema(TfToken("UsdGenRestAPI"));

    // ---- factory: Mesh serves rest at Default() -------------------------
    HdContainerDataSourceHandle c =
        usdGenImaging::UsdGenRestApiContainerFactory(meshPrim, globals);
    Check(bool(c), "rest container served for Mesh");
    (void)c;
    if (c) {
        VtValue pts = LeafValue(GetAt(
            c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"),
                                   TfToken("points"))));
        Check(pts.IsHolding<VtVec3fArray>() &&
                  pts.UncheckedGet<VtVec3fArray>() == restPts,
              "usdGen/rest/points reads Default(), not time samples");
        VtValue counts = LeafValue(GetAt(
            c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"),
                                   TfToken("faceVertexCounts"))));
        Check(counts.IsHolding<VtIntArray>() &&
                  counts.UncheckedGet<VtIntArray>() == fvc,
              "usdGen/rest/faceVertexCounts served");
        VtValue indices = LeafValue(GetAt(
            c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"),
                                   TfToken("faceVertexIndices"))));
        Check(indices.IsHolding<VtIntArray>() &&
                  indices.UncheckedGet<VtIntArray>() == fvi,
              "usdGen/rest/faceVertexIndices served");
        VtValue normals = LeafValue(GetAt(
            c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"),
                                   TfToken("normals"))));
        Check(normals.IsHolding<VtVec3fArray>() &&
                  normals.UncheckedGet<VtVec3fArray>() == restNormals,
              "usdGen/rest/normals reads Default(), not time samples");
        VtValue normalsInterpolation = LeafValue(GetAt(
            c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"),
                                   TfToken("normalsInterpolation"))));
        Check(normalsInterpolation.IsHolding<TfToken>() &&
                  normalsInterpolation.UncheckedGet<TfToken>() == TfToken("vertex"),
              "usdGen/rest/normalsInterpolation is paired with normals");
        auto pointsLeaf = HdSampledDataSource::Cast(GetAt(c, HdDataSourceLocator(
            TfToken("usdGen"), TfToken("rest"), TfToken("points"))));
        std::vector<HdSampledDataSource::Time> times;
        Check(pointsLeaf && !pointsLeaf->GetContributingSampleTimesForInterval(0, 24, &times),
              "rest leaf reports no time-varying samples");

        // The retained handle must re-read the composed Default opinion after
        // a live edit, while still remaining uniform from Hydra's time view.
        mesh.GetPointsAttr().Set(VtValue(defPts), UsdTimeCode::Default());
        Check(LeafValue(GetAt(c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"), TfToken("points")))).
                  UncheckedGet<VtVec3fArray>() == defPts,
              "cached rest handle follows Default-time points edit");
        mesh.GetFaceVertexCountsAttr().Set(VtValue(VtIntArray{3}), UsdTimeCode::Default());
        Check(LeafValue(GetAt(c, HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"), TfToken("faceVertexCounts")))).
                  UncheckedGet<VtIntArray>() == VtIntArray{3},
              "cached rest handle follows topology edit");
        UsdGeomPrimvar explicitEmpty = UsdGeomPrimvarsAPI(meshPrim).CreatePrimvar(
            TfToken("rest"), SdfValueTypeNames->Point3fArray);
        explicitEmpty.Set(VtValue(VtVec3fArray{}), UsdTimeCode::Default());
        VtValue emptyRest = LeafValue(GetAt(c, HdDataSourceLocator(
            TfToken("usdGen"), TfToken("rest"), TfToken("points"))));
        Check(emptyRest.IsHolding<VtVec3fArray>() && emptyRest.UncheckedGet<VtVec3fArray>().empty(),
              "explicit empty rest opinion wins over points fallback");
        auto sourceAttr = meshPrim.CreateAttribute(TfToken("usdGen:rest:source"), SdfValueTypeNames->Token);
        sourceAttr.Set(TfToken("asset"));
        Check(pointsLeaf->GetValue(0).IsEmpty(), "unsupported rest asset source fails closed through cached leaf");
        sourceAttr.Set(TfToken("default"));
        explicitEmpty.Set(VtValue(restPts), UsdTimeCode::Default());
        Check(pointsLeaf->GetValue(0) == VtValue(restPts), "cached leaf follows repaired authored rest");

    }

    // A time-sampled-only normals attribute has no immutable rest normal.
    // Its empty rest leaf intentionally selects geometric fallback; it must
    // not sample the current frame or shutter offset zero.
    {
        UsdGeomMesh sampledOnly = UsdGeomMesh::Define(stage, SdfPath("/sampledOnly"));
        sampledOnly.CreatePointsAttr(VtValue(restPts));
        sampledOnly.CreateFaceVertexCountsAttr(VtValue(fvc));
        sampledOnly.CreateFaceVertexIndicesAttr(VtValue(fvi));
        sampledOnly.CreateNormalsAttr();
        sampledOnly.GetNormalsAttr().Set(VtValue(sampledNormals), UsdTimeCode(24.0));
        sampledOnly.SetNormalsInterpolation(TfToken("vertex"));
        UsdPrim sampledOnlyPrim = sampledOnly.GetPrim();
        sampledOnlyPrim.AddAppliedSchema(TfToken("UsdGenRestAPI"));
        HdContainerDataSourceHandle sampledOnlyRest =
            usdGenImaging::UsdGenRestApiContainerFactory(sampledOnlyPrim, globals);
        VtValue normals = LeafValue(GetAt(sampledOnlyRest, HdDataSourceLocator(
            TfToken("usdGen"), TfToken("rest"), TfToken("normals"))));
        Check(normals.IsEmpty() ||
                  (normals.IsHolding<VtVec3fArray>() &&
                   normals.UncheckedGet<VtVec3fArray>().empty()),
              "time-sampled-only normals have no Default-time rest normal");
    }

    // ---- factory: GeomSubset refused (R15) --------------------------------
    {
        // R15 refusal warns once per prim (TF_WARN) — expected noise.
        HdContainerDataSourceHandle sc =
            usdGenImaging::UsdGenRestApiContainerFactory(subsetPrim, globals);
        Check(!sc, "rest container refused (nullptr) on GeomSubset");
    }

    // ---- adapter wiring: subprim data + invalidation ----------------------
    {
        UsdGenRestAPIAdapter adapter;
        HdContainerDataSourceHandle ac = adapter.GetImagingSubprimData(
            meshPrim, TfToken(), TfToken(), globals);
        Check(bool(ac), "adapter serves subprim data for Mesh");
        HdDataSourceLocatorSet inv = adapter.InvalidateImagingSubprim(
            meshPrim, TfToken(), TfToken(),
            TfTokenVector{TfToken("points")},
            UsdImagingPropertyInvalidationType::Update);
        Check(inv.Contains(HdDataSourceLocator(
                  TfToken("usdGen"), TfToken("rest"), TfToken("points"))),
              "points edit invalidates usdGen/rest/points");
        for (auto const& property : {TfToken("primvars:rest"), TfToken("usdGen:rest:source")}) {
            auto edited = adapter.InvalidateImagingSubprim(meshPrim, TfToken(), TfToken(),
                TfTokenVector{property}, UsdImagingPropertyInvalidationType::Update);
            Check(edited.Contains(HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"), TfToken("points"))),
                  property.GetString() + " edit invalidates rest points");
        }
        for (auto const& property : {TfToken("faceVertexCounts"), TfToken("faceVertexIndices")}) {
            auto edited = adapter.InvalidateImagingSubprim(meshPrim, TfToken(), TfToken(),
                TfTokenVector{property}, UsdImagingPropertyInvalidationType::Update);
            Check(edited.Contains(HdDataSourceLocator(TfToken("usdGen"), TfToken("rest"), property)),
                  property.GetString() + " edit invalidates precise rest topology leaf");
        }
        auto normalEdited = adapter.InvalidateImagingSubprim(
            meshPrim, TfToken(), TfToken(), TfTokenVector{TfToken("normals")},
            UsdImagingPropertyInvalidationType::Update);
        Check(normalEdited.Contains(HdDataSourceLocator(
                  TfToken("usdGen"), TfToken("rest"), TfToken("normals"))) &&
                  normalEdited.Contains(HdDataSourceLocator(
                  TfToken("usdGen"), TfToken("rest"), TfToken("normalsInterpolation"))),
              "normals edit invalidates paired rest normal leaves");
        auto interpolationEdited = adapter.InvalidateImagingSubprim(
            meshPrim, TfToken(), TfToken(), TfTokenVector{TfToken("normalsInterpolation")},
            UsdImagingPropertyInvalidationType::Update);
        Check(interpolationEdited.Contains(HdDataSourceLocator(
                  TfToken("usdGen"), TfToken("rest"), TfToken("normalsInterpolation"))),
              "normalsInterpolation edit invalidates its rest leaf");
    }

    std::printf("testUsdGenRestAdapter: %s (%d failures)\n",
                g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
