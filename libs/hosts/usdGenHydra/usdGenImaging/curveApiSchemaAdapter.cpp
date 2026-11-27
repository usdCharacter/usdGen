// UsdGenCurveAPIAdapter: live C3 rest provenance for BasisCurves.
#include "usdGenImaging/curveApiSchemaAdapter.h"

#include "pxr/base/tf/type.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/primvar.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"

#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

class _CurveRestValueDataSource final : public HdSampledDataSource
{
public:
    HD_DECLARE_DATASOURCE(_CurveRestValueDataSource);

    _CurveRestValueDataSource(UsdPrim const &prim, bool provenance)
        : _prim(prim), _provenance(provenance) {}

    VtValue GetValue(Time) override
    {
        UsdGeomBasisCurves curves(_prim);
        UsdGeomPrimvar rest =
            UsdGeomPrimvarsAPI(_prim).GetPrimvar(TfToken("rest"));
        UsdAttribute const restAttr = rest ? rest.GetAttr() : UsdAttribute();
        bool const hasAuthoredRest =
            restAttr && restAttr.GetResolveInfo().HasAuthoredValueOpinion();

        if (_provenance) {
            return VtValue(hasAuthoredRest);
        }

        // An authored rest opinion wins even when it is an explicitly empty
        // array.  Do not replace it with the current points fallback.
        VtValue value;
        if (hasAuthoredRest) {
            if (restAttr.Get(&value, UsdTimeCode::Default()) &&
                value.IsHolding<VtVec3fArray>()) {
                return value;
            }
            return VtValue(VtVec3fArray());
        }

        // C3 has no authored rest primvar: bind rest to the curve's
        // Default-time points, never to the current scene time.
        if (curves && curves.GetPointsAttr().Get(
                &value, UsdTimeCode::Default()) &&
            value.IsHolding<VtVec3fArray>()) {
            return value;
        }
        return VtValue(VtVec3fArray());
    }

    bool GetContributingSampleTimesForInterval(
        Time, Time, std::vector<Time> *) override
    {
        // The source is deliberately uniform from Hydra's point of view;
        // Default-time USD reads are performed dynamically by GetValue.
        return false;
    }

private:
    UsdPrim _prim;
    bool _provenance;
};

// Surface-cage controls are authored ordinary USD attributes on the sparse
// BasisCurves source. Keep the source live, like CurveRest, so a session does
// not retain a first-frame copy when an author edits a cage array.
class _CurveSurfaceCageValueDataSource final : public HdSampledDataSource
{
public:
    HD_DECLARE_DATASOURCE(_CurveSurfaceCageValueDataSource);

    _CurveSurfaceCageValueDataSource(UsdPrim const &prim, TfToken const &name)
        : _prim(prim), _name(name) {}

    VtValue GetValue(Time) override
    {
        VtValue value;
        UsdAttribute const attr = _prim.GetAttribute(_name);
        if (attr && attr.Get(&value, UsdTimeCode::Default()) &&
            !value.IsEmpty()) {
            return value;
        }
        return VtValue();
    }

    bool GetContributingSampleTimesForInterval(
        Time, Time, std::vector<Time> *) override
    {
        return false;
    }

private:
    UsdPrim _prim;
    TfToken _name;
};

// Built at library load, not on first use (no guarded function statics).
TfToken const s_cageNames[] = {
    TfToken("ownerIds"), TfToken("ownerDensities"),
    TfToken("ownerSeeds"), TfToken("ownerCvCounts"),
    TfToken("ownerEdgeBias"), TfToken("ownerLengthProfileOffsets"),
    TfToken("ownerLengthProfile"), TfToken("normalizedT"),
    TfToken("triangles"), TfToken("triangleOwnerIndices"),
    TfToken("triangleRootCharts"), TfToken("ownerChartCentroids"),
    TfToken("ownerChartMeanRadii")};
TfToken const s_cageAttrs[] = {
    TfToken("usdGen:surfaceCage:ownerIds"),
    TfToken("usdGen:surfaceCage:ownerDensities"),
    TfToken("usdGen:surfaceCage:ownerSeeds"),
    TfToken("usdGen:surfaceCage:ownerCvCounts"),
    TfToken("usdGen:surfaceCage:ownerEdgeBias"),
    TfToken("usdGen:surfaceCage:ownerLengthProfileOffsets"),
    TfToken("usdGen:surfaceCage:ownerLengthProfile"),
    TfToken("usdGen:surfaceCage:normalizedT"),
    TfToken("usdGen:surfaceCage:triangles"),
    TfToken("usdGen:surfaceCage:triangleOwnerIndices"),
    TfToken("usdGen:surfaceCage:triangleRootCharts"),
    TfToken("usdGen:surfaceCage:ownerChartCentroids"),
    TfToken("usdGen:surfaceCage:ownerChartMeanRadii")};

}  // namespace

TF_REGISTRY_FUNCTION(TfType)
{
    TfType t = TfType::Define<
        UsdGenCurveAPIAdapter,
        TfType::Bases<UsdImagingAPISchemaAdapter>>();
    t.SetFactory<UsdImagingAPISchemaAdapterFactory<UsdGenCurveAPIAdapter>>();
}

HdContainerDataSourceHandle
usdGenImaging::UsdGenCurveRestContainerFactory(
    UsdPrim const &prim,
    UsdImagingDataSourceStageGlobals const &stageGlobals)
{
    TF_UNUSED(stageGlobals);
    if (!prim.IsA<UsdGeomBasisCurves>()) {
        return nullptr;
    }

    HdDataSourceBaseHandle const values[2] = {
        _CurveRestValueDataSource::New(prim, false),
        _CurveRestValueDataSource::New(prim, true)};
    TfToken const names[2] = {
        TfToken("points"), TfToken("hasAuthoredRest")};
    HdContainerDataSourceHandle leaves =
        HdRetainedContainerDataSource::New(2, names, values);
    std::vector<TfToken> presentNames;
    std::vector<HdDataSourceBaseHandle> presentValues;
    for (size_t i = 0; i != sizeof(s_cageAttrs) / sizeof(s_cageAttrs[0]); ++i) {
        if (!prim.GetAttribute(s_cageAttrs[i])) continue;
        presentNames.push_back(s_cageNames[i]);
        presentValues.push_back(
            _CurveSurfaceCageValueDataSource::New(prim, s_cageAttrs[i]));
    }
    if (!presentNames.empty()) {
        HdContainerDataSourceHandle const cage =
            HdRetainedContainerDataSource::New(
                presentNames.size(), presentNames.data(), presentValues.data());
        TfToken const cageRootName("surfaceCage");
        HdDataSourceBaseHandle const cageRoot = cage;
        HdContainerDataSourceHandle const merged =
            HdRetainedContainerDataSource::New(
                1, &cageRootName, &cageRoot);
        leaves = HdOverlayContainerDataSource::OverlayedContainerDataSources(
            merged, leaves);
    }
    TfToken const rootName("usdGenCurveRest");
    HdDataSourceBaseHandle const rootValue = leaves;
    return HdRetainedContainerDataSource::New(
        1, &rootName, &rootValue);
}

HdDataSourceLocatorSet
usdGenImaging::UsdGenCurveRestInvalidateMapping(
    UsdPrim const &prim,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType)
{
    TF_UNUSED(prim);
    TF_UNUSED(invalidationType);
    HdDataSourceLocatorSet result;
    HdDataSourceLocator const root(TfToken("usdGenCurveRest"));
    bool cageDirty = false;
    for (TfToken const &property : properties) {
        std::string const name = property.GetString();
        if (name == "points" || name == "primvars:rest" ||
            name == "rest" || name.rfind("primvars:rest:", 0) == 0) {
            result.insert(root);
        }
        if (name.rfind("usdGen:surfaceCage:", 0) == 0) {
            cageDirty = true;
        }
    }
    if (cageDirty) result.insert(root);
    return result;
}

HdContainerDataSourceHandle
UsdGenCurveAPIAdapter::GetImagingSubprimData(
    UsdPrim const &prim,
    TfToken const &subprim,
    TfToken const &appliedInstanceName,
    UsdImagingDataSourceStageGlobals const &stageGlobals)
{
    if (!subprim.IsEmpty() || !appliedInstanceName.IsEmpty()) {
        return nullptr;
    }
    return usdGenImaging::UsdGenCurveRestContainerFactory(
        prim, stageGlobals);
}

HdDataSourceLocatorSet
UsdGenCurveAPIAdapter::InvalidateImagingSubprim(
    UsdPrim const &prim,
    TfToken const &subprim,
    TfToken const &appliedInstanceName,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType)
{
    if (!subprim.IsEmpty() || !appliedInstanceName.IsEmpty()) {
        return HdDataSourceLocatorSet();
    }
    return usdGenImaging::UsdGenCurveRestInvalidateMapping(
        prim, properties, invalidationType);
}

PXR_NAMESPACE_CLOSE_SCOPE
