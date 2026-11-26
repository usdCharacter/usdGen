// UsdGenRestAPIAdapter: API schema adapter for UsdGenRestAPI (S12, ADR §5.1 #3).
#include "usdGenImaging/restApiSchemaAdapter.h"
#include "usdGenImaging/usdGenRestApiDataSource.h"

#include "pxr/base/tf/type.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType)
{
    TfType t = TfType::Define<
        UsdGenRestAPIAdapter,
        TfType::Bases<UsdImagingAPISchemaAdapter>>();
    t.SetFactory<UsdImagingAPISchemaAdapterFactory<UsdGenRestAPIAdapter>>();
}

HdContainerDataSourceHandle
UsdGenRestAPIAdapter::GetImagingSubprimData(
    UsdPrim const &prim,
    TfToken const &subprim,
    TfToken const &appliedInstanceName,
    UsdImagingDataSourceStageGlobals const &stageGlobals)
{
    if (!subprim.IsEmpty() || !appliedInstanceName.IsEmpty()) {
        return nullptr;
    }
    return usdGenImaging::UsdGenRestApiContainerFactory(prim, stageGlobals);
}

HdDataSourceLocatorSet
UsdGenRestAPIAdapter::InvalidateImagingSubprim(
    UsdPrim const &prim,
    TfToken const &subprim,
    TfToken const &appliedInstanceName,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType)
{
    if (!subprim.IsEmpty() || !appliedInstanceName.IsEmpty()) {
        return HdDataSourceLocatorSet();
    }
    return usdGenImaging::UsdGenRestApiInvalidateMapping(
        prim, properties, invalidationType);
}

PXR_NAMESPACE_CLOSE_SCOPE
