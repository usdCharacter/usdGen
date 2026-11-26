// UsdGenRestAPIAdapter: API schema adapter for UsdGenRestAPI (S12, ADR §5.1 #3;
// 06-imaging.md §2.6).
//
// M0 registered the type; M1 publishes the rest container (usdGen/rest/*)
// for every prim that actually applies UsdGenRestAPI — registered by
// apiSchemaName, so it runs only on applied prims (adapterRegistry).
#ifndef USDGEN_IMAGING_REST_API_SCHEMA_ADAPTER_H
#define USDGEN_IMAGING_REST_API_SCHEMA_ADAPTER_H

#include "usdGenImaging/api.h"

#include "pxr/usdImaging/usdImaging/apiSchemaAdapter.h"
#include "pxr/pxr.h"

PXR_NAMESPACE_OPEN_SCOPE

class UsdGenRestAPIAdapter final
    : public UsdImagingAPISchemaAdapter
{
public:
    UsdGenRestAPIAdapter() = default;
    ~UsdGenRestAPIAdapter() override = default;

    /// The API adds no child prims: its contribution overlays the prim's
    /// own data source at the usdGen/rest/* locators.
    TfTokenVector GetImagingSubprims(
        UsdPrim const & /*prim*/, TfToken const & /*appliedInstanceName*/) override
    {
        return TfTokenVector();
    }

    TfToken GetImagingSubprimType(
        UsdPrim const & /*prim*/, TfToken const & /*subprim*/,
        TfToken const & /*appliedInstanceName*/) override
    {
        return TfToken();
    }

    /// Primary-prim contribution (empty subprim): the usdGen/rest container
    /// (02 §2.15). nullptr for non-Mesh targets (hard diagnostic 2, once).
    HdContainerDataSourceHandle GetImagingSubprimData(
        UsdPrim const &prim,
        TfToken const &subprim,
        TfToken const &appliedInstanceName,
        const UsdImagingDataSourceStageGlobals &stageGlobals) override;

    /// points / faceVertexCounts / faceVertexIndices / st / normals /
    /// normalsInterpolation /
    /// primvars:rest / usdGen:rest:* -> their usdGen/rest/* leaves (02 §2.15).
    HdDataSourceLocatorSet InvalidateImagingSubprim(
        UsdPrim const &prim,
        TfToken const &subprim,
        TfToken const &appliedInstanceName,
        TfTokenVector const &properties,
        UsdImagingPropertyInvalidationType invalidationType) override;
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif // USDGEN_IMAGING_REST_API_SCHEMA_ADAPTER_H
