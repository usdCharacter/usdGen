// UsdGenCurveAPIAdapter: live C3 rest provenance for BasisCurves.
//
// Unlike the surface RestAPI adapter, this API is applied to curve prims.
// Its usdGenCurveRest root contains data sources that retain the UsdPrim and
// read Default-time values when queried, so a source handle cannot retain a
// first-frame snapshot across a stage edit.
#ifndef USDGEN_IMAGING_CURVE_API_SCHEMA_ADAPTER_H
#define USDGEN_IMAGING_CURVE_API_SCHEMA_ADAPTER_H

#include "usdGenImaging/api.h"

#include "pxr/pxr.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usdImaging/usdImaging/apiSchemaAdapter.h"
#include "pxr/usdImaging/usdImaging/dataSourceStageGlobals.h"

PXR_NAMESPACE_OPEN_SCOPE

class UsdGenCurveAPIAdapter final : public UsdImagingAPISchemaAdapter
{
public:
    UsdGenCurveAPIAdapter() = default;
    ~UsdGenCurveAPIAdapter() override = default;

    TfTokenVector GetImagingSubprims(
        UsdPrim const &, TfToken const &) override
    { return TfTokenVector(); }

    TfToken GetImagingSubprimType(
        UsdPrim const &, TfToken const &, TfToken const &) override
    { return TfToken(); }

    HdContainerDataSourceHandle GetImagingSubprimData(
        UsdPrim const &prim,
        TfToken const &subprim,
        TfToken const &appliedInstanceName,
        UsdImagingDataSourceStageGlobals const &stageGlobals) override;

    HdDataSourceLocatorSet InvalidateImagingSubprim(
        UsdPrim const &prim,
        TfToken const &subprim,
        TfToken const &appliedInstanceName,
        TfTokenVector const &properties,
        UsdImagingPropertyInvalidationType invalidationType) override;
};

namespace usdGenImaging {

/// Root published by the adapter on an applied BasisCurves prim:
///   usdGenCurveRest/points          VtVec3fArray at UsdTimeCode::Default()
///   usdGenCurveRest/hasAuthoredRest bool provenance, queried live
///   usdGenCurveRest/surfaceCage/*   authored sparse-cage arrays, when present
HdContainerDataSourceHandle UsdGenCurveRestContainerFactory(
    UsdPrim const &prim,
    UsdImagingDataSourceStageGlobals const &stageGlobals);

/// Both the Default-time points fallback and the authored rest primvar feed
/// the same live root.  Dirtying the root also covers provenance changes when
/// a primvar opinion is added or removed.
HdDataSourceLocatorSet UsdGenCurveRestInvalidateMapping(
    UsdPrim const &prim,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType);

}  // namespace usdGenImaging

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_IMAGING_CURVE_API_SCHEMA_ADAPTER_H
