// usdGen imaging — UsdGenRestAPI data sourcing (02-schema.md §2.15, S12).
//
// The RestAPI container publishes the REST surface of a bound Mesh:
//   usdGen/rest/points              VtVec3fArray  (points at UsdTimeCode::Default())
//   usdGen/rest/faceVertexCounts    VtIntArray
//   usdGen/rest/faceVertexIndices   VtIntArray
//   usdGen/rest/st                  VtVec2fArray  (primary uv set)
//   usdGen/rest/normals             VtVec3fArray  (Mesh normals at Default)
//   usdGen/rest/normalsInterpolation TfToken       (the paired interpolation)
//
// Rest points are sampled at UsdTimeCode::Default() by a custom
// UsdImagingDataSourceMapped::AttributeMapping::factory whose data source
// calls Get<T>(&r, UsdTimeCode::Default()) — so the rest channel costs no
// per-frame dirty even when the scene time moves (MEASURED, research/
// G-stage-free-parameter-and-time-transport.md §3 route R2). Capture-on-first-
// cook is rejected (S12).
//
// M0 registered UsdGenRestAPIAdapter (registration only). M1 adds the factory
// and the invalidation mapping below; the imaging lane wires both into the
// adapter (docs/m1/interfaces.md §6; M0 restApiSchemaAdapter.{h,cpp} is
// extended, not replaced — see scaffold-issues.md).
#ifndef USDGEN_IMAGING_REST_API_DATA_SOURCE_H
#define USDGEN_IMAGING_REST_API_DATA_SOURCE_H

#include "usdGenImaging/api.h"

#include "pxr/pxr.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usdImaging/usdImaging/dataSourceMapped.h"
#include "pxr/usdImaging/usdImaging/sceneIndexPrimAdapter.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

/// AttributeMapping::factory for the usdGen/rest container: builds the six
/// rest leaves above on the given prim. Returns nullptr when the prim is not
/// a Mesh (the caller warns once per prim, 02 §2.15).
HdContainerDataSourceHandle UsdGenRestApiContainerFactory(
    UsdPrim const &prim,
    UsdImagingDataSourceStageGlobals const &globals);

/// Invalidation mapping required for the container to be dirtiable (02 §2.15):
///   points / faceVertexCounts / faceVertexIndices / normals /
///   normalsInterpolation -> usdGen/rest/<leaf>
///   primvars:rest                                  -> usdGen/rest/points
///   usdGen:rest:source|primvar|file                -> their own usdGen/rest/* leaves
///   usdGen:rest:file assetPath                     -> usdGen/rest/points (asset)
/// Returns the set of container locators to dirty for the given authoring
/// properties (empty set == no invalidation).
HdDataSourceLocatorSet UsdGenRestApiInvalidateMapping(
    UsdPrim const &prim,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType);

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_REST_API_DATA_SOURCE_H
