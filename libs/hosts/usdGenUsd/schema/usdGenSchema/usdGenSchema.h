// usdGenSchema — schema library (codeless domain + extent registration).
//
// M0 surface: schema TfTokens used by the imaging adapters and the
// UsdGenDescription compute-extent registration. The extent kernel body lives
// in usdGenMath (per plan §1.2); usdGenImaging installs a provider via
// UsdGenSchema_SetExtentProvider when it loads.
#ifndef USDGEN_SCHEMA_H
#define USDGEN_SCHEMA_H

#include "pxr/pxr.h"

#include "pxr/usd/usd/timeCode.h"

// TF_API would import these from libtf; they live in usdGenSchema, so the
// DSO needs its own export control (CMake defines usdGenSchema_EXPORTS).
#if defined(_WIN32)
#  if defined(usdGenSchema_EXPORTS)
#    define USDGEN_SCHEMA_API __declspec(dllexport)
#  else
#    define USDGEN_SCHEMA_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_SCHEMA_API __attribute__((visibility("default")))
#endif

PXR_NAMESPACE_OPEN_SCOPE

/// Signature for a live extent provider installed by usdGenImaging.
/// Returns true and fills outExtent (2 vec3) on success.
typedef bool (*UsdGenSchemaExtentFn)(
    const SdfPath &descriptionPath,
    const UsdTimeCode &time,
    GfVec3f *outExtent /*[2]*/);

USDGEN_SCHEMA_API void UsdGenSchema_SetExtentProvider(UsdGenSchemaExtentFn fn);
USDGEN_SCHEMA_API UsdGenSchemaExtentFn UsdGenSchema_GetExtentProvider();

PXR_NAMESPACE_CLOSE_SCOPE

#endif // USDGEN_SCHEMA_H
