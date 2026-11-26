// usdGen schema library: registers the compute-extent function for
// UsdGenDescription so usdcat/usdview can show bounds for codeless
// descriptions without any authored geometry.
//
// Rationale (ADR R19): a resource-only plugin cannot register a compute-extent
// callback, so we ship a minimal shared library whose only job is the TfType
// registration plus the UsdGeomRegisterComputeExtentFunction call.
#include "pxr/pxr.h"

#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/tf/type.h"
#include "pxr/base/vt/array.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usdGeom/boundable.h"
#include "pxr/usd/usdGeom/boundableComputeExtent.h"

#include "usdGenSchema/usdGenSchema.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace usdGen {
namespace {

// M0 extent provider. It is a placeholder, not the M1+ curve-envelope kernel:
// its job here is to PROVE the registration fires (usdcat/usdview query a
// codeless UsdGenDescription and get a non-empty extent back, which is only
// possible if Plug dlopen'd this library and called into it).
//
// Strategy: aggregate the bounds of boundable descendants. A M0 fixture
// description has no generated geometry yet, so if that aggregation comes up
// empty we emit a small default box -- a stable, deterministic volume that
// still proves the callback ran. The real curve-envelope approximation lands
// with the M1 generator pipeline.
bool
_ComputeDescriptionExtent(
    const UsdGeomBoundable &boundable,
    const UsdTimeCode &time,
    const GfMatrix4d *transformMatrix,
    VtVec3fArray *outExtent)
{
    const UsdPrim prim = boundable.GetPrim();
    GfRange3d range;
    bool any = false;
    for (const UsdPrim &child : prim.GetAllChildren()) {
        // Only ask boundable children. ComputeExtent on a non-boundable prim
        // (the Ops Scope every description carries, say) reaches
        // _ComputeExtentFromPlugins, which posts a TfError rather than
        // returning false -- and a posted error surfaces in usdview as an
        // exception out of the bbox cache, not as a skipped child.
        const UsdGeomBoundable boundableChild(child);
        if (!boundableChild) continue;
        VtVec3fArray childExtent(2);
        if (boundableChild.ComputeExtent(time, &childExtent) &&
            childExtent.size() == 2) {
            range.Union(GfRange3d(childExtent[0], childExtent[1]));
            any = true;
        }
    }
    if (!any) {
        // No boundable descendants: a fixed unit-centred box, so the prim still
        // has a stable, deterministic bounding volume for selection/culling.
        range = GfRange3d(GfVec3d(-0.5, -0.5, -0.5), GfVec3d(0.5, 0.5, 0.5));
    }
    if (transformMatrix) {
        GfBBox3d bbox(GfRange3d(range.GetMin(), range.GetMax()),
                      *transformMatrix);
        range = bbox.ComputeAlignedRange();
    }
    *outExtent = VtVec3fArray({GfVec3f(range.GetMin()),
                               GfVec3f(range.GetMax())});
    return true;
}

}  // namespace
}  // namespace usdGen

// Extent provider slot (sol S-8): usdGenSchema.h declares these with TF_API
// and usdGenImaging is expected to install a live provider when it loads.
// M0 ships the definitions so the installed surface has no undefined symbols;
// the registered compute-extent callback above stays static until M1 wires
// the provider in.
namespace {
UsdGenSchemaExtentFn _usdGenSchemaExtentProvider = nullptr;
}

void UsdGenSchema_SetExtentProvider(UsdGenSchemaExtentFn fn) {
    _usdGenSchemaExtentProvider = fn;
}

UsdGenSchemaExtentFn UsdGenSchema_GetExtentProvider() {
    return _usdGenSchemaExtentProvider;
}

TF_REGISTRY_FUNCTION(UsdGeomBoundable)
{
    // UsdGenDescription derives from UsdGeomBoundable in the codeless schema;
    // register against the concrete type so bound queries resolve it. The type
    // exists because Plug declares it from the schema plugInfo; its ancestor
    // chain reaching UsdGeomBoundable is what lets ComputeExtentFromPlugins
    // dispatch here (usdRig uses the same key-by-TfType trick for its
    // codeless types).
    const TfType &type = TfType::FindByName("UsdGenDescription");
    if (!type.IsUnknown()) {
        UsdGeomRegisterComputeExtentFunction(
            type, &usdGen::_ComputeDescriptionExtent);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
