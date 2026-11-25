// Portable C3 root binding/frame capture.  This is a pure value helper: it
// consumes an immutable graph snapshot and returns fresh arrays for a caller's
// candidate generation.  It owns no USD stage, scene-index, or device state.
#ifndef USDGEN_CURVE_ROOT_CAPTURE_H
#define USDGEN_CURVE_ROOT_CAPTURE_H

#include "usdGen/graphDesc.h"

#include <cstdint>
#include <memory>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenRestSurfaceBindingCache;

struct UsdGenCurveRootCaptureResult
{
    // All arrays are in the input curve order. Computed frames are expressed
    // in source-local coordinates, which is the coordinate space consumed by
    // point operators. For explicit surface-free `rebind=never` sources, the
    // optional rootPrim/rootUV/frames arrays remain empty and valid is true
    // for each geometry curve. Otherwise an invalid entry is explicit:
    // rootPrim is -1, rootUV is (0,0), and its frame is zeroed.
    VtIntArray rootPrim;
    VtVec2fArray rootUV;
    VtMatrix4dArray frames;
    VtArray<uint8_t> valid;
    VtArray<uint8_t> dropMask;
    uint32_t dropped = 0;
    uint32_t rebound = 0;
};

/// Captures C3 parent-face bindings and rest root frames without mutating the
/// source descriptor.  Authored frames are validated and copied verbatim;
/// absent frames are constructed from resolved bindings.  `rebind=never`
/// drops invalid/unresolved roots, `onError` repairs only invalid/missing
/// bindings, and `always` recomputes every binding.  Automatic binding uses
/// UsdGenBuildRestSurfaceRootBindings and frame construction uses
/// UsdGenBuildRestSurfaceRootFrames, both on source-order subsets.
///
/// Structural malformed input (counts/cardinalities, non-finite values,
/// malformed authored frames, invalid policy, or a missing surface required
/// for binding/frame computation) returns false and leaves `out` untouched.
/// Per-root unresolved binding and geometric degeneracy are successful,
/// explicit drop entries; no unresolved root is assigned face zero.
bool UsdGenCaptureCurveRoots(
    UsdGenCurveSetDesc const &source,
    UsdGenSurfaceDesc const *surface,
    TfToken rebind,
    UsdGenCurveRootCaptureResult *out,
    std::string *error = nullptr,
    std::shared_ptr<const UsdGenRestSurfaceBindingCache> *bindingCache = nullptr);

} // namespace usdGen

#endif // USDGEN_CURVE_ROOT_CAPTURE_H
