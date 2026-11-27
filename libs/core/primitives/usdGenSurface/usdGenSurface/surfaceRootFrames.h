// Engine-neutral construction of C3 rest root frames from already-authored
// skinprim/skinprimuv bindings.  This deliberately does not choose bindings
// or project roots: closest-point rebinding belongs to a later shared helper.
#ifndef USDGEN_SURFACE_ROOT_FRAMES_H
#define USDGEN_SURFACE_ROOT_FRAMES_H

#include "usdGen/api/graphSpec.h"
#include "usdGen/api/cowArray.h"
#include "usdGenMath/math.h"
#include "usdGenSurface/surfaceInput.h"

#include <cstdint>
#include <string>
#include "usdGenSurface/export.h"


namespace usdGen {

struct UsdGenSurfaceRootFrameResult
{
    // One row-vector Gf matrix per input binding: rows 0/1/2 are T/B/N and
    // row 3 is the surface-local origin. Invalid/dropped entries are a zero
    // matrix, never a plausible fallback frame.
    UsdGenMatrix4dArray frames;
    UsdGenCowArray<uint8_t> valid;
    UsdGenCowArray<uint8_t> dropMask;
    uint32_t dropped = 0;
};

/// Validates and constructs rest frames for parent-mesh face ids.  `surface`
/// must carry authored rest points (not a current-frame fallback); subsetFaces
/// is intentionally ignored because bindings are parent-face ids. Structural
/// and non-finite input errors return false and leave `out` unchanged.
/// Geometric/parameterisation degeneracy succeeds with valid=0/dropMask=1;
/// callers compact every C3 channel using the corresponding source index.
USDGEN_SURFACE_API bool UsdGenBuildRestSurfaceRootFrames(
    UsdGenSurfaceInputView const &surface,
    UsdGenIntArray const &rootPrim,
    UsdGenVec2fArray const &rootUV,
    UsdGenSurfaceRootFrameResult *out,
    std::string *error = nullptr);

/// Root-frame authored primvars are used verbatim, but must meet the same
/// affine, orthonormal, right-handed contract as gpu/rootFrames.cu.
USDGEN_SURFACE_API bool UsdGenValidateAuthoredRootFrame(UsdGenMatrix4d const &frame);

} // namespace usdGen

#endif // USDGEN_SURFACE_ROOT_FRAMES_H
