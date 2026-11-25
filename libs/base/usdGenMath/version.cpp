// usdGenMath — version + a trivial SoA kernel to prove the TBB/nanoflann/seexpr
// link path and keep the -ffp-contract=off rule observable (gate B-1 / §8.1).
#include "usdGenMath/usdGenMath.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/value.h"

#include <cmath>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenMath {

std::string GetVersionString() {
    // Touch tf/vt so those links are exercised and B-1 can observe them.
    // Sol S-8: renamed from usdGen::GetVersionString; the canonical core
    // declaration lives in usdGen/usdGen.h, the math lib keeps its own name.
    const TfToken v("usdGenMath");
    const VtValue marker(v);
    (void)marker;
    return "usdGenMath 0.1.0 (M0)";
}

}  // namespace usdGenMath

namespace usdGen {

// M0 placeholder for the arc-length resample kernel (full body lands with
// UsdGenResample in M2). Deterministic by construction: single-threaded,
// no intrinsics, no runtime-dependent reduction order (E-8 readiness).
VtVec3fArray ResampleByArcLength(
    VtVec3fArray const &points,
    std::vector<int> const &curveIds,
    int samplesPerCurve)
{
    TF_UNUSED(points);
    TF_UNUSED(curveIds);
    TF_UNUSED(samplesPerCurve);
    return {};
}

}  // namespace usdGen
