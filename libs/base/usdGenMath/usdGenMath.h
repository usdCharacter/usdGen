// usdGenMath - pure-math kernels (no USD, no Hydra, no I/O). Gate B-1.
#ifndef USDGEN_MATH_H
#define USDGEN_MATH_H

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"

#include <cstdint>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

// A single curve, planar SoA-friendly layout. Points are packed contiguously
// and the per-curve [start, start+count) slice is addressed via CvOffsets.
struct CurveBuffer {
    VtVec3fArray points;   // packed point stream
    std::vector<int> curveIds;     // parallel to points: curve index per point
    std::vector<int> cvOffsets;    // size = numCurves+1, cvOffsets[i..i+1] is curve i
    int numCurves = 0;

    int totalPoints() const { return static_cast<int>(points.size()); }
};

// Minimal arc-length resample: returns new point offsets for `curves` such that
// each output curve has `samplesPerCurve` points spaced by equal arc length.
// M0: linear resample only; spline resample arrives with UsdGenResample (M2).
VtVec3fArray ResampleByArcLength(
    VtVec3fArray const &points,
    std::vector<int> const &curveIds,
    int samplesPerCurve);

}  // namespace usdGen

namespace usdGenMath {

/// Monotonic version string for the math library (stage-free; gate B-1).
/// Renamed from usdGen::GetVersionString (sol S-8): the canonical core
/// declaration lives in usdGen/usdGen.h; the math lib keeps its own name.
std::string GetVersionString();

}  // namespace usdGenMath

#endif  // USDGEN_MATH_H
