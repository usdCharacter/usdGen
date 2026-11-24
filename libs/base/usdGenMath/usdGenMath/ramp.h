// usdGenMath — scalar-ramp 257-entry LUT builder (02-schema.md §2.17, S11,
// ADR §9 R11).
//
// Every scalar ramp (mask ramp, clump profile, width knots, noise magnitude,
// scale knots) is baked at capture into a 257-entry float LUT over the
// root->tip parameter u in [0,1] so the hot kernel loop is a lerp read, not a
// spline evaluation (03 §8.3 rule 2; usdRig's falloff-LUT precedent).
//
// The four R11 interpolation tokens, exactly:
//   constant     — holds the previous knot's value (step function, clamped ends)
//   linear       — piecewise-linear through the knots, clamped at both ends
//   catmullRom   — uniform Catmull-Rom spline through the knots, clamped ends
//   bspline      — uniform cubic B-spline approximating the knots
//
#ifndef USDGEN_MATH_RAMP_H
#define USDGEN_MATH_RAMP_H

#include "usdGenMath/usdGenMath.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"

#include <cstddef>
#include <cstdint>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Number of LUT entries for every scalar ramp (02 §2.17; the 257th entry
/// keeps the interpolated j+1 read in range).
constexpr size_t kUsdGenRampLutSize = 257;

/// Build a 257-entry LUT: lut[i] = ramp(i/256) for i in [0, 256].
///
/// `knots` are (position in [0,1], value), sorted by position (the adapter
/// guarantees order). Empty knots -> the identity ramp (flat 1.0).
/// `interpolation` is one of the R11 tokens (anything else falls back to
/// linear, matching 02 §2.17's "unknown token -> no special mode").
void UsdGenBuildRampLut(
    VtVec2fArray const &knots,
    TfToken const &interpolation,
    float *lut,
    size_t lutSize = kUsdGenRampLutSize);

/// Evaluate one scalar ramp at parameter u in [0,1] (for tests and the
/// single-sample paths); delegates to the same spline code as the LUT.
float UsdGenEvalRamp(
    VtVec2fArray const &knots,
    TfToken const &interpolation,
    float u);

}  // namespace usdGen

#endif  // USDGEN_MATH_RAMP_H
