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

#include "usdGenMath/math.h"
#include "usdGen/api/cowArray.h"

#include <cstddef>
#include <cstdint>
#include <string_view>


namespace usdGen {

/// Number of LUT entries for every scalar ramp (02 §2.17; the 257th entry
/// keeps the interpolated j+1 read in range).
constexpr size_t kUsdGenRampLutSize = 257;

/// The R11 interpolation modes. Callers parse the token once per build.
enum class UsdGenRampInterp : uint8_t { Linear, Constant, CatmullRom, Bspline };

/// R11 token text to a mode; anything else is Linear (02 §2.17's
/// "unknown token -> no special mode").
inline UsdGenRampInterp UsdGenParseRampInterp(std::string_view text) noexcept
{
    if (text == "constant") return UsdGenRampInterp::Constant;
    if (text == "catmullRom") return UsdGenRampInterp::CatmullRom;
    if (text == "bspline") return UsdGenRampInterp::Bspline;
    return UsdGenRampInterp::Linear;
}

/// Build a 257-entry LUT: lut[i] = ramp(i/256) for i in [0, 256].
///
/// `knots` are (position in [0,1], value), sorted by position (the adapter
/// guarantees order). Empty knots -> the identity ramp (flat 1.0).
void UsdGenBuildRampLut(
    UsdGenVec2fArray const &knots,
    UsdGenRampInterp interpolation,
    float *lut,
    size_t lutSize = kUsdGenRampLutSize);

/// Evaluate one scalar ramp at parameter u in [0,1] (for tests and the
/// single-sample paths); delegates to the same spline code as the LUT.
float UsdGenEvalRamp(
    UsdGenVec2fArray const &knots,
    UsdGenRampInterp interpolation,
    float u);

}  // namespace usdGen

#endif  // USDGEN_MATH_RAMP_H
