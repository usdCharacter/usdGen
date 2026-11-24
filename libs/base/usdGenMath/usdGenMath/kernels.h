// usdGenMath — deterministic float kernels shared by the capture-time ops
// (review M-3: raw deterministic compute lives here, compiled with
// -ffp-contract=off; the ops keep stage/parameter-binding wrappers only).
//
// Every function here is out-of-line so the generated machine code is
// identical regardless of which translation unit (or thread count) calls it;
// float results are therefore bitwise-stable across 1-thread vs N-thread
// capture runs (plan/03-execution-engine.md determinism rule E-8).
#ifndef USDGEN_MATH_KERNELS_H
#define USDGEN_MATH_KERNELS_H

#include <cstddef>
#include <cstdint>

namespace usdGen {

/// Interpolated read of a 257-entry ramp LUT (04-operators.md §5.2: "everywhere
/// rampLUT[t] is shorthand for this interpolated read, not the nearest sample"):
///   u = clamp01(t) * 256;  j = min(255, floor(u));  a = u - j;
///   value = lut[j] * (1 - a) + lut[j + 1] * a
float UsdGenEvalLut257(const float *lut, float t);

/// Interleave the low 21 bits of x, y, z into a 63-bit Morton code
/// (spatial-ordering key for capture-time root sorting; integer-only, so the
/// key is exact under any compiler flag).
std::uint64_t UsdGenMortonInterleave(std::uint32_t x, std::uint32_t y, std::uint32_t z);

/// Morton key for a world-space position: quantize each axis with
/// q = clamp(floor(v * cellScale), -(1<<20), (1<<20)-1), bias to [0, 2^21) and
/// interleave. Positions outside the quantization window clamp to the window,
/// which only affects ordering, never membership.
std::uint64_t UsdGenMortonKey3(float x, float y, float z, float cellScale);

/// Area of one triangle (a, b, c): 0.5 * |cross(b - a, c - a)|.
float UsdGenTriangleArea(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz);

/// Rest area of a planar-ish n-gon given as SoA corner positions
/// px[p0 + i], py[p0 + i], pz[p0 + i] (i in [0, n)): the fan from corner 0,
///   area = 0.5 * sum_{i=1..n-2} |cross(pi - p0, p(i+1) - p0)|
/// (matches the fan-triangle decomposition used for sampling).
float UsdGenPolygonRestArea(
    const float *px, const float *py, const float *pz,
    std::size_t p0, std::size_t n);

/// Rest length of a contiguous polyline px[base + i], i in [0, cv):
///   sum_{i=0..cv-2} |p(i+1) - p(i)|
float UsdGenPolylineRestLength(
    const float *px, const float *py, const float *pz,
    std::size_t base, std::size_t cv);

}  // namespace usdGen

#endif  // USDGEN_MATH_KERNELS_H
