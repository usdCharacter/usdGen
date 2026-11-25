// usdGenMath — scalar-ramp LUT builder implementation (02-schema.md §2.17, R11).
//
// 257-entry LUTs: the 257th entry keeps the kernel's j+1 lerp read in range and
// makes both ramp endpoints exact samples. All four R11 tokens: constant holds
// the previous knot value; linear is piecewise-linear; catmullRom is the
// uniform Catmull-Rom spline through the knots with clamped (repeated)
// endpoints; bspline is the uniform cubic B-spline approximating the knots.
#include "usdGenMath/usdGenMath/ramp.h"

#include <algorithm>
#include <cmath>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

float ClampU(float u) { return std::min(std::max(u, 0.0f), 1.0f); }

/// Find the segment index: the last knot with x <= u, clamped to size-2.
/// Knots are x-sorted (adapter guarantee); u is already clamped to [0,1]
/// and knot positions are authored in [0,1] (02 §2.17).
size_t SegmentAt(VtVec2fArray const &knots, float u)
{
    size_t lo = 0, hi = knots.size() - 1;
    while (lo < hi) {
        size_t const mid = (lo + hi + 1) / 2;
        if (float(knots[mid][0]) <= u) lo = mid;
        else hi = mid - 1;
    }
    return std::min(lo, knots.size() - 2);
}

float KnotY(VtVec2fArray const &knots, int i)
{
    // Clamp the index into [0, size-1] (endpoint clamping for CR / B-spline).
    int const n = int(knots.size());
    i = std::max(0, std::min(i, n - 1));
    return knots[size_t(i)][1];
}

float EvalOne(VtVec2fArray const &knots, TfToken const &interpolation, float u)
{
    u = ClampU(u);
    // Interned once: EvalOne runs per LUT sample (257x per chunk per node on
    // workers); each TfToken("literal") re-hashes + re-probes the global
    // token registry under a lock — the E-7 negative-scaling term.
    static const TfToken sConstant{"constant"};
    static const TfToken sLinear{"linear"};
    static const TfToken sCatmullRom{"catmullRom"};
    static const TfToken sBspline{"bspline"};
    if (knots.size() < 2) return 1.0f;
    // Endpoints: u before the first knot / after the last resolves to that
    // knot's value (clamped-end behaviour, 02 §2.17).
    if (u <= float(knots.front()[0])) return float(knots.front()[1]);
    if (u >= float(knots.back()[0]))  return float(knots.back()[1]);

    size_t const i = SegmentAt(knots, u);
    float const xa = float(knots[i][0]),     xb = float(knots[i + 1][0]);
    float const ya = float(knots[i][1]),     yb = float(knots[i + 1][1]);
    float const t = (xb - xa > 1e-9f) ? (u - xa) / (xb - xa) : 0.0f;
    const float tc = std::min(std::max(t, 0.0f), 1.0f);

    if (interpolation == sConstant) {
        return ya;  // hold the previous knot's value
    }
    if (interpolation == sLinear) {
        return ya + (yb - ya) * tc;
    }
    if (interpolation == sCatmullRom) {
        // Uniform Catmull-Rom through the knots, clamped endpoints.
        float const p0 = KnotY(knots, int(i) - 1);
        float const p1 = ya;
        float const p2 = yb;
        float const p3 = KnotY(knots, int(i) + 2);
        float const t2 = tc * tc, t3 = t2 * tc;
        return 0.5f * ((2.0f * p1) +
                       (-p0 + p2) * tc +
                       (p0 - 2.0f * p1 + p2) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    }
    if (interpolation == sBspline) {
        // Uniform cubic B-spline approximating the knots (basis on [i, i+3]).
        float const p0 = KnotY(knots, int(i));
        float const p1 = KnotY(knots, int(i) + 1);
        float const p2 = KnotY(knots, int(i) + 2);
        float const p3 = KnotY(knots, int(i) + 3);
        float const t2 = tc * tc, t3 = t2 * tc;
        return (1.0f / 6.0f) *
               ((p0 * (3.0f - 6.0f * t2 + 3.0f * t3)) +
                (p1 * (4.0f - 6.0f * tc + 3.0f * t2)) +
                (p2 * (1.0f + 3.0f * tc + 3.0f * t2 - 3.0f * t3)) +
                (p3 * t3));
    }
    // Unknown token: linear fallback.
    return ya + (yb - ya) * tc;
}

}  // namespace

void UsdGenBuildRampLut(
    VtVec2fArray const &knots,
    TfToken const &interpolation,
    float *lut,
    size_t lutSize)
{
    if (!lut) return;
    const size_t n = std::max<size_t>(1, lutSize);
    const int denom = int(n) - 1;  // lut[i] = ramp(i / (n-1)); 257 -> /256
    for (size_t i = 0; i < n; ++i) {
        lut[i] = (denom > 0) ? EvalOne(knots, interpolation, float(i) / float(denom))
                             : (knots.empty() ? 1.0f : float(knots.front()[1]));
    }
}

float UsdGenEvalRamp(
    VtVec2fArray const &knots,
    TfToken const &interpolation,
    float u)
{
    return EvalOne(knots, interpolation, u);
}

}  // namespace usdGen
