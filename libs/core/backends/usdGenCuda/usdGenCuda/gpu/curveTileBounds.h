// Device-only conservative bounds for compacted curve tiles.
#ifndef USDGEN_GPU_CURVE_TILE_BOUNDS_H
#define USDGEN_GPU_CURVE_TILE_BOUNDS_H

#include "curveTiles.h"

#include <cstdint>

namespace usdGen::gpu {

// This intentionally names only the curve paths admitted by CUDA execution.
// Centripetal Catmull-Rom needs a separate, proven bound and is not accepted.
enum class CurveTileBoundsBasis : uint8_t {
    Linear,
    BSpline,
    CatmullRom
};

struct CurveTileBoundsOptions {
    CurveTileBoundsBasis basis = CurveTileBoundsBasis::BSpline;
};

struct CurveTileBoundsInput {
    DeviceView<const float3> points;
    DeviceView<const float> widths;
    DeviceView<const CurveTileSpan> spans;
};

// Caller-provided intermediate storage. It lets finalization validate every
// tile before the publish pass touches either output range.
struct CurveTileBoundsScratch {
    float3 minimum;
    float3 maximum;
    float maximumWidth = 0.0f;
};

struct CurveTileBoundsWorkspace {
    DeviceView<CurveTileBoundsScratch> scratch;
};

struct CurveTileBoundsOutput {
    // Entry i corresponds exactly to input.spans[i], never spans[i].tile.
    DeviceView<float3> minimums;
    DeviceView<float3> maximums;
    // One GPU scalar: 0 valid; 1 invalid span/value or unrepresentable bound.
    DeviceView<uint32_t> status;
};

// Enqueue a conservative per-tile AABB reduction. The caller owns every view
// until stream completion. This function allocates nothing, reads nothing back,
// takes no lock, and has no implicit stream. Linear and uniform B-spline use
// the CV hull plus maxWidth/2. Uniform Catmull-Rom uses the CV hull expanded
// per axis by (max-min)/8 plus 9*maxWidth/16, accounting for signed basis
// weights and the renderer's width/2 radial offset. Valid empty tiles produce
// a zero box. Any device status failure leaves both output ranges untouched:
// scratch is reduced and finalized before one aggregate-status-gated publish.
// CurveTileSpan's curve fields are not consumed here; device validation covers
// only its point range because this reduction has no curve-offset input.
cudaError_t BuildCurveTileBounds(CurveTileBoundsOptions options,
    CurveTileBoundsInput input, CurveTileBoundsWorkspace workspace,
    CurveTileBoundsOutput output,
    cudaStream_t stream);

} // namespace usdGen::gpu

#endif // USDGEN_GPU_CURVE_TILE_BOUNDS_H
