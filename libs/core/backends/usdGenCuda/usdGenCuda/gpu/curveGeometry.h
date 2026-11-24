#ifndef USDGEN_GPU_CURVE_GEOMETRY_H
#define USDGEN_GPU_CURVE_GEOMETRY_H
#include "deviceBuffer.h"
#include <cstdint>
namespace usdGen { namespace gpu {
struct DeviceCurveGeometryView {
    DeviceView<const float3> points, restPoints;
    DeviceView<const float> widths;
    DeviceView<const uint32_t> curveOffsets; // curves+1, zero-based exclusive end
    DeviceView<const uint64_t> stableIds;
    size_t curveCount = 0, pointCount = 0;
};
// Persistent, GPU-only curve channels.  Inputs are copied device-to-device;
// callers retain borrowed input views until Set completes on its stream.
class CudaCurveGeometry {
public:
    cudaError_t Set(DeviceCurveGeometryView source, cudaStream_t stream);
    DeviceCurveGeometryView view() const;
    cudaError_t recordUse(cudaStream_t stream);
    cudaError_t waitOn(cudaStream_t stream) const;
    uint64_t generation() const { return generation_; }
private:
    DeviceBuffer<float3> points_, rest_; DeviceBuffer<float> widths_;
    DeviceBuffer<uint32_t> offsets_; DeviceBuffer<uint64_t> ids_;
    size_t curves_=0, pointsCount_=0; uint64_t generation_=0;
};
}}
#endif
