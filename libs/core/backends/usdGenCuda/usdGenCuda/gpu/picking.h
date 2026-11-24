#ifndef USDGEN_GPU_PICKING_H
#define USDGEN_GPU_PICKING_H

#include "curveGeometry.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace usdGen::gpu {

enum class PickingStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidValue,
    CudaError,
    NoPendingOperation
};

// The matrix is row-major and uses USD's row-vector convention:
// clip = (x,y,z,1) * viewProj.  Pixel coordinates have a top-left origin;
// x increases rightward and y increases downward.  The query is copied by
// value, while geometry views are borrowed until Finish returns.
struct PickQuery {
    float viewProj[16]{};
    int width = 0;
    int height = 0;
    float x = 0.0f;
    float y = 0.0f;
    float radiusPx = 0.0f;
};

// This is the only host-visible result of a pick.  UINT32_MAX denotes an
// unset curve/CV/flat index.  A failed operation leaves the previous result.
struct PickResult {
    bool hit = false;
    uint32_t curve = UINT32_MAX;
    uint32_t cv = UINT32_MAX;
    uint32_t flatIndex = UINT32_MAX;
    uint64_t stableId = 0;
    float distancePx = 0.0f;
};

// Device-only screen-space CV picker and footprint builder.  At most one
// operation may be pending. Finish performs scalar readback only: a pick's
// result, or a footprint count and diagnostic. Geometry is never copied to
// the host. Successful publication swaps staged results; failures preserve
// the last successful result/footprint.
class CudaPicking {
public:
    CudaPicking() = default;
    ~CudaPicking();
    CudaPicking(CudaPicking const&) = delete;
    CudaPicking& operator=(CudaPicking const&) = delete;

    PickingStatus ApplyPick(DeviceCurveGeometryView geometry,
                            PickQuery query, cudaStream_t stream);
    PickingStatus ApplyFootprint(DeviceCurveGeometryView geometry,
                                 PickQuery query, cudaStream_t stream);
    PickingStatus Finish(cudaStream_t stream);

    bool pending() const noexcept { return pending_; }
    PickResult result() const noexcept { return result_; }
    DeviceView<const int32_t> footprint() const noexcept {
        return {footprint_.data(), footprintCount_};
    }
    size_t footprintCount() const noexcept { return footprintCount_; }
    cudaError_t recordUse(cudaStream_t stream) { return footprint_.recordUse(stream); }
    cudaError_t waitOn(cudaStream_t stream) const { return footprint_.waitOn(stream); }
    int deviceIndex() const noexcept { return deviceIndex_; }
    const char* diagnostic() const noexcept { return diagnostic_.c_str(); }

private:
    PickingStatus fail(PickingStatus status, const char* message);
    PickingStatus validateGeometry(DeviceCurveGeometryView geometry) const;
    PickingStatus validateQuery(PickQuery const& query) const;
    PickingStatus validateStream(cudaStream_t stream, int current) const;
    PickingStatus begin(DeviceCurveGeometryView geometry, PickQuery query,
                        bool footprint, cudaStream_t stream);
    PickingStatus abortBegin(cudaStream_t stream, PickingStatus status,
                             const char* message);

    DeviceBuffer<int> error_;
    DeviceBuffer<uint32_t> flags_;
    DeviceBuffer<uint32_t> prefix_;
    DeviceBuffer<int32_t> pendingFootprint_;
    DeviceBuffer<unsigned char> scanTemp_;
    DeviceBuffer<unsigned char> pendingPickBytes_;
    DeviceBuffer<float> pickDistances_;
    DeviceBuffer<float> pickMinimum_;
    DeviceBuffer<unsigned char> pickIndexBytes_;
    DeviceBuffer<uint32_t> pendingCount_;
    DeviceBuffer<int32_t> footprint_;
    cudaEvent_t ready_ = nullptr;
    PickResult result_{};
    size_t footprintCount_ = 0;
    size_t pendingPoints_ = 0;
    bool pending_ = false;
    bool pendingIsFootprint_ = false;
    int deviceIndex_ = -1;
    std::string diagnostic_;
};

} // namespace usdGen::gpu

#endif // USDGEN_GPU_PICKING_H
