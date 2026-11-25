#ifndef USDGEN_GPU_POINT_OVERRIDE_H
#define USDGEN_GPU_POINT_OVERRIDE_H

#include "deviceBuffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace usdGen::gpu {

enum class PointOverrideStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    DuplicateIndex,
    CudaError,
    NoPendingUpdate
};

// Applies a sparse, indexed absolute point edit to a borrowed device base.
// The base is never modified.  A successful Finish publishes a private full
// point buffer; callers may then transfer that completed buffer to a
// generation owner with releaseCompleted().
class CudaPointOverride {
public:
    CudaPointOverride() = default;
    ~CudaPointOverride();
    CudaPointOverride(CudaPointOverride const&) = delete;
    CudaPointOverride& operator=(CudaPointOverride const&) = delete;

    // Input views are borrowed until Finish returns.  indices and positions
    // must have equal cardinality; an empty pair means an exact base copy.
    PointOverrideStatus Apply(DeviceView<const float3> basePoints,
                              DeviceView<const int32_t> indices,
                              DeviceView<const float3> absolutePositions,
                              cudaStream_t stream);
    PointOverrideStatus Finish(cudaStream_t stream);

    DeviceView<const float3> view() const noexcept;
    // Transfers the completed active allocation.  It returns empty while an
    // update is pending or when no completed allocation exists.  The moved
    // DeviceBuffer retains its consumer-use event and ownership.
    std::unique_ptr<DeviceBuffer<float3>> releaseCompleted();

    cudaError_t recordUse(cudaStream_t stream);
    cudaError_t waitOn(cudaStream_t stream) const;
    int deviceIndex() const noexcept { return deviceIndex_; }
    uint64_t generation() const noexcept { return generation_; }
    bool pending() const noexcept { return pending_; }

private:
    cudaError_t validateStream(cudaStream_t stream) const;

    DeviceBuffer<float3> active_;
    DeviceBuffer<float3> pendingStorage_;
    DeviceBuffer<int> error_;
    DeviceBuffer<uint32_t> inputSlots_;
    DeviceBuffer<int32_t> sortedIndices_;
    DeviceBuffer<uint32_t> sortedSlots_;
    DeviceBuffer<unsigned char> sortTemp_;
    cudaEvent_t ready_ = nullptr;
    bool pending_ = false;
    bool completed_ = false;
    int deviceIndex_ = -1;
    uint64_t generation_ = 0;
};

} // namespace usdGen::gpu

#endif // USDGEN_GPU_POINT_OVERRIDE_H
