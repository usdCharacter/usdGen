#ifndef USDGEN_GPU_ROOT_FRAME_GATHER_H
#define USDGEN_GPU_ROOT_FRAME_GATHER_H

#include "rootFrames.h"

#include <cstdint>
#include <string>

namespace usdGen::gpu {

enum class RootFrameGatherStatus {
    Ok, InvalidArgument, MissingStableId, CudaError, NoPendingUpdate
};

// Device-only stable-ID gather for topology survivors. Source IDs describe
// original capture order and survivor IDs describe a later Length/compaction
// order; no surface/root-frame recomputation occurs. Inputs are borrowed until
// Finish. recordUse has one last-use event per buffer, so consumers are
// scheduler-serialized rather than independently concurrent.
class CudaRootFrameGather {
public:
    CudaRootFrameGather() = default;
    ~CudaRootFrameGather();
    CudaRootFrameGather(CudaRootFrameGather const&) = delete;
    CudaRootFrameGather& operator=(CudaRootFrameGather const&) = delete;

    RootFrameGatherStatus Apply(DeviceView<const uint64_t> sourceIds,
                                RestRootFrames sourceFrames,
                                DeviceView<const float3> sourceOrigins,
                                DeviceView<const uint8_t> sourceValid,
                                DeviceView<const uint8_t> sourceDrop,
                                DeviceView<const uint64_t> survivorIds,
                                cudaStream_t stream);
    RootFrameGatherStatus Finish(cudaStream_t stream);

    RestRootFrames frames() const noexcept;
    DeviceView<const float3> origins() const noexcept;
    DeviceView<const uint8_t> valid() const noexcept;
    DeviceView<const uint8_t> dropMask() const noexcept;
    size_t count() const noexcept { return active_.count; }
    uint64_t generation() const noexcept { return generation_; }
    int deviceIndex() const noexcept { return deviceIndex_; }
    cudaError_t recordUse(cudaStream_t stream);
    cudaError_t waitOn(cudaStream_t stream) const;
    char const* diagnostic() const noexcept { return diagnostic_.c_str(); }

private:
    struct Storage {
        DeviceBuffer<float3> origins, tangent, binormal, normal;
        DeviceBuffer<uint8_t> valid, drop;
        size_t count = 0;
        void clear() noexcept;
        void quarantine() noexcept;
        void swap(Storage&) noexcept;
        cudaError_t recordUse(cudaStream_t);
        cudaError_t waitOn(cudaStream_t) const;
    };
    RootFrameGatherStatus fail(RootFrameGatherStatus, char const*);
    cudaError_t validateStream(cudaStream_t) const;
    void discardPending() noexcept;
    void quarantinePending() noexcept;

    Storage active_, pendingStorage_;
    DeviceBuffer<int> error_;
    DeviceBuffer<uint32_t> pendingCount_;
    cudaEvent_t ready_ = nullptr;
    uint64_t generation_ = 0;
    int deviceIndex_ = -1;
    bool pending_ = false;
    std::string diagnostic_;
};

} // namespace usdGen::gpu

#endif
