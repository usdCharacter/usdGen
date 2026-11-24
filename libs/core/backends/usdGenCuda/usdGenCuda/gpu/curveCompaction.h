#ifndef USDGEN_GPU_CURVE_COMPACTION_H
#define USDGEN_GPU_CURVE_COMPACTION_H

#include "curveGeometry.h"
#include "rootFrames.h"

#include <cstddef>
#include <cstdint>

namespace usdGen::gpu {

// Queries CUB's selected-device scan workspace for a compaction of `curves`.
// This is a host-only sizing call: it allocates no temporary storage and
// enqueues no CUDA work.
cudaError_t GetCudaCurveCompactionScanTemporaryBytes(
    size_t curves, size_t* bytes, cudaStream_t stream = nullptr);

enum class CurveCompactionStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    CudaError,
    NoPendingUpdate
};

// Device-only stable compaction of ragged C3 strands.  Every supplied input
// channel is copied in curve order for keep[c] != 0; optional root channels
// must be supplied together.  Legacy input views are borrowed until Finish
// returns; fresh input views remain borrowed through the final native proof
// and CommitFreshFinish.
class CudaCurveCompaction {
public:
    CudaCurveCompaction() = default;
    ~CudaCurveCompaction();
    CudaCurveCompaction(CudaCurveCompaction const&) = delete;
    CudaCurveCompaction& operator=(CudaCurveCompaction const&) = delete;

    CurveCompactionStatus Apply(
        DeviceCurveGeometryView geometry, DeviceView<const float> hairT,
        DeviceView<const int32_t> rootPrim, DeviceView<const float2> rootUV,
        DeviceView<const uint8_t> keep, cudaStream_t stream,
        UsdGenExecutionMemoryReservation* reservation = nullptr,
        RestRootFrames frames = {});
    CurveCompactionStatus Finish(cudaStream_t stream);

    // Fresh two-phase continuation.  Counts validates/scans and copies only
    // {status, exact curve count, exact point count} to owned pinned storage;
    // the parent owns the native callbacks and calls each Commit only after
    // callback success and launcher return.  Commit methods are host-only.
    CurveCompactionStatus ApplyFreshCounts(
        DeviceCurveGeometryView, DeviceView<const float>, DeviceView<const int32_t>,
        DeviceView<const float2>, DeviceView<const uint8_t>, cudaStream_t,
        UsdGenExecutionMemoryReservation* reservation = nullptr,
        RestRootFrames frames = {});
    CurveCompactionStatus FinishFreshCountsAsync(cudaStream_t,
        void (*)(cudaStream_t, cudaError_t, void*) noexcept, void*);
    CurveCompactionStatus CommitFreshCounts();
    CurveCompactionStatus ApplyFreshScatter(
        cudaStream_t, UsdGenExecutionMemoryReservation* reservation = nullptr);
    CurveCompactionStatus FinishFreshScatterAsync(cudaStream_t,
        void (*)(cudaStream_t, cudaError_t, void*) noexcept, void*);
    CurveCompactionStatus CommitFreshFinish();
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    // Sticky caller-side proof loss for raw-input consumers.  The next
    // destruction path must quarantine this compactor's device storage.
    void MarkUnprovenWork() noexcept { unprovenWork_ = true; }

    DeviceCurveGeometryView view() const noexcept;
    DeviceView<const float> hairT() const noexcept;
    DeviceView<const int32_t> rootPrim() const noexcept;
    DeviceView<const float2> rootUV() const noexcept;
    RestRootFrames frames() const noexcept;
    cudaError_t recordUse(cudaStream_t stream);
    cudaError_t waitOn(cudaStream_t stream) const;
    int deviceIndex() const noexcept { return deviceIndex_; }
    uint64_t generation() const noexcept { return generation_; }
    bool pending() const noexcept { return pending_; }
    size_t ExclusiveRetainedBytes() const noexcept;
    // Host-only: marks every completed allocation retained by the published
    // compactor as generation-pinned. Pending output storage is excluded.
    void ReclassifyPublishedGeneration() noexcept;

private:
    struct Storage {
        DeviceBuffer<float3> points;
        DeviceBuffer<float3> restPoints;
        DeviceBuffer<float> widths;
        DeviceBuffer<uint32_t> curveOffsets;
        DeviceBuffer<uint64_t> stableIds;
        DeviceBuffer<float> hairT;
        DeviceBuffer<int32_t> rootPrim;
        // DeviceBuffer<float2> is intentionally not part of the common
        // explicit-instantiation set; store the same bytes as CurveSource.
        DeviceBuffer<unsigned char> rootUV;
        DeviceBuffer<float3> rootT, rootB, rootN;
        size_t curveCount = 0;
        size_t pointCount = 0;

        void clear() noexcept;
        void quarantine() noexcept;
        void swap(Storage& other) noexcept;
        DeviceCurveGeometryView geometry() const noexcept;
        DeviceView<const float> hairTView() const noexcept;
        DeviceView<const int32_t> rootPrimView() const noexcept;
        DeviceView<const float2> rootUVView() const noexcept;
        RestRootFrames framesView() const noexcept;
        cudaError_t recordUse(cudaStream_t stream);
        cudaError_t waitOn(cudaStream_t stream) const;
        void ReclassifyPublishedGeneration() noexcept;
        size_t RetainedBytes() const noexcept;
    };

    cudaError_t validateStream(cudaStream_t stream) const;
    CurveCompactionStatus finishError(int code) const;

    Storage active_, pendingStorage_;
    DeviceBuffer<int> error_;
    DeviceBuffer<uint32_t> curveFlags_, curvePrefix_;
    DeviceBuffer<uint32_t> pointLengths_, pointPrefix_;
    DeviceBuffer<uint32_t> counts_;
    DeviceBuffer<unsigned char> scanTemp_;
    struct FreshCounts { int error; uint32_t curves; uint32_t points; };
    FreshCounts* freshHostCounts_ = nullptr;
    UsdGenExecutionResourcePermit freshHostCountsPermit_;
    DeviceCurveGeometryView freshGeometry_{};
    DeviceView<const float> freshHairT_{};
    DeviceView<const int32_t> freshRootPrim_{};
    DeviceView<const float2> freshRootUV_{};
    RestRootFrames freshFrames_{};
    DeviceView<const uint8_t> freshKeep_{};
    enum class FreshPhase { Idle, CountsPending, CountsArmed, CountsCommitted,
                            ScatterPending, ScatterArmed, Failed };
    FreshPhase freshPhase_ = FreshPhase::Idle;
    bool unprovenWork_ = false;
    cudaEvent_t ready_ = nullptr;
    bool pending_ = false;
    int deviceIndex_ = -1;
    uint64_t generation_ = 0;
};

// Test-only: fails the next fresh scatter before output allocation/submission.
void failNextCudaCurveCompactionFreshScatterAllocationForTesting();

} // namespace usdGen::gpu

#endif // USDGEN_GPU_CURVE_COMPACTION_H
