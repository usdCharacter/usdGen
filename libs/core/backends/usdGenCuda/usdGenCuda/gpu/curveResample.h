#ifndef USDGEN_GPU_CURVE_RESAMPLE_H
#define USDGEN_GPU_CURVE_RESAMPLE_H

#include "curveGeometry.h"

#include <cstddef>
#include <cstdint>

namespace usdGen::gpu {

enum class CurveResampleStatus { Ok, InvalidArgument, NonFiniteInput, CudaError, NoPendingUpdate };

// Device-only indexed-CV resampling.  target==0 clones the ragged layout;
// target>=2 makes every curve uniform.  Borrowed inputs remain live through
// Finish; active output is replaced only by a successful Finish.  Fresh
// inputs remain borrowed through the terminal callback and CommitFreshFinish.
class CudaCurveResample {
public:
    CudaCurveResample() = default;
    ~CudaCurveResample();
    CudaCurveResample(CudaCurveResample const&) = delete;
    CudaCurveResample& operator=(CudaCurveResample const&) = delete;
    CurveResampleStatus Apply(
        DeviceCurveGeometryView geometry, DeviceView<const float> hairT,
        DeviceView<const int32_t> rootPrim, DeviceView<const float2> rootUV,
        int32_t targetPointCount, cudaStream_t stream,
        UsdGenExecutionMemoryReservation* reservation = nullptr);
    CurveResampleStatus Finish(cudaStream_t);
    // Fresh-only asynchronous continuation. ApplyFresh performs the same
    // device validation as Apply but leaves its status on the stream; the
    // callback is a native CUDA callback and must only signal a host relay.
    CurveResampleStatus ApplyFresh(
        DeviceCurveGeometryView geometry, DeviceView<const float> hairT,
        DeviceView<const int32_t> rootPrim, DeviceView<const float2> rootUV,
        int32_t targetPointCount, cudaStream_t stream,
        UsdGenExecutionMemoryReservation* reservation = nullptr);
    CurveResampleStatus FinishFreshAsync(cudaStream_t,
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata);
    // Host-only after FinishFreshAsync has returned successfully and its
    // native callback reported success. It consumes the pinned status copied
    // before that callback and never calls/synchronizes CUDA.
    CurveResampleStatus CommitFreshFinish();
    // True from the first fresh submission until CommitFreshFinish establishes
    // terminal completion, and after terminal-install failure.  A caller
    // abandoning such an object must quarantine its owning workspace.
    bool HasUnprovenUpload() const noexcept { return unprovenUpload_; }
    // Used by a caller which has handed raw geometry to another asynchronous
    // candidate but then lost the candidate's completion proof.  This is a
    // host-only sticky bit; destruction will quarantine all resample storage.
    void MarkUnprovenUpload() noexcept { unprovenUpload_ = true; }
    DeviceCurveGeometryView view() const noexcept;
    DeviceView<const float> hairT() const noexcept;
    DeviceView<const int32_t> rootPrim() const noexcept;
    DeviceView<const float2> rootUV() const noexcept;
    cudaError_t recordUse(cudaStream_t);
    cudaError_t waitOn(cudaStream_t) const;
    int deviceIndex() const noexcept { return device_; }
    bool pending() const noexcept { return pending_; }
    size_t ExclusiveRetainedBytes() const noexcept;
    // Host-only: marks every completed allocation retained by this published
    // owner as generation-pinned. Only staging remains transient.
    void ReclassifyPublishedGeneration() noexcept;
private:
    struct Storage {
        DeviceBuffer<float3> points, restPoints;
        DeviceBuffer<float> widths, hairT;
        DeviceBuffer<uint32_t> curveOffsets;
        DeviceBuffer<uint64_t> stableIds;
        DeviceBuffer<int32_t> rootPrim;
        DeviceBuffer<unsigned char> rootUV;
        size_t curveCount = 0, pointCount = 0;
        void clear() noexcept;
        void quarantine() noexcept;
        void swap(Storage&) noexcept;
        DeviceCurveGeometryView geometry() const noexcept;
        DeviceView<const float2> rootUVView() const noexcept;
        cudaError_t recordUse(cudaStream_t);
        cudaError_t waitOn(cudaStream_t) const;
        void ReclassifyPublishedGeneration() noexcept;
        size_t RetainedBytes() const noexcept;
    } active_, staging_;
    CurveResampleStatus finishError(int error) const;
    void discardStaging(cudaStream_t stream) noexcept;
    DeviceBuffer<int> error_;
    int* freshHostError_ = nullptr;
    UsdGenExecutionResourcePermit freshHostErrorPermit_;
    cudaEvent_t ready_ = nullptr;
    int device_ = -1;
    bool pending_ = false;
    bool freshPending_ = false;
    bool freshCallbackArmed_ = false;
    // Set before the first fresh stream submission and cleared only by the
    // host relay's post-callback commit.  It makes destruction conservative.
    bool unprovenUpload_ = false;
    // Unlike unprovenUpload_, this denotes an unrecoverable enqueue/terminal
    // installation failure and prevents another terminal installation.
    bool freshUploadFailed_ = false;
    cudaError_t validateStream(cudaStream_t) const;
};
}
#endif
