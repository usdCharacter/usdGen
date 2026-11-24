#ifndef USDGEN_GPU_CURVE_SOURCE_H
#define USDGEN_GPU_CURVE_SOURCE_H

#include "curveGeometry.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace usdGen::gpu {

template <class T> struct CurveSourceSpan {
    const T *data = nullptr;
    size_t size = 0;
};

// Plain authored C3 input. This intentionally has no USD or Hydra types, so
// the scene-index boundary can translate graphDesc data without coupling the
// CUDA library to pxr. Spans are consumed synchronously by Set; device views
// from the last successful generation remain owned until the next publish.
struct CurveSourceInput {
    CurveSourceSpan<int32_t> curveVertexCounts;
    CurveSourceSpan<float3> points;
    CurveSourceSpan<float3> restPoints; // empty: copy points as rest
    CurveSourceSpan<float> widths;      // empty: fill with fallbackWidth
    CurveSourceSpan<float> hairT;       // empty: canonical index interpolation
    CurveSourceSpan<uint64_t> stableIds; // empty: use curve index
    CurveSourceSpan<int32_t> rootPrim;  // optional authored skinprim
    CurveSourceSpan<float2> rootUV;     // optional authored skinprimuv
    float fallbackWidth = 0.01f;       // explicit default for absent widths
};

enum class CurveSourceStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidTopology,
    DuplicateStableId,
    NoPendingUpdate,
    CudaError
};

enum CurveSourceWarningFlags : uint32_t {
    CurveSourceWarningNone = 0,
    CurveSourceWarningSynthesizedRest = 1u << 0,
    CurveSourceWarningSynthesizedWidths = 1u << 1,
    CurveSourceWarningSynthesizedHairT = 1u << 2,
    CurveSourceWarningSynthesizedStableIds = 1u << 3
};

// Authored CurveSource upload boundary. The first accepted Set binds the
// source to the current CUDA device; later Set/Finish/consumer streams must
// belong to that same device. Set validates and asynchronously copies a source
// into owned staging buffers. The previous published source remains visible
// until Finish succeeds; Finish reads no generated geometry back to the host.
// Call recordUse/waitOn when consumers use view() across streams, just as for
// CudaCurveGeometry.
class CudaCurveSource {
public:
    CudaCurveSource() = default;
    ~CudaCurveSource();
    CudaCurveSource(const CudaCurveSource&) = delete;
    CudaCurveSource& operator=(const CudaCurveSource&) = delete;

    CurveSourceStatus Set(CurveSourceInput input, cudaStream_t stream,
                          UsdGenExecutionMemoryReservation* reservation = nullptr);
    CurveSourceStatus Finish(cudaStream_t stream);
    // Fresh-source async terminal path. `callback` runs on CUDA's native
    // callback thread with the stream terminal status; it must only signal a
    // pre-reserved relay and must not destroy `userdata`. This path rejects
    // stream capture and never waits on/replaces an older active source.
    CurveSourceStatus FinishFreshAsync(cudaStream_t stream,
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata);
    // Called by the relay only after FinishFreshAsync's callback reported
    // cudaSuccess. Pure host ownership commit; no CUDA operation or wait.
    CurveSourceStatus CommitFreshFinish();
    // True after an upload path has queued work but failed to establish a
    // completion proof.  The caller must retain/quarantine this source and
    // its workspace; destroying DeviceBuffers in that state is unsafe.
    bool HasUnprovenUpload() const noexcept { return unprovenUpload_; }
    // A raw-input consumer can transfer completion-proof responsibility to
    // this owner.  Set this only when that consumer's stream proof is lost;
    // destruction then quarantines the source storage.
    void MarkUnprovenUpload() noexcept { unprovenUpload_ = true; }

    DeviceCurveGeometryView view() const;
    DeviceView<const float> hairT() const;
    DeviceView<const int32_t> rootPrim() const;
    DeviceView<const float2> rootUV() const;
    CurveSourceStatus recordUse(cudaStream_t stream);
    CurveSourceStatus waitOn(cudaStream_t stream) const;
    bool pending() const { return pending_; }
    // CUDA device that owns this source's allocations, or -1 before the
    // first accepted Set.  A source cannot be moved between CUDA devices.
    int deviceIndex() const { return deviceIndex_; }
    uint64_t generation() const { return generation_; }
    size_t curveCount() const { return curveCount_; }
    size_t pointCount() const { return pointCount_; }
    size_t ExclusiveRetainedBytes() const noexcept;
    uint32_t warningFlags() const { return warningFlags_; }
    // Reclassifies only this completed source's active output channels.  It
    // is host-only; pending upload storage is intentionally excluded.
    void ReclassifyPublishedGeneration() noexcept;

private:
    struct Storage {
        DeviceBuffer<float3> points;
        DeviceBuffer<float3> restPoints;
        DeviceBuffer<float> widths;
        DeviceBuffer<float> hairT;
        DeviceBuffer<uint32_t> offsets;
        DeviceBuffer<uint64_t> stableIds;
        DeviceBuffer<int32_t> rootPrim;
        // DeviceBuffer has explicit instantiations for the public geometry
        // channels; keep float2 UVs in an aligned CUDA byte allocation.
        DeviceBuffer<unsigned char> rootUV;

        DeviceCurveGeometryView view(size_t curves, size_t pointsCount) const;
        CurveSourceStatus recordUse(cudaStream_t stream);
        CurveSourceStatus waitOn(cudaStream_t stream) const;
        void quarantine() noexcept;
        void ReclassifyPublishedGeneration() noexcept;
        size_t RetainedBytes() const noexcept;
    };

    CurveSourceStatus validate(CurveSourceInput input,
                               std::vector<float3> &points,
                               std::vector<float3> &restPoints,
                               std::vector<float> &widths,
                               std::vector<float> &hairT,
                               std::vector<uint32_t> &offsets,
                               std::vector<uint64_t> &stableIds,
                               std::vector<int32_t> &rootPrim,
                               std::vector<float2> &rootUV,
                               uint32_t &warningFlags) const;
    CurveSourceStatus copyPending(cudaStream_t stream,
                                  UsdGenExecutionMemoryReservation* reservation);
    // Safe only before submission or after a successful terminal stream
    // proof.  It discards a partial candidate without touching active_.
    void DiscardPendingCandidate() noexcept;
    CurveSourceStatus validateStream(cudaStream_t stream) const;
    // General host commit used by the synchronous Finish path.  It permits
    // replacement of an existing active source after Finish fenced consumers.
    CurveSourceStatus CommitPending();

    Storage active_;
    Storage pendingStorage_;
    std::vector<float3> pendingPoints_, pendingRestPoints_;
    std::vector<float> pendingWidths_, pendingHairT_;
    std::vector<uint32_t> pendingOffsets_;
    std::vector<uint64_t> pendingStableIds_;
    std::vector<int32_t> pendingRootPrim_;
    std::vector<float2> pendingRootUV_;
    cudaEvent_t ready_ = nullptr;
    size_t curveCount_ = 0, pointCount_ = 0;
    size_t pendingCurveCount_ = 0, pendingPointCount_ = 0;
    uint32_t warningFlags_ = CurveSourceWarningNone;
    uint32_t pendingWarningFlags_ = CurveSourceWarningNone;
    uint64_t generation_ = 0;
    int deviceIndex_ = -1;
    bool pending_ = false;
    bool unprovenUpload_ = false;
};

} // namespace usdGen::gpu

#endif
