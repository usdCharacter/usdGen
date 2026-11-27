#include "usdGenCurves/cuda/curveSource.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <cmath>
#include <limits>
#include <new>
#include <unordered_set>
#include <utility>

namespace usdGen::gpu {
namespace {

bool ValidSpan(const CurveSourceSpan<int32_t> &span) {
    return span.size == 0 || span.data != nullptr;
}
template <class T> bool ValidSpan(CurveSourceSpan<T> const &span) {
    return span.size == 0 || span.data != nullptr;
}

template <class T> bool FiniteValues(CurveSourceSpan<T> const &) { return true; }
template <> bool FiniteValues(CurveSourceSpan<float> const &span) {
    for (size_t i = 0; i < span.size; ++i)
        if (!std::isfinite(span.data[i])) return false;
    return true;
}
template <> bool FiniteValues(CurveSourceSpan<float3> const &span) {
    for (size_t i = 0; i < span.size; ++i) {
        float3 const &v = span.data[i];
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
            return false;
    }
    return true;
}

CurveSourceStatus CudaStatus(cudaError_t error) {
    return error == cudaSuccess ? CurveSourceStatus::Ok : CurveSourceStatus::CudaError;
}

} // namespace

DeviceCurveGeometryView CudaCurveSource::Storage::view(size_t curves,
                                                        size_t pointsCount) const {
    return {points.view(), restPoints.view(), widths.view(), offsets.view(),
            stableIds.view(), curves, pointsCount};
}

CurveSourceStatus CudaCurveSource::Storage::recordUse(cudaStream_t stream) {
    cudaError_t error = points.recordUse(stream);
    if (error == cudaSuccess) error = restPoints.recordUse(stream);
    if (error == cudaSuccess) error = widths.recordUse(stream);
    if (error == cudaSuccess) error = hairT.recordUse(stream);
    if (error == cudaSuccess) error = offsets.recordUse(stream);
    if (error == cudaSuccess) error = stableIds.recordUse(stream);
    if (error == cudaSuccess) error = rootPrim.recordUse(stream);
    if (error == cudaSuccess) error = rootUV.recordUse(stream);
    return CudaStatus(error);
}

CurveSourceStatus CudaCurveSource::Storage::waitOn(cudaStream_t stream) const {
    cudaError_t error = points.waitOn(stream);
    if (error == cudaSuccess) error = restPoints.waitOn(stream);
    if (error == cudaSuccess) error = widths.waitOn(stream);
    if (error == cudaSuccess) error = hairT.waitOn(stream);
    if (error == cudaSuccess) error = offsets.waitOn(stream);
    if (error == cudaSuccess) error = stableIds.waitOn(stream);
    if (error == cudaSuccess) error = rootPrim.waitOn(stream);
    if (error == cudaSuccess) error = rootUV.waitOn(stream);
    return CudaStatus(error);
}

void CudaCurveSource::Storage::quarantine() noexcept {
    points.quarantine();
    restPoints.quarantine();
    widths.quarantine();
    hairT.quarantine();
    offsets.quarantine();
    stableIds.quarantine();
    rootPrim.quarantine();
    rootUV.quarantine();
}

void CudaCurveSource::Storage::ReclassifyPublishedGeneration() noexcept {
    points.Reclassify(UsdGenExecutionResourceKind::Pinned);
    restPoints.Reclassify(UsdGenExecutionResourceKind::Pinned);
    widths.Reclassify(UsdGenExecutionResourceKind::Pinned);
    hairT.Reclassify(UsdGenExecutionResourceKind::Pinned);
    offsets.Reclassify(UsdGenExecutionResourceKind::Pinned);
    stableIds.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootPrim.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootUV.Reclassify(UsdGenExecutionResourceKind::Pinned);
}

size_t CudaCurveSource::Storage::RetainedBytes() const noexcept {
    size_t total = 0;
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    add(points.bytes()); add(restPoints.bytes()); add(widths.bytes());
    add(hairT.bytes()); add(offsets.bytes()); add(stableIds.bytes());
    add(rootPrim.bytes()); add(rootUV.bytes());
    return total;
}

CudaCurveSource::~CudaCurveSource() {
    // A failed upload/terminal proof leaves stream ownership unknowable.  In
    // that state do not query, wait, free, or destroy its CUDA objects: the
    // allocation permits deliberately remain charged for context teardown.
    if (unprovenUpload_) {
        active_.quarantine();
        pendingStorage_.quarantine();
        ready_ = nullptr;
        return;
    }
    // DeviceBuffer::release needs the owning CUDA device.  Prove the source
    // producer event before ordinary destruction; any failed proof becomes
    // conservative quarantine.  Published consumer fences are retired by
    // SourceOwner before this owner reaches its retirement payload.  Waiting
    // for them again here can deadlock a relay gate on the default stream.
    const bool owns = active_.points.size() || active_.offsets.size() ||
        pendingStorage_.points.size() || pendingStorage_.offsets.size() || ready_;
    int previous = -1;
    const bool selected = !owns ||
        (cudaGetDevice(&previous) == cudaSuccess && deviceIndex_ >= 0 &&
         cudaSetDevice(deviceIndex_) == cudaSuccess);
    const bool producerFinished = !owns ||
        // shutdown wait: destructor.
        (selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess));
    if (!selected || !producerFinished) {
        active_.quarantine();
        pendingStorage_.quarantine();
        ready_ = nullptr;
        if (selected && previous >= 0 && previous != deviceIndex_)
            cudaSetDevice(previous);
        return;
    }
    if (ready_) {
        cudaEventDestroy(ready_);
        ready_ = nullptr;
    }
    active_ = Storage{};
    pendingStorage_ = Storage{};
    pendingPoints_.clear();
    pendingRestPoints_.clear();
    pendingWidths_.clear();
    pendingHairT_.clear();
    pendingOffsets_.clear();
    pendingStableIds_.clear();
    pendingRootPrim_.clear();
    pendingRootUV_.clear();
    if (selected && previous >= 0 && previous != deviceIndex_)
        cudaSetDevice(previous);
}

CurveSourceStatus CudaCurveSource::validateStream(cudaStream_t stream) const {
    if (deviceIndex_ < 0) return CurveSourceStatus::Ok;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
        return CurveSourceStatus::CudaError;
    if (current != deviceIndex_) return CurveSourceStatus::InvalidArgument;
    if (!stream) return CurveSourceStatus::Ok;
    int streamDevice = -1;
    if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
        return CurveSourceStatus::CudaError;
    return streamDevice == deviceIndex_ ? CurveSourceStatus::Ok
                                        : CurveSourceStatus::InvalidArgument;
}

CurveSourceStatus CudaCurveSource::validate(
    CurveSourceInput input, std::vector<float3> &points,
    std::vector<float3> &restPoints, std::vector<float> &widths,
    std::vector<float> &hairT, std::vector<uint32_t> &offsets,
    std::vector<uint64_t> &stableIds, std::vector<int32_t> &rootPrim,
    std::vector<float2> &rootUV, uint32_t &warningFlags) const {
    warningFlags = CurveSourceWarningNone;
    if (!ValidSpan(input.curveVertexCounts) || !ValidSpan(input.points) ||
        !ValidSpan(input.restPoints) || !ValidSpan(input.widths) ||
        !ValidSpan(input.hairT) || !ValidSpan(input.stableIds) ||
        !ValidSpan(input.rootPrim) || !ValidSpan(input.rootUV))
        return CurveSourceStatus::InvalidArgument;
    const size_t curves = input.curveVertexCounts.size;
    // Offsets store uint32 endpoints and the storage contract exposes counts
    // to CUDA kernels as bounded launch dimensions. Check before curves + 1,
    // reserve, or any source-array indexing can overflow.
    if (curves > size_t(std::numeric_limits<int>::max()) ||
        curves > size_t(std::numeric_limits<uint32_t>::max()) / 2u ||
        curves == std::numeric_limits<size_t>::max())
        return CurveSourceStatus::InvalidTopology;
    size_t totalPoints = 0;
    for (size_t c = 0; c < curves; ++c) {
        const int32_t count = input.curveVertexCounts.data[c];
        // C3 pinned curves need a root and a tip; a one-CV source is not a
        // valid CurveSource strand even though some downstream math supports it.
        if (count < 2) return CurveSourceStatus::InvalidTopology;
        if (totalPoints > std::numeric_limits<uint32_t>::max() - size_t(count) ||
            totalPoints > std::numeric_limits<int>::max() - size_t(count))
            return CurveSourceStatus::InvalidTopology;
        totalPoints += size_t(count);
    }
    if (input.points.size != totalPoints ||
        (input.restPoints.size != 0 && input.restPoints.size != totalPoints) ||
        (input.widths.size != 0 && input.widths.size != totalPoints) ||
        (input.hairT.size != 0 && input.hairT.size != totalPoints) ||
        (input.stableIds.size != 0 && input.stableIds.size != curves) ||
        (input.rootPrim.size != 0 && input.rootPrim.size != curves) ||
        (input.rootUV.size != 0 && input.rootUV.size != curves))
        return CurveSourceStatus::InvalidArgument;
    if (!FiniteValues(input.points) || !FiniteValues(input.restPoints) ||
        !FiniteValues(input.widths) || !FiniteValues(input.hairT))
        return CurveSourceStatus::NonFiniteInput;
    for (size_t i = 0; i < input.rootUV.size; ++i)
        if (!std::isfinite(input.rootUV.data[i].x) ||
            !std::isfinite(input.rootUV.data[i].y))
            return CurveSourceStatus::NonFiniteInput;
    if (!std::isfinite(input.fallbackWidth) || input.fallbackWidth < 0.0f)
        return std::isfinite(input.fallbackWidth)
            ? CurveSourceStatus::InvalidArgument : CurveSourceStatus::NonFiniteInput;
    for (size_t i = 0; i < input.widths.size; ++i)
        if (input.widths.data[i] < 0.0f) return CurveSourceStatus::InvalidArgument;
    for (size_t i = 0; i < input.hairT.size; ++i)
        if (input.hairT.data[i] < 0.0f || input.hairT.data[i] > 1.0f)
            return CurveSourceStatus::InvalidArgument;
    if (input.hairT.size != 0) {
        size_t base = 0;
        for (size_t c = 0; c < curves; ++c) {
            const size_t count = size_t(input.curveVertexCounts.data[c]);
            // C3 hairT is a normalized root-to-tip parameter. Preserve
            // authored interior values, but reject a source that cannot be
            // consumed as a pinned strand rather than silently renormalizing.
            if (input.hairT.data[base] != 0.0f ||
                input.hairT.data[base + count - 1] != 1.0f)
                return CurveSourceStatus::InvalidArgument;
            for (size_t i = 1; i < count; ++i)
                if (input.hairT.data[base + i] < input.hairT.data[base + i - 1])
                    return CurveSourceStatus::InvalidArgument;
            base += count;
        }
    }

    if (input.stableIds.size != 0) {
        std::unordered_set<uint64_t> ids;
        ids.reserve(curves);
        for (size_t c = 0; c < curves; ++c)
            if (!ids.insert(input.stableIds.data[c]).second)
                return CurveSourceStatus::DuplicateStableId;
    }

    if (input.points.size != 0)
        points.assign(input.points.data, input.points.data + input.points.size);
    if (input.restPoints.size != 0)
        restPoints.assign(input.restPoints.data,
                          input.restPoints.data + input.restPoints.size);
    else {
        restPoints = points;
        warningFlags |= CurveSourceWarningSynthesizedRest;
    }
    if (input.widths.size != 0)
        widths.assign(input.widths.data, input.widths.data + input.widths.size);
    else {
        widths.assign(totalPoints, input.fallbackWidth);
        warningFlags |= CurveSourceWarningSynthesizedWidths;
    }
    offsets.resize(curves + 1);
    offsets[0] = 0;
    for (size_t c = 0; c < curves; ++c)
        offsets[c + 1] = offsets[c] + uint32_t(input.curveVertexCounts.data[c]);
    if (input.hairT.size != 0) {
        hairT.assign(input.hairT.data, input.hairT.data + input.hairT.size);
    } else {
        hairT.resize(totalPoints);
        warningFlags |= CurveSourceWarningSynthesizedHairT;
        size_t base = 0;
        for (size_t c = 0; c < curves; ++c) {
            const size_t count = size_t(input.curveVertexCounts.data[c]);
            for (size_t i = 0; i < count; ++i)
                hairT[base + i] = count > 1 ? float(i) / float(count - 1) : 0.0f;
            base += count;
        }
    }
    if (input.stableIds.size != 0)
        stableIds.assign(input.stableIds.data,
                         input.stableIds.data + input.stableIds.size);
    else {
        stableIds.resize(curves);
        warningFlags |= CurveSourceWarningSynthesizedStableIds;
        for (size_t c = 0; c < curves; ++c) stableIds[c] = uint64_t(c);
    }
    if (input.rootPrim.size != 0)
        rootPrim.assign(input.rootPrim.data, input.rootPrim.data + input.rootPrim.size);
    if (input.rootUV.size != 0)
        rootUV.assign(input.rootUV.data, input.rootUV.data + input.rootUV.size);
    return CurveSourceStatus::Ok;
}

CurveSourceStatus CudaCurveSource::copyPending(
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation) {
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return CurveSourceStatus::CudaError;
    constexpr auto active = UsdGenExecutionResourceKind::Active;
    if (pendingStorage_.points.reset(pendingPointCount_, reservation, active) != cudaSuccess ||
        pendingStorage_.restPoints.reset(pendingPointCount_, reservation, active) != cudaSuccess ||
        pendingStorage_.widths.reset(pendingPointCount_, reservation, active) != cudaSuccess ||
        pendingStorage_.hairT.reset(pendingPointCount_, reservation, active) != cudaSuccess ||
        pendingStorage_.offsets.reset(pendingCurveCount_ + 1, reservation, active) != cudaSuccess ||
        pendingStorage_.stableIds.reset(pendingCurveCount_, reservation, active) != cudaSuccess ||
        pendingStorage_.rootPrim.reset(pendingRootPrim_.size(), reservation, active) != cudaSuccess ||
        pendingStorage_.rootUV.reset(pendingRootUV_.size() * sizeof(float2), reservation, active) != cudaSuccess) {
        // No upload has been submitted yet.  Release any sequentially
        // allocated partial storage and reset host candidate bookkeeping so
        // this object cannot look publishable with stale Active permits.
        DiscardPendingCandidate();
        return CurveSourceStatus::CudaError;
    }
    auto copy = [stream](void *dst, const void *src, size_t bytes) {
        return bytes == 0 ? cudaSuccess : cudaMemcpyAsync(
            dst, src, bytes, cudaMemcpyHostToDevice, stream);
    };
    cudaError_t error = copy(pendingStorage_.points.data(), pendingPoints_.data(),
                             pendingPointCount_ * sizeof(float3));
    if (error == cudaSuccess) error = copy(pendingStorage_.restPoints.data(), pendingRestPoints_.data(),
                                           pendingPointCount_ * sizeof(float3));
    if (error == cudaSuccess) error = copy(pendingStorage_.widths.data(), pendingWidths_.data(),
                                           pendingPointCount_ * sizeof(float));
    if (error == cudaSuccess) error = copy(pendingStorage_.hairT.data(), pendingHairT_.data(),
                                           pendingPointCount_ * sizeof(float));
    if (error == cudaSuccess) error = copy(pendingStorage_.offsets.data(), pendingOffsets_.data(),
                                           (pendingCurveCount_ + 1) * sizeof(uint32_t));
    if (error == cudaSuccess) error = copy(pendingStorage_.stableIds.data(), pendingStableIds_.data(),
                                           pendingCurveCount_ * sizeof(uint64_t));
    if (error == cudaSuccess) error = copy(pendingStorage_.rootPrim.data(), pendingRootPrim_.data(),
                                           pendingRootPrim_.size() * sizeof(int32_t));
    if (error == cudaSuccess) error = copy(pendingStorage_.rootUV.data(), pendingRootUV_.data(),
                                           pendingRootUV_.size() * sizeof(float2));
    if (error != cudaSuccess) {
        // Some earlier copies can already be queued.  A failed cleanup fence
        // is not permission to destroy their device buffers or host staging.
        if (CudaFenceStream(stream) != cudaSuccess)
            MarkUnprovenUpload();
        else
            DiscardPendingCandidate();
        return CurveSourceStatus::CudaError;
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) {
        // The copies preceding a failed event record still need a terminal
        // proof before ordinary source destruction is safe.
        if (CudaFenceStream(stream) != cudaSuccess)
            MarkUnprovenUpload();
        else
            DiscardPendingCandidate();
        return CurveSourceStatus::CudaError;
    }
    pending_ = true;
    return CurveSourceStatus::Ok;
}

void CudaCurveSource::DiscardPendingCandidate() noexcept {
    pendingStorage_ = Storage{};
    pendingPoints_.clear();
    pendingRestPoints_.clear();
    pendingWidths_.clear();
    pendingHairT_.clear();
    pendingOffsets_.clear();
    pendingStableIds_.clear();
    pendingRootPrim_.clear();
    pendingRootUV_.clear();
    pendingCurveCount_ = pendingPointCount_ = 0;
    pendingWarningFlags_ = CurveSourceWarningNone;
    pending_ = false;
}

CurveSourceStatus CudaCurveSource::Set(
    CurveSourceInput input, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
        return CurveSourceStatus::CudaError;
    int streamDevice = current;
    if (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
        return CurveSourceStatus::CudaError;
    if (streamDevice != current)
        return CurveSourceStatus::InvalidArgument;
    if (deviceIndex_ < 0)
        deviceIndex_ = current;
    else if (deviceIndex_ != current)
        return CurveSourceStatus::InvalidArgument;
    if (stream && streamDevice != deviceIndex_)
        return CurveSourceStatus::InvalidArgument;
    if (pending_) return CurveSourceStatus::InvalidArgument;
    // A caller which saw an unproven CUDA failure must quarantine rather than
    // attempting to reuse the source on an unknown stream state.
    if (unprovenUpload_) return CurveSourceStatus::CudaError;
    std::vector<float3> points, restPoints;
    std::vector<float> widths, hairT;
    std::vector<uint32_t> offsets;
    std::vector<uint64_t> stableIds;
    std::vector<int32_t> rootPrim;
    std::vector<float2> rootUV;
    uint32_t warningFlags = CurveSourceWarningNone;
    CurveSourceStatus status = CurveSourceStatus::Ok;
    try {
        status = validate(input, points, restPoints, widths, hairT,
                          offsets, stableIds, rootPrim, rootUV, warningFlags);
    } catch (const std::bad_alloc &) {
        // Host staging is part of the authored boundary. Treat inability to
        // build it like a failed upload and leave the active generation alone.
        return CurveSourceStatus::CudaError;
    }
    if (status != CurveSourceStatus::Ok) return status;
    pendingPoints_ = std::move(points);
    pendingRestPoints_ = std::move(restPoints);
    pendingWidths_ = std::move(widths);
    pendingHairT_ = std::move(hairT);
    pendingOffsets_ = std::move(offsets);
    pendingStableIds_ = std::move(stableIds);
    pendingRootPrim_ = std::move(rootPrim);
    pendingRootUV_ = std::move(rootUV);
    pendingCurveCount_ = input.curveVertexCounts.size;
    pendingPointCount_ = pendingPoints_.size();
    pendingWarningFlags_ = warningFlags;
    return copyPending(stream, reservation);
}

CurveSourceStatus CudaCurveSource::Finish(cudaStream_t stream) {
    if (!pending_) return CurveSourceStatus::NoPendingUpdate;
    CurveSourceStatus streamStatus = validateStream(stream);
    if (streamStatus != CurveSourceStatus::Ok) {
        MarkUnprovenUpload();
        return streamStatus;
    }
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) {
        MarkUnprovenUpload();
        return CurveSourceStatus::CudaError;
    }
    CurveSourceStatus status = active_.waitOn(stream);
    if (status != CurveSourceStatus::Ok || CudaFenceStream(stream) != cudaSuccess) {
        MarkUnprovenUpload();
        return CurveSourceStatus::CudaError;
    }
    // A legacy Finish on the valid source stream has now proven both this
    // pending upload and the prior active consumer fence.  An earlier async
    // capture/arm rejection is therefore no longer a reason to refuse the
    // otherwise established synchronous replacement contract.
    unprovenUpload_ = false;
    return CommitPending();
}

CurveSourceStatus CudaCurveSource::FinishFreshAsync(cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata) {
    if (!pending_ || !callback) return CurveSourceStatus::NoPendingUpdate;
    // This is deliberately a fresh producer path: replacing an active source
    // requires its consumer-use fence and remains the legacy Finish contract.
    if (generation_ != 0 || curveCount_ != 0 || pointCount_ != 0)
        return CurveSourceStatus::InvalidArgument;
    // Capture detection must be the first CUDA query. In particular,
    // cudaStreamGetDevice is not capture-safe on every supported runtime and
    // can turn a clean rejection into a failed/invalidated capture.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        // Set has already enqueued the upload.  The caller must retain it
        // rather than treating capture rejection as a synchronous rollback.
        MarkUnprovenUpload();
        return CurveSourceStatus::InvalidArgument;
    }
    CurveSourceStatus streamStatus = validateStream(stream);
    if (streamStatus != CurveSourceStatus::Ok) {
        MarkUnprovenUpload();
        return streamStatus;
    }
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess) {
        MarkUnprovenUpload();
        return CurveSourceStatus::CudaError;
    }
    if (cudaStreamAddCallback(stream, callback, userdata, 0) != cudaSuccess) {
        MarkUnprovenUpload();
        return CurveSourceStatus::CudaError;
    }
    return CurveSourceStatus::Ok;
}

CurveSourceStatus CudaCurveSource::CommitFreshFinish() {
    if (!pending_) return CurveSourceStatus::NoPendingUpdate;
    // This callback path is only safe for a newly-created source.  Legacy
    // replacement must retain its active-consumer fence in Finish.
    if (unprovenUpload_ || generation_ != 0 || curveCount_ != 0 || pointCount_ != 0)
        return CurveSourceStatus::InvalidArgument;
    return CommitPending();
}

CurveSourceStatus CudaCurveSource::CommitPending() {
    if (!pending_) return CurveSourceStatus::NoPendingUpdate;
    if (unprovenUpload_) return CurveSourceStatus::CudaError;
    active_ = std::move(pendingStorage_);
    curveCount_ = pendingCurveCount_;
    pointCount_ = pendingPointCount_;
    pendingPoints_.clear();
    pendingRestPoints_.clear();
    pendingWidths_.clear();
    pendingHairT_.clear();
    pendingOffsets_.clear();
    pendingStableIds_.clear();
    pendingRootPrim_.clear();
    pendingRootUV_.clear();
    pendingCurveCount_ = pendingPointCount_ = 0;
    pending_ = false;
    unprovenUpload_ = false;
    warningFlags_ = pendingWarningFlags_;
    pendingWarningFlags_ = CurveSourceWarningNone;
    ++generation_;
    return CurveSourceStatus::Ok;
}

DeviceCurveGeometryView CudaCurveSource::view() const {
    return active_.view(curveCount_, pointCount_);
}

DeviceView<const float> CudaCurveSource::hairT() const {
    return active_.hairT.view();
}

CurveSourceStatus CudaCurveSource::recordUse(cudaStream_t stream) {
    CurveSourceStatus streamStatus = validateStream(stream);
    if (streamStatus != CurveSourceStatus::Ok) return streamStatus;
    return active_.recordUse(stream);
}

CurveSourceStatus CudaCurveSource::waitOn(cudaStream_t stream) const {
    CurveSourceStatus streamStatus = validateStream(stream);
    if (streamStatus != CurveSourceStatus::Ok) return streamStatus;
    return active_.waitOn(stream);
}

DeviceView<const int32_t> CudaCurveSource::rootPrim() const {
    return active_.rootPrim.view();
}

DeviceView<const float2> CudaCurveSource::rootUV() const {
    return {reinterpret_cast<const float2 *>(active_.rootUV.data()),
            active_.rootUV.size() / sizeof(float2)};
}

void CudaCurveSource::ReclassifyPublishedGeneration() noexcept {
    // active_ is the only storage retained by a completed source.  The host
    // pending vectors and pendingStorage_ are intentionally not reclassified.
    active_.ReclassifyPublishedGeneration();
}

size_t CudaCurveSource::ExclusiveRetainedBytes() const noexcept {
    return active_.RetainedBytes();
}

} // namespace usdGen::gpu
