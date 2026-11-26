#include "usdGenCurves/cuda/curveResample.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace usdGen::gpu {
namespace {

constexpr int kBadShape = 1;
constexpr int kNonFinite = 2;
constexpr int kBadValue = 3;

__device__ void SetError(int* error, int code) { atomicCAS(error, 0, code); }
__device__ bool Finite(float value) { return isfinite(value); }
__device__ bool Finite(float2 value) { return Finite(value.x) && Finite(value.y); }
__device__ bool Finite(float3 value) { return Finite(value.x) && Finite(value.y) && Finite(value.z); }

// Double intermediates prevent (b - a) overflowing for finite float inputs.
__device__ float Blend(float a, float b, float fraction) {
    return static_cast<float>(double(a) + (double(b) - double(a)) * double(fraction));
}
__device__ float3 Blend(float3 a, float3 b, float fraction) {
    return make_float3(Blend(a.x, b.x, fraction), Blend(a.y, b.y, fraction),
                       Blend(a.z, b.z, fraction));
}

__global__ void ValidateInput(DeviceCurveGeometryView geometry,
                              DeviceView<const float> hairT,
                              DeviceView<const int32_t> rootPrim,
                              DeviceView<const float2> rootUV, int* error) {
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        (!geometry.curveOffsets.data || geometry.curveOffsets.size != geometry.curveCount + 1 ||
         geometry.curveOffsets.data[0] != 0 ||
         geometry.curveOffsets.data[geometry.curveCount] != geometry.pointCount))
        SetError(error, kBadShape);

    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t curve = first; curve < geometry.curveCount; curve += stride) {
        const uint32_t begin = geometry.curveOffsets.data[curve];
        const uint32_t end = geometry.curveOffsets.data[curve + 1];
        if (begin > end || end > geometry.pointCount || end - begin < 2 ||
            !geometry.stableIds.data)
            SetError(error, kBadShape);
        if (rootPrim.data && (rootPrim.data[curve] < 0 || !rootUV.data || !Finite(rootUV.data[curve])))
            SetError(error, kBadValue);
        if (rootUV.data && (rootUV.data[curve].x < 0.0f || rootUV.data[curve].x > 1.0f ||
                            rootUV.data[curve].y < 0.0f || rootUV.data[curve].y > 1.0f))
            SetError(error, kBadValue);
        if (hairT.data && begin < end && end <= geometry.pointCount) {
            if (hairT.data[begin] != 0.0f || hairT.data[end - 1] != 1.0f)
                SetError(error, kBadValue);
            for (uint32_t point = begin + 1; point < end; ++point)
                if (hairT.data[point] < hairT.data[point - 1]) SetError(error, kBadValue);
        }
    }
    for (size_t point = first; point < geometry.pointCount; point += stride) {
        if (!Finite(geometry.points.data[point]) ||
            (geometry.restPoints.data && !Finite(geometry.restPoints.data[point])) ||
            (geometry.widths.data && !Finite(geometry.widths.data[point])) ||
            (hairT.data && !Finite(hairT.data[point])))
            SetError(error, kNonFinite);
        if (geometry.widths.data && geometry.widths.data[point] < 0.0f) SetError(error, kBadValue);
        if (hairT.data && (hairT.data[point] < 0.0f || hairT.data[point] > 1.0f))
            SetError(error, kBadValue);
    }
}

struct ResampleOutput {
    DeviceView<float3> points;
    DeviceView<float3> restPoints;
    DeviceView<float> widths;
    DeviceView<uint32_t> curveOffsets;
    DeviceView<uint64_t> stableIds;
};

__global__ void Resample(DeviceCurveGeometryView input,
                         DeviceView<const float> inputHairT,
                         DeviceView<const int32_t> inputRootPrim,
                         DeviceView<const float2> inputRootUV,
                         uint32_t targetPointCount, ResampleOutput output,
                         DeviceView<float> outputHairT,
                         DeviceView<int32_t> outputRootPrim,
                         DeviceView<float2> outputRootUV, int* error) {
    // ApplyFresh runs ValidateInput in the same stream immediately before
    // this kernel.  Do not dereference a malformed device view when that
    // validation has already rejected it.
    if (atomicAdd(error, 0) != 0) return;
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t curve = first; curve < input.curveCount; curve += stride) {
        const uint32_t begin = input.curveOffsets.data[curve];
        const uint32_t inputCount = input.curveOffsets.data[curve + 1] - begin;
        const uint32_t outputCount = targetPointCount ? targetPointCount : inputCount;
        const uint32_t outputBegin = targetPointCount
            ? static_cast<uint32_t>(curve * size_t(targetPointCount)) : begin;
        output.curveOffsets.data[curve] = outputBegin;
        output.stableIds.data[curve] = input.stableIds.data[curve];
        if (outputRootPrim.data) outputRootPrim.data[curve] = inputRootPrim.data[curve];
        if (outputRootUV.data) outputRootUV.data[curve] = inputRootUV.data[curve];

        for (uint32_t outputPoint = 0; outputPoint < outputCount; ++outputPoint) {
            const double source = targetPointCount
                ? double(outputPoint) * double(inputCount - 1) / double(outputCount - 1)
                : double(outputPoint);
            const uint32_t lower = min(static_cast<uint32_t>(source), inputCount - 1);
            const uint32_t upper = min(lower + 1, inputCount - 1);
            const float fraction = static_cast<float>(source - double(lower));
            const uint32_t destination = outputBegin + outputPoint;
            output.points.data[destination] = Blend(input.points.data[begin + lower],
                                                     input.points.data[begin + upper], fraction);
            if (!Finite(output.points.data[destination])) SetError(error, kNonFinite);
            if (output.restPoints.data) {
                output.restPoints.data[destination] = Blend(input.restPoints.data[begin + lower],
                    input.restPoints.data[begin + upper], fraction);
                if (!Finite(output.restPoints.data[destination])) SetError(error, kNonFinite);
            }
            if (output.widths.data) {
                output.widths.data[destination] = Blend(input.widths.data[begin + lower],
                    input.widths.data[begin + upper], fraction);
                if (!Finite(output.widths.data[destination])) SetError(error, kNonFinite);
            }
            if (outputHairT.data) {
                outputHairT.data[destination] = Blend(inputHairT.data[begin + lower],
                    inputHairT.data[begin + upper], fraction);
                if (!Finite(outputHairT.data[destination])) SetError(error, kNonFinite);
            }
        }
        if (curve + 1 == input.curveCount) output.curveOffsets.data[curve + 1] = outputBegin + outputCount;
    }
}

__global__ void SetEmptyOffset(DeviceView<uint32_t> offsets) {
    if (blockIdx.x == 0 && threadIdx.x == 0) offsets.data[0] = 0;
}

unsigned Blocks(size_t count) {
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>((count + 255) / 256, 65535)));
}

cudaError_t CheckStream(int device, cudaStream_t stream) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return cudaErrorUnknown;
    if (device >= 0 && device != current) return cudaErrorInvalidDevice;
    if (!stream) return cudaSuccess;
    int streamDevice = -1;
    if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess) return cudaErrorUnknown;
    return streamDevice == current && (device < 0 || streamDevice == device)
        ? cudaSuccess : cudaErrorInvalidDevice;
}

} // namespace

void CudaCurveResample::Storage::clear() noexcept {
    points.reset(0); restPoints.reset(0); widths.reset(0); curveOffsets.reset(0);
    stableIds.reset(0); hairT.reset(0); rootPrim.reset(0); rootUV.reset(0);
    curveCount = pointCount = 0;
}
void CudaCurveResample::Storage::quarantine() noexcept {
    points.quarantine(); restPoints.quarantine(); widths.quarantine(); curveOffsets.quarantine();
    stableIds.quarantine(); hairT.quarantine(); rootPrim.quarantine(); rootUV.quarantine();
    curveCount = pointCount = 0;
}
void CudaCurveResample::Storage::swap(Storage& other) noexcept {
    using std::swap;
    swap(points, other.points); swap(restPoints, other.restPoints); swap(widths, other.widths);
    swap(curveOffsets, other.curveOffsets); swap(stableIds, other.stableIds);
    swap(hairT, other.hairT); swap(rootPrim, other.rootPrim); swap(rootUV, other.rootUV);
    swap(curveCount, other.curveCount); swap(pointCount, other.pointCount);
}
DeviceCurveGeometryView CudaCurveResample::Storage::geometry() const noexcept {
    return {points.view(), restPoints.view(), widths.view(), curveOffsets.view(), stableIds.view(),
            curveCount, pointCount};
}
DeviceView<const float2> CudaCurveResample::Storage::rootUVView() const noexcept {
    return {reinterpret_cast<float2 const*>(rootUV.data()), rootUV.size() / sizeof(float2)};
}
cudaError_t CudaCurveResample::Storage::recordUse(cudaStream_t stream) {
    cudaError_t result = points.recordUse(stream);
    if (result == cudaSuccess) result = restPoints.recordUse(stream);
    if (result == cudaSuccess) result = widths.recordUse(stream);
    if (result == cudaSuccess) result = curveOffsets.recordUse(stream);
    if (result == cudaSuccess) result = stableIds.recordUse(stream);
    if (result == cudaSuccess) result = hairT.recordUse(stream);
    if (result == cudaSuccess) result = rootPrim.recordUse(stream);
    if (result == cudaSuccess) result = rootUV.recordUse(stream);
    return result;
}
cudaError_t CudaCurveResample::Storage::waitOn(cudaStream_t stream) const {
    cudaError_t result = points.waitOn(stream);
    if (result == cudaSuccess) result = restPoints.waitOn(stream);
    if (result == cudaSuccess) result = widths.waitOn(stream);
    if (result == cudaSuccess) result = curveOffsets.waitOn(stream);
    if (result == cudaSuccess) result = stableIds.waitOn(stream);
    if (result == cudaSuccess) result = hairT.waitOn(stream);
    if (result == cudaSuccess) result = rootPrim.waitOn(stream);
    if (result == cudaSuccess) result = rootUV.waitOn(stream);
    return result;
}

void CudaCurveResample::Storage::ReclassifyPublishedGeneration() noexcept {
    points.Reclassify(UsdGenExecutionResourceKind::Pinned);
    restPoints.Reclassify(UsdGenExecutionResourceKind::Pinned);
    widths.Reclassify(UsdGenExecutionResourceKind::Pinned);
    hairT.Reclassify(UsdGenExecutionResourceKind::Pinned);
    curveOffsets.Reclassify(UsdGenExecutionResourceKind::Pinned);
    stableIds.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootPrim.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootUV.Reclassify(UsdGenExecutionResourceKind::Pinned);
}

size_t CudaCurveResample::Storage::RetainedBytes() const noexcept {
    size_t total = 0;
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    add(points.bytes()); add(restPoints.bytes()); add(widths.bytes());
    add(hairT.bytes()); add(curveOffsets.bytes()); add(stableIds.bytes());
    add(rootPrim.bytes()); add(rootUV.bytes());
    return total;
}

CudaCurveResample::~CudaCurveResample() {
    const bool owns = active_.curveOffsets.size() || staging_.curveOffsets.size() || error_.size() || ready_ || freshHostError_;
    // ready_ is recorded before the fresh D2H/status callback, so it cannot
    // prove a nonblocking producer stream reached that later terminal work.
    const bool freshCompletionUnproven = freshPending_ || unprovenUpload_;
    if (freshCompletionUnproven) {
        // Do not issue any CUDA query, wait, or free: an enqueue/terminal
        // failure can leave work on a stream which is no longer provable.
        active_.quarantine(); staging_.quarantine(); error_.quarantine(); ready_ = nullptr;
        freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
        return;
    }
    int previous = -1;
    const bool selected = !owns || (cudaGetDevice(&previous) == cudaSuccess && device_ >= 0 &&
                                    cudaSetDevice(device_) == cudaSuccess);
    const bool producerFinished = !owns ||
        (selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess));
    // DeviceBuffer::release does not wait for a recorded consumer event.
    // Honor recordUse() before freeing output storage on the owning device.
    const bool consumersFinished = !owns || (selected && producerFinished &&
        active_.waitOn(nullptr) == cudaSuccess && staging_.waitOn(nullptr) == cudaSuccess &&
        cudaStreamSynchronize(nullptr) == cudaSuccess);
    if (!selected || !producerFinished || !consumersFinished) {
        active_.quarantine(); staging_.quarantine(); error_.quarantine(); ready_ = nullptr;
        // The pinned status is callback/D2H storage.  With no completion
        // proof it must remain allocated with the quarantined payload rather
        // than being freed while a stream may still write it.
        freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
    } else {
        if (ready_) cudaEventDestroy(ready_);
        active_.clear(); staging_.clear(); error_.reset(0);
        if (freshHostError_) {
            if (cudaFreeHost(freshHostError_) == cudaSuccess) freshHostErrorPermit_.Release();
            else freshHostErrorPermit_.Abandon();
        }
        freshHostError_ = nullptr;
    }
    if (selected && previous >= 0 && previous != device_) cudaSetDevice(previous);
}
cudaError_t CudaCurveResample::validateStream(cudaStream_t stream) const { return CheckStream(device_, stream); }
CurveResampleStatus CudaCurveResample::finishError(int error) const {
    return error == kNonFinite ? CurveResampleStatus::NonFiniteInput : CurveResampleStatus::InvalidArgument;
}
void CudaCurveResample::discardStaging(cudaStream_t stream) noexcept {
    if (CudaFenceStream(stream) == cudaSuccess) staging_.clear(); else staging_.quarantine();
}

CurveResampleStatus CudaCurveResample::Apply(DeviceCurveGeometryView geometry,
    DeviceView<const float> inputHairT, DeviceView<const int32_t> inputRootPrim,
    DeviceView<const float2> inputRootUV, int32_t targetPointCount,
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation) {
    // A fresh submission has a callback/D2H lifetime which legacy Finish
    // cannot establish.  Fail closed instead of letting legacy reuse staging
    // after any queued fresh work or failed terminal installation.
    if (pending_ || freshPending_ || unprovenUpload_ || freshUploadFailed_ ||
        targetPointCount < 0 || targetPointCount == 1 || validateStream(stream) != cudaSuccess)
        return CurveResampleStatus::InvalidArgument;
    if (device_ < 0 && cudaGetDevice(&device_) != cudaSuccess) return CurveResampleStatus::CudaError;
    const size_t maxU32 = std::numeric_limits<uint32_t>::max();
    if (geometry.curveCount > maxU32 || geometry.pointCount > maxU32 ||
        (geometry.curveCount == 0) != (geometry.pointCount == 0) || !geometry.curveOffsets.data ||
        geometry.curveOffsets.size != geometry.curveCount + 1 ||
        (geometry.pointCount && (!geometry.points.data || geometry.points.size != geometry.pointCount)) ||
        (!geometry.pointCount && geometry.points.size != 0) ||
        (!geometry.restPoints.data && geometry.restPoints.size != 0) ||
        (geometry.restPoints.data && geometry.restPoints.size != geometry.pointCount) ||
        (!geometry.widths.data && geometry.widths.size != 0) ||
        (geometry.widths.data && geometry.widths.size != geometry.pointCount) ||
        (!geometry.stableIds.data && geometry.stableIds.size != 0) ||
        (geometry.stableIds.data && geometry.stableIds.size != geometry.curveCount) ||
        (!inputHairT.data && inputHairT.size != 0) || (inputHairT.data && inputHairT.size != geometry.pointCount) ||
        (!inputRootPrim.data && inputRootPrim.size != 0) || (!inputRootUV.data && inputRootUV.size != 0) ||
        ((inputRootPrim.size != 0) != (inputRootUV.size != 0)) ||
        (inputRootPrim.size && inputRootPrim.size != geometry.curveCount) ||
        (inputRootUV.size && inputRootUV.size != geometry.curveCount) ||
        (targetPointCount > 0 &&
         (geometry.curveCount > maxU32 / size_t(targetPointCount) ||
          geometry.curveCount > size_t(std::numeric_limits<int32_t>::max()) / size_t(targetPointCount))))
        return CurveResampleStatus::InvalidArgument;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return CurveResampleStatus::CudaError;
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return CurveResampleStatus::CudaError;
    ValidateInput<<<Blocks(std::max(geometry.curveCount, geometry.pointCount)), 256, 0, stream>>>(
        geometry, inputHairT, inputRootPrim, inputRootUV, error_.data());
    int error = 0;
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) return CurveResampleStatus::CudaError;
    if (error) return finishError(error);

    const size_t outputPointCount = targetPointCount ? geometry.curveCount * size_t(targetPointCount) : geometry.pointCount;
    staging_.clear(); staging_.curveCount = geometry.curveCount; staging_.pointCount = outputPointCount;
    constexpr auto active = UsdGenExecutionResourceKind::Active;
    if (staging_.curveOffsets.reset(geometry.curveCount + 1, reservation, active) != cudaSuccess ||
        (outputPointCount && staging_.points.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (geometry.restPoints.size && outputPointCount && staging_.restPoints.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (geometry.widths.size && outputPointCount && staging_.widths.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (geometry.curveCount && staging_.stableIds.reset(geometry.curveCount, reservation, active) != cudaSuccess) ||
        (inputHairT.size && outputPointCount && staging_.hairT.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (inputRootPrim.size && geometry.curveCount && staging_.rootPrim.reset(geometry.curveCount, reservation, active) != cudaSuccess) ||
        (inputRootUV.size && geometry.curveCount && staging_.rootUV.reset(geometry.curveCount * sizeof(float2), reservation, active) != cudaSuccess)) {
        staging_.clear(); return CurveResampleStatus::CudaError;
    }
    if (geometry.curveCount) {
        Resample<<<Blocks(geometry.curveCount), 256, 0, stream>>>(
            geometry, inputHairT, inputRootPrim, inputRootUV, static_cast<uint32_t>(targetPointCount),
            {staging_.points.view(), staging_.restPoints.view(), staging_.widths.view(),
             staging_.curveOffsets.view(), staging_.stableIds.view()},
            staging_.hairT.view(), staging_.rootPrim.view(),
            {reinterpret_cast<float2*>(staging_.rootUV.data()), geometry.curveCount}, error_.data());
    } else SetEmptyOffset<<<1, 1, 0, stream>>>(staging_.curveOffsets.view());
    if (cudaGetLastError() != cudaSuccess || cudaEventRecord(ready_, stream) != cudaSuccess) {
        discardStaging(stream); return CurveResampleStatus::CudaError;
    }
    pending_ = true;
    return CurveResampleStatus::Ok;
}

CurveResampleStatus CudaCurveResample::Finish(cudaStream_t stream) {
    if (freshPending_ || unprovenUpload_ || freshUploadFailed_)
        return CurveResampleStatus::InvalidArgument;
    if (validateStream(stream) != cudaSuccess) return CurveResampleStatus::InvalidArgument;
    if (!pending_) return CurveResampleStatus::NoPendingUpdate;
    int error = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) return CurveResampleStatus::CudaError;
    if (error) { pending_ = false; staging_.clear(); return finishError(error); }
    if (active_.curveOffsets.size() &&
        (active_.waitOn(stream) != cudaSuccess || CudaFenceStream(stream) != cudaSuccess))
        return CurveResampleStatus::CudaError;
    active_.swap(staging_); staging_.clear(); pending_ = false;
    return CurveResampleStatus::Ok;
}

CurveResampleStatus CudaCurveResample::ApplyFresh(DeviceCurveGeometryView geometry,
    DeviceView<const float> inputHairT, DeviceView<const int32_t> inputRootPrim,
    DeviceView<const float2> inputRootUV, int32_t targetPointCount,
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation) {
    if (pending_ || freshPending_ || unprovenUpload_ || active_.curveOffsets.size() ||
        targetPointCount < 0 || targetPointCount == 1)
        return CurveResampleStatus::InvalidArgument;
    // This must precede CheckStream/cudaStreamGetDevice: those queries are
    // not capture-safe on every supported CUDA runtime.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return CurveResampleStatus::InvalidArgument;
    if (validateStream(stream) != cudaSuccess) return CurveResampleStatus::InvalidArgument;
    if (device_ < 0 && cudaGetDevice(&device_) != cudaSuccess) return CurveResampleStatus::CudaError;
    const size_t maxU32 = std::numeric_limits<uint32_t>::max();
    if (geometry.curveCount > maxU32 || geometry.pointCount > maxU32 ||
        (geometry.curveCount == 0) != (geometry.pointCount == 0) || !geometry.curveOffsets.data ||
        geometry.curveOffsets.size != geometry.curveCount + 1 ||
        (geometry.pointCount && (!geometry.points.data || geometry.points.size != geometry.pointCount)) ||
        (!geometry.pointCount && geometry.points.size != 0) ||
        (!geometry.restPoints.data && geometry.restPoints.size != 0) ||
        (geometry.restPoints.data && geometry.restPoints.size != geometry.pointCount) ||
        (!geometry.widths.data && geometry.widths.size != 0) ||
        (geometry.widths.data && geometry.widths.size != geometry.pointCount) ||
        (!geometry.stableIds.data && geometry.stableIds.size != 0) ||
        (geometry.stableIds.data && geometry.stableIds.size != geometry.curveCount) ||
        (!inputHairT.data && inputHairT.size != 0) || (inputHairT.data && inputHairT.size != geometry.pointCount) ||
        (!inputRootPrim.data && inputRootPrim.size != 0) || (!inputRootUV.data && inputRootUV.size != 0) ||
        ((inputRootPrim.size != 0) != (inputRootUV.size != 0)) ||
        (inputRootPrim.size && inputRootPrim.size != geometry.curveCount) ||
        (inputRootUV.size && inputRootUV.size != geometry.curveCount) ||
        (targetPointCount > 0 && (geometry.curveCount > maxU32 / size_t(targetPointCount) ||
         geometry.curveCount > size_t(std::numeric_limits<int32_t>::max()) / size_t(targetPointCount))))
        return CurveResampleStatus::InvalidArgument;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return CurveResampleStatus::CudaError;
    // The pinned allocation is charged as Scratch.  Only payload bytes are
    // charged; allocator page/driver overhead remains resource-pool headroom.
    if (!freshHostError_) {
        auto permit = TryReserveCudaExecutionBytes(
            sizeof(int), UsdGenExecutionResourceKind::Scratch, reservation);
        int* hostError = nullptr;
        if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&hostError), sizeof(int), cudaHostAllocDefault) != cudaSuccess)
            return CurveResampleStatus::CudaError;
        freshHostError_ = hostError;
        freshHostErrorPermit_ = std::move(*permit);
    }
    const size_t outputPointCount = targetPointCount ? geometry.curveCount * size_t(targetPointCount) : geometry.pointCount;
    staging_.clear(); staging_.curveCount = geometry.curveCount; staging_.pointCount = outputPointCount;
    constexpr auto active = UsdGenExecutionResourceKind::Active;
    if (staging_.curveOffsets.reset(geometry.curveCount + 1, reservation, active) != cudaSuccess ||
        (outputPointCount && staging_.points.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (geometry.restPoints.size && outputPointCount && staging_.restPoints.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (geometry.widths.size && outputPointCount && staging_.widths.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (geometry.curveCount && staging_.stableIds.reset(geometry.curveCount, reservation, active) != cudaSuccess) ||
        (inputHairT.size && outputPointCount && staging_.hairT.reset(outputPointCount, reservation, active) != cudaSuccess) ||
        (inputRootPrim.size && geometry.curveCount && staging_.rootPrim.reset(geometry.curveCount, reservation, active) != cudaSuccess) ||
        (inputRootUV.size && geometry.curveCount && staging_.rootUV.reset(geometry.curveCount * sizeof(float2), reservation, active) != cudaSuccess)) {
        staging_.clear(); return CurveResampleStatus::CudaError;
    }
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
        return CurveResampleStatus::CudaError;
    // From the first queued operation onward a failed terminal installation
    // must not permit destruction/freeing.  FinishFreshAsync separately uses
    // freshUploadFailed_ so it can install the normal terminal callback.
    unprovenUpload_ = true;
    freshUploadFailed_ = false;
    freshCallbackArmed_ = false;
    *freshHostError_ = std::numeric_limits<int>::min();
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess) {
        freshUploadFailed_ = true;
        return CurveResampleStatus::CudaError;
    }
    ValidateInput<<<Blocks(std::max(geometry.curveCount, geometry.pointCount)), 256, 0, stream>>>(
        geometry, inputHairT, inputRootPrim, inputRootUV, error_.data());
    if (cudaGetLastError() != cudaSuccess) {
        freshUploadFailed_ = true;
        return CurveResampleStatus::CudaError;
    }
    if (geometry.curveCount) {
        Resample<<<Blocks(geometry.curveCount), 256, 0, stream>>>(geometry, inputHairT, inputRootPrim,
            inputRootUV, static_cast<uint32_t>(targetPointCount),
            {staging_.points.view(), staging_.restPoints.view(), staging_.widths.view(),
             staging_.curveOffsets.view(), staging_.stableIds.view()}, staging_.hairT.view(),
            staging_.rootPrim.view(), {reinterpret_cast<float2*>(staging_.rootUV.data()), geometry.curveCount}, error_.data());
    } else SetEmptyOffset<<<1, 1, 0, stream>>>(staging_.curveOffsets.view());
    if (cudaGetLastError() != cudaSuccess || cudaEventRecord(ready_, stream) != cudaSuccess) {
        freshUploadFailed_ = true;
        return CurveResampleStatus::CudaError;
    }
    pending_ = true;
    freshPending_ = true;
    return CurveResampleStatus::Ok;
}

CurveResampleStatus CudaCurveResample::FinishFreshAsync(cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata) {
    if (!pending_ || !freshPending_ || !callback || freshUploadFailed_ || freshCallbackArmed_)
        return CurveResampleStatus::NoPendingUpdate;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        freshUploadFailed_ = true;
        return CurveResampleStatus::InvalidArgument;
    }
    if (validateStream(stream) != cudaSuccess || !freshHostError_ ||
        cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(freshHostError_, error_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamAddCallback(stream, callback, userdata, 0) != cudaSuccess) {
        freshUploadFailed_ = true;
        return CurveResampleStatus::CudaError;
    }
    freshCallbackArmed_ = true;
    return CurveResampleStatus::Ok;
}

CurveResampleStatus CudaCurveResample::CommitFreshFinish() {
    if (!pending_ || !freshPending_ || !freshCallbackArmed_) return CurveResampleStatus::NoPendingUpdate;
    if (freshUploadFailed_ || active_.curveOffsets.size()) return CurveResampleStatus::InvalidArgument;
    freshPending_ = false;
    pending_ = false;
    freshCallbackArmed_ = false;
    freshUploadFailed_ = false;
    unprovenUpload_ = false;
    if (!freshHostError_ || *freshHostError_ != 0)
        return freshHostError_ ? finishError(*freshHostError_) : CurveResampleStatus::CudaError;
    active_.swap(staging_);
    return CurveResampleStatus::Ok;
}

DeviceCurveGeometryView CudaCurveResample::view() const noexcept { return active_.geometry(); }
DeviceView<const float> CudaCurveResample::hairT() const noexcept { return active_.hairT.view(); }
DeviceView<const int32_t> CudaCurveResample::rootPrim() const noexcept { return active_.rootPrim.view(); }
DeviceView<const float2> CudaCurveResample::rootUV() const noexcept { return active_.rootUVView(); }
cudaError_t CudaCurveResample::recordUse(cudaStream_t stream) {
    return validateStream(stream) == cudaSuccess ? active_.recordUse(stream) : cudaErrorInvalidDevice;
}
cudaError_t CudaCurveResample::waitOn(cudaStream_t stream) const {
    return validateStream(stream) == cudaSuccess ? active_.waitOn(stream) : cudaErrorInvalidDevice;
}

void CudaCurveResample::ReclassifyPublishedGeneration() noexcept {
    // The status buffers remain allocated with this owner after publication,
    // so they are retained auxiliary storage rather than operation Scratch.
    active_.ReclassifyPublishedGeneration();
    error_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    freshHostErrorPermit_.Reclassify(UsdGenExecutionResourceKind::Pinned);
}

size_t CudaCurveResample::ExclusiveRetainedBytes() const noexcept {
    size_t total = active_.RetainedBytes();
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    add(error_.bytes());
    add(freshHostErrorPermit_.Bytes());
    return total;
}

} // namespace usdGen::gpu
