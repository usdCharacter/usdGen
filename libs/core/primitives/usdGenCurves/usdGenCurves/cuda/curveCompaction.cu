#include "usdGenCurves/cuda/curveCompaction.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <utility>

namespace usdGen::gpu {
namespace {

constexpr int kBadShape = 1;
constexpr int kNonFinite = 2;
constexpr int kBadValue = 3;
std::atomic<bool> s_failNextFreshScatterAllocation{false};

__device__ void SetError(int *error, int code) { atomicCAS(error, 0, code); }
__device__ bool Finite(float x) { return isfinite(x); }
__device__ bool Finite(float3 const &p) {
    return Finite(p.x) && Finite(p.y) && Finite(p.z);
}
__device__ bool Finite(float2 const &p) { return Finite(p.x) && Finite(p.y); }

__global__ void ValidateInput(
    DeviceCurveGeometryView geometry, DeviceView<const float> hairT,
    DeviceView<const int32_t> rootPrim, DeviceView<const float2> rootUV,
    RestRootFrames frames,
    DeviceView<const uint8_t> keep, DeviceView<uint32_t> flags,
    DeviceView<uint32_t> pointLengths, int *error) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (!geometry.curveOffsets.data || geometry.curveOffsets.size != geometry.curveCount + 1 ||
            geometry.curveOffsets.data[0] != 0 ||
            geometry.curveOffsets.data[geometry.curveCount] != geometry.pointCount)
            SetError(error, kBadShape);
    }
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t curve = i; curve < geometry.curveCount; curve += stride) {
        uint32_t begin = geometry.curveOffsets.data[curve];
        uint32_t end = geometry.curveOffsets.data[curve + 1];
        bool valid = end > begin && end <= geometry.pointCount;
        if (!valid) SetError(error, kBadShape);
        uint8_t value = keep.data[curve];
        if (value > 1u) SetError(error, kBadValue);
        flags.data[curve] = valid && value ? 1u : 0u;
        pointLengths.data[curve] = valid && value ? end - begin : 0u;
        if (rootPrim.data && (rootPrim.data[curve] < 0 || !rootUV.data ||
                              !Finite(rootUV.data[curve]) ||
                              rootUV.data[curve].x < 0 || rootUV.data[curve].x > 1 ||
                              rootUV.data[curve].y < 0 || rootUV.data[curve].y > 1))
            SetError(error, kBadValue);
        if (frames.tangent.data) {
            if (!Finite(frames.tangent.data[curve]) ||
                !Finite(frames.binormal.data[curve]) ||
                !Finite(frames.normal.data[curve]) ||
                (frames.stableIds.data &&
                 frames.stableIds.data[curve] != geometry.stableIds.data[curve]))
                SetError(error, kBadValue);
        }
        if (hairT.data && valid &&
            (hairT.data[begin] != 0.0f || hairT.data[end - 1] != 1.0f))
            SetError(error, kBadValue);
        if (hairT.data && valid)
            for (uint32_t p = begin + 1; p < end; ++p)
                if (hairT.data[p] < hairT.data[p - 1]) SetError(error, kBadValue);
    }
    for (size_t point = i; point < geometry.pointCount; point += stride) {
        if ((geometry.points.data && !Finite(geometry.points.data[point])) ||
            (geometry.restPoints.data && !Finite(geometry.restPoints.data[point])) ||
            (geometry.widths.data && !Finite(geometry.widths.data[point])))
            SetError(error, kNonFinite);
        if (geometry.widths.data && Finite(geometry.widths.data[point]) &&
            geometry.widths.data[point] < 0)
            SetError(error, kBadValue);
        if (hairT.data && (!Finite(hairT.data[point]) || hairT.data[point] < 0 || hairT.data[point] > 1))
            SetError(error, kBadValue);
    }
}

__global__ void FinalizeCounts(DeviceView<const uint32_t> flags,
                               DeviceView<const uint32_t> curvePrefix,
                               DeviceView<const uint32_t> pointLengths,
                               DeviceView<const uint32_t> pointPrefix,
                               DeviceView<uint32_t> counts) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        size_t last = flags.size - 1;
        counts.data[0] = curvePrefix.data[last] + flags.data[last];
        counts.data[1] = pointPrefix.data[last] + pointLengths.data[last];
    }
}

__global__ void ScatterCurves(
    DeviceCurveGeometryView input, DeviceView<const float> inputHairT,
    DeviceView<const int32_t> inputRootPrim, DeviceView<const float2> inputRootUV,
    DeviceView<const uint8_t> keep, DeviceView<const uint32_t> curvePrefix,
    DeviceView<const uint32_t> pointPrefix, DeviceView<uint32_t> outputOffsets,
    DeviceView<float3> outputPoints, DeviceView<float3> outputRest,
    DeviceView<float> outputWidths, DeviceView<uint64_t> outputIds,
    DeviceView<float> outputHairT, DeviceView<int32_t> outputRootPrim,
    DeviceView<float2> outputRootUV, RestRootFrames inputFrames,
    DeviceView<float3> outputT, DeviceView<float3> outputB,
    DeviceView<float3> outputN) {
    size_t firstCurve = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t curve = firstCurve; curve < input.curveCount; curve += stride) {
        if (!keep.data[curve]) continue;
        uint32_t begin = input.curveOffsets.data[curve];
        uint32_t end = input.curveOffsets.data[curve + 1];
        uint32_t outputCurve = curvePrefix.data[curve];
        uint32_t outputPoint = pointPrefix.data[curve];
        outputOffsets.data[outputCurve] = outputPoint;
        if (outputIds.data) outputIds.data[outputCurve] = input.stableIds.data[curve];
        if (outputRootPrim.data) outputRootPrim.data[outputCurve] = inputRootPrim.data[curve];
        if (outputRootUV.data) outputRootUV.data[outputCurve] = inputRootUV.data[curve];
        if (outputT.data) outputT.data[outputCurve] = inputFrames.tangent.data[curve];
        if (outputB.data) outputB.data[outputCurve] = inputFrames.binormal.data[curve];
        if (outputN.data) outputN.data[outputCurve] = inputFrames.normal.data[curve];
        for (uint32_t p = begin; p < end; ++p) {
            uint32_t q = outputPoint + p - begin;
            if (outputPoints.data) outputPoints.data[q] = input.points.data[p];
            if (outputRest.data) outputRest.data[q] = input.restPoints.data[p];
            if (outputWidths.data) outputWidths.data[q] = input.widths.data[p];
            if (outputHairT.data) outputHairT.data[q] = inputHairT.data[p];
        }
    }
}

__global__ void FinalizeOutputOffset(DeviceView<uint32_t> offsets,
                                     uint32_t outputCurves,
                                     uint32_t outputPoints) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        offsets.data[outputCurves] = outputPoints;
}

unsigned Blocks(size_t count) {
    size_t blocks = (count + 255) / 256;
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(blocks, 65535)));
}

cudaError_t CheckStream(int deviceIndex, cudaStream_t stream) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return cudaErrorUnknown;
    if (deviceIndex >= 0 && current != deviceIndex) return cudaErrorInvalidDevice;
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
            return cudaErrorUnknown;
        if (streamDevice != current || (deviceIndex >= 0 && streamDevice != deviceIndex))
            return cudaErrorInvalidDevice;
    }
    return cudaSuccess;
}

template <class T>
DeviceView<const T> ReadView(DeviceBuffer<T> const& buffer) {
    return {buffer.data(), buffer.size()};
}

bool ValidFrames(RestRootFrames frames, DeviceCurveGeometryView geometry) {
    bool const any = frames.tangent.data || frames.binormal.data || frames.normal.data ||
        frames.stableIds.data || frames.tangent.size || frames.binormal.size ||
        frames.normal.size || frames.stableIds.size;
    if (!any) return true;
    if (!frames.tangent.data || !frames.binormal.data || !frames.normal.data ||
        frames.tangent.size != geometry.curveCount ||
        frames.binormal.size != geometry.curveCount ||
        frames.normal.size != geometry.curveCount ||
        (frames.stableIds.data && frames.stableIds.size != geometry.curveCount) ||
        (frames.stableIds.size && !frames.stableIds.data) ||
        (geometry.curveCount && frames.stableIds.data && !geometry.stableIds.data) ||
        (frames.stableIds.data && geometry.stableIds.size != geometry.curveCount))
        return false;
    return geometry.curveCount != 0 ||
        (frames.tangent.size == 0 && frames.binormal.size == 0 && frames.normal.size == 0);
}

} // namespace

cudaError_t GetCudaCurveCompactionScanTemporaryBytes(
    size_t curves, size_t* bytes, cudaStream_t stream) {
    if (!bytes || curves > std::numeric_limits<uint32_t>::max())
        return cudaErrorInvalidValue;
    *bytes = 0;
    if (!curves) return cudaSuccess;
    size_t curveBytes = 0, pointBytes = 0;
    // CUB's null-storage form only reports the temporary requirement; it
    // neither dereferences these placeholders nor launches a scan.
    auto* input = static_cast<uint32_t const*>(nullptr);
    auto* output = static_cast<uint32_t*>(nullptr);
    cudaError_t status = cub::DeviceScan::ExclusiveSum(
        nullptr, curveBytes, input, output, curves, stream);
    if (status != cudaSuccess) return status;
    status = cub::DeviceScan::ExclusiveSum(nullptr, pointBytes, input, output,
                                           curves, stream);
    if (status != cudaSuccess) return status;
    *bytes = std::max(curveBytes, pointBytes);
    return cudaSuccess;
}

void CudaCurveCompaction::Storage::clear() noexcept {
    points.reset(0); restPoints.reset(0); widths.reset(0); curveOffsets.reset(0);
    stableIds.reset(0); hairT.reset(0); rootPrim.reset(0); rootUV.reset(0);
    rootT.reset(0); rootB.reset(0); rootN.reset(0);
    curveCount = pointCount = 0;
}

void CudaCurveCompaction::Storage::quarantine() noexcept {
    points.quarantine(); restPoints.quarantine(); widths.quarantine();
    curveOffsets.quarantine(); stableIds.quarantine(); hairT.quarantine();
    rootPrim.quarantine(); rootUV.quarantine();
    rootT.quarantine(); rootB.quarantine(); rootN.quarantine();
    curveCount = pointCount = 0;
}

void CudaCurveCompaction::Storage::swap(Storage& other) noexcept {
    using std::swap;
    swap(points, other.points); swap(restPoints, other.restPoints);
    swap(widths, other.widths); swap(curveOffsets, other.curveOffsets);
    swap(stableIds, other.stableIds); swap(hairT, other.hairT);
    swap(rootPrim, other.rootPrim); swap(rootUV, other.rootUV);
    swap(rootT, other.rootT); swap(rootB, other.rootB); swap(rootN, other.rootN);
    swap(curveCount, other.curveCount); swap(pointCount, other.pointCount);
}

DeviceCurveGeometryView CudaCurveCompaction::Storage::geometry() const noexcept {
    return {points.view(), restPoints.view(), widths.view(), curveOffsets.view(),
            stableIds.view(), curveCount, pointCount};
}

DeviceView<const float> CudaCurveCompaction::Storage::hairTView() const noexcept {
    return hairT.view();
}
DeviceView<const int32_t> CudaCurveCompaction::Storage::rootPrimView() const noexcept {
    return rootPrim.view();
}
DeviceView<const float2> CudaCurveCompaction::Storage::rootUVView() const noexcept {
    return {reinterpret_cast<const float2*>(rootUV.data()), rootUV.size() / sizeof(float2)};
}
RestRootFrames CudaCurveCompaction::Storage::framesView() const noexcept {
    return {rootT.view(), rootB.view(), rootN.view(),
            rootT.size() ? stableIds.view() : DeviceView<const uint64_t>{}};
}

cudaError_t CudaCurveCompaction::Storage::recordUse(cudaStream_t stream) {
    cudaError_t result = points.recordUse(stream);
    if (result == cudaSuccess) result = restPoints.recordUse(stream);
    if (result == cudaSuccess) result = widths.recordUse(stream);
    if (result == cudaSuccess) result = curveOffsets.recordUse(stream);
    if (result == cudaSuccess) result = stableIds.recordUse(stream);
    if (result == cudaSuccess) result = hairT.recordUse(stream);
    if (result == cudaSuccess) result = rootPrim.recordUse(stream);
    if (result == cudaSuccess) result = rootUV.recordUse(stream);
    if (result == cudaSuccess) result = rootT.recordUse(stream);
    if (result == cudaSuccess) result = rootB.recordUse(stream);
    if (result == cudaSuccess) result = rootN.recordUse(stream);
    return result;
}

cudaError_t CudaCurveCompaction::Storage::waitOn(cudaStream_t stream) const {
    cudaError_t result = points.waitOn(stream);
    if (result == cudaSuccess) result = restPoints.waitOn(stream);
    if (result == cudaSuccess) result = widths.waitOn(stream);
    if (result == cudaSuccess) result = curveOffsets.waitOn(stream);
    if (result == cudaSuccess) result = stableIds.waitOn(stream);
    if (result == cudaSuccess) result = hairT.waitOn(stream);
    if (result == cudaSuccess) result = rootPrim.waitOn(stream);
    if (result == cudaSuccess) result = rootUV.waitOn(stream);
    if (result == cudaSuccess) result = rootT.waitOn(stream);
    if (result == cudaSuccess) result = rootB.waitOn(stream);
    if (result == cudaSuccess) result = rootN.waitOn(stream);
    return result;
}

void CudaCurveCompaction::Storage::ReclassifyPublishedGeneration() noexcept {
    points.Reclassify(UsdGenExecutionResourceKind::Pinned);
    restPoints.Reclassify(UsdGenExecutionResourceKind::Pinned);
    widths.Reclassify(UsdGenExecutionResourceKind::Pinned);
    curveOffsets.Reclassify(UsdGenExecutionResourceKind::Pinned);
    stableIds.Reclassify(UsdGenExecutionResourceKind::Pinned);
    hairT.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootPrim.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootUV.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootT.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootB.Reclassify(UsdGenExecutionResourceKind::Pinned);
    rootN.Reclassify(UsdGenExecutionResourceKind::Pinned);
}

size_t CudaCurveCompaction::Storage::RetainedBytes() const noexcept {
    size_t total = 0;
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    add(points.bytes()); add(restPoints.bytes()); add(widths.bytes());
    add(curveOffsets.bytes()); add(stableIds.bytes()); add(hairT.bytes());
    add(rootPrim.bytes()); add(rootUV.bytes());
    add(rootT.bytes()); add(rootB.bytes()); add(rootN.bytes());
    return total;
}

CudaCurveCompaction::~CudaCurveCompaction() {
    if (unprovenWork_) {
        active_.quarantine(); pendingStorage_.quarantine(); error_.quarantine();
        curveFlags_.quarantine(); curvePrefix_.quarantine(); pointLengths_.quarantine();
        pointPrefix_.quarantine(); counts_.quarantine(); scanTemp_.quarantine();
        ready_ = nullptr; freshHostCounts_ = nullptr; freshHostCountsPermit_.Abandon();
        return;
    }
    const bool owns = active_.points.size() || active_.curveOffsets.size() ||
        pendingStorage_.points.size() || pendingStorage_.curveOffsets.size() ||
        error_.size() || ready_;
    int previous = -1;
    const bool selected = !owns ||
        (cudaGetDevice(&previous) == cudaSuccess && deviceIndex_ >= 0 &&
         cudaSetDevice(deviceIndex_) == cudaSuccess);
    const bool synchronized = !owns ||
        // shutdown wait: destructor.
        (selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess));
    if (!selected || !synchronized) {
        active_.quarantine(); pendingStorage_.quarantine();
        error_.quarantine(); curveFlags_.quarantine(); curvePrefix_.quarantine();
        pointLengths_.quarantine(); pointPrefix_.quarantine(); counts_.quarantine();
        scanTemp_.quarantine(); ready_ = nullptr;
        freshHostCounts_ = nullptr; freshHostCountsPermit_.Abandon();
        if (selected && previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
        return;
    }
    if (ready_) { cudaEventDestroy(ready_); ready_ = nullptr; }
    active_.clear(); pendingStorage_.clear(); error_.reset(0);
    curveFlags_.reset(0); curvePrefix_.reset(0); pointLengths_.reset(0);
    pointPrefix_.reset(0); counts_.reset(0); scanTemp_.reset(0);
    if (freshHostCounts_) {
        if (cudaFreeHost(freshHostCounts_) == cudaSuccess) freshHostCountsPermit_.Release();
        else freshHostCountsPermit_.Abandon();
        freshHostCounts_ = nullptr;
    }
    if (previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
}

cudaError_t CudaCurveCompaction::validateStream(cudaStream_t stream) const {
    return CheckStream(deviceIndex_, stream);
}

CurveCompactionStatus CudaCurveCompaction::finishError(int code) const {
    if (code == kNonFinite) return CurveCompactionStatus::NonFiniteInput;
    return CurveCompactionStatus::InvalidArgument;
}

CurveCompactionStatus CudaCurveCompaction::Apply(
    DeviceCurveGeometryView geometry, DeviceView<const float> inputHairT,
    DeviceView<const int32_t> inputRootPrim, DeviceView<const float2> inputRootUV,
    DeviceView<const uint8_t> keep, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation, RestRootFrames frames) {
    if (pending_ || freshPhase_ != FreshPhase::Idle || unprovenWork_) return CurveCompactionStatus::InvalidArgument;
    cudaError_t streamStatus = validateStream(stream);
    if (streamStatus != cudaSuccess) return CurveCompactionStatus::InvalidArgument;
    if (deviceIndex_ < 0 && cudaGetDevice(&deviceIndex_) != cudaSuccess)
        return CurveCompactionStatus::CudaError;
    const size_t max = std::numeric_limits<uint32_t>::max();
    if (geometry.curveCount > max || geometry.pointCount > max ||
        (geometry.curveCount == 0) != (geometry.pointCount == 0) ||
        geometry.curveOffsets.size != geometry.curveCount + 1 ||
        !geometry.curveOffsets.data ||
        (geometry.pointCount && (!geometry.points.data || geometry.points.size != geometry.pointCount)) ||
        (!geometry.pointCount && geometry.points.size != 0) ||
        (geometry.restPoints.size && (!geometry.restPoints.data || geometry.restPoints.size != geometry.pointCount)) ||
        (geometry.widths.size && (!geometry.widths.data || geometry.widths.size != geometry.pointCount)) ||
        (geometry.stableIds.size && (!geometry.stableIds.data || geometry.stableIds.size != geometry.curveCount)) ||
        (!inputHairT.data && inputHairT.size != 0) ||
        (inputHairT.data && inputHairT.size != geometry.pointCount) ||
        (!inputRootPrim.data && inputRootPrim.size != 0) ||
        (!inputRootUV.data && inputRootUV.size != 0) ||
        ((inputRootPrim.size != 0) != (inputRootUV.size != 0)) ||
        (inputRootPrim.size && inputRootPrim.size != geometry.curveCount) ||
        (inputRootUV.size && inputRootUV.size != geometry.curveCount) ||
        (!keep.data && keep.size != 0) || keep.size != geometry.curveCount ||
        !ValidFrames(frames, geometry))
        return CurveCompactionStatus::InvalidArgument;
    if (geometry.curveCount && (!geometry.points.data || !keep.data))
        return CurveCompactionStatus::InvalidArgument;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return CurveCompactionStatus::CudaError;
    auto resolveFailedSubmit = [&]() {
        // A successful terminal synchronize proves every submission made by
        // this legacy phase before its candidate storage is released.  If it
        // cannot prove completion, destruction must quarantine instead.
        if (CudaFenceStream(stream) == cudaSuccess) {
            pendingStorage_.clear();
            pending_ = false;
            unprovenWork_ = false;
        } else {
            unprovenWork_ = true;
        }
        return CurveCompactionStatus::CudaError;
    };
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        counts_.reset(2, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        curveFlags_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        curvePrefix_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        pointLengths_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        pointPrefix_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
        return CurveCompactionStatus::CudaError;
    // From the first enqueue onward, failed cleanup must never free storage
    // unless the stream has supplied a terminal completion proof.
    unprovenWork_ = true;
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return resolveFailedSubmit();
    const size_t workCount = std::max(geometry.curveCount, geometry.pointCount);
    ValidateInput<<<Blocks(workCount), 256, 0, stream>>>(
        geometry, inputHairT, inputRootPrim, inputRootUV, frames, keep,
        curveFlags_.view(), pointLengths_.view(), error_.data());
    if (cudaGetLastError() != cudaSuccess) return resolveFailedSubmit();
    int error = 0;
    if (cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess)
        return resolveFailedSubmit();
    unprovenWork_ = false;
    if (error) return finishError(error);

    uint32_t outputCurves = 0, outputPoints = 0;
    if (geometry.curveCount) {
        size_t curveTempBytes = 0, pointTempBytes = 0;
        if (cub::DeviceScan::ExclusiveSum(nullptr, curveTempBytes,
                curveFlags_.data(), curvePrefix_.data(), geometry.curveCount, stream) != cudaSuccess ||
            cub::DeviceScan::ExclusiveSum(nullptr, pointTempBytes,
                pointLengths_.data(), pointPrefix_.data(), geometry.curveCount, stream) != cudaSuccess)
            return CurveCompactionStatus::CudaError;
        if (scanTemp_.reset(std::max(curveTempBytes, pointTempBytes), reservation,
                            UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
            return CurveCompactionStatus::CudaError;
        unprovenWork_ = true;
        if (cub::DeviceScan::ExclusiveSum(scanTemp_.data(), curveTempBytes,
                curveFlags_.data(), curvePrefix_.data(), geometry.curveCount, stream) != cudaSuccess ||
            cub::DeviceScan::ExclusiveSum(scanTemp_.data(), pointTempBytes,
                pointLengths_.data(), pointPrefix_.data(), geometry.curveCount, stream) != cudaSuccess)
            return resolveFailedSubmit();
        FinalizeCounts<<<1, 1, 0, stream>>>(
            ReadView(curveFlags_), ReadView(curvePrefix_), ReadView(pointLengths_),
            ReadView(pointPrefix_), counts_.view());
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpyAsync(&outputCurves, counts_.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(&outputPoints, counts_.data() + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            CudaFenceStream(stream) != cudaSuccess)
            return resolveFailedSubmit();
        unprovenWork_ = false;
    }
    pendingStorage_.clear();
    pendingStorage_.curveCount = outputCurves;
    pendingStorage_.pointCount = outputPoints;
    if (pendingStorage_.curveOffsets.reset(size_t(outputCurves) + 1, reservation) != cudaSuccess ||
        (outputPoints && pendingStorage_.points.reset(outputPoints, reservation) != cudaSuccess) ||
        (geometry.restPoints.size && outputPoints && pendingStorage_.restPoints.reset(outputPoints, reservation) != cudaSuccess) ||
        (geometry.widths.size && outputPoints && pendingStorage_.widths.reset(outputPoints, reservation) != cudaSuccess) ||
        (geometry.stableIds.size && outputCurves && pendingStorage_.stableIds.reset(outputCurves, reservation) != cudaSuccess) ||
        (inputHairT.size && outputPoints && pendingStorage_.hairT.reset(outputPoints, reservation) != cudaSuccess) ||
        (inputRootPrim.size && outputCurves && pendingStorage_.rootPrim.reset(outputCurves, reservation) != cudaSuccess) ||
        (inputRootUV.size && outputCurves && pendingStorage_.rootUV.reset(size_t(outputCurves) * sizeof(float2), reservation) != cudaSuccess) ||
        (frames.tangent.size && outputCurves && pendingStorage_.rootT.reset(outputCurves, reservation) != cudaSuccess) ||
        (frames.binormal.size && outputCurves && pendingStorage_.rootB.reset(outputCurves, reservation) != cudaSuccess) ||
        (frames.normal.size && outputCurves && pendingStorage_.rootN.reset(outputCurves, reservation) != cudaSuccess)) {
        pendingStorage_.clear();
        return CurveCompactionStatus::CudaError;
    }
    if (outputCurves) {
        unprovenWork_ = true;
        ScatterCurves<<<Blocks(geometry.curveCount), 256, 0, stream>>>(
            geometry, inputHairT, inputRootPrim, inputRootUV, keep,
            ReadView(curvePrefix_), ReadView(pointPrefix_), pendingStorage_.curveOffsets.view(),
            pendingStorage_.points.view(), pendingStorage_.restPoints.view(),
            pendingStorage_.widths.view(), pendingStorage_.stableIds.view(),
            pendingStorage_.hairT.view(), pendingStorage_.rootPrim.view(),
            {reinterpret_cast<float2*>(pendingStorage_.rootUV.data()), outputCurves},
            frames, pendingStorage_.rootT.view(), pendingStorage_.rootB.view(),
            pendingStorage_.rootN.view());
        if (cudaGetLastError() != cudaSuccess) return resolveFailedSubmit();
    }
    unprovenWork_ = true;
    FinalizeOutputOffset<<<1, 1, 0, stream>>>(pendingStorage_.curveOffsets.view(),
                                               outputCurves, outputPoints);
    if (cudaGetLastError() != cudaSuccess) return resolveFailedSubmit();
    if (cudaEventRecord(ready_, stream) != cudaSuccess) return resolveFailedSubmit();
    unprovenWork_ = false;
    pending_ = true;
    return CurveCompactionStatus::Ok;
}

CurveCompactionStatus CudaCurveCompaction::Finish(cudaStream_t stream) {
    if (freshPhase_ != FreshPhase::Idle || unprovenWork_) return CurveCompactionStatus::InvalidArgument;
    cudaError_t streamStatus = validateStream(stream);
    if (streamStatus != cudaSuccess) return CurveCompactionStatus::InvalidArgument;
    if (!pending_) return CurveCompactionStatus::NoPendingUpdate;
    auto resolveFailedFinish = [&]() {
        // The producer event proves pending output; the caller's stream can
        // additionally contain the error readback. Both must complete before
        // this candidate is released.
        if (ready_ && cudaStreamWaitEvent(stream, ready_, 0) == cudaSuccess &&
            CudaFenceStream(stream) == cudaSuccess) {
            pendingStorage_.clear();
            pending_ = false;
            unprovenWork_ = false;
        } else {
            unprovenWork_ = true;
        }
        return CurveCompactionStatus::CudaError;
    };
    int error = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess)
        return resolveFailedFinish();
    if (error) {
        pending_ = false;
        pendingStorage_.clear();
        unprovenWork_ = false;
        return finishError(error);
    }
    if (active_.curveCount || active_.pointCount || active_.curveOffsets.size()) {
        if (active_.waitOn(stream) != cudaSuccess || CudaFenceStream(stream) != cudaSuccess)
            return resolveFailedFinish();
    }
    active_.swap(pendingStorage_);
    pendingStorage_.clear();
    pending_ = false;
    unprovenWork_ = false;
    ++generation_;
    return CurveCompactionStatus::Ok;
}

CurveCompactionStatus CudaCurveCompaction::ApplyFreshCounts(
    DeviceCurveGeometryView geometry, DeviceView<const float> hairT,
    DeviceView<const int32_t> rootPrim, DeviceView<const float2> rootUV,
    DeviceView<const uint8_t> keep, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation, RestRootFrames frames) {
    if (pending_ || freshPhase_ != FreshPhase::Idle || unprovenWork_ || active_.curveOffsets.size())
        return CurveCompactionStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return CurveCompactionStatus::InvalidArgument;
    if (validateStream(stream) != cudaSuccess) return CurveCompactionStatus::InvalidArgument;
    if (deviceIndex_ < 0 && cudaGetDevice(&deviceIndex_) != cudaSuccess) return CurveCompactionStatus::CudaError;
    const size_t max = std::numeric_limits<uint32_t>::max();
    if (geometry.curveCount > max || geometry.pointCount > max ||
        (geometry.curveCount == 0) != (geometry.pointCount == 0) ||
        !geometry.curveOffsets.data || geometry.curveOffsets.size != geometry.curveCount + 1 ||
        (geometry.pointCount && (!geometry.points.data || geometry.points.size != geometry.pointCount)) ||
        (!geometry.pointCount && geometry.points.size != 0) ||
        (geometry.restPoints.size && (!geometry.restPoints.data || geometry.restPoints.size != geometry.pointCount)) ||
        (geometry.widths.size && (!geometry.widths.data || geometry.widths.size != geometry.pointCount)) ||
        (geometry.stableIds.size && (!geometry.stableIds.data || geometry.stableIds.size != geometry.curveCount)) ||
        (!hairT.data && hairT.size != 0) || (hairT.data && hairT.size != geometry.pointCount) ||
        (!rootPrim.data && rootPrim.size != 0) || (!rootUV.data && rootUV.size != 0) ||
        ((rootPrim.size != 0) != (rootUV.size != 0)) ||
        (rootPrim.size && rootPrim.size != geometry.curveCount) ||
        (rootUV.size && rootUV.size != geometry.curveCount) ||
        (!keep.data && keep.size != 0) || keep.size != geometry.curveCount ||
        !ValidFrames(frames, geometry) ||
        (geometry.curveCount && (!geometry.points.data || !keep.data)))
        return CurveCompactionStatus::InvalidArgument;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess) return CurveCompactionStatus::CudaError;
    size_t curveTemp = 0, pointTemp = 0;
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess || counts_.reset(2, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        curveFlags_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess || curvePrefix_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        pointLengths_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess || pointPrefix_.reset(geometry.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        (geometry.curveCount && (cub::DeviceScan::ExclusiveSum(nullptr, curveTemp, curveFlags_.data(), curvePrefix_.data(), geometry.curveCount, stream) != cudaSuccess ||
                                 cub::DeviceScan::ExclusiveSum(nullptr, pointTemp, pointLengths_.data(), pointPrefix_.data(), geometry.curveCount, stream) != cudaSuccess ||
                                 scanTemp_.reset(std::max(curveTemp, pointTemp), reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess))) return CurveCompactionStatus::CudaError;
    if (!freshHostCounts_) {
        auto permit = TryReserveCudaExecutionBytes(sizeof(FreshCounts), UsdGenExecutionResourceKind::Scratch, reservation);
        FreshCounts* hostCounts = nullptr;
        if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&hostCounts), sizeof(FreshCounts), cudaHostAllocDefault) != cudaSuccess)
            return CurveCompactionStatus::CudaError;
        freshHostCounts_ = hostCounts;
        freshHostCountsPermit_ = std::move(*permit);
    }
    freshGeometry_ = geometry; freshHairT_ = hairT; freshRootPrim_ = rootPrim; freshRootUV_ = rootUV; freshKeep_ = keep; freshFrames_ = frames;
    freshHostCounts_->error = std::numeric_limits<int>::min();
    freshHostCounts_->curves = freshHostCounts_->points = UINT32_MAX;
    unprovenWork_ = true; freshPhase_ = FreshPhase::CountsPending;
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess) { freshPhase_ = FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    const size_t work = std::max(geometry.curveCount, geometry.pointCount);
    ValidateInput<<<Blocks(work),256,0,stream>>>(geometry,hairT,rootPrim,rootUV,frames,keep,curveFlags_.view(),pointLengths_.view(),error_.data());
    if (cudaGetLastError() != cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    if (geometry.curveCount) {
        if (cub::DeviceScan::ExclusiveSum(scanTemp_.data(),curveTemp,curveFlags_.data(),curvePrefix_.data(),geometry.curveCount,stream)!=cudaSuccess ||
            cub::DeviceScan::ExclusiveSum(scanTemp_.data(),pointTemp,pointLengths_.data(),pointPrefix_.data(),geometry.curveCount,stream)!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
        FinalizeCounts<<<1,1,0,stream>>>(ReadView(curveFlags_),ReadView(curvePrefix_),ReadView(pointLengths_),ReadView(pointPrefix_),counts_.view());
        if (cudaGetLastError()!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    } else if (cudaMemsetAsync(counts_.data(),0,2*sizeof(uint32_t),stream)!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    if (cudaEventRecord(ready_,stream)!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    return CurveCompactionStatus::Ok;
}

CurveCompactionStatus CudaCurveCompaction::FinishFreshCountsAsync(cudaStream_t stream,
    void (*callback)(cudaStream_t,cudaError_t,void*) noexcept, void* userdata) {
    if (freshPhase_ != FreshPhase::CountsPending || !callback || !freshHostCounts_) return CurveCompactionStatus::NoPendingUpdate;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone || validateStream(stream)!=cudaSuccess ||
        cudaStreamWaitEvent(stream,ready_,0)!=cudaSuccess ||
        cudaMemcpyAsync(&freshHostCounts_->error,error_.data(),sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess ||
        cudaMemcpyAsync(&freshHostCounts_->curves,counts_.data(),2*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)!=cudaSuccess ||
        cudaStreamAddCallback(stream,callback,userdata,0)!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    freshPhase_=FreshPhase::CountsArmed; return CurveCompactionStatus::Ok;
}

CurveCompactionStatus CudaCurveCompaction::CommitFreshCounts() {
    if (freshPhase_ != FreshPhase::CountsArmed || !freshHostCounts_ || freshHostCounts_->error==std::numeric_limits<int>::min()) return CurveCompactionStatus::NoPendingUpdate;
    unprovenWork_=false;
    if (freshHostCounts_->error) { freshPhase_=FreshPhase::Idle; return finishError(freshHostCounts_->error); }
    if (freshHostCounts_->curves > freshGeometry_.curveCount ||
        freshHostCounts_->points > freshGeometry_.pointCount) {
        freshPhase_=FreshPhase::Idle;
        return CurveCompactionStatus::InvalidArgument;
    }
    freshPhase_=FreshPhase::CountsCommitted; return CurveCompactionStatus::Ok;
}

CurveCompactionStatus CudaCurveCompaction::ApplyFreshScatter(
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation) {
    if (freshPhase_ != FreshPhase::CountsCommitted || !freshHostCounts_) return CurveCompactionStatus::InvalidArgument;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone || validateStream(stream)!=cudaSuccess) return CurveCompactionStatus::InvalidArgument;
    if (s_failNextFreshScatterAllocation.exchange(false, std::memory_order_acq_rel))
        return CurveCompactionStatus::CudaError;
    uint32_t curves=freshHostCounts_->curves, points=freshHostCounts_->points;
    pendingStorage_.clear(); pendingStorage_.curveCount=curves; pendingStorage_.pointCount=points;
    if (pendingStorage_.curveOffsets.reset(size_t(curves)+1, reservation)!=cudaSuccess ||
        (points && pendingStorage_.points.reset(points, reservation)!=cudaSuccess) ||
        (freshGeometry_.restPoints.size && points && pendingStorage_.restPoints.reset(points, reservation)!=cudaSuccess) ||
        (freshGeometry_.widths.size && points && pendingStorage_.widths.reset(points, reservation)!=cudaSuccess) ||
        (freshGeometry_.stableIds.size && curves && pendingStorage_.stableIds.reset(curves, reservation)!=cudaSuccess) ||
        (freshHairT_.size && points && pendingStorage_.hairT.reset(points, reservation)!=cudaSuccess) ||
        (freshRootPrim_.size && curves && pendingStorage_.rootPrim.reset(curves, reservation)!=cudaSuccess) ||
        (freshRootUV_.size && curves && pendingStorage_.rootUV.reset(size_t(curves)*sizeof(float2), reservation)!=cudaSuccess) ||
        (freshFrames_.tangent.size && curves && pendingStorage_.rootT.reset(curves, reservation)!=cudaSuccess) ||
        (freshFrames_.binormal.size && curves && pendingStorage_.rootB.reset(curves, reservation)!=cudaSuccess) ||
        (freshFrames_.normal.size && curves && pendingStorage_.rootN.reset(curves, reservation)!=cudaSuccess)) { pendingStorage_.clear(); return CurveCompactionStatus::CudaError; }
    unprovenWork_=true; freshPhase_=FreshPhase::ScatterPending;
    if (curves) ScatterCurves<<<Blocks(freshGeometry_.curveCount),256,0,stream>>>(freshGeometry_,freshHairT_,freshRootPrim_,freshRootUV_,freshKeep_,ReadView(curvePrefix_),ReadView(pointPrefix_),pendingStorage_.curveOffsets.view(),pendingStorage_.points.view(),pendingStorage_.restPoints.view(),pendingStorage_.widths.view(),pendingStorage_.stableIds.view(),pendingStorage_.hairT.view(),pendingStorage_.rootPrim.view(),{reinterpret_cast<float2*>(pendingStorage_.rootUV.data()),curves},freshFrames_,pendingStorage_.rootT.view(),pendingStorage_.rootB.view(),pendingStorage_.rootN.view());
    if (cudaGetLastError()!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    FinalizeOutputOffset<<<1,1,0,stream>>>(pendingStorage_.curveOffsets.view(),curves,points);
    if (cudaGetLastError()!=cudaSuccess || cudaEventRecord(ready_,stream)!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    return CurveCompactionStatus::Ok;
}

void failNextCudaCurveCompactionFreshScatterAllocationForTesting() {
    s_failNextFreshScatterAllocation.store(true, std::memory_order_release);
}

CurveCompactionStatus CudaCurveCompaction::FinishFreshScatterAsync(cudaStream_t stream,
    void (*callback)(cudaStream_t,cudaError_t,void*) noexcept, void* userdata) {
    if (freshPhase_ != FreshPhase::ScatterPending || !callback)
        return CurveCompactionStatus::NoPendingUpdate;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone ||
        validateStream(stream)!=cudaSuccess || cudaStreamWaitEvent(stream,ready_,0)!=cudaSuccess ||
        cudaStreamAddCallback(stream,callback,userdata,0)!=cudaSuccess) { freshPhase_=FreshPhase::Failed; return CurveCompactionStatus::CudaError; }
    freshPhase_=FreshPhase::ScatterArmed; return CurveCompactionStatus::Ok;
}

CurveCompactionStatus CudaCurveCompaction::CommitFreshFinish() {
    if (freshPhase_!=FreshPhase::ScatterArmed) return CurveCompactionStatus::NoPendingUpdate;
    active_.swap(pendingStorage_); ++generation_;
    unprovenWork_=false; freshPhase_=FreshPhase::Idle; return CurveCompactionStatus::Ok;
}

DeviceCurveGeometryView CudaCurveCompaction::view() const noexcept { return active_.geometry(); }
DeviceView<const float> CudaCurveCompaction::hairT() const noexcept { return active_.hairTView(); }
DeviceView<const int32_t> CudaCurveCompaction::rootPrim() const noexcept { return active_.rootPrimView(); }
DeviceView<const float2> CudaCurveCompaction::rootUV() const noexcept { return active_.rootUVView(); }
RestRootFrames CudaCurveCompaction::frames() const noexcept { return active_.framesView(); }

cudaError_t CudaCurveCompaction::recordUse(cudaStream_t stream) {
    if (validateStream(stream) != cudaSuccess) return cudaErrorInvalidDevice;
    return active_.recordUse(stream);
}
cudaError_t CudaCurveCompaction::waitOn(cudaStream_t stream) const {
    if (validateStream(stream) != cudaSuccess) return cudaErrorInvalidDevice;
    return active_.waitOn(stream);
}

void CudaCurveCompaction::ReclassifyPublishedGeneration() noexcept {
    // The scratch buffers and pinned count readback are retained by this
    // owner after commit.  Reclassify them rather than leaving long-lived
    // charge in Scratch; only pendingStorage_ is intentionally excluded.
    active_.ReclassifyPublishedGeneration();
    error_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    curveFlags_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    curvePrefix_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    pointLengths_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    pointPrefix_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    counts_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    scanTemp_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    freshHostCountsPermit_.Reclassify(UsdGenExecutionResourceKind::Pinned);
}

size_t CudaCurveCompaction::ExclusiveRetainedBytes() const noexcept {
    size_t total = active_.RetainedBytes();
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    add(error_.bytes()); add(curveFlags_.bytes()); add(curvePrefix_.bytes());
    add(pointLengths_.bytes()); add(pointPrefix_.bytes()); add(counts_.bytes());
    add(scanTemp_.bytes()); add(freshHostCountsPermit_.Bytes());
    return total;
}

} // namespace usdGen::gpu
