#include "usdGenCurves/cuda/pointOverride.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <limits>
#include <utility>

namespace usdGen::gpu {
namespace {

constexpr int kBadIndex = 1;
constexpr int kNonFinite = 2;
constexpr int kDuplicate = 3;

__device__ void SetError(int *error, int code) { atomicCAS(error, 0, code); }

__device__ bool Finite(float3 const& p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}

__global__ void ValidateInput(DeviceView<const float3> base,
                              DeviceView<const int32_t> indices,
                              DeviceView<const float3> replacements,
                              int* error) {
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = first; i < base.size; i += stride)
        if (!Finite(base.data[i])) SetError(error, kNonFinite);
    for (size_t i = first; i < indices.size; i += stride) {
        const int32_t index = indices.data[i];
        if (index < 0 || size_t(index) >= base.size)
            SetError(error, kBadIndex);
        if (!Finite(replacements.data[i])) SetError(error, kNonFinite);
    }
}

__global__ void CheckDuplicateIndices(DeviceView<const int32_t> indices,
                                       int* error) {
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = first; i < indices.size; i += stride)
        if (i && indices.data[i] == indices.data[i - 1])
            SetError(error, kDuplicate);
}

__global__ void InitializeSlots(DeviceView<uint32_t> slots) {
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = first; i < slots.size; i += stride)
        slots.data[i] = static_cast<uint32_t>(i);
}

__global__ void ApplyOverrides(DeviceView<const int32_t> sortedIndices,
                               DeviceView<const uint32_t> sortedSlots,
                               DeviceView<const float3> replacements,
                               DeviceView<float3> output) {
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = first; i < sortedIndices.size; i += stride)
        output.data[sortedIndices.data[i]] = replacements.data[sortedSlots.data[i]];
}

unsigned Blocks(size_t count) {
    const size_t blocks = (count + 255) / 256;
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(blocks, 65535)));
}

cudaError_t CheckStream(int deviceIndex, cudaStream_t stream) {
    int current = -1;
    cudaError_t result = cudaGetDevice(&current);
    if (result != cudaSuccess) return result;
    if (deviceIndex >= 0 && current != deviceIndex) return cudaErrorInvalidDevice;
    if (!stream) return cudaSuccess;
    int streamDevice = -1;
    result = cudaStreamGetDevice(stream, &streamDevice);
    if (result != cudaSuccess) return result;
    return streamDevice == current && (deviceIndex < 0 || streamDevice == deviceIndex)
        ? cudaSuccess : cudaErrorInvalidDevice;
}

cudaError_t CheckDevicePointer(const void* pointer, int device) {
    if (!pointer) return cudaErrorInvalidValue;
    cudaPointerAttributes attributes{};
    const cudaError_t result = cudaPointerGetAttributes(&attributes, pointer);
    if (result != cudaSuccess) return result;
#if CUDART_VERSION >= 10000
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        return cudaErrorInvalidDevice;
#else
    if (attributes.memoryType != cudaMemoryTypeDevice || attributes.device != device)
        return cudaErrorInvalidDevice;
#endif
    return cudaSuccess;
}

PointOverrideStatus PointerStatus(cudaError_t error) {
    return error == cudaErrorInvalidValue || error == cudaErrorInvalidDevice
        ? PointOverrideStatus::InvalidArgument : PointOverrideStatus::CudaError;
}

PointOverrideStatus DeviceErrorStatus(int error) {
    if (error == kNonFinite) return PointOverrideStatus::NonFiniteInput;
    if (error == kDuplicate) return PointOverrideStatus::DuplicateIndex;
    return PointOverrideStatus::InvalidArgument;
}

} // namespace

CudaPointOverride::~CudaPointOverride() {
    const bool owns = active_.size() || pendingStorage_.size() || error_.size() ||
        inputSlots_.size() ||
        sortedIndices_.size() || sortedSlots_.size() || sortTemp_.size() || ready_;
    int previous = -1;
    const bool selected = !owns ||
        (cudaGetDevice(&previous) == cudaSuccess && deviceIndex_ >= 0 &&
         cudaSetDevice(deviceIndex_) == cudaSuccess);
    // DeviceBuffer's event is also used by consumers, so synchronizing only
    // the publication event would permit cudaFree while a consumer stream is
    // still reading the active generation.  Destruction is rare; synchronize
    // the owning device to cover both producer and consumer events.
    // shutdown wait: destructor.
    const bool synchronized = !owns || (selected && cudaDeviceSynchronize() == cudaSuccess);
    if (!selected || !synchronized) {
        active_.quarantine(); pendingStorage_.quarantine(); error_.quarantine();
        inputSlots_.quarantine();
        sortedIndices_.quarantine(); sortedSlots_.quarantine(); sortTemp_.quarantine();
        ready_ = nullptr;
        if (selected && previous >= 0 && previous != deviceIndex_)
            cudaSetDevice(previous);
        return;
    }
    if (ready_) { cudaEventDestroy(ready_); ready_ = nullptr; }
    active_.release(); pendingStorage_.release(); error_.release(); inputSlots_.release();
    sortedIndices_.release(); sortedSlots_.release(); sortTemp_.release();
    if (previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
}

cudaError_t CudaPointOverride::validateStream(cudaStream_t stream) const {
    return CheckStream(deviceIndex_, stream);
}

PointOverrideStatus CudaPointOverride::Apply(
    DeviceView<const float3> basePoints, DeviceView<const int32_t> indices,
    DeviceView<const float3> absolutePositions, cudaStream_t stream) {
    if (pending_ || indices.size != absolutePositions.size)
        return PointOverrideStatus::InvalidArgument;
    const cudaError_t streamError = validateStream(stream);
    if (streamError != cudaSuccess) return PointerStatus(streamError);
    if (basePoints.size > size_t(std::numeric_limits<int32_t>::max()) ||
        indices.size > size_t(std::numeric_limits<int>::max()) ||
        indices.size > basePoints.size)
        return PointOverrideStatus::InvalidArgument;
    if ((!basePoints.data && basePoints.size) ||
        (!indices.data && indices.size) ||
        (!absolutePositions.data && absolutePositions.size))
        return PointOverrideStatus::InvalidArgument;
    if (deviceIndex_ < 0) {
        if (cudaGetDevice(&deviceIndex_) != cudaSuccess)
            return PointOverrideStatus::CudaError;
    }
    // Zero-sized views do not need a meaningful pointer.  Every non-empty
    // view must be a device allocation on the bound device.
    if (basePoints.size) {
        const cudaError_t e = CheckDevicePointer(basePoints.data, deviceIndex_);
        if (e != cudaSuccess) return PointerStatus(e);
    }
    if (indices.size) {
        const cudaError_t e = CheckDevicePointer(indices.data, deviceIndex_);
        if (e != cudaSuccess) return PointerStatus(e);
    }
    if (absolutePositions.size) {
        const cudaError_t e = CheckDevicePointer(absolutePositions.data, deviceIndex_);
        if (e != cudaSuccess) return PointerStatus(e);
    }
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return PointOverrideStatus::CudaError;
    if (error_.reset(1) != cudaSuccess ||
        inputSlots_.reset(indices.size) != cudaSuccess ||
        sortedIndices_.reset(indices.size) != cudaSuccess ||
        sortedSlots_.reset(indices.size) != cudaSuccess ||
        cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return PointOverrideStatus::CudaError;
    const size_t work = std::max(basePoints.size, indices.size);
    ValidateInput<<<Blocks(work), 256, 0, stream>>>(
        basePoints, indices, absolutePositions, error_.data());
    if (cudaGetLastError() != cudaSuccess)
        return PointOverrideStatus::CudaError;
    int error = 0;
    if (cudaMemcpyAsync(&error, error_.data(), sizeof(error),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess)
        return PointOverrideStatus::CudaError;
    if (error) return DeviceErrorStatus(error);

    if (indices.size) {
        InitializeSlots<<<Blocks(indices.size), 256, 0, stream>>>(inputSlots_.view());
        if (cudaGetLastError() != cudaSuccess)
            return PointOverrideStatus::CudaError;
        size_t tempBytes = 0;
        if (cub::DeviceRadixSort::SortPairs(
                nullptr, tempBytes, indices.data, sortedIndices_.data(),
                inputSlots_.data(), sortedSlots_.data(), int(indices.size),
                0, sizeof(int32_t) * 8, stream) != cudaSuccess)
            return PointOverrideStatus::CudaError;
        if (sortTemp_.reset(tempBytes) != cudaSuccess ||
            cub::DeviceRadixSort::SortPairs(
                sortTemp_.data(), tempBytes, indices.data, sortedIndices_.data(),
                inputSlots_.data(), sortedSlots_.data(), int(indices.size),
                0, sizeof(int32_t) * 8, stream) != cudaSuccess)
            return PointOverrideStatus::CudaError;
        CheckDuplicateIndices<<<Blocks(indices.size), 256, 0, stream>>>(
            {sortedIndices_.data(), sortedIndices_.size()}, error_.data());
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpyAsync(&error, error_.data(), sizeof(error),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            CudaFenceStream(stream) != cudaSuccess)
            return PointOverrideStatus::CudaError;
        if (error) return DeviceErrorStatus(error);
    }

    if (pendingStorage_.reset(basePoints.size) != cudaSuccess)
        return PointOverrideStatus::CudaError;
    if (basePoints.size && cudaMemcpyAsync(
            pendingStorage_.data(), basePoints.data,
            basePoints.size * sizeof(float3), cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        { pendingStorage_.quarantine(); return PointOverrideStatus::CudaError; }
    if (indices.size) {
        ApplyOverrides<<<Blocks(indices.size), 256, 0, stream>>>(
            {sortedIndices_.data(), sortedIndices_.size()},
            {sortedSlots_.data(), sortedSlots_.size()}, absolutePositions,
            pendingStorage_.view());
        if (cudaGetLastError() != cudaSuccess)
            { pendingStorage_.quarantine(); return PointOverrideStatus::CudaError; }
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess)
        { pendingStorage_.quarantine(); return PointOverrideStatus::CudaError; }
    pending_ = true;
    return PointOverrideStatus::Ok;
}

PointOverrideStatus CudaPointOverride::Finish(cudaStream_t stream) {
    const cudaError_t streamError = validateStream(stream);
    if (streamError != cudaSuccess) return PointerStatus(streamError);
    if (!pending_) return PointOverrideStatus::NoPendingUpdate;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess)
        return PointOverrideStatus::CudaError;
    if (active_.waitOn(stream) != cudaSuccess || CudaFenceStream(stream) != cudaSuccess)
        return PointOverrideStatus::CudaError;
    using std::swap;
    swap(active_, pendingStorage_);
    pendingStorage_.release();
    pending_ = false;
    completed_ = true;
    ++generation_;
    return PointOverrideStatus::Ok;
}

DeviceView<const float3> CudaPointOverride::view() const noexcept {
    return active_.view();
}

std::unique_ptr<DeviceBuffer<float3>> CudaPointOverride::releaseCompleted() {
    if (pending_ || !completed_) return {};
    completed_ = false;
    return std::unique_ptr<DeviceBuffer<float3>>(
        new DeviceBuffer<float3>(std::move(active_)));
}

cudaError_t CudaPointOverride::recordUse(cudaStream_t stream) {
    if (validateStream(stream) != cudaSuccess) return cudaErrorInvalidDevice;
    return active_.recordUse(stream);
}

cudaError_t CudaPointOverride::waitOn(cudaStream_t stream) const {
    if (validateStream(stream) != cudaSuccess) return cudaErrorInvalidDevice;
    return active_.waitOn(stream);
}

} // namespace usdGen::gpu
