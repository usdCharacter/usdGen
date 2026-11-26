#include "usdGenCurves/cuda/topology.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"
#include "usdGenCuda/gpu/deviceBuffer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace usdGen::gpu {
namespace {

__device__ void SetError(int* error) { atomicCAS(error, 0, 1); }

__global__ void InitializeResult(CurveTopologyCompareResult* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result->error = 0;
        result->equal = 1;
    }
}

__global__ void CompareKernel(DeviceCurveGeometryView a,
                              DeviceCurveGeometryView b, CurveTopologyCompareResult* result) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;

    for (size_t curve = index; curve < a.curveCount; curve += stride) {
        const uint32_t begin = a.curveOffsets.data[curve];
        const uint32_t end = a.curveOffsets.data[curve + 1];
        if (end <= begin || end > a.pointCount) SetError(&result->error);
    }
    for (size_t curve = index; curve < b.curveCount; curve += stride) {
        const uint32_t begin = b.curveOffsets.data[curve];
        const uint32_t end = b.curveOffsets.data[curve + 1];
        if (end <= begin || end > b.pointCount) SetError(&result->error);
    }

    // The first offset is checked separately so an empty curve set accepts
    // the canonical one-element offset array {0}.
    if (index == 0) {
        if (a.curveOffsets.data[0] != 0 ||
            a.curveOffsets.data[a.curveCount] != a.pointCount ||
            b.curveOffsets.data[0] != 0 ||
            b.curveOffsets.data[b.curveCount] != b.pointCount)
            SetError(&result->error);
    }

    const size_t curves = min(a.curveCount, b.curveCount);
    for (size_t curve = index; curve < curves; curve += stride) {
        if (a.stableIds.data[curve] != b.stableIds.data[curve] ||
            a.curveOffsets.data[curve] != b.curveOffsets.data[curve] ||
            a.curveOffsets.data[curve + 1] != b.curveOffsets.data[curve + 1])
            atomicExch(&result->equal, 0);
    }
    if (index == 0 && (a.curveCount != b.curveCount ||
                       a.pointCount != b.pointCount))
        atomicExch(&result->equal, 0);
}

cudaError_t CheckDevicePointer(const void* pointer, size_t count, int device) {
    if (!count) return cudaSuccess;
    if (!pointer) return cudaErrorInvalidValue;
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) return status;
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        return cudaErrorInvalidDevicePointer;
    return cudaSuccess;
}

cudaError_t Validate(DeviceCurveGeometryView const& geometry, int device) {
    if (geometry.curveCount == std::numeric_limits<size_t>::max() ||
        geometry.curveCount > std::numeric_limits<uint32_t>::max() ||
        geometry.pointCount > std::numeric_limits<uint32_t>::max() ||
        geometry.curveOffsets.size != geometry.curveCount + 1 ||
        geometry.stableIds.size != geometry.curveCount)
        return cudaErrorInvalidValue;
    cudaError_t status = CheckDevicePointer(geometry.curveOffsets.data,
                                            geometry.curveOffsets.size, device);
    if (status != cudaSuccess) return status;
    return CheckDevicePointer(geometry.stableIds.data, geometry.stableIds.size, device);
}

unsigned Blocks(size_t count) {
    const size_t needed = (count + 255) / 256;
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(needed, 65535)));
}

} // namespace

cudaError_t CompareCurveTopology(DeviceCurveGeometryView a,
                                 DeviceCurveGeometryView b,
                                 cudaStream_t stream, bool* equal,
                                 UsdGenExecutionMemoryReservation* reservation) {
    if (!equal) return cudaErrorInvalidValue;
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (stream) {
        int streamDevice = -1;
        status = cudaStreamGetDevice(stream, &streamDevice);
        if (status != cudaSuccess) return status;
        if (streamDevice != device) return cudaErrorInvalidDevice;
    }
    status = Validate(a, device);
    if (status != cudaSuccess) return status;
    status = Validate(b, device);
    if (status != cudaSuccess) return status;

    DeviceBuffer<unsigned char> resultStorage;
    status = resultStorage.reset(sizeof(CurveTopologyCompareResult), reservation,
                                 UsdGenExecutionResourceKind::Scratch);
    if (status != cudaSuccess) return status;
    CurveTopologyCompareResult* const deviceResult =
        reinterpret_cast<CurveTopologyCompareResult*>(resultStorage.data());
    bool workEnqueued = false;
    InitializeResult<<<1, 1, 0, stream>>>(deviceResult);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        workEnqueued = true;
        const size_t work = std::max(a.curveCount, b.curveCount);
        CompareKernel<<<Blocks(work), 256, 0, stream>>>(a, b, deviceResult);
        status = cudaGetLastError();
    }
    CurveTopologyCompareResult host{};
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(&host, deviceResult, sizeof(host),
                                 cudaMemcpyDeviceToHost, stream);
    cudaError_t syncStatus = cudaSuccess;
    if (workEnqueued) syncStatus = CudaFenceStream(stream);
    if (status == cudaSuccess && syncStatus != cudaSuccess) status = syncStatus;
    // Do not free or refund an allocation after an unproven stream failure.
    if (workEnqueued && syncStatus != cudaSuccess) resultStorage.quarantine();
    if (status != cudaSuccess) return status;
    if (host.error) return cudaErrorInvalidValue;
    *equal = host.equal != 0;
    return cudaSuccess;
}

CudaCurveTopologyCompare::~CudaCurveTopologyCompare() {
    if (unprovenWork_) {
        result_.quarantine(); ready_ = nullptr; hostResult_ = nullptr;
        hostResultPermit_.Abandon();
        return;
    }
    const bool owns = result_.size() || ready_ || hostResult_;
    if (!owns) return;
    int previous = -1;
    const bool gotPrevious = cudaGetDevice(&previous) == cudaSuccess;
    const bool selected = deviceIndex_ >= 0 && cudaSetDevice(deviceIndex_) == cudaSuccess;
    // Fresh work marks itself unproven before its first enqueue.  Therefore an
    // unproven object took the branch above, while a proven object was already
    // covered by the parent's native terminal callback and launcher return.
    // Do not add a destructor-side synchronization to that contract.
    if (!selected) {
        result_.quarantine(); ready_ = nullptr; hostResult_ = nullptr;
        hostResultPermit_.Abandon();
        return;
    }
    if (ready_) cudaEventDestroy(ready_);
    ready_ = nullptr; result_.release();
    if (hostResult_) {
        if (cudaFreeHost(hostResult_) == cudaSuccess) hostResultPermit_.Release();
        else hostResultPermit_.Abandon();
        hostResult_ = nullptr;
    }
    if (gotPrevious && previous != deviceIndex_) cudaSetDevice(previous);
}

cudaError_t CudaCurveTopologyCompare::BeginFresh(DeviceCurveGeometryView a,
                                                 DeviceCurveGeometryView b,
                                                 cudaStream_t stream,
                                                 UsdGenExecutionMemoryReservation* reservation) {
    if (freshRunning_ || freshStatusEnqueued_ || freshFailed_ || unprovenWork_)
        return cudaErrorInvalidValue;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return cudaErrorInvalidValue;
    int device = -1, streamDevice = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (stream && ((status = cudaStreamGetDevice(stream, &streamDevice)) != cudaSuccess ||
                   streamDevice != device))
        return status == cudaSuccess ? cudaErrorInvalidDevice : status;
    if (deviceIndex_ >= 0 && deviceIndex_ != device) return cudaErrorInvalidDevice;
    if ((status = Validate(a, device)) != cudaSuccess || (status = Validate(b, device)) != cudaSuccess)
        return status;
    if (deviceIndex_ < 0) deviceIndex_ = device;
    if (!ready_ && (status = cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming)) != cudaSuccess)
        return status;
    if ((status = result_.reset(sizeof(CurveTopologyCompareResult), reservation,
                                UsdGenExecutionResourceKind::Scratch)) != cudaSuccess)
        return status;
    if (!hostResult_) {
        auto permit = TryReserveCudaExecutionBytes(
            sizeof(CurveTopologyCompareResult),
            UsdGenExecutionResourceKind::Scratch, reservation);
        CurveTopologyCompareResult* host = nullptr;
        if (!permit) return cudaErrorMemoryAllocation;
        status = cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(*host), cudaHostAllocDefault);
        if (status != cudaSuccess) {
            // A failing allocation is permitted to return a partial host
            // pointer.  Do not let that pointer escape, and never refund its
            // resource charge unless cudaFreeHost proves it was released.
            if (host && cudaFreeHost(host) != cudaSuccess) permit->Abandon();
            return status;
        }
        hostResult_ = host;
        hostResultPermit_ = std::move(*permit);
    }
    hostResult_->error = std::numeric_limits<int>::min();
    hostResult_->equal = 0;
    unprovenWork_ = true;
    auto* const deviceResult = reinterpret_cast<CurveTopologyCompareResult*>(result_.data());
    InitializeResult<<<1, 1, 0, stream>>>(deviceResult);
    if ((status = cudaGetLastError()) != cudaSuccess) { freshFailed_ = true; return status; }
    CompareKernel<<<Blocks(std::max(a.curveCount, b.curveCount)), 256, 0, stream>>>(a, b, deviceResult);
    if ((status = cudaGetLastError()) != cudaSuccess ||
        (status = cudaEventRecord(ready_, stream)) != cudaSuccess) {
        freshFailed_ = true;
        return status;
    }
    freshRunning_ = true;
    return cudaSuccess;
}

cudaError_t CudaCurveTopologyCompare::EnqueueFreshStatus(cudaStream_t stream) {
    if (!freshRunning_ || freshStatusEnqueued_ || freshFailed_ || !hostResult_)
        return cudaErrorInvalidValue;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) { freshFailed_ = true; return cudaErrorInvalidValue; }
    int current = -1, streamDevice = -1;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess || current != deviceIndex_ ||
        (stream && ((status = cudaStreamGetDevice(stream, &streamDevice)) != cudaSuccess || streamDevice != deviceIndex_)) ||
        (status = cudaStreamWaitEvent(stream, ready_, 0)) != cudaSuccess ||
        (status = cudaMemcpyAsync(hostResult_, result_.data(), sizeof(*hostResult_), cudaMemcpyDeviceToHost, stream)) != cudaSuccess) {
        freshFailed_ = true;
        return status == cudaSuccess ? cudaErrorInvalidDevice : status;
    }
    freshStatusEnqueued_ = true;
    return cudaSuccess;
}

cudaError_t CudaCurveTopologyCompare::CommitFreshFinish(bool* equal) {
    if (!equal || !freshRunning_ || !freshStatusEnqueued_ || freshFailed_ || !hostResult_ ||
        hostResult_->error == std::numeric_limits<int>::min())
        return cudaErrorInvalidValue;
    freshRunning_ = false; freshStatusEnqueued_ = false; unprovenWork_ = false;
    if (hostResult_->error) return cudaErrorInvalidValue;
    *equal = hostResult_->equal != 0;
    return cudaSuccess;
}

} // namespace usdGen::gpu
