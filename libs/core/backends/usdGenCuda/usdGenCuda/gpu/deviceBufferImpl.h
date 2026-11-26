// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// DeviceBuffer<T> member definitions. A module includes this in one source per
// element type it instantiates; the block cache stays in usdGenCuda behind the
// detail functions, so every instantiation shares it.
#ifndef USDGEN_GPU_DEVICE_BUFFER_IMPL_H
#define USDGEN_GPU_DEVICE_BUFFER_IMPL_H

#include "deviceBuffer.h"
#include "cudaCompat.h"
#include "usdGenCuda/export.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace usdGen { namespace gpu {
namespace detail {

struct CachedBlock {
    void* ptr = nullptr;
    cudaEvent_t ready = nullptr;
    size_t bytes = 0;     // requested by the last owner (the gauge)
    size_t capacity = 0;  // the class allocated
    int device = -1;
};
USDGEN_CUDA_API size_t CacheClassBytes(size_t bytes) noexcept;
USDGEN_CUDA_API bool CachePop(int device, size_t capacity, CachedBlock* out);
USDGEN_CUDA_API void CachePush(CachedBlock const& block);
USDGEN_CUDA_API void CacheDrain(int device) noexcept;
USDGEN_CUDA_API void CacheFreeBlock(CachedBlock const& block) noexcept;

}  // namespace detail

template <class T> cudaError_t DeviceBuffer<T>::reset(
    size_t count, UsdGenExecutionMemoryReservation* reservation,
    UsdGenExecutionResourceKind kind) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return cudaErrorInvalidValue;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return cudaErrorInvalidDevice;
    if (count == size_ && data_) return device == device_ ? cudaSuccess : cudaErrorInvalidDevice;
    if (!count) { release(); return cudaSuccess; }
    size_t const bytes = count * sizeof(T);
    size_t const capacity = detail::CacheClassBytes(bytes);
    if (data_ && device == device_ && capacity == capacity_) {
        // Same class: keep the block and its event, recharge the new size.
        auto permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
        if (!permit) {
            detail::CacheDrain(device);
            permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
            if (!permit) return cudaErrorMemoryAllocation;
        }
        permit_.Release();
        permit_ = std::move(*permit);
        size_ = count;
        return cudaSuccess;
    }
    detail::CachedBlock hit;
    bool const haveHit = detail::CachePop(device, capacity, &hit);
    auto permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
    if (!permit) {
        // Admission failed: the cache may hold physical memory the budget
        // cannot admit. Shed the candidate and this device's whole cache,
        // then retry once so failure means live allocations genuinely
        // exceed the budget.
        if (haveHit) detail::CacheFreeBlock(hit);
        detail::CacheDrain(device);
        permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
        if (!permit) return cudaErrorMemoryAllocation;
    } else if (haveHit) {
        release();
        data_ = static_cast<T*>(hit.ptr); size_ = count; ready_ = hit.ready; device_ = device;
        capacity_ = hit.capacity;
        permit_ = std::move(*permit);
        return cudaSuccess;
    }
    T* next = nullptr;
    cudaEvent_t nextReady = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&next), capacity);
    if (e != cudaSuccess) return e;
    e = cudaEventCreateWithFlags(&nextReady, cudaEventDisableTiming);
    if (e != cudaSuccess) {
        if (cudaFree(next) != cudaSuccess) permit->Abandon();
        return e;
    }
    release();
    data_ = next; size_ = count; ready_ = nextReady; device_ = device; capacity_ = capacity;
    permit_ = std::move(*permit);
    return cudaSuccess;
}
template <class T> void DeviceBuffer<T>::release() noexcept {
    if (!data_) { ready_ = nullptr; size_ = 0; device_ = -1; capacity_ = 0; return; }
    int previous = -1;
    bool const selected = cudaGetDevice(&previous) == cudaSuccess &&
        device_ >= 0 && cudaSetDevice(device_) == cudaSuccess;
    // Device provenance cannot be proved: do not issue event/free calls in a
    // possibly foreign context, and retain the resource charge conservatively.
    if (!selected) {
        permit_.Abandon();
        ready_ = nullptr; data_ = nullptr; size_ = 0; device_ = -1; capacity_ = 0;
        return;
    }
    // Proven-complete storage joins the reuse cache instead of freeing: the
    // permit releases now (accounting reads as freed) while the raw block
    // waits for a reuser of its class. See the cache note above.
    detail::CachedBlock const block{data_, ready_, size_ * sizeof(T), capacity_, device_};
    int const dev = device_;
    data_ = nullptr; size_ = 0; ready_ = nullptr; device_ = -1; capacity_ = 0;
    permit_.Release();
    detail::CachePush(block);
    if (previous != dev) cudaSetDevice(previous);
}
template <class T> cudaError_t DeviceBuffer<T>::recordUse(cudaStream_t stream) {
    if (!ready_) return cudaSuccess;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != device_) return cudaErrorInvalidDevice;
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess || streamDevice != device_)
            return cudaErrorInvalidDevice;
    }
    return cudaEventRecord(ready_, stream);
}
template <class T> cudaError_t DeviceBuffer<T>::waitOn(cudaStream_t stream) const {
    if (!ready_) return cudaSuccess;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != device_) return cudaErrorInvalidDevice;
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess || streamDevice != device_)
            return cudaErrorInvalidDevice;
    }
    return cudaStreamWaitEvent(stream, ready_, 0);
}
template <class T> cudaError_t DeviceBuffer<T>::queryUse() const {
    if (!ready_) return cudaSuccess;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != device_)
        return cudaErrorInvalidDevice;
    return cudaEventQuery(ready_);
}
template <class T> cudaError_t DeviceBuffer<T>::synchronizeUse() const {
    if (!ready_) return cudaSuccess;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || device_ < 0 ||
        cudaSetDevice(device_) != cudaSuccess)
        return cudaErrorInvalidDevice;
    cudaError_t const status = cudaEventSynchronize(ready_);
    if (previous != device_) (void)cudaSetDevice(previous);
    return status;
}
template <class T> void DeviceBuffer<T>::moveFrom(DeviceBuffer& o) noexcept {
    data_ = o.data_; size_ = o.size_; ready_ = o.ready_; device_ = o.device_;
    capacity_ = o.capacity_;
    permit_ = std::move(o.permit_);
    o.data_ = nullptr; o.size_ = 0; o.ready_ = nullptr; o.device_ = -1; o.capacity_ = 0;
}
}} // namespace usdGen::gpu
#endif
