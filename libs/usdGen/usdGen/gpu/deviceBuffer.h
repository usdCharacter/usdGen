// Small ownership primitives for buffers which never acquire a host mirror.
#ifndef USDGEN_GPU_DEVICE_BUFFER_H
#define USDGEN_GPU_DEVICE_BUFFER_H

#include <cuda_runtime.h>
#include "deviceResources.h"

#include <cstddef>
#include <limits>
#include <utility>

namespace usdGen { namespace gpu {

template <class T> struct DeviceView {
    T* data = nullptr;
    size_t size = 0;
    constexpr explicit operator bool() const { return data != nullptr || size == 0; }
};

// Diagnostic gauge for tests: idle device bytes currently retained by the
// DeviceBuffer reuse cache process-wide. Cached storage is uncharged (its
// permit released at insert), so pool accounting reads as if freed; this
// gauge is the only window into the retained physical bytes.
size_t UsdGenGpuDeviceBufferCacheBytes() noexcept;

// A move-only allocation.  The event is deliberately owned by the allocation:
// a published generation records its last producer use, and a consumer can put
// a wait on its own stream before dereferencing view().
template <class T> class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { moveFrom(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) { release(); moveFrom(other); }
        return *this;
    }

    // A non-null reservation is consumed only when this call must replace an
    // allocation; same-size reuse retains its existing permit unchanged.
    cudaError_t reset(size_t count,
                      UsdGenExecutionMemoryReservation* reservation = nullptr,
                      UsdGenExecutionResourceKind kind =
                          UsdGenExecutionResourceKind::Active);
    // Host-only accounting transition for storage which has become part of an
    // immutable published generation.  It deliberately does not touch the
    // CUDA allocation, event, or device selection.
    void Reclassify(UsdGenExecutionResourceKind kind) noexcept {
        permit_.Reclassify(kind);
    }
    void release() noexcept;
    // Only for a lost device or unprovable completion: abandon handles rather
    // than free storage still potentially in use. Context teardown reclaims
    // these intentionally quarantined allocations/events.
    void quarantine() noexcept { permit_.Abandon(); data_ = nullptr; size_ = 0; ready_ = nullptr; device_ = -1; }
    DeviceView<T> view() { return {data_, size_}; }
    DeviceView<const T> view() const { return {data_, size_}; }
    size_t size() const { return size_; }
    size_t bytes() const noexcept { return size_ * sizeof(T); }
    T* data() { return data_; }
    const T* data() const { return data_; }
    cudaError_t recordUse(cudaStream_t stream);
    cudaError_t waitOn(cudaStream_t stream) const;
    // Poll the last-use event without synchronizing the host.  A caller may
    // reclaim the allocation only after this reports cudaSuccess.
    cudaError_t queryUse() const;
    // Establish a final completion proof for destruction of paired host
    // sources. Selects the allocation's device for the event wait.
    cudaError_t synchronizeUse() const;

private:
    void moveFrom(DeviceBuffer& other) noexcept;
    T* data_ = nullptr;
    size_t size_ = 0;
    cudaEvent_t ready_ = nullptr;
    int device_ = -1;
    UsdGenExecutionResourcePermit permit_;
};

}} // namespace usdGen::gpu
#endif
