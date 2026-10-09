#include "deviceBuffer.h"
#include "cudaCompat.h"
#include "curveTileBounds.h"
#include "curveTiles.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace usdGen { namespace gpu {
namespace {
// Process-wide cache of freed device blocks, keyed by exact (device, byte
// size). Flows such as CudaScatterGrow::BeginFresh allocate ~19 buffers and
// free them per operation while a cold cudaMalloc costs ~1ms (~17ms per
// grow), so proven-complete blocks are retained and reissued on exact-size
// match, eliding the malloc/free/event round-trip.
//
// Accounting: the cached block's permit is RELEASED on insert, so pool
// usedBytes/byKind read exactly as if the storage had been freed —
// teardown baselines and tight-budget admission are unchanged. Reuse
// re-reserves through the normal TryReserve path (charging the reusing
// job and consuming its reservation), and a failed reserve drains this
// device's cache and retries once, so failure still means live
// allocations genuinely exceed the budget rather than cache pressure.
// Idle physical retention is bounded by the cap below.
//
// Safety: release() already requires proven completion (unproven work
// quarantines instead of freeing), so a cached block is exactly as
// reusable as a freed address is remallocable. The cached event is
// re-recorded by the reuser's recordUse before any wait, as with a
// fresh event. cudaMalloc's 256-byte alignment covers every cached T.
struct CachedBlock {
    void* ptr = nullptr;
    cudaEvent_t ready = nullptr;
    size_t bytes = 0;
    int device = -1;
};
// Deliberately leaked: entries must never run cudaFree during process
// teardown (the CUDA context may already be gone); the driver reclaims
// device memory with the context.
std::mutex& CacheMutex() {
    static auto* m = new std::mutex;
    return *m;
}
std::vector<CachedBlock>& Cache() {
    static auto* c = new std::vector<CachedBlock>;
    return *c;
}
size_t& CacheBytes() {
    static auto* b = new size_t(0);
    return *b;
}
// Idle physical retention bound. Covers one scatter-grow peak (~380MB)
// with headroom; drain-on-reserve-failure keeps tight budgets correct
// regardless of this value.
constexpr size_t kCacheCapBytes = size_t{1} << 30;

void FreeBlock(CachedBlock const& b) noexcept {
    if (b.ready) cudaEventDestroy(b.ready);
    if (b.ptr) cudaFree(b.ptr);
}

// Exact (device, bytes) match, most-recently-freed first. On hit *out
// takes the entry out of the cache.
bool PopCache(int device, size_t bytes, CachedBlock* out) {
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& c = Cache();
    for (size_t i = c.size(); i-- > 0;) {
        if (c[i].device == device && c[i].bytes == bytes) {
            *out = c[i];
            c.erase(c.begin() + std::ptrdiff_t(i));
            CacheBytes() -= bytes;
            return true;
        }
    }
    return false;
}

// Insert a freed block, evicting oldest-first while over the cap. The
// caller has `device` selected: same-device evictions issue directly,
// foreign ones save/set/restore around the free. A failed set keeps the
// entry and stops eviction (soft cap) rather than leaking or spinning.
void PushCache(int device, size_t bytes, void* ptr, cudaEvent_t ready) {
    {
        std::lock_guard<std::mutex> lock(CacheMutex());
        Cache().push_back(CachedBlock{ptr, ready, bytes, device});
        CacheBytes() += bytes;
    }
    for (;;) {
        CachedBlock victim;
        bool sameDevice = true;
        {
            std::lock_guard<std::mutex> lock(CacheMutex());
            auto& c = Cache();
            if (CacheBytes() <= kCacheCapBytes || c.empty()) return;
            size_t pick = c.size();
            for (size_t i = 0; i < c.size(); ++i) {
                if (c[i].device == device) { pick = i; break; }
            }
            if (pick == c.size()) { pick = 0; sameDevice = false; }
            victim = c[pick];
            c.erase(c.begin() + std::ptrdiff_t(pick));
            CacheBytes() -= victim.bytes;
        }
        if (!sameDevice && cudaSetDevice(victim.device) != cudaSuccess) {
            std::lock_guard<std::mutex> lock(CacheMutex());
            Cache().insert(Cache().begin(), victim);
            CacheBytes() += victim.bytes;
            return;
        }
        FreeBlock(victim);
        if (!sameDevice) cudaSetDevice(device);
    }
}

// Free every cached block for `device`, already selected by the caller.
// Best-effort: a backend failure just drops the handles.
void DrainCacheForDevice(int device) noexcept {
    std::vector<CachedBlock> doomed;
    {
        std::lock_guard<std::mutex> lock(CacheMutex());
        auto& c = Cache();
        for (size_t i = 0; i < c.size();) {
            if (c[i].device == device) {
                doomed.push_back(c[i]);
                CacheBytes() -= c[i].bytes;
                c.erase(c.begin() + std::ptrdiff_t(i));
            } else {
                ++i;
            }
        }
    }
    for (auto const& b : doomed) FreeBlock(b);
}
}  // namespace

size_t UsdGenGpuDeviceBufferCacheBytes() noexcept {
    std::lock_guard<std::mutex> lock(CacheMutex());
    return CacheBytes();
}

template <class T> cudaError_t DeviceBuffer<T>::reset(
    size_t count, UsdGenExecutionMemoryReservation* reservation,
    UsdGenExecutionResourceKind kind) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return cudaErrorInvalidValue;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return cudaErrorInvalidDevice;
    if (count == size_ && data_) return device == device_ ? cudaSuccess : cudaErrorInvalidDevice;
    if (!count) { release(); return cudaSuccess; }
    size_t const bytes = count * sizeof(T);
    CachedBlock hit;
    bool const haveHit = PopCache(device, bytes, &hit);
    auto permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
    if (!permit) {
        // Admission failed: the cache may hold physical memory the budget
        // cannot admit. Shed the candidate and this device's whole cache,
        // then retry once so failure means live allocations genuinely
        // exceed the budget.
        if (haveHit) FreeBlock(hit);
        DrainCacheForDevice(device);
        permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
        if (!permit) return cudaErrorMemoryAllocation;
    } else if (haveHit) {
        release();
        data_ = static_cast<T*>(hit.ptr); size_ = count; ready_ = hit.ready; device_ = device;
        permit_ = std::move(*permit);
        return cudaSuccess;
    }
    T* next = nullptr;
    cudaEvent_t nextReady = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&next), bytes);
    if (e != cudaSuccess) return e;
    e = cudaEventCreateWithFlags(&nextReady, cudaEventDisableTiming);
    if (e != cudaSuccess) {
        if (cudaFree(next) != cudaSuccess) permit->Abandon();
        return e;
    }
    release();
    data_ = next; size_ = count; ready_ = nextReady; device_ = device;
    permit_ = std::move(*permit);
    return cudaSuccess;
}
template <class T> void DeviceBuffer<T>::release() noexcept {
    if (!data_) { ready_ = nullptr; size_ = 0; device_ = -1; return; }
    int previous = -1;
    bool const selected = cudaGetDevice(&previous) == cudaSuccess &&
        device_ >= 0 && cudaSetDevice(device_) == cudaSuccess;
    // Device provenance cannot be proved: do not issue event/free calls in a
    // possibly foreign context, and retain the resource charge conservatively.
    if (!selected) {
        permit_.Abandon();
        ready_ = nullptr; data_ = nullptr; size_ = 0; device_ = -1;
        return;
    }
    // Proven-complete storage joins the reuse cache instead of freeing: the
    // permit releases now (accounting reads as freed) while the raw block
    // waits for an exact-size reuser. See the cache note above.
    T* const ptr = data_;
    cudaEvent_t const ready = ready_;
    size_t const bytes = size_ * sizeof(T);
    int const dev = device_;
    data_ = nullptr; size_ = 0; ready_ = nullptr; device_ = -1;
    permit_.Release();
    PushCache(dev, bytes, ptr, ready);
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
    permit_ = std::move(o.permit_);
    o.data_ = nullptr; o.size_ = 0; o.ready_ = nullptr; o.device_ = -1;
}
template class DeviceBuffer<float>;
template class DeviceBuffer<double>;
template class DeviceBuffer<int>;
template class DeviceBuffer<unsigned char>;
template class DeviceBuffer<uint32_t>;
template class DeviceBuffer<uint64_t>;
template class DeviceBuffer<float3>;
template class DeviceBuffer<float2>;
template class DeviceBuffer<CurveTileBoundsScratch>;
template class DeviceBuffer<CurveTileSpan>;
}}
