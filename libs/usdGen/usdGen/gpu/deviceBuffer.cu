#include "deviceBuffer.h"
#include "cudaCompat.h"
#include "curveTileBounds.h"
#include "curveTiles.h"
#include <cstddef>
#include <cstdint>
#include <atomic>

namespace usdGen { namespace gpu {
namespace {
// Process-wide size-class pool of freed device blocks, keyed by (device,
// class). Classes are 256 bytes, then four per power of two, so a block
// serves any request within 25% below its class: a curve or CV count that
// moves within its class reuses storage, and only a topology change that
// crosses a class allocates. Flows such as CudaScatterGrow::BeginFresh
// allocate ~19 buffers and free them per operation while a cold cudaMalloc
// costs ~1ms (~17ms per grow), so proven-complete blocks are retained and
// reissued, eliding the malloc/free/event round-trip.
//
// Accounting: permits and the cache gauge charge the bytes requested, not
// the class. The cached block's permit is RELEASED on insert, so pool
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
    size_t bytes = 0;     // requested by the last owner (the gauge)
    size_t capacity = 0;  // the class allocated
    int device = -1;
};
// Lock-free and allocation-free: blocks hang off fixed nodes in per-(device,
// bytes) buckets. A bucket is a Treiber stack whose head packs a 32-bit ABA
// tag with a 1-based node index; bucket keys are CAS-inserted and never
// removed, so the table only grows with new sizes (topology changes). A full
// table or node array frees the block instead of caching it.
// Deliberately leaked: entries must never run cudaFree during process
// teardown (the CUDA context may already be gone); the driver reclaims
// device memory with the context. All state is constant-initialized.
constexpr uint32_t kCacheNodes = 8192;
constexpr uint32_t kCacheBuckets = 1024;  // power of two
struct CacheNode {
    void* ptr = nullptr;
    cudaEvent_t ready = nullptr;
    size_t bytes = 0;
    std::atomic<uint32_t> next{0};
};
struct CacheBucket {
    std::atomic<uint64_t> key{0};   // 0 = empty
    std::atomic<uint64_t> head{0};  // tag << 32 | node index + 1
};
CacheNode g_nodes[kCacheNodes];
std::atomic<uint32_t> g_nodesUsed{0};
std::atomic<uint64_t> g_freeNodes{0};
CacheBucket g_buckets[kCacheBuckets];
std::atomic<size_t> g_cacheBytes{0};     // requested bytes (gauge)
std::atomic<size_t> g_cachePhysical{0};  // class bytes (cap)
// Idle physical retention bound. Covers one scatter-grow peak (~380MB)
// with headroom; drain-on-reserve-failure keeps tight budgets correct
// regardless of this value.
constexpr size_t kCacheCapBytes = size_t{1} << 30;

// 256 bytes, then four classes per power of two (at most 25% slack).
size_t ClassBytes(size_t bytes) noexcept {
    if (bytes <= 256) return 256;
    unsigned top = 0;
    for (size_t v = bytes - 1; v >>= 1;) ++top;  // floor(log2(bytes - 1))
    size_t const step = size_t{1} << (top - 2);
    size_t const rounded = (bytes + step - 1) & ~(step - 1);
    return rounded < bytes ? bytes : rounded;
}

uint64_t Pack(uint64_t head, uint32_t index) noexcept {
    return (((head >> 32) + 1) << 32) | index;
}
uint32_t PopIndex(std::atomic<uint64_t>& head) noexcept {
    uint64_t h = head.load(std::memory_order_acquire);
    for (;;) {
        uint32_t const i = uint32_t(h);
        if (!i) return 0;
        uint32_t const next = g_nodes[i - 1].next.load(std::memory_order_relaxed);
        if (head.compare_exchange_weak(h, Pack(h, next), std::memory_order_acq_rel,
                                       std::memory_order_acquire))
            return i;
    }
}
void PushIndex(std::atomic<uint64_t>& head, uint32_t i) noexcept {
    uint64_t h = head.load(std::memory_order_relaxed);
    do {
        g_nodes[i - 1].next.store(uint32_t(h), std::memory_order_relaxed);
    } while (!head.compare_exchange_weak(h, Pack(h, i), std::memory_order_release,
                                         std::memory_order_relaxed));
}
uint32_t AllocNode() noexcept {
    if (uint32_t i = PopIndex(g_freeNodes)) return i;
    uint32_t used = g_nodesUsed.load(std::memory_order_relaxed);
    while (used < kCacheNodes) {
        if (g_nodesUsed.compare_exchange_weak(used, used + 1, std::memory_order_relaxed))
            return used + 1;
    }
    return 0;
}
void FreeNode(uint32_t i) noexcept {
    g_nodes[i - 1].ptr = nullptr; g_nodes[i - 1].ready = nullptr; g_nodes[i - 1].bytes = 0;
    PushIndex(g_freeNodes, i);
}
uint64_t Key(int device, size_t bytes) noexcept {
    if (device < 0 || device >= 0xFFFE || bytes == 0 || bytes >= (uint64_t{1} << 48)) return 0;
    return (uint64_t(device + 1) << 48) | uint64_t(bytes);
}
int KeyDevice(uint64_t key) noexcept { return int(key >> 48) - 1; }
size_t KeyBytes(uint64_t key) noexcept { return size_t(key & ((uint64_t{1} << 48) - 1)); }
CacheBucket* FindBucket(uint64_t key, bool insert) noexcept {
    if (!key) return nullptr;
    uint64_t h = key * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    for (uint32_t probe = 0; probe != kCacheBuckets; ++probe) {
        CacheBucket& b = g_buckets[(h + probe) & (kCacheBuckets - 1)];
        uint64_t k = b.key.load(std::memory_order_acquire);
        if (k == key) return &b;
        if (k) continue;
        if (!insert) return nullptr;  // keys are never removed
        if (b.key.compare_exchange_strong(k, key, std::memory_order_acq_rel,
                                          std::memory_order_acquire) || k == key)
            return &b;
    }
    return nullptr;
}
bool PopBucket(CacheBucket& b, CachedBlock* out) noexcept {
    uint32_t const i = PopIndex(b.head);
    if (!i) return false;
    uint64_t const key = b.key.load(std::memory_order_relaxed);
    *out = CachedBlock{g_nodes[i - 1].ptr, g_nodes[i - 1].ready, g_nodes[i - 1].bytes,
                       KeyBytes(key), KeyDevice(key)};
    FreeNode(i);
    g_cacheBytes.fetch_sub(out->bytes, std::memory_order_acq_rel);
    g_cachePhysical.fetch_sub(out->capacity, std::memory_order_acq_rel);
    return true;
}
// Pops any block, preferring `device` (or only `device` when `only`).
bool PopAny(int device, bool only, CachedBlock* out) noexcept {
    for (int pass = 0; pass != (only ? 1 : 2); ++pass) {
        for (auto& b : g_buckets) {
            uint64_t const key = b.key.load(std::memory_order_acquire);
            if (!key || (KeyDevice(key) == device) != (pass == 0)) continue;
            if (PopBucket(b, out)) return true;
        }
    }
    return false;
}

void FreeBlock(CachedBlock const& b) noexcept {
    if (b.ready) cudaEventDestroy(b.ready);
    if (b.ptr) cudaFree(b.ptr);
}

// Same (device, class), most-recently-freed first.
bool PopCache(int device, size_t capacity, CachedBlock* out) {
    CacheBucket* b = FindBucket(Key(device, capacity), false);
    return b && PopBucket(*b, out);
}

// Insert a freed block, then evict while over the cap. The caller has
// `device` selected: same-device evictions issue directly, foreign ones
// save/set/restore around the free. A failed set keeps the entry and stops
// eviction (soft cap) rather than leaking or spinning.
bool PushBlock(CachedBlock const& block) noexcept {
    CacheBucket* b = FindBucket(Key(block.device, block.capacity), true);
    uint32_t const i = b ? AllocNode() : 0;
    if (!i) return false;
    g_nodes[i - 1].ptr = block.ptr; g_nodes[i - 1].ready = block.ready;
    g_nodes[i - 1].bytes = block.bytes;
    g_cacheBytes.fetch_add(block.bytes, std::memory_order_acq_rel);
    g_cachePhysical.fetch_add(block.capacity, std::memory_order_acq_rel);
    PushIndex(b->head, i);
    return true;
}
void PushCache(CachedBlock const& block) {
    int const device = block.device;
    if (!PushBlock(block)) {
        FreeBlock(block);
        return;
    }
    while (g_cachePhysical.load(std::memory_order_acquire) > kCacheCapBytes) {
        CachedBlock victim;
        if (!PopAny(device, false, &victim)) return;
        bool const sameDevice = victim.device == device;
        if (!sameDevice && cudaSetDevice(victim.device) != cudaSuccess) {
            // Keep it (soft cap); with no slot left, leak rather than free
            // in a foreign context.
            (void)PushBlock(victim);
            return;
        }
        FreeBlock(victim);
        if (!sameDevice) cudaSetDevice(device);
    }
}

// Free every cached block for `device`, already selected by the caller.
// Best-effort: a backend failure just drops the handles.
void DrainCacheForDevice(int device) noexcept {
    CachedBlock b;
    while (PopAny(device, true, &b)) FreeBlock(b);
}
}  // namespace

size_t UsdGenGpuDeviceBufferCacheBytes() noexcept {
    return g_cacheBytes.load(std::memory_order_acquire);
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
    size_t const capacity = ClassBytes(bytes);
    if (data_ && device == device_ && capacity == capacity_) {
        // Same class: keep the block and its event, recharge the new size.
        auto permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
        if (!permit) {
            DrainCacheForDevice(device);
            permit = TryReserveCudaExecutionBytes(bytes, kind, reservation);
            if (!permit) return cudaErrorMemoryAllocation;
        }
        permit_.Release();
        permit_ = std::move(*permit);
        size_ = count;
        return cudaSuccess;
    }
    CachedBlock hit;
    bool const haveHit = PopCache(device, capacity, &hit);
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
    CachedBlock const block{data_, ready_, size_ * sizeof(T), capacity_, device_};
    int const dev = device_;
    data_ = nullptr; size_ = 0; ready_ = nullptr; device_ = -1; capacity_ = 0;
    permit_.Release();
    PushCache(block);
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
