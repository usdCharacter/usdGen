#include "deviceBufferImpl.h"
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
using detail::CachedBlock;
// Lock-free and allocation-free: blocks hang off fixed nodes in one bucket
// per (device, size class). A bucket is a Treiber stack whose head packs a
// 32-bit ABA tag with a 1-based node index. Buckets are indexed directly by
// device and class (classes are bounded: 161 below 2^48 bytes), so no key is
// ever claimed or exhausted and a lookup never probes; a device past
// kCacheDevices, or a full node array, frees the block instead of caching it.
// Deliberately leaked: entries must never run cudaFree during process
// teardown (the CUDA context may already be gone); the driver reclaims
// device memory with the context. All state is constant-initialized.
constexpr uint32_t kCacheNodes = 8192;
constexpr uint32_t kCacheDevices = 16;
constexpr uint32_t kCacheClasses = 1 + 4 * (48 - 8);  // 256 bytes, then 4 per power of two
struct CacheNode {
    void* ptr = nullptr;
    cudaEvent_t ready = nullptr;
    size_t bytes = 0;
    std::atomic<uint32_t> next{0};
};
CacheNode g_nodes[kCacheNodes];
std::atomic<uint32_t> g_nodesUsed{0};
std::atomic<uint64_t> g_freeNodes{0};
std::atomic<uint64_t> g_buckets[kCacheDevices * kCacheClasses];  // tag << 32 | node index + 1
std::atomic<uint32_t> g_deviceBlocks[kCacheDevices];  // cached blocks per device (scan hint)
std::atomic<size_t> g_cacheBytes{0};     // requested bytes (gauge)
std::atomic<size_t> g_cachePhysical{0};  // class bytes (cap)
// Idle physical retention bound. Covers one scatter-grow peak (~380MB)
// with headroom; drain-on-reserve-failure keeps tight budgets correct
// regardless of this value.
constexpr size_t kCacheCapBytes = size_t{1} << 30;

unsigned FloorLog2(size_t v) noexcept {
    unsigned top = 0;
    while (v >>= 1) ++top;
    return top;
}

// 256 bytes, then four classes per power of two (at most 25% slack).
size_t ClassBytes(size_t bytes) noexcept {
    if (bytes <= 256) return 256;
    unsigned const top = FloorLog2(bytes - 1);
    size_t const step = size_t{1} << (top - 2);
    size_t const rounded = (bytes + step - 1) & ~(step - 1);
    return rounded < bytes ? bytes : rounded;
}

// The bucket class of a ClassBytes result; kCacheClasses when it has none
// (2^48 bytes or more).
uint32_t ClassIndex(size_t capacity) noexcept {
    if (capacity <= 256) return 0;
    unsigned const top = FloorLog2(capacity - 1);  // capacity in (2^top, 2^(top+1)]
    if (top >= 48) return kCacheClasses;
    size_t const step = size_t{1} << (top - 2);
    size_t const k = (capacity - (size_t{1} << top)) / step;  // 1..4
    return 1 + (top - 8) * 4 + uint32_t(k - 1);
}
size_t ClassCapacity(uint32_t index) noexcept {
    if (!index) return 256;
    unsigned const top = 8 + (index - 1) / 4;
    return (size_t{1} << top) + size_t((index - 1) % 4 + 1) * (size_t{1} << (top - 2));
}
std::atomic<uint64_t>* Bucket(int device, size_t capacity) noexcept {
    uint32_t const c = ClassIndex(capacity);
    if (device < 0 || uint32_t(device) >= kCacheDevices || c >= kCacheClasses ||
        ClassCapacity(c) != capacity)
        return nullptr;
    return &g_buckets[uint32_t(device) * kCacheClasses + c];
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
bool PopBucket(int device, uint32_t c, CachedBlock* out) noexcept {
    uint32_t const i = PopIndex(g_buckets[uint32_t(device) * kCacheClasses + c]);
    if (!i) return false;
    *out = CachedBlock{g_nodes[i - 1].ptr, g_nodes[i - 1].ready, g_nodes[i - 1].bytes,
                       ClassCapacity(c), device};
    FreeNode(i);
    g_deviceBlocks[device].fetch_sub(1, std::memory_order_acq_rel);
    g_cacheBytes.fetch_sub(out->bytes, std::memory_order_acq_rel);
    g_cachePhysical.fetch_sub(out->capacity, std::memory_order_acq_rel);
    return true;
}
// Pops one of `device`'s blocks, largest class first (most memory back per
// eviction, and big blocks of sizes no longer in use do not outstay small
// fresh ones).
bool PopLargest(int device, CachedBlock* out) noexcept {
    if (!g_deviceBlocks[device].load(std::memory_order_acquire)) return false;
    for (uint32_t c = kCacheClasses; c-- != 0;)
        if (PopBucket(device, c, out)) return true;
    return false;
}
// Pops any block, preferring `device` (or only `device` when `only`).
bool PopAny(int device, bool only, CachedBlock* out) noexcept {
    bool const local = device >= 0 && uint32_t(device) < kCacheDevices;
    if (local && PopLargest(device, out)) return true;
    if (only) return false;
    for (uint32_t d = 0; d != kCacheDevices; ++d)
        if (int(d) != device && PopLargest(int(d), out)) return true;
    return false;
}

void FreeBlock(CachedBlock const& b) noexcept {
    if (b.ready) cudaEventDestroy(b.ready);
    if (b.ptr) cudaFree(b.ptr);
}

// Same (device, class), most-recently-freed first.
bool PopCache(int device, size_t capacity, CachedBlock* out) {
    return Bucket(device, capacity) && PopBucket(device, ClassIndex(capacity), out);
}

// Insert a freed block, then evict while over the cap. The caller has
// `device` selected: same-device evictions issue directly, foreign ones
// save/set/restore around the free. A failed set keeps the entry and stops
// eviction (soft cap) rather than leaking or spinning.
bool PushBlock(CachedBlock const& block) noexcept {
    std::atomic<uint64_t>* b = Bucket(block.device, block.capacity);
    uint32_t const i = b ? AllocNode() : 0;
    if (!i) return false;
    g_nodes[i - 1].ptr = block.ptr; g_nodes[i - 1].ready = block.ready;
    g_nodes[i - 1].bytes = block.bytes;
    g_cacheBytes.fetch_add(block.bytes, std::memory_order_acq_rel);
    g_cachePhysical.fetch_add(block.capacity, std::memory_order_acq_rel);
    g_deviceBlocks[block.device].fetch_add(1, std::memory_order_acq_rel);
    PushIndex(*b, i);
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

namespace detail {
size_t CacheClassBytes(size_t bytes) noexcept { return gpu::ClassBytes(bytes); }
bool CachePop(int device, size_t capacity, CachedBlock* out) {
    return gpu::PopCache(device, capacity, out);
}
void CachePush(CachedBlock const& block) { gpu::PushCache(block); }
void CacheDrain(int device) noexcept { gpu::DrainCacheForDevice(device); }
void CacheFreeBlock(CachedBlock const& block) noexcept { gpu::FreeBlock(block); }
}  // namespace detail

size_t UsdGenGpuDeviceBufferCacheBytes() noexcept {
    return g_cacheBytes.load(std::memory_order_acquire);
}

template class DeviceBuffer<float>;
template class DeviceBuffer<double>;
template class DeviceBuffer<int>;
template class DeviceBuffer<unsigned char>;
template class DeviceBuffer<uint32_t>;
template class DeviceBuffer<uint64_t>;
template class DeviceBuffer<float3>;
template class DeviceBuffer<float2>;
}}
