// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — size-class pools. See pool.h.
#include "usdGen/api/pool.h"

#include "usdGen/exec/allocCounter.h"

#include <atomic>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace usdGen {
namespace {

// 16-byte steps to 256, then four classes per power of two to 64 MiB.
constexpr int kSmallClasses = 16;
constexpr int kClasses = kSmallClasses + 18 * 4;
// Block indices are 32-bit in 16-byte units.
constexpr std::size_t kRegionBytes = std::size_t(32) << 30;
constexpr std::size_t kSlabBytes = std::size_t(64) << 10;

constexpr int ClassOf(std::size_t n)
{
    if (n <= 256) return n ? int((n + 15) >> 4) - 1 : 0;
    int p = 8;
    while ((std::size_t(2) << p) < n) ++p;          // 2^p < n <= 2^(p+1)
    std::size_t const step = (std::size_t(1) << p) >> 2;
    int const sub = int((n - (std::size_t(1) << p) + step - 1) / step);   // 1..4
    return kSmallClasses + (p - 8) * 4 + sub - 1;
}

constexpr std::size_t ClassBytes(int c)
{
    if (c < kSmallClasses) return std::size_t(c + 1) << 4;
    int const p = 8 + (c - kSmallClasses) / 4;
    int const sub = (c - kSmallClasses) % 4 + 1;
    return (std::size_t(1) << p) + std::size_t(sub) * ((std::size_t(1) << p) >> 2);
}

static_assert(ClassBytes(kClasses - 1) == kUsdGenPoolMaxBytes, "class table");
static_assert(ClassOf(kUsdGenPoolMaxBytes) == kClasses - 1, "class table");
static_assert(ClassOf(257) == kSmallClasses && ClassBytes(kSmallClasses) == 320, "class table");

// Blocks a magazine holds: about 16 KiB of small blocks, one large block.
constexpr uint32_t Capacity(int c)
{
    std::size_t const n = (std::size_t(16) << 10) / ClassBytes(c);
    return n < 1 ? 1u : n > 32 ? 32u : uint32_t(n);
}

struct alignas(64) Depot
{
    std::atomic<uint64_t> head{0};   // low: chain block index (0 empty), high: tag
};

std::atomic<char *> g_base{nullptr};
std::atomic<bool> g_reserveFailed{false};
alignas(64) std::atomic<std::size_t> g_frontier{0};
Depot g_depot[kClasses];
std::atomic<uint64_t> g_refills{0};
std::atomic<uint64_t> g_heapFallbacks{0};
std::atomic<bool> g_denyGrowth{false};  // tests: force the heap fallback

// Requests above 256 bytes are 64-byte aligned (pool.h), from the heap too;
// the matching free uses the same rule, keyed by the request size.
constexpr std::size_t kSmallMax = 256;
constexpr std::align_val_t kLargeAlign{64};

char *Reserve()
{
#ifdef _WIN32
    return static_cast<char *>(VirtualAlloc(nullptr, kRegionBytes, MEM_RESERVE, PAGE_NOACCESS));
#else
    void *p = mmap(nullptr, kRegionBytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<char *>(p);
#endif
}

void Unreserve(char *p)
{
#ifdef _WIN32
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, kRegionBytes);
#endif
}

bool Commit(char *p, std::size_t bytes)
{
#ifdef _WIN32
    return VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    (void)p; (void)bytes;   // MAP_NORESERVE pages commit on first touch
    return true;
#endif
}

// The region, reserved by whichever thread gets here first.
char *Base()
{
    char *base = g_base.load(std::memory_order_acquire);
    if (base || g_reserveFailed.load(std::memory_order_relaxed)) return base;
    char *mine = Reserve();
    if (!mine) {
        g_reserveFailed.store(true, std::memory_order_relaxed);
        return nullptr;
    }
    if (!g_base.compare_exchange_strong(base, mine, std::memory_order_acq_rel)) {
        Unreserve(mine);
        return base;
    }
    return mine;
}

inline bool InRegion(void const *p)
{
    char const *base = g_base.load(std::memory_order_acquire);
    return base && static_cast<char const *>(p) >= base &&
           static_cast<char const *>(p) < base + kRegionBytes;
}

inline uint32_t IndexOf(void *p)
{
    return uint32_t((static_cast<char *>(p) - g_base.load(std::memory_order_relaxed)) >> 4) + 1;
}

inline void *BlockAt(uint32_t index)
{
    return g_base.load(std::memory_order_relaxed) + (std::size_t(index - 1) << 4);
}

// Word 0 links a magazine's blocks; word 1 of a chain's first block holds
// the depot link (low) and the chain length (high).
inline void *&Next(void *block) { return *static_cast<void **>(block); }
inline std::atomic<uint64_t> &ChainWord(void *block)
{
    return *reinterpret_cast<std::atomic<uint64_t> *>(static_cast<char *>(block) + 8);
}

void PushChain(int c, void *head, uint32_t count)
{
    uint32_t const index = IndexOf(head);
    std::atomic<uint64_t> &word = ChainWord(head);
    Depot &d = g_depot[c];
    uint64_t old = d.head.load(std::memory_order_relaxed);
    for (;;) {
        word.store((uint64_t(count) << 32) | uint32_t(old), std::memory_order_relaxed);
        uint64_t const next = (uint64_t(uint32_t(old >> 32) + 1) << 32) | index;
        if (d.head.compare_exchange_weak(old, next, std::memory_order_release,
                                         std::memory_order_relaxed))
            return;
    }
}

bool PopChain(int c, void **head, uint32_t *count)
{
    Depot &d = g_depot[c];
    uint64_t old = d.head.load(std::memory_order_acquire);
    for (;;) {
        uint32_t const index = uint32_t(old);
        if (!index) return false;
        void *block = BlockAt(index);
        // A racing pop may hand this block out and overwrite the word; the
        // tag then fails the CAS below. The range is never unmapped.
        uint64_t const word = ChainWord(block).load(std::memory_order_relaxed);
        uint64_t const next = (uint64_t(uint32_t(old >> 32) + 1) << 32) | uint32_t(word);
        if (d.head.compare_exchange_weak(old, next, std::memory_order_acquire,
                                         std::memory_order_acquire)) {
            *head = block;
            *count = uint32_t(word >> 32);
            return true;
        }
    }
}

struct Magazine
{
    void *head = nullptr;
    uint32_t count = 0;
};

// Carve a slab: the first magazine's worth goes to `out`, the rest to the
// depot in magazine-sized chains. False when the region is exhausted.
bool Refill(int c, Magazine *out)
{
    if (g_denyGrowth.load(std::memory_order_relaxed)) return false;
    char *base = Base();
    if (!base) return false;
    std::size_t const cb = ClassBytes(c);
    uint32_t const cap = Capacity(c);
    std::size_t n = kSlabBytes / cb;
    if (n < cap) n = cap;
    std::size_t const bytes = (n * cb + 63) & ~std::size_t(63);
    std::size_t const offset = g_frontier.fetch_add(bytes, std::memory_order_relaxed);
    if (offset + bytes > kRegionBytes) return false;
    char *slab = base + offset;
    std::size_t const page = 4096;
    char *pageStart = base + (offset & ~(page - 1));
    char *pageEnd = base + ((offset + bytes + page - 1) & ~(page - 1));
    if (!Commit(pageStart, std::size_t(pageEnd - pageStart))) return false;
    g_refills.fetch_add(1, std::memory_order_relaxed);
    UsdGenEngineAllocNote(bytes);   // pool growth is an engine allocation
    std::size_t i = 0;
    auto chain = [&](uint32_t want, Magazine *m) {
        for (uint32_t k = 0; k < want && i < n; ++k, ++i) {
            void *block = slab + i * cb;
            Next(block) = m->head;
            m->head = block;
            ++m->count;
        }
    };
    chain(cap, out);
    while (i < n) {
        Magazine rest;
        chain(cap, &rest);
        PushChain(c, rest.head, rest.count);
    }
    return true;
}

struct ThreadCache
{
    Magazine loaded[kClasses];
    Magazine spare[kClasses];
    bool dead = false;

    // Hand everything back so blocks outlive the thread. Lock-free and
    // non-blocking, so safe at thread and process exit.
    ~ThreadCache()
    {
        for (int c = 0; c < kClasses; ++c) {
            if (loaded[c].count) PushChain(c, loaded[c].head, loaded[c].count);
            if (spare[c].count) PushChain(c, spare[c].head, spare[c].count);
            loaded[c] = spare[c] = Magazine();
        }
        dead = true;
    }
};

thread_local ThreadCache t_cache;

void *HeapFallback(std::size_t bytes)
{
    g_heapFallbacks.fetch_add(1, std::memory_order_relaxed);
    return bytes > kSmallMax ? ::operator new(bytes, kLargeAlign) : ::operator new(bytes);
}

void HeapFree(void *p, std::size_t bytes) noexcept
{
    if (bytes > kSmallMax) ::operator delete(p, kLargeAlign);
    else ::operator delete(p);
}

}  // namespace

std::size_t UsdGenPoolClassBytes(std::size_t bytes) noexcept
{
    return bytes > kUsdGenPoolMaxBytes ? bytes : ClassBytes(ClassOf(bytes));
}

void *UsdGenPoolAlloc(std::size_t bytes)
{
    if (bytes > kUsdGenPoolMaxBytes) return HeapFallback(bytes);
    int const c = ClassOf(bytes);
    ThreadCache &tc = t_cache;
    if (tc.dead) {
        // After this thread's cache is gone (exit): take one block from a
        // depot chain and return the remainder.
        void *head = nullptr;
        uint32_t count = 0;
        Magazine fresh;
        if (!PopChain(c, &head, &count)) {
            if (!Refill(c, &fresh)) return HeapFallback(bytes);
            head = fresh.head;
            count = fresh.count;
        }
        if (count > 1) PushChain(c, Next(head), count - 1);
        return head;
    }
    Magazine &m = tc.loaded[c];
    if (!m.count) {
        Magazine &s = tc.spare[c];
        if (s.count) {
            m = s;
            s = Magazine();
        } else {
            m = Magazine();
            if (!PopChain(c, &m.head, &m.count) && !Refill(c, &m)) {
                m = Magazine();
                return HeapFallback(bytes);
            }
        }
    }
    void *block = m.head;
    m.head = Next(block);
    --m.count;
    return block;
}

void UsdGenPoolFree(void *p, std::size_t bytes) noexcept
{
    if (!p) return;
    if (bytes > kUsdGenPoolMaxBytes || !InRegion(p)) {
        HeapFree(p, bytes);
        return;
    }
    int const c = ClassOf(bytes);
    ThreadCache &tc = t_cache;
    if (tc.dead) {
        Next(p) = nullptr;
        PushChain(c, p, 1);
        return;
    }
    Magazine &m = tc.loaded[c];
    if (m.count >= Capacity(c)) {
        Magazine &s = tc.spare[c];
        if (s.count) PushChain(c, s.head, s.count);
        s = m;
        m = Magazine();
    }
    Next(p) = m.head;
    m.head = p;
    ++m.count;
}

void UsdGenPoolTestDenyGrowth(bool deny) noexcept
{
    g_denyGrowth.store(deny, std::memory_order_relaxed);
}

UsdGenPoolStats UsdGenPoolGetStats() noexcept
{
    UsdGenPoolStats s;
    s.refills = g_refills.load(std::memory_order_relaxed);
    s.heapFallbacks = g_heapFallbacks.load(std::memory_order_relaxed);
    s.reservedBytes = g_frontier.load(std::memory_order_relaxed);
    return s;
}

}  // namespace usdGen
