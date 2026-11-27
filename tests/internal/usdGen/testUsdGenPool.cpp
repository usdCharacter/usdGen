// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenPool: size-class pools and per-thread bump scratch.
//   - every class: alignment, usable size, reuse on the same thread
//   - steady state: alloc/free after warm-up makes no counted allocation
//   - stress: many threads allocate, exchange blocks through shared slots
//     (so most frees are cross-thread and move through the depots) and
//     verify that no block is ever handed to two owners
//   - scratch: nesting, rewind, growth, one arena per thread
//   - heap fallback (growth denied) keeps the 64-byte alignment above 256

#include "usdGen/exec/allocCounter.h"
#include "usdGen/api/pool.h"
#include "usdGen/api/ext/scratch.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

uint64_t Mix(uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

// A block carries its size and a fill derived from its owner tag, so a block
// handed to two owners at once shows up as a corrupted fill.
constexpr uint64_t kMagic = 0x5553444750304f4cull;
void Stamp(void *p, std::size_t bytes, uint64_t tag)
{
    auto *w = static_cast<uint64_t *>(p);
    std::size_t const words = bytes / 8;
    w[0] = bytes;
    w[1] = kMagic ^ tag;
    for (std::size_t i = 2; i < words && i < 64; ++i) w[i] = tag + i;
    if (words > 64) w[words - 1] = ~tag;
}
bool Verify(void const *p, std::size_t *bytes)
{
    auto const *w = static_cast<uint64_t const *>(p);
    *bytes = std::size_t(w[0]);
    uint64_t const tag = w[1] ^ kMagic;
    std::size_t const words = *bytes / 8;
    for (std::size_t i = 2; i < words && i < 64; ++i)
        if (w[i] != tag + i) return false;
    return words <= 64 || w[words - 1] == ~tag;
}

std::size_t SizeFor(uint64_t r)
{
    // Mostly small, sometimes up to 1 MiB; at least 16 for the stamp.
    switch (r & 7) {
    case 0: return 16 + (r >> 8) % (1u << 20);
    case 1: case 2: return 16 + (r >> 8) % 4096;
    default: return 16 + (r >> 8) % 256;
    }
}

void TestClasses()
{
    for (std::size_t n = 1; n <= (std::size_t(4) << 20); n = n < 64 ? n + 1 : n + n / 3) {
        void *p = UsdGenPoolAlloc(n);
        std::size_t const cb = UsdGenPoolClassBytes(n);
        Check(cb >= n, "class size covers the request");
        Check((reinterpret_cast<uintptr_t>(p) & 15) == 0, "16-byte alignment");
        if (n > 256) Check((reinterpret_cast<uintptr_t>(p) & 63) == 0, "64-byte alignment above 256");
        std::memset(p, 0xab, cb);
        UsdGenPoolFree(p, n);
        void *q = UsdGenPoolAlloc(n);
        Check(q == p, "a freed block comes back to the same thread first");
        UsdGenPoolFree(q, n);
    }
    void *big = UsdGenPoolAlloc(kUsdGenPoolMaxBytes + 1);
    Check(big != nullptr, "oversize requests fall back to the heap");
    UsdGenPoolFree(big, kUsdGenPoolMaxBytes + 1);
}

void TestSteadyState()
{
    std::vector<void *> held(512);
    auto round = [&] {
        for (std::size_t i = 0; i < held.size(); ++i)
            held[i] = UsdGenPoolAlloc(16 + (i * 97) % 20000);
        for (std::size_t i = 0; i < held.size(); ++i)
            UsdGenPoolFree(held[i], 16 + (i * 97) % 20000);
    };
    round();
    round();
    UsdGenEngineAllocCounting(true);
    for (int r = 0; r < 50; ++r) round();
    UsdGenEngineAllocCounting(false);
    Check(UsdGenEngineAllocCount() == 0, "warm alloc/free makes no counted allocation");
}

void TestStress(int threads, int iterations)
{
    constexpr int kSlots = 256;
    std::vector<std::atomic<void *>> slots(kSlots);
    for (auto &s : slots) s.store(nullptr);
    std::atomic<int> bad{0};
    std::atomic<int> start{0};
    auto worker = [&](int t) {
        start.fetch_add(1);
        while (start.load() < threads) {}
        std::vector<std::pair<void *, std::size_t>> mine;
        uint64_t seed = Mix(uint64_t(t) + 1);
        for (int i = 0; i < iterations; ++i) {
            seed = Mix(seed);
            std::size_t const n = SizeFor(seed);
            void *p = UsdGenPoolAlloc(n);
            Stamp(p, n, seed);
            if (seed & 0x100) {
                // Trade with another thread: whatever comes back is theirs.
                void *other = slots[(seed >> 16) % kSlots].exchange(p);
                if (other) {
                    std::size_t bytes = 0;
                    if (!Verify(other, &bytes)) bad.fetch_add(1);
                    else UsdGenPoolFree(other, bytes);
                }
            } else {
                mine.push_back({p, n});
            }
            if (mine.size() > 64 || (seed & 0x3000) == 0) {
                for (auto &b : mine) {
                    std::size_t bytes = 0;
                    if (!Verify(b.first, &bytes) || bytes != b.second) bad.fetch_add(1);
                    UsdGenPoolFree(b.first, b.second);
                }
                mine.clear();
            }
        }
        for (auto &b : mine) {
            std::size_t bytes = 0;
            if (!Verify(b.first, &bytes) || bytes != b.second) bad.fetch_add(1);
            UsdGenPoolFree(b.first, b.second);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();
    for (auto &s : slots) {
        if (void *p = s.exchange(nullptr)) {
            std::size_t bytes = 0;
            if (!Verify(p, &bytes)) bad.fetch_add(1);
            else UsdGenPoolFree(p, bytes);
        }
    }
    Check(bad.load() == 0, "no block is ever owned twice under contention");
}

// With growth denied, a class whose depot runs dry falls back to the heap;
// the fallback keeps the documented alignment (64 above 256 bytes) and
// frees through the matching path.
void TestFallbackAlignment()
{
    std::thread([] {
        UsdGenPoolTestDenyGrowth(true);
        uint64_t const f0 = UsdGenPoolGetStats().heapFallbacks;
        constexpr std::size_t kBytes = 300;  // the 320-byte class
        std::vector<void *> held;
        int misaligned = 0;
        while (UsdGenPoolGetStats().heapFallbacks - f0 < 256 && held.size() < 400000) {
            void *p = UsdGenPoolAlloc(kBytes);
            if (reinterpret_cast<uintptr_t>(p) & 63) ++misaligned;
            std::memset(p, 0x5a, kBytes);
            held.push_back(p);
        }
        uint64_t const fallbacks = UsdGenPoolGetStats().heapFallbacks - f0;
        UsdGenPoolTestDenyGrowth(false);
        for (void *p : held) UsdGenPoolFree(p, kBytes);
        std::printf("fallback: %zu blocks, %llu from the heap\n", held.size(),
                    (unsigned long long)fallbacks);
        Check(fallbacks >= 256, "denied growth reaches the heap fallback");
        Check(misaligned == 0, "heap fallback above 256 bytes is 64-byte aligned");
    }).join();
}

void TestScratch()
{
    UsdGenScratchArena &arena = UsdGenScratchArena::ForThread();
    void *first = nullptr;
    {
        UsdGenScratchScope outer;
        float *a = outer.Filled<float>(100, 2.0f);
        first = a;
        Check((reinterpret_cast<uintptr_t>(a) & 15) == 0, "scratch alignment");
        {
            UsdGenScratchScope inner;
            double *b = inner.Array<double>(1000);
            for (int i = 0; i < 1000; ++i) b[i] = i;
            Check(static_cast<void *>(b) != static_cast<void *>(a), "nested scope allocates past the outer");
        }
        {
            UsdGenScratchScope inner;
            // Grows past the first block while the outer allocation lives.
            uint8_t *big = inner.Filled<uint8_t>(std::size_t(3) << 20, 7);
            Check(big[(std::size_t(3) << 20) - 1] == 7, "scratch growth is writable");
        }
        Check(a[99] == 2.0f, "outer scratch survives inner scopes and growth");
    }
    {
        UsdGenScratchScope again;
        float *a = again.Array<float>(100);
        Check(static_cast<void *>(a) == first || a != nullptr, "rewound scratch reuses its block");
    }
    // Warm: the same pattern makes no counted allocation.
    auto pattern = [] {
        UsdGenScratchScope s;
        float *x = s.Filled<float>(std::size_t(1) << 18, 1.0f);
        UsdGenScratchScope t;
        uint32_t *y = t.Filled<uint32_t>(std::size_t(1) << 20, 3u);
        return x[5] + float(y[7]);
    };
    pattern();
    UsdGenEngineAllocCounting(true);
    float sum = 0.0f;
    for (int i = 0; i < 100; ++i) sum += pattern();
    UsdGenEngineAllocCounting(false);
    Check(sum == 400.0f, "scratch pattern values");
    Check(UsdGenEngineAllocCount() == 0, "warm scratch scopes make no counted allocation");
    (void)arena;
    // One arena per thread: concurrent scopes never share memory.
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&bad, t] {
            for (int r = 0; r < 200; ++r) {
                UsdGenScratchScope s;
                std::size_t const n = 1000 + std::size_t(r) * 37;
                uint32_t *v = s.Filled<uint32_t>(n, uint32_t(t * 100000 + r));
                for (std::size_t i = 0; i < n; ++i)
                    if (v[i] != uint32_t(t * 100000 + r)) { bad.fetch_add(1); break; }
            }
        });
    }
    for (auto &th : threads) th.join();
    Check(bad.load() == 0, "per-thread scratch arenas are disjoint");
}

}  // namespace

int main()
{
    TestClasses();
    TestSteadyState();
    unsigned const hw = std::thread::hardware_concurrency();
    int const threads = hw > 64 ? 64 : hw < 8 ? 8 : int(hw);
    for (int round = 0; round < 3; ++round) TestStress(threads, 20000);
    TestScratch();
    UsdGenPoolStats const s = UsdGenPoolGetStats();
    std::printf("pool: refills=%llu heapFallbacks=%llu reserved=%llu bytes\n",
                (unsigned long long)s.refills, (unsigned long long)s.heapFallbacks,
                (unsigned long long)s.reservedBytes);
    Check(s.heapFallbacks == 1, "only the deliberate oversize request reached the heap");
    TestFallbackAlignment();
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
