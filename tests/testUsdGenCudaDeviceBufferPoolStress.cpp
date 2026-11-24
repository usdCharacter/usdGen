// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCudaDeviceBufferPoolStress — the lock-free DeviceBuffer block pool
// under contention: 16 threads reset and release buffers of a few sizes, so
// blocks cycle through the same buckets from every thread. A block must never
// be live in two buffers at once (checked by a CAS registry and a per-owner
// tag written to and read back from the block), and accounting must return to
// zero with the cache gauge equal to the idle bytes.
#include "usdGen/gpu/deviceBuffer.h"
#include "usdGen/gpu/deviceResources.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
constexpr size_t kThreads = 16;
constexpr size_t kIterations = 1500;
constexpr size_t kSizes[] = {64, 256, 1000, 4096};
constexpr uintptr_t kTombstone = 1;
constexpr size_t kRegistry = size_t{1} << 17;
std::atomic<uintptr_t> g_live[kRegistry];
std::atomic<int> g_errors{0};

size_t Slot(uintptr_t p) { return size_t((p >> 8) * 0x9E3779B97F4A7C15ull) & (kRegistry - 1); }
// Claims p; fails if p is already live. Tombstones are never reused, so a
// live entry is always found before an empty slot.
bool Claim(uintptr_t p) {
    for (size_t i = Slot(p), n = 0; n != kRegistry; i = (i + 1) & (kRegistry - 1), ++n) {
        uintptr_t v = g_live[i].load(std::memory_order_acquire);
        if (v == p) return false;
        if (v) continue;
        if (g_live[i].compare_exchange_strong(v, p, std::memory_order_acq_rel)) return true;
        if (v == p) return false;
    }
    return false;
}
void Unclaim(uintptr_t p) {
    for (size_t i = Slot(p), n = 0; n != kRegistry; i = (i + 1) & (kRegistry - 1), ++n) {
        uintptr_t v = g_live[i].load(std::memory_order_acquire);
        if (v == p) { g_live[i].store(kTombstone, std::memory_order_release); return; }
        if (!v) break;
    }
    g_errors.fetch_add(1);
}

void Worker(int device, uint32_t id) {
    if (cudaSetDevice(device) != cudaSuccess) { g_errors.fetch_add(1); return; }
    gpu::DeviceBuffer<uint32_t> held[2];
    for (size_t i = 0; i != kIterations; ++i) {
        auto& buffer = held[i & 1];
        if (buffer.data()) {
            Unclaim(reinterpret_cast<uintptr_t>(buffer.data()));
            buffer.release();
        }
        size_t const count = kSizes[(i * 7 + id) % 4];
        if (buffer.reset(count) != cudaSuccess || !buffer.data()) { g_errors.fetch_add(1); return; }
        if (!Claim(reinterpret_cast<uintptr_t>(buffer.data()))) { g_errors.fetch_add(1); return; }
        uint32_t const tag = (id << 20) | uint32_t(i);
        uint32_t back = 0;
        if (cudaMemcpy(buffer.data(), &tag, sizeof(tag), cudaMemcpyHostToDevice) != cudaSuccess ||
            buffer.recordUse(nullptr) != cudaSuccess) { g_errors.fetch_add(1); return; }
        std::this_thread::yield();
        if (cudaMemcpy(&back, buffer.data(), sizeof(back), cudaMemcpyDeviceToHost) != cudaSuccess ||
            back != tag) { g_errors.fetch_add(1); return; }
    }
    for (auto& buffer : held) {
        if (!buffer.data()) continue;
        Unclaim(reinterpret_cast<uintptr_t>(buffer.data()));
        buffer.release();
    }
}
}  // namespace

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(gpu::ConfigureCudaExecutionResources(device, {size_t{1} << 30, 0}));
    auto resources = FindUsdGenExecutionResourcePool({UsdGenExecutionResourceBackend::Cuda, device});
    CHECK(resources && resources->Snapshot().usedBytes == 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t != kThreads; ++t) threads.emplace_back(Worker, device, t);
    for (auto& t : threads) t.join();
    CHECK(g_errors.load() == 0);
    CHECK(resources->Snapshot().usedBytes == 0);
    // Every block is idle now: at most two per thread, each of a known size.
    size_t const cached = gpu::UsdGenGpuDeviceBufferCacheBytes();
    CHECK(cached > 0 && cached <= kThreads * 2 * 4096 * sizeof(uint32_t));
    // Draining by pressure: a budget-sized request sheds the whole cache.
    gpu::DeviceBuffer<uint8_t> big;
    CHECK(big.reset((size_t{1} << 30) + 1) == cudaErrorMemoryAllocation);
    CHECK(gpu::UsdGenGpuDeviceBufferCacheBytes() == 0 && resources->Snapshot().usedBytes == 0);
    std::printf("pool stress: %zu threads x %zu iterations, %zu idle bytes before drain\n",
                kThreads, kIterations, cached);
    return 0;
}
