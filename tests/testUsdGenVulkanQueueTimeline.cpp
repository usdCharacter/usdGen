// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenVulkanQueueTimeline — the per-queue timeline semaphore and the
// size-class device memory pool under many threads.
//
// Timeline: 16 threads submit command buffers concurrently (single and
// three-batch calls) through one VulkanQueueTimeline. Every value handed out
// is unique and increasing per thread, the k-batch calls get k consecutive
// values, the batches execute in value order (each fills its own word, and
// the last writer per word is the newest), Check never blocks and Wait
// proves the newest value. Memory pool: 16 threads acquire and release
// blocks of mixed sizes; a held block is never handed to anyone else (each
// holder tags its block and finds the tag intact at release), idle blocks
// are reused, and growth stops once the classes are warm.
#include "usdGen/vulkan/deviceContext.h"
#include "usdGen/vulkan/memoryPool.h"
#include "usdGen/vulkan/queueTimeline.h"

#include "vulkanNativeFixture.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
uint32_t FindType(VkPhysicalDevice physical, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties p{};
    vkGetPhysicalDeviceMemoryProperties(physical, &p);
    for (uint32_t i = 0; i < p.memoryTypeCount; ++i)
        if ((p.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
}

int main() {
    bool unavailable = false;
    auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8101;
    ci.nativeLifetime = native;
    ci.resources = {size_t{64} << 20, 0};
    ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci);
    CHECK(context && context->computeTimeline() && context->memoryPool());
    VkDevice const d = context->device();
    VulkanQueueTimeline& timeline = *context->computeTimeline();
    CHECK(timeline.Check(0) == VK_SUCCESS);

    // ---- Timeline stress ----
    constexpr int kThreads = 16, kIters = 120, kBatch = 3;
    uint32_t const hostType = FindType(native->physical,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(hostType != UINT32_MAX);
    VkBuffer words = VK_NULL_HANDLE;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = kThreads * sizeof(uint32_t);
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(d, &bi, nullptr, &words) == VK_SUCCESS);
    VkResult r = VK_SUCCESS;
    auto* wordsMem = context->memoryPool()->Acquire(hostType, 4096, &r);
    CHECK(wordsMem && vkBindBufferMemory(d, words, wordsMem->memory, 0) == VK_SUCCESS);

    std::atomic<int> failures{0};
    std::vector<std::vector<uint64_t>> values(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            VkCommandPoolCreateInfo pc{};
            pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pc.queueFamilyIndex = native->family;
            VkCommandPool pool = VK_NULL_HANDLE;
            if (vkCreateCommandPool(d, &pc, nullptr, &pool) != VK_SUCCESS) { ++failures; return; }
            std::vector<VkCommandBuffer> cmds(size_t(kIters) * kBatch);
            VkCommandBufferAllocateInfo ca{};
            ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ca.commandPool = pool;
            ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ca.commandBufferCount = uint32_t(cmds.size());
            if (vkAllocateCommandBuffers(d, &ca, cmds.data()) != VK_SUCCESS) { ++failures; return; }
            uint32_t word = 0;
            for (int i = 0; i < kIters; ++i) {
                int const count = (i % 4 == 3) ? kBatch : 1;
                for (int b = 0; b < count; ++b) {
                    VkCommandBuffer cmd = cmds[size_t(i) * kBatch + size_t(b)];
                    VkCommandBufferBeginInfo begin{};
                    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    vkBeginCommandBuffer(cmd, &begin);
                    VkMemoryBarrier mb{};
                    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                    vkCmdFillBuffer(cmd, words, VkDeviceSize(t) * 4u, 4u, ++word);
                    vkEndCommandBuffer(cmd);
                }
                VkResult sr = VK_SUCCESS;
                uint64_t const last = timeline.Submit(&cmds[size_t(i) * kBatch], uint32_t(count), &sr);
                if (!last || sr != VK_SUCCESS) { ++failures; break; }
                for (int b = count - 1; b >= 0; --b) values[t].push_back(last - uint64_t(b));
                VkResult cr = timeline.Check(last);
                if (cr != VK_SUCCESS && cr != VK_NOT_READY) { ++failures; break; }
                if (i % 16 == 15 && timeline.Wait(last, 10000000000ull) != VK_SUCCESS) { ++failures; break; }
            }
            // The pool outlives its buffers' execution: wait our newest.
            if (!values[t].empty() && timeline.Wait(values[t].back(), 10000000000ull) != VK_SUCCESS)
                ++failures;
            vkDestroyCommandPool(d, pool, nullptr);
        });
    }
    for (auto& th : threads) th.join();
    CHECK(failures.load() == 0);
    std::set<uint64_t> all;
    size_t total = 0;
    for (int t = 0; t < kThreads; ++t) {
        for (size_t i = 1; i < values[t].size(); ++i) CHECK(values[t][i] > values[t][i - 1]);
        all.insert(values[t].begin(), values[t].end());
        total += values[t].size();
    }
    CHECK(all.size() == total);                       // unique
    CHECK(*all.begin() == 1 && *all.rbegin() == total); // dense: no hole, no failed value
    CHECK(timeline.Submitted() == total);
    CHECK(timeline.Wait(total, 10000000000ull) == VK_SUCCESS);
    CHECK(timeline.Completed() == total);
    {
        void* p = nullptr;
        CHECK(vkMapMemory(d, wordsMem->memory, 0, kThreads * 4u, 0, &p) == VK_SUCCESS);
        uint32_t got[kThreads];
        std::memcpy(got, p, sizeof(got));
        vkUnmapMemory(d, wordsMem->memory);
        // Each thread's batches ran in its submission order: its word holds
        // its last fill.
        for (int t = 0; t < kThreads; ++t) {
            uint32_t const fills = uint32_t(values[t].size());
            CHECK(got[t] == fills);
        }
    }
    auto ts = timeline.stats();
    std::printf("timeline: %zu values, %llu vkQueueSubmit calls, %llu combined, %llu parked\n",
        total, (unsigned long long)ts.submits, (unsigned long long)ts.combined,
        (unsigned long long)ts.parks);
    vkDestroyBuffer(d, words, nullptr);
    context->memoryPool()->Release(wordsMem);
    std::puts("Q1 (timeline: unique dense values, per-thread order, waits): PASS");

    // ---- Memory pool stress ----
    {
        VulkanDeviceMemoryPool pool(d, VkDeviceSize(256) << 20);
        VkDeviceSize const sizes[] = {256, 4096, 5000, 65536, 300000, 3u << 20};
        constexpr int kPoolThreads = 16, kPoolIters = 200;
        std::atomic<int> bad{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < kPoolThreads; ++t) {
            workers.emplace_back([&, t] {
                uint64_t s = 0x9E3779B97F4A7C15ull * uint64_t(t + 1);
                VulkanDeviceMemoryPool::Block* held[3] = {};
                for (int i = 0; i < kPoolIters; ++i) {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    int const slot = int(s % 3u);
                    if (held[slot]) {
                        uint32_t tag = 0;
                        void* p = nullptr;
                        if (vkMapMemory(d, held[slot]->memory, 0, 4, 0, &p) != VK_SUCCESS) { ++bad; return; }
                        std::memcpy(&tag, p, 4);
                        vkUnmapMemory(d, held[slot]->memory);
                        if ((tag >> 12) != uint32_t(t)) ++bad;  // someone else got our block
                        pool.Release(held[slot]);
                        held[slot] = nullptr;
                        continue;
                    }
                    VkDeviceSize const bytes = sizes[(s >> 8) % (sizeof(sizes) / sizeof(sizes[0]))];
                    VkResult ar = VK_SUCCESS;
                    auto* b = pool.Acquire(hostType, bytes, &ar);
                    if (!b || b->bytes < bytes) { ++bad; return; }
                    void* p = nullptr;
                    if (vkMapMemory(d, b->memory, 0, 4, 0, &p) != VK_SUCCESS) { ++bad; return; }
                    uint32_t const tag = uint32_t(t) << 12 | uint32_t(i % 4096);
                    std::memcpy(p, &tag, 4);
                    vkUnmapMemory(d, b->memory);
                    held[slot] = b;
                }
                for (auto*& b : held) if (b) { pool.Release(b); b = nullptr; }
            });
        }
        for (auto& w : workers) w.join();
        CHECK(bad.load() == 0);
        auto st = pool.stats();
        std::printf("memory pool: %llu allocations, %llu reuses, %llu frees, %llu idle bytes\n",
            (unsigned long long)st.allocations, (unsigned long long)st.reuses,
            (unsigned long long)st.frees, (unsigned long long)st.idleBytes);
        CHECK(st.reuses > 0);
        CHECK(st.allocations <= uint64_t(kPoolThreads) * 3u * 6u);
        // Warm: another round of the same sizes allocates nothing new.
        uint64_t const warm = st.allocations;
        for (VkDeviceSize bytes : sizes) {
            VkResult ar = VK_SUCCESS;
            auto* b = pool.Acquire(hostType, bytes, &ar);
            CHECK(b);
            pool.Release(b);
        }
        CHECK(pool.stats().allocations == warm);
        CHECK(VulkanDeviceMemoryPool::ClassBytes(5000) == 5120);
        CHECK(VulkanDeviceMemoryPool::ClassBytes(8192) == 8192);
        CHECK(VulkanDeviceMemoryPool::ClassBytes(8193) == 10240);
        CHECK(VulkanDeviceMemoryPool::ClassBytes(1) == 4096);
    }
    std::puts("Q2 (memory pool: exclusive blocks, reuse, warm growth stops): PASS");
    return 0;
}
