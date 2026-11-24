// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Retirement lane under contention: many producers enqueue into the bounded
// ring while it is full most of the time; every accepted callback runs once,
// in per-producer order, off the producers' threads; concurrent closers all
// return after the lane joined, and admission then fails.
#include "usdGen/vulkan/externalRetirement.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "retirement stress: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

int main() {
    constexpr unsigned kProducers = 16, kEach = 4000;
    std::atomic<uint64_t> ran{0};
    std::vector<uint32_t> last(kProducers, 0);  // lane-only
    std::atomic<uint64_t> rejectedFull{0};
    std::vector<std::thread> producers;
    for (unsigned p = 0; p != kProducers; ++p)
        producers.emplace_back([&, p] {
            auto const self = std::this_thread::get_id();
            for (uint32_t i = 1; i <= kEach;) {
                if (EnqueueVulkanExternalRetirement([&, p, i, self] {
                        CHECK(std::this_thread::get_id() != self);
                        CHECK(last[p] + 1 == i);
                        last[p] = i;
                        ran.fetch_add(1, std::memory_order_relaxed);
                    })) ++i;
                else { rejectedFull.fetch_add(1, std::memory_order_relaxed); std::this_thread::yield(); }
            }
        });
    for (auto& t : producers) t.join();
    std::vector<std::thread> closers;
    for (int c = 0; c != 4; ++c)
        closers.emplace_back([&] {
            CloseVulkanExternalRetirement();
            CHECK(ran.load() == uint64_t(kProducers) * kEach);
        });
    for (auto& t : closers) t.join();
    for (unsigned p = 0; p != kProducers; ++p) CHECK(last[p] == kEach);
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
    std::printf("retirement stress: %llu callbacks, %llu full rejections: PASS\n",
                (unsigned long long)ran.load(), (unsigned long long)rejectedFull.load());
    return 0;
}
