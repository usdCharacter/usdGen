// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Retirement lane under contention. The queue is process-global and closes
// for good, so each mode is its own process:
//   (default)     many producers enqueue into the bounded ring (full most of
//                 the time) while several closers race them: every accepted
//                 callback runs exactly once, in per-producer order, off the
//                 producers' threads, before Close returns; nothing rejected
//                 runs; admission fails after Close.
//   --start-fail  a producer that arrives while another thread's lane start
//                 is about to fail must not be admitted without a lane: its
//                 callback runs (it starts the lane itself) or it is rejected.
#include "usdGenVulkan/vulkan/externalRetirement.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "retirement stress: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {

int RaceClose() {
    constexpr unsigned kProducers = 16, kEach = 20000, kClosers = 4;
    std::vector<uint32_t> last(kProducers, 0);            // lane-only
    std::vector<std::atomic<uint64_t>> accepted(kProducers);
    std::atomic<uint64_t> ran{0}, rejected{0};
    std::atomic<bool> closing{false};
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
                    })) {
                    accepted[p].fetch_add(1, std::memory_order_relaxed);
                    ++i;
                    continue;
                }
                rejected.fetch_add(1, std::memory_order_relaxed);
                // Full or closed: once a close began, a rejection may be final.
                if (closing.load(std::memory_order_acquire)) return;
                std::this_thread::yield();
            }
        });
    // Close while every producer is still enqueuing.
    while (ran.load(std::memory_order_relaxed) < uint64_t(kProducers) * kEach / 4)
        std::this_thread::yield();
    std::vector<std::thread> closers;
    std::atomic<uint64_t> ranAtClose[kClosers];
    for (unsigned c = 0; c != kClosers; ++c)
        closers.emplace_back([&, c] {
            closing.store(true, std::memory_order_release);
            CloseVulkanExternalRetirement();
            ranAtClose[c].store(ran.load(std::memory_order_acquire));
            // After any Close returns, admission is closed for good.
            CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
        });
    for (auto& t : closers) t.join();
    uint64_t const ranAfterClose = ran.load(std::memory_order_acquire);
    for (auto& t : producers) t.join();
    uint64_t total = 0;
    for (unsigned p = 0; p != kProducers; ++p) {
        // Exactly the accepted callbacks ran, in order, and none of them
        // after Close returned.
        CHECK(last[p] == accepted[p].load());
        total += accepted[p].load();
    }
    CHECK(ran.load() == total && ranAfterClose == total);
    for (unsigned c = 0; c != kClosers; ++c) CHECK(ranAtClose[c].load() == total);
    CHECK(total < uint64_t(kProducers) * kEach);  // the close really raced admission
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
    std::printf("retirement stress: %llu accepted and run, %llu rejections: PASS\n",
                (unsigned long long)total, (unsigned long long)rejected.load());
    return 0;
}

std::atomic<int> hookCalls{0};
std::atomic<bool> secondReturned{false};
bool FailFirstStart() {
    if (hookCalls.fetch_add(1) != 0) return true;
    // Give the second producer every chance to be admitted against this
    // start before it fails.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (!secondReturned.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    return false;
}

int StartFail() {
    SetVulkanExternalRetirementLaneStartHookForTesting(&FailFirstStart);
    std::atomic<unsigned> ran{0};
    bool first = true, second = false;
    std::thread p1([&] { first = EnqueueVulkanExternalRetirement([&] { ran.fetch_add(1); }); });
    while (hookCalls.load() == 0) std::this_thread::yield();
    std::thread p2([&] {
        second = EnqueueVulkanExternalRetirement([&] { ran.fetch_add(1); });
        secondReturned.store(true);
    });
    p1.join();
    p2.join();
    CHECK(!first);  // its own start failed
    CloseVulkanExternalRetirement();
    // An accepted callback always runs; the start-in-flight producer started
    // the lane itself once the first start failed.
    CHECK(ran.load() == unsigned(first) + unsigned(second));
    CHECK(second && hookCalls.load() == 2);
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
    SetVulkanExternalRetirementLaneStartHookForTesting(nullptr);
    std::printf("retirement start-fail: PASS\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--start-fail") == 0) return StartFail();
    CHECK(argc == 1);
    return RaceClose();
}
