// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenCleanupQueue — T0: the imaging cleanup queue
// (usdGenImaging/cleanupQueue.h) that replaced the retirement flow graphs.
//
//   0. An action never runs on the posting thread, and Wait returns only
//      after it finished.
//   1. 32 threads post 2000 actions each while 4 threads Wait repeatedly;
//      every action runs exactly once and a final Wait sees all of them.
//   2. Actions run concurrently (unlimited, like the flow-graph node): two
//      actions that each wait for the other both finish.
//   3. An action may post another action; Wait covers both.
//   4. The first exception an action throws is rethrown by the next Wait,
//      once; the queue keeps working afterwards.

#include "usdGenImaging/cleanupQueue.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using usdGenImaging::UsdGenCleanupQueue;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

void NotInline()
{
    auto *q = new UsdGenCleanupQueue;  // leaked, as in production
    std::atomic<bool> ran{false}, inlineRun{false};
    auto const poster = std::this_thread::get_id();
    q->Post([&] {
        inlineRun.store(std::this_thread::get_id() == poster);
        std::this_thread::yield();
        ran.store(true);
    });
    q->Wait();
    Check(ran.load(), "Wait returns after the action finished");
    Check(!inlineRun.load(), "the action does not run on the posting thread");
    Check(q->Outstanding() == 0, "nothing outstanding after Wait");
}

void Stress()
{
    auto *q = new UsdGenCleanupQueue;
    constexpr int kPosters = 32, kEach = 2000, kWaiters = 4;
    std::vector<std::atomic<uint8_t>> runs(size_t(kPosters) * kEach);
    std::atomic<int> postersDone{0};
    std::vector<std::thread> threads;
    for (int p = 0; p < kPosters; ++p)
        threads.emplace_back([&, p] {
            for (int i = 0; i < kEach; ++i) {
                size_t const id = size_t(p) * kEach + size_t(i);
                q->Post([&runs, id] { runs[id].fetch_add(1, std::memory_order_relaxed); });
            }
            postersDone.fetch_add(1);
        });
    std::atomic<uint64_t> waits{0};
    for (int w = 0; w < kWaiters; ++w)
        threads.emplace_back([&] {
            while (postersDone.load() != kPosters) {
                q->Wait();
                waits.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (auto &t : threads) t.join();
    q->Wait();
    size_t once = 0;
    for (auto &r : runs) once += r.load() == 1;
    Check(once == runs.size(), "every one of " + std::to_string(runs.size()) +
                                   " actions ran exactly once (" +
                                   std::to_string(waits.load()) + " concurrent waits)");
}

void Concurrent()
{
    auto *q = new UsdGenCleanupQueue;
    std::atomic<int> arrived{0};
    auto meet = [&arrived] {
        arrived.fetch_add(1);
        // Bounded: a serial queue would leave the other action unstarted.
        for (int i = 0; i < 2000000 && arrived.load() < 2; ++i) std::this_thread::yield();
    };
    q->Post(meet);
    q->Post(meet);
    q->Wait();
    Check(arrived.load() == 2, "two actions run at the same time");
}

void Nested()
{
    auto *q = new UsdGenCleanupQueue;
    std::atomic<int> ran{0};
    q->Post([&] {
        ran.fetch_add(1);
        q->Post([&] { ran.fetch_add(1); });
    });
    q->Wait();
    Check(ran.load() == 2, "an action posted by an action is covered by Wait");
}

void Exceptions()
{
    auto *q = new UsdGenCleanupQueue;
    q->Post([] { throw std::runtime_error("boom"); });
    std::string what;
    try {
        q->Wait();
    } catch (std::runtime_error const &e) {
        what = e.what();
    }
    Check(what == "boom", "Wait rethrows the action's exception");
    bool again = false;
    try {
        q->Wait();
    } catch (...) {
        again = true;
    }
    Check(!again, "the exception is rethrown once");
    std::atomic<int> after{0};
    q->Post([&] { after.fetch_add(1); });
    q->Wait();
    Check(after.load() == 1, "the queue keeps working after a failure");
}

}  // namespace

int main()
{
    NotInline();
    Stress();
    Concurrent();
    Nested();
    Exceptions();
    std::printf("%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
