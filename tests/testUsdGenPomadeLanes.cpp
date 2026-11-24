// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenPomadeLanes — T0: the lock-free handoffs Pomade runs on.
//
//   * PomadeMailbox + PomadeWakeWord: many producers, one parked consumer;
//     newest wins, nothing leaks, no wakeup is lost.
//   * PomadeRegistry: concurrent register / activate / attach / publish
//     over the copy-on-write tables.
//   * PomadeModel published state: one authoring thread edits and
//     publishes while readers acquire states and check each one is a
//     consistent copy of a single version.
#include "usdGenPomade/pomadeLane.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeModelState.h"
#include "usdGenPomade/pomadeRegistry.h"

#include "usdGen/epoch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace usdGenPomade;

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
    std::fflush(stdout);
}

std::atomic<int> g_live{0};

struct Item {
    int producer = 0;
    int value = 0;
    Item(int p, int v) : producer(p), value(v) { g_live.fetch_add(1); }
    ~Item() { g_live.fetch_sub(1); }
};

void TestMailbox()
{
    constexpr int kProducers = 32;
    constexpr int kValues = 2000;
    {
        PomadeMailbox<Item> box;
        PomadeWakeWord wake;
        std::atomic<int> done{0};
        std::atomic<bool> stop{false};
        std::vector<int> lastSeen(kProducers, -1);
        bool monotonic = true;
        size_t taken = 0;
        std::thread consumer([&] {
            for (;;) {
                uint32_t const seen = wake.Sequence();
                if (std::unique_ptr<Item> item = box.Take()) {
                    ++taken;
                    if (item->value <= lastSeen[size_t(item->producer)]) {
                        monotonic = false;
                    }
                    lastSeen[size_t(item->producer)] = item->value;
                    continue;
                }
                if (stop.load(std::memory_order_acquire)) {
                    return;
                }
                wake.Park(seen);
            }
        });
        std::vector<std::thread> producers;
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p] {
                for (int v = 0; v < kValues; ++v) {
                    box.Publish(std::unique_ptr<Item>(new Item(p, v)));
                    wake.Notify();
                }
                done.fetch_add(1);
            });
        }
        for (std::thread &t : producers) {
            t.join();
        }
        // One last value per producer would be lost to newest-wins, so the
        // proof of no lost wakeup is: a final publish after the joins is
        // always taken before the consumer is told to stop.
        box.Publish(std::unique_ptr<Item>(new Item(0, kValues)));
        wake.Notify();
        auto const deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (lastSeen[0] != kValues &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        bool const finalSeen = lastSeen[0] == kValues;
        stop.store(true, std::memory_order_release);
        wake.Notify();
        consumer.join();
        Check(done.load() == kProducers, "mailbox: every producer finished");
        Check(finalSeen, "mailbox: the parked consumer took the final value");
        Check(monotonic, "mailbox: each producer's values arrive in order");
        Check(taken > 0 && taken <= size_t(kProducers) * kValues + 1,
              "mailbox: newest wins (taken " + std::to_string(taken) + ")");
    }
    Check(g_live.load() == 0, "mailbox: displaced and taken values freed");
}

void TestTimedPark()
{
    PomadeWakeWord word;
    auto const t0 = std::chrono::steady_clock::now();
    bool const notExpired =
        word.ParkUntil(word.Sequence(), t0 + std::chrono::milliseconds(20));
    auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    Check(notExpired && ms >= 10, "wake word: a timed park returns on time");
    Check(!word.ParkUntil(word.Sequence(), t0),
          "wake word: a passed deadline does not park");
}

class Target final : public PomadePublishTarget {
public:
    bool PublishModel(PomadeModel *, uint32_t) override
    {
        calls.fetch_add(1);
        return true;
    }
    bool QueryPublishedLevel(int, int *, int *, int *) const override
    {
        return false;
    }
    std::atomic<int> calls{0};
};

void TestRegistry()
{
    constexpr int kThreads = 32;
    constexpr int kRounds = 200;
    PomadeRegistry &registry = PomadeRegistry::Get();
    int const modelsBefore = registry.ModelCount();
    int const indicesBefore = registry.IndexCount();
    // Distinct fake model addresses: the registry never dereferences them.
    std::vector<char> storage(size_t(kThreads) * kRounds);
    std::vector<Target> targets(kThreads);
    std::atomic<int> badIds{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int r = 0; r < kRounds; ++r) {
                PomadeModel *fake = reinterpret_cast<PomadeModel *>(
                    &storage[size_t(t) * kRounds + size_t(r)]);
                int const id = registry.Register(fake);
                if (id <= 0 || registry.Register(fake) != id) {
                    badIds.fetch_add(1);
                }
                registry.AttachIndex(&targets[size_t(t)]);
                registry.SetActive(id);
                (void)registry.Active();
                (void)registry.Publish(0u);
                registry.DetachIndex(&targets[size_t(t)]);
                registry.Unregister(id);
                if (registry.SetActive(id)) {
                    badIds.fetch_add(1);  // an unregistered id activates
                }
            }
        });
    }
    for (std::thread &t : threads) {
        t.join();
    }
    Check(badIds.load() == 0,
          "registry: ids are stable and unregistered ids refuse");
    Check(registry.ModelCount() == modelsBefore,
          "registry: every concurrent register was unregistered");
    Check(registry.IndexCount() == indicesBefore,
          "registry: every concurrent attach was detached");
    int calls = 0;
    for (Target const &target : targets) {
        calls += target.calls.load();
    }
    Check(calls > 0, "registry: publishes reached attached targets");
    usdGen::UsdGenEpochDrain();
}

void TestModelState()
{
    constexpr int kReaders = 16;
    constexpr int kEdits = 1500;
    PomadeModel model;
    Check(model.BuildTestTube(), "state: test tube builds");
    // expected[v]: centerX[0] the authoring thread saw at version v,
    // written before the state carrying v is published.
    std::vector<float> expected(size_t(kEdits) * 4 + 64, 0.0f);
    expected[size_t(model.GetVersion())] = model.Snapshot().centerX[0];
    model.PublishState();
    std::atomic<bool> stop{false};
    std::atomic<int> torn{0};
    std::atomic<long> reads{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < kReaders; ++r) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                PomadeModelStateRef const state =
                    PomadeAcquireModelState(model);
                if (!state || state->version != state->tube.version ||
                    state->version >= expected.size() ||
                    state->tube.centerX.empty() ||
                    state->tube.centerX[0] !=
                        expected[size_t(state->version)]) {
                    torn.fetch_add(1);
                }
                reads.fetch_add(1);
            }
        });
    }
    for (int i = 0; i < kEdits; ++i) {
        model.MoveCenterRing(0, 0.001f, 0.0f);
        model.Sync();
        expected[size_t(model.GetVersion())] = model.Snapshot().centerX[0];
        model.PublishState();
    }
    stop.store(true, std::memory_order_release);
    for (std::thread &t : readers) {
        t.join();
    }
    Check(torn.load() == 0, "state: every acquired state is one version");
    Check(reads.load() > 0, "state: readers ran (" +
                                std::to_string(reads.load()) + " reads)");
    PomadeModelStateRef const last = PomadeAcquireModelState(model);
    Check(last && last->version == model.GetVersion(),
          "state: the newest publish is what a reader sees");
    usdGen::UsdGenEpochDrain();
}

}  // namespace

int
main()
{
    TestMailbox();
    TestTimedPark();
    TestRegistry();
    TestModelState();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
