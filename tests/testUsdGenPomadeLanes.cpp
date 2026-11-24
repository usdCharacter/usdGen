// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenPomadeLanes — T0: the lock-free handoffs Pomade runs on.
//
//   * The lane handoff (usdGen::UsdGenNewestWins plus a parking word, as
//     the committer and bake worker use them): many producers, one parked
//     consumer; every value is freed exactly once, no wakeup is lost.
//   * PomadeRegistry: concurrent register / activate / attach / publish
//     over the copy-on-write tables; every registered id stays findable.
//   * PomadeModel published state: one authoring thread edits and
//     publishes while readers acquire states and check each one is a
//     consistent copy of a single version.
//   * PomadeBakeWorker: a completion the UI has taken but not swapped yet
//     survives the stale-file sweep of a newer bake; a timed wait returns.
#include "usdGenPomade/pomadeBake.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeModelState.h"
#include "usdGenPomade/pomadeRegistry.h"

#include "usdGen/epoch.h"
#include "usdGen/parkingWord.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
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
std::atomic<int> g_freedUntaken{0};

struct Item : usdGen::UsdGenRefCounted {
    int producer = 0;
    int value = 0;
    bool taken = false;
    Item(int p, int v) : producer(p), value(v) { g_live.fetch_add(1); }
    ~Item() override
    {
        if (!taken) {
            g_freedUntaken.fetch_add(1);
        }
        g_live.fetch_sub(1);
    }
};

// The lane pattern: producers Publish then bump the wake word; the consumer
// reads the word before it looks for work and parks against that value.
void TestLaneHandoff()
{
    constexpr int kProducers = 32;
    constexpr int kValues = 2000;
    g_freedUntaken.store(0);
    size_t taken = 0;
    {
        usdGen::UsdGenNewestWins<Item> box;
        std::atomic<uint32_t> wake{0};
        auto notify = [&] {
            wake.fetch_add(1u, std::memory_order_seq_cst);
            usdGen::UsdGenUnparkAll(wake);
        };
        std::atomic<bool> stop{false};
        std::atomic<int> finalSeen{0};
        std::thread consumer([&] {
            for (;;) {
                uint32_t const seen = wake.load(std::memory_order_seq_cst);
                if (usdGen::UsdGenRef<Item> item = box.Take()) {
                    item->taken = true;
                    ++taken;
                    if (item->producer == -1) {
                        finalSeen.store(1, std::memory_order_release);
                    }
                    continue;
                }
                if (stop.load(std::memory_order_acquire)) {
                    return;
                }
                usdGen::UsdGenParkWhileEqual(wake, seen);
            }
        });
        std::vector<std::thread> producers;
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p] {
                for (int v = 0; v < kValues; ++v) {
                    box.Publish(usdGen::UsdGenMakeRef<Item>(p, v));
                    notify();
                }
            });
        }
        for (std::thread &t : producers) {
            t.join();
        }
        // Newest-wins drops values, so the proof of no lost wakeup is: a
        // final publish after the joins is always taken by the parked
        // consumer (no polling: the consumer is woken or the test hangs).
        box.Publish(usdGen::UsdGenMakeRef<Item>(-1, 0));
        notify();
        auto const deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!finalSeen.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        Check(finalSeen.load() == 1,
              "lane: the parked consumer took the final value");
        stop.store(true, std::memory_order_release);
        notify();
        consumer.join();
    }
    size_t const published = size_t(kProducers) * kValues + 1;
    Check(g_live.load() == 0, "lane: displaced and taken values freed");
    Check(taken > 0 && taken + size_t(g_freedUntaken.load()) == published,
          "lane: every value is either taken or displaced, exactly once "
          "(taken " + std::to_string(taken) + ", displaced " +
              std::to_string(g_freedUntaken.load()) + ")");
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
    std::atomic<int> lostIds{0};
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
                // Only this thread unregisters `id`, so it must be found
                // however the other threads' registrations interleaved.
                if (!registry.SetActive(id)) {
                    lostIds.fetch_add(1);
                }
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
    Check(lostIds.load() == 0,
          "registry: every registered id activates (lost " +
              std::to_string(lostIds.load()) + ")");
    Check(registry.ModelCount() == modelsBefore,
          "registry: every concurrent register was unregistered");
    Check(registry.IndexCount() == indicesBefore,
          "registry: every concurrent attach was detached");
    int calls = 0;
    for (Target const &target : targets) {
        calls += target.calls.load();
    }
    Check(calls > 0, "registry: publishes reached attached targets");

    // Concurrent registrations that stay: every id in the final table is
    // found, and activating it makes that model the active one.
    std::vector<int> ids(size_t(kThreads) * kRounds, 0);
    threads.clear();
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int r = 0; r < kRounds; ++r) {
                size_t const k = size_t(t) * kRounds + size_t(r);
                ids[k] = registry.Register(
                    reinterpret_cast<PomadeModel *>(&storage[k]));
            }
        });
    }
    for (std::thread &t : threads) {
        t.join();
    }
    int unfound = 0;
    for (size_t k = 0; k < ids.size(); ++k) {
        if (!registry.SetActive(ids[k]) ||
            registry.Active() != reinterpret_cast<PomadeModel *>(&storage[k])) {
            ++unfound;
        }
    }
    Check(unfound == 0, "registry: the final table finds every id (" +
                            std::to_string(unfound) + " unfound)");
    registry.SetActive(0);
    for (int id : ids) {
        registry.Unregister(id);
    }
    Check(registry.ModelCount() == modelsBefore,
          "registry: the kept registrations unregister");
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

// A 4x4 quad grid on y = 0 with one rectangular region, enough for the
// worker to bake real files.
struct BakeScene {
    PomadeModel model;
    bool ok = false;
    BakeScene()
    {
        constexpr int n = 4;
        std::vector<float> points;
        std::vector<int> counts, indices;
        for (int ix = 0; ix <= n; ++ix) {
            for (int iz = 0; iz <= n; ++iz) {
                points.insert(points.end(), {float(ix), 0.0f, float(iz)});
            }
        }
        auto pid = [&](int ix, int iz) { return ix * (n + 1) + iz; };
        for (int ix = 0; ix < n; ++ix) {
            for (int iz = 0; iz < n; ++iz) {
                counts.push_back(4);
                indices.insert(indices.end(), {pid(ix, iz), pid(ix, iz + 1),
                                               pid(ix + 1, iz + 1),
                                               pid(ix + 1, iz)});
            }
        }
        if (!model.BindScalp(points, counts, indices)) {
            return;
        }
        PomadeScalpMesh const &mesh = *model.GetScalp();
        auto locate = [&](float x, float z) {
            int const ix = std::min(std::max(int(x), 0), n - 1);
            int const iz = std::min(std::max(int(z), 0), n - 1);
            PomadeHit hit;
            hit.hit = true;
            hit.faceId = ix * n + iz;
            hit.u = z - float(iz);
            hit.v = x - float(ix);
            PomadeFacePosition(mesh, hit.faceId, hit.u, hit.v, &hit.px,
                               &hit.py, &hit.pz);
            hit.ny = 1.0f;
            return hit;
        };
        int const a = model.GraphAddNode(locate(0.0f, 1.0f));
        int const b = model.GraphAddNode(locate(2.0f, 1.0f));
        int const c = model.GraphAddNode(locate(2.0f, 3.0f));
        int const d = model.GraphAddNode(locate(0.0f, 3.0f));
        model.GraphConnect(a, b);
        model.GraphConnect(b, c);
        model.GraphConnect(c, d);
        model.GraphConnect(d, a);
        ok = model.Rasterise();
    }
    PomadeBakeInput Input(std::filesystem::path const &dir) const
    {
        PomadeBakeInput input;
        input.scalp = model.GetScalp();
        input.graph = model.GetGraph();
        input.outDir = dir.string();
        return input;
    }
};

bool WaitSweeps(PomadeBakeWorker const &worker, size_t count)
{
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (worker.SweepCount() < count &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    return worker.SweepCount() >= count;
}

// The UI takes a completion and swaps the live layer to it on a later idle
// turn. A newer bake that lands in between must not sweep the file the UI is
// about to reference (the sweep used to keep only the previous swap and the
// newest completion).
void TestBakeSweepKeepsTaken()
{
    namespace fs = std::filesystem;
    BakeScene scene;
    Check(scene.ok, "bake: the scene binds and rasterises");
    fs::path const dir =
        fs::temp_directory_path() / "testUsdGenPomadeLanesSweep";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    {
        PomadeBakeWorker worker;
        uint64_t version = 0;
        std::string path1, path2, path3;
        worker.Enqueue(1, scene.Input(dir));
        Check(worker.WaitCompleted(1) && WaitSweeps(worker, 1),
              "bake: v1 bakes");
        Check(worker.TakeCompleted(&version, &path1) && version == 1,
              "bake: v1 is taken");
        worker.NoteSwapped(1, path1);
        Check(WaitSweeps(worker, 2), "bake: the v1 swap sweeps");
        worker.Enqueue(2, scene.Input(dir));
        Check(worker.WaitCompleted(2) && WaitSweeps(worker, 3),
              "bake: v2 bakes");
        // Taken, not yet swapped: the UI is between TakeCompleted and the
        // one-attribute swap.
        Check(worker.TakeCompleted(&version, &path2) && version == 2,
              "bake: v2 is taken");
        worker.Enqueue(3, scene.Input(dir));
        Check(worker.WaitCompleted(3) && WaitSweeps(worker, 4),
              "bake: v3 bakes and sweeps");
        Check(fs::is_regular_file(path2, ec),
              "bake: the taken, unswapped v2 survives the v3 sweep");
        Check(fs::is_regular_file(path1, ec),
              "bake: the live v1 survives the v3 sweep");
        // The UI swaps to v2 now; the next sweep frees v1 and keeps v2/v3.
        worker.NoteSwapped(2, path2);
        Check(WaitSweeps(worker, 5), "bake: the v2 swap sweeps");
        Check(!fs::is_regular_file(path1, ec) &&
                  fs::is_regular_file(path2, ec),
              "bake: the v2 swap frees v1 and keeps v2");
        Check(worker.TakeCompleted(&version, &path3) && version == 3 &&
                  fs::is_regular_file(path3, ec),
              "bake: v3 is still on disk");

        auto const t0 = std::chrono::steady_clock::now();
        bool const landed = worker.WaitCompleted(1000, 20);
        auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        Check(!landed && ms >= 10,
              "bake: a timed wait for a version never enqueued times out");
    }
    fs::remove_all(dir, ec);
}

}  // namespace

int
main()
{
    TestLaneHandoff();
    TestRegistry();
    TestModelState();
    TestBakeSweepKeepsTaken();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
