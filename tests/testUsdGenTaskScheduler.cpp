// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Engine task scheduler: multi-producer posting, ring overflow, pending-count
// parking, fork/join regions (nested, concurrent, throwing) and a 64-thread
// pipeline / task-graph submit stress.
#include "usdGen/executionPipeline.h"
#include "usdGen/executionTaskGraph.h"
#include "usdGen/taskScheduler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

struct CountTask : UsdGenTask {
    CountTask() noexcept : UsdGenTask(&Run) {}
    std::atomic<int> *ran = nullptr;
    UsdGenPendingCount *pending = nullptr;
    static void Run(UsdGenTask *t) noexcept {
        auto *self = static_cast<CountTask *>(t);
        self->ran->fetch_add(1, std::memory_order_relaxed);
        self->pending->Release();
    }
};

int TestMultiProducer()
{
    UsdGenTaskScheduler scheduler(4);
    constexpr int kThreads = 64, kPerThread = 500;
    std::vector<CountTask> tasks(kThreads * kPerThread);
    std::atomic<int> ran{0};
    UsdGenPendingCount pending;
    pending.Add(uint32_t(tasks.size()));
    std::vector<std::thread> producers;
    for (int t = 0; t != kThreads; ++t)
        producers.emplace_back([&, t] {
            for (int i = 0; i != kPerThread; ++i) {
                CountTask &task = tasks[size_t(t * kPerThread + i)];
                task.ran = &ran;
                task.pending = &pending;
                scheduler.Post(&task);
            }
        });
    for (auto &p : producers) p.join();
    pending.Wait();
    CHECK(ran.load() == kThreads * kPerThread);
    CHECK(pending.Count() == 0);
    return 0;
}

struct GateTask : UsdGenTask {
    GateTask() noexcept : UsdGenTask(&Run) {}
    std::atomic<bool> *open = nullptr;
    std::atomic<bool> *entered = nullptr;
    static void Run(UsdGenTask *t) noexcept {
        auto *self = static_cast<GateTask *>(t);
        self->entered->store(true);
        while (!self->open->load()) std::this_thread::yield();  // test rendezvous only
    }
};

int TestOverflowAndDrainOnDestroy()
{
    std::atomic<int> ran{0};
    UsdGenPendingCount pending;
    std::vector<CountTask> tasks(5000);  // well past the 1024-entry ring
    std::atomic<bool> open{false}, entered{false};
    GateTask gate;
    gate.open = &open;
    gate.entered = &entered;
    {
        UsdGenTaskScheduler scheduler(1);
        scheduler.Post(&gate);
        while (!entered.load()) std::this_thread::yield();
        pending.Add(uint32_t(tasks.size()));
        for (auto &t : tasks) {
            t.ran = &ran;
            t.pending = &pending;
            scheduler.Post(&t);
        }
        open.store(true);
        // Destruction runs everything still queued before joining.
    }
    CHECK(ran.load() == int(tasks.size()));
    CHECK(pending.Count() == 0);
    return 0;
}

int TestPendingCountStress()
{
    UsdGenPendingCount pending;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> waits{0};
    std::thread waiter([&] {
        while (!stop.load()) {
            pending.Wait();
            waits.fetch_add(1);
        }
    });
    std::vector<std::thread> workers;
    for (int t = 0; t != 64; ++t)
        workers.emplace_back([&] {
            for (int i = 0; i != 20000; ++i) {
                pending.Add();
                pending.Release();
            }
        });
    for (auto &w : workers) w.join();
    stop.store(true);
    pending.Add();
    pending.Release();
    waiter.join();
    CHECK(pending.Count() == 0);
    pending.Wait();  // idle: returns at once
    CHECK(waits.load() > 0);
    return 0;
}

int TestParallelFor()
{
    constexpr size_t kCount = 100000;
    std::vector<std::atomic<int>> hits(kCount);
    for (auto &h : hits) h.store(0);
    UsdGenParallelFor(kCount, [](size_t i, void *p) {
        (*static_cast<std::vector<std::atomic<int>> *>(p))[i].fetch_add(1);
    }, &hits);
    for (auto &h : hits) CHECK(h.load() == 1);

    // Concurrent callers, each nesting a region inside its body.
    std::atomic<int> nested{0};
    std::vector<std::thread> callers;
    for (int t = 0; t != 16; ++t)
        callers.emplace_back([&] {
            for (int r = 0; r != 20; ++r)
                UsdGenParallelFor(8, [](size_t, void *p) {
                    UsdGenParallelFor(16, [](size_t, void *q) {
                        static_cast<std::atomic<int> *>(q)->fetch_add(1);
                    }, p);
                }, &nested);
        });
    for (auto &c : callers) c.join();
    CHECK(nested.load() == 16 * 20 * 8 * 16);

    bool threw = false;
    try {
        UsdGenParallelFor(1000, [](size_t i, void *) {
            if (i == 517) throw std::runtime_error("body failure");
        }, nullptr);
    } catch (std::runtime_error const &) { threw = true; }
    CHECK(threw);
    CHECK(UsdGenParallelForConcurrency() >= 2);
    return 0;
}

int TestPipelineSubmitStress()
{
    UsdGenExecutionRuntime runtime(4);
    UsdGenExecutionPipeline pipeline(runtime, 1u << 16, 1u << 16);
    std::atomic<int> completions{0}, commands{0};
    std::atomic<uint64_t> published{0};
    std::atomic<bool> orderError{false};
    std::vector<std::thread> producers;
    for (int t = 0; t != 64; ++t)
        producers.emplace_back([&] {
            for (int i = 0; i != 50; ++i) {
                pipeline.Submit([&](UsdGenExecutionPipeline::Cancellation const &cancel)
                                    -> UsdGenExecutionPipeline::Publish {
                    return [&, epoch = cancel.epoch] {
                        if (published.exchange(epoch) >= epoch) orderError = true;
                    };
                }, [&](uint64_t, UsdGenExecutionPipeline::Outcome, std::exception_ptr) {
                    completions.fetch_add(1);
                });
                pipeline.PostCommand([&] { commands.fetch_add(1); });
            }
        });
    std::thread drainer([&] { for (int i = 0; i != 50; ++i) pipeline.Drain(); });
    for (auto &p : producers) p.join();
    drainer.join();
    pipeline.Drain();
    CHECK(completions.load() == 64 * 50);
    CHECK(commands.load() == 64 * 50);
    CHECK(!orderError.load());
    CHECK(published.load() == pipeline.AcceptedEpoch());
    return 0;
}

int TestTaskGraphSubmitStress()
{
    UsdGenExecutionRuntime runtime(2);
    auto graph = UsdGenExecutionTaskGraph::GetOrCreate(
        runtime, "schedulerStress", 0, {256, 4096, 16384, 8, 3, 32});
    CHECK(graph);
    std::atomic<int> ran{0}, completed{0}, admitted{0};
    std::vector<std::thread> producers;
    for (int t = 0; t != 64; ++t)
        producers.emplace_back([&, t] {
            for (int i = 0; i != 40; ++i) {
                UsdGenExecutionTaskGraph::Job job;
                job.requestClass = (t & 1) ? UsdGenExecutionTaskGraph::RequestClass::Background
                                           : UsdGenExecutionTaskGraph::RequestClass::Interactive;
                job.tasks.resize(3);
                for (auto &task : job.tasks)
                    task.run = [&](auto const &, auto done) { ran.fetch_add(1); done([] {}, {}); };
                job.tasks[1].dependencies = {0};
                job.tasks[2].dependencies = {0, 1};
                job.completion = [&](auto publish, std::exception_ptr error) {
                    if (!error && publish) publish();
                    completed.fetch_add(1);
                };
                // Admission is bounded: retry a rejected job after a yield.
                while (!graph->Submit(std::move(job))) {
                    std::this_thread::yield();
                    job = {};
                    job.tasks.resize(3);
                    for (auto &task : job.tasks)
                        task.run = [&](auto const &, auto done) { ran.fetch_add(1); done([] {}, {}); };
                    job.tasks[1].dependencies = {0};
                    job.tasks[2].dependencies = {0, 1};
                    job.completion = [&](auto publish, std::exception_ptr error) {
                        if (!error && publish) publish();
                        completed.fetch_add(1);
                    };
                }
                admitted.fetch_add(1);
            }
        });
    for (auto &p : producers) p.join();
    graph->Drain();
    CHECK(admitted.load() == 64 * 40);
    CHECK(completed.load() == admitted.load());
    CHECK(ran.load() == 3 * admitted.load());
    auto usage = graph->GetAdmissionUsage();
    CHECK(usage.jobs == 0 && usage.tasks == 0 && usage.dependencyEdges == 0);
    return 0;
}

}  // namespace

int main()
{
    if (TestMultiProducer()) return 1;
    if (TestOverflowAndDrainOnDestroy()) return 1;
    if (TestPendingCountStress()) return 1;
    if (TestParallelFor()) return 1;
    if (TestPipelineSubmitStress()) return 1;
    if (TestTaskGraphSubmitStress()) return 1;
    std::printf("testUsdGenTaskScheduler: PASS\n");
    return 0;
}
