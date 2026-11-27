// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — the engine task scheduler. Workers pull intrusive tasks from a
// bounded MPMC ring (Vyukov) with a Treiber-stack overflow, and park on a
// parkingWord when idle; Post wakes one only when a worker sleeps, so a busy
// scheduler makes no system call. No locks anywhere.
//
//   struct MyTask : UsdGenTask { ... };   // taskRun set, owned by the poster
//   scheduler.Post(&task);                // never blocks
//
// - A task must stay alive until its run starts and must not be posted again
//   while queued. Runs are noexcept.
// - Serial lanes, pending counts and fork/join regions are built on Post.
// - Destruction stops and joins the workers after the queue drains. It never
//   joins the calling worker, and during process detach it joins nothing.
#pragma once

#include "usdGen/api/export.h"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace usdGen {

/// Intrusive unit of work.
struct UsdGenTask {
    using Run = void (*)(UsdGenTask *) noexcept;
    explicit constexpr UsdGenTask(Run run = nullptr) noexcept : taskRun(run) {}
    UsdGenTask *taskNext = nullptr;
    Run taskRun = nullptr;
};

class USDGEN_API UsdGenTaskScheduler {
public:
    /// Spawns max(1, workers) threads.
    explicit UsdGenTaskScheduler(int workers);
    ~UsdGenTaskScheduler();
    UsdGenTaskScheduler(UsdGenTaskScheduler const &) = delete;
    UsdGenTaskScheduler &operator=(UsdGenTaskScheduler const &) = delete;

    int Workers() const noexcept;
    /// Queues `task`; never blocks or fails.
    void Post(UsdGenTask *task) noexcept;
    /// True on one of this scheduler's worker threads.
    bool OnWorker() const noexcept;

    /// Process-wide scheduler, one worker per hardware thread. Made on first
    /// use and never destroyed.
    static UsdGenTaskScheduler &Process();

    struct State;  // private to taskScheduler.cpp

private:
    State *_state = nullptr;
};

/// Outstanding work a boundary waits for. Wait parks until the count is
/// zero; Release wakes only when someone is parked. The waking Release reads
/// nothing after its decrement, so a waiter may free the counter as soon as
/// it sees zero (the wake is keyed by address only).
class USDGEN_API UsdGenPendingCount {
public:
    void Add(uint32_t n = 1) noexcept { _word.fetch_add(n, std::memory_order_seq_cst); }
    void Release(uint32_t n = 1) noexcept;
    /// Parks until the count reaches zero at some instant.
    void Wait() noexcept;
    uint32_t Count() const noexcept { return _word.load(std::memory_order_acquire) & ~kWaiter; }

private:
    static constexpr uint32_t kWaiter = 0x80000000u;
    std::atomic<uint32_t> _word{0};
};

/// body(i, payload) for i in [0, count) on the process scheduler; the caller
/// runs chunks too and returns once every index ran. Helpers that start after
/// the range is exhausted leave without touching it, so nesting is safe.
/// Rethrows the first body exception after every started chunk finished.
/// maxWorkers caps the threads used (0: the scheduler's worker count + 1).
USDGEN_API void UsdGenParallelFor(size_t count, void (*body)(size_t, void *), void *payload,
                       size_t maxWorkers = 0);

/// Threads UsdGenParallelFor may use: the process workers plus the caller.
USDGEN_API int UsdGenParallelForConcurrency();

/// UsdGenParallelFor as a block runner (usdGenMath::rbf::ParallelFor::run):
/// body(user, i, i + 1) for i in [0, count). ctx is unused.
USDGEN_API void UsdGenParallelForBlocks(void *ctx, size_t count,
                             void (*body)(void *user, size_t begin, size_t end), void *user);

}  // namespace usdGen
