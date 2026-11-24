// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/tbbFastCores.h"

#include "usdGen/fastCores.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

namespace usdGen {

#if defined(__linux__) && !defined(__ANDROID__)

namespace {

class FastCoreObserver : public tbb::task_scheduler_observer {
public:
    FastCoreObserver(tbb::task_arena &arena, cpu_set_t fast)
        : tbb::task_scheduler_observer(arena), _fast(fast)
    {
    }

    void on_scheduler_entry(bool isWorker) override
    {
        if (!isWorker) return;  // never pin the calling (application) thread
        thread_local cpu_set_t pinnedTo;
        thread_local bool havePinned = false;
        if (havePinned && CPU_EQUAL(&pinnedTo, &_fast)) return;
        cpu_set_t current, target;
        CPU_ZERO(&current);
        CPU_ZERO(&target);
        if (pthread_getaffinity_np(pthread_self(), sizeof(current), &current) != 0)
            return;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &current) && CPU_ISSET(cpu, &_fast))
                CPU_SET(cpu, &target);
        if (CPU_COUNT(&target) == 0) return;
        // The target always intersects the thread's current affinity, so
        // taskset, cgroups, and narrower pre-existing pins are respected,
        // never widened.
        if (!CPU_EQUAL(&target, &current) &&
            pthread_setaffinity_np(pthread_self(), sizeof(target), &target) != 0)
            return;
        pinnedTo = target;
        havePinned = true;
    }

private:
    cpu_set_t _fast;
};

} // namespace

#endif // defined(__linux__) && !defined(__ANDROID__)

std::unique_ptr<tbb::task_scheduler_observer> ObserveFastCores(
    tbb::task_arena& arena, int workers)
{
#if defined(__linux__) && !defined(__ANDROID__)
    std::vector<int> const list = FastCoreList(workers);
    if (list.empty()) return nullptr;
    cpu_set_t fast;
    CPU_ZERO(&fast);
    for (int cpu : list) CPU_SET(cpu, &fast);
    auto observer = std::make_unique<FastCoreObserver>(arena, fast);
    // Local observers activate only while attached, so observe from
    // inside the arena; late-joining workers fire on entry.
    arena.execute([&]() { observer->observe(true); });
    if (!observer->is_observing()) return nullptr;
    return observer;
#else
    (void)arena;
    (void)workers;
    return nullptr;
#endif
}

} // namespace usdGen
