// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/tbbFastCores.h"

#include "pxr/base/tf/getenv.h"
#include "pxr/pxr.h"

#include <cstdio>
#include <optional>

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

#if defined(__linux__) && !defined(__ANDROID__)

namespace {

long ReadCpuMaxFreqKhz(int cpu)
{
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    FILE *f = std::fopen(path, "r");
    if (!f) return -1;
    long freq = -1;
    if (std::fscanf(f, "%ld", &freq) != 1) freq = -1;
    std::fclose(f);
    return freq;
}

struct FastTopology {
    cpu_set_t fast;
    int fastCount = 0;
    int allowedCount = 0;
};

// The topology scan behind FastCores/FastCoreCount: null when disabled,
// unreadable, trivially small, or homogeneous. Uncached (sysfs reads are
// microseconds, arenas are built rarely), so a mid-process
// USDGEN_NO_FAST_CORE_PIN change takes effect on the next query.
std::optional<FastTopology> ReadFastTopology()
{
    if (TfGetenv("USDGEN_NO_FAST_CORE_PIN") == "1") return std::nullopt;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return std::nullopt;
    long maxFreq = -1;
    int allowedCount = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        ++allowedCount;
        long const freq = ReadCpuMaxFreqKhz(cpu);
        if (freq <= 0) return std::nullopt;
        if (freq > maxFreq) maxFreq = freq;
    }
    if (allowedCount <= 1) return std::nullopt;
    FastTopology topo;
    topo.allowedCount = allowedCount;
    CPU_ZERO(&topo.fast);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (ReadCpuMaxFreqKhz(cpu) == maxFreq) {
            CPU_SET(cpu, &topo.fast);
            ++topo.fastCount;
        }
    }
    if (topo.fastCount == topo.allowedCount) return std::nullopt;  // homogeneous
    return topo;
}

std::optional<cpu_set_t> FastCores(int workers)
{
    if (workers <= 0) return std::nullopt;
    auto topo = ReadFastTopology();
    if (!topo) return std::nullopt;
    if (topo->fastCount < workers) return std::nullopt;  // would oversubscribe
    return topo->fast;
}

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
    auto fast = FastCores(workers);
    if (!fast) return nullptr;
    auto observer =
        std::make_unique<FastCoreObserver>(arena, *fast);
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

int FastCoreCount()
{
#if defined(__linux__) && !defined(__ANDROID__)
    auto topo = ReadFastTopology();
    return topo ? topo->fastCount : 0;
#else
    return 0;
#endif
}

std::vector<int> FastCoreList(int workers)
{
    std::vector<int> out;
#if defined(__linux__) && !defined(__ANDROID__)
    auto fast = FastCores(workers);
    if (!fast) return out;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, fast)) out.push_back(cpu);
#else
    (void)workers;
#endif
    return out;
}

} // namespace usdGen
