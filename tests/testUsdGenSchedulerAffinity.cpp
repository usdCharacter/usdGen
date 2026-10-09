// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenSchedulerAffinity — T0: on heterogeneous big.LITTLE Linux,
// UsdGenScheduler pins its worker-pool threads to max-frequency cores.
// The thread-limit resolution checks run on every platform; the affinity
// checks skip (77) on non-Linux, homogeneous, unreadable-topology, or
// too-few-fast-core hosts.
//
// Asserted (every platform):
//   0. Thread-limit resolution: an explicit count wins, else
//      USDGEN_THREAD_LIMIT, else the product default.
// Asserted (heterogeneous Linux only):
//   1. A parallel region runs tasks on at least two distinct worker
//      threads (parallelism actually happened).
//   2. Every observed worker thread's affinity mask is a nonempty
//      subset of the process-allowed max-frequency set (fast cores
//      only, never widened; one worker per fast CPU).
//   3. The calling thread's affinity is unchanged (never pinned).
//   4. The product default is max(8, fast-core count), and the
//      USDGEN_NO_FAST_CORE_PIN kill switch restores 8.

#include "usdGen/scheduler.h"
#include "usdGen/tbbFastCores.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

using namespace usdGen;

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

#if defined(__linux__) && !defined(__ANDROID__)
long CpuMaxFreqKhz(int cpu)
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

struct TaskSlot {
    pthread_t thread;
    cpu_set_t mask;
};

void Burn()
{
    for (volatile int k = 0; k < 4000; ++k) {
    }
}

void RecordBody(size_t i, void *payload)
{
    Burn();  // keep every arena worker occupied so all of them join
    auto *slots = static_cast<std::vector<TaskSlot> *>(payload);
    TaskSlot slot;
    slot.thread = pthread_self();
    CPU_ZERO(&slot.mask);
    sched_getaffinity(0, sizeof(slot.mask), &slot.mask);
    (*slots)[i] = slot;
}
#endif  // defined(__linux__) && !defined(__ANDROID__)

}  // namespace

int main()
{
    // Resolution precedence runs everywhere (no topology needed): an
    // explicit count wins over the environment, which wins over the
    // default. The outer environment is saved and restored.
    char const *outerLimit = std::getenv("USDGEN_THREAD_LIMIT");
    std::string const savedLimit = outerLimit ? outerLimit : "";
    bool const hadLimit = outerLimit != nullptr;
    setenv("USDGEN_THREAD_LIMIT", "3", 1);
    {
        UsdGenScheduler fromEnv;
        Check(fromEnv.ThreadLimit() == 3, "USDGEN_THREAD_LIMIT=3 sizes a default scheduler");
        UsdGenScheduler explicitWins(5);
        Check(explicitWins.ThreadLimit() == 5, "an explicit count wins over USDGEN_THREAD_LIMIT");
    }
    if (hadLimit) setenv("USDGEN_THREAD_LIMIT", savedLimit.c_str(), 1);
    else unsetenv("USDGEN_THREAD_LIMIT");
    if (g_failures) return 1;

#if !defined(__linux__) || defined(__ANDROID__)
    std::printf("SKIP: fast-core affinity is Linux-only\n");
    return 77;
#else
    // Force-enable even if the outer environment disables the feature: this
    // test owns the scheduler under test.
    setenv("USDGEN_NO_FAST_CORE_PIN", "0", 1);

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        std::printf("SKIP: cannot read process affinity\n");
        return 77;
    }
    long maxFreq = -1;
    int allowedCount = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        ++allowedCount;
        long const freq = CpuMaxFreqKhz(cpu);
        if (freq <= 0) {
            std::printf("SKIP: unreadable CPU topology\n");
            return 77;
        }
        if (freq > maxFreq) maxFreq = freq;
    }
    cpu_set_t fast;
    CPU_ZERO(&fast);
    int fastCount = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (CpuMaxFreqKhz(cpu) == maxFreq) {
            CPU_SET(cpu, &fast);
            ++fastCount;
        }
    }
    if (fastCount == allowedCount) {
        std::printf("SKIP: homogeneous frequencies\n");
        return 77;
    }
    int const workers = 8;
    if (fastCount < workers) {
        std::printf("SKIP: only %d fast cores for %d workers\n", fastCount, workers);
        return 77;
    }

    // The product default follows the independently recomputed fast-core
    // count (never below the 8-thread knee), and the kill switch
    // restores 8 exactly.
    Check(FastCoreCount() == fastCount, "FastCoreCount() matches the recomputed fast set");
    {
        UsdGenScheduler productDefault;
        Check(productDefault.ThreadLimit() == std::max(8, fastCount),
              "a default scheduler spans max(8, fast-core count)");
    }
    setenv("USDGEN_NO_FAST_CORE_PIN", "1", 1);
    Check(FastCoreCount() == 0, "the kill switch empties FastCoreCount()");
    {
        UsdGenScheduler killedDefault;
        Check(killedDefault.ThreadLimit() == 8, "the kill switch restores the 8-thread default");
    }
    setenv("USDGEN_NO_FAST_CORE_PIN", "0", 1);
    if (g_failures) return 1;

    cpu_set_t mainBefore;
    CPU_ZERO(&mainBefore);
    sched_getaffinity(0, sizeof(mainBefore), &mainBefore);
    pthread_t const mainThread = pthread_self();

    UsdGenScheduler scheduler(workers);
    UsdGenWorkDispatcher dispatcher = scheduler.MakeWorkDispatcher();
    size_t const tasks = 20000;
    std::vector<TaskSlot> slots(tasks);
    dispatcher.ParallelFor(tasks, RecordBody, &slots);

    std::set<pthread_t> workerThreads;
    bool masksOk = true;
    for (TaskSlot const &slot : slots) {
        if (pthread_equal(slot.thread, mainThread)) continue;
        workerThreads.insert(slot.thread);
        // Nonempty subset of the fast set: at least one bit set, and
        // every set bit is fast. The pool pins one worker per fast CPU,
        // so masks are singletons in practice; the contract is the subset.
        bool anySet = false, noneSlow = true;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &slot.mask)) continue;
            anySet = true;
            if (!CPU_ISSET(cpu, &fast)) noneSlow = false;
        }
        if (!(anySet && noneSlow)) masksOk = false;
    }
    Check(workerThreads.size() >= 2, "at least two distinct worker threads ran tasks");
    Check(masksOk, "every worker mask is a nonempty subset of the fast-core set");

    cpu_set_t mainAfter;
    CPU_ZERO(&mainAfter);
    sched_getaffinity(0, sizeof(mainAfter), &mainAfter);
    Check(CPU_EQUAL(&mainBefore, &mainAfter) != 0, "calling thread affinity unchanged");

    if (g_failures) return 1;
    std::printf("testUsdGenSchedulerAffinity: PASS\n");
    return 0;
#endif
}
