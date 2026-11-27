// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/exec/fastCores.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

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

// The topology scan: null when disabled, unreadable, trivially small, or
// homogeneous. Uncached (sysfs reads are microseconds, worker sets are built
// rarely), so a mid-process USDGEN_NO_FAST_CORE_PIN change takes effect on
// the next query.
std::optional<FastTopology> ReadFastTopology()
{
    char const *disabled = std::getenv("USDGEN_NO_FAST_CORE_PIN");
    if (disabled && std::strcmp(disabled, "1") == 0) return std::nullopt;
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

} // namespace

#endif // defined(__linux__) && !defined(__ANDROID__)

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
    if (workers <= 0) return out;
    auto topo = ReadFastTopology();
    if (!topo || topo->fastCount < workers) return out;  // would oversubscribe
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &topo->fast)) out.push_back(cpu);
#else
    (void)workers;
#endif
    return out;
}

} // namespace usdGen
