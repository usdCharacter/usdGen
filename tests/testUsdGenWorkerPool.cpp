// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenWorkerPool — T0: the scheduler's persistent fork/join pool.
//
// Asserted (every platform):
//   0. Every index runs exactly once (disjoint writes over several
//      counts, including 0, 1, and counts below the worker number).
//   1. At least two distinct worker threads run tasks on dispatches
//      large enough that scheduling granularity cannot decide the race
//      (count above 8 per worker; parallelism actually happened).
//   2. A body exception reaches the caller (first error rethrown) after
//      every chunk completed, and the pool stays usable.
//   3. The park path is correct: with USDGEN_POOL_SPIN_US=0 every
//      dispatch parks and wakes through the parking word, on both sides
//      (idle workers and the joining master), including long bodies.
//   4. Repeated dispatches are all correct (generation counter wraps
//      safely over 300 back-to-back regions with gaps).
//   5. A single-wide pool runs inline (no threads needed for Dispatch).
//   6. An explicit claim chunk keeps every-index-once (prime counts,
//      chunk 1 and a chunk past the count).

#include "usdGen/workerPool.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#else
#include <time.h>
#endif

using namespace usdGen;

namespace {

#if defined(_WIN32)
// MSVC has no setenv/unsetenv. _putenv_s covers set; an empty value
// reads back as unset for SpinBudgetUs (empty means default), which is
// all this test's save/restore needs.
inline int PoolSetEnv(char const *name, char const *value) { return _putenv_s(name, value); }
inline int PoolUnsetEnv(char const *name) { return _putenv_s(name, ""); }
#else
inline int PoolSetEnv(char const *name, char const *value) { return ::setenv(name, value, 1); }
inline int PoolUnsetEnv(char const *name) { return ::unsetenv(name); }
#endif

int g_failures = 0;

// CPU time in milliseconds: the calling thread's, and the whole process's.
// Windows: cycle counts (GetThreadTimes is charged per 15.6 ms tick, far too
// coarse), converted with the time-stamp counter rate measured against QPC.
#if defined(_WIN32)
double CyclesPerMs()
{
    static double rate = 0.0;  // test-only, single-threaded use
    if (rate == 0.0) {
        LARGE_INTEGER f, q0, q1;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&q0);
        uint64_t const c0 = __rdtsc();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        uint64_t const c1 = __rdtsc();
        QueryPerformanceCounter(&q1);
        rate = double(c1 - c0) / (double(q1.QuadPart - q0.QuadPart) * 1e3 / double(f.QuadPart));
    }
    return rate;
}
#endif

double ThreadCpuMs()
{
#if defined(_WIN32)
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return double(cycles) / CyclesPerMs();
#else
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
#endif
}

double ProcessCpuMs()
{
#if defined(_WIN32)
    ULONG64 cycles = 0;
    QueryProcessCycleTime(GetCurrentProcess(), &cycles);
    return double(cycles) / CyclesPerMs();
#else
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
#endif
}

struct SlowPayload {
    std::atomic<int> *hits;
    std::atomic<int> *inFlight;
    int sleepMs;
};

void SlowBody(size_t i, void *p)
{
    auto *pl = static_cast<SlowPayload *>(p);
    pl->inFlight->fetch_add(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(pl->sleepMs));
    pl->hits[i].fetch_add(1, std::memory_order_relaxed);
    pl->inFlight->fetch_sub(1);
}

// A region of long bodies: every index once and nothing still running when
// ParallelFor returns. Returns the caller's CPU time spent in the call.
double RunSlowRegion(UsdGenWorkerPool &pool, size_t count, int sleepMs, bool *exact)
{
    std::vector<std::atomic<int>> hits(count);
    for (auto &h : hits) h.store(0);
    std::atomic<int> inFlight{0};
    SlowPayload pl{hits.data(), &inFlight, sleepMs};
    double const cpu0 = ThreadCpuMs();
    pool.ParallelFor(count, SlowBody, &pl, 1);
    double const cpu = ThreadCpuMs() - cpu0;
    bool ok = inFlight.load() == 0;
    for (auto &h : hits) ok = ok && h.load() == 1;
    *exact = ok;
    return cpu;
}

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

struct CountPayload {
    std::atomic<int> *hits;
    std::set<std::thread::id> *threads;
    std::mutex *threadsMutex;
};

void CountBody(size_t i, void *p)
{
    auto *pl = static_cast<CountPayload *>(p);
    pl->hits[i].fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(*pl->threadsMutex);
        pl->threads->insert(std::this_thread::get_id());
    }
    // Pace the claimer: without this a single hot thread can win every
    // fetch_add race while peers are descheduled (oversubscribed CI),
    // which the distribution asserts below would misread as a pool bug.
    std::this_thread::yield();
}

void ThrowBody(size_t i, void *p)
{
    auto *hits = static_cast<std::atomic<int> *>(p);
    hits[i].fetch_add(1, std::memory_order_relaxed);
    if (i == 5) throw std::runtime_error("pool test throw");
}

void CheckCountsChunk(UsdGenWorkerPool &pool, size_t count, size_t chunk,
                      char const *what)
{
    std::vector<std::atomic<int>> hits(count);
    for (auto &h : hits) h.store(0, std::memory_order_relaxed);
    std::set<std::thread::id> threads;
    std::mutex threadsMutex;
    CountPayload pl{count ? hits.data() : nullptr, &threads, &threadsMutex};
    pool.ParallelFor(count, CountBody, &pl, chunk);
    bool once = true;
    for (auto &h : hits)
        if (h.load(std::memory_order_relaxed) != 1) once = false;
    Check(once, std::string("every index runs exactly once (") + what + ")");
    // Spread needs a race longer than scheduler wakeup granularity: a
    // tiny dispatch can finish on the first woken thread while peers are
    // still waking, which the scheduler decides, not the pool. Assert
    // spread only where the pool determines it (enough chunks that
    // staggered peers join the race); exactness is asserted always.
    if (count > size_t(8) * size_t(pool.Workers()) && pool.Workers() > 1)
        Check(threads.size() >= 2,
              std::string("at least two workers ran tasks (") + what + ")");
}

void CheckCounts(UsdGenWorkerPool &pool, size_t count, char const *what)
{
    std::vector<std::atomic<int>> hits(count);
    for (auto &h : hits) h.store(0, std::memory_order_relaxed);
    std::set<std::thread::id> threads;
    std::mutex threadsMutex;
    CountPayload pl{count ? hits.data() : nullptr, &threads, &threadsMutex};
    pool.ParallelFor(count, CountBody, &pl);
    bool once = true;
    for (auto &h : hits)
        if (h.load(std::memory_order_relaxed) != 1) once = false;
    Check(once, std::string("every index runs exactly once (") + what + ")");
    // Spread needs a race longer than scheduler wakeup granularity: a
    // tiny dispatch can finish on the first woken thread while peers are
    // still waking, which the scheduler decides, not the pool. Assert
    // spread only where the pool determines it (enough chunks that
    // staggered peers join the race); exactness is asserted always.
    if (count > size_t(8) * size_t(pool.Workers()) && pool.Workers() > 1)
        Check(threads.size() >= 2,
              std::string("at least two workers ran tasks (") + what + ")");
}

// Prove every worker started and is spinning before a measured
// dispatch: the first woken thread otherwise claims small counts alone
// (cold start), which reads as a pool bug on slow CI runners.
void WarmUpPool(UsdGenWorkerPool &pool)
{
    std::vector<std::atomic<int>> warm(64);
    for (auto &h : warm) h.store(0, std::memory_order_relaxed);
    std::set<std::thread::id> ignore;
    std::mutex ignoreMutex;
    CountPayload pl{warm.data(), &ignore, &ignoreMutex};
    pool.ParallelFor(warm.size(), CountBody, &pl);
}

}  // namespace

int main()
{
    {
        UsdGenWorkerPool pool(8);
        Check(pool.Workers() == 8, "worker count reports");
        WarmUpPool(pool); // all workers started and spinning before measured dispatches
        CheckCounts(pool, 0, "empty");
        CheckCounts(pool, 1, "single");
        CheckCounts(pool, 3, "fewer than workers");
        CheckCounts(pool, 8, "exactly workers");
        CheckCounts(pool, 20000, "many");
    }
    if (g_failures) return 1;

    // Exceptions: the throw escapes to the caller, every sibling chunk
    // still ran (the pool joins instead of cancelling), the throwing
    // worker abandons the rest of its own chunk (here [4,8): indices 6
    // and 7 never run), and the pool stays usable afterwards.
    {
        UsdGenWorkerPool pool(4);
        size_t const count = 64;
        std::vector<std::atomic<int>> hits(count);
        for (auto &h : hits) h.store(0, std::memory_order_relaxed);
        bool threw = false;
        try {
            pool.ParallelFor(count, ThrowBody, hits.data());
        } catch (std::runtime_error const &) {
            threw = true;
        } catch (...) {
        }
        Check(threw, "a body throw reaches the caller");
        bool siblingsRan = true;
        for (size_t i = 0; i < count; ++i) {
            int const want = (i == 6 || i == 7) ? 0 : 1;
            if (hits[i].load(std::memory_order_relaxed) != want)
                siblingsRan = false;
        }
        Check(siblingsRan, "siblings ran; the throwing chunk abandoned its tail");
        CheckCounts(pool, 100, "usable after a throw");
    }
    if (g_failures) return 1;

    // Park path: force every dispatch and every join through the parking word.
    {
        char const *outer = std::getenv("USDGEN_POOL_SPIN_US");
        std::string const saved = outer ? outer : "";
        bool const had = outer != nullptr;
        PoolSetEnv("USDGEN_POOL_SPIN_US", "0");
        {
            UsdGenWorkerPool pool(8);
            CheckCounts(pool, 1000, "forced sleep/wake");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            CheckCounts(pool, 1000, "wake after an idle gap");
            // Many forced-sleep dispatches: each park/wake cycle is a
            // missed-wakeup lottery ticket, so two dispatches pass by
            // luck (the pre-fix pool hung ~1% per sleep dispatch).
            bool slept = true;
            for (int round = 0; round < 100; ++round) {
                std::vector<std::atomic<int>> hits(257);
                for (auto &h : hits) h.store(0, std::memory_order_relaxed);
                std::set<std::thread::id> threads;
                std::mutex threadsMutex;
                CountPayload pl{hits.data(), &threads, &threadsMutex};
                pool.ParallelFor(257, CountBody, &pl);
                for (auto &h : hits)
                    if (h.load(std::memory_order_relaxed) != 1) slept = false;
            }
            Check(slept, "100 forced-sleep dispatches are all exact");
            // Long bodies: the master parks in the join while workers still
            // run, and is woken by the last one.
            bool exact = true;
            for (int round = 0; round < 20 && exact; ++round)
                RunSlowRegion(pool, size_t(8 + round % 5), 1 + round % 3, &exact);
            Check(exact, "forced-park joins over long bodies are exact and complete");
        }
        if (had) PoolSetEnv("USDGEN_POOL_SPIN_US", saved.c_str());
        else PoolUnsetEnv("USDGEN_POOL_SPIN_US");
    }
    if (g_failures) return 1;

    // Generations: many back-to-back regions stay correct.
    {
        UsdGenWorkerPool pool(8);
        size_t const count = 257;
        std::vector<std::atomic<int>> hits(count);
        bool ok = true;
        for (int round = 0; round < 300; ++round) {
            for (auto &h : hits) h.store(0, std::memory_order_relaxed);
            std::set<std::thread::id> threads;
            std::mutex threadsMutex;
            CountPayload pl{hits.data(), &threads, &threadsMutex};
            pool.ParallelFor(count, CountBody, &pl);
            for (auto &h : hits)
                if (h.load(std::memory_order_relaxed) != 1) ok = false;
            if (round == 150)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        Check(ok, "300 back-to-back dispatches are all exact");
    }
    if (g_failures) return 1;

    // The join and the idle wait are bounded spins: a master waiting on a
    // long region parks instead of burning its core for the region, and
    // idle workers park soon after a region instead of spinning for tens of
    // milliseconds. Skipped when the spin budget is overridden.
    if (!std::getenv("USDGEN_POOL_SPIN_US") || !*std::getenv("USDGEN_POOL_SPIN_US")) {
#if defined(_WIN32)
        CyclesPerMs();  // calibrate before anything is measured
#endif
        UsdGenWorkerPool pool(8);
        WarmUpPool(pool);
        bool exact = true;
        double const joinCpu = RunSlowRegion(pool, 8, 60, &exact);
        std::printf("join over a 60 ms region: %.1f ms caller CPU\n", joinCpu);
        Check(exact, "long region is exact and complete");
        Check(joinCpu < 25.0, "the joining master parks instead of spinning through a long region");
        WarmUpPool(pool);
        double const cpu0 = ProcessCpuMs();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        double const idleCpu = ProcessCpuMs() - cpu0;
        std::printf("8 idle workers, 100 ms after a region: %.1f ms process CPU\n", idleCpu);
        Check(idleCpu < 60.0, "idle workers park soon after a region");
    }
    if (g_failures) return 1;

    // Single-wide pools run inline.
    {
        UsdGenWorkerPool pool(1);
        std::set<std::thread::id> threads;
        std::mutex threadsMutex;
        std::vector<std::atomic<int>> hits(16);
        for (auto &h : hits) h.store(0, std::memory_order_relaxed);
        CountPayload pl{hits.data(), &threads, &threadsMutex};
        pool.ParallelFor(16, CountBody, &pl);
        bool once = true;
        for (auto &h : hits)
            if (h.load(std::memory_order_relaxed) != 1) once = false;
        Check(once, "a single-wide pool runs every index");
        Check(threads.size() == 1 &&
                  *threads.begin() == std::this_thread::get_id(),
              "a single-wide pool runs inline on the caller");
    }

    if (g_failures) return 1;

    // Explicit claim chunks keep the contract (prime counts defeat even
    // splits; a chunk past the count still covers every index once, on
    // one worker, since a single claim takes the whole range).
    {
        UsdGenWorkerPool pool(8);
        WarmUpPool(pool); // all workers started and spinning before measured dispatches
        CheckCountsChunk(pool, 101, 1, "chunk 1 over prime 101");
        CheckCountsChunk(pool, 101, 7, "chunk 7 over prime 101");
        {
            std::vector<std::atomic<int>> hits(17);
            for (auto &h : hits) h.store(0, std::memory_order_relaxed);
            std::set<std::thread::id> threads;
            std::mutex threadsMutex;
            CountPayload pl{hits.data(), &threads, &threadsMutex};
            pool.ParallelFor(17, CountBody, &pl, 64);
            bool once = true;
            for (auto &h : hits)
                if (h.load(std::memory_order_relaxed) != 1) once = false;
            Check(once, "every index runs exactly once (chunk past the count)");
        }
    }

    if (g_failures) return 1;
    std::printf("testUsdGenWorkerPool: PASS\n");
    return 0;
}
