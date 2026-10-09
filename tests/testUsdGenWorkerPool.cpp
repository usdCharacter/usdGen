// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenWorkerPool — T0: the scheduler's persistent fork/join pool.
//
// Asserted (every platform):
//   0. Every index runs exactly once (disjoint writes over several
//      counts, including 0, 1, and counts below the worker number).
//   1. At least two distinct worker threads run tasks when the pool is
//      wide (parallelism actually happened).
//   2. A body exception reaches the caller (first error rethrown) after
//      every chunk completed, and the pool stays usable.
//   3. The sleep path is correct: with USDGEN_POOL_SPIN_US=0 every
//      dispatch parks and wakes through the condition variable.
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

struct CountPayload {
    std::atomic<int> *hits;
    std::set<std::thread::id> *threads;
    std::mutex *threadsMutex;
};

void CountBody(size_t i, void *p)
{
    auto *pl = static_cast<CountPayload *>(p);
    pl->hits[i].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(*pl->threadsMutex);
    pl->threads->insert(std::this_thread::get_id());
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
    if (count > 1 && pool.Workers() > 1)
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
    if (count > 1 && pool.Workers() > 1)
        Check(threads.size() >= 2,
              std::string("at least two workers ran tasks (") + what + ")");
}

}  // namespace

int main()
{
    {
        UsdGenWorkerPool pool(8);
        Check(pool.Workers() == 8, "worker count reports");
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

    // Sleep path: force every dispatch through the condition variable.
    {
        char const *outer = std::getenv("USDGEN_POOL_SPIN_US");
        std::string const saved = outer ? outer : "";
        bool const had = outer != nullptr;
        setenv("USDGEN_POOL_SPIN_US", "0", 1);
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
        }
        if (had) setenv("USDGEN_POOL_SPIN_US", saved.c_str(), 1);
        else unsetenv("USDGEN_POOL_SPIN_US");
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
