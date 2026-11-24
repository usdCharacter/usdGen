// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/workerPool.h"
#include "usdGen/parkingWord.h"
#include "usdGen/fastCores.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

namespace usdGen {

namespace {

// Spin budget before a parked worker blocks: bench gaps (~0.5ms),
// interactive frame gaps (16ms), and scheduling hiccups never sleep, so
// the wake path stays out of every hot loop; a truly idle pool still
// blocks instead of burning cores. USDGEN_POOL_SPIN_US overrides (0
// forces the sleep path; the pool test uses it).
long SpinBudgetUs()
{
    char const *env = std::getenv("USDGEN_POOL_SPIN_US");
    if (env && *env) return std::atol(env);
    return 50000;
}

// Spin-hint most iterations, with a periodic true yield so a parked
// spinner (or a joining master) surrenders its core to real work —
// latency-sensitive calls between dispatches (a sub-millisecond pick)
// must not queue behind ten spinning threads. The yield costs ~100ns
// against an uncontended core and returns immediately, so dispatch
// observation stays hot; only genuinely contended cores deschedule the
// spinner, which is exactly when yielding is correct.
void CpuRelax(unsigned spin)
{
    if ((spin & 63) == 0) {
        std::this_thread::yield();
        return;
    }
#if defined(_MSC_VER) && defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

} // namespace

struct UsdGenWorkerPool::State {
    std::vector<std::thread> threads;
    // Dispatch generation: even values are idle. The master publishes
    // body/payload/count/chunk, then release-stores an odd generation and
    // wakes; workers acquire-load it. finished counts workers done with
    // the current odd generation; observed counts workers that parked on
    // the following even marker. The master joins finished, bumps even,
    // then joins observed before the next dispatch, so no worker can be
    // a whole cycle behind when a new payload publishes (a descheduled
    // sleeper delays the next dispatch; it can never miss one).
    // Workers that spun out park on this word (parkingWord.h); the master
    // only makes the wake call when one has.
    std::atomic<uint32_t> epoch{0};
    std::atomic<uint32_t> sleepers{0};
    std::atomic<unsigned> finished{0};
    std::atomic<unsigned> observed{0};
    std::atomic<bool> stop{false};
    void (*body)(size_t, void *) = nullptr;
    void *payload = nullptr;
    size_t count = 0;
    // Dynamic queue: workers claim fixed chunks, so a stacked or slow
    // worker takes fewer chunks instead of setting the wall the way a
    // static split would.
    std::atomic<size_t> next{0};
    size_t chunk = 1;
    // The first worker to throw claims the slot; the master reads it after
    // the finished join, which orders the write.
    std::atomic<bool> errorClaimed{false};
    std::exception_ptr firstError;
    // One fast CPU per worker where the topology qualifies, else empty
    // (workers float). Read once: affinity is process-stable in practice.
    std::vector<int> pinCpus;
    int workers = 0;

    void WorkerMain(int id)
    {
#if defined(__linux__) && !defined(__ANDROID__)
        // One worker per fast CPU: pinning the whole set lets the
        // scheduler stack workers on a few cores for longer than a
        // sub-millisecond region lasts. The target always intersects the
        // thread's current affinity, so taskset, cgroups, and narrower
        // pre-existing pins are respected, never widened; when the
        // 1:1 CPU falls outside the current set the worker takes the
        // intersected set instead.
        if (!pinCpus.empty()) {
            cpu_set_t current, target;
            CPU_ZERO(&current);
            CPU_ZERO(&target);
            if (pthread_getaffinity_np(pthread_self(), sizeof(current),
                                       &current) == 0) {
                if (id < int(pinCpus.size()) &&
                    CPU_ISSET(pinCpus[size_t(id)], &current)) {
                    CPU_SET(pinCpus[size_t(id)], &target);
                } else {
                    for (int cpu : pinCpus)
                        if (CPU_ISSET(cpu, &current)) CPU_SET(cpu, &target);
                }
                if (CPU_COUNT(&target) > 0 && !CPU_EQUAL(&target, &current))
                    pthread_setaffinity_np(pthread_self(), sizeof(target),
                                           &target);
            }
        }
#else
        (void)id;
#endif
        uint32_t seen = 0;
        for (;;) {
            // Spin phase: a dispatch landing here costs no mutex. The
            // clock is sampled sparingly (a vDSO read per iteration
            // would double the spin's core burn while siblings still
            // compute).
            long const spinUs = SpinBudgetUs();
            auto const spinUntil =
                std::chrono::steady_clock::now() + std::chrono::microseconds(spinUs);
            for (unsigned spin = 0;; ++spin) {
                uint32_t const e = epoch.load(std::memory_order_acquire);
                if (e != seen) break;
                if (stop.load(std::memory_order_relaxed)) return;
                if (spinUs <= 0) break;
                if ((spin & 255) == 0 &&
                    std::chrono::steady_clock::now() >= spinUntil)
                    break;
                CpuRelax(spin);
            }
            uint32_t e = epoch.load(std::memory_order_acquire);
            if (e == seen && !stop.load(std::memory_order_relaxed)) {
                // Announce before the final check (both seq_cst): either the
                // master's bump is seen here, or the master sees a sleeper
                // and wakes. The kernel re-compares the word, so a bump
                // between the check and the park returns at once.
                sleepers.fetch_add(1, std::memory_order_seq_cst);
                while ((e = epoch.load(std::memory_order_seq_cst)) == seen &&
                       !stop.load(std::memory_order_relaxed))
                    UsdGenParkWhileEqual(epoch, seen);
                sleepers.fetch_sub(1, std::memory_order_relaxed);
                if (stop.load(std::memory_order_relaxed)) return;
            }
            if (stop.load(std::memory_order_relaxed)) return;
            if (e == seen) continue;  // defensive: re-park on a stale load
            seen = e;
            if (!(e & 1)) {
                // The master's all-done marker: acknowledge it so the
                // next dispatch knows every worker parked here.
                observed.fetch_add(1, std::memory_order_release);
                continue;
            }
            size_t const n = count;
            void (*runBody)(size_t, void *) = body;
            void *runPayload = payload;
            size_t const step = chunk;
            try {
                for (;;) {
                    size_t const lo =
                        next.fetch_add(step, std::memory_order_relaxed);
                    if (lo >= n) break;
                    size_t const hi = std::min(lo + step, n);
                    for (size_t i = lo; i < hi; ++i) runBody(i, runPayload);
                }
            } catch (...) {
                if (!errorClaimed.exchange(true, std::memory_order_acq_rel))
                    firstError = std::current_exception();
            }
            finished.fetch_add(1, std::memory_order_release);
        }
    }
};

UsdGenWorkerPool::UsdGenWorkerPool(int workers) : _workers(workers)
{
    if (workers <= 1) {
        _workers = workers < 1 ? 1 : workers;
        return;  // inline path; no threads, no state
    }
    _state = new State();
    _state->workers = workers;
    _state->pinCpus = FastCoreList(workers);
    _state->threads.reserve(size_t(workers));
    for (int i = 0; i < workers; ++i)
        _state->threads.emplace_back(&State::WorkerMain, _state, i);
}

UsdGenWorkerPool::~UsdGenWorkerPool()
{
    if (!_state) return;
    _state->stop.store(true, std::memory_order_relaxed);
    _state->epoch.fetch_add(1, std::memory_order_seq_cst);
    UsdGenUnparkAll(_state->epoch);
    for (std::thread &t : _state->threads) t.join();
    delete _state;
}

void UsdGenWorkerPool::ParallelFor(size_t count, void (*body)(size_t, void *),
                                   void *payload, size_t claimChunk)
{
    if (count == 0 || !body) return;
    if (!_state) {
        for (size_t i = 0; i < count; ++i) body(i, payload);
        return;
    }
    _state->firstError = nullptr;
    _state->errorClaimed.store(false, std::memory_order_relaxed);
    // Publish the dispatch before the release-store hands it over.
    // Claim chunks of ~4 indices: a stacked worker's straggler bound is
    // one chunk, and the fetch_add traffic is one op per chunk. An
    // explicit claimChunk overrides the heuristic for callers whose
    // bodies are big enough that the quantum strands measurable wall.
    _state->body = body;
    _state->payload = payload;
    _state->count = count;
    _state->chunk = claimChunk != 0 ? claimChunk : std::max<size_t>(
        1, std::min<size_t>(4, (count + size_t(_state->workers) - 1) /
                                   size_t(_state->workers)));
    _state->next.store(0, std::memory_order_relaxed);
    _state->finished.store(0, std::memory_order_relaxed);
    _state->observed.store(0, std::memory_order_relaxed);
    // No wake is lost: the park re-compares the epoch in the kernel, and a
    // worker announces itself in sleepers before its last check (see
    // WorkerMain), so a bump it missed is one this side wakes for.
    auto bump = [s = _state] {
        s->epoch.fetch_add(1, std::memory_order_seq_cst);
        if (s->sleepers.load(std::memory_order_seq_cst) != 0)
            UsdGenUnparkAll(s->epoch);
    };
    bump();  // idle even -> work odd
    for (unsigned spin = 0;
         _state->finished.load(std::memory_order_acquire) !=
         unsigned(_state->workers);
         ++spin)
        CpuRelax(spin);
    // Sleepers parked between their finished++ and this bump need the
    // wake to observe the marker, or the observed join below never ends.
    bump();  // work odd -> idle even
    for (unsigned spin = 0;
         _state->observed.load(std::memory_order_acquire) !=
         unsigned(_state->workers);
         ++spin)
        CpuRelax(spin);
    std::exception_ptr error = std::move(_state->firstError);
    _state->firstError = nullptr;
    if (error) std::rethrow_exception(error);
}

} // namespace usdGen
