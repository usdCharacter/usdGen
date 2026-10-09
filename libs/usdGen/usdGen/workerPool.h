// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// workerPool.h — a persistent fork/join pool for the scheduler's parallel
// regions.
//
// TBB's arena pays ~0.2ms per region on this machine (wakeup stagger plus
// the calling thread's lottery: the master joins the arena, so a master
// parked on a slow core drags the span). The pool instead keeps one worker
// per fast CPU parked on a generation counter: dispatch is one
// release-store plus a wake, the master coordinates without computing, and
// small claimed chunks keep a stacked or slow worker from setting the
// wall the way a static split would. Workers spin generously (bench and
// frame gaps never sleep) and then block on a condition variable, so a
// truly idle pool costs no power while hot loops never touch the wake
// path.
//
// Semantics mirror UsdGenWorkDispatcher::ParallelFor: body(0..count-1),
// joined before return, any partition (callers must be partition-agnostic,
// as they already are for TBB). The first worker exception is rethrown on
// the caller after every worker finished its chunk (TBB would cancel
// pending siblings instead; no body throws today outside OOM).
//
// One pool per owner (a scheduler, a Vulkan pipeline): concurrent
// ParallelFor on the same pool from two threads is not supported, exactly
// like concurrent use of one TBB arena from two masters is not attempted.
// Bodies must not re-enter the pool (no nested ParallelFor); nothing does.

#pragma once

#include <cstddef>

namespace usdGen {

class UsdGenWorkerPool
{
public:
    /// Spawns `workers` persistent threads, pinned to the fast-core set
    /// where the topology qualifies (tbbFastCores.h policy). `workers <= 1`
    /// spawns nothing: ParallelFor runs inline on the caller.
    explicit UsdGenWorkerPool(int workers);
    ~UsdGenWorkerPool();

    UsdGenWorkerPool(const UsdGenWorkerPool &) = delete;
    UsdGenWorkerPool &operator=(const UsdGenWorkerPool &) = delete;

    int Workers() const noexcept { return _workers; }

    /// Runs body(i, payload) for i in [0, count), joined before return.
    /// Empty ranges return without dispatching. Rethrows the first worker
    /// exception, if any, after every chunk completed.
    /// claimChunk is the indices per atomic claim (0 selects the default
    /// heuristic, currently ~4): smaller chunks de-quantize big-body
    /// regions (a 40-index quantum strands up to a full chunk per worker
    /// past the last full round), while the default preserves locality
    /// for small locality-sensitive bodies.
    void ParallelFor(size_t count, void (*body)(size_t, void *), void *payload,
                     size_t claimChunk = 0);

private:
    struct State;
    State *_state = nullptr;
    int _workers = 0;
};

} // namespace usdGen
