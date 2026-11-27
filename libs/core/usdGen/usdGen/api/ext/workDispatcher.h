// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The work dispatcher an op's Capture() gets for intra-node parallelism (03 §8.1).
#pragma once
#include "usdGen/api/export.h"

#include <cstddef>

namespace usdGen {

class UsdGenWorkerPool;

class UsdGenWorkDispatcher
{
public:
    explicit UsdGenWorkDispatcher(UsdGenWorkerPool *pool) : _pool(pool) {}
    /// Run a per-index body 0..count-1 over the scheduler's worker pool
    /// (capture phase). A null pool runs the body inline, serially.
    /// claimChunk overrides the pool's claim heuristic (0 keeps it); see
    /// UsdGenWorkerPool::ParallelFor.
    USDGEN_CORE_API void ParallelFor(size_t count, void (*body)(size_t, void *), void *payload,
                                     size_t claimChunk = 0);
    /// Pool worker count (1 when null/serial), so Capture() can size
    /// partitions instead of oversubscribing ranges onto one worker.
    USDGEN_CORE_API int MaxConcurrency() const;
private:
    UsdGenWorkerPool *_pool;
};

}  // namespace usdGen
