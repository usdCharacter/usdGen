// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// tbbFastCores.h — fastCores.h placement for a TBB arena. The only TBB piece
// of the CPU engine; it goes when the last TBB arena (GPU backends) does.

#pragma once

#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>

#include <memory>

namespace usdGen {

// Pins `arena`'s workers to FastCoreList(workers). Returns the observer,
// which must stay alive while the arena runs (declare it after the arena so
// it detaches first), or nullptr when the topology does not qualify.
std::unique_ptr<tbb::task_scheduler_observer> ObserveFastCores(
    tbb::task_arena& arena, int workers);

} // namespace usdGen
