// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/serialLane.h"

namespace usdGen {

// SCHEDULED while a pass is queued or running; DIRTY when a kick arrived
// since the current pass began, so the runner passes again.
void UsdGenSerialLane::Kick() noexcept
{
    uint32_t const old = _state.fetch_or(kScheduled | kDirty, std::memory_order_acq_rel);
    if (!(old & kScheduled)) UsdGenTaskScheduler::Process().Post(this);
}

void UsdGenSerialLane::_Run(UsdGenTask *task) noexcept
{
    auto *lane = static_cast<UsdGenSerialLane *>(task);
    lane->_state.fetch_and(~kDirty, std::memory_order_acq_rel);
    for (;;) {
        lane->_passes.fetch_add(1, std::memory_order_relaxed);
        lane->_pass(lane->_user);
        uint32_t s = kScheduled;
        if (lane->_state.compare_exchange_strong(s, 0, std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
            return;
        lane->_state.fetch_and(~kDirty, std::memory_order_acq_rel);  // kicked meanwhile
    }
}

}  // namespace usdGen
