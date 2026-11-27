// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — a serial lane: at most one pass at a time on the process
// scheduler, no kick lost. Kick is wait-free from any thread, including a
// device stream callback; it never runs the pass inline.
#pragma once

#include "usdGen/api/export.h"
#include "usdGen/api/taskScheduler.h"

#include <atomic>
#include <cstdint>

namespace usdGen {

// Not class-exported: under dllimport &_Run would not be a constant
// expression on MSVC and the lane would lose constant initialization.
class UsdGenSerialLane : private UsdGenTask {
public:
    using Pass = void (*)(void *user) noexcept;
    /// Constant-initialized: a namespace-scope lane outlives late callbacks at exit.
    constexpr UsdGenSerialLane(Pass pass, void *user) noexcept
        : UsdGenTask(&UsdGenSerialLane::_Run), _pass(pass), _user(user) {}
    UsdGenSerialLane(UsdGenSerialLane const &) = delete;
    UsdGenSerialLane &operator=(UsdGenSerialLane const &) = delete;
    /// Schedules a pass that starts after this call.
    USDGEN_CORE_API void Kick() noexcept;
    /// Passes run so far.
    uint64_t Passes() const noexcept { return _passes.load(std::memory_order_relaxed); }

private:
    static void _Run(UsdGenTask *task) noexcept;  // passes until no kick is pending
    static constexpr uint32_t kScheduled = 1u, kDirty = 2u;
    Pass _pass;
    void *_user;
    std::atomic<uint32_t> _state{0};
    std::atomic<uint64_t> _passes{0};
};

}  // namespace usdGen
