// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — parking on a 32-bit word, the engine's only blocking primitive.
// A futex on Linux, WaitOnAddress on Windows. Waiting compares the word in
// the kernel, so a wake that lands between the caller's check and its park
// is never lost, and no mutex or condition variable is involved.
#pragma once

#include "usdGen/api/export.h"
#include <atomic>
#include <cstdint>

namespace usdGen {

/// Block while `word` still holds `expected`. May return spuriously; callers
/// re-check in a loop.
USDGEN_API void UsdGenParkWhileEqual(std::atomic<uint32_t> &word, uint32_t expected);

/// As UsdGenParkWhileEqual, giving up after about `timeoutMs`. False when it
/// timed out; true on a wake or a changed word (or spuriously).
USDGEN_API bool UsdGenParkWhileEqualFor(std::atomic<uint32_t> &word, uint32_t expected,
                             uint32_t timeoutMs);

/// Wake every thread parked on `word`. Call after changing it.
USDGEN_API void UsdGenUnparkAll(std::atomic<uint32_t> &word);

/// Wake at most one thread parked on `word`. Call after changing it.
USDGEN_API void UsdGenUnparkOne(std::atomic<uint32_t> &word);

/// The idle-wait protocol for "sleep until there may be work": a waiter
/// announces itself, re-checks its condition, then parks; a notifier makes
/// the condition true, then notifies, which makes a system call only when a
/// waiter announced. The announce and the notifier's sleeper check are both
/// behind seq_cst fences (a store-load pair: required on ARM64 as on x64),
/// so either the waiter's re-check sees the work or the notifier sees the
/// waiter. The kernel re-compares the word, so a notify between the
/// re-check and the park is not lost.
///
///   uint32_t const token = wake.Announce();
///   if (TryTakeWork()) { wake.Cancel(); ... } else wake.Park(token);
///   // notifier: publish work; wake.NotifyOne();
class UsdGenWakeWord {
public:
    uint32_t Announce() noexcept
    {
        uint32_t const token = _word.load(std::memory_order_acquire);
        _sleepers.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return token;
    }
    void Cancel() noexcept { _sleepers.fetch_sub(1, std::memory_order_relaxed); }
    /// Parks until a notify after Announce (or spuriously), then withdraws.
    void Park(uint32_t token) noexcept
    {
        UsdGenParkWhileEqual(_word, token);
        _sleepers.fetch_sub(1, std::memory_order_relaxed);
    }
    void NotifyOne() noexcept
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (_sleepers.load(std::memory_order_relaxed)) {
            _word.fetch_add(1, std::memory_order_release);
            UsdGenUnparkOne(_word);
        }
    }
    void NotifyAll() noexcept
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (_sleepers.load(std::memory_order_relaxed)) {
            _word.fetch_add(1, std::memory_order_release);
            UsdGenUnparkAll(_word);
        }
    }
    /// Unconditional wake of everyone (shutdown).
    void WakeAll() noexcept
    {
        _word.fetch_add(1, std::memory_order_seq_cst);
        UsdGenUnparkAll(_word);
    }

private:
    std::atomic<uint32_t> _word{0};
    std::atomic<uint32_t> _sleepers{0};
};

}  // namespace usdGen
