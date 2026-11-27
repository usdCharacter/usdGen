// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGenPomade — the handoff between the UI thread and a Pomade lane (the
// committer and the bake worker). No locks: requests and results move
// through newest-wins mailboxes, and an idle lane parks on a wake word
// (usdGen/parkingWord.h). Internal to usdGenPomade's sources.
#ifndef USDGEN_POMADE_LANE_H
#define USDGEN_POMADE_LANE_H

#include "usdGen/exec/parkingWord.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace usdGenPomade {

// Single-slot newest-wins mailbox: Publish displaces (and frees) whatever
// the consumer has not taken yet; Take empties the slot. Values are never
// read in place, so no reclamation scheme is needed: whoever exchanges a
// pointer out of the slot owns it alone.
template <class T>
class PomadeMailbox {
public:
    PomadeMailbox() = default;
    ~PomadeMailbox() { delete _slot.load(std::memory_order_acquire); }
    PomadeMailbox(PomadeMailbox const &) = delete;
    PomadeMailbox &operator=(PomadeMailbox const &) = delete;

    void Publish(std::unique_ptr<T> value)
    {
        delete _slot.exchange(value.release(), std::memory_order_acq_rel);
    }
    std::unique_ptr<T> Take()
    {
        return std::unique_ptr<T>(
            _slot.exchange(nullptr, std::memory_order_acq_rel));
    }
    bool Empty() const
    {
        return _slot.load(std::memory_order_acquire) == nullptr;
    }

private:
    std::atomic<T *> _slot{nullptr};
};

// A sequence word a lane parks on. A waiter reads Sequence() BEFORE
// checking for work and parks against that value, so a Notify that lands
// in between changes the word and the park returns at once.
class PomadeWakeWord {
public:
    uint32_t Sequence() const { return _word.load(std::memory_order_acquire); }
    void Notify()
    {
        _word.fetch_add(1u, std::memory_order_acq_rel);
        usdGen::UsdGenUnparkAll(_word);
    }
    void Park(uint32_t seen) { usdGen::UsdGenParkWhileEqual(_word, seen); }
    // False once `deadline` has passed.
    bool ParkUntil(uint32_t seen,
                   std::chrono::steady_clock::time_point deadline)
    {
        auto const now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }
        auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - now).count() + 1;
        usdGen::UsdGenParkWhileEqualFor(
            _word, seen, uint32_t(ms > 0x7fffffff ? 0x7fffffff : ms));
        return true;
    }

private:
    std::atomic<uint32_t> _word{0};
};

}  // namespace usdGenPomade

#endif  // USDGEN_POMADE_LANE_H
