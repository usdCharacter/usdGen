#include "usdGen/executionSequenceWindow.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace usdGen {

UsdGenExecutionSequenceWindow::UsdGenExecutionSequenceWindow(uint64_t capacity)
    : capacity_(capacity)
{
    if (!capacity || capacity > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::invalid_argument("execution sequence window capacity is invalid");
    if (capacity > static_cast<uint64_t>(completed_.max_size()))
        throw std::length_error("execution sequence window capacity is too large");
    completed_.resize(static_cast<std::size_t>(capacity), 0);
}

uint64_t UsdGenExecutionSequenceWindow::TryIssue() noexcept {
    auto issued = lastIssued_.load(std::memory_order_relaxed);
    for (;;) {
        const auto through = completedThrough_.load(std::memory_order_acquire);
        // A concurrent issuer/owner pair may have advanced both counters
        // after this caller's local issued load. Refresh rather than report a
        // false full result; this is a CAS reservation retry, never a wait.
        if (issued < through) {
            issued = lastIssued_.load(std::memory_order_acquire);
            continue;
        }
        if (issued == UINT64_MAX || issued - through >= capacity_)
            return 0;
        if (lastIssued_.compare_exchange_weak(issued, issued + 1,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed))
            return issued + 1;
    }
}

bool UsdGenExecutionSequenceWindow::Complete(uint64_t sequence) noexcept {
    auto through = completedThrough_.load(std::memory_order_relaxed);
    const auto issued = lastIssued_.load(std::memory_order_acquire);
    if (sequence <= through || sequence > issued) return false;

    auto& slot = completed_[static_cast<std::size_t>(sequence % capacity_)];
    if (slot == sequence) return false;
    // The release-store below clears retired slots before reuse becomes
    // visible; therefore a non-duplicate valid completion owns an empty slot.
    slot = sequence;

    while (through != UINT64_MAX) {
        const auto next = through + 1;
        auto& nextSlot = completed_[static_cast<std::size_t>(next % capacity_)];
        if (nextSlot != next) break;
        nextSlot = 0;
        through = next;
    }
    // Clear all retired slots before exposing capacity to concurrent issuers.
    completedThrough_.store(through, std::memory_order_release);
    return true;
}

uint64_t UsdGenExecutionSequenceWindow::LastIssued() const noexcept {
    return lastIssued_.load(std::memory_order_acquire);
}
uint64_t UsdGenExecutionSequenceWindow::CompletedThrough() const noexcept {
    return completedThrough_.load(std::memory_order_acquire);
}
uint64_t UsdGenExecutionSequenceWindow::Capacity() const noexcept { return capacity_; }

} // namespace usdGen
