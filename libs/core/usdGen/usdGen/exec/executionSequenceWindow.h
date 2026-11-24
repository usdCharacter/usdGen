#ifndef USDGEN_EXECUTION_SEQUENCE_WINDOW_H
#define USDGEN_EXECUTION_SEQUENCE_WINDOW_H

#include <atomic>
#include <cstdint>
#include <vector>

namespace usdGen {

// Bounded contiguous sequence admission. TryIssue may be called concurrently;
// Complete belongs to one externally scheduled owner. The ring is owner-only:
// frontend code may read counters and issue, but never examines completion
// slots. CAS reserves an available sequence and never waits for another call.
class UsdGenExecutionSequenceWindow {
public:
    explicit UsdGenExecutionSequenceWindow(uint64_t capacity);
    uint64_t TryIssue() noexcept;
    // Returns false for stale, future, or duplicate completion. A successful
    // out-of-order completion is retained until all earlier sequences finish.
    bool Complete(uint64_t sequence) noexcept;
    uint64_t LastIssued() const noexcept;
    uint64_t CompletedThrough() const noexcept;
    uint64_t Capacity() const noexcept;

private:
    const uint64_t capacity_;
    std::vector<uint64_t> completed_;
    std::atomic<uint64_t> lastIssued_{0};
    std::atomic<uint64_t> completedThrough_{0};
};

} // namespace usdGen

#endif
