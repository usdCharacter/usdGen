// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// queueTimeline.h — one timeline semaphore per VkQueue. Every submission
// through it signals the next value, so completion is one monotonic 64-bit
// number: a non-blocking check compares it, nothing is pooled, and nothing
// is reset. Replaces per-submit VkFences.
//
// vkQueueSubmit needs its queue externally synchronized and timeline values
// must be signaled in increasing order, so submissions are flat-combined:
// a producer pushes its request on a lock-free stack and whichever thread
// holds the submit role pops the whole stack, assigns values in arrival
// order and submits. A producer that finds the role taken parks on its own
// request word until the submitter has handled it (one vkQueueSubmit call
// at most); nobody spins and nobody waits for a lock holder that is idle.
// Values are assigned only to submissions that succeed, so a failed submit
// never leaves a hole the timeline cannot pass.
#ifndef USDGEN_VULKAN_QUEUE_TIMELINE_H
#define USDGEN_VULKAN_QUEUE_TIMELINE_H

#include "usdGenVulkan/export.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace usdGen::vulkan {

class USDGEN_VULKAN_API VulkanQueueTimeline final {
public:
    /// Creates the semaphore (initial value 0). The device must have the
    /// timelineSemaphore feature enabled.
    static std::unique_ptr<VulkanQueueTimeline> Create(VkDevice device, VkQueue queue,
                                                       VkResult* result = nullptr);
    /// Destroys the semaphore; the caller proves the queue idle for it.
    ~VulkanQueueTimeline();
    VulkanQueueTimeline(VulkanQueueTimeline const&) = delete;
    VulkanQueueTimeline& operator=(VulkanQueueTimeline const&) = delete;

    /// Most command buffers one Submit call takes.
    static constexpr uint32_t kMaxBatches = 32;

    /// Submits `count` (1..kMaxBatches) command buffers as `count` consecutive batches; batch
    /// i signals first + i. Returns the value of the last batch (wait or
    /// check it to prove the whole call), or 0 with *result set when the
    /// submit failed (nothing was queued then). Thread-safe; never spins.
    uint64_t Submit(VkCommandBuffer const* commands, uint32_t count,
                    VkResult* result = nullptr) noexcept;

    /// Submits one empty batch that signals the next value. Same-queue
    /// ordering makes it prove every earlier submission on this queue,
    /// timeline or not. Returns the value, or 0 with *result set.
    uint64_t Signal(VkResult* result = nullptr) noexcept;

    /// Non-blocking: VK_SUCCESS once `value` has signaled, VK_NOT_READY
    /// before, or the device error (VK_ERROR_DEVICE_LOST).
    VkResult Check(uint64_t value) noexcept;

    /// Non-blocking: reads the signaled value into *value (one driver call).
    VkResult Poll(uint64_t* value) noexcept;

    /// Blocks until `value` signals or `timeoutNs` passes (VK_TIMEOUT).
    /// For synchronous API boundaries, shutdown, device loss and tests.
    /// Every call counts in VulkanQueueTimelineWaitsForTesting().
    VkResult Wait(uint64_t value, uint64_t timeoutNs) noexcept;

    /// The newest value known to have signaled (no driver call).
    uint64_t Completed() const noexcept { return completed_.load(std::memory_order_acquire); }
    /// The newest value handed out to a successful submit.
    uint64_t Submitted() const noexcept { return submitted_.load(std::memory_order_acquire); }
    VkSemaphore semaphore() const noexcept { return semaphore_; }

    struct Stats {
        uint64_t submits = 0;     // vkQueueSubmit calls
        uint64_t combined = 0;    // requests submitted by another thread
        uint64_t parks = 0;       // producers that parked for a submitter
    };
    Stats stats() const noexcept;

private:
    struct Request;
    VulkanQueueTimeline(VkDevice, VkQueue, VkSemaphore) noexcept;
    uint64_t Enqueue(VkCommandBuffer const* commands, uint32_t count,
                     VkResult* result) noexcept;
    void Drain(Request* self) noexcept;
    void SubmitOne(Request&) noexcept;
    void Observe(uint64_t value) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    std::atomic<Request*> pending_{nullptr};
    std::atomic<uint32_t> submitting_{0};
    std::atomic<uint64_t> submitted_{0};  // written by the submitter only
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> submits_{0}, combined_{0}, parks_{0};
};

/// Process-wide count of VulkanQueueTimeline::Wait calls (every timeline).
/// Tests assert deltas: a warm cook makes none.
USDGEN_VULKAN_INTERNAL_API uint64_t VulkanQueueTimelineWaitsForTesting() noexcept;

} // namespace usdGen::vulkan

#endif
