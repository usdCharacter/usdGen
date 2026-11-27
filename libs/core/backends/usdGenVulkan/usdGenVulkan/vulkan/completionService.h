// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_COMPLETION_SERVICE_H
#define USDGEN_VULKAN_COMPLETION_SERVICE_H

#include "deviceContext.h"
#include "queueTimeline.h"
#include "usdGen/exec/executionPipeline.h"

#include <cstdint>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace usdGen::vulkan {

// Device-local, event-driven proof delivery. It is not an execution backend
// or a general-purpose reactor: its one native waiter (a driver wait, one per
// service; the default provider makes one service per VkDevice) blocks in
// vkWaitSemaphores on the context's VulkanQueueTimeline and a private
// host-control semaphore. A watch completes when the queue timeline passes
// its value. The bound pipeline owner serializes watch mutation.
class VulkanCompletionService final : public std::enable_shared_from_this<VulkanCompletionService> {
public:
    enum class DeliveryResult : uint8_t { Posted, Stale, LostProof };
    using Delivery = std::function<DeliveryResult(VkResult)>;
    // May run thread-neutrally only if the pre-reserved owner route itself is
    // unavailable; it must not inspect or release native job ownership.
    using Failure = std::function<void(VkResult)>;
    struct CreateInfo {
        std::shared_ptr<DeviceContext> context;
        UsdGenExecutionPipeline* queueOwner = nullptr;
        uint32_t maxWatches = 0;
    };

    class Watch final {
    public:
        // Destruction after MarkPhaseSubmitted is owner-confined: destroy it
        // on the bound queue owner, or keep it through explicit external
        // CloseAndJoin. Off-owner marked destruction does not signal, mutate
        // transport state, or terminal-route sibling work.
        ~Watch();
        Watch(Watch const&) = delete;
        Watch& operator=(Watch const&) = delete;
        Watch(Watch&&) noexcept;
        Watch& operator=(Watch&&) noexcept;
        // Must occur on the bound queue owner immediately before the first
        // native submission whose lifetime this watch retains.
        bool MarkPhaseSubmitted() noexcept;
        // Owner-only acknowledgement after the posted completion callback has
        // consumed its proof. This is the only normal path that may release
        // the slot's native/job lifetime.
        bool Retire() noexcept;
    private:
        friend class VulkanCompletionService;
        friend uint64_t SubmitOnTimeline(VulkanCompletionService&, VkCommandBuffer const*, uint32_t,
                                         std::shared_ptr<const void>, Delivery, VkResult*,
                                         Failure) noexcept;
        Watch(std::shared_ptr<VulkanCompletionService>, uint32_t) noexcept;
        std::shared_ptr<VulkanCompletionService> service_;
        uint32_t slot_ = UINT32_MAX;
        bool marked_ = false;
        bool armed_ = false;
    };

    static std::shared_ptr<VulkanCompletionService> Create(CreateInfo, VkResult* = nullptr);
    ~VulkanCompletionService();
    VulkanCompletionService(VulkanCompletionService const&) = delete;
    VulkanCompletionService& operator=(VulkanCompletionService const&) = delete;

    // Acquires a preallocated bounded slot and retains `nativeJobLifetime`
    // before the caller marks or submits native work. `delivery` runs only on
    // the bound owner after proof and may complete there or issue an optional
    // pre-reserved owner post; successful proof is acknowledged by Retire.
    std::unique_ptr<Watch> Reserve(std::shared_ptr<const void> nativeJobLifetime,
                                   Delivery delivery, Failure failure = {}) noexcept;
    std::shared_ptr<DeviceContext> const& context() const noexcept { return context_; }
    UsdGenExecutionPipeline* queueOwner() const noexcept { return queueOwner_; }
    // Owner-only. Records the expected job epoch, signals the queue timeline
    // after phase work (an empty same-queue batch), then publishes and wakes
    // the waiter. *completionEpoch receives the timeline value.
    bool Arm(Watch&, uint64_t jobEpoch, uint64_t* completionEpoch = nullptr) noexcept;
    // Owner-only. As Arm, at a value the queue timeline already handed out:
    // every phase submission of the watch went through the timeline and
    // `timelineValue` is the last one. Submits nothing.
    bool ArmAt(Watch&, uint64_t timelineValue, uint64_t jobEpoch) noexcept;
    // Owner-only; only an unmarked reservation may be released normally.
    bool CancelBeforeSubmit(Watch&) noexcept;
    // External boundary only. The caller must first stop/join registration
    // that can Reserve/Mark/Arm (serializing host control signals); already
    // reserved owner terminal deliveries may still run. Close closes admission,
    // wakes the host-control timeline, joins the waiter, then failure-routes
    // only remaining Marked/Armed watches. True proves the waiter was joined;
    // a failed wake retains/detaches it and returns false on every later call.
    bool CloseAndJoin() noexcept;

private:
    struct Slot;
    friend uint64_t SubmitOnTimeline(VulkanCompletionService&, VkCommandBuffer const*, uint32_t,
                                     std::shared_ptr<const void>, Delivery, VkResult*,
                                     Failure) noexcept;
    explicit VulkanCompletionService(CreateInfo);
    bool Publish(Watch&, uint64_t timelineValue, uint64_t jobEpoch, uint64_t* completionEpoch) noexcept;
    bool Unsubmit(Watch&) noexcept;
    void Detach(Watch&) noexcept;
    bool Mark(Watch&) noexcept;
    bool Retire(Watch&) noexcept;
    void Abandon(Watch&) noexcept;
    void Quarantine(uint32_t) noexcept;
    void RouteFailure(uint32_t, VkResult) noexcept;
    void DeliverOnOwner(uint32_t, uint64_t, VkResult) noexcept;
    void FailOnOwner(uint32_t, uint64_t, VkResult) noexcept;
    void WaitMain() noexcept;
    bool Wake() noexcept;
    std::shared_ptr<DeviceContext> context_;
    UsdGenExecutionPipeline* queueOwner_ = nullptr;
    std::vector<std::unique_ptr<Slot>> slots_;
    VulkanQueueTimeline* timeline_ = nullptr;  // owned by context_
    VkSemaphore control_ = VK_NULL_HANDLE;
    std::thread waiter_;
    std::unique_ptr<std::shared_ptr<VulkanCompletionService>> quarantine_;
    std::atomic<bool> quarantined_{false};  // elects the one quarantine writer
    std::atomic<bool> closing_{false}, lost_{false};
    bool waiterJoined_ = false; // Serialized external lifecycle calls only.
    std::atomic<uint64_t> publishedControlEpoch_{0};
};

// SubmitOnTimeline: the one submit path for a phase whose completion runs a
// continuation. Owner-only (call it inside a queueOwner command).
//   uint64_t v = SubmitOnTimeline(*service, &cmd, 1, lifetime,
//       [state](VkResult proof) { /* continue on the owner */
//           return VulkanCompletionService::DeliveryResult::Posted; }, &r);
// It reserves a watch slot, submits `commands` (1..kMaxBatches) through the
// context's VulkanQueueTimeline and arms the slot at the returned value; no
// VkFence, no wait. `lifetime` (command buffer, buffers, pipeline state)
// stays alive until `delivery` has returned on the owner. `delivery` runs
// exactly once: VK_SUCCESS after the GPU passed the value, or an error on
// device loss / close. The slot frees itself unless delivery says LostProof.
// Returns 0 when nothing was queued (*result says why; no free slot is
// VK_ERROR_TOO_MANY_OBJECTS): `delivery` is dropped unrun, unless a device
// loss raced it. Keep `v` for a non-blocking computeTimeline()->Check(v).
uint64_t SubmitOnTimeline(VulkanCompletionService& service, VkCommandBuffer const* commands,
                          uint32_t count, std::shared_ptr<const void> lifetime,
                          VulkanCompletionService::Delivery delivery, VkResult* result = nullptr,
                          VulkanCompletionService::Failure failure = {}) noexcept;
} // namespace usdGen::vulkan
#endif
