// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_EXTERNAL_RETIREMENT_H
#define USDGEN_VULKAN_EXTERNAL_RETIREMENT_H

#include "usdGenVulkan/export.h"

#include <functional>
#include <atomic>
#include <cstdint>
#include <memory>

namespace usdGen { class UsdGenExecutionPipeline; }

namespace usdGen::vulkan {
class VulkanCompletionService;

// Retained by the provider and its deferred retirement, weakly tracked by the
// process queue. A lifecycle state word orders ordinary provider closure with
// final quiescence: a close never waits (it defers to the close in flight, or
// retains once quiescence began); only Quiesce, a shutdown path, parks.
class USDGEN_VULKAN_API VulkanExternalRetirementDomain final {
public:
    enum class CloseEntry { Run, Deferred, Retain };
    // Run: the caller closes, then calls EndClose. Deferred: a close is in
    // flight and reruns for this caller. Retain: quiesced or quiescing.
    CloseEntry BeginClose() noexcept;
    // Ends a Run close; true when a deferred close arrived and must run now.
    bool EndClose() noexcept;
    void Quiesce() noexcept;
    bool IsQuiesced() const noexcept { return (state_.load(std::memory_order_acquire) & kQuiesced) != 0; }
private:
    static constexpr uint32_t kClosing = 1;
    static constexpr uint32_t kRerun = 2;
    static constexpr uint32_t kQuiescing = 4;
    static constexpr uint32_t kQuiesced = 8;
    std::shared_ptr<UsdGenExecutionPipeline> owner_;
    std::shared_ptr<VulkanCompletionService> service_;
    std::atomic<uint32_t> state_{0};
    friend USDGEN_VULKAN_FRIEND_API std::shared_ptr<VulkanExternalRetirementDomain>
    TrackVulkanExternalRetirementDomain(std::shared_ptr<UsdGenExecutionPipeline>,
                                       std::shared_ptr<VulkanCompletionService>) noexcept;
};

// Register after creating the completion service, before publishing a native
// provider. A null result rejects admission; callers must close that service.
USDGEN_VULKAN_API std::shared_ptr<VulkanExternalRetirementDomain> TrackVulkanExternalRetirementDomain(
    std::shared_ptr<UsdGenExecutionPipeline>, std::shared_ptr<VulkanCompletionService>) noexcept;

// Rearm once per provider Shutdown, after lazy backend initialization. Later
// backend exit handlers must not unload code before accepted native cleanup.
USDGEN_VULKAN_API bool RegisterVulkanExternalRetirementAtExit() noexcept;

// Runs accepted close callbacks in order on the retirement lane, a
// UsdGenSerialLane on the process scheduler outside any execution graph, fed
// by a bounded lock-free ring. A running callback occupies one scheduler
// worker. Admission is bounded and permanently closes at shutdown.
// A rejected callback must remain retained by its native domain.
USDGEN_VULKAN_API bool EnqueueVulkanExternalRetirement(std::function<void()> callback) noexcept;

// External lifecycle boundary, also registered as a process exit hook after
// first use. Parks until every accepted callback ran, then until the lane
// stopped and joined every tracked domain's owner and completion waiter.
// Unproved ownership remains retained without waiting for consumers.
// Subsequent admissions fail closed. Never call this from a retirement
// callback, nor from a scheduler worker when no other worker is free.
USDGEN_VULKAN_API void CloseVulkanExternalRetirement() noexcept;

// Test-only: called on the thread that claims the lane start, before any
// admission; returning false fails that start. Null restores the default.
USDGEN_VULKAN_INTERNAL_API void SetVulkanExternalRetirementLaneStartHookForTesting(bool (*hook)()) noexcept;

} // namespace usdGen::vulkan
#endif
