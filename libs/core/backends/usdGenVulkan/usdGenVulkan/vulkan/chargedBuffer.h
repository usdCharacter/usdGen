// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_CHARGED_BUFFER_H
#define USDGEN_VULKAN_CHARGED_BUFFER_H

#include "deviceContext.h"
#include "usdGenVulkan/export.h"

#include <memory>

namespace usdGen::vulkan {
class USDGEN_VULKAN_API ChargedBuffer final {
public:
    static std::shared_ptr<ChargedBuffer> Create(std::shared_ptr<DeviceContext> const&, VkBufferCreateInfo const&,
                                                 VkMemoryPropertyFlags, UsdGenExecutionResourceKind, VkResult* = nullptr);
    ~ChargedBuffer();
    ChargedBuffer(ChargedBuffer const&) = delete;
    ChargedBuffer& operator=(ChargedBuffer const&) = delete;

    VkBuffer buffer() const noexcept { return buffer_; }
    VkDeviceMemory memory() const noexcept { return memory_; }
    // Logical buffer extent, distinct from the charged native allocation size.
    VkDeviceSize sizeBytes() const noexcept { return sizeBytes_; }
    VkBufferUsageFlags usage() const noexcept { return usage_; }
    // The bound memory type's actual property flags (a superset of the
    // requested Create flags), so callers can detect host-mappable memory
    // without probing vkMapMemory.
    VkMemoryPropertyFlags memoryPropertyFlags() const noexcept { return memoryPropertyFlags_; }
    std::shared_ptr<DeviceContext> const& context() const noexcept { return context_; }
    VkDeviceSize allocationBytes() const noexcept { return allocationBytes_; }
    bool unproven() const noexcept { return unproven_; }

    // Call before vkQueueSubmit while the scheduled owner holds its queue
    // serialization. `submissionLifetime` keeps the fence unreset/alive.
    VkResult MarkSubmitted(VkFence, std::shared_ptr<const void> submissionLifetime);
    // Queue-timeline form: call right after the context's computeTimeline()
    // Submit returned `timelineValue` for work that uses this buffer.
    VkResult MarkSubmittedAt(uint64_t timelineValue, std::shared_ptr<const void> submissionLifetime);
    // Nonblocking proof: VK_NOT_READY leaves ownership and accounting intact.
    VkResult PollComplete();
    void Quarantine() noexcept;

private:
    explicit ChargedBuffer(std::shared_ptr<DeviceContext>);
    std::shared_ptr<DeviceContext> context_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    // The pooled block behind memory_ (offset 0), returned on destruction.
    VulkanDeviceMemoryPool::Block* block_ = nullptr;
    VkDeviceSize allocationBytes_ = 0;
    VkDeviceSize sizeBytes_ = 0;
    VkBufferUsageFlags usage_ = 0;
    VkMemoryPropertyFlags memoryPropertyFlags_ = 0;
    VkFence fence_ = VK_NULL_HANDLE;
    uint64_t timelineValue_ = 0;  // MarkSubmittedAt proof (0: none)
    std::shared_ptr<const void> submissionLifetime_;
    UsdGenExecutionResourcePermit permit_;
    // Allocated before first submission so lost-proof quarantine is noexcept.
    std::unique_ptr<std::shared_ptr<DeviceContext>> quarantineContext_;
    std::unique_ptr<std::shared_ptr<const void>> quarantineSubmission_;
    bool submitted_ = false;
    bool unproven_ = false;
};
} // namespace usdGen::vulkan
#endif
