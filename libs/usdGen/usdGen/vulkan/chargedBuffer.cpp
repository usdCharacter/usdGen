// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "chargedBuffer.h"

#include <limits>

namespace usdGen::vulkan {
namespace {
uint32_t FindMemoryType(VkPhysicalDevice physical, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((bits & (uint32_t(1) << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return UINT32_MAX;
}
}

ChargedBuffer::ChargedBuffer(std::shared_ptr<DeviceContext> context) : context_(std::move(context)) {}

std::shared_ptr<ChargedBuffer> ChargedBuffer::Create(std::shared_ptr<DeviceContext> const& context,
    VkBufferCreateInfo const& createInfo, VkMemoryPropertyFlags properties,
    UsdGenExecutionResourceKind kind, VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context || !createInfo.size || createInfo.sType != VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO) return {};
    std::shared_ptr<ChargedBuffer> owner;
    try {
        owner = std::shared_ptr<ChargedBuffer>(new ChargedBuffer(context));
        owner->quarantineContext_ = std::make_unique<std::shared_ptr<DeviceContext>>(context);
        owner->quarantineSubmission_ = std::make_unique<std::shared_ptr<const void>>();
    } catch (...) { if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY; return {}; }
    VkResult status = vkCreateBuffer(context->device(), &createInfo, nullptr, &owner->buffer_);
    if (status != VK_SUCCESS) { if (result) *result = status; return {}; }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context->device(), owner->buffer_, &requirements);
    if (!requirements.size || requirements.size > std::numeric_limits<size_t>::max()) {
        return {};
    }
    auto permit = context->resources()->TryReserve(static_cast<size_t>(requirements.size), kind);
    if (!permit) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
    owner->permit_ = std::move(*permit);
    uint32_t memoryType = FindMemoryType(context->physicalDevice(), requirements.memoryTypeBits, properties);
    if (memoryType == UINT32_MAX) { if (result) *result = VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
    {
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(context->physicalDevice(), &memProps);
        if (memoryType < memProps.memoryTypeCount)
            owner->memoryPropertyFlags_ = memProps.memoryTypes[memoryType].propertyFlags;
    }
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = memoryType;
    status = vkAllocateMemory(context->device(), &allocation, nullptr, &owner->memory_);
    if (status == VK_SUCCESS) status = vkBindBufferMemory(context->device(), owner->buffer_, owner->memory_, 0);
    if (status != VK_SUCCESS) { if (result) *result = status; return {}; }
    owner->allocationBytes_ = requirements.size;
    owner->sizeBytes_ = createInfo.size;
    owner->usage_ = createInfo.usage;
    if (result) *result = VK_SUCCESS;
    return owner;
}

ChargedBuffer::~ChargedBuffer() {
    if (unproven_) { Quarantine(); return; }
    if (submitted_) { Quarantine(); return; }
    if (buffer_) vkDestroyBuffer(context_->device(), buffer_, nullptr);
    if (memory_) vkFreeMemory(context_->device(), memory_, nullptr);
}

VkResult ChargedBuffer::MarkSubmitted(VkFence fence, std::shared_ptr<const void> submissionLifetime) {
    if (!buffer_ || submitted_ || unproven_ || !fence || !submissionLifetime) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult fenceStatus = vkGetFenceStatus(context_->device(), fence);
    if (fenceStatus != VK_NOT_READY) return fenceStatus == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : fenceStatus;
    *quarantineSubmission_ = submissionLifetime;
    submitted_ = true; unproven_ = true; fence_ = fence; submissionLifetime_ = std::move(submissionLifetime);
    return VK_SUCCESS;
}
VkResult ChargedBuffer::PollComplete() {
    if (!submitted_ || !fence_) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult status = vkGetFenceStatus(context_->device(), fence_);
    if (status == VK_NOT_READY) return status;
    if (status != VK_SUCCESS) { Quarantine(); return status; }
    submitted_ = false; unproven_ = false; fence_ = VK_NULL_HANDLE; submissionLifetime_.reset(); quarantineSubmission_->reset();
    return VK_SUCCESS;
}
void ChargedBuffer::Quarantine() noexcept {
    unproven_ = true; buffer_ = VK_NULL_HANDLE; memory_ = VK_NULL_HANDLE; fence_ = VK_NULL_HANDLE;
    // Permanently retain context/submission lifetimes and the allocation
    // charge when proof is lost. This helper has no reclamation path; it
    // never destroys the device or waits for queue idleness to infer safety.
    if (quarantineContext_) quarantineContext_.release();
    if (quarantineSubmission_) quarantineSubmission_.release();
    permit_.Abandon();
}
} // namespace usdGen::vulkan
