// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "chargedBuffer.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
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

// Verification mode (USDGEN_VULKAN_POISON_MEMORY=1): every buffer is placed
// in host-visible memory where the device offers it and filled with 0xFF
// bytes (NaN floats, all-ones words) before its first use, fresh or reused
// from the pool, so a consumer that reads memory it never wrote (relying on
// the zeroes a fresh driver allocation happens to hold) fails visibly.
std::atomic<int> gPoison{-1};  // -1 unread, 0 off, 1 on
bool PoisonMemory() noexcept {
    int p = gPoison.load(std::memory_order_relaxed);
    if (p < 0) {
        char const* v = std::getenv("USDGEN_VULKAN_POISON_MEMORY");
        p = v && *v && *v != '0' ? 1 : 0;
        gPoison.store(p, std::memory_order_relaxed);
    }
    return p == 1;
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
    bool const poison = PoisonMemory();
    uint32_t memoryType = UINT32_MAX;
    if (poison)
        memoryType = FindMemoryType(context->physicalDevice(), requirements.memoryTypeBits,
            properties | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memoryType == UINT32_MAX)
        memoryType = FindMemoryType(context->physicalDevice(), requirements.memoryTypeBits, properties);
    if (memoryType == UINT32_MAX) { if (result) *result = VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
    {
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(context->physicalDevice(), &memProps);
        if (memoryType < memProps.memoryTypeCount)
            owner->memoryPropertyFlags_ = memProps.memoryTypes[memoryType].propertyFlags;
    }
    // Size-class pooled memory: a steady-state create reuses an idle block
    // of the same class instead of calling vkAllocateMemory.
    owner->block_ = context->memoryPool()->Acquire(memoryType, requirements.size, &status);
    if (!owner->block_) { if (result) *result = status; return {}; }
    owner->memory_ = owner->block_->memory;
    if (poison && (owner->memoryPropertyFlags_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        void* mapped = nullptr;
        if (vkMapMemory(context->device(), owner->memory_, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
            std::memset(mapped, 0xFF, size_t(owner->block_->bytes));
            vkUnmapMemory(context->device(), owner->memory_);
        }
    }
    status = vkBindBufferMemory(context->device(), owner->buffer_, owner->memory_, 0);
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
    if (block_) context_->memoryPool()->Release(block_);
}

VkResult ChargedBuffer::MarkSubmitted(VkFence fence, std::shared_ptr<const void> submissionLifetime) {
    if (!buffer_ || submitted_ || unproven_ || !fence || !submissionLifetime) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult fenceStatus = vkGetFenceStatus(context_->device(), fence);
    if (fenceStatus != VK_NOT_READY) return fenceStatus == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : fenceStatus;
    *quarantineSubmission_ = submissionLifetime;
    submitted_ = true; unproven_ = true; fence_ = fence; submissionLifetime_ = std::move(submissionLifetime);
    return VK_SUCCESS;
}
VkResult ChargedBuffer::MarkSubmittedAt(uint64_t timelineValue, std::shared_ptr<const void> submissionLifetime) {
    if (!buffer_ || submitted_ || unproven_ || !timelineValue || !submissionLifetime ||
        !context_->computeTimeline()) return VK_ERROR_INITIALIZATION_FAILED;
    *quarantineSubmission_ = submissionLifetime;
    submitted_ = true; unproven_ = true; timelineValue_ = timelineValue;
    submissionLifetime_ = std::move(submissionLifetime);
    return VK_SUCCESS;
}
VkResult ChargedBuffer::PollComplete() {
    if (!submitted_ || (!fence_ && !timelineValue_)) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult status = fence_ ? vkGetFenceStatus(context_->device(), fence_)
                             : context_->computeTimeline()->Check(timelineValue_);
    if (status == VK_NOT_READY) return status;
    if (status != VK_SUCCESS) { Quarantine(); return status; }
    submitted_ = false; unproven_ = false; fence_ = VK_NULL_HANDLE; timelineValue_ = 0;
    submissionLifetime_.reset(); quarantineSubmission_->reset();
    return VK_SUCCESS;
}
void ChargedBuffer::Quarantine() noexcept {
    unproven_ = true; buffer_ = VK_NULL_HANDLE; memory_ = VK_NULL_HANDLE; block_ = nullptr; fence_ = VK_NULL_HANDLE;
    timelineValue_ = 0;
    // Permanently retain context/submission lifetimes and the allocation
    // charge when proof is lost. This helper has no reclamation path; it
    // never destroys the device or waits for queue idleness to infer safety.
    if (quarantineContext_) quarantineContext_.release();
    if (quarantineSubmission_) quarantineSubmission_.release();
    permit_.Abandon();
}
} // namespace usdGen::vulkan
