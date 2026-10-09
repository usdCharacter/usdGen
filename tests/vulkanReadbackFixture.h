// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_TEST_VULKAN_READBACK_H
#define USDGEN_TEST_VULKAN_READBACK_H

#include "vulkanNativeFixture.h"
#include "usdGen/vulkan/chargedBuffer.h"
#include <algorithm>
#include <cstring>

// Upload through a transfer into device memory, so compute fixtures exercise
// buffers that production cannot read by mapping their inputs on the host.
inline std::shared_ptr<usdGen::vulkan::ChargedBuffer> UploadVulkanDeviceBytes(
    std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<usdGen::vulkan::DeviceContext> const& context,
    void const* values, size_t bytes) {
    using namespace usdGen;
    using namespace usdGen::vulkan;
    if (!native || !context || !values || !bytes) return {};
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto staging = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto output = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging || !output) return {};
    void* mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS)
        return {};
    std::memcpy(mapped, values, bytes);
    vkUnmapMemory(native->device, staging->memory());
    VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return {};
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return {};
    VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(command, staging->buffer(), output->buffer(), 1, &copy);
    VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return {};
    struct Keep { std::shared_ptr<NativeOwner> native; std::shared_ptr<ChargedBuffer> output; };
    auto keep = std::make_shared<Keep>(Keep{native, output});
    if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) return {};
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) != VK_SUCCESS ||
        staging->PollComplete() != VK_SUCCESS) return {};
    vkFreeCommandBuffers(native->device, native->commands, 1, &command);
    return output;
}

// Geometry readback is test-only. A failed submission/proof intentionally
// retains staging, source and actual native command/device ownership.
inline bool ReadVulkanBytes(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<usdGen::vulkan::DeviceContext> const& context,
    VkBuffer source, size_t bytes, std::shared_ptr<const void> sourceLifetime,
    std::vector<uint8_t>* output) {
    if (!output || !native || !context || context->device() != native->device ||
        context->computeQueue() != native->queue) return false;
    if (!bytes) { output->clear(); return true; }
    if (!source || !sourceLifetime) return false;
    using namespace usdGen;
    using namespace usdGen::vulkan;
    // One region copy per call; the caller loops it over chunks. Split out
    // so the tight-budget fallback below reuses the proven sequence.
    auto copyRegion = [&](size_t srcOffset, size_t dstOffset, size_t size) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto staging = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch);
        if (!staging) return false;
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return false;
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
        VkMemoryBarrier before{};
        before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        VkBufferCopy region{srcOffset, 0, size};
        vkCmdCopyBuffer(command, source, staging->buffer(), 1, &region);
        VkMemoryBarrier after{};
        after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &after, 0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(command) != VK_SUCCESS ||
            vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
        struct Keep { std::shared_ptr<NativeOwner> native; std::shared_ptr<const void> source; };
        auto keep = std::make_shared<Keep>(Keep{native, sourceLifetime});
        if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) return false;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
            vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) !=
                VK_SUCCESS ||
            staging->PollComplete() != VK_SUCCESS) return false;
        void* mapped = nullptr;
        if (vkMapMemory(native->device, staging->memory(), 0, size, 0, &mapped) != VK_SUCCESS)
            return false;
        std::memcpy(output->data() + dstOffset, mapped, size);
        vkUnmapMemory(native->device, staging->memory());
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return true;
    };
    output->resize(bytes);
    if (copyRegion(0, 0, bytes)) return true;
    // Tight budget (a funded cache may leave only its ~1MiB margin free):
    // retry in 256KiB chunks, which fit any post-funding remainder. Same
    // bytes, slower; only reachable when the single staging missed.
    size_t const kChunk = size_t{256} << 10;
    for (size_t off = 0; off < bytes; off += kChunk) {
        size_t const size = std::min(kChunk, bytes - off);
        if (!copyRegion(off, off, size)) return false;
    }
    return true;
}
#endif
