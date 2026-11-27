// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/chargedBuffer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan resource check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

#include "hosts/usdGenUsd/vulkanNativeFixture.h"

static bool Read(VkDevice device, ChargedBuffer const& buffer, std::vector<uint32_t>* values) {
    void* mapped = nullptr;
    size_t bytes = values->size() * sizeof(uint32_t);
    if (vkMapMemory(device, buffer.memory(), 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(values->data(), mapped, bytes);
    vkUnmapMemory(device, buffer.memory());
    return true;
}

int main(int argc, char** argv) {
    bool quarantine = argc == 2 && std::string(argv[1]) == "--quarantine";
    CHECK(argc == 1 || quarantine);
    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable) return 77; // No alternate CPU/llvmpipe execution is substituted.
    CHECK(native);
    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    // This isolated factory fixture owns the key. Cross-context UUID registry
    // validation remains a separate requirement before Session integration.
    info.resourceDeviceId = 7001;
    info.resources = {size_t{8} << 20, 0}; info.nativeLifetime = native;
    VkResult result = VK_ERROR_UNKNOWN;
    auto context = DeviceContext::Create(info, &result);
    CHECK(context && result == VK_SUCCESS);
    info.nativeLifetime.reset();
    auto pool = context->resources();
    auto baseline = pool->Snapshot();
    VkPhysicalDeviceIDProperties ids{}; ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props{}; props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &ids; vkGetPhysicalDeviceProperties2(native->physical, &props);
    CHECK(std::memcmp(context->deviceUUID().data(), ids.deviceUUID, VK_UUID_SIZE) == 0);

    constexpr size_t count = 257;
    constexpr size_t bytes = count * sizeof(uint32_t);
    VkBufferCreateInfo create{}; create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create.size = bytes;
    create.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto first = ChargedBuffer::Create(context, create, memory,
        UsdGenExecutionResourceKind::Active, &result);
    CHECK(first && result == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(native->device, first->buffer(), &requirements);
    CHECK(first->allocationBytes() == requirements.size);
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + requirements.size);

    // Impossible memory properties reject after reservation and restore it.
    auto before = pool->Snapshot();
    auto rejected = ChargedBuffer::Create(context, create, VkMemoryPropertyFlags(0x80000000u),
        UsdGenExecutionResourceKind::Active, &result);
    CHECK(!rejected && result == VK_ERROR_FEATURE_NOT_PRESENT);
    CHECK(pool->Snapshot().usedBytes == before.usedBytes && pool->Snapshot().byKind == before.byKind);
    {
        auto filler = pool->TryReserve(before.usableBytes - before.usedBytes,
                                      UsdGenExecutionResourceKind::Active);
        CHECK(filler);
        auto full = pool->Snapshot();
        rejected = ChargedBuffer::Create(context, create, memory,
            UsdGenExecutionResourceKind::Active, &result);
        CHECK(!rejected && result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        auto after = pool->Snapshot();
        CHECK(after.usedBytes == full.usedBytes && after.byKind == full.byKind);
    }

    auto second = ChargedBuffer::Create(context, create, memory,
        UsdGenExecutionResourceKind::Active, &result);
    CHECK(second && second->buffer() != first->buffer() && second->memory() != first->memory());
    VkMemoryRequirements secondRequirements{};
    vkGetBufferMemoryRequirements(native->device, second->buffer(), &secondRequirements);
    CHECK(second->allocationBytes() == secondRequirements.size);
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + requirements.size + secondRequirements.size);
    std::vector<uint32_t> original(count, 0x40400000u), observed(count);
    void* mapped = nullptr;
    CHECK(vkMapMemory(native->device, first->memory(), 0, bytes, 0, &mapped) == VK_SUCCESS);
    std::memcpy(mapped, original.data(), bytes); vkUnmapMemory(native->device, first->memory());
    auto held = first; first.reset();

    if (quarantine) {
        CHECK(second->MarkSubmitted(native->fence, native) == VK_SUCCESS);
        auto pending = pool->Snapshot();
        CHECK(second->PollComplete() == VK_NOT_READY);
        CHECK(pool->Snapshot().usedBytes == pending.usedBytes && pool->Snapshot().byKind == pending.byKind);
        // Explicit parent proof-loss injection: no GPU work is fabricated.
        // A later signaled fence cannot refund this retained allocation.
        second->Quarantine();
        CHECK(second->MarkSubmitted(native->fence, native) != VK_SUCCESS);
        VkSubmitInfo emptySubmit{}; emptySubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        CHECK(vkQueueSubmit(native->queue, 1, &emptySubmit, native->fence) == VK_SUCCESS);
        CHECK(vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);
        CHECK(second->PollComplete() == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(pool->Snapshot().usedBytes == pending.usedBytes);
        second.reset(); held.reset();
        CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + secondRequirements.size);
        context.reset(); native.reset();
        CHECK(!weakNative.expired());
        return 0; // Intentionally retained native owner; no manual device destroy.
    }

    VkCommandBufferAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = native->commands; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(native->device, &allocate, &command) == VK_SUCCESS);
    for (uint32_t iteration = 0; iteration < 2; ++iteration) {
        if (iteration) {
            CHECK(vkResetFences(native->device, 1, &native->fence) == VK_SUCCESS);
            CHECK(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
        }
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CHECK(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
        uint32_t fill = iteration ? 0x40000000u : 0x3f800000u;
        vkCmdFillBuffer(command, second->buffer(), 0, bytes, fill);
        VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(command) == VK_SUCCESS);
        CHECK(second->MarkSubmitted(native->fence, native) == VK_SUCCESS);
        auto pending = pool->Snapshot();
        CHECK(second->PollComplete() == VK_NOT_READY);
        CHECK(pool->Snapshot().usedBytes == pending.usedBytes && pool->Snapshot().byKind == pending.byKind);
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        // Arm above covers every post-submission failure. If a check fails,
        // buffer destruction quarantines the real native lifetime.
        CHECK(vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS);
        CHECK(vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);
        CHECK(second->PollComplete() == VK_SUCCESS);
        CHECK(second->MarkSubmitted(native->fence, native) == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(Read(native->device, *second, &observed));
        CHECK(std::all_of(observed.begin(), observed.end(), [&](uint32_t value) { return value == fill; }));
        CHECK(Read(native->device, *held, &observed) && observed == original);
    }
    second.reset();
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + requirements.size);
    native.reset(); context.reset(); CHECK(!weakNative.expired());
    held.reset();
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes && pool->Snapshot().byKind == baseline.byKind);
    CHECK(weakNative.expired());
    return 0;
}
