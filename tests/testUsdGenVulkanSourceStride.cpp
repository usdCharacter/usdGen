// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "vulkanNativeFixture.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan source stride check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

// Deliberately permits the non-four-byte logical span under test. The shared
// geometry helper rejects this case because its callers are word-oriented.
static bool ReadBytes9(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context, VkBuffer source, size_t bytes,
    std::shared_ptr<const void> sourceLifetime, std::vector<uint8_t>* output) {
    if (!native || !context || context->device() != native->device ||
        context->computeQueue() != native->queue || !source || !sourceLifetime || bytes == 0) return false;
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging) return false;
    VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
    VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(command, source, staging->buffer(), 1, &region);
    VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &after, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    struct Keep { std::shared_ptr<NativeOwner> native; std::shared_ptr<const void> source; };
    auto keep = std::make_shared<Keep>(Keep{native, std::move(sourceLifetime)});
    if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) != VK_SUCCESS ||
        staging->PollComplete() != VK_SUCCESS) return false;
    output->resize(bytes);
    void* mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(output->data(), mapped, bytes); vkUnmapMemory(native->device, staging->memory());
    vkFreeCommandBuffers(native->device, native->commands, 1, &command);
    return true;
}

static VulkanSourceGenerationCreateInfo Fixture(std::shared_ptr<DeviceContext> context,
    std::vector<uint8_t> payload) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& b = info.source;
    b.totalCurves = 1; b.totalCvs = 2; b.topologyVersion = 1; b.valueVersion = 1;
    b.px = {0, 1}; b.py = {0, 1}; b.pz = {0, 1}; b.curveId = {42}; b.cvOffsets = {0, 2};
    info.additionalNamed.push_back({{TfToken("stride5"), UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Point, 2, 1, 5, true, UsdGenDeviceChannelSemantic::Generic},
        std::move(payload)});
    return info;
}

int main() {
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family; ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 7013; ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true; ci.resources = {size_t{8} << 20, 0};
    VkResult result = VK_ERROR_UNKNOWN; std::string reason;
    auto context = DeviceContext::Create(ci, &result); CHECK(context && result == VK_SUCCESS);
    std::vector<uint8_t> expected{1,2,3,4,5,6,7,8,9};
    auto info = Fixture(context, expected);
    auto upload = VulkanSourceUpload::Create(info, &result, &reason);
    CHECK(upload && result == VK_SUCCESS);
    CHECK(upload->Submit() == SourceGenerationStatus::Submitted);
    CHECK(Prove(native)); CHECK(upload->Poll() == SourceGenerationStatus::Ready);
    auto generation = upload->TakeReady(); CHECK(generation);
    auto plane = generation->PlaneOwner(TfToken("stride5")); CHECK(plane && plane->sizeBytes() == 9);
    auto const& views = generation->planes();
    bool saw = false;
    for (auto const& view : views) if (view.metadata.name == "stride5") {
        saw = true; CHECK(view.metadata.elementCount == 2 && view.metadata.strideBytes == 5);
        CHECK(view.bytes == 9 && ReadBytes9(native, context, view.buffer, view.bytes, generation, &expected));
    }
    CHECK(saw && expected == std::vector<uint8_t>({1,2,3,4,5,6,7,8,9}));
    for (size_t count : {size_t{8}, size_t{10}}) {
        auto bad = info; bad.additionalNamed[0].bytes.resize(count);
        auto before = context->resources()->Snapshot();
        CHECK(!VulkanSourceUpload::Create(bad, &result, &reason));
        CHECK(context->resources()->Snapshot().usedBytes == before.usedBytes);
    }
    generation.reset(); upload.reset(); info.context.reset(); context.reset(); native.reset();
    return 0;
}
