// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/widthPipeline.h"
#include "vulkanNativeFixture.h"
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan Width pipeline check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

// Test-only queue proof. Production Candidate::Poll never waits or reads back
// geometry. This fence is submitted after the candidate on its confined queue.
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static bool Download(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<const ChargedBuffer> const& source, std::vector<float>* values) {
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = values->size() * sizeof(float); bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(source->context(), bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging) return false;
    VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
    VkBufferCopy region{0, 0, bi.size};
    vkCmdCopyBuffer(command, source->buffer(), staging->buffer(), 1, &region);
    VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    // Retain the source and real command/device owner if test readback loses proof.
    struct Keep { std::shared_ptr<NativeOwner> native; std::shared_ptr<const ChargedBuffer> source; };
    auto keep = std::make_shared<Keep>(Keep{native, source});
    if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) != VK_SUCCESS ||
        staging->PollComplete() != VK_SUCCESS) return false;
    void* mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(values->data(), mapped, bi.size); vkUnmapMemory(native->device, staging->memory());
    vkFreeCommandBuffers(native->device, native->commands, 1, &command);
    return true;
}

int main(int argc, char** argv) {
    bool quarantine = argc == 3 && std::string(argv[2]) == "--quarantine";
    CHECK(argc == 2 || quarantine);
    std::ifstream shader(argv[1], std::ios::binary);
    CHECK(shader);
    std::vector<char> bytes((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!bytes.empty() && bytes.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
    std::memcpy(code.data(), bytes.data(), bytes.size());
    bool unavailable = false;
    auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info; info.timelineSemaphoreEnabled = true;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7003; info.nativeLifetime = native;
    info.resources = {size_t{8} << 20, 0};
    auto context = DeviceContext::Create(info);
    CHECK(context); info.nativeLifetime.reset();
    auto pool = context->resources(); auto baseline = pool->Snapshot();
    VkResult status = VK_SUCCESS;
    CHECK(!WidthPipeline::Create(context, {0x07230203u}, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline = WidthPipeline::Create(context, code, &status);
    CHECK(pipeline && status == VK_SUCCESS);
    std::vector<float> original(257, 3.f), observed(257);
    auto upload = [&](std::vector<float> const& values) {
        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = values.size() * sizeof(float);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto buffer = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Active);
        if (!buffer) return buffer;
        void* mapped = nullptr;
        if (vkMapMemory(native->device, buffer->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS)
            return std::shared_ptr<ChargedBuffer>{};
        std::memcpy(mapped, values.data(), bi.size); vkUnmapMemory(native->device, buffer->memory());
        return buffer;
    };
    auto input = upload(original); CHECK(input);
    std::weak_ptr<const ChargedBuffer> weakInput = input;
    auto before = pool->Snapshot();
    CHECK(!pipeline->Begin(input, 258, 1.f, 0, &status));
    CHECK(!pipeline->Begin(input, 257, -1.f, 0, &status));
    CHECK(!pipeline->Begin(input, 257, 1.f, 2, &status));
    CHECK(!pipeline->Begin({}, 1, 1.f, 0, &status));
    CHECK(!pipeline->Begin({}, 0, -1.f, 0, &status));
    CHECK(pool->Snapshot().usedBytes == before.usedBytes);
    {
        auto otherInfo = info; otherInfo.nativeLifetime = native;
        auto otherContext = DeviceContext::Create(otherInfo);
        CHECK(otherContext && otherContext != context);
        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = 257 * sizeof(float); bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        auto otherInput = ChargedBuffer::Create(otherContext, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Active);
        CHECK(otherInput && !pipeline->Begin(otherInput, 257, 1.f, 0, &status));
        CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        auto wrongUsage = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Active);
        CHECK(wrongUsage && !pipeline->Begin(wrongUsage, 257, 1.f, 0, &status));
        CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    }
    CHECK(pool->Snapshot().usedBytes == before.usedBytes && pool->Snapshot().byKind == before.byKind);
    {
        auto empty = pipeline->Begin({}, 0, 2.f, 1, &status);
        uint32_t semantic = UINT32_MAX;
        CHECK(empty && status == VK_SUCCESS && empty->Poll(&semantic) == VK_SUCCESS && semantic == 0);
        CHECK(!empty->output() && pool->Snapshot().usedBytes == before.usedBytes);
        auto filler = pool->TryReserve(before.usableBytes - before.usedBytes, UsdGenExecutionResourceKind::Active);
        CHECK(filler); auto full = pool->Snapshot();
        CHECK(!pipeline->Begin(input, 257, 2.f, 1, &status));
        CHECK(status == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        CHECK(pool->Snapshot().usedBytes == full.usedBytes && pool->Snapshot().byKind == full.byKind);
    }
    auto first = pipeline->Begin(input, 257, 2.f, 1, &status);
    CHECK(first && status == VK_SUCCESS && !first->output());
    if (quarantine) {
        auto charged = pool->Snapshot();
        CHECK(charged.usedBytes > before.usedBytes);
        first->Quarantine(); first.reset(); input.reset(); pipeline.reset(); context.reset();
        CHECK(!weakInput.expired());
        CHECK(Prove(native)); native.reset();
        CHECK(!weakNative.expired() && !weakInput.expired());
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes && pool->Snapshot().byKind == charged.byKind);
        return 0;
    }
    CHECK(Prove(native));
    uint32_t semantic = UINT32_MAX;
    CHECK(first->Poll(&semantic) == VK_SUCCESS && semantic == 0);
    auto firstOutput = first->output(); CHECK(firstOutput && firstOutput->buffer() != input->buffer());
    CHECK(Download(native, firstOutput, &observed));
    CHECK(observed == std::vector<float>(257, 2.f));
    auto second = pipeline->Begin(firstOutput, 257, 4.f, 0, &status);
    CHECK(second && Prove(native));
    CHECK(second->Poll(&semantic) == VK_SUCCESS && semantic == 0);
    CHECK(Download(native, second->output(), &observed));
    CHECK(observed == std::vector<float>(257, 8.f));
    CHECK(Download(native, firstOutput, &observed) && observed == std::vector<float>(257, 2.f));
    // Input host bytes are untouched by both output generations.
    void* mapped = nullptr;
    CHECK(vkMapMemory(native->device, input->memory(), 0, original.size() * sizeof(float), 0, &mapped) == VK_SUCCESS);
    CHECK(std::memcmp(mapped, original.data(), original.size() * sizeof(float)) == 0);
    vkUnmapMemory(native->device, input->memory());
    second.reset(); first.reset(); firstOutput.reset();
    auto badValues = original; badValues[0] = std::numeric_limits<float>::quiet_NaN();
    auto bad = upload(badValues); CHECK(bad);
    auto invalid = pipeline->Begin(bad, 257, 2.f, 1, &status);
    CHECK(invalid && Prove(native));
    CHECK(invalid->Poll(&semantic) == VK_SUCCESS && semantic == 1 && !invalid->output());
    invalid.reset(); bad.reset();
    auto huge = upload(std::vector<float>(257, std::numeric_limits<float>::max())); CHECK(huge);
    auto overflow = pipeline->Begin(huge, 257, 2.f, 0, &status);
    CHECK(overflow && Prove(native));
    CHECK(overflow->Poll(&semantic) == VK_SUCCESS && semantic == 2 && !overflow->output());
    overflow.reset(); huge.reset(); input.reset(); pipeline.reset(); context.reset(); native.reset();
    CHECK(weakNative.expired() && weakInput.expired());
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes && pool->Snapshot().byKind == baseline.byKind);
    return 0;
}
