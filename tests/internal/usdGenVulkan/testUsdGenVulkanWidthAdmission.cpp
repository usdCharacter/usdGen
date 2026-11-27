// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/completionService.h"
#include "usdGenCurves/vulkan/widthPipeline.h"
#include "vulkanReadbackFixture.h"
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan width admission check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

int main(int argc, char** argv) {
    CHECK(argc == 2);
    std::ifstream shader(argv[1], std::ios::binary); CHECK(shader);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!raw.empty() && raw.size() % 4 == 0);
    std::vector<uint32_t> code(raw.size() / 4); std::memcpy(code.data(), raw.data(), raw.size());
    bool unavailable = false;
    auto probe = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(probe);
    VkPhysicalDeviceTimelineSemaphoreFeatures supported{};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features{}; features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &supported;
    vkGetPhysicalDeviceFeatures2(probe->physical, &features);
    if (!supported.timelineSemaphore) return 77;
    probe.reset();
    auto native = CreateNative(&unavailable, {}, &supported); CHECK(native);
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family; ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 7018; ci.nativeLifetime = native;
    ci.resources = {size_t{8} << 20, 0}; ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci); CHECK(context);
    auto pipeline = WidthPipeline::Create(context, code); CHECK(pipeline);
    UsdGenExecutionRuntime runtime{2}; UsdGenExecutionPipeline owner(runtime);
    auto service = VulkanCompletionService::Create({context, &owner, 1}); CHECK(service);
    struct CloseService { std::shared_ptr<VulkanCompletionService> service; ~CloseService() { service->CloseAndJoin(); } } closeService{service};
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 8; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto input = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Pinned); CHECK(input);
    void* mapped = nullptr;
    CHECK(vkMapMemory(native->device, input->memory(), 0, 8, 0, &mapped) == VK_SUCCESS);
    float original[2] = {1, 2}; std::memcpy(mapped, original, sizeof(original));
    vkUnmapMemory(native->device, input->memory());
    auto pool = context->resources(); auto baseline = pool->Snapshot();
    using Result = VulkanCompletionService::DeliveryResult;
    bool ok = false;
    owner.InvokeOwner([&] {
        auto watch = service->Reserve(input, [](VkResult) { return Result::Stale; });
        if (!watch) return;
        auto pressure = pool->TryReserve(baseline.usableBytes - baseline.usedBytes, UsdGenExecutionResourceKind::Scratch);
        if (!pressure) return;
        unsigned hooks = 0; VkResult status = VK_SUCCESS;
        auto rejected = pipeline->Begin(input, 2, 3, 1, &status, [&] { ++hooks; return watch->MarkPhaseSubmitted(); });
        std::fprintf(stderr, "admission budget: candidate=%d status=%d hooks=%u\n", bool(rejected), int(status), hooks);
        ok = !rejected && status != VK_SUCCESS && hooks == 0 && service->CancelBeforeSubmit(*watch);
        pressure->Release();
        ok = ok && pool->Snapshot().usedBytes == baseline.usedBytes;
        // Preparation completed, but caller admission rejected or threw.
        rejected = pipeline->Begin(input, 2, 3, 1, &status, [&] { ++hooks; return false; });
        std::fprintf(stderr, "admission reject: candidate=%d status=%d hooks=%u used=%zu baseline=%zu ok=%d\n", bool(rejected), int(status), hooks, pool->Snapshot().usedBytes, baseline.usedBytes, ok);
        ok = ok && !rejected && hooks == 1 && pool->Snapshot().usedBytes == baseline.usedBytes;
        rejected = pipeline->Begin(input, 2, 3, 1, &status, [&]() -> bool { ++hooks; throw std::runtime_error("admission"); });
        std::fprintf(stderr, "admission throw: candidate=%d status=%d hooks=%u used=%zu\n", bool(rejected), int(status), hooks, pool->Snapshot().usedBytes);
        ok = ok && !rejected && status == VK_ERROR_UNKNOWN && hooks == 2 && pool->Snapshot().usedBytes == baseline.usedBytes;
        auto empty = pipeline->Begin({}, 0, 3, 1, &status, [&] { ++hooks; return true; });
        std::fprintf(stderr, "admission empty: candidate=%d status=%d hooks=%u success=%d\n", bool(empty), int(status), hooks, empty && empty->succeeded());
        ok = ok && empty && empty->succeeded() && hooks == 2;
        rejected = pipeline->Begin(input, 2, -1, 1, &status, [&] { ++hooks; return true; });
        ok = ok && !rejected && hooks == 2;
    });
    CHECK(ok && owner.OutstandingCommands() == 0);

    // The same one-slot service must remain usable after budget rejection.
    // Native completion, not a test-driven fence wait, finishes this dispatch.
    std::unique_ptr<VulkanCompletionService::Watch> watch;
    std::unique_ptr<WidthPipeline::Candidate> candidate;
    unsigned hooks = 0, callbacks = 0;
    ok = false;
    owner.Await([&](auto finish) {
        if (!owner.PostCommand([&, finish] {
            watch = service->Reserve(input, [&, finish](VkResult proof) {
                ++callbacks; uint32_t semantic = UINT32_MAX;
                ok = owner.IsExecutingOwner() && proof == VK_SUCCESS && candidate &&
                    candidate->Poll(&semantic) == VK_SUCCESS && semantic == 0;
                ok = watch->Retire() && ok;
                finish(); return Result::Posted;
            });
            if (!watch) { finish(); return; }
            candidate = pipeline->Begin(input, 2, 3, 1, nullptr, [&] { ++hooks; return watch->MarkPhaseSubmitted(); });
            if (!candidate || !service->Arm(*watch, 1)) finish();
        })) finish();
    });
    service->CloseAndJoin(); owner.Drain();
    CHECK(ok && hooks == 1 && callbacks == 1 && owner.OutstandingCommands() == 0);
    std::vector<uint8_t> bytes;
    CHECK(ReadVulkanBytes(native, context, candidate->output()->buffer(), 8, candidate->output(), &bytes));
    float output[2]{}; std::memcpy(output, bytes.data(), sizeof(output));
    CHECK(output[0] == 3 && output[1] == 3);
    CHECK(ReadVulkanBytes(native, context, input->buffer(), 8, input, &bytes));
    CHECK(std::memcmp(bytes.data(), original, sizeof(original)) == 0);
    owner.InvokeOwner([&] { watch.reset(); candidate.reset(); });
    input.reset(); service.reset();
    CHECK(pool->Snapshot().usedBytes == 0);
    return 0;
}
