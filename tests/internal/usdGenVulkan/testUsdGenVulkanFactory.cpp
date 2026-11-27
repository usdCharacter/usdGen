// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/deviceFactory.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "hosts/usdGenUsd/vulkanNativeFixture.h"

#include <algorithm>
#include <atomic>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan factory check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static std::array<uint8_t, VK_UUID_SIZE> Uuid(VkPhysicalDevice physical) {
    VkPhysicalDeviceIDProperties ids{}; ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props{}; props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &ids; vkGetPhysicalDeviceProperties2(physical, &props);
    std::array<uint8_t, VK_UUID_SIZE> result{};
    std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), result.begin());
    return result;
}

static DeviceFactory::CreateInfo Info(std::shared_ptr<NativeOwner> const& native) {
    DeviceFactory::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.nativeLifetime = native;
    return info;
}

int main() {
    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    auto uuid = Uuid(native->physical);
    UsdGenExecutionRuntime runtime{2};
    UsdGenExecutionPipeline replies(runtime);
    auto lifetime = std::make_shared<int>(0);
    UsdGenExecutionResourceConfig config{size_t{8} << 20, 0};
    using Entry = std::shared_ptr<const DeviceFactory::Entry>;
    auto resolve = [&](auto const& key, auto const& requested, bool requireMatchingConfig = true) {
        Entry entry;
        replies.Await([&](auto finish) {
            if (!DeviceFactory::ResolveAsync(key, requested, replies, lifetime,
                [&, finish](Entry result, DeviceFactory::ResolveOutcome outcome) {
                    if (outcome == DeviceFactory::ResolveOutcome::Resolved) entry = std::move(result);
                    finish();
                }, requireMatchingConfig)) finish();
        });
        return entry;
    };
    auto entry = resolve(uuid, config);
    CHECK(entry && entry->deviceUUID() == uuid);
    {
        UsdGenExecutionPipeline bounded(runtime, 4, 1);
        auto ticket = bounded.ReserveCommandTicket(); CHECK(ticket);
        bool called = false;
        auto rejectedLife = std::make_shared<int>(0);
        std::weak_ptr<int> weak = rejectedLife;
        CHECK(!DeviceFactory::ResolveAsync(uuid, config, bounded, std::move(rejectedLife),
            [&](Entry, DeviceFactory::ResolveOutcome) { called = true; }));
        CHECK(!called && weak.expired() && bounded.OutstandingCommands() == 1);
        ticket = {}; bounded.Drain();
        CHECK(bounded.OutstandingCommands() == 0);
    }
    {
        auto replyState = std::make_shared<int>(0);
        std::weak_ptr<int> weak = replyState;
        bool retainedDuringCallback = false;
        replies.Await([&](auto finish) {
            if (!DeviceFactory::ResolveAsync(uuid, config, replies, std::move(replyState),
                [&, finish, weak](Entry result, DeviceFactory::ResolveOutcome outcome) {
                    retainedDuringCallback = !weak.expired() && result == entry &&
                        outcome == DeviceFactory::ResolveOutcome::Resolved;
                    finish();
                })) finish();
        });
        replies.Drain();
        CHECK(retainedDuringCallback && weak.expired());
    }
    CHECK(entry->resources().limitBytes == config.limitBytes);
    CHECK(resolve(uuid, config) == entry);
    auto mismatch = config; mismatch.headroomBytes = 4096;
    CHECK(!resolve(uuid, mismatch));
    mismatch = config; mismatch.limitBytes += 4096;
    CHECK(!resolve(uuid, mismatch));
    // A later automatic free-memory snapshot cannot replace the shared pool
    // or reject a second Session. Explicit mismatches above remain rejected.
    CHECK(resolve(uuid, mismatch, false) == entry);
    CHECK(entry->resources().limitBytes == config.limitBytes);
    CHECK(entry->resources().headroomBytes == config.headroomBytes);
    auto otherUuid = uuid; otherUuid[0] ^= 0x80;
    auto other = resolve(otherUuid, config);
    CHECK(other && other != entry && other->resourceDeviceId() != entry->resourceDeviceId());
    CHECK(other->resourcePool() != entry->resourcePool());

    // Many accepted requests converge on one immutable entry. Requests are
    // initiated inside an owner frame: no synchronous InvokeOwner is allowed.
    std::array<Entry, 32> results;
    std::atomic<unsigned> remaining{results.size()};
    bool allAdmitted = true;
    replies.Await([&](auto finish) {
        if (!replies.PostCommand([&, finish] {
            for (size_t i = 0; i < results.size(); ++i) {
                bool admitted = DeviceFactory::ResolveAsync(uuid, config, replies, lifetime,
                    [&, finish, i](Entry result, DeviceFactory::ResolveOutcome outcome) {
                        if (outcome == DeviceFactory::ResolveOutcome::Resolved) results[i] = std::move(result);
                        if (remaining.fetch_sub(1) == 1) finish();
                    });
                if (!admitted) {
                    allAdmitted = false;
                    if (remaining.fetch_sub(1) == 1) finish();
                }
            }
        })) { allAdmitted = false; finish(); }
    });
    CHECK(allAdmitted);
    CHECK(std::all_of(results.begin(), results.end(), [&](auto const& value) { return value == entry; }));

    VkResult status = VK_SUCCESS;
    auto info = Info(native);
    CHECK(!DeviceFactory::CreateContext(other, info, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    auto context = DeviceFactory::CreateContext(entry, info, &status);
    CHECK(context && status == VK_SUCCESS && context->resources() == entry->resourcePool());
    CHECK(!context->timelineSemaphoreEnabled());
    {
        VkPhysicalDeviceTimelineSemaphoreFeatures supported{};
        supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &supported;
        vkGetPhysicalDeviceFeatures2(native->physical, &features);
        if (supported.timelineSemaphore) {
            VkPhysicalDeviceTimelineSemaphoreFeatures enabled{};
            enabled.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
            enabled.timelineSemaphore = VK_TRUE;
            auto timelineNative = CreateNative(&unavailable, {}, &enabled);
            CHECK(timelineNative);
            auto timelineInfo = Info(timelineNative);
            // Even an enabled device is not inferred enabled from support.
            auto unclaimed = DeviceFactory::CreateContext(entry, timelineInfo, &status);
            CHECK(unclaimed && !unclaimed->timelineSemaphoreEnabled());
            timelineInfo.timelineSemaphoreEnabled = true;
            auto claimed = DeviceFactory::CreateContext(entry, timelineInfo, &status);
            CHECK(claimed && claimed->timelineSemaphoreEnabled());
            CHECK(claimed->resources() == context->resources());
        }
    }
    auto secondNative = CreateNative(&unavailable);
    CHECK(secondNative && Uuid(secondNative->physical) == uuid);
    auto secondInfo = Info(secondNative);
    auto second = DeviceFactory::CreateContext(entry, secondInfo, &status);
    CHECK(second && status == VK_SUCCESS && second->resources() == context->resources());
    auto baseline = entry->resourcePool()->Snapshot();
    VkBufferCreateInfo create{}; create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create.size = 1028; create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto a = ChargedBuffer::Create(context, create, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &status);
    auto b = ChargedBuffer::Create(second, create, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &status);
    CHECK(a && b);
    CHECK(context->resources()->Snapshot().usedBytes == baseline.usedBytes + a->allocationBytes() + b->allocationBytes());
    a.reset(); b.reset();
    CHECK(entry->resourcePool()->Snapshot().usedBytes == baseline.usedBytes);
    replies.Drain();
    CHECK(replies.OutstandingCommands() == 0 && replies.CallbackFailures() == 0);
    return 0;
}
