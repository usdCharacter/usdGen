// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "deviceFactory.h"
#include "usdGenVulkan/backendId.h"

#include "deviceContext.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <utility>

namespace usdGen::vulkan {
using Uuid = std::array<uint8_t, VK_UUID_SIZE>;

struct RegistryAuthority {
    UsdGenExecutionRuntime runtime{1};
    UsdGenExecutionPipeline owner{runtime};
    std::map<Uuid, std::shared_ptr<const DeviceFactory::Entry>> entries;
    int nextResourceDeviceId = 0;

    std::shared_ptr<const DeviceFactory::Entry> Resolve(
        Uuid const&, UsdGenExecutionResourceConfig, bool requireMatchingConfig);
};

namespace {

Uuid QueryDeviceUuid(VkPhysicalDevice physicalDevice) noexcept {
    VkPhysicalDeviceIDProperties ids{};
    ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &ids;
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
    Uuid uuid{};
    std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), uuid.begin());
    return uuid;
}

bool SameConfig(UsdGenExecutionResourceConfig const& a,
                UsdGenExecutionResourceConfig const& b) noexcept {
    return a.limitBytes == b.limitBytes && a.headroomBytes == b.headroomBytes;
}

// This authority intentionally leaks at process exit: a quarantined Vulkan
// allocation must never lose its canonical accounting state during static
// teardown while native callbacks/lifetimes can still exist.
// Published by CAS on first use (no guarded static); a racing loser's
// authority is discarded before anyone saw it.
std::atomic<RegistryAuthority*> gAuthority{nullptr};
RegistryAuthority& Registry() {
    if (auto* authority = gAuthority.load(std::memory_order_acquire)) return *authority;
    auto* fresh = new RegistryAuthority;
    RegistryAuthority* expected = nullptr;
    if (gAuthority.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel)) return *fresh;
    delete fresh;
    return *expected;
}

} // namespace

std::shared_ptr<const DeviceFactory::Entry> RegistryAuthority::Resolve(
    Uuid const& uuid, UsdGenExecutionResourceConfig config, bool requireMatchingConfig) {
    auto existing = entries.find(uuid);
    if (existing != entries.end()) {
        return !requireMatchingConfig || SameConfig(existing->second->resources(), config)
            ? existing->second : nullptr;
    }
    if (!config.limitBytes || nextResourceDeviceId == std::numeric_limits<int>::max())
        return {};
    int const resourceDeviceId = nextResourceDeviceId++;
    auto pool = GetOrCreateUsdGenExecutionResourcePool(
        {UsdGenVulkanBackendId(), resourceDeviceId}, config);
    if (!pool) return {};
    auto entry = std::shared_ptr<const DeviceFactory::Entry>(new DeviceFactory::Entry(
        uuid, resourceDeviceId, config, std::move(pool)));
    entries.emplace(uuid, entry);
    return entry;
}

namespace {
std::shared_ptr<DeviceContext> BuildContext(DeviceFactory::CreateInfo const& info,
                                            Uuid const& queriedUuid,
                                            std::shared_ptr<const DeviceFactory::Entry> const& entry,
                                            VkResult* result) {
    if (!entry || entry->deviceUUID() != queriedUuid ||
        QueryDeviceUuid(info.physicalDevice) != entry->deviceUUID()) {
        if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
        return {};
    }
    DeviceContext::CreateInfo contextInfo;
    contextInfo.instance = info.instance;
    contextInfo.physicalDevice = info.physicalDevice;
    contextInfo.device = info.device;
    contextInfo.computeQueue = info.computeQueue;
    contextInfo.computeQueueFamily = info.computeQueueFamily;
    contextInfo.physicalIndex = info.physicalIndex;
    contextInfo.resourceDeviceId = entry->resourceDeviceId();
    contextInfo.nativeLifetime = info.nativeLifetime;
    contextInfo.gpuLabel = info.gpuLabel;
    contextInfo.resources = entry->resources();
    contextInfo.timelineSemaphoreEnabled = info.timelineSemaphoreEnabled;
    contextInfo.shaderFloat64Enabled = info.shaderFloat64Enabled;
    contextInfo.shaderInt64Enabled = info.shaderInt64Enabled;
    auto context = DeviceContext::Create(contextInfo, result);
    if (context && context->resources() != entry->resourcePool()) {
        if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
        return {};
    }
    return context;
}
} // namespace

DeviceFactory::Entry::Entry(Uuid uuid, int resourceDeviceId,
                            UsdGenExecutionResourceConfig resources,
                            std::shared_ptr<UsdGenExecutionResourcePool> resourcePool)
    : deviceUUID_(uuid), resourceDeviceId_(resourceDeviceId), resources_(resources),
      resourcePool_(std::move(resourcePool)) {}

bool DeviceFactory::ResolveAsync(Uuid const& deviceUUID,
                                 UsdGenExecutionResourceConfig resources,
                                 UsdGenExecutionPipeline& replyOwner,
                                 std::shared_ptr<const void> replyLifetime,
                                 ResolveCompletion completion, bool requireMatchingConfig) {
    if (!replyLifetime || !completion) return false;
    auto ticket = replyOwner.ReserveCommandTicket();
    if (!ticket) return false;

    // The relay is copyable for both pipeline callbacks while its ticket stays
    // move-only. Complete/Cancel are mutually exclusive and release captures
    // exactly once, including registry cancellation or a failed reply post.
    struct Relay : std::enable_shared_from_this<Relay> {
        UsdGenExecutionPipeline* replyOwner = nullptr;
        UsdGenExecutionPipeline::CommandTicket ticket;
        std::shared_ptr<const void> lifetime;
        ResolveCompletion completion;
        std::atomic<bool> settled{false};

        void Finish(std::shared_ptr<const Entry> entry, ResolveOutcome outcome) noexcept {
            if (settled.exchange(true, std::memory_order_acq_rel)) return;
            auto keepAlive = std::move(lifetime);
            auto callback = std::move(completion);
            completion = {};
            ticket = {};
            try { callback(std::move(entry), outcome); }
            catch (...) {}
        }
        void Cancel() noexcept {
            Finish({}, ResolveOutcome::Cancelled);
        }
        void Discard() noexcept {
            if (settled.exchange(true, std::memory_order_acq_rel)) return;
            completion = {};
            lifetime.reset();
            ticket = {};
        }
        void Complete(std::shared_ptr<const Entry> entry, ResolveOutcome outcome) noexcept {
            Finish(std::move(entry), outcome);
        }
        void Post(std::shared_ptr<const Entry> entry, ResolveOutcome outcome) noexcept {
            auto self = std::shared_ptr<Relay>();
            try { self = shared_from_this(); }
            catch (...) { Cancel(); return; }
            bool posted = false;
            try {
                posted = replyOwner->PostCommand(std::move(ticket),
                    [self, entry=std::move(entry), outcome]() mutable {
                        self->Complete(std::move(entry), outcome);
                    },
                    [self] { self->Cancel(); });
            } catch (...) { Cancel(); return; }
            if (!posted) Cancel();
        }
    };
    auto relay = std::make_shared<Relay>();
    relay->replyOwner = &replyOwner;
    relay->ticket = std::move(ticket);
    relay->lifetime = std::move(replyLifetime);
    relay->completion = std::move(completion);

    auto& registry = Registry();
    bool posted = registry.owner.PostCommand(
        [deviceUUID, resources, requireMatchingConfig, relay, &registry]() mutable {
            std::shared_ptr<const Entry> entry;
            try { entry = registry.Resolve(deviceUUID, resources, requireMatchingConfig); }
            catch (...) { entry.reset(); }
            relay->Post(entry, entry ? ResolveOutcome::Resolved : ResolveOutcome::Rejected);
        },
        [relay] { relay->Cancel(); });
    if (!posted) relay->Discard();
    return posted;
}

std::shared_ptr<DeviceContext> DeviceFactory::CreateContext(
    std::shared_ptr<const Entry> const& entry, CreateInfo const& info,
    VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!info.physicalDevice) return {};
    return BuildContext(info, QueryDeviceUuid(info.physicalDevice), entry, result);
}

} // namespace usdGen::vulkan
