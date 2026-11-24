// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_DEVICE_FACTORY_H
#define USDGEN_VULKAN_DEVICE_FACTORY_H

#include "usdGen/executionPipeline.h"
#include "usdGen/executionResources.h"

#include <vulkan/vulkan.h>

#include <array>
#include <functional>
#include <memory>
#include <string>

namespace usdGen::vulkan {

class DeviceContext;

// Canonical Vulkan physical-device identity. The factory assigns the resource
// identity after reading the physical-device UUID; callers never supply a
// resource-device id.
class DeviceFactory {
public:
    class Entry {
    public:
        std::array<uint8_t, VK_UUID_SIZE> const& deviceUUID() const noexcept {
            return deviceUUID_;
        }
        int resourceDeviceId() const noexcept { return resourceDeviceId_; }
        UsdGenExecutionResourceConfig const& resources() const noexcept {
            return resources_;
        }
        std::shared_ptr<UsdGenExecutionResourcePool> const& resourcePool() const noexcept {
            return resourcePool_;
        }

    private:
        Entry(std::array<uint8_t, VK_UUID_SIZE>, int,
              UsdGenExecutionResourceConfig,
              std::shared_ptr<UsdGenExecutionResourcePool>);

        std::array<uint8_t, VK_UUID_SIZE> deviceUUID_{};
        int resourceDeviceId_ = -1;
        UsdGenExecutionResourceConfig resources_;
        std::shared_ptr<UsdGenExecutionResourcePool> resourcePool_;
        friend class DeviceFactory;
        friend struct RegistryAuthority;
    };

    struct CreateInfo {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeQueueFamily = UINT32_MAX;
        int physicalIndex = -1;
        // Owns the externally created instance/device/queue relationship.
        std::shared_ptr<const void> nativeLifetime;
        std::string gpuLabel;
        // True only when the supplied logical device was created with this
        // feature enabled; this is not merely a physical capability flag.
        bool timelineSemaphoreEnabled = false;
        bool shaderFloat64Enabled = false;
        bool shaderInt64Enabled = false;
    };

    enum class ResolveOutcome { Resolved, Rejected, Cancelled };
    using ResolveCompletion = std::function<void(std::shared_ptr<const Entry>,
                                                 ResolveOutcome)>;

    // Resolves only immutable UUID/configuration state. `replyLifetime` keeps
    // caller-owned reply state alive; it must not own Vulkan native objects.
    // Resolved/Rejected callbacks run on replyOwner; Rejected has a null Entry.
    // A Cancelled callback has a null Entry and can run in the registry owner
    // only when replyOwner rejects the reserved reply or registry shutdown
    // discards its command; that exception must be thread-neutral cleanup.
    // Resolver callbacks/lifetime must not create, destroy, or be the final
    // owner of native Vulkan objects. The caller must stop/join replyOwner
    // only after its terminal callback, and replyLifetime must not be the last
    // owner of replyOwner. False means ingress was not admitted and invokes no
    // callback. Every accepted request reports exactly one outcome.
    // Explicit configurations match exactly by default. Pass false only for
    // automatic available-memory snapshots: the first configuration remains
    // authoritative, as on CUDA, even when subsequent free memory changes.
    static bool ResolveAsync(std::array<uint8_t, VK_UUID_SIZE> const& deviceUUID,
                             UsdGenExecutionResourceConfig resources,
                             UsdGenExecutionPipeline& replyOwner,
                             std::shared_ptr<const void> replyLifetime,
                             ResolveCompletion completion,
                             bool requireMatchingConfig = true);

    // Native validation and DeviceContext construction are deliberately
    // separate from ResolveAsync and never run in registry-owner access.
    static std::shared_ptr<DeviceContext> CreateContext(
        std::shared_ptr<const Entry> const&, CreateInfo const&, VkResult* result = nullptr);
};

} // namespace usdGen::vulkan

#endif
