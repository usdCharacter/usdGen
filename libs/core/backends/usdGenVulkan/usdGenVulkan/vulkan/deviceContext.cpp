// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "deviceContext.h"
#include "usdGenVulkan/backendId.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace usdGen::vulkan {
DeviceContext::DeviceContext(CreateInfo const& info, std::shared_ptr<UsdGenExecutionResourcePool> pool)
    : instance_(info.instance), physicalDevice_(info.physicalDevice), device_(info.device),
      computeQueue_(info.computeQueue), computeQueueFamily_(info.computeQueueFamily),
      physicalIndex_(info.physicalIndex), resourceDeviceId_(info.resourceDeviceId),
      gpuLabel_(info.gpuLabel), nativeLifetime_(info.nativeLifetime), resources_(std::move(pool)),
      timelineSemaphoreEnabled_(info.timelineSemaphoreEnabled),
      shaderFloat64Enabled_(info.shaderFloat64Enabled),
      shaderInt64Enabled_(info.shaderInt64Enabled) {
    VkPhysicalDeviceIDProperties ids{};
    ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &ids;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties);
    std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), deviceUUID_.begin());
}

std::shared_ptr<DeviceContext> DeviceContext::Create(CreateInfo const& info, VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!info.instance || !info.physicalDevice || !info.device || !info.computeQueue ||
        info.computeQueueFamily == UINT32_MAX || info.physicalIndex < 0 || info.resourceDeviceId < 0 || !info.nativeLifetime || !info.resources.limitBytes)
        return {};
    try {
        if (info.timelineSemaphoreEnabled) {
            VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
            timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &timeline;
            vkGetPhysicalDeviceFeatures2(info.physicalDevice, &features);
            if (!timeline.timelineSemaphore) {
                if (result) *result = VK_ERROR_FEATURE_NOT_PRESENT;
                return {};
            }
        }
        if (info.shaderFloat64Enabled || info.shaderInt64Enabled) {
            // Mirror of the timeline gate: the flag asserts the VkDevice was
            // created with shaderFloat64 enabled (factory contract); physical
            // support is re-verified so a lying producer fails at Create.
            VkPhysicalDeviceFeatures2 fp64{};
            fp64.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            vkGetPhysicalDeviceFeatures2(info.physicalDevice, &fp64);
            if ((info.shaderFloat64Enabled && !fp64.features.shaderFloat64) ||
                (info.shaderInt64Enabled && !fp64.features.shaderInt64)) {
                if (result) *result = VK_ERROR_FEATURE_NOT_PRESENT;
                return {};
            }
        }
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(info.physicalDevice, &familyCount, nullptr);
        if (info.computeQueueFamily >= familyCount) return {};
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(info.physicalDevice, &familyCount, families.data());
        auto const& family = families[info.computeQueueFamily];
        if (!family.queueCount || !(family.queueFlags & VK_QUEUE_COMPUTE_BIT)) return {};
        auto pool = GetOrCreateUsdGenExecutionResourcePool(
            {UsdGenVulkanBackendId(), info.resourceDeviceId}, info.resources);
        if (!pool) return {};
        auto context = std::shared_ptr<DeviceContext>(new DeviceContext(info, std::move(pool)));
        // Idle device memory kept for reuse: a quarter of the budget, at
        // least 16 MiB.
        VkDeviceSize const retain = std::max<VkDeviceSize>(VkDeviceSize(16) << 20,
                                                           VkDeviceSize(info.resources.limitBytes / 4));
        context->memoryPool_ = std::make_unique<VulkanDeviceMemoryPool>(info.device, retain);
        if (info.timelineSemaphoreEnabled) {
            VkResult tr = VK_SUCCESS;
            context->timeline_ = VulkanQueueTimeline::Create(info.device, info.computeQueue, &tr);
            if (!context->timeline_) {
                if (result) *result = tr;
                return {};
            }
        }
        if (result) *result = VK_SUCCESS;
        return context;
    } catch (...) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}
} // namespace usdGen::vulkan
