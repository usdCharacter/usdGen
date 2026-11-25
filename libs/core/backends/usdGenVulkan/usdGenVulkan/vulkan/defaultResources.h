// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_DEFAULT_RESOURCES_H
#define USDGEN_VULKAN_DEFAULT_RESOURCES_H

#include "usdGen/executionResources.h"

#include <vulkan/vulkan.h>
#include <algorithm>
#include <limits>

namespace usdGen::vulkan {

// Match ChargedBuffer's first eligible device-local memory type. All native
// allocations, including host-visible staging, share this conservative pool.
// EXT_memory_budget supplies a best-effort available-byte snapshot. Without
// that extension Vulkan exposes heap capacity, not third-party free memory;
// allocation failure remains the final guard against external competition.
inline UsdGenExecutionResourceConfig VulkanDefaultResourceConfig(
    VkPhysicalDeviceMemoryProperties const& memory,
    VkPhysicalDeviceMemoryBudgetPropertiesEXT const* budget = nullptr) noexcept {
    for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
        if (!(memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            continue;
        uint32_t const heap = memory.memoryTypes[type].heapIndex;
        if (heap >= memory.memoryHeapCount) return {};
        VkDeviceSize available = memory.memoryHeaps[heap].size;
        if (budget) {
            auto const capacity = std::min(available, budget->heapBudget[heap]);
            available = budget->heapUsage[heap] >= capacity
                ? 0 : capacity - budget->heapUsage[heap];
        }
        size_t const bytes = size_t(std::min<VkDeviceSize>(available,
            std::numeric_limits<size_t>::max()));
        // CUDA's default: retain at least 1 GiB or 20%, capped by availability.
        size_t const headroom = std::min(bytes, std::max(size_t{1} << 30, bytes / 5));
        return {bytes, headroom};
    }
    return {};
}

} // namespace usdGen::vulkan
#endif
