// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenVulkan/vulkan/defaultResources.h"

#include <cstdio>
#include <limits>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Default resource policy failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

int main() {
    using namespace usdGen::vulkan;
    constexpr VkDeviceSize GiB = VkDeviceSize{1} << 30;
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 2;
    memory.memoryHeaps[0] = {64 * GiB, 0};
    memory.memoryHeaps[1] = {16 * GiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    memory.memoryTypeCount = 2;
    memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0};
    memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};
    auto config = VulkanDefaultResourceConfig(memory);
    CHECK(config.limitBytes == 16 * GiB && config.headroomBytes == (16 * GiB) / 5);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    budget.heapBudget[1] = 14 * GiB;
    budget.heapUsage[1] = 4 * GiB;
    config = VulkanDefaultResourceConfig(memory, &budget);
    CHECK(config.limitBytes == 10 * GiB && config.headroomBytes == 2 * GiB);
    // More than the old fixed 64 MiB is usable even under a small budget.
    budget.heapBudget[1] = GiB + (GiB >> 3);
    budget.heapUsage[1] = 0;
    config = VulkanDefaultResourceConfig(memory, &budget);
    CHECK(config.headroomBytes == GiB);
    CHECK(config.limitBytes - config.headroomBytes == GiB / 8);
    // Headroom saturates below 1 GiB, matching CUDA, without underflow.
    budget.heapBudget[1] = GiB / 2;
    config = VulkanDefaultResourceConfig(memory, &budget);
    CHECK(config.limitBytes == GiB / 2 && config.headroomBytes == config.limitBytes);
    budget.heapUsage[1] = GiB / 2;
    CHECK(VulkanDefaultResourceConfig(memory, &budget).limitBytes == 0);
    budget.heapUsage[1] = GiB;
    CHECK(VulkanDefaultResourceConfig(memory, &budget).limitBytes == 0);
    budget.heapBudget[1] = std::numeric_limits<VkDeviceSize>::max();
    budget.heapUsage[1] = 0;
    CHECK(VulkanDefaultResourceConfig(memory, &budget).limitBytes == memory.memoryHeaps[1].size);
    budget.heapBudget[1] = 0;
    CHECK(VulkanDefaultResourceConfig(memory, &budget).limitBytes == 0);
    memory.memoryTypes[1].heapIndex = memory.memoryHeapCount;
    CHECK(VulkanDefaultResourceConfig(memory).limitBytes == 0);
    memory.memoryTypeCount = 1;
    CHECK(VulkanDefaultResourceConfig(memory).limitBytes == 0);
    std::puts("testUsdGenVulkanDefaultResources: PASS");
}
