// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Host-independent coverage for usdGen/vulkan/errors.h: every documented
// enumerant renders its canonical name, unknown values fall back without an
// empty diagnostic, and the one-line format keeps the numeric code.
#include "usdGenVulkan/vulkan/errors.h"

#include <cstdio>
#include <string>

using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan errors check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

int main() {
    CHECK(std::string(VulkanResultName(VK_SUCCESS)) == "VK_SUCCESS");
    CHECK(std::string(VulkanResultName(VK_NOT_READY)) == "VK_NOT_READY");
    CHECK(std::string(VulkanResultName(VK_TIMEOUT)) == "VK_TIMEOUT");
    CHECK(std::string(VulkanResultName(VK_EVENT_SET)) == "VK_EVENT_SET");
    CHECK(std::string(VulkanResultName(VK_EVENT_RESET)) == "VK_EVENT_RESET");
    CHECK(std::string(VulkanResultName(VK_INCOMPLETE)) == "VK_INCOMPLETE");
    CHECK(std::string(VulkanResultName(VK_ERROR_OUT_OF_HOST_MEMORY)) == "VK_ERROR_OUT_OF_HOST_MEMORY");
    CHECK(std::string(VulkanResultName(VK_ERROR_OUT_OF_DEVICE_MEMORY)) == "VK_ERROR_OUT_OF_DEVICE_MEMORY");
    CHECK(std::string(VulkanResultName(VK_ERROR_INITIALIZATION_FAILED)) == "VK_ERROR_INITIALIZATION_FAILED");
    CHECK(std::string(VulkanResultName(VK_ERROR_DEVICE_LOST)) == "VK_ERROR_DEVICE_LOST");
    CHECK(std::string(VulkanResultName(VK_ERROR_MEMORY_MAP_FAILED)) == "VK_ERROR_MEMORY_MAP_FAILED");
    CHECK(std::string(VulkanResultName(VK_ERROR_LAYER_NOT_PRESENT)) == "VK_ERROR_LAYER_NOT_PRESENT");
    CHECK(std::string(VulkanResultName(VK_ERROR_EXTENSION_NOT_PRESENT)) == "VK_ERROR_EXTENSION_NOT_PRESENT");
    CHECK(std::string(VulkanResultName(VK_ERROR_FEATURE_NOT_PRESENT)) == "VK_ERROR_FEATURE_NOT_PRESENT");
    CHECK(std::string(VulkanResultName(VK_ERROR_INCOMPATIBLE_DRIVER)) == "VK_ERROR_INCOMPATIBLE_DRIVER");
    CHECK(std::string(VulkanResultName(VK_ERROR_TOO_MANY_OBJECTS)) == "VK_ERROR_TOO_MANY_OBJECTS");
    CHECK(std::string(VulkanResultName(VK_ERROR_FORMAT_NOT_SUPPORTED)) == "VK_ERROR_FORMAT_NOT_SUPPORTED");
    CHECK(std::string(VulkanResultName(VK_ERROR_FRAGMENTED_POOL)) == "VK_ERROR_FRAGMENTED_POOL");
    CHECK(std::string(VulkanResultName(VK_ERROR_UNKNOWN)) == "VK_ERROR_UNKNOWN");
    CHECK(std::string(VulkanResultName(VK_ERROR_OUT_OF_POOL_MEMORY)) == "VK_ERROR_OUT_OF_POOL_MEMORY");
    CHECK(std::string(VulkanResultName(VK_ERROR_INVALID_EXTERNAL_HANDLE)) == "VK_ERROR_INVALID_EXTERNAL_HANDLE");
    CHECK(std::string(VulkanResultName(VK_ERROR_FRAGMENTATION)) == "VK_ERROR_FRAGMENTATION");
    CHECK(std::string(VulkanResultName(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)) ==
          "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS");
    CHECK(std::string(VulkanResultName(VK_SUBOPTIMAL_KHR)) == "VK_SUBOPTIMAL_KHR");
    CHECK(std::string(VulkanResultName(VK_ERROR_OUT_OF_DATE_KHR)) == "VK_ERROR_OUT_OF_DATE_KHR");
    // Future/extension values never render empty; the numeric code survives.
    CHECK(std::string(VulkanResultName(static_cast<VkResult>(0x7fffffff))) == "VK_UNKNOWN_RESULT");
    CHECK(std::string(VulkanResultName(static_cast<VkResult>(-99999))) == "VK_UNKNOWN_RESULT");
    CHECK(FormatVulkanError("queue submit", VK_ERROR_DEVICE_LOST) ==
          "queue submit: VK_ERROR_DEVICE_LOST (-4)");
    CHECK(FormatVulkanError("fence wait", VK_TIMEOUT) == "fence wait: VK_TIMEOUT (2)");
    CHECK(FormatVulkanError("probe", static_cast<VkResult>(-99999)) ==
          "probe: VK_UNKNOWN_RESULT (-99999)");
    return 0;
}
