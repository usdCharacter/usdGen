// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_TEST_VULKAN_NATIVE_H
#define USDGEN_TEST_VULKAN_NATIVE_H

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>

// Native lifetime is real ownership, not a dummy token. Quarantined buffers
// retain this object, including the fence/command pool needed by pending work.
struct NativeOwner {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    int physicalIndex = -1;
    VkPhysicalDeviceFeatures supportedFeatures{};
    VkPhysicalDeviceFeatures enabledCoreFeatures{};
    ~NativeOwner() {
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

static std::shared_ptr<NativeOwner> CreateNative(bool* unavailable,
    std::vector<char const*> const& extensions = {}, void const* enabledFeatures = nullptr,
    VkPhysicalDeviceFeatures const* coreFeatures = nullptr) {
    *unavailable = false;
    auto native = std::make_shared<NativeOwner>();
    VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "usdGen Vulkan resource proof";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create{}; create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &app;
    VkResult result = vkCreateInstance(&create, nullptr, &native->instance);
    if (result != VK_SUCCESS) {
        *unavailable = result == VK_ERROR_INCOMPATIBLE_DRIVER;
        std::fprintf(stderr, "Vulkan fixture: vkCreateInstance failed (%d)\n", int(result));
        return {};
    }
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(native->instance, &count, nullptr) != VK_SUCCESS) return {};
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(native->instance, &count, devices.data()) != VK_SUCCESS) return {};
    for (uint32_t index = 0; index < count && !native->physical; ++index) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[index], &props);
        if (props.apiVersion < VK_API_VERSION_1_2) continue;
        char const* hardware = std::getenv("USDGEN_VULKAN_REQUIRE_HARDWARE");
        if (hardware && std::strcmp(hardware, "1") == 0 &&
            props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) continue;
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(devices[index], &supported);
        if (coreFeatures) {
            // VkPhysicalDeviceFeatures consists entirely of VkBool32 fields.
            VkBool32 requested[sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32)]{};
            VkBool32 available[sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32)]{};
            std::memcpy(requested, coreFeatures, sizeof(requested));
            std::memcpy(available, &supported, sizeof(available));
            bool missing = false;
            for (size_t f = 0; f < sizeof(requested) / sizeof(requested[0]); ++f)
                if (requested[f] && !available[f]) missing = true;
            if (missing) continue;
        }
        bool missingExtension = false;
        if (!extensions.empty()) {
            uint32_t extensionCount = 0;
            if (vkEnumerateDeviceExtensionProperties(devices[index], nullptr,
                    &extensionCount, nullptr) != VK_SUCCESS) return {};
            std::vector<VkExtensionProperties> available(extensionCount);
            if (vkEnumerateDeviceExtensionProperties(devices[index], nullptr,
                    &extensionCount, available.data()) != VK_SUCCESS) return {};
            for (char const* requested : extensions) {
                bool found = false;
                for (auto const& ext : available)
                    if (std::strcmp(requested, ext.extensionName) == 0) found = true;
                if (!found) missingExtension = true;
            }
        }
        if (missingExtension) continue;
        // Query the feature chain used by the timeline tests before enabling it.
        bool missingChainedFeature = false;
        for (auto feature = static_cast<VkBaseInStructure const*>(enabledFeatures);
             feature; feature = feature->pNext) {
            if (feature->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
                VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
                timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
                VkPhysicalDeviceFeatures2 features{};
                features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                features.pNext = &timeline;
                vkGetPhysicalDeviceFeatures2(devices[index], &features);
                auto const* requested = reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreFeatures const*>(feature);
                if (requested->timelineSemaphore && !timeline.timelineSemaphore)
                    missingChainedFeature = true;
            }
        }
        if (missingChainedFeature) continue;
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, nullptr);
        std::vector<VkQueueFamilyProperties> info(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, info.data());
        for (uint32_t f = 0; f < families; ++f) {
            if (!info[f].queueCount || !(info[f].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            native->physical = devices[index]; native->physicalIndex = int(index);
            native->family = f;
            native->supportedFeatures = supported;
            if (coreFeatures) native->enabledCoreFeatures = *coreFeatures;
            std::printf("Vulkan resource device: %s\n", props.deviceName);
            break;
        }
    }
    if (!native->physical) { *unavailable = true; return {}; }
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue{}; queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = native->family; queue.queueCount = 1; queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo device{}; device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.pNext = enabledFeatures;
    device.pEnabledFeatures = coreFeatures;
    device.queueCreateInfoCount = 1; device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = uint32_t(extensions.size());
    device.ppEnabledExtensionNames = extensions.data();
    result = vkCreateDevice(native->physical, &device, nullptr, &native->device);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Vulkan fixture: vkCreateDevice failed (%d)\n", int(result));
        return {};
    }
    vkGetDeviceQueue(native->device, native->family, 0, &native->queue);
    VkCommandPoolCreateInfo pool{}; pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = native->family;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(native->device, &pool, nullptr, &native->commands) != VK_SUCCESS) return {};
    VkFenceCreateInfo fence{}; fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(native->device, &fence, nullptr, &native->fence) != VK_SUCCESS) return {};
    return native;
}

// A native device with the timeline semaphore feature enabled, for contexts
// whose pipelines submit on the queue timeline (set
// DeviceContext::CreateInfo::timelineSemaphoreEnabled with it).
static std::shared_ptr<NativeOwner> CreateTimelineNative(bool* unavailable,
    VkPhysicalDeviceFeatures const* coreFeatures = nullptr,
    std::vector<char const*> const& extensions = {}) {
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline.timelineSemaphore = VK_TRUE;
    return CreateNative(unavailable, extensions, &timeline, coreFeatures);
}

#endif
