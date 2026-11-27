// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "vulkanNativeFixture.h"
#include <cstring>
#include <poll.h>
#include <unistd.h>

struct NotificationOwner {
    std::shared_ptr<NativeOwner> native;
    VkFence fence = VK_NULL_HANDLE;
    ~NotificationOwner() { if (fence) vkDestroyFence(native->device, fence, nullptr); }
};
struct PendingGuard {
    std::unique_ptr<std::shared_ptr<NotificationOwner>> keep;
    bool pending = false;
    ~PendingGuard() { if (pending) (void)keep.release(); }
};
struct FdOwner {
    int value = -1;
    ~FdOwner() { if (value >= 0) close(value); }
};
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan fence FD check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

int main() {
    bool unavailable = false;
    auto probe = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(probe);
    uint32_t count = 0;
    CHECK(vkEnumerateDeviceExtensionProperties(probe->physical, nullptr, &count, nullptr) == VK_SUCCESS);
    std::vector<VkExtensionProperties> extensions(count);
    CHECK(vkEnumerateDeviceExtensionProperties(probe->physical, nullptr, &count, extensions.data()) == VK_SUCCESS);
    bool supported = false;
    for (auto const& extension : extensions)
        supported |= std::strcmp(extension.extensionName, VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME) == 0;
    VkPhysicalDeviceExternalFenceInfo query{};
    query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO;
    query.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    VkExternalFenceProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES;
    vkGetPhysicalDeviceExternalFenceProperties(probe->physical, &query, &properties);
    if (!supported || !(properties.externalFenceFeatures & VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT)) {
        std::puts("Vulkan SYNC_FD export unavailable");
        return 77;
    }
    probe.reset();
    auto native = CreateNative(&unavailable, {VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME});
    CHECK(native);
    auto exportFd = reinterpret_cast<PFN_vkGetFenceFdKHR>(vkGetDeviceProcAddr(native->device, "vkGetFenceFdKHR"));
    CHECK(exportFd);
    auto notification = std::make_shared<NotificationOwner>(); notification->native = native;
    VkExportFenceCreateInfo exportInfo{}; exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    VkFenceCreateInfo create{}; create.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; create.pNext = &exportInfo;
    CHECK(vkCreateFence(native->device, &create, nullptr, &notification->fence) == VK_SUCCESS);
    PendingGuard retained{std::make_unique<std::shared_ptr<NotificationOwner>>(notification)};
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    retained.pending = true;
    // The original proof fence is never exported or reset. Only a separate
    // follow-up notification fence uses SYNC_FD's copy-transfer semantics.
    CHECK(vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS);
    CHECK(vkQueueSubmit(native->queue, 1, &submit, notification->fence) == VK_SUCCESS);
    VkFenceGetFdInfoKHR get{}; get.sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR;
    get.fence = notification->fence; get.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    FdOwner fd;
    CHECK(exportFd(native->device, &get, &fd.value) == VK_SUCCESS);
    CHECK(fd.value >= -1);
    if (fd.value >= 0) {
        pollfd event{fd.value, POLLIN, 0};
        CHECK(poll(&event, 1, 10000) == 1);
        CHECK((event.revents & POLLIN) && !(event.revents & (POLLERR | POLLNVAL)));
        std::puts("Vulkan dedicated SYNC_FD notification: pollable fd signaled");
    } else {
        std::puts("Vulkan dedicated SYNC_FD notification: already signaled (-1)");
    }
    // Test-only wait verifies that export did not consume the original proof.
    CHECK(vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);
    CHECK(vkGetFenceStatus(native->device, native->fence) == VK_SUCCESS);
    retained.pending = false;
    return 0;
}
