// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "vulkanNativeFixture.h"

#include <cstring>
#include <poll.h>
#include <unistd.h>

struct NotificationOwner {
    std::shared_ptr<NativeOwner> native;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    ~NotificationOwner() {
        if (semaphore) vkDestroySemaphore(native->device, semaphore, nullptr);
    }
};

// Once a submission has been attempted, failures must retain both the native
// queue proof and notification semaphore rather than destroy possibly-live
// Vulkan objects.  The owner is allocated before submitting so this path is
// noexcept and cannot itself lose the proof.
struct PendingGuard {
    std::unique_ptr<std::shared_ptr<NotificationOwner>> keep;
    bool pending = false;
    ~PendingGuard() {
        if (pending) (void)keep.release();
    }
};

struct FdOwner {
    int value = -1;
    ~FdOwner() {
        if (value >= 0) close(value);
    }
};

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan semaphore FD check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

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
        supported |= std::strcmp(extension.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) == 0;

    VkPhysicalDeviceExternalSemaphoreInfo query{};
    query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    query.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkExternalSemaphoreProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(probe->physical, &query, &properties);
    if (!supported || !(properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)) {
        std::puts("Vulkan semaphore SYNC_FD export unavailable");
        return 77;
    }

    probe.reset();
    auto native = CreateNative(&unavailable, {VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME});
    CHECK(native);
    auto exportFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(native->device, "vkGetSemaphoreFdKHR"));
    CHECK(exportFd);

    auto notification = std::make_shared<NotificationOwner>();
    notification->native = native;
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkSemaphoreCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create.pNext = &exportInfo;
    CHECK(vkCreateSemaphore(native->device, &create, nullptr, &notification->semaphore) == VK_SUCCESS);

    PendingGuard retained{std::make_unique<std::shared_ptr<NotificationOwner>>(notification)};
    VkSubmitInfo originalSubmit{};
    originalSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    retained.pending = true;
    // The dedicated, exportable binary semaphore is signaled only by this
    // ordered follow-up submission.  The original proof fence is neither
    // exported nor reset.
    CHECK(vkQueueSubmit(native->queue, 1, &originalSubmit, native->fence) == VK_SUCCESS);
    VkSubmitInfo notificationSubmit{};
    notificationSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    notificationSubmit.signalSemaphoreCount = 1;
    notificationSubmit.pSignalSemaphores = &notification->semaphore;
    CHECK(vkQueueSubmit(native->queue, 1, &notificationSubmit, VK_NULL_HANDLE) == VK_SUCCESS);

    VkSemaphoreGetFdInfoKHR get{};
    get.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    get.semaphore = notification->semaphore;
    get.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    FdOwner fd;
    CHECK(exportFd(native->device, &get, &fd.value) == VK_SUCCESS);
    CHECK(fd.value >= -1);
    if (fd.value >= 0) {
        pollfd event{fd.value, POLLIN, 0};
        CHECK(poll(&event, 1, 10000) == 1);
        CHECK((event.revents & POLLIN) && !(event.revents & (POLLERR | POLLNVAL)));
        std::puts("Vulkan dedicated semaphore SYNC_FD notification: pollable fd signaled");
    } else {
        std::puts("Vulkan dedicated semaphore SYNC_FD notification: already signaled (-1)");
    }

    // Test-only proof that the export did not consume the original fence.
    CHECK(vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);
    CHECK(vkGetFenceStatus(native->device, native->fence) == VK_SUCCESS);
    retained.pending = false;
    return 0;
}
