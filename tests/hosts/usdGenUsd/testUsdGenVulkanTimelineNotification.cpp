// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "vulkanNativeFixture.h"

#include <atomic>
#include <climits>
#include <thread>

struct TimelineOwner {
    std::shared_ptr<NativeOwner> native;
    VkSemaphore completion = VK_NULL_HANDLE;
    VkSemaphore control = VK_NULL_HANDLE;
    ~TimelineOwner() {
        if (completion) vkDestroySemaphore(native->device, completion, nullptr);
        if (control) vkDestroySemaphore(native->device, control, nullptr);
    }
};

// Submission can leave native work pending even when the caller cannot
// continue the proof path.  Allocate the retention holder before submitting,
// so an uncertain return never destroys the device or timeline payloads.
struct PendingGuard {
    std::unique_ptr<std::shared_ptr<TimelineOwner>> keep;
    bool pending = false;
    ~PendingGuard() {
        if (pending) (void)keep.release();
    }
};

struct JoiningThread {
    std::thread value;
    ~JoiningThread() {
        if (value.joinable()) value.join();
    }
};

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan timeline notification check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static VkResult WaitAny(std::shared_ptr<TimelineOwner> const& owner,
                        uint64_t completionValue, uint64_t controlValue) {
    VkSemaphore semaphores[] = {owner->completion, owner->control};
    uint64_t values[] = {completionValue, controlValue};
    VkSemaphoreWaitInfo wait{};
    wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
    wait.semaphoreCount = 2;
    wait.pSemaphores = semaphores;
    wait.pValues = values;
    // This is a hardware-test bound, not a production notification loop.
    return vkWaitSemaphores(owner->native->device, &wait, 10000000000ull);
}

int main() {
    bool unavailable = false;
    auto probe = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(probe);

    VkPhysicalDeviceTimelineSemaphoreFeatures available{};
    available.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 featureQuery{};
    featureQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    featureQuery.pNext = &available;
    vkGetPhysicalDeviceFeatures2(probe->physical, &featureQuery);
    if (!available.timelineSemaphore) {
        std::puts("Vulkan timeline semaphore unavailable");
        return 77;
    }

    probe.reset();
    VkPhysicalDeviceTimelineSemaphoreFeatures enabled{};
    enabled.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    enabled.timelineSemaphore = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, &enabled);
    CHECK(native);

    auto timelines = std::make_shared<TimelineOwner>();
    timelines->native = native;
    VkSemaphoreTypeCreateInfo type{};
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    VkSemaphoreCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create.pNext = &type;
    CHECK(vkCreateSemaphore(native->device, &create, nullptr, &timelines->completion) == VK_SUCCESS);
    CHECK(vkCreateSemaphore(native->device, &create, nullptr, &timelines->control) == VK_SUCCESS);

    PendingGuard retained{std::make_unique<std::shared_ptr<TimelineOwner>>(timelines)};
    std::atomic<int> completionResult{INT_MIN};
    JoiningThread completionWaiter;
    try {
        completionWaiter.value = std::thread([timelines, &completionResult] {
            completionResult.store(int(WaitAny(timelines, 1, 1)), std::memory_order_release);
        });
    } catch (...) {
        return 1;
    }

    VkSubmitInfo originalSubmit{};
    originalSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    retained.pending = true;
    // The proof fence remains independent.  This same-queue follow-up has a
    // dedicated timeline signal and therefore cannot consume or reset it.
    CHECK(vkQueueSubmit(native->queue, 1, &originalSubmit, native->fence) == VK_SUCCESS);
    uint64_t completionValue = 1;
    VkTimelineSemaphoreSubmitInfo timelineSubmit{};
    timelineSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmit.signalSemaphoreValueCount = 1;
    timelineSubmit.pSignalSemaphoreValues = &completionValue;
    VkSubmitInfo signalSubmit{};
    signalSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    signalSubmit.pNext = &timelineSubmit;
    signalSubmit.signalSemaphoreCount = 1;
    signalSubmit.pSignalSemaphores = &timelines->completion;
    CHECK(vkQueueSubmit(native->queue, 1, &signalSubmit, VK_NULL_HANDLE) == VK_SUCCESS);

    if (completionWaiter.value.joinable()) completionWaiter.value.join();
    CHECK(completionResult.load(std::memory_order_acquire) == VK_SUCCESS);
    uint64_t counter = 0;
    CHECK(vkGetSemaphoreCounterValue(native->device, timelines->completion, &counter) == VK_SUCCESS && counter >= 1);
    CHECK(vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);
    CHECK(vkGetFenceStatus(native->device, native->fence) == VK_SUCCESS);
    retained.pending = false;

    // A fresh waiter asks for an unreachable GPU value or the host-control
    // value.  Host signaling control is the bounded shutdown wake path.
    std::atomic<int> controlResult{INT_MIN};
    JoiningThread controlWaiter;
    try {
        controlWaiter.value = std::thread([timelines, &controlResult] {
            controlResult.store(int(WaitAny(timelines, 2, 1)), std::memory_order_release);
        });
    } catch (...) {
        return 1;
    }
    VkSemaphoreSignalInfo wake{};
    wake.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    wake.semaphore = timelines->control;
    wake.value = 1;
    CHECK(vkSignalSemaphore(native->device, &wake) == VK_SUCCESS);
    if (controlWaiter.value.joinable()) controlWaiter.value.join();
    CHECK(controlResult.load(std::memory_order_acquire) == VK_SUCCESS);
    CHECK(vkGetSemaphoreCounterValue(native->device, timelines->control, &counter) == VK_SUCCESS && counter == 1);
    return 0;
}
