// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "queueTimeline.h"

#include "usdGen/parkingWord.h"

namespace usdGen::vulkan {

// One producer's call, on its stack until the submitter marks it done.
struct VulkanQueueTimeline::Request {
    VkCommandBuffer const* commands = nullptr;
    uint32_t count = 0;  // 0: one empty batch (Signal)
    Request* next = nullptr;
    uint64_t last = 0;
    VkResult result = VK_SUCCESS;
    // kQueued -> kDone; kWaiter is or'ed in by a producer about to park.
    std::atomic<uint32_t> state{0};
    bool combined = false;
};

namespace {
constexpr uint32_t kQueued = 0, kDone = 1, kWaiter = 2;
constexpr uint32_t kMaxBatches = VulkanQueueTimeline::kMaxBatches;
std::atomic<uint64_t> g_waits{0};
}

uint64_t VulkanQueueTimelineWaitsForTesting() noexcept {
    return g_waits.load(std::memory_order_relaxed);
}

VulkanQueueTimeline::VulkanQueueTimeline(VkDevice d, VkQueue q, VkSemaphore s) noexcept
    : device_(d), queue_(q), semaphore_(s) {}

std::unique_ptr<VulkanQueueTimeline> VulkanQueueTimeline::Create(VkDevice device, VkQueue queue,
                                                                 VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!device || !queue) return {};
    VkSemaphoreTypeCreateInfo type{};
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &type;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VkResult r = vkCreateSemaphore(device, &info, nullptr, &semaphore);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    std::unique_ptr<VulkanQueueTimeline> timeline(new (std::nothrow) VulkanQueueTimeline(device, queue, semaphore));
    if (!timeline) {
        vkDestroySemaphore(device, semaphore, nullptr);
        finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
    finish(VK_SUCCESS);
    return timeline;
}

VulkanQueueTimeline::~VulkanQueueTimeline() {
    if (semaphore_) vkDestroySemaphore(device_, semaphore_, nullptr);
}

uint64_t VulkanQueueTimeline::Submit(VkCommandBuffer const* commands, uint32_t count,
                                     VkResult* result) noexcept {
    if (!commands || count == 0 || count > kMaxBatches) {
        if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
        return 0;
    }
    return Enqueue(commands, count, result);
}

uint64_t VulkanQueueTimeline::Signal(VkResult* result) noexcept {
    return Enqueue(nullptr, 0, result);
}

uint64_t VulkanQueueTimeline::Enqueue(VkCommandBuffer const* commands, uint32_t count,
                                      VkResult* result) noexcept {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    Request request;
    request.commands = commands;
    request.count = count;
    // Push (seq_cst pairs with the submitter's release-then-recheck below).
    Request* head = pending_.load(std::memory_order_relaxed);
    do { request.next = head; }
    while (!pending_.compare_exchange_weak(head, &request, std::memory_order_seq_cst,
                                           std::memory_order_relaxed));
    Drain(&request);
    uint32_t s = request.state.load(std::memory_order_acquire);
    if (!(s & kDone)) {
        // Another thread holds the submit role and has seen (or will see,
        // on its recheck) this request. Park until it marks it done.
        parks_.fetch_add(1, std::memory_order_relaxed);
        while (!(s & kDone)) {
            if (!(s & kWaiter) &&
                !request.state.compare_exchange_weak(s, s | kWaiter, std::memory_order_acq_rel,
                                                     std::memory_order_acquire))
                continue;
            UsdGenParkWhileEqual(request.state, kQueued | kWaiter);
            s = request.state.load(std::memory_order_acquire);
        }
    }
    if (request.combined) combined_.fetch_add(1, std::memory_order_relaxed);
    finish(request.result);
    return request.result == VK_SUCCESS ? request.last : 0;
}

void VulkanQueueTimeline::Drain(Request* self) noexcept {
    // Take the role, submit everything queued, give the role back, and
    // recheck: a push that lost the race for the role relies on this.
    while (pending_.load(std::memory_order_seq_cst)) {
        if (submitting_.exchange(1, std::memory_order_seq_cst)) return;
        Request* list = pending_.exchange(nullptr, std::memory_order_acquire);
        // The stack is newest-first; submit in arrival order.
        Request* ordered = nullptr;
        while (list) {
            Request* next = list->next;
            list->next = ordered;
            ordered = list;
            list = next;
        }
        while (ordered) {
            Request* next = ordered->next;  // read before the request is released
            ordered->combined = ordered != self;
            SubmitOne(*ordered);
            ordered = next;
        }
        submitting_.store(0, std::memory_order_seq_cst);
    }
}

void VulkanQueueTimeline::SubmitOne(Request& request) noexcept {
    uint64_t const base = submitted_.load(std::memory_order_relaxed);
    uint32_t const count = request.count ? request.count : 1;
    uint64_t values[kMaxBatches];
    VkTimelineSemaphoreSubmitInfo timeline[kMaxBatches]{};
    VkSubmitInfo batches[kMaxBatches]{};
    for (uint32_t i = 0; i < count; ++i) {
        values[i] = base + 1 + i;
        timeline[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline[i].signalSemaphoreValueCount = 1;
        timeline[i].pSignalSemaphoreValues = &values[i];
        batches[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        batches[i].pNext = &timeline[i];
        batches[i].commandBufferCount = request.count ? 1 : 0;
        batches[i].pCommandBuffers = request.count ? &request.commands[i] : nullptr;
        batches[i].signalSemaphoreCount = 1;
        batches[i].pSignalSemaphores = &semaphore_;
    }
    submits_.fetch_add(1, std::memory_order_relaxed);
    VkResult r = vkQueueSubmit(queue_, count, batches, VK_NULL_HANDLE);
    if (r == VK_SUCCESS) submitted_.store(base + count, std::memory_order_release);
    request.result = r;
    request.last = r == VK_SUCCESS ? base + count : 0;
    uint32_t const old = request.state.exchange(kDone, std::memory_order_acq_rel);
    // Nothing of the request is read after this point: its owner may return.
    if (old & kWaiter) UsdGenUnparkAll(request.state);
}

void VulkanQueueTimeline::Observe(uint64_t value) noexcept {
    uint64_t seen = completed_.load(std::memory_order_relaxed);
    while (seen < value &&
           !completed_.compare_exchange_weak(seen, value, std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {}
}

VkResult VulkanQueueTimeline::Check(uint64_t value) noexcept {
    if (value <= completed_.load(std::memory_order_acquire)) return VK_SUCCESS;
    uint64_t now = 0;
    VkResult r = vkGetSemaphoreCounterValue(device_, semaphore_, &now);
    if (r != VK_SUCCESS) return r;
    Observe(now);
    return now >= value ? VK_SUCCESS : VK_NOT_READY;
}

VkResult VulkanQueueTimeline::Poll(uint64_t* value) noexcept {
    uint64_t now = 0;
    VkResult r = vkGetSemaphoreCounterValue(device_, semaphore_, &now);
    if (r != VK_SUCCESS) return r;
    Observe(now);
    if (value) *value = now;
    return VK_SUCCESS;
}

VkResult VulkanQueueTimeline::Wait(uint64_t value, uint64_t timeoutNs) noexcept {
    g_waits.fetch_add(1, std::memory_order_relaxed);
    VkResult r = Check(value);
    if (r != VK_NOT_READY) return r;
    VkSemaphoreWaitInfo wait{};
    wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait.semaphoreCount = 1;
    wait.pSemaphores = &semaphore_;
    wait.pValues = &value;
    r = vkWaitSemaphores(device_, &wait, timeoutNs);
    if (r == VK_SUCCESS) Observe(value);
    return r;
}

VulkanQueueTimeline::Stats VulkanQueueTimeline::stats() const noexcept {
    Stats s;
    s.submits = submits_.load(std::memory_order_relaxed);
    s.combined = combined_.load(std::memory_order_relaxed);
    s.parks = parks_.load(std::memory_order_relaxed);
    return s;
}

} // namespace usdGen::vulkan
