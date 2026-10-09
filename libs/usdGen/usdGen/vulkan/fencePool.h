// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// fencePool.h — a tiny freelist of idle VkFences, one per pipeline/binding.
//
// vkCreateFence/vkDestroyFence cost ~0.7ms each on the qualified NVIDIA
// driver (phase timers + nsys: median 729/753us), and every per-pose submit
// used to create and destroy its fences, so a deform pose paid ~2.2ms in
// fence lifecycle alone. Pooled fences are reset at checkout and behave
// exactly like fresh ones; only proven-idle fences are ever released back
// (a fence from a failed or unproven submission stays with its owner for
// quarantine or destruction, never pooled).
#ifndef USDGEN_VULKAN_FENCE_POOL_H
#define USDGEN_VULKAN_FENCE_POOL_H

#include <vulkan/vulkan.h>

#include <mutex>
#include <vector>

namespace usdGen::vulkan {

class VulkanFencePool
{
public:
    VulkanFencePool() = default;
    VulkanFencePool(VulkanFencePool const &) = delete;
    VulkanFencePool &operator=(VulkanFencePool const &) = delete;

    /// An unsignaled fence: a reset idle one when the pool holds one, else a
    /// fresh vkCreateFence. VK_NULL_HANDLE (with *result set) when creation
    /// fails. Thread-safe.
    inline VkFence Acquire(VkDevice device, VkResult *result = nullptr);

    /// Returns a proven-idle fence to the pool. Never throws and never
    /// fails: when the freelist cannot grow the fence is destroyed instead.
    /// Thread-safe. Only fences whose completion was proven (a successful
    /// wait or a VK_SUCCESS fence-status poll) may be released; an unproven
    /// fence must stay with (or be destroyed by) its owner.
    inline void Release(VkDevice device, VkFence fence) noexcept;

    /// Destroys every pooled fence. The pool must outlive its checkouts:
    /// owners hold their Native (and its pool) by shared_ptr.
    inline void Clear(VkDevice device) noexcept;

private:
    std::mutex mutex_;
    std::vector<VkFence> idle_;
};

inline VkFence VulkanFencePool::Acquire(VkDevice device, VkResult *result)
{
    auto finish = [&](VkResult r) { if (result) *result = r; };
    VkFence fence = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!idle_.empty()) {
            fence = idle_.back();
            idle_.pop_back();
        }
    }
    if (fence != VK_NULL_HANDLE) {
        // Reset outside the lock: an idle fence resets without blocking.
        if (vkResetFences(device, 1, &fence) == VK_SUCCESS) {
            finish(VK_SUCCESS);
            return fence;
        }
        vkDestroyFence(device, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkResult r = vkCreateFence(device, &info, nullptr, &fence);
    finish(r);
    return r == VK_SUCCESS ? fence : VK_NULL_HANDLE;
}

inline void VulkanFencePool::Release(VkDevice device, VkFence fence) noexcept
{
    if (fence == VK_NULL_HANDLE) return;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        idle_.push_back(fence);
    } catch (...) {
        vkDestroyFence(device, fence, nullptr);
    }
}

inline void VulkanFencePool::Clear(VkDevice device) noexcept
{
    std::vector<VkFence> fences;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        fences.swap(idle_);
    } catch (...) {
        return;
    }
    for (VkFence fence : fences) vkDestroyFence(device, fence, nullptr);
}

}  // namespace usdGen::vulkan

#endif  // USDGEN_VULKAN_FENCE_POOL_H
