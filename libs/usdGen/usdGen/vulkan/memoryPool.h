// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// memoryPool.h — size-class device memory for ChargedBuffer. Each block is
// its own VkDeviceMemory rounded up to a size class (quarter-power-of-two
// steps above 4 KiB, so at most 25% slack), bound at offset 0, so every
// existing map/bind of buffer->memory() is unchanged. A released block goes
// back to its class's idle stack and the next buffer of that class reuses
// it, so steady-state cooks make no vkAllocateMemory/vkFreeMemory call: the
// pool grows only when a new size appears (a topology change). Idle memory
// is capped; past the cap a release frees instead.
//
// Lock-free: idle stacks are Treiber stacks with push and pop-all only (no
// ABA); an acquire pops the whole stack, keeps the head and pushes the rest
// back, so a racing acquire that finds the stack momentarily empty simply
// allocates (bounded by the peak concurrency).
#ifndef USDGEN_VULKAN_MEMORY_POOL_H
#define USDGEN_VULKAN_MEMORY_POOL_H

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>

namespace usdGen::vulkan {

class VulkanDeviceMemoryPool final {
public:
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;  // class size (the allocation size)
        uint32_t type = 0;
        uint32_t sizeClass = 0;
        Block* next = nullptr;
    };
    struct Stats {
        uint64_t allocations = 0;  // vkAllocateMemory calls
        uint64_t reuses = 0;       // acquires served from an idle block
        uint64_t frees = 0;        // vkFreeMemory calls (cap overflow, teardown)
        uint64_t idleBytes = 0;
    };

    VulkanDeviceMemoryPool(VkDevice device, VkDeviceSize retainBytes) noexcept;
    /// Frees every idle block. Blocks still held must not be released after.
    ~VulkanDeviceMemoryPool();
    VulkanDeviceMemoryPool(VulkanDeviceMemoryPool const&) = delete;
    VulkanDeviceMemoryPool& operator=(VulkanDeviceMemoryPool const&) = delete;

    /// A block of at least `bytes` of `memoryType`; null with *result set
    /// when allocation fails. Thread-safe.
    Block* Acquire(uint32_t memoryType, VkDeviceSize bytes, VkResult* result) noexcept;
    /// Returns a GPU-idle, unmapped block. Thread-safe.
    void Release(Block* block) noexcept;

    /// The allocation size a request of `bytes` gets.
    static VkDeviceSize ClassBytes(VkDeviceSize bytes) noexcept;
    Stats stats() const noexcept;

private:
    static constexpr uint32_t kClasses = 4u * 48u;
    static uint32_t ClassOf(VkDeviceSize bytes, VkDeviceSize* classBytes) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDeviceSize retainBytes_ = 0;
    std::atomic<uint64_t> idleBytes_{0};
    std::atomic<uint64_t> allocations_{0}, reuses_{0}, frees_{0};
    std::atomic<Block*> idle_[VK_MAX_MEMORY_TYPES][kClasses] = {};
};

} // namespace usdGen::vulkan

#endif
