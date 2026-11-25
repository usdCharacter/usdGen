// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "memoryPool.h"

#include <new>

namespace usdGen::vulkan {

namespace {
constexpr VkDeviceSize kMinClassBytes = 4096;
int Log2Floor(VkDeviceSize v) noexcept {
    int e = 0;
    while (v >>= 1) ++e;
    return e;
}
}

VulkanDeviceMemoryPool::VulkanDeviceMemoryPool(VkDevice device, VkDeviceSize retainBytes) noexcept
    : device_(device), retainBytes_(retainBytes) {}

VulkanDeviceMemoryPool::~VulkanDeviceMemoryPool() {
    for (auto& type : idle_)
        for (auto& head : type) {
            Block* b = head.exchange(nullptr, std::memory_order_acquire);
            while (b) {
                Block* next = b->next;
                vkFreeMemory(device_, b->memory, nullptr);
                delete b;
                b = next;
            }
        }
}

uint32_t VulkanDeviceMemoryPool::ClassOf(VkDeviceSize bytes, VkDeviceSize* classBytes) noexcept {
    if (bytes <= kMinClassBytes) {
        *classBytes = kMinClassBytes;
        return 0;
    }
    // (2^e, 2^(e+1)] in four steps of 2^(e-2).
    int const e = Log2Floor(bytes - 1);
    VkDeviceSize const step = VkDeviceSize(1) << (e - 2);
    VkDeviceSize const rounded = (bytes + step - 1) / step * step;
    *classBytes = rounded;
    uint32_t const q = uint32_t(rounded / step) - 4u;  // 1..4
    uint32_t const c = uint32_t(e - 12) * 4u + q;       // e >= 12 above 4 KiB
    return c < kClasses ? c : UINT32_MAX;
}

VkDeviceSize VulkanDeviceMemoryPool::ClassBytes(VkDeviceSize bytes) noexcept {
    VkDeviceSize c = bytes;
    (void)ClassOf(bytes, &c);
    return c;
}

VulkanDeviceMemoryPool::Block* VulkanDeviceMemoryPool::Acquire(uint32_t memoryType, VkDeviceSize bytes,
                                                               VkResult* result) noexcept {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    if (memoryType >= VK_MAX_MEMORY_TYPES || bytes == 0) {
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return nullptr;
    }
    VkDeviceSize classBytes = bytes;
    uint32_t const c = ClassOf(bytes, &classBytes);
    if (c != UINT32_MAX) {
        auto& head = idle_[memoryType][c];
        if (head.load(std::memory_order_relaxed)) {
            if (Block* b = head.exchange(nullptr, std::memory_order_acquire)) {
                if (Block* rest = b->next) {
                    Block* tail = rest;
                    while (tail->next) tail = tail->next;
                    Block* h = head.load(std::memory_order_relaxed);
                    do { tail->next = h; }
                    while (!head.compare_exchange_weak(h, rest, std::memory_order_release,
                                                       std::memory_order_relaxed));
                }
                b->next = nullptr;
                idleBytes_.fetch_sub(b->bytes, std::memory_order_relaxed);
                reuses_.fetch_add(1, std::memory_order_relaxed);
                finish(VK_SUCCESS);
                return b;
            }
        }
    } else {
        classBytes = bytes;  // beyond the class table: exact size, never pooled
    }
    Block* b = new (std::nothrow) Block;
    if (!b) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return nullptr; }
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = classBytes;
    allocation.memoryTypeIndex = memoryType;
    VkResult r = vkAllocateMemory(device_, &allocation, nullptr, &b->memory);
    if (r != VK_SUCCESS) { delete b; finish(r); return nullptr; }
    allocations_.fetch_add(1, std::memory_order_relaxed);
    b->bytes = classBytes;
    b->type = memoryType;
    b->sizeClass = c;
    finish(VK_SUCCESS);
    return b;
}

void VulkanDeviceMemoryPool::Release(Block* b) noexcept {
    if (!b) return;
    bool keep = b->sizeClass != UINT32_MAX;
    if (keep) {
        uint64_t idle = idleBytes_.load(std::memory_order_relaxed);
        do {
            if (idle + b->bytes > retainBytes_) { keep = false; break; }
        } while (!idleBytes_.compare_exchange_weak(idle, idle + b->bytes, std::memory_order_relaxed));
    }
    if (!keep) {
        vkFreeMemory(device_, b->memory, nullptr);
        frees_.fetch_add(1, std::memory_order_relaxed);
        delete b;
        return;
    }
    auto& head = idle_[b->type][b->sizeClass];
    Block* h = head.load(std::memory_order_relaxed);
    do { b->next = h; }
    while (!head.compare_exchange_weak(h, b, std::memory_order_release, std::memory_order_relaxed));
}

VulkanDeviceMemoryPool::Stats VulkanDeviceMemoryPool::stats() const noexcept {
    Stats s;
    s.allocations = allocations_.load(std::memory_order_relaxed);
    s.reuses = reuses_.load(std::memory_order_relaxed);
    s.frees = frees_.load(std::memory_order_relaxed);
    s.idleBytes = idleBytes_.load(std::memory_order_relaxed);
    return s;
}

} // namespace usdGen::vulkan
