// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Atomic publication of a shared_ptr without std::atomic_load/store, which
// MSVC and libstdc++ implement with a lock. The value sits in an immutable
// box published by atomic pointer; readers copy it out under an epoch guard
// and a displaced box retires through the epochs. Writers drain right away,
// so with no reader pinned the displaced value is released before the write
// returns, as the shared_ptr store it replaces did. For Vulkan objects that
// are still shared_ptr-owned; new code publishes UsdGenAtomicRef instead.
#ifndef USDGEN_VULKAN_EPOCH_CELL_H
#define USDGEN_VULKAN_EPOCH_CELL_H

#include "usdGen/epoch.h"

#include <atomic>
#include <memory>
#include <utility>

namespace usdGen::vulkan {

template <class T>
class VulkanEpochCell final {
public:
    VulkanEpochCell() noexcept = default;
    explicit VulkanEpochCell(std::shared_ptr<T> v) : p_(Box::Make(std::move(v))) {}
    ~VulkanEpochCell() { delete p_.load(std::memory_order_acquire); }
    VulkanEpochCell(VulkanEpochCell const&) = delete;
    VulkanEpochCell& operator=(VulkanEpochCell const&) = delete;

    /// Any thread, never waits.
    std::shared_ptr<T> Load() const noexcept {
        UsdGenEpochGuard guard;
        Box const* b = p_.load(std::memory_order_seq_cst);
        return b ? b->value : std::shared_ptr<T>{};
    }
    /// Publishes v; only a non-null value allocates (and may throw).
    void Store(std::shared_ptr<T> v) { Retire(p_.exchange(Box::Make(std::move(v)), std::memory_order_seq_cst)); }
    /// Clears the cell; never allocates.
    void Reset() noexcept { Retire(p_.exchange(nullptr, std::memory_order_seq_cst)); }
    /// Clears the cell and returns what it held; never allocates.
    std::shared_ptr<T> Take() noexcept {
        Box* old = p_.exchange(nullptr, std::memory_order_seq_cst);
        if (!old) return {};
        std::shared_ptr<T> v = old->value;  // concurrent readers only copy it
        Retire(old);
        return v;
    }
    /// Publishes desired if the cell still holds expected (by pointer).
    bool CompareExchange(std::shared_ptr<T> const& expected, std::shared_ptr<T> desired) {
        Box* fresh = Box::Make(std::move(desired));
        Box* b;
        {
            UsdGenEpochGuard guard;  // a box seen here cannot be reused (no ABA)
            b = p_.load(std::memory_order_seq_cst);
            if ((b ? b->value.get() : nullptr) != expected.get() ||
                !p_.compare_exchange_strong(b, fresh, std::memory_order_seq_cst)) {
                delete fresh;
                return false;
            }
        }
        Retire(b);
        return true;
    }

private:
    struct Box final : UsdGenEpochNode {
        explicit Box(std::shared_ptr<T> v) noexcept : value(std::move(v)) {}
        static Box* Make(std::shared_ptr<T> v) { return v ? new Box(std::move(v)) : nullptr; }
        std::shared_ptr<T> const value;
    };
    static void Retire(Box* b) noexcept {
        if (!b) return;
        UsdGenEpochRetire(b, [](UsdGenEpochNode* n) { delete static_cast<Box*>(n); });
        UsdGenEpochDrain();  // no-op while this thread is pinned
    }
    std::atomic<Box*> p_{nullptr};
};

} // namespace usdGen::vulkan
#endif
