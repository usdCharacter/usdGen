// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenCow<T>: O(1) copy, detach on Edit. The value lives in a refcounted
// block from the pools; copies share it until one of them edits.
#pragma once
#include "usdGen/api/epoch.h"
#include "usdGen/api/pool.h"

#include <atomic>
#include <utility>

namespace usdGen {

template <class T>
class UsdGenCow {
public:
    UsdGenCow() : _block(new Block()) {}                     // T{}
    explicit UsdGenCow(T value) : _block(new Block(std::move(value))) {}
    // Copy only: a moved-from Cow would have no block.
    UsdGenCow(UsdGenCow const &) noexcept = default;
    UsdGenCow &operator=(UsdGenCow const &) noexcept = default;

    T const &Get() const noexcept { return _block->value; }
    T const &operator*() const noexcept { return _block->value; }
    T const *operator->() const noexcept { return &_block->value; }
    /// Clones the value first when another Cow shares it.
    T &Edit()
    {
        if (!IsUnique()) _block = UsdGenRef<Block>(new Block(_block->value));
        return _block->value;
    }
    bool IsUnique() const noexcept
    {
        if (_block->RefCount() != 1) return false;
        std::atomic_thread_fence(std::memory_order_acquire);   // pairs with the other owner's release
        return true;
    }
    /// Same block (undo budget: unique bytes).
    bool SameAs(UsdGenCow const &o) const noexcept { return _block == o._block; }
    void const *Identity() const noexcept { return _block.Get(); }

private:
    struct Block final : UsdGenRefCounted, UsdGenPooled {
        Block() : value() {}
        explicit Block(T v) : value(std::move(v)) {}
        T value;
    };
    UsdGenRef<Block> _block;
};

}  // namespace usdGen
