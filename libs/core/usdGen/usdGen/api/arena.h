// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// One refcounted block of column memory.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/exec/epoch.h"        // UsdGenRefCounted, UsdGenRef (moves to api/ in step 12)
#include "usdGen/exec/pool.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

namespace usdGen {

/// Header (refcount on its own lines), then a 256-aligned payload, in one
/// pool block. Released through usdGen.dll (UsdGenRefCountedDestroy).
class alignas(64) UsdGenArena final : public UsdGenRefCounted {
public:
    static constexpr size_t kColumnAlign = 256;
    /// Payload of at least 'payloadBytes' (contents undefined).
    USDGEN_CORE_API static UsdGenRef<UsdGenArena> Acquire(size_t payloadBytes);
    std::byte *Payload() noexcept { return reinterpret_cast<std::byte *>(this) + _payloadOffset; }
    std::byte const *Payload() const noexcept
    {
        return reinterpret_cast<std::byte const *>(this) + _payloadOffset;
    }
    size_t Capacity() const noexcept { return _capacity; }
    /// Set when the arena becomes reachable outside the cook (cache store,
    /// host handoff). A shared arena is never rewritten in place again.
    void MarkShared() const noexcept { _shared.store(1, std::memory_order_release); }
    bool IsShared() const noexcept { return _shared.load(std::memory_order_acquire) != 0; }
    /// Unique per Acquire; the device mirror's cache key.
    uint64_t Serial() const noexcept { return _serial; }

    /// The block starts kPrefix bytes before the object and stores its request size.
    static constexpr size_t kPrefix = 64;
    static void operator delete(void *p) noexcept
    {
        std::byte *block = static_cast<std::byte *>(p) - kPrefix;
        UsdGenPoolFree(block, *reinterpret_cast<size_t *>(block));
    }
    static void operator delete(void *p, std::align_val_t) noexcept { operator delete(p); }

private:
    UsdGenArena(size_t capacity, uint64_t serial, uint32_t payloadOffset) noexcept
        : _capacity(capacity), _serial(serial), _payloadOffset(payloadOffset) {}
    ~UsdGenArena() override = default;
    size_t _capacity;
    uint64_t _serial;
    uint32_t _payloadOffset;
    mutable std::atomic<uint8_t> _shared{0};
};

}  // namespace usdGen
