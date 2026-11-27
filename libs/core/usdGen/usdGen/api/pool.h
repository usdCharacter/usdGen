// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — size-class pools for engine objects (review: "Size-class pools").
// Blocks come from one reserved address range, so a block is named by a
// 32-bit index. Each thread keeps two magazines per class with no atomics on
// the fast path; full magazines move through a per-class depot, a Treiber
// stack whose head packs the chain index with a 32-bit tag bumped on every
// push and pop (ABA-safe; the range is never unmapped). The pool grows only
// when every magazine and the depot are empty, and each growth counts as one
// engine allocation (allocCounter.h). No locks, no once_flag.
#pragma once

#include "usdGen/api/export.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace usdGen {

/// Largest pooled request; bigger ones go to operator new.
constexpr std::size_t kUsdGenPoolMaxBytes = std::size_t(64) << 20;

/// A block of at least `bytes`, 16-byte aligned (64-byte above 256 bytes,
/// heap fallback included). Never returns null: falls back to operator new
/// (throws bad_alloc).
USDGEN_API void *UsdGenPoolAlloc(std::size_t bytes);
/// Return a block; `bytes` must be the size it was requested with.
USDGEN_API void UsdGenPoolFree(void *p, std::size_t bytes) noexcept;
/// The usable size of a request (its class size), for growable buffers.
USDGEN_API std::size_t UsdGenPoolClassBytes(std::size_t bytes) noexcept;

struct UsdGenPoolStats
{
    uint64_t refills = 0;        // slabs carved from the reserved range
    uint64_t heapFallbacks = 0;  // requests served by operator new
    uint64_t reservedBytes = 0;  // carved so far
};
USDGEN_API UsdGenPoolStats UsdGenPoolGetStats() noexcept;
/// Tests only: while set, the pools do not grow, so empty classes fall back
/// to the heap.
USDGEN_API void UsdGenPoolTestDenyGrowth(bool deny) noexcept;

/// Derive to allocate a type (and everything derived from it, through a
/// virtual destructor) from the pools.
struct UsdGenPooled
{
    static void *operator new(std::size_t n) { return UsdGenPoolAlloc(n); }
    static void operator delete(void *p, std::size_t n) noexcept { UsdGenPoolFree(p, n); }
    // Classes above 256 bytes are 64-byte aligned; wider goes to the heap.
    static void *operator new(std::size_t n, std::align_val_t a)
    {
        if (std::size_t(a) <= 16) return UsdGenPoolAlloc(n);
        if (std::size_t(a) <= 64) return UsdGenPoolAlloc(n < 257 ? 257 : n);
        return ::operator new(n, a);
    }
    static void operator delete(void *p, std::size_t n, std::align_val_t a) noexcept
    {
        if (std::size_t(a) <= 16) UsdGenPoolFree(p, n);
        else if (std::size_t(a) <= 64) UsdGenPoolFree(p, n < 257 ? 257 : n);
        else ::operator delete(p, a);
    }
};

/// std allocator over the pools, for engine-owned containers.
template <class T>
struct UsdGenPoolAllocator
{
    using value_type = T;
    UsdGenPoolAllocator() noexcept = default;
    template <class U>
    UsdGenPoolAllocator(UsdGenPoolAllocator<U> const &) noexcept {}
    T *allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
        return static_cast<T *>(UsdGenPoolAlloc(n * sizeof(T)));
    }
    void deallocate(T *p, std::size_t n) noexcept { UsdGenPoolFree(p, n * sizeof(T)); }
    template <class U>
    bool operator==(UsdGenPoolAllocator<U> const &) const noexcept { return true; }
    template <class U>
    bool operator!=(UsdGenPoolAllocator<U> const &) const noexcept { return false; }
};

/// A vector whose storage recycles through the pools.
template <class T>
using UsdGenPoolVector = std::vector<T, UsdGenPoolAllocator<T>>;

}  // namespace usdGen
