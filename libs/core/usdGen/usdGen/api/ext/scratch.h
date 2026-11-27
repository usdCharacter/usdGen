// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — per-thread bump scratch. Each thread (in practice each pool
// worker) owns one arena; a UsdGenScratchScope marks it and rewinds on exit,
// so chunk-local temporaries cost a pointer bump and are released per chunk.
// Blocks come from the size-class pools and are kept for the thread's life,
// so a warm cook never reaches the heap. Single owner: no atomics.
#pragma once

#include "usdGen/api/export.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace usdGen {

class UsdGenScratchArena
{
public:
    struct Block;
    struct Mark
    {
        Block *block;
        std::size_t top;
    };

    /// The calling thread's arena.
    USDGEN_API static UsdGenScratchArena &ForThread();

    /// A base block allocated by one thread for an arena that another
    /// thread adopts (UsdGenWorkerPool, for its workers): the adopting
    /// thread's first scope then needs no heap. FreeBaseBlock returns one
    /// that was never adopted.
    USDGEN_API static Block *NewBaseBlock();
    USDGEN_API static void FreeBaseBlock(Block *block) noexcept;
    /// Makes `block` this arena's base block; frees it instead when the
    /// arena already holds one.
    USDGEN_API void AdoptBaseBlock(Block *block) noexcept;

    USDGEN_API void *Alloc(std::size_t bytes, std::size_t align);
    Mark Save() const { return {_block, _top}; }
    USDGEN_API void Rewind(Mark mark) noexcept;
    USDGEN_API ~UsdGenScratchArena();

    constexpr UsdGenScratchArena() = default;
    UsdGenScratchArena(UsdGenScratchArena const &) = delete;
    UsdGenScratchArena &operator=(UsdGenScratchArena const &) = delete;

private:
    Block *_block = nullptr;   // current block; earlier ones chain through prev
    std::size_t _top = 0;      // bytes used in _block's payload
    Block *_spare = nullptr;   // the largest block retired by a rewind, kept
};

/// Marks the calling thread's arena; everything allocated through it is
/// released when the scope ends. Scopes nest in stack order.
class UsdGenScratchScope
{
public:
    UsdGenScratchScope()
        : _arena(UsdGenScratchArena::ForThread()), _mark(_arena.Save()) {}
    ~UsdGenScratchScope() { _arena.Rewind(_mark); }
    UsdGenScratchScope(UsdGenScratchScope const &) = delete;
    UsdGenScratchScope &operator=(UsdGenScratchScope const &) = delete;

    /// `count` uninitialized elements of trivial T.
    template <class T>
    T *Array(std::size_t count)
    {
        static_assert(std::is_trivially_copyable<T>::value &&
                      std::is_trivially_destructible<T>::value,
                      "scratch holds trivial types only");
        return static_cast<T *>(_arena.Alloc(count * sizeof(T), alignof(T) < 16 ? 16 : alignof(T)));
    }
    /// `count` elements of trivial T, each set to `value`.
    template <class T>
    T *Filled(std::size_t count, T const &value)
    {
        T *p = Array<T>(count);
        for (std::size_t i = 0; i < count; ++i) p[i] = value;
        return p;
    }

private:
    UsdGenScratchArena &_arena;
    UsdGenScratchArena::Mark _mark;
};

}  // namespace usdGen
