// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — per-thread bump scratch. See scratch.h.
#include "usdGen/exec/scratch.h"

#include "usdGen/exec/pool.h"

#include <cstdint>

namespace usdGen {

struct UsdGenScratchArena::Block
{
    Block *prev;
    std::size_t bytes;   // payload bytes after this header
    char *Payload() { return reinterpret_cast<char *>(this) + kHeader; }
    static constexpr std::size_t kHeader = 64;
};

namespace {

constexpr std::size_t kMinBlock = std::size_t(64) << 10;
thread_local UsdGenScratchArena t_arena;

UsdGenScratchArena::Block *NewBlock(std::size_t payload, UsdGenScratchArena::Block *prev)
{
    std::size_t const total =
        UsdGenPoolClassBytes(payload + UsdGenScratchArena::Block::kHeader);
    auto *b = static_cast<UsdGenScratchArena::Block *>(UsdGenPoolAlloc(total));
    b->prev = prev;
    b->bytes = total - UsdGenScratchArena::Block::kHeader;
    return b;
}

void FreeBlock(UsdGenScratchArena::Block *b) noexcept
{
    UsdGenPoolFree(b, b->bytes + UsdGenScratchArena::Block::kHeader);
}

}  // namespace

UsdGenScratchArena &UsdGenScratchArena::ForThread() { return t_arena; }

UsdGenScratchArena::Block *UsdGenScratchArena::NewBaseBlock() { return NewBlock(kMinBlock, nullptr); }

void UsdGenScratchArena::FreeBaseBlock(Block *block) noexcept
{
    if (block) FreeBlock(block);
}

void UsdGenScratchArena::AdoptBaseBlock(Block *block) noexcept
{
    if (!block) return;
    if (_block || _spare) {
        FreeBlock(block);
        return;
    }
    block->prev = nullptr;
    _block = block;
    _top = 0;
}

void *UsdGenScratchArena::Alloc(std::size_t bytes, std::size_t align)
{
    std::size_t at = (_top + align - 1) & ~(align - 1);
    if (!_block || at + bytes > _block->bytes) {
        // Grow: the spare when it fits, else a block at least double the
        // current one. Earlier blocks stay live until a rewind passes them.
        std::size_t want = _block ? _block->bytes * 2 : kMinBlock;
        if (want < bytes + align) want = bytes + align;
        Block *next;
        if (_spare && _spare->bytes >= bytes + align) {
            next = _spare;
            next->prev = _block;
            _spare = nullptr;
        } else {
            next = NewBlock(want, _block);
        }
        _block = next;
        at = 0;
        // Payload starts 64-aligned (pool blocks above 256 bytes are).
    }
    _top = at + bytes;
    return _block->Payload() + at;
}

void UsdGenScratchArena::Rewind(Mark mark) noexcept
{
    // Blocks added after the mark go back; keep the largest as the spare so
    // the next scope that needs it reuses it instead of growing again.
    while (_block != mark.block) {
        Block *b = _block;
        _block = b->prev;
        if (!_spare || b->bytes > _spare->bytes) {
            if (_spare) FreeBlock(_spare);
            _spare = b;
        } else {
            FreeBlock(b);
        }
    }
    _top = mark.top;
    // Back at the base with a bigger spare: promote it, so steady state runs
    // in one block.
    if (!_block && _spare) {
        _block = _spare;
        _block->prev = nullptr;
        _spare = nullptr;
        _top = 0;
    }
}

UsdGenScratchArena::~UsdGenScratchArena()
{
    while (_block) {
        Block *b = _block;
        _block = b->prev;
        FreeBlock(b);
    }
    if (_spare) FreeBlock(_spare);
    _spare = nullptr;
}

}  // namespace usdGen
