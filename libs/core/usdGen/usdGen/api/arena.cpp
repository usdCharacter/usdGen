// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/arena.h"

namespace usdGen {

namespace {
std::atomic<uint64_t> gArenaSerial{0};

// The object is alignas(64): pool blocks above 256 bytes are 64-aligned.
constexpr size_t kMinRequest = 257;

std::byte *Block(size_t *request)
{
    static_assert(UsdGenArena::kPrefix >= sizeof(size_t) && UsdGenArena::kPrefix % alignof(UsdGenArena) == 0,
                  "arena prefix");
    if (*request < kMinRequest) *request = kMinRequest;
    std::byte *const block = static_cast<std::byte *>(UsdGenPoolAlloc(*request));
    *reinterpret_cast<size_t *>(block) = *request;
    return block;
}
}  // namespace

UsdGenRef<UsdGenArena> UsdGenArena::Acquire(size_t payloadBytes, size_t align)
{
    if (align < alignof(std::max_align_t)) align = alignof(std::max_align_t);
    if (align > kColumnAlign) align = kColumnAlign;
    size_t request = kPrefix + sizeof(UsdGenArena) + (align > 64 ? align : 0) + payloadBytes;
    std::byte *const block = Block(&request);
    std::byte *const object = block + kPrefix;
    uintptr_t const end = reinterpret_cast<uintptr_t>(object) + sizeof(UsdGenArena);
    uintptr_t const payload = (end + align - 1) & ~uintptr_t(align - 1);
    size_t const usable = UsdGenPoolClassBytes(request) > request ? UsdGenPoolClassBytes(request) : request;
    size_t const capacity = size_t(reinterpret_cast<uintptr_t>(block) + usable - payload);
    uint64_t const serial = gArenaSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    auto *arena = ::new (static_cast<void *>(object))
        UsdGenArena(capacity, serial, reinterpret_cast<std::byte *>(payload), nullptr);
    return UsdGenRef<UsdGenArena>(arena);
}

UsdGenRef<UsdGenArena> UsdGenArena::Adopt(void const *data, size_t bytes, UsdGenExternal *owner)
{
    size_t request = kPrefix + sizeof(UsdGenArena);
    std::byte *const block = Block(&request);
    uint64_t const serial = gArenaSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    auto *arena = ::new (static_cast<void *>(block + kPrefix))
        UsdGenArena(bytes, serial, static_cast<std::byte *>(const_cast<void *>(data)), owner);
    return UsdGenRef<UsdGenArena>(arena);
}

}  // namespace usdGen
