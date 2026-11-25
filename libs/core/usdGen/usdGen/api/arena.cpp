// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/arena.h"

namespace usdGen {

namespace {
std::atomic<uint64_t> gArenaSerial{0};
}  // namespace

UsdGenRef<UsdGenArena> UsdGenArena::Acquire(size_t payloadBytes)
{
    static_assert(kPrefix >= sizeof(size_t) && kPrefix % alignof(UsdGenArena) == 0, "arena prefix");
    // Pool blocks above 256 bytes are 64-aligned: room to align the payload to 256.
    size_t const request = kPrefix + sizeof(UsdGenArena) + kColumnAlign + payloadBytes;
    std::byte *const block = static_cast<std::byte *>(UsdGenPoolAlloc(request));
    *reinterpret_cast<size_t *>(block) = request;
    std::byte *const object = block + kPrefix;
    uintptr_t const end = reinterpret_cast<uintptr_t>(object) + sizeof(UsdGenArena);
    uintptr_t const payload = (end + kColumnAlign - 1) & ~uintptr_t(kColumnAlign - 1);
    size_t const usable = UsdGenPoolClassBytes(request) > request ? UsdGenPoolClassBytes(request) : request;
    size_t const capacity = size_t(reinterpret_cast<uintptr_t>(block) + usable - payload);
    uint64_t const serial = gArenaSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    auto *arena = ::new (static_cast<void *>(object))
        UsdGenArena(capacity, serial, uint32_t(payload - reinterpret_cast<uintptr_t>(object)));
    return UsdGenRef<UsdGenArena>(arena);
}

}  // namespace usdGen
