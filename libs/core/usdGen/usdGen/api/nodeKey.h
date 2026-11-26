// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Host-assigned node identity (step 7). The engine hashes, compares and
// stores keys; only hosts know what they name.
#pragma once
#include <cstddef>
#include <cstdint>

namespace usdGen {

/// Opaque node identity. 0 is invalid.
struct UsdGenNodeKey {
    uint64_t value = 0;
    constexpr bool IsValid() const noexcept { return value != 0; }
    friend constexpr bool operator==(UsdGenNodeKey a, UsdGenNodeKey b) noexcept { return a.value == b.value; }
    friend constexpr bool operator!=(UsdGenNodeKey a, UsdGenNodeKey b) noexcept { return a.value != b.value; }
    friend constexpr bool operator<(UsdGenNodeKey a, UsdGenNodeKey b) noexcept { return a.value < b.value; }
};

struct UsdGenNodeKeyHash {
    size_t operator()(UsdGenNodeKey k) const noexcept { return size_t(k.value ^ (k.value >> 32)); }
};

}  // namespace usdGen
