// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Registry handles and element-model limits.
#pragma once
#include <cstdint>

namespace usdGen {

enum class UsdGenOpId : uint16_t { Invalid = 0xFFFFu };
enum class UsdGenTypeId : uint16_t { Invalid = 0xFFFFu };

constexpr uint32_t UsdGenIndex(UsdGenOpId id) noexcept { return uint32_t(id); }
constexpr uint32_t UsdGenIndex(UsdGenTypeId id) noexcept { return uint32_t(id); }

using UsdGenSlot = uint16_t;                       // index into a UsdGenChannelTable
constexpr UsdGenSlot kUsdGenNoSlot = 0xFFFFu;
constexpr uint8_t kUsdGenMaxLevels = 3;            // levels 1..3 under the constant level 0
// Plan compacts past this. Every writer on a chain adds one arena to the
// sets below it and a compacted set re-copies its shared columns on every
// cook, so the limit sits above the chains in use (5.4: 4 compacted G3).
constexpr uint8_t kUsdGenMaxArenasPerSet = 16;

}  // namespace usdGen
