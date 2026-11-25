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
constexpr uint8_t kUsdGenMaxArenasPerSet = 4;      // Plan compacts past this

}  // namespace usdGen
