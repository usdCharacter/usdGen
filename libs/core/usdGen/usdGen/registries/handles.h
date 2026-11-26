// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#pragma once
#include "usdGen/api/ids.h"
#include <cstdint>

namespace usdGen {

/// Registered handle; 0 is the CPU (registries/backend.h). Until step 8.4
/// a handle equals the UsdGenExecutionBackend value of its name.
enum class UsdGenBackendId : uint8_t { Invalid = 0xFFu };

using UsdGenParamSlot = uint16_t;
constexpr UsdGenParamSlot kUsdGenNoParam = 0xFFFFu;

constexpr uint32_t kUsdGenMaxOps = 256;
constexpr uint32_t kUsdGenMaxTypes = 64;
constexpr uint32_t kUsdGenMaxBackends = 8;
constexpr uint32_t kUsdGenMaxParamsPerOp = 128;   // Clump declares ~50 today
constexpr uint32_t kUsdGenMaxOpTokens = 32;

constexpr uint32_t UsdGenIndex(UsdGenBackendId id) noexcept { return uint32_t(id); }

}  // namespace usdGen
