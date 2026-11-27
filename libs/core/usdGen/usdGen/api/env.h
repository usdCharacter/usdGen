// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Environment reads. Alias phase (step 11.1): read on each call; 12.0
// freezes a table at Engine::Create (no getenv on hot paths).
#pragma once
#include "usdGen/api/export.h"

namespace usdGen {

/// Null when unset.
USDGEN_CORE_API char const *UsdGenGetEnv(char const *name) noexcept;
/// Unset or empty: fallback. True for "true", "yes", "on" or "1", any case.
USDGEN_CORE_API bool UsdGenGetEnvBool(char const *name, bool fallback) noexcept;
/// Unset or empty: fallback. Otherwise atoi.
USDGEN_CORE_API int UsdGenGetEnvInt(char const *name, int fallback) noexcept;

}  // namespace usdGen
