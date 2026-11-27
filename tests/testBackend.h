// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Backends by registered name for tests: the engine names none (step 8.4).
#pragma once
#include "usdGen/api/ext/backendDef.h"

namespace usdGenTestUtils {

/// The handle registered for name ("cpu" is the built-in CPU); Invalid if none.
inline usdGen::UsdGenBackendId Backend(char const *name) noexcept
{
    return usdGen::UsdGenFindBackend(name);
}

/// Handles no library registers: distinct backend identities for tests of
/// keys, caches and ledgers, never a runnable backend.
constexpr usdGen::UsdGenBackendId kTestBackendA =
    usdGen::UsdGenBackendId(usdGen::kUsdGenMaxBackends - 1);
constexpr usdGen::UsdGenBackendId kTestBackendB =
    usdGen::UsdGenBackendId(usdGen::kUsdGenMaxBackends - 2);

}  // namespace usdGenTestUtils
