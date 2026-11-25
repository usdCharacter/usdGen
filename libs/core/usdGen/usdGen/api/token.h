// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Interim: UsdGenToken is TfToken until step 11 swaps in the engine's own
// interned token. The pxr include comes through registries/token.h.
#pragma once
#include "usdGen/registries/token.h"

namespace usdGen {
using UsdGenToken = PXR_NS::TfToken;
}  // namespace usdGen
