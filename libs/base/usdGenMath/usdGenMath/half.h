// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenHalf: IEEE binary16. Alias phase (step 11.1): GfHalf until the
// math swap (11.4) brings the engine's own type with the same bits.
#pragma once
#include "pxr/base/gf/half.h"

namespace usdGen {
using UsdGenHalf = PXR_NS::GfHalf;
}  // namespace usdGen
