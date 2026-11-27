// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenRange3{f,d}. Alias phase (step 11.1): the Gf types.
#pragma once
#include "usdGenMath/vec.h"

#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/range3f.h"

namespace usdGen {
using UsdGenRange3f = PXR_NS::GfRange3f;
using UsdGenRange3d = PXR_NS::GfRange3d;
}  // namespace usdGen
