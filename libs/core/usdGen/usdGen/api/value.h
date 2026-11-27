// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenValue: a type-erased value with VtValue's exact type dispatch.
// Alias phase (step 11.1): VtValue until the value swap (11.7).
#pragma once
#include "usdGen/api/cowArray.h"

#include "pxr/base/gf/timeCode.h"
#include "pxr/base/vt/value.h"

namespace usdGen {
using UsdGenValue = PXR_NS::VtValue;
/// The value table's time code row.
using UsdGenTimeCode = PXR_NS::GfTimeCode;
}  // namespace usdGen
