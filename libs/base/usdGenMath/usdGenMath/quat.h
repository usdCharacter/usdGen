// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenQuat{f,d,h}. Alias phase (step 11.1): the Gf types.
#pragma once
#include "usdGenMath/vec.h"

#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/quath.h"

namespace usdGen {
using UsdGenQuatf = PXR_NS::GfQuatf;
using UsdGenQuatd = PXR_NS::GfQuatd;
using UsdGenQuath = PXR_NS::GfQuath;
}  // namespace usdGen
