// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenMatrix{2d,3d,4d,4f}. Alias phase (step 11.1): the Gf types.
#pragma once
#include "usdGenMath/vec.h"

#include "pxr/base/gf/matrix2d.h"
#include "pxr/base/gf/matrix3d.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/matrix4f.h"

namespace usdGen {
using UsdGenMatrix2d = PXR_NS::GfMatrix2d;
using UsdGenMatrix3d = PXR_NS::GfMatrix3d;
using UsdGenMatrix4d = PXR_NS::GfMatrix4d;
using UsdGenMatrix4f = PXR_NS::GfMatrix4f;
}  // namespace usdGen
