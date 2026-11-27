// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenCowArray<T>: a copy-on-write array with VtArray's read API and
// detach rule. Alias phase (step 11.1): VtArray until the array swap (11.6).
#pragma once
#include "usdGen/api/token.h"

#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"

#include <cstdint>

namespace usdGen {
template <class T>
using UsdGenCowArray = PXR_NS::VtArray<T>;

using UsdGenFloatArray = PXR_NS::VtFloatArray;
using UsdGenIntArray = PXR_NS::VtIntArray;
using UsdGenBoolArray = PXR_NS::VtBoolArray;
using UsdGenUInt64Array = PXR_NS::VtUInt64Array;
using UsdGenTokenArray = PXR_NS::VtTokenArray;
using UsdGenVec2fArray = PXR_NS::VtVec2fArray;
using UsdGenVec3fArray = PXR_NS::VtVec3fArray;
using UsdGenVec4fArray = PXR_NS::VtVec4fArray;
using UsdGenVec2dArray = PXR_NS::VtVec2dArray;
using UsdGenVec3dArray = PXR_NS::VtVec3dArray;
using UsdGenVec4dArray = PXR_NS::VtVec4dArray;
using UsdGenVec2hArray = PXR_NS::VtVec2hArray;
using UsdGenVec3hArray = PXR_NS::VtVec3hArray;
using UsdGenVec4hArray = PXR_NS::VtVec4hArray;
using UsdGenVec2iArray = PXR_NS::VtVec2iArray;
using UsdGenVec3iArray = PXR_NS::VtVec3iArray;
using UsdGenVec4iArray = PXR_NS::VtVec4iArray;
using UsdGenMatrix4dArray = PXR_NS::VtMatrix4dArray;
}  // namespace usdGen
