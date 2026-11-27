// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenCowArray<T>: a copy-on-write array with VtArray's read API and
// detach rule. Alias phase (step 11.1): VtArray until the array swap (11.6).
// Since the math swap (11.4) the math arrays hold usdGenMath types; hosts
// convert Gf arrays with usdGenUsd/convert/math.h.
#pragma once
#include "usdGen/api/token.h"
#include "usdGenMath/math.h"

#include "pxr/base/vt/array.h"

#include <cstdint>

namespace usdGen {
template <class T>
using UsdGenCowArray = PXR_NS::VtArray<T>;

using UsdGenFloatArray = UsdGenCowArray<float>;
using UsdGenIntArray = UsdGenCowArray<int>;
using UsdGenBoolArray = UsdGenCowArray<bool>;
using UsdGenUInt64Array = UsdGenCowArray<uint64_t>;
using UsdGenTokenArray = UsdGenCowArray<UsdGenToken>;
using UsdGenVec2fArray = UsdGenCowArray<UsdGenVec2f>;
using UsdGenVec3fArray = UsdGenCowArray<UsdGenVec3f>;
using UsdGenVec4fArray = UsdGenCowArray<UsdGenVec4f>;
using UsdGenVec2dArray = UsdGenCowArray<UsdGenVec2d>;
using UsdGenVec3dArray = UsdGenCowArray<UsdGenVec3d>;
using UsdGenVec4dArray = UsdGenCowArray<UsdGenVec4d>;
using UsdGenVec2hArray = UsdGenCowArray<UsdGenVec2h>;
using UsdGenVec3hArray = UsdGenCowArray<UsdGenVec3h>;
using UsdGenVec4hArray = UsdGenCowArray<UsdGenVec4h>;
using UsdGenVec2iArray = UsdGenCowArray<UsdGenVec2i>;
using UsdGenVec3iArray = UsdGenCowArray<UsdGenVec3i>;
using UsdGenVec4iArray = UsdGenCowArray<UsdGenVec4i>;
using UsdGenMatrix4dArray = UsdGenCowArray<UsdGenMatrix4d>;
}  // namespace usdGen
