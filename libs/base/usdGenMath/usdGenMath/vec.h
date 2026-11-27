// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenVec{2,3,4}{f,d,h,i}, UsdGenDot, UsdGenCross, UsdGenRotateInto.
// Alias phase (step 11.1): the Gf types and forwarders to Gf's functions.
#pragma once
#include "usdGenMath/half.h"

#include "pxr/base/gf/rotation.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3h.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/gf/vec4h.h"
#include "pxr/base/gf/vec4i.h"

#include <utility>

namespace usdGen {

/// GF_MIN_VECTOR_LENGTH: the default Normalize epsilon.
constexpr double kUsdGenMinVectorLength = 1e-10;

using UsdGenVec2f = PXR_NS::GfVec2f;
using UsdGenVec3f = PXR_NS::GfVec3f;
using UsdGenVec4f = PXR_NS::GfVec4f;
using UsdGenVec2d = PXR_NS::GfVec2d;
using UsdGenVec3d = PXR_NS::GfVec3d;
using UsdGenVec4d = PXR_NS::GfVec4d;
using UsdGenVec2i = PXR_NS::GfVec2i;
using UsdGenVec3i = PXR_NS::GfVec3i;
using UsdGenVec4i = PXR_NS::GfVec4i;
using UsdGenVec2h = PXR_NS::GfVec2h;
using UsdGenVec3h = PXR_NS::GfVec3h;
using UsdGenVec4h = PXR_NS::GfVec4h;

/// Gf's overload set, unchanged (mixed types convert as Gf's would).
template <class... A>
inline auto UsdGenDot(A &&...a) -> decltype(PXR_NS::GfDot(std::forward<A>(a)...))
{
    return PXR_NS::GfDot(std::forward<A>(a)...);
}

template <class... A>
inline auto UsdGenCross(A &&...a) -> decltype(PXR_NS::GfCross(std::forward<A>(a)...))
{
    return PXR_NS::GfCross(std::forward<A>(a)...);
}

/// GfRotation(from, to).TransformDir(v).
inline UsdGenVec3d UsdGenRotateInto(UsdGenVec3d const &from, UsdGenVec3d const &to, UsdGenVec3d const &v)
{
    return PXR_NS::GfRotation(from, to).TransformDir(v);
}

}  // namespace usdGen
