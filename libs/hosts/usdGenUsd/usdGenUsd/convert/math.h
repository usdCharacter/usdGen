// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Host and engine math. The usdGenMath types have Gf's bytes, so single
// values convert implicitly (in) and explicitly (out) through
// UsdGenHostMath; values convert explicitly here and arrays in
// convert/array.h (included at the end).
#pragma once
#include "usdGenMath/math.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/matrix2d.h"
#include "pxr/base/gf/matrix3d.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/range3f.h"
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
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"

#include <cstring>
#include <type_traits>
#include <utility>

namespace usdGenUsd {
/// The host type with an engine type's bytes (the inverse of UsdGenHostMath).
template <class Engine> struct HostMath { using Type = void; };
}  // namespace usdGenUsd

#define USDGEN_HOST_MATH(GF, ENGINE)                                                                   \
    namespace usdGen { template <> struct UsdGenHostMath<PXR_NS::GF> { using Type = ENGINE; }; }      \
    namespace usdGenUsd { template <> struct HostMath<usdGen::ENGINE> { using Type = PXR_NS::GF; }; } \
    static_assert(sizeof(PXR_NS::GF) == sizeof(usdGen::ENGINE) &&                                      \
                  alignof(PXR_NS::GF) == alignof(usdGen::ENGINE) &&                                    \
                  std::is_trivially_copyable<PXR_NS::GF>::value, #GF " layout");

USDGEN_HOST_MATH(GfVec2f, UsdGenVec2f)
USDGEN_HOST_MATH(GfVec3f, UsdGenVec3f)
USDGEN_HOST_MATH(GfVec4f, UsdGenVec4f)
USDGEN_HOST_MATH(GfVec2d, UsdGenVec2d)
USDGEN_HOST_MATH(GfVec3d, UsdGenVec3d)
USDGEN_HOST_MATH(GfVec4d, UsdGenVec4d)
USDGEN_HOST_MATH(GfVec2i, UsdGenVec2i)
USDGEN_HOST_MATH(GfVec3i, UsdGenVec3i)
USDGEN_HOST_MATH(GfVec4i, UsdGenVec4i)
USDGEN_HOST_MATH(GfVec2h, UsdGenVec2h)
USDGEN_HOST_MATH(GfVec3h, UsdGenVec3h)
USDGEN_HOST_MATH(GfVec4h, UsdGenVec4h)
USDGEN_HOST_MATH(GfMatrix2d, UsdGenMatrix2d)
USDGEN_HOST_MATH(GfMatrix3d, UsdGenMatrix3d)
USDGEN_HOST_MATH(GfMatrix4d, UsdGenMatrix4d)
USDGEN_HOST_MATH(GfMatrix4f, UsdGenMatrix4f)
USDGEN_HOST_MATH(GfQuatf, UsdGenQuatf)
USDGEN_HOST_MATH(GfQuatd, UsdGenQuatd)
USDGEN_HOST_MATH(GfQuath, UsdGenQuath)
USDGEN_HOST_MATH(GfRange3f, UsdGenRange3f)
USDGEN_HOST_MATH(GfRange3d, UsdGenRange3d)
#undef USDGEN_HOST_MATH
namespace usdGenUsd { template <> struct HostMath<usdGen::UsdGenHalf> { using Type = PXR_NS::GfHalf; }; }
static_assert(sizeof(PXR_NS::GfHalf) == sizeof(usdGen::UsdGenHalf), "GfHalf layout");

namespace usdGenUsd {

namespace convert_detail {
template <class T> struct EngineOfImpl { using Type = typename usdGen::UsdGenHostMath<T>::Type; };
template <> struct EngineOfImpl<PXR_NS::GfHalf> { using Type = usdGen::UsdGenHalf; };

}  // namespace convert_detail

/// The engine type with a host type's bytes (void when none).
template <class Host> using EngineMathOf = typename convert_detail::EngineOfImpl<Host>::Type;
template <class Engine> using HostMathOf = typename HostMath<Engine>::Type;

/// True for a usdGenMath type or an array of one: such a value must not
/// reach Hydra or USD unconverted (they read Gf types). Data-source helpers
/// static_assert on it.
template <class T> struct IsEngineMath : std::integral_constant<bool, !std::is_void<HostMathOf<T>>::value> {};
template <class T> constexpr bool kIsEngineMath = IsEngineMath<std::decay_t<T>>::value;

/// Single values.
inline usdGen::UsdGenHalf ToEngine(PXR_NS::GfHalf h) noexcept { usdGen::UsdGenHalf e; e.setBits(h.bits()); return e; }
inline PXR_NS::GfHalf ToHost(usdGen::UsdGenHalf h) noexcept { PXR_NS::GfHalf g; g.setBits(h.bits()); return g; }
template <class H, class E = EngineMathOf<H>, std::enable_if_t<!std::is_void<E>::value, int> = 0>
E ToEngine(H const &h) noexcept { E e; std::memcpy(static_cast<void *>(&e), static_cast<void const *>(&h), sizeof(E)); return e; }
template <class E, class H = HostMathOf<E>, std::enable_if_t<!std::is_void<H>::value, int> = 0>
H ToHost(E const &e) noexcept { H h; std::memcpy(static_cast<void *>(&h), static_cast<void const *>(&e), sizeof(H)); return h; }

/// A copy of a host array of math values as engine values (std::vector).
template <class E, class H>
void CopyToEngine(H const *data, size_t n, E *out) noexcept
{
    static_assert(sizeof(E) == sizeof(H), "same bytes");
    if (n) std::memcpy(static_cast<void *>(out), static_cast<void const *>(data), n * sizeof(E));
}

/// How a USD or Hydra value of engine type T is held on the host: T itself,
/// or the Gf type (array) with its bytes. Read a value as HostRep<T>::Type,
/// then In() gives T. Builders read engine fields through it so a host
/// value never silently misses an engine-typed IsHolding.
template <class T, class = void>
struct HostRep {
    using Type = T;
    static T const &In(T const &h) noexcept { return h; }
};
template <class E>
struct HostRep<E, std::enable_if_t<!std::is_void<HostMathOf<E>>::value>> {
    using Type = HostMathOf<E>;
    static E In(Type const &h) noexcept { return ToEngine(h); }
};
/// VtValues: a held Gf math value becomes the usdGenMath one (ToEngineValue),
/// and back (ToHostValue); a TfToken becomes a UsdGenToken and back
/// (convert/token.h); a VtArray of a convertible element becomes a
/// UsdGenCowArray and back (convert/array.h), and a string array a
/// std::vector<std::string>. Anything else is unchanged. Every value the
/// engine reads goes through ToEngineValue (UsdGenValue is VtValue until the
/// value swap, and IsHolding is exact).
PXR_NS::VtValue ToEngineValue(PXR_NS::VtValue const &v);
PXR_NS::VtValue ToHostValue(PXR_NS::VtValue const &v);

}  // namespace usdGenUsd

#include "usdGenUsd/convert/array.h"
