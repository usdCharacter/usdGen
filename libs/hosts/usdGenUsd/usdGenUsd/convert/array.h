// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Host and engine arrays, converted explicitly and without copying where the
// element bytes match: ToCow adopts the VtArray's buffer (an external arena
// pins a copy of the VtArray, so an unchanged host array keeps its address);
// ToVt hands a VtArray the engine arena through a foreign data source.
// Token arrays convert per element. Included by convert/math.h.
#pragma once
#include "usdGen/api/cowArray.h"
#include "usdGen/api/token.h"
#include "usdGen/api/value.h"
#include "usdGenUsd/convert/math.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/timeCode.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace usdGenUsd {

usdGen::UsdGenToken ToToken(PXR_NS::TfToken const &t);   // convert/token.h
PXR_NS::TfToken const &ToTf(usdGen::UsdGenToken t);

namespace convert_detail {
/// The engine element with a host element's bytes (void when none).
template <class H, class = void> struct EngineElemImpl { using Type = void; };
template <class H> struct EngineElemImpl<H, std::enable_if_t<std::is_arithmetic<H>::value>> { using Type = H; };
template <class H>
struct EngineElemImpl<H, std::enable_if_t<!std::is_void<typename EngineOfImpl<H>::Type>::value>> {
    using Type = typename EngineOfImpl<H>::Type;
};
template <> struct EngineElemImpl<PXR_NS::GfTimeCode, void> { using Type = usdGen::UsdGenTimeCode; };
/// The host element of an engine element (void when none).
template <class E, class = void> struct HostElemImpl { using Type = void; };
template <class E> struct HostElemImpl<E, std::enable_if_t<std::is_arithmetic<E>::value>> { using Type = E; };
template <class E>
struct HostElemImpl<E, std::enable_if_t<!std::is_void<typename HostMath<E>::Type>::value>> {
    using Type = typename HostMath<E>::Type;
};
template <> struct HostElemImpl<usdGen::UsdGenTimeCode, void> { using Type = PXR_NS::GfTimeCode; };

/// Keeps a host VtArray alive under an engine arena.
template <class A>
struct VtPin final : usdGen::UsdGenExternal {
    explicit VtPin(A a) : array(std::move(a)) {}
    void Release() noexcept override { delete this; }
    A array;
};

/// Keeps an engine arena alive under a VtArray.
struct ArenaSource final : PXR_NS::Vt_ArrayForeignDataSource {
    explicit ArenaSource(usdGen::UsdGenRef<const usdGen::UsdGenArena> a)
        : PXR_NS::Vt_ArrayForeignDataSource(&Detached), arena(std::move(a)) {}
    static void Detached(PXR_NS::Vt_ArrayForeignDataSource *self) { delete static_cast<ArenaSource *>(self); }
    usdGen::UsdGenRef<const usdGen::UsdGenArena> arena;
};
}  // namespace convert_detail

/// Element types: the engine's for a host one, the host's for an engine one.
/// Tokens convert per element (TfToken <-> UsdGenToken).
template <class H> using EngineElemOf = std::conditional_t<std::is_same<H, PXR_NS::TfToken>::value,
    usdGen::UsdGenToken, typename convert_detail::EngineElemImpl<H>::Type>;
template <class E> using HostElemOf = std::conditional_t<std::is_same<E, usdGen::UsdGenToken>::value,
    PXR_NS::TfToken, typename convert_detail::HostElemImpl<E>::Type>;

/// Zero copy: the engine array adopts the host buffer.
template <class H, class E = EngineElemOf<H>,
          std::enable_if_t<!std::is_void<E>::value && !std::is_same<H, PXR_NS::TfToken>::value, int> = 0>
usdGen::UsdGenCowArray<E> ToCow(PXR_NS::VtArray<H> const &a)
{
    static_assert(sizeof(E) == sizeof(H) && std::is_trivially_copyable<H>::value, "same bytes");
    if (a.empty()) return usdGen::UsdGenCowArray<E>();
    auto *pin = new convert_detail::VtPin<PXR_NS::VtArray<H>>(a);
    return usdGen::UsdGenCowArray<E>::Adopt(reinterpret_cast<E const *>(pin->array.cdata()), pin->array.size(), pin);
}
/// Interns each token text the engine has not seen (compile phase).
usdGen::UsdGenTokenArray ToCow(PXR_NS::VtArray<PXR_NS::TfToken> const &a);

/// Zero copy: the VtArray reads the engine arena (it holds a reference).
template <class E, class H = HostElemOf<E>,
          std::enable_if_t<!std::is_void<H>::value && !std::is_same<E, usdGen::UsdGenToken>::value, int> = 0>
PXR_NS::VtArray<H> ToVt(usdGen::UsdGenCowArray<E> const &a)
{
    static_assert(sizeof(E) == sizeof(H) && std::is_trivially_copyable<H>::value, "same bytes");
    if (a.empty()) return PXR_NS::VtArray<H>();
    auto *source = new convert_detail::ArenaSource(a.ArenaRef());
    return PXR_NS::VtArray<H>(source, reinterpret_cast<H *>(const_cast<E *>(a.cdata())), a.size());
}
PXR_NS::VtArray<PXR_NS::TfToken> ToVt(usdGen::UsdGenTokenArray const &a);

/// The array spellings of ToEngine and ToHost (convert/math.h).
template <class H>
auto ToEngine(PXR_NS::VtArray<H> const &a) -> decltype(ToCow(a)) { return ToCow(a); }
template <class E>
auto ToHost(usdGen::UsdGenCowArray<E> const &a) -> decltype(ToVt(a)) { return ToVt(a); }

/// Engine arrays are read from USD and Hydra as their VtArray.
template <class E>
struct HostRep<usdGen::UsdGenCowArray<E>, std::enable_if_t<!std::is_void<HostElemOf<E>>::value>> {
    using Type = PXR_NS::VtArray<HostElemOf<E>>;
    static usdGen::UsdGenCowArray<E> In(Type const &h) { return ToCow(h); }
};

/// An engine array never reaches Hydra or USD unconverted.
template <class E> struct IsEngineMath<usdGen::UsdGenCowArray<E>> : std::true_type {};

}  // namespace usdGenUsd
