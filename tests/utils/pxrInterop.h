// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Test interop with pxr types. Math (step 11.4): Gf values convert to the
// usdGenMath types implicitly (same bytes); usdGenUsd::ToHost converts back
// and usdGenUsd::ToEngine / ToEngineValue convert arrays and VtValues the
// engine reads. Tokens (step 11.5): TfToken and UsdGenToken convert
// implicitly both ways (usdGenUsd::ToToken / ToTf), and mixed comparisons
// compare text. Arrays (step 11.6): a VtArray converts implicitly both ways
// to the UsdGenCowArray of its engine element (usdGenUsd::ToCow / ToVt, zero
// copy where the bytes match), and mixed comparisons compare elements.
// Values (step 11.7): a VtValue converts implicitly both ways to a
// UsdGenValue (usdGenUsd::ToEngineValue / ToHostValue).
// Hosts never include this header.
#pragma once
#include "usdGenUsd/convert/array.h"
#include "usdGenUsd/convert/math.h"
#include "usdGenUsd/convert/token.h"
#include "usdGenUsd/convert/value.h"

#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"

#include <string_view>
#include <type_traits>

namespace usdGen {
template <>
struct UsdGenHostToken<PXR_NS::TfToken> : std::true_type {
    static UsdGenToken In(PXR_NS::TfToken const &t) { return usdGenUsd::ToToken(t); }
    static PXR_NS::TfToken Out(UsdGenToken t) { return usdGenUsd::ToTf(t); }
};

namespace pxr_interop_detail {
inline std::string_view View(PXR_NS::TfToken const &t) { return std::string_view(t.GetText(), t.size()); }
}  // namespace pxr_interop_detail

inline bool operator==(PXR_NS::TfToken const &a, UsdGenToken b) { return pxr_interop_detail::View(a) == b.View(); }
inline bool operator==(UsdGenToken a, PXR_NS::TfToken const &b) { return b == a; }
inline bool operator!=(PXR_NS::TfToken const &a, UsdGenToken b) { return !(a == b); }
inline bool operator!=(UsdGenToken a, PXR_NS::TfToken const &b) { return !(b == a); }
inline bool operator<(PXR_NS::TfToken const &a, UsdGenToken b) { return a < PXR_NS::TfToken(usdGenUsd::ToTf(b)); }
inline bool operator<(UsdGenToken a, PXR_NS::TfToken const &b) { return usdGenUsd::ToTf(a) < b; }

template <class H>
struct UsdGenHostArray<PXR_NS::VtArray<H>>
    : std::integral_constant<bool, !std::is_void<usdGenUsd::EngineElemOf<H>>::value> {
    using Element = usdGenUsd::EngineElemOf<H>;
    static UsdGenCowArray<Element> In(PXR_NS::VtArray<H> const &h) { return usdGenUsd::ToCow(h); }
    static PXR_NS::VtArray<H> Out(UsdGenCowArray<Element> const &a) { return usdGenUsd::ToVt(a); }
};

template <>
struct UsdGenHostValue<PXR_NS::VtValue> : std::true_type {
    static UsdGenValue In(PXR_NS::VtValue const &v) { return usdGenUsd::ToEngineValue(v); }
    static PXR_NS::VtValue Out(UsdGenValue const &v) { return usdGenUsd::ToHostValue(v); }
};

/// Mixed comparisons compare engine values (pxr's VtValue == T would never match).
inline bool operator==(UsdGenValue const &a, PXR_NS::VtValue const &b) { return a == usdGenUsd::ToEngineValue(b); }
inline bool operator==(PXR_NS::VtValue const &b, UsdGenValue const &a) { return a == b; }
inline bool operator!=(UsdGenValue const &a, PXR_NS::VtValue const &b) { return !(a == b); }
inline bool operator!=(PXR_NS::VtValue const &b, UsdGenValue const &a) { return !(a == b); }

namespace pxr_interop_detail {
template <class E, class H>
using IfSameElem = std::enable_if_t<std::is_same<usdGenUsd::EngineElemOf<H>, E>::value, bool>;
}  // namespace pxr_interop_detail

template <class E, class H>
pxr_interop_detail::IfSameElem<E, H> operator==(UsdGenCowArray<E> const &a, PXR_NS::VtArray<H> const &b)
{
    return a == UsdGenCowArray<E>(usdGenUsd::ToCow(b));
}
template <class E, class H>
pxr_interop_detail::IfSameElem<E, H> operator==(PXR_NS::VtArray<H> const &b, UsdGenCowArray<E> const &a) { return a == b; }
template <class E, class H>
pxr_interop_detail::IfSameElem<E, H> operator!=(UsdGenCowArray<E> const &a, PXR_NS::VtArray<H> const &b) { return !(a == b); }
template <class E, class H>
pxr_interop_detail::IfSameElem<E, H> operator!=(PXR_NS::VtArray<H> const &b, UsdGenCowArray<E> const &a) { return !(a == b); }
}  // namespace usdGen

using usdGenUsd::ToCow;
using usdGenUsd::ToEngine;
using usdGenUsd::ToEngineValue;
using usdGenUsd::ToHost;
using usdGenUsd::ToHostValue;
using usdGenUsd::ToTf;
using usdGenUsd::ToToken;
using usdGenUsd::ToVt;
