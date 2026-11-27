// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Test interop with pxr types. Math (step 11.4): Gf values convert to the
// usdGenMath types implicitly (same bytes); usdGenUsd::ToHost converts back
// and usdGenUsd::ToEngine / ToEngineValue convert arrays and VtValues the
// engine reads. Tokens (step 11.5): TfToken and UsdGenToken convert
// implicitly both ways (usdGenUsd::ToToken / ToTf), and mixed comparisons
// compare text. Hosts never include this header.
#pragma once
#include "usdGenUsd/convert/math.h"
#include "usdGenUsd/convert/token.h"

#include "pxr/base/tf/token.h"
#include "pxr/base/vt/types.h"

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
}  // namespace usdGen

using usdGenUsd::ToEngine;
using usdGenUsd::ToEngineValue;
using usdGenUsd::ToHost;
using usdGenUsd::ToHostValue;
using usdGenUsd::ToTf;
using usdGenUsd::ToToken;
