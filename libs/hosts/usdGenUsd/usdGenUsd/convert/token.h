// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Host and engine tokens, converted explicitly. ToToken looks the text up
// first and interns (as a compile-phase intern) only text the engine has
// never seen. ToTf returns an immortal TfToken cached per token Id().
#pragma once
#include "usdGen/api/cowArray.h"
#include "usdGenUsd/export.h"
#include "usdGen/api/token.h"
#include "usdGenUsd/convert/math.h"

#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"

#include <string>
#include <string_view>
#include <vector>

namespace usdGenUsd {

USDGEN_USD_API usdGen::UsdGenToken ToToken(PXR_NS::TfToken const &t);
USDGEN_USD_API usdGen::UsdGenToken ToToken(std::string_view text);
inline usdGen::UsdGenToken ToToken(char const *text) { return ToToken(std::string_view(text ? text : "")); }
inline usdGen::UsdGenToken ToToken(std::string const &text) { return ToToken(std::string_view(text)); }
PXR_NS::TfToken const &ToTf(usdGen::UsdGenToken t);

USDGEN_USD_API usdGen::UsdGenTokenVector ToTokens(std::vector<PXR_NS::TfToken> const &t);
USDGEN_USD_API std::vector<PXR_NS::TfToken> ToTfs(usdGen::UsdGenTokenVector const &t);
USDGEN_USD_API usdGen::UsdGenTokenArray ToTokens(PXR_NS::VtArray<PXR_NS::TfToken> const &t);
USDGEN_USD_API PXR_NS::VtArray<PXR_NS::TfToken> ToTfs(usdGen::UsdGenTokenArray const &t);

/// Engine token fields read a host value as TfToken (convert/math.h HostRep).
template <>
struct HostRep<usdGen::UsdGenToken> {
    using Type = PXR_NS::TfToken;
    static usdGen::UsdGenToken In(Type const &h) { return ToToken(h); }
};

}  // namespace usdGenUsd
