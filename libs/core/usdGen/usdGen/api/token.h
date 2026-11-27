// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenToken: an interned string. Alias phase (step 11.1): TfToken until
// the token swap (11.5). UsdGenInternToken (registries/token.h) is the only
// text -> token door; UsdGenFindToken only looks up.
#pragma once
#include "pxr/base/tf/token.h"

#include <string>
#include <string_view>
#include <vector>

namespace usdGen {
using UsdGenToken = PXR_NS::TfToken;
using UsdGenTokenVector = std::vector<UsdGenToken>;

/// Empty when `text` was never interned. Never counts as an intern.
inline UsdGenToken UsdGenFindToken(std::string_view text)
{
    return UsdGenToken::Find(std::string(text));
}
}  // namespace usdGen
