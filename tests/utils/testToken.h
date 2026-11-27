// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// TestToken: engine tokens in tests that do not link usdGenUsd (and so have
// no pxrInterop.h). Looks the text up first; a new text interns as test
// data (a compile-phase intern), like usdGenUsd::ToToken.
#pragma once
#include "usdGen/api/token.h"
#include "usdGen/api/ext/tokenScope.h"

#include <string_view>

inline usdGen::UsdGenToken TestToken(std::string_view text)
{
    usdGen::UsdGenToken const found = usdGen::UsdGenFindToken(text);
    if (!found.IsEmpty() || text.empty()) return found;
    usdGen::UsdGenCompileTokenScope scope;
    return usdGen::UsdGenInternToken(text);
}
