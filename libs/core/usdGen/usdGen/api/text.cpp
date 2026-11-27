// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/text.h"

namespace usdGen {

UsdGenText::UsdGenText(std::string_view s)
{
    if (s.empty()) return;
    _chars.resize(s.size() + 1);
    char *out = _chars.data();
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
}

}  // namespace usdGen
