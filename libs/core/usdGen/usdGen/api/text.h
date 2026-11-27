// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenText: engine-owned immutable text (labels, asset paths, messages).
#pragma once
#include "usdGen/api/cowArray.h"
#include "usdGen/api/export.h"

#include <cstring>
#include <string_view>

namespace usdGen {

class UsdGenText {
public:
    UsdGenText() noexcept = default;
    /// Allocates in usdGen.
    USDGEN_CORE_API explicit UsdGenText(std::string_view s);

    std::string_view View() const noexcept
    {
        return _chars.empty() ? std::string_view() : std::string_view(_chars.cdata(), _chars.size() - 1);
    }
    /// NUL-terminated.
    char const *c_str() const noexcept { return _chars.empty() ? "" : _chars.cdata(); }
    bool empty() const noexcept { return _chars.size() <= 1; }

    friend bool operator==(UsdGenText const &a, UsdGenText const &b) noexcept { return a.View() == b.View(); }
    friend bool operator!=(UsdGenText const &a, UsdGenText const &b) noexcept { return !(a == b); }
    /// Unsigned byte order, shorter prefix first.
    friend bool operator<(UsdGenText const &a, UsdGenText const &b) noexcept
    {
        std::string_view const x = a.View(), y = b.View();
        size_t const n = x.size() < y.size() ? x.size() : y.size();
        int const c = n ? std::memcmp(x.data(), y.data(), n) : 0;
        return c < 0 || (c == 0 && x.size() < y.size());
    }

private:
    UsdGenCowArray<char> _chars;   // size includes the NUL; empty when default
};

}  // namespace usdGen
