// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenToken: an interned string. Reps are immortal (never freed) and live
// in usdGen. UsdGenInternToken is the only text -> token door;
// UsdGenFindToken only looks up. Equality is identity; order is TfToken's
// (empty first, then its 8-byte compare code, then unsigned bytes); Hash()
// is FNV-1a 64 of the bytes.
#pragma once
#include "usdGen/api/export.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace usdGen {

/// Immortal interned rep. Read-only once published.
struct UsdGenTokenRep {
    uint64_t hash;                 // FNV-1a 64 of the bytes
    uint64_t compareCode;          // Tf's token compare code: the first 8 chars, widened from char
    uint32_t id;                   // unique, intern order, 1-based
    uint32_t size;
    char const *text;              // NUL-terminated
    std::string const *string;     // GetString()
};

class UsdGenToken;
USDGEN_CORE_API UsdGenToken UsdGenInternToken(std::string_view text);
USDGEN_CORE_API UsdGenToken UsdGenFindToken(std::string_view text) noexcept;

/// Host token interop (tests only; tests/utils/pxrInterop.h specialises it).
/// A specialisation has value = true, static UsdGenToken In(H const &) and
/// static H Out(UsdGenToken). The primary admits nothing.
template <class H> struct UsdGenHostToken : std::false_type {};

class UsdGenToken {
public:
    constexpr UsdGenToken() noexcept = default;                     // empty

    template <class H, std::enable_if_t<UsdGenHostToken<H>::value, int> = 0>
    UsdGenToken(H const &h) : UsdGenToken(UsdGenHostToken<H>::In(h)) {}
    template <class H, std::enable_if_t<UsdGenHostToken<H>::value, int> = 0>
    operator H() const { return UsdGenHostToken<H>::Out(*this); }

    bool IsEmpty() const noexcept { return !_rep; }
    char const *GetText() const noexcept { return _rep ? _rep->text : ""; }
    char const *data() const noexcept { return GetText(); }
    /// The empty string when empty.
    USDGEN_CORE_API std::string const &GetString() const noexcept;
    std::string_view View() const noexcept
    {
        return _rep ? std::string_view(_rep->text, _rep->size) : std::string_view();
    }
    size_t size() const noexcept { return _rep ? _rep->size : 0; }
    size_t Hash() const noexcept { return _rep ? size_t(_rep->hash) : 0; }
    uint32_t Id() const noexcept { return _rep ? _rep->id : 0; }

    friend bool operator==(UsdGenToken a, UsdGenToken b) noexcept { return a._rep == b._rep; }
    friend bool operator!=(UsdGenToken a, UsdGenToken b) noexcept { return a._rep != b._rep; }
    // Text compares (no intern).
    friend bool operator==(UsdGenToken a, char const *s) noexcept { return a.View() == std::string_view(s ? s : ""); }
    friend bool operator!=(UsdGenToken a, char const *s) noexcept { return !(a == s); }
    friend bool operator==(char const *s, UsdGenToken a) noexcept { return a == s; }
    friend bool operator!=(char const *s, UsdGenToken a) noexcept { return !(a == s); }
    friend bool operator==(UsdGenToken a, std::string const &s) noexcept { return a.View() == std::string_view(s); }
    friend bool operator!=(UsdGenToken a, std::string const &s) noexcept { return !(a == s); }
    friend bool operator==(std::string const &s, UsdGenToken a) noexcept { return a == s; }
    friend bool operator!=(std::string const &s, UsdGenToken a) noexcept { return !(a == s); }
    /// TfToken's order (tf/token.h operator<, token.cpp _ComputeCompareCode):
    /// empty first, then the compare code, then unsigned bytes.
    friend bool operator<(UsdGenToken a, UsdGenToken b) noexcept
    {
        if (a._rep == b._rep) return false;
        if (!a._rep || !b._rep) return !a._rep;
        if (a._rep->compareCode != b._rep->compareCode) return a._rep->compareCode < b._rep->compareCode;
        size_t const n = a._rep->size < b._rep->size ? a._rep->size : b._rep->size;
        int const c = n ? std::memcmp(a._rep->text, b._rep->text, n) : 0;
        return c < 0 || (c == 0 && a._rep->size < b._rep->size);
    }
    friend bool operator>(UsdGenToken a, UsdGenToken b) noexcept { return b < a; }
    friend bool operator<=(UsdGenToken a, UsdGenToken b) noexcept { return !(b < a); }
    friend bool operator>=(UsdGenToken a, UsdGenToken b) noexcept { return !(a < b); }

    struct HashFunctor { size_t operator()(UsdGenToken t) const noexcept { return t.Hash(); } };

private:
    friend USDGEN_CORE_API UsdGenToken UsdGenInternToken(std::string_view);
    friend USDGEN_CORE_API UsdGenToken UsdGenFindToken(std::string_view) noexcept;
    constexpr explicit UsdGenToken(UsdGenTokenRep const *r) noexcept : _rep(r) {}
    UsdGenTokenRep const *_rep = nullptr;
};
using UsdGenTokenVector = std::vector<UsdGenToken>;

/// TfHash and VtValue find it (deterministic, process independent).
inline size_t hash_value(UsdGenToken t) noexcept { return t.Hash(); }

}  // namespace usdGen

namespace std {
template <>
struct hash<usdGen::UsdGenToken> {
    size_t operator()(usdGen::UsdGenToken t) const noexcept { return t.Hash(); }
};
}  // namespace std
