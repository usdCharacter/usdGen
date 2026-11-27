// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#  define _CRT_SECURE_NO_WARNINGS   // std::getenv
#endif
#include "usdGen/api/env.h"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace usdGen {

char const *UsdGenGetEnv(char const *name) noexcept
{
    return name ? std::getenv(name) : nullptr;
}

// TfGetenvBool: "true", "yes", "on" or "1", any case.
bool UsdGenGetEnvBool(char const *name, bool fallback) noexcept
{
    char const *const v = UsdGenGetEnv(name);
    if (!v || !*v) return fallback;
    char lower[8];
    size_t const n = std::strlen(v);
    if (n >= sizeof(lower)) return false;
    for (size_t i = 0; i <= n; ++i) lower[i] = char(std::tolower(static_cast<unsigned char>(v[i])));
    return !std::strcmp(lower, "true") || !std::strcmp(lower, "yes") || !std::strcmp(lower, "on") ||
           !std::strcmp(lower, "1");
}

// TfGetenvInt as the engine called it (a conversion failure gave the
// fallback): leading digits, no digits or out of int range is the fallback.
int UsdGenGetEnvInt(char const *name, int fallback) noexcept
{
    char const *const v = UsdGenGetEnv(name);
    if (!v || !*v) return fallback;
    char *end = nullptr;
    errno = 0;
    long const r = std::strtol(v, &end, 10);
    if (end == v || errno == ERANGE || r < long(INT_MIN) || r > long(INT_MAX)) return fallback;
    return int(r);
}

}  // namespace usdGen
