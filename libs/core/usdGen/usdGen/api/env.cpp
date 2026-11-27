// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#  define _CRT_SECURE_NO_WARNINGS   // std::getenv
#endif
#include "usdGen/api/env.h"

#include "pxr/base/tf/getenv.h"

#include <cstdlib>

namespace usdGen {

char const *UsdGenGetEnv(char const *name) noexcept
{
    return name ? std::getenv(name) : nullptr;
}

bool UsdGenGetEnvBool(char const *name, bool fallback) noexcept
{
    try {
        return name ? PXR_NS::TfGetenvBool(name, fallback) : fallback;
    } catch (...) {
        return fallback;
    }
}

int UsdGenGetEnvInt(char const *name, int fallback) noexcept
{
    try {
        return name ? PXR_NS::TfGetenvInt(name, fallback) : fallback;
    } catch (...) {
        return fallback;
    }
}

}  // namespace usdGen
