// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Engine vocabulary that names no op, type or primitive.
#pragma once
#include "usdGen/export.h"
#include "usdGen/api/token.h"

namespace usdGen {

struct UsdGenEngineTokens {
    // plane rate / scalar (UsdGenPlane, types.h)
    UsdGenToken vertex, uniform, constant, float_, int_;
    // base node parameters (compiler.cpp:779-801) and the "usdGen:" prefix
    UsdGenToken input, surface, seed, enabled, guides, mask, usdGenPrefix;
    // the expression output default
    UsdGenToken result;
    // S4-3 (Core-B): varying rate; surface defaults; the preview evaluation default
    UsdGenToken varying, none, rightHanded, edgeAndCorner, cornersPlus1, catmullClark,
        primitive;
};

/// Valid from the first registry access for the life of the process.
USDGEN_CORE_API UsdGenEngineTokens const &UsdGenTokens() noexcept;

}  // namespace usdGen
