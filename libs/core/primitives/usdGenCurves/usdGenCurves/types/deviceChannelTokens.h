// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Device channel names of the CurveSet backends, interned once (8.9).
#pragma once
#include "usdGenCurves/export.h"
#include "usdGen/api/token.h"

namespace usdGen {

struct UsdGenCurveChannelTokens {
    UsdGenToken points, rest, width, widths, hairT, curveOffsets, curveId, stableIds, skinprim, skinprimuv, rootPrim, rootUV, sourceRootT, sourceRootB, sourceRootN, restPoints, sourceChunks;
};

/// Built by usdGenRegister (compile scope) or on first use; never freed. The
/// cook path only reads it.
USDGEN_CURVES_API UsdGenCurveChannelTokens const &UsdGenCurveChannels() noexcept;

}  // namespace usdGen
