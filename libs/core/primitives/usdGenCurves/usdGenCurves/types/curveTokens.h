// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curve primitive's input vocabulary: BasisCurves type, basis, wrap and
// widths words, and the curve loader's option words. Interned once (first
// use; usdGenRegister touches it before the registries freeze).
#pragma once
#include "usdGenCurves/export.h"
#include "usdGen/api/token.h"

namespace usdGen {

struct UsdGenCurveTokenTable {
    // curve input words (UsdGenCurveSetData defaults, admission checks)
    UsdGenToken hair, linear, cubic, bspline, catmullRom, pinned, nonperiodic, periodic,
        vertexWidths;
    // curve loader options (curveLoader.cpp)
    UsdGenToken useRest, usdGenUseRest, resampleTo, idSource, staleAction, rebind, expectEpoch,
        primvar, index, warn, ignore, block, never, onError, always;
};

/// Valid for the life of the process.
USDGEN_CURVES_API UsdGenCurveTokenTable const &UsdGenCurveTokens() noexcept;

}  // namespace usdGen
