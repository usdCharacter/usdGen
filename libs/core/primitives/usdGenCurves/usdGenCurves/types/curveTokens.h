// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curve primitive's input vocabulary: BasisCurves type, basis, wrap and
// widths words, the curve loader's option words and the plane, task and
// preview names the curve sources use. Interned once (first use;
// usdGenRegister touches it before the registries freeze). Never interned
// by load-time code.
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
    // the forwarded source displayColor plane (UsdGenSourceColorPlane)
    UsdGenToken sourceColor;
    // whole-strand planes (wholeStrand.cpp) and cage ownership planes
    // (surfaceCageSource.cpp)
    UsdGenToken strandSeed, strandIndex, tubeId, regionId, hierarchyLevel;
    // CUDA task types (cuda/cudaExecution.cpp)
    UsdGenToken cudaScatterGrow, cudaPublication;
    // value preview (preview/valuePreview.cpp)
    UsdGenToken previewDestination, groom, point, map, mapChannel, mapFirstChannel,
        mapChannelCount;
    UsdGenToken previewChannels[3], rgbChannels[3];
};

/// Valid for the life of the process.
USDGEN_CURVES_API UsdGenCurveTokenTable const &UsdGenCurveTokens() noexcept;

}  // namespace usdGen
