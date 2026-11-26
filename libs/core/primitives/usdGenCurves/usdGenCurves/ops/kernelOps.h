// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Op ids of the built-in curve ops that GPU kernel rows attach to. Filled by
// usdGenCurves' usdGenRegister; a backend library that loads after it reads
// them from its own usdGenRegister (step 8.6 moves the rows into the curves'
// GPU programs).
#pragma once
#include "usdGenCurves/export.h"
#include "usdGen/api/ids.h"

namespace usdGen {

struct UsdGenCurvesKernelOps {
    UsdGenOpId curveSource = UsdGenOpId::Invalid, referenceSource = UsdGenOpId::Invalid,
               scatter = UsdGenOpId::Invalid, grow = UsdGenOpId::Invalid,
               width = UsdGenOpId::Invalid, length = UsdGenOpId::Invalid,
               widthBlend = UsdGenOpId::Invalid, noise = UsdGenOpId::Invalid,
               deform = UsdGenOpId::Invalid, resample = UsdGenOpId::Invalid;
};

/// Invalid ids until usdGenCurves has registered.
USDGEN_CURVES_API UsdGenCurvesKernelOps const &UsdGenCurvesBuiltinKernelOps() noexcept;

}  // namespace usdGen
