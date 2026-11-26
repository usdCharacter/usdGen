// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CurveSet's publish description (api/publishDesc.h): basis curves with
// value-preview and occluder hooks.
#pragma once
#include "usdGen/api/publishDesc.h"

namespace usdGen {

/// Fills the description (interns its tokens) and returns it; registration
/// only. It outlives the registry.
UsdGenPublishDesc const *UsdGenMakeCurvePublishDesc();

}  // namespace usdGen
