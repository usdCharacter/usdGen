// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/rbfAnimation.h"
#include "pxr/base/tf/envSetting.h"
namespace usdGen {
TF_DEFINE_ENV_SETTING(USDGEN_STORM_GPU_DEFORM, false,
    "Use resident Storm GPU points for compatible terminal RBF pose previews.");
bool UsdGenRbfAnimationEnabled() { return TfGetEnvSetting(USDGEN_STORM_GPU_DEFORM); }
}
