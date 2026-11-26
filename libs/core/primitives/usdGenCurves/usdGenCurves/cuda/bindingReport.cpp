// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/cuda/bindingReport.h"
#include "usdGen/exec/session.h"

namespace usdGen {

std::vector<UsdGenCudaBindingStats> UsdGenCudaBindingStatsOf(UsdGenSessionSnapshot const *snapshot)
{
    if (!snapshot || !snapshot->deviceReport ||
        snapshot->deviceReport->kind != UsdGenCudaBindingReport::kKind)
        return {};
    return static_cast<UsdGenCudaBindingReport const &>(*snapshot->deviceReport).stats;
}

}  // namespace usdGen
