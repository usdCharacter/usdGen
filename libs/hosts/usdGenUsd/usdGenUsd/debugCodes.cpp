// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/debugCodes.h"

#include "usdGen/api/debug.h"

#include "pxr/base/tf/registryManager.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfDebug)
{
    TF_DEBUG_ENVIRONMENT_SYMBOL(USDGEN_COMMIT,
        "usdGen: one line per cook (compile, cache, tiles, timings)");
    TF_DEBUG_ENVIRONMENT_SYMBOL(USDGEN_SCHEDULE,
        "usdGen: one line per operator per cook (re-captured or reused, chunks)");
    TF_DEBUG_ENVIRONMENT_SYMBOL(USDGEN_INGRESS,
        "usdGen: groom scene index notices and description captures");
    TF_DEBUG_ENVIRONMENT_SYMBOL(USDGEN_FUR,
        "usdGen: one line per fur optical-depth bake (grid, voxel size, items)");
}

PXR_NAMESPACE_CLOSE_SCOPE

namespace usdGenUsd {

void SyncDebugCodes() noexcept
{
    using PXR_NS::TfDebug;
    using usdGen::UsdGenDebugCode;
    usdGen::UsdGenSetDebugEnabled(UsdGenDebugCode::Commit, TfDebug::IsEnabled(PXR_NS::USDGEN_COMMIT));
    usdGen::UsdGenSetDebugEnabled(UsdGenDebugCode::Schedule, TfDebug::IsEnabled(PXR_NS::USDGEN_SCHEDULE));
    usdGen::UsdGenSetDebugEnabled(UsdGenDebugCode::Ingress, TfDebug::IsEnabled(PXR_NS::USDGEN_INGRESS));
    usdGen::UsdGenSetDebugEnabled(UsdGenDebugCode::Fur, TfDebug::IsEnabled(PXR_NS::USDGEN_FUR));
}

}  // namespace usdGenUsd
