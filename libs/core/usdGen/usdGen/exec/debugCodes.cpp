#include "usdGen/debugCodes.h"

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
        "usdGen: one line per fur optical-depth bake (grid, voxel size, CVs)");
}

PXR_NAMESPACE_CLOSE_SCOPE
