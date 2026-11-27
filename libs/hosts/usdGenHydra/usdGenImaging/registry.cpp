#include "usdGenImaging/registry.h"

namespace usdGenImaging {

UsdGenImagingRegistry& UsdGenImagingRegistry::GetInstance() {
    static UsdGenImagingRegistry instance;
    return instance;
}

void UsdGenImagingRegistry::RegisterRenderInstance(
    const std::string& renderInstanceId) {
    (void)renderInstanceId;
    // M1: build the session key (weak stage | usdGen:sessionId) and hold the
    // generation store here. M0 is pass-through only.
}

}  // namespace usdGenImaging
