// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenVulkan's registration: the Vulkan kernel rows of the built-in curve
// ops and, when the runtime is built, the "vulkan" backend. Loaded after
// usdGenCurves (USDGEN_DEFAULT_EXTENSIONS), whose op ids the rows attach to.
#include "usdGenVulkan/export.h"
#include "usdGenVulkan/vulkan/defaultProvider.h"
#include "usdGenVulkan/vulkan/vkStages.h"
#include "usdGenCurves/ops/kernelOps.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

#include <memory>
#include <string>

namespace usdGen {

uint32_t UsdGenVulkanCapabilityVersionForBackendContract() noexcept;   // executionBackendVulkanProvider.cpp

}  // namespace usdGen

using namespace usdGen;

namespace {

#if defined(USDGEN_ENABLE_VULKAN_RUNTIME)
std::shared_ptr<UsdGenSessionDeviceProvider> CreateProvider(UsdGenBackendProviderDesc const &,
                                                            std::string *reason)
{
    return vulkan::CreateDefaultVulkanSessionProvider(reason);
}
#endif

}  // namespace

extern "C" USDGEN_VULKAN_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
#if defined(USDGEN_ENABLE_VULKAN_RUNTIME)
    // Without the runtime (test-only builds) the backend stays unregistered:
    // a Session reaches it only through an injected provider.
    UsdGenBackendDef def;
    def.name = "vulkan";
    def.label = "Vulkan";
    def.capabilityVersion = UsdGenVulkanCapabilityVersionForBackendContract();
    def.flags = kUsdGenBackendAuthoritativeRevisions | kUsdGenBackendNeedsDevicePublication;
    def.createProvider = &CreateProvider;
    r.RegisterBackend(def);
#endif
    UsdGenCurvesKernelOps const &ops = UsdGenCurvesBuiltinKernelOps();
    UsdGenVulkanKernelOps k;
    k.curveSource = ops.curveSource;
    k.referenceSource = ops.referenceSource;
    k.scatter = ops.scatter;
    k.grow = ops.grow;
    k.width = ops.width;
    k.length = ops.length;
    k.widthBlend = ops.widthBlend;
    k.noise = ops.noise;
    k.deform = ops.deform;
    k.resample = ops.resample;
    UsdGenRegisterVulkanKernels(r, k);
}
