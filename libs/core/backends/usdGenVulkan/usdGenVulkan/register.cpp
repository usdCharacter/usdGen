// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenVulkan's registration: the "vulkan" backend (unavailable without the
// runtime). The primitives' kernel rows and programs register from their own
// modules, loaded after this one (USDGEN_DEFAULT_EXTENSIONS).
#include "usdGenVulkan/export.h"
#include "usdGenVulkan/backendId.h"
#include "usdGenVulkan/vulkan/defaultProvider.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

#include <atomic>
#include <memory>
#include <string>

namespace usdGen {

uint32_t UsdGenVulkanCapabilityVersionForBackendContract() noexcept;   // executionBackendVulkanProvider.cpp

}  // namespace usdGen

using namespace usdGen;

namespace {

std::atomic<UsdGenBackendId> gVulkanBackend{UsdGenBackendId::Invalid};

#if defined(USDGEN_ENABLE_VULKAN_RUNTIME)
std::shared_ptr<UsdGenSessionDeviceProvider> CreateProvider(UsdGenBackendProviderDesc const &,
                                                            std::string *reason)
{
    return vulkan::CreateDefaultVulkanSessionProvider(reason);
}
#else
char const kNoRuntime[] = "Vulkan support was not enabled in this build";

bool Unavailable(std::string *reason)
{
    if (reason) *reason = kNoRuntime;
    return false;
}

std::shared_ptr<UsdGenSessionDeviceProvider> CreateProvider(UsdGenBackendProviderDesc const &,
                                                            std::string *reason)
{
    Unavailable(reason);
    return nullptr;
}
#endif

}  // namespace

UsdGenBackendId usdGen::UsdGenVulkanBackendId() noexcept
{
    UsdGenBackendId const id = gVulkanBackend.load(std::memory_order_acquire);
    return id != UsdGenBackendId::Invalid ? id : UsdGenFindBackend("vulkan");
}

extern "C" USDGEN_VULKAN_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    // Without the runtime (test-only builds) the backend registers
    // unavailable: a Session reaches it only through an injected provider.
    UsdGenBackendDef def;
    def.name = "vulkan";
    def.label = "Vulkan";
    def.capabilityVersion = UsdGenVulkanCapabilityVersionForBackendContract();
    def.flags = kUsdGenBackendAuthoritativeRevisions | kUsdGenBackendNeedsDevicePublication;
    def.createProvider = &CreateProvider;
#if !defined(USDGEN_ENABLE_VULKAN_RUNTIME)
    def.available = &Unavailable;
    def.unavailableReason = kNoRuntime;
#endif
    gVulkanBackend.store(r.RegisterBackend(def), std::memory_order_release);
}
