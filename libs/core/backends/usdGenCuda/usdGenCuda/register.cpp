// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCuda's registration: the "cuda" backend. The curves' kernel rows and
// device program register from usdGenCurvesCuda, loaded after this module.
// Without the CUDA toolkit the backend registers unavailable, so its
// contract keeps the capability version.
#include "usdGenCuda/export.h"
#include "usdGenCuda/compile/program.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ext/op.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/api/ext/backendDef.h"
#include "usdGen/api/ext/registrar.h"
#if defined(USDGEN_ENABLE_CUDA)
#include "usdGenCuda/exec/provider.h"
#endif

#include <atomic>
#include <memory>
#include <string>

using namespace usdGen;

namespace {

std::atomic<UsdGenBackendId> gCudaBackend{UsdGenBackendId::Invalid};

// Admission is the program's: the graph's primitive registered it.
bool Validate(UsdGenGraphSpec const &spec, UsdGenDiagnostics *diagnostics)
{
    cuda::ProgramDef const *program = cuda::ProgramFor(spec);
    if (program && program->validate) return program->validate(spec, diagnostics);
    if (diagnostics) diagnostics->Error("no CUDA program is registered for this graph");
    return false;
}

#if !defined(USDGEN_ENABLE_CUDA)
bool Unavailable(std::string *reason)
{
    if (reason) *reason = "CUDA support was not enabled in this build";
    return false;
}

std::shared_ptr<UsdGenSessionDeviceProvider> NoProvider(UsdGenBackendProviderDesc const &,
                                                        std::string *reason)
{
    Unavailable(reason);
    return nullptr;
}
#endif

}  // namespace

UsdGenBackendId usdGen::UsdGenCudaBackendId() noexcept
{
    UsdGenBackendId const id = gCudaBackend.load(std::memory_order_acquire);
    return id != UsdGenBackendId::Invalid ? id : UsdGenFindBackend("cuda");
}

extern "C" USDGEN_CUDA_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    UsdGenBackendDef def;
    def.name = "cuda";
    def.label = "CUDA";
    def.capabilityVersion = kUsdGenCudaCapabilityVersion;
    def.flags = kUsdGenBackendPoisonable | kUsdGenBackendNeedsDevicePublication |
                kUsdGenBackendReusesPlan;
    def.validate = &Validate;
#if defined(USDGEN_ENABLE_CUDA)
    def.captureCallerDevice = &CaptureCudaCallerDevice;
    def.createProvider = &CreateCudaSessionProvider;
#else
    def.available = &Unavailable;
    def.unavailableReason = "CUDA support was not enabled in this build";
    def.createProvider = &NoProvider;
#endif
    gCudaBackend.store(r.RegisterBackend(def), std::memory_order_release);
}
