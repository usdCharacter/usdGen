// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCuda's registration: the CUDA kernel rows of the built-in curve ops
// (the capability matrix reads them in every build) and the "cuda" backend.
// Loaded after usdGenCurves (USDGEN_DEFAULT_EXTENSIONS), whose op ids the
// rows attach to. Without the CUDA toolkit the backend registers unavailable,
// so its contract keeps the capability version.
#include "usdGenCuda/export.h"
#include "usdGenCuda/backendId.h"
#include "usdGenCuda/cudaExecution.h"
#include "usdGenCuda/cudaStages.h"
#include "usdGenCurves/ops/kernelOps.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"
#if defined(USDGEN_ENABLE_CUDA)
#include "usdGenCuda/cudaSessionProvider.h"
#endif

#include <atomic>
#include <memory>
#include <string>

using namespace usdGen;

namespace {

std::atomic<UsdGenBackendId> gCudaBackend{UsdGenBackendId::Invalid};

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
    def.validate = &ValidateCudaGraph;
#if defined(USDGEN_ENABLE_CUDA)
    def.captureCallerDevice = &CaptureCudaCallerDevice;
    def.createProvider = &CreateCudaSessionProvider;
#else
    def.available = &Unavailable;
    def.unavailableReason = "CUDA support was not enabled in this build";
    def.createProvider = &NoProvider;
#endif
    gCudaBackend.store(r.RegisterBackend(def), std::memory_order_release);
    UsdGenCurvesKernelOps const &ops = UsdGenCurvesBuiltinKernelOps();
    UsdGenCudaKernelOps cuda;
    cuda.scatter = ops.scatter;
    cuda.grow = ops.grow;
    cuda.noise = ops.noise;
    cuda.length = ops.length;
    cuda.width = ops.width;
    cuda.widthBlend = ops.widthBlend;
    cuda.curveSource = ops.curveSource;
    cuda.deform = ops.deform;
    cuda.referenceSource = ops.referenceSource;
    UsdGenRegisterCurvesCuda(r, cuda);
}
