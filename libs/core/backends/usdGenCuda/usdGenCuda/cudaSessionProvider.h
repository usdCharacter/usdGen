// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CUDA Session device provider (CUDA builds only).
#pragma once
#include "usdGenCuda/export.h"
#include "usdGen/exec/sessionDeviceProvider.h"

#include <memory>
#include <string>

namespace usdGen {

struct UsdGenBackendProviderDesc;

/// One execution workspace, replaced when the cook's device or graph
/// description changes; each cook runs source -> operator* -> final as a
/// task graph on the device's shared dispatcher.
USDGEN_CUDA_API std::shared_ptr<UsdGenSessionDeviceProvider> CreateCudaSessionProvider(
    UsdGenBackendProviderDesc const &, std::string *reason);

/// cudaGetDevice on the calling thread, -2 on failure.
USDGEN_CUDA_API int CaptureCudaCallerDevice() noexcept;

}  // namespace usdGen
