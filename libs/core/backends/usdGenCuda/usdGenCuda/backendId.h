// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The handle usdGenCuda registered as "cuda" (registries/backend.h). Light:
// no pxr, safe in .cu sources.
#pragma once
#include "usdGenCuda/export.h"
#include "usdGen/registries/handles.h"

namespace usdGen {

/// Invalid until usdGenCuda's usdGenRegister ran; asking builds the registries.
USDGEN_CUDA_API UsdGenBackendId UsdGenCudaBackendId() noexcept;

}  // namespace usdGen
