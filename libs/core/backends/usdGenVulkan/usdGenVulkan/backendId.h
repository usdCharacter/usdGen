// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The handle usdGenVulkan registered as "vulkan" (registries/backend.h). Light:
// no pxr, safe in .cu sources.
#pragma once
#include "usdGenVulkan/export.h"
#include "usdGen/api/ids.h"

namespace usdGen {

/// Invalid until usdGenVulkan's usdGenRegister ran; asking builds the registries.
USDGEN_VULKAN_API UsdGenBackendId UsdGenVulkanBackendId() noexcept;

}  // namespace usdGen
