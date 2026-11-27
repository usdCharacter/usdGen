// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curves' Vulkan program (registered by register.cpp).
#pragma once
#include "usdGenVulkan/compile/program.h"

#include <memory>
#include <string>

namespace usdGen::vulkan {

/// Builds the curve pipelines on `context` from the registered SPIR-V. Null
/// with a reason when the width module is missing or a required pipeline
/// cannot be created.
UsdGenRef<Program> CreateCurvesProgram(std::shared_ptr<DeviceContext> const &context,
                                       ShaderLibrary const &library, std::string *reason);

}  // namespace usdGen::vulkan
