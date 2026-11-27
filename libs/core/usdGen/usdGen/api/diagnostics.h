// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Compile and cook diagnostics.
#pragma once
#include "usdGen/api/export.h"

#include <string>
#include <string_view>
#include <vector>

namespace usdGen {

/// Diagnostics collected by a node; the imaging layer is the one that logs
/// (the engine core stays TF_WARN-free on hot paths, 03 §9.2).
struct UsdGenDiagnostics
{
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    bool HasErrors() const { return !errors.empty(); }
    /// Out of line: every message string is allocated in usdGen.
    USDGEN_API void Error(std::string_view msg);
    USDGEN_API void Warn(std::string_view msg);
};

}  // namespace usdGen
