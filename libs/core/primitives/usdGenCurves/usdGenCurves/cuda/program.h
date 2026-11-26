// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curves' CUDA device program (usdGenCuda/compile/program.h).
#pragma once
#include "usdGenCuda/compile/program.h"

namespace usdGen {

/// Compile, admission, workspace state and staged job of a curve graph.
cuda::ProgramDef const &UsdGenCurvesCudaProgram() noexcept;

}  // namespace usdGen
