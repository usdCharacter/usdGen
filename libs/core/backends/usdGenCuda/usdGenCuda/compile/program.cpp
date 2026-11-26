// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Key functions of the program interfaces: one typeinfo, in usdGenCuda.
#include "usdGenCuda/compile/program.h"

namespace usdGen::cuda {

Job::~Job() = default;

UsdGenRef<const UsdGenPublication> Job::Report() const noexcept
{
    return {};
}

ProgramState::~ProgramState() = default;

}  // namespace usdGen::cuda
