// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenPlane: one named, typed array plane (a primitive's extra planes and
// a host adapter's published ones). The engine never reads it; it lives in
// api/ so a primitive and its Hydra adapter share it without including each
// other (gate 9b).
#pragma once

#include "usdGen/api/cowArray.h"
#include "usdGen/api/token.h"

#include <cstdint>

namespace usdGen {

/// Exactly one of `f`/`i` holds the payload; `arity` values per element.
struct UsdGenPlane
{
    UsdGenToken      name;
    UsdGenToken      interpolation;   // vertex | uniform | constant
    UsdGenToken      type;            // float | int
    uint8_t      arity = 1;
    UsdGenFloatArray f;
    UsdGenIntArray   i;
};

}  // namespace usdGen
