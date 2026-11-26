// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Value shapes of USD attribute types. Host side: the engine resolves native
// type tokens through its closed table (usdGenExpr/expressions/valueShape.h).
#pragma once

#include "usdGenExpr/expressions/valueShape.h"

#include "pxr/pxr.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/valueTypeName.h"

#include <cstdint>

namespace usdGen::expr {

// Decode the actual USD value type rather than parsing its serialization
// token. `arrayCountKnown` as in ValueShapeFromValueType.
ValueShape ValueShapeFromSdfType(PXR_NS::SdfValueTypeName const &type,
                                 uint32_t arrayElementCount = 0,
                                 bool arrayCountKnown = false);

// Returns false for a missing, incompatible, non-array, or overlarge typed
// value. A correctly authored empty array succeeds with *count == 0.
bool FixedArrayElementCount(PXR_NS::SdfValueTypeName const &type,
                            PXR_NS::VtValue const &value,
                            uint32_t *count);

} // namespace usdGen::expr
