// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_EXPRESSIONS_VALUE_SHAPE_H
#define USDGEN_EXPRESSIONS_VALUE_SHAPE_H

#include "usdGenExpr/expressions/context.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"

#include "usdGen/api/value.h"

#include <cstdint>
#include "usdGenExpr/export.h"

namespace usdGen::expr {

// One row of the closed value-type table: the native type names the
// builders emit (Sdf's value type names, scalar and array). Tuple dims mirror
// SdfTupleDimensions; scalar is Invalid for types the evaluator cannot carry.
// makeDefault yields Sdf's default value (every row has one).
// testUsdGenValueTypeTable compares every row against SdfSchema.
struct UsdGenValueTypeInfo {
    char const *name = nullptr;
    ScalarType scalar = ScalarType::Invalid;
    uint8_t dimensionCount = 0;
    uint8_t dimension0 = 0;
    uint8_t dimension1 = 0;
    bool isArray = false;
    UsdGenValue (*makeDefault)() = nullptr;

    explicit operator bool() const { return name != nullptr; }
};

// Rows sorted by name; an unknown name yields an empty row. No allocation.
USDGEN_EXPR_API UsdGenValueTypeInfo FindValueType(UsdGenToken const &nativeType);
USDGEN_EXPR_API UsdGenValueTypeInfo FindValueType(char const *nativeType);
USDGEN_EXPR_INTERNAL_API UsdGenSpan<const UsdGenValueTypeInfo> AllValueTypes();

// Sdf's default value of the type; empty for an unknown or sdf-only type.
USDGEN_EXPR_API UsdGenValue ValueTypeDefault(UsdGenValueTypeInfo const &type);

// `arrayCountKnown` distinguishes an authored empty fixed array from an
// array declaration with no typed value from which a fixed count can be
// obtained; the latter is deliberately returned as an invalid shape.
USDGEN_EXPR_INTERNAL_API ValueShape ValueShapeFromValueType(UsdGenValueTypeInfo const &type,
                                   uint32_t arrayElementCount = 0,
                                   bool arrayCountKnown = false);

// Hydra transports USD's native type token; resolved through the table so
// role aliases and tuple dimensions take the same path as direct USD reads
// (usdGenUsd's ValueShapeFromSdfType).
USDGEN_EXPR_API ValueShape ValueShapeFromNativeType(UsdGenToken const &nativeType,
                                    uint32_t arrayElementCount = 0,
                                    bool arrayCountKnown = false);

} // namespace usdGen::expr

#endif
