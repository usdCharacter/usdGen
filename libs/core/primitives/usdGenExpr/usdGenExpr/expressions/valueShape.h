#ifndef USDGEN_EXPRESSIONS_VALUE_SHAPE_H
#define USDGEN_EXPRESSIONS_VALUE_SHAPE_H

#include "usdGen/expressions/context.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/valueTypeName.h"

#include <cstdint>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen::expr {

// Decode the actual USD value type rather than parsing its serialization
// token.  `arrayCountKnown` distinguishes an authored empty fixed array from
// an array declaration with no typed value from which a fixed count can be
// obtained; the latter is deliberately returned as an invalid shape.
ValueShape ValueShapeFromSdfType(SdfValueTypeName const& type,
                                 uint32_t arrayElementCount = 0,
                                 bool arrayCountKnown = false);

// Hydra transports USD's native type token.  Resolve it through SdfSchema so
// role aliases and tuple dimensions take the same path as direct USD reads.
ValueShape ValueShapeFromNativeType(TfToken const& nativeType,
                                    uint32_t arrayElementCount = 0,
                                    bool arrayCountKnown = false);

// Returns false for a missing, incompatible, non-array, or overlarge typed
// value.  A correctly authored empty array succeeds with *count == 0.
bool FixedArrayElementCount(SdfValueTypeName const& type,
                            VtValue const& value,
                            uint32_t* count);

} // namespace usdGen::expr

#endif
