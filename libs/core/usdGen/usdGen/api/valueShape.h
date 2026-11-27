// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Value domains, scalar types and shapes a connected parameter carries
// (moved from usdGenExpr/expressions/context.h, which includes this).
#ifndef USDGEN_API_VALUE_SHAPE_H
#define USDGEN_API_VALUE_SHAPE_H

#include <cstdint>

namespace usdGen::expr {

enum class Domain : uint8_t { None = 0, Groom = 1, Primitive = 2, Point = 4,
                              All = 7 };
constexpr Domain operator|(Domain a, Domain b) {
    return Domain(uint8_t(a) | uint8_t(b));
}
constexpr bool HasDomain(Domain mask, Domain d) {
    return (uint8_t(mask) & uint8_t(d)) != 0;
}

enum class ScalarType : uint8_t { Invalid, Bool, Int32, UInt32, Int64, UInt64,
                                  Float16, Float32, Float64 };

struct ValueShape {
    ScalarType scalar = ScalarType::Invalid;
    uint32_t elementCount = 1;
    uint32_t components = 1;
    uint32_t rows = 1;
    uint32_t columns = 1;
    bool isArray = false;
};

} // namespace usdGen::expr
#endif
