// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Input kinds a host reads into input records (step 7). Declared in 7.2;
// registered and consumed from 7.3a, when typed input payloads become
// records.
#ifndef USDGEN_API_INPUT_DESC_H
#define USDGEN_API_INPUT_DESC_H

#include "usdGen/api/elementSet.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"
#include "usdGen/exec/epoch.h"

#include <cstdint>

namespace usdGen {

struct UsdGenInputRecord;

enum UsdGenInputSlotFlag : uint8_t {
    kUsdGenInputRequired = 1,
    kUsdGenInputPrimvar = 2,   // read primvars:<attribute>, flattening indexed
    kUsdGenInputCounts = 4,    // integer counts that become Offsets(level)
};

/// One attribute a host reads into one channel of the input record.
struct UsdGenInputSlot {
    UsdGenToken attribute;     // "points", "curveVertexCounts", "usdGen:curveId"
    UsdGenSlot channel = kUsdGenNoSlot;  // kUsdGenNoSlot for kUsdGenInputCounts
    uint8_t level = 0;         // kUsdGenInputCounts: the level these counts define
    uint8_t flags = 0;
};

/// Registered per input kind. A host reads 'schemaType' prims per 'slots',
/// time-sampled, into a UsdGenInputRecord; it never links the primitive.
struct UsdGenInputDesc {
    UsdGenTypeId kind = UsdGenTypeId::Invalid;   // registered type with the record layout
    UsdGenToken schemaType;                       // "BasisCurves", "Mesh"
    UsdGenSpan<const UsdGenInputSlot> slots;
    UsdGenSpan<const UsdGenToken> attributes;     // scalar/token attributes copied as values
    UsdGenToken dynamicPrimvarPrefix;             // authored planes; empty = none
    /// Optional engine-side derivation run at compile.
    UsdGenRef<const UsdGenElementSet> (*materialize)(UsdGenInputRecord const &) = nullptr;
};

}  // namespace usdGen

#endif  // USDGEN_API_INPUT_DESC_H
