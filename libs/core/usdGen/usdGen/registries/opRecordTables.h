// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Private: the tables a UsdGenOpRecord owns, built once by RegisterOp.
#pragma once
#include "usdGen/api/ext/opRegistry.h"

#include <utility>
#include <vector>

namespace usdGen {

struct UsdGenOpRecord::Tables {
    std::vector<UsdGenToken> paramNames;                 // per slot, no "usdGen:"
    std::vector<std::pair<UsdGenToken, UsdGenParamSlot>> hash;   // open addressing; both spellings
    size_t mask = 0;
    std::vector<std::vector<UsdGenToken>> enumerants;    // per slot
    std::vector<std::pair<UsdGenToken, uint32_t>> routing;
    std::vector<UsdGenParamSlot> digest;
    std::vector<UsdGenToken> referenceInputs;
    std::vector<UsdGenChannelDesc> inputPlanes, outputPlanes;   // layouts, registry order
    std::vector<UsdGenToken> mapRelationships;
    std::vector<UsdGenToken> tokens;
    UsdGenToken outputType;                              // empty: the default primitive
    std::vector<UsdGenToken> inputTypes;                 // an empty entry: the default
};

}  // namespace usdGen
