// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Private: the tables a UsdGenOpRecord owns, built once by RegisterOp.
#pragma once
#include "usdGen/registries/opRegistry.h"

#include <utility>
#include <vector>

namespace usdGen {

struct UsdGenOpRecord::Tables {
    std::vector<TfToken> paramNames;                 // per slot, no "usdGen:"
    std::vector<std::pair<TfToken, UsdGenParamSlot>> hash;   // open addressing; both spellings
    size_t mask = 0;
    std::vector<std::vector<TfToken>> enumerants;    // per slot
    std::vector<std::pair<TfToken, uint32_t>> routing;
    std::vector<UsdGenParamSlot> digest;
    std::vector<TfToken> referenceInputs;
    std::vector<UsdGenPlane> inputPlanes, outputPlanes;
    std::vector<TfToken> mapRelationships;
    std::vector<TfToken> tokens;
    TfToken outputType;                              // empty: the default primitive
    std::vector<TfToken> inputTypes;                 // an empty entry: the default
};

}  // namespace usdGen
