// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Compile checks and dependencies of connected parameters, registered as the
// evaluator's UsdGenParamEvaluatorDef hooks.
#pragma once
#include "usdGen/api/graphSpec.h"

#include <string>
#include <vector>

namespace usdGen {

/// Geometry, map and expression inputs an expression names exist and are unique.
bool UsdGenValidateExpressionInputs(UsdGenGraphSpec const &, std::vector<std::string> *errors);
/// One node's bindings resolve to an expression output of the destination's shape.
bool UsdGenValidateExpressionBindingSources(UsdGenGraphSpec const &, UsdGenOpNode const &,
                                            std::vector<std::string> *errors);
/// The keys an expression 'source' samples: targets, geometries, maps, per input.
void UsdGenExpressionDependencies(UsdGenGraphSpec const &, UsdGenNodeKey source, void *context,
                                  void (*add)(void *context, UsdGenNodeKey));

}  // namespace usdGen
