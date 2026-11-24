// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Private, temporary (S4-1 to S4-8): a def for an op that still describes
// itself through UsdGenOp's static virtuals.
#pragma once
#include "usdGen/registries/opDef.h"   // types.h brings TfToken

#include <functional>
#include <memory>

namespace usdGen {

/// Probes one instance. Storage lives for the process; nullptr on failure.
UsdGenOpDef const *UsdGenMakeLegacyOpDef(TfToken const &type,
                                         std::function<std::unique_ptr<UsdGenOp>()> factory);
/// Frees a def RegisterOp rejected.
void UsdGenDiscardLegacyOpDef(UsdGenOpDef const *def);

}  // namespace usdGen
