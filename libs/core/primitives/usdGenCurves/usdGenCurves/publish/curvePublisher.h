// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curve primitive's UsdGenPublisherDef (registries/publisher.h): builds
// a UsdGenCurvePublication (curvePublication.h) from a curve terminal.
#pragma once

#include "usdGen/api/export.h"
#include "usdGen/registries/publisher.h"

namespace usdGen {

class UsdGenRegistrar;

/// Registers the curve publisher for UsdGenCurveSetType(); call after that
/// type is registered.
USDGEN_CORE_API bool UsdGenRegisterCurvePublisher(UsdGenRegistrar &);

/// The publisher's diff of two curve publications (a UsdGenCurveDirtyReport);
/// `prev` may be null.
USDGEN_CORE_API UsdGenRef<const UsdGenPublication> UsdGenDiffCurvePublications(
    UsdGenPublication const *prev, UsdGenPublication const &next);

}  // namespace usdGen