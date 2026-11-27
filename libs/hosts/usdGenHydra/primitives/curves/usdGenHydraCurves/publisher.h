// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curves' UsdGenPublisherDef (registries/publisher.h): builds a
// UsdGenCurvePublication (publication.h) from a curve terminal. Registered by
// usdGenRegister_usdGenHydraCurves (register.h).
#pragma once

#include "usdGenHydraCurves/export.h"
#include "usdGen/api/ext/publisher.h"

namespace usdGen {

class UsdGenRegistrar;

/// Registers the curve publisher for the type named UsdGenCurveSet; call
/// after that type is registered.
USDGEN_HYDRA_CURVES_API bool UsdGenRegisterCurvePublisher(UsdGenRegistrar &);

/// True when this adapter's publisher serves UsdGenCurveSet.
USDGEN_HYDRA_CURVES_API bool UsdGenCurvePublisherRegistered() noexcept;

/// Whether the publisher builds the renderer playback binding
/// (USDGEN_STORM_GPU_DEFORM).
USDGEN_HYDRA_CURVES_API bool UsdGenRbfAnimationEnabled();

/// The publisher's diff of two curve publications (a UsdGenCurveDirtyReport);
/// `prev` may be null.
USDGEN_HYDRA_CURVES_API UsdGenRef<const UsdGenPublication> UsdGenDiffCurvePublications(
    UsdGenPublication const *prev, UsdGenPublication const &next);

}  // namespace usdGen