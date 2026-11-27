// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenHydraCurves: the curves' Hydra adapter. Registers the publisher for
// the type named UsdGenCurveSet. Call before the first compile; idempotent and
// lock-free (an atomic state word, no once_flag).
#pragma once
#include "usdGenHydraCurves/export.h"
#include <string>
namespace usdGen { class UsdGenRegistrar; }
namespace usdGenHydra {
/// False with a reason when the registries are frozen or the type is missing.
USDGEN_HYDRA_CURVES_API bool UsdGenHydraCurvesEnsureRegistered(std::string *error = nullptr);
}  // namespace usdGenHydra
extern "C" USDGEN_HYDRA_CURVES_API void usdGenRegister_usdGenHydraCurves(usdGen::UsdGenRegistrar &);
