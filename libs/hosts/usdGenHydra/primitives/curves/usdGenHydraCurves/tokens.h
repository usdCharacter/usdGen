// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curves' Hydra adapter vocabulary: published primvar, material-purpose
// and fur-occlusion plane names (step 9.3: out of the engine tokens).
// Header-safe static tokens, built on first use.
#pragma once
#include "usdGenUsd/presentation.h"
#include "pxr/base/tf/token.h"

namespace usdGen {
namespace hydraCurvesTokens {

#define USDGEN_HYDRA_CURVES_TOKEN(name, text)                                          \
    inline constexpr char k_##name##_text[] = text;                                    \
    inline ::usdGenUsd::UsdGenTextToken<k_##name##_text> const s_##name{};                \
    inline PXR_NS::TfToken const &name() { return *s_##name; }

USDGEN_HYDRA_CURVES_TOKEN(allPurpose, "allPurpose")
USDGEN_HYDRA_CURVES_TOKEN(none, "none")
USDGEN_HYDRA_CURVES_TOKEN(displayColor, "displayColor")
USDGEN_HYDRA_CURVES_TOKEN(hairT, "hairT")
USDGEN_HYDRA_CURVES_TOKEN(hairId, "hairId")
USDGEN_HYDRA_CURVES_TOKEN(st, "st")
USDGEN_HYDRA_CURVES_TOKEN(bakeColor, "bakeColor")
USDGEN_HYDRA_CURVES_TOKEN(velocities, "velocities")
USDGEN_HYDRA_CURVES_TOKEN(furTauP, "furTauP")
USDGEN_HYDRA_CURVES_TOKEN(furTauN, "furTauN")
USDGEN_HYDRA_CURVES_TOKEN(furDensity, "furDensity")
USDGEN_HYDRA_CURVES_TOKEN(furCapAlbedo, "furCapAlbedo")

#undef USDGEN_HYDRA_CURVES_TOKEN

}  // namespace hydraCurvesTokens
}  // namespace usdGen
