// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenHydraCurves export macro (step 12.3d). Annotation only until the visibility flip (12.3f).
#ifndef USDGEN_HYDRA_CURVES_EXPORT_H
#define USDGEN_HYDRA_CURVES_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenHydraCurves_EXPORTS)
#    define USDGEN_HYDRA_CURVES_API __declspec(dllexport)
#  else
#    define USDGEN_HYDRA_CURVES_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_HYDRA_CURVES_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_HYDRA_CURVES_EXPORT_H
