// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurvesCuda export macro. Default visibility on ELF until step 12.
#ifndef USDGEN_CURVES_CUDA_EXPORT_H
#define USDGEN_CURVES_CUDA_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenCurvesCuda_EXPORTS)
#    define USDGEN_CURVES_CUDA_API __declspec(dllexport)
#  else
#    define USDGEN_CURVES_CUDA_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_CURVES_CUDA_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_CURVES_CUDA_EXPORT_H
