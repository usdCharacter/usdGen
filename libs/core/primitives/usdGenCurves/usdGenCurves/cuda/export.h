// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurvesCuda export macros (brief 3.3). USDGEN_CURVES_CUDA_API: used by other libraries.
// USDGEN_CURVES_CUDA_INTERNAL_API: driven only by tests; exported when
// USDGEN_EXPORT_INTERNALS is defined (step 12.3f makes it a PUBLIC define).
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

#if defined(USDGEN_EXPORT_INTERNALS)
#  define USDGEN_CURVES_CUDA_INTERNAL_API USDGEN_CURVES_CUDA_API
#else
#  define USDGEN_CURVES_CUDA_INTERNAL_API
#endif

#endif  // USDGEN_CURVES_CUDA_EXPORT_H
