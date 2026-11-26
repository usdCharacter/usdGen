// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCuda export macro. Default visibility on ELF until step 12.
#ifndef USDGEN_CUDA_EXPORT_H
#define USDGEN_CUDA_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenCuda_EXPORTS)
#    define USDGEN_CUDA_API __declspec(dllexport)
#  else
#    define USDGEN_CUDA_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_CUDA_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_CUDA_EXPORT_H
