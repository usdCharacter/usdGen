// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCuda export macros (brief 3.3). The library is built hidden; what
// another DSO reaches is tagged USDGEN_CUDA_API. Symbols only tests drive carry
// USDGEN_CUDA_INTERNAL_API: exported when USDGEN_EXPORT_INTERNALS is defined
// (a PUBLIC definition of every library when the option is ON).
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

// DeviceBuffer<T> is a template other modules also instantiate (their own element
// types), so its members cannot carry dll linkage: Windows exports only the
// explicit instantiations (USDGEN_CUDA_TEMPLATE_API); ELF exports through the class.
#if defined(_WIN32)
#  define USDGEN_CUDA_TEMPLATE_API USDGEN_CUDA_API
#  define USDGEN_CUDA_TEMPLATE_CLASS_API
#else
#  define USDGEN_CUDA_TEMPLATE_API
#  define USDGEN_CUDA_TEMPLATE_CLASS_API USDGEN_CUDA_API
#endif

#if defined(USDGEN_EXPORT_INTERNALS)
#  define USDGEN_CUDA_INTERNAL_API USDGEN_CUDA_API
#else
#  define USDGEN_CUDA_INTERNAL_API
#endif

#endif  // USDGEN_CUDA_EXPORT_H
