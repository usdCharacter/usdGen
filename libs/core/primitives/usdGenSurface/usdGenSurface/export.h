// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenSurface export macro. Default visibility on ELF until step 12.
#ifndef USDGEN_SURFACE_EXPORT_H
#define USDGEN_SURFACE_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenSurface_EXPORTS)
#    define USDGEN_SURFACE_API __declspec(dllexport)
#  else
#    define USDGEN_SURFACE_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_SURFACE_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_SURFACE_EXPORT_H
