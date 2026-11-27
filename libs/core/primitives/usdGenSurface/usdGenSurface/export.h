// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenSurface export macros (brief 3.3). USDGEN_SURFACE_API: used by other libraries.
// USDGEN_SURFACE_INTERNAL_API: driven only by tests; exported when
// USDGEN_EXPORT_INTERNALS is defined (step 12.3f makes it a PUBLIC define).
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

#if defined(USDGEN_EXPORT_INTERNALS)
#  define USDGEN_SURFACE_INTERNAL_API USDGEN_SURFACE_API
#else
#  define USDGEN_SURFACE_INTERNAL_API
#endif

#endif  // USDGEN_SURFACE_EXPORT_H
