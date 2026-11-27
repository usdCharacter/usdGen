// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd export macro (step 12.3d). Class-level tags export a class's
// vtable and typeinfo too. Annotation only until the visibility flip (12.3f).
#ifndef USDGENUSD_EXPORT_H
#define USDGENUSD_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenUsd_EXPORTS)
#    define USDGEN_USD_API __declspec(dllexport)
#  else
#    define USDGEN_USD_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_USD_API __attribute__((visibility("default")))
#endif

#endif  // USDGENUSD_EXPORT_H
