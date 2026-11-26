// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenExpr export macro. Default visibility on ELF until step 12.
#ifndef USDGEN_EXPR_EXPORT_H
#define USDGEN_EXPR_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenExpr_EXPORTS)
#    define USDGEN_EXPR_API __declspec(dllexport)
#  else
#    define USDGEN_EXPR_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_EXPR_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_EXPR_EXPORT_H
