// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenPomade export macro (hierarchical groom authoring model, plan/17).
#ifndef USDGEN_POMADE_API_H
#define USDGEN_POMADE_API_H

#ifndef USDGENPOMADE_API
#  if defined(_WIN32)
#    if defined(usdGenPomade_EXPORTS)
#      define USDGENPOMADE_API __declspec(dllexport)
#    else
#      define USDGENPOMADE_API __declspec(dllimport)
#    endif
#  else
#    define USDGENPOMADE_API __attribute__((visibility("default")))
#  endif
#endif

#endif // USDGEN_POMADE_API_H
