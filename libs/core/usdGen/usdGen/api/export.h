// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen export macros (brief 3.3). USDGEN_API marks the installed surface
// (api/ and api/ext/); USDGEN_INTERNAL_API marks private classes and
// functions that tests drive, exported only with USDGEN_EXPORT_INTERNALS.
#ifndef USDGEN_EXPORT_H
#define USDGEN_EXPORT_H

#if defined(_WIN32)
#  define USDGEN_DLL_EXPORT __declspec(dllexport)
#  define USDGEN_DLL_IMPORT __declspec(dllimport)
#else
#  define USDGEN_DLL_EXPORT __attribute__((visibility("default")))
#  define USDGEN_DLL_IMPORT __attribute__((visibility("default")))
#endif

#if defined(usdGen_EXPORTS)
#  define USDGEN_API USDGEN_DLL_EXPORT
#else
#  define USDGEN_API USDGEN_DLL_IMPORT
#endif

// For a function whose address must stay a constant expression in importers
// (MSVC: the address of a dllimport function is not one). Importers call it
// through the import library's thunk.
#if defined(usdGen_EXPORTS)
#  define USDGEN_API_NO_IMPORT USDGEN_DLL_EXPORT
#else
#  define USDGEN_API_NO_IMPORT
#endif

// A PUBLIC compile definition of usdGen when the option is ON (12.3f), so
// importers agree.
#if defined(USDGEN_EXPORT_INTERNALS)
#  define USDGEN_INTERNAL_API USDGEN_API
#else
#  define USDGEN_INTERNAL_API
#endif

#endif  // USDGEN_EXPORT_H
