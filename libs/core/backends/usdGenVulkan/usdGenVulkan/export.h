// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenVulkan export macros (brief 3.3). The library is built hidden; what
// another DSO reaches is tagged USDGEN_VULKAN_API. Symbols only tests drive carry
// USDGEN_VULKAN_INTERNAL_API: exported when USDGEN_EXPORT_INTERNALS is defined
// (a PUBLIC definition of every library when the option is ON).
#ifndef USDGEN_VULKAN_EXPORT_H
#define USDGEN_VULKAN_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenVulkan_EXPORTS)
#    define USDGEN_VULKAN_API __declspec(dllexport)
#  else
#    define USDGEN_VULKAN_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_VULKAN_API __attribute__((visibility("default")))
#endif

// A friend declaration that precedes the function's exported declaration must
// carry the same dll linkage on MSVC; ELF merges visibility and needs nothing.
#if defined(_WIN32)
#  define USDGEN_VULKAN_FRIEND_API USDGEN_VULKAN_API
#else
#  define USDGEN_VULKAN_FRIEND_API
#endif

#if defined(USDGEN_EXPORT_INTERNALS)
#  define USDGEN_VULKAN_INTERNAL_API USDGEN_VULKAN_API
#else
#  define USDGEN_VULKAN_INTERNAL_API
#endif

#endif  // USDGEN_VULKAN_EXPORT_H
