// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenVulkan export macro. Default visibility on ELF until step 12.
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

#endif  // USDGEN_VULKAN_EXPORT_H
