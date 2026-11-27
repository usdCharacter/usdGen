// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Debug messages (TF_DEBUG=USDGEN_* keeps working):
//
//   Commit    one line per cook: why it ran, compile, cache, timings
//   Schedule  one line per node per cook: re-captured or reused, chunks
//   Ingress   scene index notices and description captures
//   Fur       one line per optical-depth bake
//
// The engine reads TF_DEBUG on the first query (TfDebug's syntax: words,
// a trailing '*' matches a prefix, a leading '-' turns codes off). Hosts
// that register the codes with TfDebug (usdGenUsd/debugCodes.h) mirror
// TfDebug's state into the engine once per commit.
#pragma once
#include "usdGen/api/export.h"

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define USDGEN_DEBUG_PRINTF_FORMAT(fmt, first) __attribute__((format(printf, fmt, first)))
#else
#  define USDGEN_DEBUG_PRINTF_FORMAT(fmt, first)
#endif

namespace usdGen {

enum class UsdGenDebugCode : uint8_t { Commit, Schedule, Ingress, Fur, Count };
USDGEN_CORE_API bool UsdGenDebugEnabled(UsdGenDebugCode c) noexcept;
USDGEN_CORE_API void UsdGenSetDebugEnabled(UsdGenDebugCode c, bool on) noexcept;
/// printf-style, to stdout (TfDebug's default output).
USDGEN_CORE_API void UsdGenDebugMsg(char const *fmt, ...) noexcept USDGEN_DEBUG_PRINTF_FORMAT(1, 2);

}  // namespace usdGen

/// Arguments are evaluated only when the code is on.
#define USDGEN_DEBUG_MSG(code, ...) \
    do { if (::usdGen::UsdGenDebugEnabled(::usdGen::UsdGenDebugCode::code)) ::usdGen::UsdGenDebugMsg(__VA_ARGS__); } while (0)
