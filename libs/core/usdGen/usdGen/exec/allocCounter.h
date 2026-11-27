// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — heap-allocation counter (the warm-allocation gate's proof).
// On Windows each DLL links its own operator new, and the pxr DLLs allocate
// VtArray storage inside themselves, so a counting operator new in a test exe
// sees almost nothing. Starting a count there wraps the CRT heap imports
// (malloc, calloc, realloc, _aligned_malloc/_realloc) of every loaded module,
// usdGen.dll included. On ELF an exe-level operator new interposes on every
// module; it reports each call through UsdGenEngineAllocNote. Either way the
// count is process-wide while it runs, and pool growth counts too (pool.h).
// Counting is off by default and costs one relaxed load per allocation.
#pragma once

#include "usdGen/api/export.h"

#include <cstddef>
#include <cstdint>

namespace usdGen {

/// Start or stop counting. Starting resets the count and the trace.
USDGEN_CORE_API void UsdGenEngineAllocCounting(bool on);
/// Allocations counted since counting last started.
USDGEN_CORE_API uint64_t UsdGenEngineAllocCount();
/// A counting operator new (ELF: the exe's replacement) calls this per call;
/// on Windows the CRT import wrappers do.
USDGEN_CORE_API void UsdGenEngineAllocNote(std::size_t bytes);

/// Record a return-address stack for the first kUsdGenAllocTraceSlots counted
/// allocations (Windows; elsewhere only the immediate caller). No allocation.
/// The test prints them as module+offset (USDGEN_ALLOC_TRACE=1).
constexpr int kUsdGenAllocTraceSlots = 256;
constexpr int kUsdGenAllocTraceFrames = 16;
USDGEN_CORE_API void UsdGenEngineAllocTracing(bool on);
/// Frames of traced allocation `slot` into `frames`; returns the frame count
/// (0 past the recorded ones). `bytes` receives the request size.
USDGEN_CORE_API int UsdGenEngineAllocTrace(int slot, void **frames, int maxFrames,
                                           std::size_t *bytes);

}  // namespace usdGen
