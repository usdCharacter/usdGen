// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd — TfDebug codes for what the engine does per commit.
//
//   TF_DEBUG=USDGEN_COMMIT     one line per cook: why it ran, compile,
//                              cache, tiles rebuilt/carried, timings
//   TF_DEBUG=USDGEN_SCHEDULE   one line per node per cook: re-captured or
//                              reused, chunks evaluated, timings
//   TF_DEBUG=USDGEN_INGRESS    groom scene index: dirty notices received,
//                              which grooms re-capture, capture cost
//   TF_DEBUG=USDGEN_FUR        one line per fur optical-depth bake: grid,
//                              voxel size, items, work split
//
// Codes may be combined (TF_DEBUG="USDGEN_*"). The engine reads TF_DEBUG
// itself (usdGen/api/debug.h); SyncDebugCodes() mirrors TfDebug's state
// (TfDebug::Enable, usdview's debug menu) into it, once per commit. The
// engine also emits trace scopes for the same stages: record them with
// `usdview --traceToFile trace.json --traceFormat chrome` or
// bin/trace_playback.ps1.
#ifndef USDGENUSD_DEBUGCODES_H
#define USDGENUSD_DEBUGCODES_H

#include "pxr/pxr.h"
#include "usdGenUsd/export.h"
#include "pxr/base/tf/debug.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEBUG_CODES(
    USDGEN_COMMIT,
    USDGEN_SCHEDULE,
    USDGEN_INGRESS,
    USDGEN_FUR
);

PXR_NAMESPACE_CLOSE_SCOPE

namespace usdGenUsd {

/// Copies the four TfDebug codes into the engine's debug bits.
USDGEN_USD_API void SyncDebugCodes() noexcept;

}  // namespace usdGenUsd

#endif  // USDGENUSD_DEBUGCODES_H
