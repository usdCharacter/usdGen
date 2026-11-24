// usdGen — TfDebug codes for tracing what the engine does per commit.
//
//   TF_DEBUG=USDGEN_COMMIT     one line per cook: why it ran, compile,
//                              cache, tiles rebuilt/carried, timings
//   TF_DEBUG=USDGEN_SCHEDULE   one line per node per cook: re-captured or
//                              reused, chunks evaluated, timings
//   TF_DEBUG=USDGEN_INGRESS    groom scene index: dirty notices received,
//                              which grooms re-capture, capture cost
//   TF_DEBUG=USDGEN_FUR        one line per fur optical-depth bake: grid,
//                              voxel size, CVs, work split
//
// Codes may be combined (TF_DEBUG="USDGEN_*"). The engine also emits
// TRACE_SCOPE events for the same stages: record them with
// `usdview --traceToFile trace.json --traceFormat chrome` or
// bin/trace_playback.ps1.
#ifndef USDGEN_DEBUG_CODES_H
#define USDGEN_DEBUG_CODES_H

#include "pxr/pxr.h"
#include "pxr/base/tf/debug.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEBUG_CODES(
    USDGEN_COMMIT,
    USDGEN_SCHEDULE,
    USDGEN_INGRESS,
    USDGEN_FUR
);

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_DEBUG_CODES_H
