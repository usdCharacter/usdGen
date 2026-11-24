// usdGen engine — diagnostics counters (03-execution-engine.md §9.2;
// 09-performance-and-benchmarks.md §6.1 — 03 §9.2 is normative).
//
// Enabled by USDGEN_DIAGNOSTICS (0|counters|trace; 10-build-dependencies-testing.md §3.5).
// One number source, two consumers: the groom-panel stack profiler and the CI gates
// (via UsdGenImaging_GetStatsJson() at M5).
#ifndef USDGEN_STATS_H
#define USDGEN_STATS_H

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"

#include <array>
#include <cstdint>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Last 120 commit durations, split by phase (ring buffer; SI-3 reads this).
struct UsdGenCommitTiming
{
    int64_t generation = -1;
    double routeMs = 0.0;
    double compileMs = 0.0;
    double captureMs = 0.0;
    double evaluateMs = 0.0;
    double interleaveMs = 0.0;
    double publishMs = 0.0;
    double totalMs = 0.0;
};

/// Artist-facing per-node stats: the stack profiler column
/// ("op03_clumpBig is 60 % of your frame").
struct UsdGenNodeStats
{
    TfToken  type;
    SdfPath  path;
    double   captureMs = 0.0, lastEvalMs = 0.0, meanEvalMs = 0.0;
    uint64_t curvesIn = 0, curvesOut = 0;
    uint64_t chunksDirty = 0, chunksTotal = 0;
    uint64_t bytesOwned = 0;
    uint64_t captureHits = 0, captureMisses = 0;
    uint32_t warnings = 0;   // e.g. "guide angle rejected 42% of candidates"
};

/// Plain values collected by their execution owner and published as immutable
/// snapshots; the fields themselves are not atomic. Imaging counters include
/// usdGenImaging: commits, publishedTiles, noticeEntries, supersessions
/// (06-imaging.md §9).
struct UsdGenStats
{
    uint64_t commits = 0;
    uint64_t cookedNodes = 0;
    uint64_t cookedChunks = 0;
    uint64_t capturedNodes = 0;
    uint64_t publishedTiles = 0;
    uint64_t interleavedBytes = 0;
    uint64_t noticeEntries = 0;
    uint64_t recompiles = 0;
    uint64_t evictions = 0;
    uint64_t executionCacheHits = 0;
    uint64_t executionCacheMisses = 0;
    uint64_t executionCacheAdmissions = 0;
    uint64_t executionCacheAdmissionFailures = 0;
    uint64_t executionCacheCoalesced = 0;
    uint64_t motionSamples = 0;
    uint64_t supersessions = 0;
    std::array<UsdGenCommitTiming, 120> ring;
    size_t ringHead = 0;
};

}  // namespace usdGen

#endif  // USDGEN_STATS_H
