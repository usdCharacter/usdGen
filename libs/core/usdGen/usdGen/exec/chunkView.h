// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// What a worker hands Evaluate: one chunk of a node's set and its primary input.
// Note: the legacy curve types of the same names (usdGenCurves/types/
// curveBuffer.h) are renamed or deleted by 5.3a/5.4; until then no TU may
// include both headers.
#pragma once
#include "usdGen/api/elementSet.h"
#include <cstdint>

namespace usdGen {

struct UsdGenChunkExtent { float min[3], max[3]; UsdGenChunkRange range; };

struct UsdGenChunkView {
    UsdGenChunkRange const *range = nullptr;
    uint32_t const *offsets[kUsdGenMaxLevels + 1] = {};    // node's, pre-shifted to range->first[L-1]; null = uniform
    uint32_t const *inOffsets[kUsdGenMaxLevels + 1] = {};  // primary input's, same shift
    uint32_t inFirst[kUsdGenMaxLevels + 1] = {};           // primary input's first item per level
    uint32_t inStride[kUsdGenMaxLevels + 1] = {};          // primary input's stride
    void *const *out = nullptr;            // per node slot: chunk-local base (written, or the shared column)
    void const *const *in = nullptr;       // per primary-input slot: chunk-local base
    void const *const *in2 = nullptr;      // second ordered input, or null
    UsdGenSlot const *emitted = nullptr;   // node slots of the declared emitted channels (outF/outI order)
    UsdGenSlot const *emittedIn = nullptr; // input slot of the same name or kUsdGenNoSlot
    UsdGenSlot const *consumed = nullptr;  // input slots of the declared consumed channels (inF/inI)
    UsdGenSlot const *remapped = nullptr;  // remapsDeepestDynamic: deepest-level dynamic slots
    UsdGenChannelTable const *channels = nullptr, *inChannels = nullptr;
    uint16_t emittedCount = 0, consumedCount = 0, remappedCount = 0;
    void *pointScratch = nullptr;          // 24-byte double3 elements
    double *scalarScratch = nullptr;
    UsdGenChunkExtent *extentSlot = nullptr;
    float const *rampLut = nullptr;
};

struct UsdGenReferenceSet {
    UsdGenRef<const UsdGenElementSet> set; // root-local offsets are dynamic channels "__localX/Y/Z"
    uint64_t generation = 0;
};

}  // namespace usdGen
