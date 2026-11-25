// Stable capture-order tile ranges for compacted CUDA curve geometry.
#ifndef USDGEN_GPU_CURVE_TILES_H
#define USDGEN_GPU_CURVE_TILES_H

#include "deviceBuffer.h"

#include <cstddef>
#include <cstdint>

namespace usdGen::gpu {

struct CurveTileOptions {
    uint32_t chunkSize = 512;
    uint32_t tileTarget = 64;
};

// All ranges are local to the compacted survivor geometry.  `tile` is the
// stable capture-order tile ordinal; a range may be empty after culling.
struct CurveTileSpan {
    uint32_t tile = 0;
    uint32_t firstCurve = 0;
    uint32_t curveCount = 0;
    uint32_t firstPoint = 0;
    uint32_t pointCount = 0;
};

// The default mode consumes a strictly sorted survivor subset and derives tile
// boundaries with stable-ID lower bounds.  IdentityCaptureOrder is for
// authoritative generators whose capture order is meaningful (for example,
// Scatter's Morton order): the survivor list must have the same cardinality
// and contain the capture ID at the same index, so tile boundaries can be
// emitted directly without reordering the generated curves.
enum class CurveTileOrder : uint8_t {
    SortedSurvivorSubset = 0,
    IdentityCaptureOrder = 1,
    CaptureOrderSurvivorSubset = 2,
};

struct CurveTileInput {
    DeviceView<const uint64_t> captureStableIds;
    DeviceView<const uint64_t> survivorStableIds;
    DeviceView<const uint32_t> survivorCurveOffsets;
    size_t captureCurveCount = 0;
    size_t survivorCurveCount = 0;
    size_t survivorPointCount = 0;
    CurveTileOrder order = CurveTileOrder::SortedSurvivorSubset;
    // Required only for CaptureOrderSurvivorSubset: sorted unique IDs and
    // their original capture ordinals. Both borrowed domains must outlive
    // the caller's native completion proof.
    DeviceView<const uint64_t> sortedCaptureStableIds{};
    DeviceView<const uint32_t> sortedCaptureOrdinals{};
};

struct CurveTileRequirements {
    size_t chunkCount = 0;
    size_t tileCount = 0;
    size_t chunksPerTile = 0;
    size_t curvesPerTile = 0;
    uint32_t chunkSize = 0;
    uint32_t tileTarget = 0;
    int deviceIndex = -1;
    cudaStream_t stream = nullptr;
};

struct CurveTileOutput {
    DeviceView<CurveTileSpan> spans; // capacity >= requirements.tileCount
    DeviceView<uint32_t> status;     // one scalar: 0 valid, 1 invalid input
};

// Setup-only scalar arithmetic.  Zero selects the 512/64 defaults; other
// values clamp chunkSize to [128,1024] and tileTarget to [32,256].  It binds
// the current device and (when supplied) stream, and never reads device
// geometry or allocates.
cudaError_t GetCurveTileRequirements(CurveTileOptions options,
    size_t captureCurveCount, size_t survivorCurveCount,
    size_t survivorPointCount, CurveTileRequirements* result,
    cudaStream_t stream = nullptr);

// Enqueue-only borrowed-device-view builder.  In SortedSurvivorSubset mode,
// capture and survivor IDs must be strictly increasing and survivors must be
// an ordered subset of capture IDs.  In IdentityCaptureOrder mode, both lists
// must have equal cardinality and survivorStableIds[i] must equal
// captureStableIds[i]; capture IDs are required to have been uniqueness-
// validated by the authoritative capture producer (the unsorted identity
// path intentionally has no device sort/hash oracle).
// In CaptureOrderSurvivorSubset mode capture IDs may be unsorted; the supplied
// sorted lookup is validated as an exact permutation of capture IDs. Survivor
// IDs must form a strictly increasing subsequence of original capture ordinals.
// This preserves frozen capture-tile membership after generated curves cull.
// Survivor offsets must
// be canonical nonempty-curve offsets.  Invalid device data sets status=1 and
// leaves spans untouched; invalid mode or identity cardinality is rejected at
// enqueue time.  Tile membership is frozen by capture ordinal: every emitted
// tile initially owns at least one capture chunk, and empty post-cull tiles
// are emitted deliberately.
// No allocation, host wait/readback, locks, or geometry/index/draw-count
// publication occurs here.  Prepare requirements before CUDA graph capture.
cudaError_t BuildCurveTiles(CurveTileInput input,
    CurveTileRequirements const& requirements, CurveTileOutput output,
    cudaStream_t stream);

} // namespace usdGen::gpu

#endif // USDGEN_GPU_CURVE_TILES_H
