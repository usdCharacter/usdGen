// usdGen engine — shared type aliases, enums, constants and tile arithmetic.
//
// Interface source of truth: docs/m1/interfaces.md §4 (engine API).
// Plan sources: 03-execution-engine.md §1.2/§1.4, 02-schema.md §2.3/§2.3.1,
// ADR §9 R1/R9/R12/R14/R21.
#ifndef USDGEN_TYPES_H
#define USDGEN_TYPES_H

#include "pxr/pxr.h"
#include "pxr/base/gf/range3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/tf/span.h"

#include <array>
#include <cmath>
#include <cstdint>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

// ---------------------------------------------------------------------------
// Ids (03-execution-engine.md §1.2)
// ---------------------------------------------------------------------------
using UsdGenNodeId    = uint32_t;   // dense, assigned by the compiler in topo order
using UsdGenChunkId   = uint32_t;   // dense within a description
using UsdGenTileId    = uint16_t;
using UsdGenSurfaceId = uint16_t;   // index into UsdGenGraphDesc::surfaces

constexpr UsdGenNodeId kUsdGenInvalidNode = ~0u;

/// CPU-lane ceiling on CVs per generated or resampled strand (Grow segments,
/// GuideInterpolate/CardSource cvCount, Resample cvCount/spacing). The CUDA
/// and Vulkan lanes keep their own, smaller kernel limits and refuse a graph
/// above them at admission.
constexpr uint32_t kUsdGenMaxCvsPerCurve = 4096u;

/// 128-bit digest; used for the Merkle structural digest and the capture epoch.
using UsdGenEpoch = std::array<uint64_t, 2>;

inline bool operator==(UsdGenEpoch const &a, UsdGenEpoch const &b)
{
    // Explicit element compare: a bare `a == b` here would resolve to this
    // free operator (it shadows std::array's member-adjacent ==), recursing
    // forever. Pinned so the T0 graph tests do not hang.
    return a[0] == b[0] && a[1] == b[1];
}
inline bool operator!=(UsdGenEpoch const &a, UsdGenEpoch const &b)
{
    return !(a == b);
}

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------


/// R14 static topology effect. `CurveCount` is the plan's `MayChangeCurveCount`.
enum class UsdGenTopoFx : uint8_t { None, CurveCount, CvCount, Both };

/// I3 reference lane: un-chunked buffers evaluated to completion before consumers.
enum class UsdGenRole : uint8_t { Curves, Reference };


/// UsdGenDirtyBits — unscoped integer constants (ADR §9 R3: never written
/// `UsdGenDirtyBits::X`). Mapping from 02-schema.md §2.1 dirty classes:
///   structural          -> UsdGenDirtyStructural (graph digest term)
///   structural(topology) -> UsdGenDirtyTopology  (not a digest term)
///   capture             -> UsdGenDirtyCapture
///   value               -> UsdGenDirtyParameter
///   toggle              -> UsdGenDirtyParameter (topology-preserving types)
///                          / UsdGenDirtyTopology (generators, Resample, Length)
/// Publication/cosmetic classes never reach the engine (02-schema.md §2.1).
constexpr uint32_t UsdGenDirtyNone          = 0u;
constexpr uint32_t UsdGenDirtyParameter     = 1u << 0;  // re-evaluate node chunks + descendants
constexpr uint32_t UsdGenDirtyCapture       = 1u << 1;  // bump capture epoch, re-capture node + descendants
constexpr uint32_t UsdGenDirtyTopology      = 1u << 2;  // curve/CV/published-prim-count change; re-allocate, no digest
constexpr uint32_t UsdGenDirtySurfacePoints = 1u << 3;  // deformed points of a bound surface
constexpr uint32_t UsdGenDirtySurfaceXform  = 1u << 4;
constexpr uint32_t UsdGenDirtySurfaceTopo   = 1u << 5;  // surface resync
constexpr uint32_t UsdGenDirtyMap           = 1u << 6;  // asset path / textureGeneration bump
constexpr uint32_t UsdGenDirtyStructural    = 1u << 7;  // recompile (Merkle digest term)
constexpr uint32_t UsdGenDirtyLiveOverride  = 1u << 8;  // brush stroke, chunk-scoped

/// Commit triggers (ADR §4.3 as amended by ADR §9 R32; 06-imaging.md §3.9).
enum class UsdGenCommitReason : uint8_t {
    SetTime,         // (a) UsdGenImaging_SetTime() / app driver currentFrameChanged
    LiveOverride,    // (a) explicit Commit() after a live-override change
    SceneFrameDirty, // (b) / sceneGlobals/currentFrame dirty, no app driver attached
    NoticeBatchEnd,  // (c) end of the _PrimsDirtied batch that carried a routed dirty
};

/// Session context (02-schema.md §2.3.2; R13 — the two are exclusive).
enum class UsdGenContext : uint8_t { Interactive, Render };

// ---------------------------------------------------------------------------
// Chunk / tile constants (S23, 02 §2.3, ADR §9 R21)
// ---------------------------------------------------------------------------
constexpr int kUsdGenMinChunkSize      = 128;   // USDGEN_CHUNK_SIZE clamp
constexpr int kUsdGenDefaultChunkSize  = 512;
constexpr int kUsdGenMaxChunkSize      = 1024;
constexpr int kUsdGenTileTargetDefault = 64;    // uniform int usdGen:tileTarget
constexpr int kUsdGenTileMin           = 32;
constexpr int kUsdGenTileMax           = 256;

/// Clamp an env-supplied chunk size into [128, 1024]; 0/negative -> 512.
inline int ClampChunkSize(int chunkSize)
{
    int c = (chunkSize <= 0) ? kUsdGenDefaultChunkSize : chunkSize;
    return std::max(kUsdGenMinChunkSize, std::min(kUsdGenMaxChunkSize, c));
}

// ---------------------------------------------------------------------------
// Tile arithmetic (ADR §9 R21 verbatim; 03-execution-engine.md §1.4)
//
//   chunkSize     = USDGEN_CHUNK_SIZE                      # default 512, clamped [128,1024]
//   nChunks       = ceil(maxCurves / chunkSize)
//   chunksPerTile = max(1, ceil(nChunks / tileTarget))      # tileTarget default 64
//   nTiles        = min(nChunks, clamp(ceil(nChunks / chunksPerTile), 32, 256))
//
// Worked example (frozen in C2): 100 000 curves, chunkSize 512, tileTarget 64
//   => nChunks = 196, chunksPerTile = 4, nTiles = 49, ~2 048 curves/tile.
// ---------------------------------------------------------------------------
inline int ComputeNumChunks(int maxCurves, int chunkSize = kUsdGenDefaultChunkSize)
{
    const int cs = ClampChunkSize(chunkSize);
    if (maxCurves <= 0) return 0;
    return (maxCurves + cs - 1) / cs;
}

inline int ComputeChunksPerTile(int nChunks, int tileTarget = kUsdGenTileTargetDefault)
{
    if (nChunks <= 0) return 0;
    const int t = tileTarget < 1 ? 1 : tileTarget;
    return std::max(1, (nChunks + t - 1) / t);
}

inline int ComputeNumTiles(int nChunks, int tileTarget = kUsdGenTileTargetDefault)
{
    if (nChunks <= 0) return 0;
    const int cpt = ComputeChunksPerTile(nChunks, tileTarget);
    int n = (nChunks + cpt - 1) / cpt;
    n = std::max(kUsdGenTileMin, std::min(kUsdGenTileMax, n));
    return std::min(nChunks, n);
}


}  // namespace usdGen

#endif  // USDGEN_TYPES_H
