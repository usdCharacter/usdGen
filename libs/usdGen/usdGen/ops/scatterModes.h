// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenScatter distribution modes beyond the per-face random path
// (private to the Scatter operator; tests include it for the internals).
//
//   usdGen:distribution = random | uvGrid | vertex | even | perGuide
//   usdGen:countMode    = perFace | surface
//   usdGen:proximity:mode = none | distance | coverage
//   rel usdGen:guides, rel usdGen:densityMap
//
// The non-random modes reproduce the reference root generator: one surface
// wide target `clamp(roundf(A * density), 1, 2e6)`, std::mt19937 draws keyed
// by usdGen:seed in a fixed order, guide-proximity calibration with the fixed
// seeds 90210 / 42, and density-map thinning keyed by the reference root hash.
// Roots are produced in GENERATION order (that order is the strand identity
// `strandIndex`), thinned, then Morton-sorted like the random path.
#ifndef USDGEN_OPS_SCATTER_MODES_H
#define USDGEN_OPS_SCATTER_MODES_H

#include "usdGen/curveBuffer.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace scatter {

enum class Distribution { Random, UvGrid, Vertex, Even, PerGuide };
enum class CountMode { PerFace, Surface };
enum class ProximityMode { None, Distance, Coverage };

/// The authored mode parameters (no `usdGen:` prefix).
struct ModeConfig
{
    Distribution distribution = Distribution::Random;
    CountMode countMode = CountMode::PerFace;
    ProximityMode proximity = ProximityMode::None;
    float proximityDistance = 0.01f;   // scene units
    float proximityCoverage = 1.0f;    // 0..1
    float density = 100.0f;            // roots per unit area
    bool flip = false;
    /// usdGen:subdivisionLevel. Above 0 on a Catmull-Clark surface, roots are
    /// still distributed over the cage, then moved with their frames onto the
    /// limit surface (any polygon topology; this is the patch isolation level).
    int subdivisionLevel = 0;

    /// True when the roots come from the generation-order generator below
    /// (every mode except the default random + perFace).
    bool GenerationPath() const
    {
        return distribution != Distribution::Random || countMode == CountMode::Surface;
    }
};

/// Parses and validates the mode parameters. False (with `error`) on an
/// unknown token or a non-finite value.
bool ReadModeConfig(UsdGenParamView const *params, ModeConfig *out, std::string *error);

/// Per-capture numbers of the generator, mirroring the reference log lines
/// (`base-setup`, `scatter-even`). For tests and diagnostics only.
struct ModeStats
{
    float areaMesh = 0.0f;       // A_mesh, bound surface rest area
    float areaDomain = 0.0f;     // A_dom (0 without a proximity domain)
    float areaEffective = 0.0f;  // A_eff
    uint32_t calibrationHits = 0;
    float cutoff2 = -1.0f;       // proximity cutoff^2 (-1 = off)
    bool proximity = false;
    uint32_t target = 0;
    float spacing = 0.0f;        // even: r0
    float finalRadius = 0.0f;    // even: radius after the last pass
    uint32_t candidates = 0;     // even
    uint32_t accepted = 0;       // even
    uint32_t firstPass = 0;      // even
    uint32_t relaxations = 0;    // even
    uint64_t relaxationDarts = 0;// even
    uint32_t barren = 0;         // even
    bool lastResort = false;     // even
    uint32_t placed = 0;         // roots generated (before density-map thinning)
    uint32_t emitted = 0;        // roots after thinning
};

/// The reference per-root existence hash: boost hash_combine of
/// (index, tri+1, vertex+1, bits(pos.xyz), bits(bary.xyz)) seeded with
/// 0x14650fb0739d0383, murmur fmix64, 24-bit float. `tri` is the fan
/// triangle index over the WHOLE mesh (OBJ face order), -1 for vertex roots.
float RootHash01(uint64_t index, int tri, int vertex, GfVec3f const &pos, GfVec3f const &bary);

/// Builds the generator's output into `buf` (roots, frames, Morton order,
/// planes strandIndex/strandSeed). `stats` is optional.
bool CaptureModes(UsdGenCaptureContext const &ctx, ModeConfig const &config,
                  UsdGenSurfaceDesc const &surface, UsdGenCurveBuffer *buf,
                  UsdGenDiagnostics *diag, ModeStats *stats = nullptr);

/// Density-map / guide-proximity thinning for the default random + perFace
/// roots. Inactive (Keep() always true) unless a density map is bound or a
/// guide-proximity mode is active with guides.
class RandomPathFilter
{
public:
    RandomPathFilter();
    ~RandomPathFilter();
    RandomPathFilter(RandomPathFilter const &) = delete;
    RandomPathFilter &operator=(RandomPathFilter const &) = delete;

    /// False (with a diagnostic) on malformed inputs.
    bool Init(UsdGenCaptureContext const &ctx, ModeConfig const &config,
              UsdGenSurfaceDesc const &surface, UsdGenDiagnostics *diag);
    bool Active() const;
    /// Keeps a root when it lies within the guide-proximity region and the
    /// density-map product m passes `m > 0 && (m >= 0.999 || h <= m)`, with h
    /// a curveId-keyed hash (stable under density edits, like the ids).
    bool Keep(GfVec3f const &pos, uint64_t curveId) const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// Inserts the uniform int planes strandIndex / strandSeed (sorted by name).
void SetStrandPlanes(UsdGenCurveBuffer *buf, VtIntArray index, VtIntArray seed);

}  // namespace scatter
}  // namespace usdGen

#endif  // USDGEN_OPS_SCATTER_MODES_H
