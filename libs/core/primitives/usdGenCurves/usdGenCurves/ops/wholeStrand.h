// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — base class for "whole-strand" stylers: operators whose result for
// one strand depends on other strands, on reference curve sets, or on a
// whole-strand computation (arc-length resampling, frames, cutting), so they
// cannot be expressed as an independent per-chunk Evaluate.
//
// The contract mirrors UsdGenClump (03 §8.1, I3):
//   * Capture copies the upstream points, calls Compute() over ALL strands,
//     and keeps the resulting positions plus any output planes.
//   * The capture is re-run whenever the upstream topology OR values move
//     (ValidForTopology checks valueVersion), and whenever a structural
//     parameter, the seed, the bound surface, a reference set or a map
//     changes (CaptureDigest).
//   * Evaluate only applies usdGen:mask (lerp input -> captured result per
//     CV) and copies the captured output planes into the chunk's slots.
// TopologyEffect is None: a whole-strand styler keeps every curve and its CV
// count (operators that resample keep the count and move points).
#ifndef USDGEN_OPS_WHOLE_STRAND_H
#define USDGEN_OPS_WHOLE_STRAND_H

#include "usdGen/ops/recordedOp.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/api/token.h"

#include <cstdint>
#include <memory>
#include <vector>


namespace usdGen {

/// One output plane a whole-strand styler writes (built from its record).
struct UsdGenStrandPlaneSpec
{
    UsdGenToken name;
    bool cv = false;      // true: vertex (per CV); false: uniform (per curve)
    bool isInt = false;   // int or float payload, arity 1
};

/// Output plane payload, parallel to OutputPlaneSpecs(): `i` or `f` sized
/// totalCurves (uniform) or totalCvs (vertex).
struct UsdGenStrandPlaneData
{
    std::vector<int32_t> i;
    std::vector<float> f;
};

/// The strands handed to Compute(): every upstream curve, points copied into
/// one mutable xyz array. Compute edits `points` in place.
struct UsdGenStrandSet
{
    UsdGenCurveSetView const *upstream = nullptr;
    std::vector<uint32_t> spans;          // [spans[c], spans[c+1]) CV range of curve c
    std::vector<float> points;            // 3 * totalCvs, starts as the input

    uint32_t Count() const { return spans.empty() ? 0u : uint32_t(spans.size() - 1); }
    size_t First(uint32_t c) const { return spans[c]; }
    size_t Size(uint32_t c) const { return size_t(spans[c + 1] - spans[c]); }
    float *P(uint32_t c) { return points.data() + 3 * size_t(spans[c]); }
    float const *P(uint32_t c) const { return points.data() + 3 * size_t(spans[c]); }

    /// Upstream per-curve int plane (arity 1), or null when absent.
    int const *CurveInt(UsdGenToken const &name) const;
    /// Upstream per-curve float plane (arity 1), or null when absent.
    float const *CurveFloat(UsdGenToken const &name) const;
    /// Upstream per-CV float plane (arity 1), or null when absent.
    float const *CvFloat(UsdGenToken const &name) const;
    /// Per-strand random key: the upstream `strandSeed` plane when present,
    /// else Index(c).
    uint32_t Seed(uint32_t c) const;
    /// Generation-order index of the strand: the upstream `strandIndex`
    /// plane when present, else the curve index. Operators whose result
    /// depends on strand ORDER (flyaway hashes, first-pick clump seeds) sort
    /// or key by this, not by the (Morton-sorted) curve index.
    uint32_t Index(uint32_t c) const;
};

/// The def of a subclass declares, in order: input, surface, seed, its
/// structural parameters, then enabled and mask (Value class), and its output
/// planes (UsdGenOpDef::outputPlanes). TopologyEffect stays None.
class UsdGenWholeStrandOp : public UsdGenOp
{
public:
    explicit UsdGenWholeStrandOp(UsdGenOpRecord const &record);
    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const final;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenOpInputs const &in,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) final;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenChunkView const &chunk) const final;
    std::unique_ptr<UsdGenCapture> CreateCapture() const final;
    uint32_t PlanesTouched() const final { return kPlanePoints; }

protected:
    /// Moves `strands.points` and fills `planes` (pre-sized to the specs,
    /// default -1 for ints and 0 for floats). Return false (with a
    /// diagnostic) to fail the capture; the node then passes its input
    /// through.
    virtual bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                         std::vector<UsdGenStrandPlaneData> &planes,
                         UsdGenDiagnostics *diag) = 0;
    /// Short type name for diagnostics ("UsdGenTwist").
    std::string Label() const { return Record().Name().GetString(); }

private:
    UsdGenParamSlot _mask = kUsdGenNoParam;
    std::vector<UsdGenStrandPlaneSpec> _specs;   // the record's output planes, in slot order
};

}  // namespace usdGen

#endif  // USDGEN_OPS_WHOLE_STRAND_H
