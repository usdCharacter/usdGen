// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenTieClumpOp. Pulls the strands of tied guides toward their
// hair-tie positions (the imported groom's tie-clump stage).
//
// Inputs: `rel usdGen:guides` (the guide curve set, reference slot
// "guides"); the strand's guide is its upstream `primaryGuide` plane.
// Ties are attributes on the op prim: `point3f[] usdGen:tie:positions` (one
// per tie), `int[] usdGen:tie:entryCounts` (entries per tie) and, per entry,
// `int[] usdGen:tie:guideIndices` (curve index in the guide set) and
// `int[] usdGen:tie:vertexIndices` (guide point index). Controls
// `float usdGen:tie:amount = 1` and `float usdGen:tie:spread = 0.25`
// (both clamped to [0,1]); the stage runs when amount > 1e-4 and any guide
// has a tie.
//
// Tie table: each entry adds {position, tieArc} to its guide's list (tie
// order, then entry order), tieArc = guide polyline length from point 0 to
// point clamp(vertexIndex, 1, nv-1). Per strand of a tied guide (root never
// moves), L = current length, for each tie with tieArc < L0 (L0 = the
// strand length on entry), with m = max(1, n-1):
//   inner = max(1e-4, 0.04 L, 1.5 L/m); outer = max(0.4 L, 8 L/m);
//   W = max(inner, outer); s* = arc length of the strand point closest to
//   the tie (segment clamp-projection, first minimum);
//   x = clamp01(|s_k - s*| / (W spread + (1 - spread) inner));
//   f = 1 - smoothstep(x); w = clamp01((1-spread) f + spread sqrt(f)) amount;
//   P_k = lerp(P_k, tie, w) for k >= 1 when w > 1e-6;
// then the arc lengths and L are recomputed before the next tie.
// Strands whose guide has no tie are untouched. Deterministic (no
// randomness, per-strand independent).
//
// Whole-strand styler (ops/wholeStrand.h): Capture computes the result over
// every strand; Evaluate applies usdGen:mask. TopologyEffect = None.
#ifndef USDGEN_OP_TIECLUMP_H
#define USDGEN_OP_TIECLUMP_H

#include "usdGen/ops/wholeStrand.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenTieClumpOp final : public UsdGenWholeStrandOp
{
public:
    UsdGenTieClumpOp() = default;
    ~UsdGenTieClumpOp() override = default;
    TfToken Type() const override { return TfToken("UsdGenTieClump"); }
    TfSpan<const TfToken> ReferenceInputs() const override;

protected:
    TfTokenVector StructuralParameters() const override;
    std::vector<UsdGenStrandPlaneSpec> OutputPlaneSpecs() const override;
    bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                 std::vector<UsdGenStrandPlaneData> &planes,
                 UsdGenDiagnostics *diag) override;
};

}  // namespace usdGen

#endif  // USDGEN_OP_TIECLUMP_H
