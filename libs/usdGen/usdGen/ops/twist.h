// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenTwistOp. Rotates strands about the mean curve of their guide
// group or clump, by a per-point absolute angle from a root/mid/tip Bezier
// profile over the mean curve's normalised arc length.
//
// Parameters (no usdGen: prefix): twist:amount (multiplier, default 1),
// twist:root / twist:mid / twist:tip (degrees, default 0), twist:midShift
// (0.5, clamped 0.1..0.9), twist:noise (0..1), twist:useClumpIds (false),
// twist:resnapRoot (true). Gate: amount > 0 and any of root/mid/tip != 0.
//
// Upstream planes: clumpId_0 (group by clump when useClumpIds), primaryGuide
// (group by dominant guide otherwise, and for unclumped strands), mirrorSign
// (the group's first strand decides the rotation sense), strandSeed (noise
// key) and strandIndex (summation / representative order). A strand whose
// group has fewer than two strands, or that has no group, is not twisted.
//
// Whole-strand styler (ops/wholeStrand.h): Capture computes the result over
// every strand; Evaluate applies usdGen:mask. TopologyEffect = None.
#ifndef USDGEN_OP_TWIST_H
#define USDGEN_OP_TWIST_H

#include "usdGen/ops/wholeStrand.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenTwistOp final : public UsdGenWholeStrandOp
{
public:
    UsdGenTwistOp() = default;
    ~UsdGenTwistOp() override = default;
    TfToken Type() const override { return TfToken("UsdGenTwist"); }

protected:
    TfTokenVector StructuralParameters() const override;
    std::vector<UsdGenStrandPlaneSpec> OutputPlaneSpecs() const override;
    bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                 std::vector<UsdGenStrandPlaneData> &planes,
                 UsdGenDiagnostics *diag) override;
};

}  // namespace usdGen

#endif  // USDGEN_OP_TWIST_H
