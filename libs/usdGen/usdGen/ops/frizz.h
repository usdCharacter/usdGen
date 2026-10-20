// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenFrizzOp. Two-axis wave plus value-noise frizz in a transport
// frame, relative to a per-strand reference length.
//
// Parameters (no usdGen: prefix):
//   frizz:amplitude  peak offset as a fraction of the reference length (0)
//   frizz:wavesX / frizz:wavesZ  waves per reference length (0)
//   frizz:noise      noise weight 0..1 (0)
//   frizz:noiseSize  scene units per noise cell (0.1)
//   frizz:sharpness  0 = smooth noise, 1 = piecewise linear (0)
//   frizz:root / mid / midShift / tip  Bezier weight profile over the
//                    normalised arc length (0 / 0.5 / 0.5 / 1, all 0..1)
//   frizz:useClumpIds    clump members share seed and reference length
//   frizz:preserveKnots  multiply by the upstream braidKnotMask plane
// Reference length = upstream restLength (else the current length); the
// root never moves.
//
// Whole-strand styler (ops/wholeStrand.h): Capture computes the result over
// every strand; Evaluate applies usdGen:mask. TopologyEffect = None.
#ifndef USDGEN_OP_FRIZZ_H
#define USDGEN_OP_FRIZZ_H

#include "usdGen/ops/wholeStrand.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenFrizzOp final : public UsdGenWholeStrandOp
{
public:
    UsdGenFrizzOp() = default;
    ~UsdGenFrizzOp() override = default;
    TfToken Type() const override { return TfToken("UsdGenFrizz"); }

protected:
    TfTokenVector StructuralParameters() const override;
    std::vector<UsdGenStrandPlaneSpec> OutputPlaneSpecs() const override;
    bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                 std::vector<UsdGenStrandPlaneData> &planes,
                 UsdGenDiagnostics *diag) override;
};

}  // namespace usdGen

#endif  // USDGEN_OP_FRIZZ_H
