// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenBraidOp. Weaves the strands of each guide into a three-lane
// braid around the guide curve (reference behaviour of the imported groom's
// "braid" clump type).
//
// Members of guide g are the strands whose upstream `nearestGuide` plane is g
// (or, without that plane, whose root is nearest g's root), in strandIndex
// order. Per guide: the spine is the guide sampled uniformly in arc length,
// with transported frames; the members' roots are ranked along the principal
// axis of their spread (PCA in the guide's root frame) and cut into three
// lanes; each lane follows a phase-shifted cross-section curve
// x = f(phi) = 2.2 cos(phi) / (1.2 |cos(phi)| + 1), y = sin(2 phi) around the
// spine; a member keeps a concentric-disk slot inside its lane (pinched at lane
// crossings, bulged between them); fades, root/tip spread, a loose tail with
// cinch and relax, and the root/mid/tip profile shape the pull, which is an
// absolute lerp toward the woven target.
//
// Parameters (no usdGen: prefix here): braid:frequency (knots per reference
// length), braid:width (relative to the reference length), braid:flatness,
// braid:knotTightness, braid:fadeIn, braid:fadeOut, braid:rootSpread,
// braid:tipSpread, braid:tailLength, braid:tailRelax, braid:tailCinch,
// braid:referenceLength (0 = diagonal of the bound surface's rest bounding
// box, 1 without a surface), the profile braid:root/mid/midShift/tip/amount,
// braid:flyaway:fraction / braid:flyaway:amount, and braid:knotMask.
//
// Outputs: clumpId_0 (uniform int, 3 * guide + lane; -1 for flyaways and
// unbraided strands) and braidKnotMask (vertex float, 1 - knot(u) for braided
// non-flyaway strands when braid:knotMask is on, else 1).
//
// Whole-strand styler (ops/wholeStrand.h): Capture computes the result over
// every strand; Evaluate applies usdGen:mask. TopologyEffect = None.
#ifndef USDGEN_OP_BRAID_H
#define USDGEN_OP_BRAID_H

#include "usdGen/ops/wholeStrand.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenBraidOp final : public UsdGenWholeStrandOp
{
public:
    UsdGenBraidOp() = default;
    ~UsdGenBraidOp() override = default;
    TfToken Type() const override { return TfToken("UsdGenBraid"); }
    TfSpan<const TfToken> ReferenceInputs() const override;

protected:
    TfTokenVector StructuralParameters() const override;
    std::vector<UsdGenStrandPlaneSpec> OutputPlaneSpecs() const override;
    bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                 std::vector<UsdGenStrandPlaneData> &planes,
                 UsdGenDiagnostics *diag) override;
};

}  // namespace usdGen

#endif  // USDGEN_OP_BRAID_H
