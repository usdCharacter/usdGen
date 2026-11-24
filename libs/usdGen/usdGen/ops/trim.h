// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenTrimOp. Cuts strands where they cross blade ribbons standing on trim curves.
// Whole-strand styler (ops/wholeStrand.h): Capture computes the result over
// every strand; Evaluate applies usdGen:mask. TopologyEffect = None.
#ifndef USDGEN_OP_TRIM_H
#define USDGEN_OP_TRIM_H

#include "usdGen/ops/wholeStrand.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenTrimOp final : public UsdGenWholeStrandOp
{
public:
    UsdGenTrimOp() = default;
    ~UsdGenTrimOp() override = default;
    TfToken Type() const override { return TfToken("UsdGenTrim"); }
    TfSpan<const TfToken> ReferenceInputs() const override;

protected:
    TfTokenVector StructuralParameters() const override;
    std::vector<UsdGenStrandPlaneSpec> OutputPlaneSpecs() const override;
    bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                 std::vector<UsdGenStrandPlaneData> &planes,
                 UsdGenDiagnostics *diag) override;
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterTrimOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_TRIM_H
