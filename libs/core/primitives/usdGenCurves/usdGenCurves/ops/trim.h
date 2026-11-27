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


namespace usdGen {

namespace trimP {
/// Slots, in decl order.
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kAngle, kInvert, kDistance, kNormalSearch, kEnabled, kMask, kCount
};
enum Tok : uint16_t { kTokAngle, kTokInvert, kTokDistance, kTokNormal };
}  // namespace trimP

class UsdGenTrimOp final : public UsdGenWholeStrandOp
{
public:
    explicit UsdGenTrimOp(UsdGenOpRecord const &record) : UsdGenWholeStrandOp(record) {}
    ~UsdGenTrimOp() override = default;

protected:
    bool Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                 std::vector<UsdGenStrandPlaneData> &planes,
                 UsdGenDiagnostics *diag) override;
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterTrimOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_TRIM_H
