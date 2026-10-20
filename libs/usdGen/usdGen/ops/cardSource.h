// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenCardSourceOp. Grows strands on hair cards: every card
// (a spine curve with card parameters, usdGen:cards) is meshed into a
// rows x cols surface grid (ops/cardGeometry.h), and the strands of its
// texture tile (a strand library authored in tile space, usdGen:cardStrands)
// are mapped onto that grid, rooted on the scalp surface.
//
// Reference inputs (told apart by their planes, not by order):
//   * the card spine set: one linear curve per card, uniform planes carrying
//     the card parameters (`cardType` marks the set);
//   * the tile strand library: tile-space strands with uniform int planes
//     `uvId`, `tile` (the tile entry), `index` (local strand index inside the
//     entry) and `tileStrandCount` (entry size).
// The plane contract is documented in docs/groom-import.md.
//
// Per card strip whose (layer, uvId, tile) passes the node's filters, every
// library strand of the strip's tile entry with 2+ points grows one strand
// with id SplitMix64(SplitMix64(cardId) ^ j ^ ((strip + 1) << 40)) | 1, kept
// when U(SplitMix64(id ^ 0x5a17c4)) < amount * min(1, maxStrands / total).
// Its points are mapped by the card surface (tile x -> across, y -> along,
// z -> depth x (1 - rootFlatten) row widths), the root snaps to the closest
// scalp point (the strand's base follows with a smooth falloff), and the
// strand is resampled to usdGen:cvCount CVs uniform in arc length.
// Outputs curveId = strand id, rootPrim/rootUV/root frame at the snapped root,
// and uniform int planes strandIndex (generation order), strandSeed
// (SplitMix64(id) & 0x7fffffff) and cardIndex (spine curve). Curves are
// Morton-ordered by root. Generator: TopologyEffect = CurveCount.
#ifndef USDGEN_OP_CARDSOURCE_H
#define USDGEN_OP_CARDSOURCE_H

#include "usdGen/op.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenCardSourceOp final : public UsdGenOp
{
public:
    UsdGenCardSourceOp() = default;
    ~UsdGenCardSourceOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenCardSource"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CurveCount; }
    UsdGenRole Role() const override { return UsdGenRole::Curves; }
    bool IsGenerator() const override { return true; }
    size_t GeometryInputArity() const override { return 0; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;
    TfSpan<const TfToken> ReferenceInputs() const override;
    TfSpan<const TfToken> OutputPrimvars() const override;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx, UsdGenCurveBuffer const &upstream,
                 UsdGenCapture *out, UsdGenDiagnostics *diag) override;
    /// Identity: Capture fills the whole node buffer.
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView *) const override {}
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override { return 0; }
};

}  // namespace usdGen

#endif  // USDGEN_OP_CARDSOURCE_H
