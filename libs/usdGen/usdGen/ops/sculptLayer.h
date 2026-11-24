// usdGen — UsdGenSculptLayerOp. 02-schema.md §2.10.
// Hand-authored per-CV deltas keyed by stable curveId, applied at a layer
// weight in the rest root frame (or object space):
//   P[j] += frame(deltas[first + j]) * weight * mask,
// where frame(d) = T*dx + B*dy + N*dz in space = rootFrame and d as-is in
// space = object. curveIds is sorted strictly ascending; cvOffsets has
// curveIds.size() + 1 entries with cvOffsets[0] == 0 and the last entry ==
// deltas.size(). Entries whose id is absent upstream are ignored (kept, never
// dropped); strands without an entry pass through bit-for-bit. rootPrims and
// rootUVs are rebase metadata the kernel validates but never reads;
// lockedCurves names Freeze-brush ids for downstream stylers, so this layer
// still sculpts them. Deltas and weight are value-class (read live in
// Evaluate); ids, offsets, epoch, locks and rebase metadata are
// capture-class (snapshotted in Capture). TopologyEffect = None.
#ifndef USDGEN_OP_SCULPT_LAYER_H
#define USDGEN_OP_SCULPT_LAYER_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenSculptLayerOp final : public UsdGenOp
{
public:
    UsdGenSculptLayerOp() = default;
    ~UsdGenSculptLayerOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenSculptLayer"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }

    TfSpan<const TfToken> TopologyParameters() const override;
    TfSpan<const TfToken> ValueParameters() const override;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenCurveBuffer const &upstream,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenChunkView *view) const override;
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override;
private:
    const TfToken sWeight{"sculpt:weight"}, sSpace{"sculpt:space"};
    const TfToken sCurveIds{"sculpt:curveIds"}, sCvOffsets{"sculpt:cvOffsets"};
    const TfToken sDeltas{"sculpt:deltas"}, sEpoch{"sculpt:epoch"};
    const TfToken sLocked{"sculpt:lockedCurves"}, sRootPrims{"sculpt:rootPrims"};
    const TfToken sRootUVs{"sculpt:rootUVs"};
    const TfToken sRootFrame{"rootFrame"}, sObject{"object"}, sMask{"mask"};
};

class UsdGenRegistrar;
/// Legacy registration stub; S4-4 replaces it with a def.
UsdGenOpId UsdGenRegisterSculptLayerOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_SCULPT_LAYER_H
