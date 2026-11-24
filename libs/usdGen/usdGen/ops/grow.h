// usdGen — UsdGenGrowOp (M1). 02-schema.md §2.6, 04-operators.md §2.2.
// Roots -> straight strands: sets the CV count (usdGen:segments), the length,
// and the lift off the surface. TopologyEffect = CvCount (a generator that
// re-partitions the CV layout of its input's curve set).
#ifndef USDGEN_OP_GROW_H
#define USDGEN_OP_GROW_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenGrowOp final : public UsdGenOp
{
public:
    explicit UsdGenGrowOp(UsdGenOpRecord const &record) noexcept : _record(&record) {}
    ~UsdGenGrowOp() override = default;

    UsdGenOpRecord const &Record() const noexcept { return *_record; }
    // Registered values served from the record; removed with the legacy virtuals (S4-8).
    TfToken Type() const override { return _record->Name(); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::CvCount; }
    bool IsGenerator() const override { return true; }

    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }

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
    bool CaptureDigestComplete() const override { return true; }
private:
    UsdGenOpRecord const *_record;
};

/// Parameter slots: the order of UsdGenRegisterGrowOp's decl table.
namespace growP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kEnabled, kSegments, kDirection, kLength, kLengthRandom,
    kDirectionVector, kLift, kAzimuth, kAzimuthRandom,
    kCount
};
enum Direction : int { kDirSurfaceNormal, kDirVector };
enum Tok : uint16_t { kTokSurfaceNormal, kTokVector };
}  // namespace growP

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterGrowOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_GROW_H
