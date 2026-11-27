#include "usdGenCurves/ops/referenceSource.h"
#include "usdGenCurves/types/curveCapture.h"

#include <memory>
#include "usdGen/api/ext/registrar.h"
#include "usdGen/api/span.h"


namespace usdGen {
namespace {

/// Owns its output: the compiled reference set itself (immutable, exported),
/// handed on without a copy.
struct UsdGenReferenceSourceCapture final : public UsdGenCurveCapture
{
    UsdGenRef<const UsdGenElementSet> set;
    bool OwnsBuffer() const override { return true; }
    UsdGenRef<const UsdGenElementSet> Output() const override { return set; }
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenReferenceSourceCapture>(*this);
    }
};

constexpr char const *kReferenceInputs[] = {"reference"};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &record, void *)
{
    return std::make_unique<UsdGenReferenceSourceOp>(record);
}

}  // namespace

UsdGenEpoch UsdGenReferenceSourceOp::CaptureDigest(
    UsdGenCaptureContext const &ctx) const
{
    // The scheduler folds resolved identity/generation into this digest.
    (void)ctx;
    return {0x524546534f555243ULL, 0x455f4350555f7631ULL};
}

std::unique_ptr<UsdGenCapture> UsdGenReferenceSourceOp::CreateCapture() const
{
    return std::make_unique<UsdGenReferenceSourceCapture>();
}

bool UsdGenReferenceSourceOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &, UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    if (!out || ctx.referenceCount != 1 || !ctx.resolvedReferences ||
        !ctx.resolvedReferences[0] || !ctx.resolvedReferences[0]->value) {
        if (diag) diag->Error(
            "UsdGenReferenceSource: exactly one resolved reference value is required");
        return false;
    }

    // The compiled set (rest included: the compiler materializes it from
    // authored rest or the points fallback). It is exported, so downstream
    // writers derive their own arenas and never write it in place.
    UsdGenRef<const UsdGenElementSet> const &set = ctx.resolvedReferences[0]->value->set;
    if (!set) {
        if (diag) diag->Error("UsdGenReferenceSource: the resolved reference value has no set");
        return false;
    }
    static_cast<UsdGenReferenceSourceCapture *>(out)->set = set;
    return true;
}

UsdGenOpId UsdGenRegisterReferenceSourceOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenReferenceSource";
    d.displayName = "ReferenceSource";
    d.create = &Create;
    d.countChanges = UsdGenCountsAtLevel(1);
    d.lane = UsdGenLane::Reference;
    d.geometryInputArity = 0;
    d.traits = kUsdGenOpGenerator;
    d.referenceInputs = UsdGenSpan(kReferenceInputs);
    return r.RegisterOp(d);
}

}  // namespace usdGen
