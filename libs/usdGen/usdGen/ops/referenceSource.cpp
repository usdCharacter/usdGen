#include "usdGen/ops/referenceSource.h"

#include <memory>
#include "usdGen/registries/registrar.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

struct UsdGenReferenceSourceCapture final : public UsdGenCapture
{
    bool OwnsBuffer() const override { return true; }
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenReferenceSourceCapture>(*this);
    }
};

TfTokenVector const kReferenceInputs{TfToken("reference")};

}  // namespace

TfSpan<const TfToken> UsdGenReferenceSourceOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(kReferenceInputs.data(), kReferenceInputs.size());
}

UsdGenEpoch UsdGenReferenceSourceOp::CaptureDigest(
    UsdGenCaptureContext const &ctx) const
{
    // The scheduler folds resolved identity/generation into this digest.
    TF_UNUSED(ctx);
    return {0x524546534f555243ULL, 0x455f4350555f7631ULL};
}

std::unique_ptr<UsdGenCapture> UsdGenReferenceSourceOp::CreateCapture() const
{
    return std::make_unique<UsdGenReferenceSourceCapture>();
}

bool UsdGenReferenceSourceOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &, UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    if (!out || ctx.referenceCount != 1 || !ctx.resolvedReferences ||
        !ctx.resolvedReferences[0] || !ctx.resolvedReferences[0]->value) {
        if (diag) diag->Error(
            "UsdGenReferenceSource: exactly one resolved reference value is required");
        return false;
    }

    // Buffer assignment shares immutable VtArray/plane storage, including the
    // optional C3 rest plane (which the compiler materializes from authored
    // rest or points fallback). Downstream writers are made private by
    // PrepareNodeForEval before raw writes.
    static_cast<UsdGenReferenceSourceCapture *>(out)->MutableBuffer() =
        ctx.resolvedReferences[0]->value->buffer;
    return true;
}

UsdGenOpId UsdGenRegisterReferenceSourceOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenReferenceSourceOp().Type(), [] { return std::make_unique<UsdGenReferenceSourceOp>(); });
}

}  // namespace usdGen
