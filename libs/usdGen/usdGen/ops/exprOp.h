// usdGen — UsdGenExprOp. 02-schema.md §2.7, 04-operators.md §3, 07 §7.8.
//
// A styler whose kernel is a SeExpr expression, evaluated ONCE PER CAPTURE
// into a per-CV or per-curve array; Evaluate is a memcpy/add of that array
// (04 §3). Capture-time only (S37): no expression runs in Evaluate.
//
//   usdGen:expr:source     string, default "" — the SeExpr program.
//   usdGen:expr:maps       rel — accepted structurally (02 §6.1), never
//                          sampled in the kernel: UsdGenResolvedMapValue is
//                          identity-only, and sampler calls resolve against a
//                          UsdGenExpression prim's input:<name> bindings,
//                          which a bare source string has. Any ptex() /
//                          geoSampler() call therefore fails closed, naming
//                          the expr:maps targets that do exist; maps under
//                          the rel with a sampler-free expression warn once
//                          and are ignored (displace:map precedent).
//   usdGen:expr:returnType uniform token "displacement", displacement|width|
//                          color. displacement adds a captured vec3 to the
//                          points; width SETS the widths from a captured
//                          scalar (Width's set-path envelope). color fails
//                          closed: the operator-emitted primvar transport
//                          (R24/DefaultOutputPlane) cannot carry a vec3 color
//                          plane without a scheduler change, so publishing
//                          one would overrun its allocation — the same class
//                          of fail-closed as Displace mode=vector.
//   usdGen:mode            uniform token "cv", cv|curve — the granularity the
//                          expression is captured at (02 §2.7.2): cv
//                          evaluates per CV (point domain), curve evaluates
//                          per root (primitive domain) and broadcasts the
//                          per-curve value to every CV of the strand.
//
// Compile happens at Capture through the existing lane (Frontend::Compile +
// CpuExpressionContext::Build + EvaluateProgram, the cpuParameters.cpp path);
// no new machinery. An empty or blank source fails closed (a compiler-side
// UsdGenExpression with an empty source is likewise a compile error), as
// does any syntax error, with the frontend diagnostics quoted. $frame/$time
// fail closed: capture cannot see time, so a time-varying source would go
// stale without recapturing. $value fails closed: it is the connected
// parameter's literal, and a bare source has none (width authors use $cWidth
// instead). Non-finite or unrepresentable results fail closed via the
// interpreter's InvalidValue (Store admits finite values only).
//
// The mask envelope (literal or connected "mask", ValueParameters like every
// other styler) scales the displacement add and blends set-widths toward the
// upstream width, exactly like Width's replace path; a zero mask is a
// bit-for-bit pass-through of the target plane. TopologyEffect = None.
#ifndef USDGEN_OP_EXPROP_H
#define USDGEN_OP_EXPROP_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenExprOp final : public UsdGenOp
{
public:
    UsdGenExprOp() = default;
    ~UsdGenExprOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenExprOp"); }
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

};

namespace exprOpP {
/// Declaration order == today's TopologyParameters() then ValueParameters().
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kMode, kReturnType, kSource, kMaps, kEnabled, kMask, kCount
};
enum Tok : uint16_t { kTokCv, kTokCurve, kTokDisplacement, kTokWidth, kTokColor };
}  // namespace exprOpP

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterExprOpOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_EXPROP_H
