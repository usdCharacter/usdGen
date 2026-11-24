// usdGen — UsdGenDirectionOp. 02-schema.md §2.7.1, 04-operators.md §2.13.
// a host groomer Tilt: comb every strand toward an authored target direction.
//
// Per strand, with root frame (T, B, N) and input points P, the kernel builds
// a unit target direction and rotates the strand toward it:
//
//   1. Base target: the bound direction:source tangents (see below) when the
//      relationship targets a curve set or an operator, else the authored
//      usdGen:direction (literal vector3f, or a groom/primitive-connected
//      expression sampled at the strand root).
//   2. Frame shaping, in this order, all in degrees: tiltU about T, tiltV
//      about B, tiltN about N, aroundN about N (composes with tiltN), then
//      lift about B (the Grow lift axis: N x T == B). followSkinContour in
//      [0,1] then blends the target toward its own projection onto the root
//      tangent plane span(T, B) (dPdu/dPdv at the root); a target already
//      parallel to N is kept as is.
//   3. Rotation: mode = rigid rotates every CV offset about the root by the
//      fraction w_i = amount x ramp(t_i) x mask_i of the minimal rotation
//      taking the input root segment onto the target (w clamped to [0,1]).
//      mode = perSegment walks the segments Bend-style, accumulating a
//      cumulative rotation: segment k turns by amount x (R(t_k)-R(t_{k-1}))
//      x mask_k of its remaining angle to the target (signed deltas, each
//      clamped to [-1,1]; R(t_{-1}) := 0, so a flat ramp turns once at the
//      root exactly like rigid). Segment lengths are preserved by
//      construction in perSegment; in rigid they are preserved when w is
//      uniform along the strand.
//
// Roots never move in either mode. amount = 0 (or a zero-length input root
// segment, or a zero-length target) copies the input; with amount = 0 the
// copy is bitwise. amount/lift are sampled per curve root, the mask per CV
// (clamped to [0,1]); tiltU/tiltV/tiltN/aroundN/followSkinContour are
// groom-wide. direction:source tangents are captured (nearest-sample tangent
// per strand root, Part-style, on rest positions); everything else evaluates
// per chunk, so value edits sweep without recapture. A map-driven
// direction:source (04 §2.13) fails closed: no CPU kernel consumes the map
// lane yet, so only curve-set/operator targets are supported. No random
// draws, so no salt is used. TopologyEffect = None.
#ifndef USDGEN_OP_DIRECTION_H
#define USDGEN_OP_DIRECTION_H

#include "usdGen/op.h"
#include "usdGen/registries/handles.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace directionP {
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kMode, kSource,
    kEnabled, kDirection, kAmount, kLift, kTiltU, kTiltV, kTiltN, kAroundN, kFollow, kKnots,
    kInterp, kMask,
    kCount
};
enum Mode : int { kModeRigid, kModePerSegment };
enum Interp : int { kInterpLinear, kInterpCatmullRom, kInterpBspline, kInterpConstant };
}  // namespace directionP

class UsdGenDirectionOp final : public UsdGenOp
{
public:
    explicit UsdGenDirectionOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    ~UsdGenDirectionOp() override = default;


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

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterDirectionOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_DIRECTION_H
