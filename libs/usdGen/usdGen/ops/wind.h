// usdGen — UsdGenWindOp. 02-schema.md §2.8, 04-operators.md §4.
//
// Time-dependent force field with constant and gust terms. Per curve c with
// rest root R_c and per CV i with hairT t:
//
//   D      = normalize(direction)                      // (0, 0, 1); per root when connected
//   gust_c = gustStrength * (2 * FBM3(R_c * 0.5 + h_c + (tSec * 0.5, 0, 0)) - 1)
//   flex   = clamp(1 - stiffness * stiffLUT(t), 0, 1)   // groom stiffness, ramp over hairT
//   d      = (constStrength + gust_c) * flex * mask * t // t locks the root
//   P[i]   = inP[i] + D * d                            // bitwise copy when d == 0
//
// h_c is the per-curve hash vector from UsdGenDraw01 over (seed, curveId),
// mirroring UsdGenNoiseOp's deterministic CPU noise (SeExpr2::FBM, S38 — the
// single noise implementation); the FBM shape is fixed at 3 octaves,
// lacunarity 2.0, gain 0.5. gustStrength 0 is pure constant deflection and is
// time-independent. Roots (t = 0) never move. tSec is the cook time in
// seconds: UsdGenEvalContext.time divided by desc->timeCodesPerSecond, with
// the scheduler's own finite/positive guard (scheduler.cpp), so gusts animate
// yet stay deterministic per time + seed.
//
// Optional coherent billow adds two traveling waves. Each samples the CV's
// rest position projected onto the normalized direction, with a seed-wide
// phase; strengths default to zero for the legacy result. Low and high use
// t^2 and t^3 envelopes to curve shafts while holding their roots.
// When an upstream Clump supplies the native ID/center/weight quartet, Wind
// blends its animated field with stable center-seeded group fields according
// to the Clump's effective per-CV weight. Gust and low billow resolve from
// coarse to fine; high billow resolves from fine to coarse. Each component's
// unclaimed fraction keeps the independent strand field. Constant deflection
// remains authored per strand; strength, direction, mask and stiffness remain
// per-strand/per-CV where authored. Missing quartets or zero weights take the
// exact legacy arithmetic path, including older clumpId-only sources.
//
// Capture pins only time-independent data (per-CV rest positions and roots); the gust is
// re-sampled every Evaluate from the live time. That is the readsTime/rebuild
// split of cpuParameters' SamplerState: time-dependent state is never pinned
// in the capture, so no time term enters the capture digest. All Wind schema
// parameters are value-class (C1); TopologyParameters is the base triple.
// CPU-only: the CUDA capability matrix has no UsdGenWind row, so the planner
// rejects a Wind graph exactly like a Clump graph. TopologyEffect = None.
#ifndef USDGEN_OP_WIND_H
#define USDGEN_OP_WIND_H

#include "usdGen/op.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenWindOp final : public UsdGenOp
{
public:
    UsdGenWindOp() = default;
    ~UsdGenWindOp() override = default;

    TfToken Type() const override { return TfToken("UsdGenWind"); }
    UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }
    bool ReadsTime() const override { return true; }

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
    bool ReadsTime(UsdGenParamView const &params) const override {
        auto const gust = params.GetScalarField(sGust, 0.0);
        return gust.connected || gust.literal != 0.0;
    }
private:
    const TfToken sDirection{"direction"}, sConst{"constStrength"};
    const TfToken sGust{"gustStrength"}, sStiffness{"stiffness"};
    const TfToken sLowStrength{"billowLowStrength"}, sHighStrength{"billowHighStrength"};
    const TfToken sLowFrequency{"billowLowFrequency"}, sHighFrequency{"billowHighFrequency"};
    const TfToken sLowRate{"billowLowRate"}, sHighRate{"billowHighRate"};
    const TfToken sKnots{"stiffness:knots"}, sInterp{"stiffness:interpolation"};
    const TfToken sLinear{"linear"}, sCatmullRom{"catmullRom"};
    const TfToken sBspline{"bspline"}, sConstant{"constant"};
    const TfToken sMask{"mask"};
};

}  // namespace usdGen

#endif  // USDGEN_OP_WIND_H
