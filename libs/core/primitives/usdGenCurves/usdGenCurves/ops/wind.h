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

#include "usdGenCurves/ops/recordedOp.h"
#include "usdGen/api/ids.h"
#include "usdGenCurves/export.h"


namespace usdGen {

namespace windP {
/// Slots, in decl order.
enum : UsdGenParamSlot {
    kInput, kSurface, kSeed, kEnabled, kDirection, kConst, kGust, kLowStrength,
    kLowFrequency, kLowRate, kHighStrength, kHighFrequency, kHighRate, kStiffness,
    kKnots, kInterp, kMask, kCount
};
enum Tok : uint16_t { kTokLinear, kTokCatmullRom, kTokBspline, kTokConstant };
}  // namespace windP

class USDGEN_CURVES_INTERNAL_API UsdGenWindOp final : public UsdGenOp
{
public:
    explicit UsdGenWindOp(UsdGenOpRecord const &record) : UsdGenOp(record) {}
    ~UsdGenWindOp() override = default;

    bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) override;
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const override;
    bool Capture(UsdGenCaptureContext const &ctx,
                 UsdGenOpInputs const &in,
                 UsdGenCapture *out,
                 UsdGenDiagnostics *diag) override;
    void Evaluate(UsdGenEvalContext const &ctx,
                  UsdGenCapture const &capture,
                  UsdGenChunkView const &chunk) const override;
    std::unique_ptr<UsdGenCapture> CreateCapture() const override;
    uint32_t PlanesTouched() const override;
    bool ReadsTime(UsdGenParamView const &params) const override {
        auto const gust = params.GetScalarField(windP::kGust, 0.0);
        return gust.connected || gust.literal != 0.0;
    }
};

class UsdGenRegistrar;
UsdGenOpId UsdGenRegisterWindOp(UsdGenRegistrar &);

}  // namespace usdGen

#endif  // USDGEN_OP_WIND_H
