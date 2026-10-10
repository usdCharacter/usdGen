// usdGen — UsdGenCollideOp. 02-schema.md §2.8, 04-operators.md §4,
// docs/freezes/C1.md §2 (UsdGenCollide row).
//
// Iterative push-out from collider meshes. Every surface named in the node's
// surfaces list is a collider: the inherited bound skin (front) plus the
// usdGen:colliders targets the builders append after it, so this matches 02
// §1's "out of colliders and the skin" with no boundary marker and no change
// to root-surface/front semantics. Only deformed points are read.
//
// Per iteration, per CV p: closest point q over all collider triangles
// (node order, ascending faces, fan triangulation, first-win ties),
// penetration pen = offset - |p - q|; no push when pen <= 0. Colliders past
// the brute-force threshold are indexed by a Morton-LBVH rebuilt per surface
// epoch (plan/04 §4 cost C); exact ties break by (collider, face, fan)
// ordinal, so any visit order reproduces the first-win exactly. The push is
// dir * pen * pushAmount * mask with dir = (p - q)/|p - q| (the winning
// triangle's unit normal when p == q), computed in double precision.
// resolveType = flexible pushes each CV independently (A7 S10: "push CVs to
// closest surface point"); stiff finds the first penetrating CV and moves
// the strand rigidly — a translation when the root penetrates, otherwise a
// rotation of the downstream CVs about the last clean CV (the hinge), which
// preserves every segment length (A7 S10: "rotate first intersecting
// segment"). mask (the UsdGenDeformer envelope) clamps to [0,1]; with
// pushAmount == 0, mask == 0, or no penetration, Evaluate copies the input
// bit-for-bit. TopologyEffect = None. CPU lane only: the CUDA capability
// matrix has no UsdGenCollide row, so the planner rejects the graph exactly
// like any other CPU-only operator (Clump precedent).
#ifndef USDGEN_OP_COLLIDE_H
#define USDGEN_OP_COLLIDE_H

#include "usdGen/op.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

// Per-phase wall time accumulated by Evaluate while the phase probe is on
// (bench-only diagnostic, default off; see benchUsdGenHairCollide). Query
// time covers the whole closest-point search (broad + narrow; the bench
// splits them with a brute-force calibration leg), response covers the
// push math. Counts stay exact; production cost when off is one relaxed
// atomic load per chunk plus predicted branches on the search loops.
struct UsdGenCollidePhaseStats
{
    double queryMs = 0.0;    // whole closest-point search (broad + narrow)
    double respMs = 0.0;     // penetration/response math
    uint64_t queries = 0;    // closest-point queries issued
    uint64_t triTests = 0;   // triangle tests evaluated
    uint64_t nodeVisits = 0; // BVH nodes visited (0 on the brute-force path)
};

// Arming BVH memo, defined in collide.cpp: the last build's exact inputs
// per collider plus the built BVH. Lets groom-only recaptures reuse the
// identical BVH instead of rebuilding it. Owned by the op object (which
// survives stable Recompiles); never shared, never copied.
struct UsdGenCollideBvhMemo;

class UsdGenCollideOp final : public UsdGenOp
{
public:
    UsdGenCollideOp();
    ~UsdGenCollideOp() override;
    UsdGenCollideOp(UsdGenCollideOp const &) = delete;
    UsdGenCollideOp &operator=(UsdGenCollideOp const &) = delete;

    static void SetPhaseProbe(bool enabled);
    static void ResetPhaseProbe();
    static UsdGenCollidePhaseStats ReadPhaseProbe();
    // Test/diagnostic hooks (defaults: brute force off, threshold 16).
    // ForceBruteForce skips every BVH and runs the brute-force loop, so
    // differential tests can prove the BVH bit-identical; USDGEN_COLLIDE_BRUTE
    // in the environment turns it on at startup. A collider with more than
    // BruteForceThreshold non-degenerate triangles gets a BVH, smaller ones
    // stay brute force (measured crossover; mixed colliders compose by
    // ordinal, so either side of the cut is exact).
    static void SetForceBruteForce(bool enabled);
    static void SetBruteForceThreshold(size_t tris);

    TfToken Type() const override { return TfToken("UsdGenCollide"); }
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
    std::unique_ptr<UsdGenCollideBvhMemo> _bvhMemo;
};

}  // namespace usdGen

#endif  // USDGEN_OP_COLLIDE_H
