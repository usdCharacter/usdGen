#ifndef USDGEN_OP_DEFORM_H
#define USDGEN_OP_DEFORM_H

#include "usdGen/op.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/rbfField.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"

#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// UsdGenDeform: topology-preserving RBF or centerline deformation.
///
/// CUDA lane: driven by samples of the bound surface (the persistent RBF
/// library in cudaExecution.cpp); this class supplies its metadata only.
/// CPU lane: driven by usdGen:guides, or the bound surface when absent. Every
/// driver CV (up to usdGen:rbfSamples of them, farthest-point sampled) is an
/// RBF sample bound at its rest position and moved to its current one, and
/// every incoming CV moves by the resulting field. Surface input requires
/// Default-time rest data. Its input is surface rest-local and its output is
/// in current groom space, so different parents work and shared ancestor
/// motion is applied once by publication.
/// CPU mode=curveWrap instead transports the incoming geometry around one
/// open guide using rotation-minimizing frames. It supports straight rest
/// curves and preserves transverse offsets; the centerline supplies no roll.
/// Multiple open guides use a categorical Ptex regionMap at rest roots and
/// guideRegions IDs to select exactly one centerline per incoming strand.
class UsdGenDeformOp final : public UsdGenOp
{
public:
   UsdGenDeformOp();
   ~UsdGenDeformOp() override = default;

   TfToken Type() const override { return TfToken("UsdGenDeform"); }
   UsdGenTopoFx TopologyEffect() const override { return UsdGenTopoFx::None; }
   UsdGenRole Role() const override { return UsdGenRole::Curves; }
   bool IsGenerator() const override { return false; }

   TfSpan<const TfToken> TopologyParameters() const override;
   TfSpan<const TfToken> ValueParameters() const override;
   TfSpan<const TfToken> ReferenceInputs() const override;
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

   // Testing only: the cached surface selection (empty until the first
   // surface-driven capture settles it), and how many digests hashed
   // only the chosen posed drivers. The scheduler runs digest and
   // capture back to back on the commit thread, so the counter needs
   // no synchronization.
   std::vector<size_t> SurfaceSelectionForTesting() const { return surfaceSelection_; }
   uint64_t ChosenDigestHitsForTesting() const { return chosenDigestHits_; }

private:
   TfTokenVector topologyParameters_;
   TfTokenVector valueParameters_;
   // The factored system survives poses that keep the same rest samples.
   rbf::CubicField field_;
   std::vector<GfVec3d> boundRest_;
   // The farthest-point selection is a pure function of the rest drivers, so
   // it is reused while they are bitwise unchanged (the steady-state pose
   // path); any rest edit re-selects. selectRest_ holds the drivers the
   // cached selection was made from.
   std::vector<GfVec3d> selectRest_;
   std::vector<size_t> selection_;
   size_t selectionBudget_ = 0;
   bool selectionValid_ = false;
   // The surface-driven twin of the selection cache, keyed by the rest
   // points' shared-buffer identity (pointer + size, the ContentDigestCache
   // rule) instead of a 1.2MB memcmp. The cache holds a VtArray reference,
   // and VtArray is copy-on-write, so any in-place edit detaches the
   // writer to a new buffer and the key misses; a hit therefore proves the
   // bytes are unchanged and reads them straight from the surface
   // (float->double conversion is injective, so equal bytes mean equal
   // rest drivers). The held reference also keeps the buffer alive, so a
   // recycled address can never false-hit. A hit skips the conversion and
   // transforms only the chosen posed samples instead of every driver.
   // Mutable: CaptureDigest reads the key (never writes it) so a pose can
   // hash only the posed drivers Capture will touch; the scheduler runs
   // digest and capture back to back on the commit thread, so the key
   // cannot change between the two. See MixChosenPosed.
   mutable VtVec3fArray surfaceRestRef_;
   mutable std::vector<size_t> surfaceSelection_;
   mutable size_t surfaceBudget_ = 0;
   mutable bool surfaceValid_ = false;
   // The rest drivers' digest contribution, memoized across poses (the
   // surface branch then hashes only the chosen posed drivers; the guide
   // branch still hashes every frame). Shared by the surface and guide
   // branches; a rewire misses once and rehashes.
   mutable opUtil::ContentDigestCache<VtVec3fArray> restDigest_;

   // Hashes the posed surface drivers Capture reads on a selection-cache
   // hit (surface-driven RBF only): the rest decides the selection, so
   // only posed[chosen] can move the output, and a move anywhere else
   // must not re-capture. False (caller hashes the whole posed buffer)
   // without a cache hit, in wrap mode (which binds every driver), on a
   // rest/posed size mismatch (Capture fails there; the subset would
   // read out of bounds), or when the field is unbound or the selection
   // degenerate (Capture fails there too; the full hash keeps the
   // failure cadence what it was).
   bool MixChosenPosed(opUtil::Digest *d, UsdGenCaptureContext const &ctx,
                       UsdGenSurfaceDesc const &surface) const;
   mutable uint64_t chosenDigestHits_ = 0;
};

}  // namespace usdGen

#endif  // USDGEN_OP_DEFORM_H
