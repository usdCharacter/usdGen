#ifndef USDGEN_GPU_DEFORM_CURVES_H
#define USDGEN_GPU_DEFORM_CURVES_H
#include "curveGeometry.h"
#include "rbf.h"
#include "width.h"
namespace usdGen { namespace gpu {

// Runtime controls for the C3 deformation. `mask` IS the operator envelope
// (usdGen:mask) and may be a Groom-, Primitive-, or Point-domain field in
// [0,1]; exactly 0 is a bitwise pass-through. enabled is Groom-only;
// lockRoots is Groom- or Primitive-domain (a point-domain root lock is
// ambiguous for a whole-strand correction and is rejected).
struct DeformParameters {
 ScalarField mask = ScalarField::Literal(1.0f);
 BoolField enabled = BoolField::Literal(true);
 BoolField lockRoots = BoolField::Literal(true);
};

class CudaRbfCurveDeformer {
public:
 ~CudaRbfCurveDeformer();
 // primitive/point envelopes may be empty (identity 1); rootTargets is one per curve.
 RbfStatus Deform(CudaRbfBinding& rbf, DeviceCurveGeometryView geometry, DeviceView<const float3> rootTargets, float groomEnvelope, DeviceView<const float> primitiveEnvelope, DeviceView<const float> pointEnvelope, DeviceView<float3> output, cudaStream_t stream);
 // New typed path. Geometry points are the incoming styled REST-domain input;
 // restPoints remains canonical binding data and is not used as the evaluated
 // source. All output is private until Finish returns Ok.
 RbfStatus Deform(CudaRbfBinding& rbf, DeviceCurveGeometryView geometry,
                  DeviceView<const float3> rootTargets,
                  DeformParameters parameters, DeviceView<float3> output,
                  cudaStream_t stream);
 RbfStatus Finish(CudaRbfBinding& rbf,cudaStream_t stream);
 // Fresh staged path. Geometry, roots, parameter fields, and output are
 // borrowed through CommitFreshFinish. The caller owns the native terminal callbacks and
 // calls each Commit only after native success and the corresponding launcher
 // has returned.  Commits are host-only; no fresh phase publishes output
 // until CommitFreshFinish succeeds.
 RbfStatus BeginFreshShape(DeviceCurveGeometryView geometry,
                           DeviceView<const float3> rootTargets,
                           DeformParameters parameters,
                           DeviceView<float3> output, cudaStream_t stream,
                           UsdGenExecutionMemoryReservation* reservation = nullptr);
 RbfStatus CommitFreshShape();
 // This forwards to CudaRbfBinding's fresh evaluator.  The binding remains
 // externally owned and its terminal callback/proof belongs to the caller.
 RbfStatus BeginFreshEvaluate(CudaRbfBinding& rbf, cudaStream_t stream,
                              UsdGenExecutionMemoryReservation* reservation = nullptr);
 RbfStatus CommitFreshEvaluate(CudaRbfBinding& rbf);
 RbfStatus BeginFreshApply(cudaStream_t stream);
 RbfStatus CommitFreshApply();
 RbfStatus BeginFreshCopy(cudaStream_t stream);
 RbfStatus CommitFreshFinish();
 // True only while a submitted fresh phase lacks terminal proof, or after a
 // post-submit failure.  Such objects must be quarantined by their owner.
 bool HasUnprovenWork() const noexcept;
private: DeviceBuffer<float3> warped_; DeviceBuffer<int> flags_; cudaEvent_t ready_=nullptr; bool pending_=false, poisoned_=false;
 DeviceBuffer<float3> staged_; DeviceView<float3> output_{}; size_t outputCount_=0;
 int* freshHostError_ = nullptr;
 UsdGenExecutionResourcePermit freshHostErrorPermit_;
 enum class FreshPhase { None, Shape, ShapeReady, Evaluate, EvaluateReady, Apply, ApplyReady, Copy };
 FreshPhase freshPhase_ = FreshPhase::None;
 bool freshArmed_ = false, freshUnproven_ = false, freshFailed_ = false;
 bool freshUsed_ = false;
 bool legacyUsed_ = false;
 CudaRbfBinding* freshRbf_ = nullptr;
 DeviceCurveGeometryView freshGeometry_{};
 DeviceView<const float3> freshRoots_{};
 DeformParameters freshParameters_{};
 int deviceIndex_=-1;
 RbfStatus deformImpl(CudaRbfBinding&, DeviceCurveGeometryView,
                      DeviceView<const float3>, DeformParameters,
                      float, DeviceView<const float>, DeviceView<const float>,
                      DeviceView<float3>, cudaStream_t);
};
}}
#endif
