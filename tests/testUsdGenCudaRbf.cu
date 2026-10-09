#include "gpu/rbf.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace usdGen::gpu;
static float3 f(float x,float y,float z){return make_float3(x,y,z);}
static bool near(float3 a,float3 b,float e=2e-4f){return fabs(a.x-b.x)<e&&fabs(a.y-b.y)<e&&fabs(a.z-b.z)<e;}
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(1);}}
int main(){
  cudaStream_t s, s2; check(cudaStreamCreate(&s)); check(cudaStreamCreate(&s2));
  // Non-coplanar support is deliberately required by the full affine basis.
  std::vector<float3> rest={f(0,0,0),f(1,0,0),f(0,1,0),f(0,0,1),f(1,1,1)};
  float3 *dr,*dp,*do_; check(cudaMalloc(&dr,5*sizeof(float3)));check(cudaMalloc(&dp,5*sizeof(float3)));check(cudaMalloc(&do_,5*sizeof(float3)));
  check(cudaMemcpyAsync(dr,rest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  CudaRbfBinding r; assert(r.Bind({dr,5},0,s)==RbfStatus::Ok);
  // Bind exposes the explicit identity state; no uninitialized coefficients.
  assert(r.Evaluate({dr,5},{do_,5},s2)==RbfStatus::Ok); assert(r.Finish(s2)==RbfStatus::Ok);
  std::vector<float3> got(5);check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s2));check(cudaStreamSynchronize(s2));for(int i=0;i<5;i++)assert(near(got[i],rest[i]));
  // Identity and repeated Solve reuse the persistent factorization.
  check(cudaMemcpyAsync(dp,rest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s)); assert(r.Solve({dp,5},s)==RbfStatus::Ok); assert(r.Evaluate({dr,5},{do_,5},s)==RbfStatus::Ok); assert(r.Finish(s)==RbfStatus::Ok);
  check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));for(int i=0;i<5;i++)assert(near(got[i],rest[i]));
  // Affine rotation around Z plus translation must be represented exactly.
  std::vector<float3> pose(5);for(int i=0;i<5;i++)pose[i]=f(-rest[i].y+2,rest[i].x-3,rest[i].z+4);
  check(cudaMemcpyAsync(dp,pose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));assert(r.Solve({dp,5},s)==RbfStatus::Ok);assert(r.Evaluate({dr,5},{do_,5},s2)==RbfStatus::Ok); assert(r.Finish(s2)==RbfStatus::Ok); check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s2));check(cudaStreamSynchronize(s2));for(int i=0;i<5;i++)assert(near(got[i],pose[i]));
  // Interpolation at drivers and a nonlinear displacement (test-only readback oracle).
  for(int i=0;i<5;i++)pose[i].z+=rest[i].x*rest[i].y+.2f*rest[i].z*rest[i].z;
  check(cudaMemcpyAsync(dp,pose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));assert(r.Solve({dp,5},s)==RbfStatus::Ok);assert(r.Evaluate({dr,5},{do_,5},s)==RbfStatus::Ok);assert(r.Finish(s)==RbfStatus::Ok);check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));for(int i=0;i<5;i++)assert(near(got[i],pose[i],1e-3f));
  // Multiple async evaluations share one completion diagnostic; a NaN poisons
  // the generation and prevents a subsequent publish/evaluate.
  pose[0].x=NAN; check(cudaMemcpyAsync(dp,pose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));assert(r.Evaluate({dp,5},{do_,5},s)==RbfStatus::Ok);assert(r.Evaluate({dr,5},{do_,5},s2)==RbfStatus::Ok);assert(r.Finish(s2)==RbfStatus::NonFiniteInput);assert(r.Evaluate({dr,5},{do_,5},s)==RbfStatus::InvalidArgument);
  // A planar binding is rejected even with smoothing: polynomial rank is absent.
  std::vector<float3> plane={f(0,0,0),f(1,0,0),f(0,1,0),f(1,1,0),f(.2f,.3f,0)};check(cudaMemcpyAsync(dr,plane.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));CudaRbfBinding bad;assert(bad.Bind({dr,5},1e-3,s)==RbfStatus::RankDeficient);
  // A non-finite rest sample is rejected at the extent proof on both the
  // direct and fresh paths, and the binding stays usable afterwards.
  std::vector<float3> nanRest=rest; nanRest[2].y=NAN;
  check(cudaMemcpyAsync(dr,nanRest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  CudaRbfBinding nanDirect; assert(nanDirect.Bind({dr,5},0,s)==RbfStatus::NonFiniteInput);
  CudaRbfBinding nanStaged;
  assert(nanStaged.BeginFreshBind({dr,5},0,s)==RbfStatus::Ok);
  check(cudaStreamSynchronize(s)); assert(nanStaged.CommitFreshBindExtent()==RbfStatus::NonFiniteInput);
  check(cudaMemcpyAsync(dr,rest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(nanDirect.Bind({dr,5},0,s)==RbfStatus::Ok);
  // A non-finite posed sample is rejected by the direct solve flag proof,
  // and the binding stays usable afterwards.
  std::vector<float3> nanPose=rest; nanPose[1].z=NAN;
  check(cudaMemcpyAsync(dp,nanPose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(nanDirect.Solve({dp,5},s)==RbfStatus::NonFiniteInput);
  check(cudaMemcpyAsync(dp,rest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(nanDirect.Solve({dp,5},s)==RbfStatus::Ok);
  // Fresh binding has three externally-proved, host-only commit boundaries.
  // These calls deliberately synchronize only in the test as the parent's
  // native-completion proof stand-in; production must not do so in commits.
  check(cudaMemcpyAsync(dr,rest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  CudaRbfBinding staged;
  assert(!staged.CanAcceptFreshSolve() && !staged.CanRollbackFreshSolve());
  // Commits cannot be guessed or replayed, and capture rejection has no
  // candidate side effect (the ordinary stream remains reusable).
  assert(staged.CommitFreshBindExtent()==RbfStatus::InvalidArgument);
  cudaGraph_t graph=nullptr; check(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
  assert(staged.BeginFreshBind({dr,5},0,s)==RbfStatus::InvalidArgument);
  check(cudaStreamEndCapture(s,&graph)); check(cudaGraphDestroy(graph));
  assert(staged.BeginFreshBind({dr,5},0,s)==RbfStatus::Ok && staged.HasUnprovenWork() && staged.freshSampleCount()==0);
  check(cudaStreamSynchronize(s)); assert(staged.CommitFreshBindExtent()==RbfStatus::Ok && !staged.HasUnprovenWork());
  assert(staged.CommitFreshBindExtent()==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshBindRank(s)==RbfStatus::Ok && staged.HasUnprovenWork());
  check(cudaStreamSynchronize(s)); assert(staged.CommitFreshBindRank()==RbfStatus::Ok && !staged.HasUnprovenWork());
  assert(staged.BeginFreshBindLu(s)==RbfStatus::Ok && staged.HasUnprovenWork());
  check(cudaStreamSynchronize(s)); assert(staged.CommitFreshBindLu()==RbfStatus::Ok && !staged.HasUnprovenWork());
  assert(staged.CommitFreshBindLu()==RbfStatus::InvalidArgument);
  assert(staged.freshSampleCount()==5);
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok && staged.HasUnprovenWork());
  check(cudaStreamSynchronize(s)); assert(staged.CommitFreshEvaluate()==RbfStatus::Ok && !staged.HasUnprovenWork());
  check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s));
  for(int i=0;i<5;i++) assert(near(got[i],rest[i]));
  // The fresh pose has two proof boundaries.  It writes only private
  // coefficients until solver status is proven, then exposes them only to a
  // downstream fresh evaluation.  Acceptance remains an explicit final
  // transaction boundary.
  std::vector<float3> freshPose(5); for(int i=0;i<5;i++) freshPose[i]=f(rest[i].x+3,rest[i].y-2,rest[i].z+1);
  check(cudaMemcpyAsync(dp,freshPose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(staged.CommitFreshSolveInput()==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshSolve({dp,5},s)==RbfStatus::Ok && staged.HasUnprovenWork());
  check(cudaStreamSynchronize(s)); assert(staged.CommitFreshSolveInput()==RbfStatus::Ok && !staged.HasUnprovenWork());
  // The private RHS is tied to the exact accepted factorization captured by
  // BeginFreshSolve; a rebind cannot interleave before its solve/commit.
  assert(staged.BeginFreshBind({dr,5},0,s)==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshSolveFactors(s)==RbfStatus::Ok && staged.HasUnprovenWork());
  check(cudaStreamSynchronize(s)); assert(staged.CommitFreshSolve()==RbfStatus::Ok && !staged.HasUnprovenWork());
  // A solved pose cannot be overtaken by a rebind/new solve, and once
  // downstream evaluation is submitted neither decision may be guessed
  // before that proof is complete.
  assert(staged.BeginFreshBind({dr,5},0,s)==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshSolve({dp,5},s)==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(!staged.CanAcceptFreshSolve() && !staged.CanRollbackFreshSolve());
  assert(staged.AcceptFreshSolve()==RbfStatus::InvalidArgument);
  assert(staged.RollbackFreshSolve()==RbfStatus::InvalidArgument);
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok); check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s));
  for(int i=0;i<5;i++) assert(near(got[i],freshPose[i]));
  // Simulate a downstream publication rejection after a successful solve:
  // rollback swaps the old accepted coefficients back without rebinding.
  assert(staged.CanAcceptFreshSolve() && staged.CanRollbackFreshSolve());
  assert(staged.RollbackFreshSolve()==RbfStatus::Ok);
  assert(!staged.CanAcceptFreshSolve() && !staged.CanRollbackFreshSolve());
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok); check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s));
  for(int i=0;i<5;i++) assert(near(got[i],rest[i]));
  // A clean retry can solve the same pose and make it durable only through
  // AcceptFreshSolve after its downstream proof.
  assert(staged.BeginFreshSolve({dp,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshSolveInput()==RbfStatus::Ok);
  assert(staged.BeginFreshSolveFactors(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshSolve()==RbfStatus::Ok);
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok);
  assert(staged.CanAcceptFreshSolve() && staged.CanRollbackFreshSolve());
  assert(staged.AcceptFreshSolve()==RbfStatus::Ok);
  assert(!staged.CanAcceptFreshSolve() && !staged.CanRollbackFreshSolve());
  assert(staged.RollbackFreshSolve()==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok); check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s));
  for(int i=0;i<5;i++) assert(near(got[i],freshPose[i]));
  // A rejected posed candidate cannot alter the last accepted coefficients.
  pose[0]=f(NAN,0,0); check(cudaMemcpyAsync(dp,pose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(staged.BeginFreshSolve({dp,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshSolveInput()==RbfStatus::NonFiniteInput && !staged.HasUnprovenWork());
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok); check(cudaMemcpyAsync(got.data(),do_,5*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s));
  for(int i=0;i<5;i++) assert(near(got[i],freshPose[i]));
  TestFailNextFreshRbfSolveAllocation();
  assert(staged.BeginFreshSolve({dr,5},s)==RbfStatus::CudaError && !staged.HasUnprovenWork());
  pose[0]=f(NAN,0,0); check(cudaMemcpyAsync(dp,pose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(staged.BeginFreshEvaluate({dp,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::NonFiniteInput);
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok);
  // Test seams distinguish clean pre-enqueue failure (retryable, old field
  // retained) from a failure after a submitted D2H proof (permanent unsafe
  // candidate that cannot be committed or reused).
  CudaRbfBinding allocFault;
  TestFailNextFreshRbfBindAllocation();
  assert(allocFault.BeginFreshBind({dr,5},0,s)==RbfStatus::CudaError && !allocFault.HasUnprovenWork());
  assert(allocFault.BeginFreshBind({dr,5},0,s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(allocFault.CommitFreshBindExtent()==RbfStatus::Ok);
  assert(allocFault.BeginFreshBindRank(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(allocFault.CommitFreshBindRank()==RbfStatus::Ok);
  assert(allocFault.BeginFreshBindLu(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(allocFault.CommitFreshBindLu()==RbfStatus::Ok && allocFault.freshSampleCount()==5);
  CudaRbfBinding inputUnsafe;
  assert(inputUnsafe.BeginFreshBind({dr,5},0,s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(inputUnsafe.CommitFreshBindExtent()==RbfStatus::Ok);
  assert(inputUnsafe.BeginFreshBindRank(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(inputUnsafe.CommitFreshBindRank()==RbfStatus::Ok);
  assert(inputUnsafe.BeginFreshBindLu(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(inputUnsafe.CommitFreshBindLu()==RbfStatus::Ok);
  TestFailNextFreshRbfSolveAfterInputSubmit();
  assert(inputUnsafe.BeginFreshSolve({dr,5},s)==RbfStatus::CudaError && inputUnsafe.HasUnprovenWork());
  check(cudaStreamSynchronize(s));
  assert(inputUnsafe.CommitFreshSolveInput()==RbfStatus::InvalidArgument); inputUnsafe.AbandonFresh();
  assert(allocFault.BeginFreshSolve({dr,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(allocFault.CommitFreshSolveInput()==RbfStatus::Ok);
  TestFailNextFreshRbfSolveAfterSolverSubmit();
  assert(allocFault.BeginFreshSolveFactors(s)==RbfStatus::CudaError && allocFault.HasUnprovenWork());
  check(cudaStreamSynchronize(s));
  assert(allocFault.CommitFreshSolve()==RbfStatus::InvalidArgument); allocFault.AbandonFresh();
  TestFailNextFreshRbfEvaluateAllocation();
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::CudaError && !staged.HasUnprovenWork());
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(staged.CommitFreshEvaluate()==RbfStatus::Ok);
  CudaRbfBinding bindUnsafe;
  TestFailNextFreshRbfBindAfterSubmit();
  assert(bindUnsafe.BeginFreshBind({dr,5},0,s)==RbfStatus::CudaError && bindUnsafe.HasUnprovenWork());
  assert(bindUnsafe.CommitFreshBindExtent()==RbfStatus::InvalidArgument); bindUnsafe.AbandonFresh();
  // A proven rejected rebind cannot replace the accepted identity field.
  check(cudaMemcpyAsync(dr,plane.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(staged.BeginFreshBind({dr,5},0,s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshBindExtent()==RbfStatus::Ok);
  assert(staged.BeginFreshBindRank(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshBindRank()==RbfStatus::RankDeficient && staged.freshSampleCount()==5);
  check(cudaMemcpyAsync(dr,rest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
  assert(staged.CommitFreshEvaluate()==RbfStatus::Ok);
  // Unsafe evaluation is terminal: quarantine is permanent, so this is the
  // final use of staged and both fresh entry points reject thereafter.
  TestFailNextFreshRbfEvaluateAfterSubmit();
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::CudaError && staged.HasUnprovenWork());
  assert(staged.CommitFreshEvaluate()==RbfStatus::InvalidArgument); staged.AbandonFresh();
  assert(staged.BeginFreshEvaluate({dr,5},{do_,5},s)==RbfStatus::InvalidArgument);
  assert(staged.BeginFreshBind({dr,5},0,s)==RbfStatus::InvalidArgument);
  // Direct-evaluate R cache: the cold miss, the hit, refills, rebinding,
  // reshaping, and the flag path are all bitwise against the seam-disabled
  // uncached kernel on the same binding; the path counters prove which
  // path each Evaluate took.
  {
    auto readback = [&](float3* d, int nn){ std::vector<float3> v(nn); check(cudaMemcpyAsync(v.data(),d,size_t(nn)*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s)); return v; };
    auto sameBits = [&](std::vector<float3> const& a, std::vector<float3> const& b){ return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float3))==0; };
    std::vector<float3> crest={f(0,0,0),f(1,0,0),f(0,1,0),f(0,0,1),f(1,1,1)};
    std::vector<float3> cpose(5); for(int i=0;i<5;i++) cpose[i]=f(-crest[i].y+2,crest[i].x-3,crest[i].z+4);
    std::vector<float3> ccvs={f(.1f,.2f,.3f),f(.4f,.5f,.6f),f(.7f,.8f,.9f),f(1.1f,1.2f,1.3f),f(-.5f,.25f,2.f)};
    float3 *cr,*cp,*cc,*co; check(cudaMalloc(&cr,5*sizeof(float3))); check(cudaMalloc(&cp,5*sizeof(float3))); check(cudaMalloc(&cc,5*sizeof(float3))); check(cudaMalloc(&co,5*sizeof(float3)));
    check(cudaMemcpyAsync(cr,crest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(cp,cpose.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(cc,ccvs.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    CudaRbfBinding cb; assert(cb.Bind({cr,5},0,s)==RbfStatus::Ok); assert(cb.Solve({cp,5},s)==RbfStatus::Ok);
    uint64_t h0=CudaRbfEvalCacheHitsForTesting(), m0=CudaRbfEvalCacheMissesForTesting();
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> missOut=readback(co,5);
    assert(CudaRbfEvalCacheMissesForTesting()==m0+1 && CudaRbfEvalCacheHitsForTesting()==h0);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> hitOut=readback(co,5);
    assert(CudaRbfEvalCacheHitsForTesting()==h0+1);
    assert(sameBits(missOut,hitOut));
    // The seam-disabled uncached kernel is bitwise the same output, and
    // moves neither counter.
    TestDisableCudaRbfEvalCache(true);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(co,5),hitOut));
    assert(CudaRbfEvalCacheHitsForTesting()==h0+1 && CudaRbfEvalCacheMissesForTesting()==m0+1);
    TestDisableCudaRbfEvalCache(false);
    // Rebinding identical rest keeps the cache: still a hit, same bits.
    assert(cb.Bind({cr,5},0,s)==RbfStatus::Ok); assert(cb.Solve({cp,5},s)==RbfStatus::Ok);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(co,5),hitOut));
    assert(CudaRbfEvalCacheHitsForTesting()==h0+2);
    // A 1-ULP rest change misses and refills; the refill matches uncached.
    std::vector<float3> crest2=crest; crest2[0].x=std::nextafterf(crest2[0].x,2.f);
    check(cudaMemcpyAsync(cr,crest2.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(cb.Bind({cr,5},0,s)==RbfStatus::Ok); assert(cb.Solve({cp,5},s)==RbfStatus::Ok);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> refillOut=readback(co,5);
    assert(CudaRbfEvalCacheMissesForTesting()==m0+2);
    TestDisableCudaRbfEvalCache(true);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(co,5),refillOut));
    TestDisableCudaRbfEvalCache(false);
    // A 1-ULP CV change misses and refills; the refill matches uncached.
    check(cudaMemcpyAsync(cr,crest.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(cb.Bind({cr,5},0,s)==RbfStatus::Ok); assert(cb.Solve({cp,5},s)==RbfStatus::Ok);
    std::vector<float3> ccvs2=ccvs; ccvs2[3].z=std::nextafterf(ccvs2[3].z,-2.f);
    check(cudaMemcpyAsync(cc,ccvs2.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> cvsRefill=readback(co,5);
    assert(CudaRbfEvalCacheMissesForTesting()==m0+3);
    TestDisableCudaRbfEvalCache(true);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(co,5),cvsRefill));
    TestDisableCudaRbfEvalCache(false);
    // In-place evaluation through a warm cache matches uncached bitwise.
    check(cudaMemcpyAsync(cc,ccvs.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(cb.Evaluate({cc,5},{cc,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> inplace=readback(cc,5);
    assert(CudaRbfEvalCacheMissesForTesting()==m0+4);
    check(cudaMemcpyAsync(cc,ccvs.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    TestDisableCudaRbfEvalCache(true);
    assert(cb.Evaluate({cc,5},{cc,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(cc,5),inplace));
    TestDisableCudaRbfEvalCache(false);
    // A count change reshapes the cache; coming back restores the bits.
    float3 *cc8,*co8; check(cudaMalloc(&cc8,8*sizeof(float3))); check(cudaMalloc(&co8,8*sizeof(float3)));
    std::vector<float3> ccvs8(8); for(int i=0;i<8;i++) ccvs8[i]=f(.1f*float(i),.2f*float(i)+.05f,.3f*float(i)-.07f);
    check(cudaMemcpyAsync(cc8,ccvs8.data(),8*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(cb.Evaluate({cc8,8},{co8,8},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> wideOut=readback(co8,8);
    assert(CudaRbfEvalCacheMissesForTesting()==m0+5);
    TestDisableCudaRbfEvalCache(true);
    assert(cb.Evaluate({cc8,8},{co8,8},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(co8,8),wideOut));
    TestDisableCudaRbfEvalCache(false);
    check(cudaMemcpyAsync(cc,ccvs.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok); assert(cb.Finish(s)==RbfStatus::Ok);
    assert(sameBits(readback(co,5),hitOut));
    // A NaN CV refills (miss) and then hits over the NaN proof copy: the
    // hit-path evaluator raises the flag exactly like the direct kernel,
    // and the generation stays poisoned the same way.
    std::vector<float3> nanCvs=ccvs; nanCvs[1].y=NAN;
    check(cudaMemcpyAsync(cc,nanCvs.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    uint64_t h1=CudaRbfEvalCacheHitsForTesting(), m1=CudaRbfEvalCacheMissesForTesting();
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok);
    assert(CudaRbfEvalCacheMissesForTesting()==m1+1);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::Ok);
    assert(CudaRbfEvalCacheHitsForTesting()==h1+1);
    assert(cb.Finish(s)==RbfStatus::NonFiniteInput);
    assert(cb.Evaluate({cc,5},{co,5},s)==RbfStatus::InvalidArgument);
    cudaFree(cr); cudaFree(cp); cudaFree(cc); cudaFree(co); cudaFree(cc8); cudaFree(co8);
  }
  // Over the cache cap the fallback kernel runs: neither counter moves
  // and the identity field reproduces its inputs.
  {
    int const bn=1000, bp=200000;
    std::vector<float3> brest(size_t(bn), f(0,0,0));
    for(int i=0;i<bn;i++){ int x=i%10,y=(i/10)%10,z=i/100; brest[size_t(i)]=f(float(x)+.01f*float(i%7),float(y)+.01f*float((i+3)%7),float(z)+.01f*float((i+5)%7)); }
    float3 *bdr,*bdp,*bdc,*bdo;
    check(cudaMalloc(&bdr,size_t(bn)*sizeof(float3))); check(cudaMalloc(&bdp,size_t(bn)*sizeof(float3)));
    check(cudaMalloc(&bdc,size_t(bp)*sizeof(float3))); check(cudaMalloc(&bdo,size_t(bp)*sizeof(float3)));
    check(cudaMemcpyAsync(bdr,brest.data(),size_t(bn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(bdp,brest.data(),size_t(bn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    std::vector<float3> bcvs(size_t(bp), f(0,0,0));
    for(int i=0;i<bp;i++) bcvs[size_t(i)]=f(.01f*float(i%1000),.02f*float((i+7)%1000),.03f*float((i+13)%1000));
    check(cudaMemcpyAsync(bdc,bcvs.data(),size_t(bp)*sizeof(float3),cudaMemcpyHostToDevice,s));
    CudaRbfBinding bb; assert(bb.Bind({bdr,size_t(bn)},0,s)==RbfStatus::Ok); assert(bb.Solve({bdp,size_t(bn)},s)==RbfStatus::Ok);
    uint64_t h0=CudaRbfEvalCacheHitsForTesting(), m0=CudaRbfEvalCacheMissesForTesting();
    assert(bb.Evaluate({bdc,size_t(bp)},{bdo,size_t(bp)},s)==RbfStatus::Ok); assert(bb.Finish(s)==RbfStatus::Ok);
    assert(CudaRbfEvalCacheHitsForTesting()==h0 && CudaRbfEvalCacheMissesForTesting()==m0);
    std::vector<float3> bout(size_t(bp), f(0,0,0)); check(cudaMemcpyAsync(bout.data(),bdo,size_t(bp)*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s));
    for(int i=0;i<bp;i++){ assert(std::isfinite(bout[size_t(i)].x)&&std::isfinite(bout[size_t(i)].y)&&std::isfinite(bout[size_t(i)].z)); assert(near(bout[size_t(i)],bcvs[size_t(i)])); }
    cudaFree(bdr); cudaFree(bdp); cudaFree(bdc); cudaFree(bdo);
  }
  // Fresh-evaluate R cache: over the pair floor the fresh path verifies
  // into a device word and predicates its fill on it, so verify/fill/eval
  // submit as one stream slice with no host round-trip. The predicate is
  // device-side, so engagement is proven two ways: a fresh fill serves a
  // later direct Evaluate as a counted host-side hit, and every cached
  // fresh output is bitwise the seam-disabled uncached kernel. The direct
  // path counters never move on the fresh path.
  {
    int const fn=100, fc=12000;
    auto h01 = [&](uint64_t k){ k+=0x9e3779b97f4a7c15ULL; k=(k^(k>>30))*0xbf58476d1ce4e5b9ULL; k=(k^(k>>27))*0x94d049bb133111ebULL; return double((k^(k>>31))>>11)/double(1ull<<53); };
    std::vector<float3> frest(fn), fpose(fn), fcvs(fc);
    for(int i=0;i<fn;i++){ int x=i%5,y=(i/5)%5,z=(i/25)%4; frest[size_t(i)]=f(float(x)+.01f*float(i%7),float(y)+.01f*float((i+3)%7),float(z)+.01f*float((i+5)%7)); fpose[size_t(i)]=f(frest[size_t(i)].x+.1f*float(h01(uint64_t(i))-0.5),frest[size_t(i)].y+.1f*float(h01(uint64_t(i)+1000)-0.5),frest[size_t(i)].z+.1f*float(h01(uint64_t(i)+2000)-0.5)); }
    for(int i=0;i<fc;i++) fcvs[size_t(i)]=f(float(4*h01(uint64_t(i))-2),float(4*h01(uint64_t(i)+500000)-2),float(4*h01(uint64_t(i)+1000000)-2));
    float3 *fdr,*fdp,*fdc,*fdo; check(cudaMalloc(&fdr,size_t(fn)*sizeof(float3))); check(cudaMalloc(&fdp,size_t(fn)*sizeof(float3))); check(cudaMalloc(&fdc,size_t(fc)*sizeof(float3))); check(cudaMalloc(&fdo,size_t(fc)*sizeof(float3)));
    check(cudaMemcpyAsync(fdr,frest.data(),size_t(fn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(fdp,fpose.data(),size_t(fn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(fdc,fcvs.data(),size_t(fc)*sizeof(float3),cudaMemcpyHostToDevice,s));
    auto freadback = [&](float3* d){ std::vector<float3> v(fc); check(cudaMemcpyAsync(v.data(),d,size_t(fc)*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s)); return v; };
    auto fsameBits = [&](std::vector<float3> const& a, std::vector<float3> const& b){ return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float3))==0; };
    auto freshBind = [&](CudaRbfBinding& b, float3* rr){ assert(b.BeginFreshBind({rr,size_t(fn)},0,s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(b.CommitFreshBindExtent()==RbfStatus::Ok); assert(b.BeginFreshBindRank(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(b.CommitFreshBindRank()==RbfStatus::Ok); assert(b.BeginFreshBindLu(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(b.CommitFreshBindLu()==RbfStatus::Ok); };
    auto freshSolve = [&](CudaRbfBinding& b, float3* pp){ assert(b.BeginFreshSolve({pp,size_t(fn)},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(b.CommitFreshSolveInput()==RbfStatus::Ok); assert(b.BeginFreshSolveFactors(s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(b.CommitFreshSolve()==RbfStatus::Ok); };
    auto freshEval = [&](CudaRbfBinding& b){ assert(b.BeginFreshEvaluate({fdc,size_t(fc)},{fdo,size_t(fc)},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(b.CommitFreshEvaluate()==RbfStatus::Ok); };
    CudaRbfBinding fb; freshBind(fb,fdr); freshSolve(fb,fdp);
    uint64_t fh0=CudaRbfEvalCacheHitsForTesting(), fm0=CudaRbfEvalCacheMissesForTesting();
    freshEval(fb); std::vector<float3> fo0=freadback(fdo);
    freshEval(fb); std::vector<float3> fo1=freadback(fdo);
    assert(fsameBits(fo0,fo1));
    assert(CudaRbfEvalCacheHitsForTesting()==fh0 && CudaRbfEvalCacheMissesForTesting()==fm0);
    // The seam-disabled uncached kernel is bitwise the cached path.
    TestDisableCudaRbfEvalCache(true);
    freshEval(fb); assert(fsameBits(freadback(fdo),fo0));
    TestDisableCudaRbfEvalCache(false);
    // A 1-ULP CV change misses and refills; restoring re-hits the bits.
    std::vector<float3> fcvs2=fcvs; fcvs2[7].x=std::nextafterf(fcvs2[7].x,2.f);
    check(cudaMemcpyAsync(fdc,fcvs2.data(),size_t(fc)*sizeof(float3),cudaMemcpyHostToDevice,s));
    freshEval(fb); std::vector<float3> frefill=freadback(fdo);
    TestDisableCudaRbfEvalCache(true);
    freshEval(fb); assert(fsameBits(freadback(fdo),frefill));
    TestDisableCudaRbfEvalCache(false);
    check(cudaMemcpyAsync(fdc,fcvs.data(),size_t(fc)*sizeof(float3),cudaMemcpyHostToDevice,s));
    freshEval(fb); assert(fsameBits(freadback(fdo),fo0));
    // The fresh fill serves the direct path: a direct bind/solve over the
    // same samples evaluates as a counted host-side hit with the same bits.
    assert(fb.Bind({fdr,size_t(fn)},0,s)==RbfStatus::Ok); assert(fb.Solve({fdp,size_t(fn)},s)==RbfStatus::Ok);
    assert(fb.Evaluate({fdc,size_t(fc)},{fdo,size_t(fc)},s)==RbfStatus::Ok); assert(fb.Finish(s)==RbfStatus::Ok);
    assert(CudaRbfEvalCacheHitsForTesting()==fh0+1 && CudaRbfEvalCacheMissesForTesting()==fm0);
    assert(fsameBits(freadback(fdo),fo0));
    // A fresh rebind over moved rest misses, refills, and matches a clean
    // binding bitwise (new center/scale included).
    std::vector<float3> frest2=frest; for(int i=0;i<fn;i++) frest2[size_t(i)]=f(frest[size_t(i)].x+.5f,frest[size_t(i)].y-.25f,frest[size_t(i)].z+.125f);
    std::vector<float3> fpose2=fpose; for(int i=0;i<fn;i++) fpose2[size_t(i)]=f(fpose[size_t(i)].x+.5f,fpose[size_t(i)].y-.25f,fpose[size_t(i)].z+.125f);
    check(cudaMemcpyAsync(fdr,frest2.data(),size_t(fn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(fdp,fpose2.data(),size_t(fn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(fb.AcceptFreshSolve()==RbfStatus::Ok);
    freshBind(fb,fdr); freshSolve(fb,fdp); freshEval(fb); std::vector<float3> frebind=freadback(fdo);
    CudaRbfBinding fb2; freshBind(fb2,fdr); freshSolve(fb2,fdp);
    assert(fb2.BeginFreshEvaluate({fdc,size_t(fc)},{fdo,size_t(fc)},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s)); assert(fb2.CommitFreshEvaluate()==RbfStatus::Ok);
    assert(fsameBits(freadback(fdo),frebind));
    TestDisableCudaRbfEvalCache(true);
    freshEval(fb); assert(fsameBits(freadback(fdo),frebind));
    TestDisableCudaRbfEvalCache(false);
    // A NaN CV fails the flag proof exactly like the direct kernel, and
    // the binding recovers on the next stable pose.
    std::vector<float3> fnan=fcvs; fnan[3].z=NAN;
    check(cudaMemcpyAsync(fdc,fnan.data(),size_t(fc)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(fb.BeginFreshEvaluate({fdc,size_t(fc)},{fdo,size_t(fc)},s)==RbfStatus::Ok); check(cudaStreamSynchronize(s));
    assert(fb.CommitFreshEvaluate()==RbfStatus::NonFiniteInput);
    check(cudaMemcpyAsync(fdc,fcvs.data(),size_t(fc)*sizeof(float3),cudaMemcpyHostToDevice,s));
    freshEval(fb); assert(fsameBits(freadback(fdo),frebind));
    // Rollback restores the prior pose through the warm cache.
    check(cudaMemcpyAsync(fdp,fpose.data(),size_t(fn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(fb.AcceptFreshSolve()==RbfStatus::Ok);
    freshSolve(fb,fdp); freshEval(fb); std::vector<float3> frb=freadback(fdo);
    assert(fb.AcceptFreshSolve()==RbfStatus::Ok);
    check(cudaMemcpyAsync(fdp,fpose2.data(),size_t(fn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    freshSolve(fb,fdp); freshEval(fb);
    assert(fb.RollbackFreshSolve()==RbfStatus::Ok);
    freshEval(fb); assert(fsameBits(freadback(fdo),frb));
    assert(CudaRbfEvalCacheHitsForTesting()==fh0+1 && CudaRbfEvalCacheMissesForTesting()==fm0);
    cudaFree(fdr); cudaFree(fdp); cudaFree(fdc); cudaFree(fdo);
  }
  // Direct-bind memo: rebinding bitwise-identical rest skips the factor
  // slice (a counted hit); the solve+evaluate over the memo-hit binding
  // is bitwise the full-slice path. A 1-ULP rest change misses and
  // re-arms, a smoothing change skips the probe, the seam-disabled path
  // never probes, and a failed bind disarms the memo.
  {
    int const mn=100;
    auto mh01 = [&](uint64_t k){ k+=0x9e3779b97f4a7c15ULL; k=(k^(k>>30))*0xbf58476d1ce4e5b9ULL; k=(k^(k>>27))*0x94d049bb133111ebULL; return double((k^(k>>31))>>11)/double(1ull<<53); };
    std::vector<float3> mrest(mn), mpose(mn);
    for(int i=0;i<mn;i++){ int x=i%5,y=(i/5)%5,z=(i/25)%4; mrest[size_t(i)]=f(float(x)+.01f*float(i%7),float(y)+.01f*float((i+3)%7),float(z)+.01f*float((i+5)%7)); mpose[size_t(i)]=f(mrest[size_t(i)].x+.1f*float(mh01(uint64_t(i))-0.5),mrest[size_t(i)].y+.1f*float(mh01(uint64_t(i)+1000)-0.5),mrest[size_t(i)].z+.1f*float(mh01(uint64_t(i)+2000)-0.5)); }
    std::vector<float3> mcvs={f(.1f,.2f,.3f),f(.4f,.5f,.6f),f(.7f,.8f,.9f),f(1.1f,1.2f,1.3f),f(-.5f,.25f,2.f)};
    float3 *mr,*mp,*mc,*mo; check(cudaMalloc(&mr,size_t(mn)*sizeof(float3))); check(cudaMalloc(&mp,size_t(mn)*sizeof(float3))); check(cudaMalloc(&mc,5*sizeof(float3))); check(cudaMalloc(&mo,5*sizeof(float3)));
    check(cudaMemcpyAsync(mr,mrest.data(),size_t(mn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(mp,mpose.data(),size_t(mn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(mc,mcvs.data(),5*sizeof(float3),cudaMemcpyHostToDevice,s));
    auto mreadback = [&](){ std::vector<float3> v(5); check(cudaMemcpyAsync(v.data(),mo,5*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s)); return v; };
    auto msameBits = [&](std::vector<float3> const& a, std::vector<float3> const& b){ return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float3))==0; };
    CudaRbfBinding mb; assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok); assert(mb.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> mout0=mreadback();
    uint64_t bh0=CudaRbfBindMemoHitsForTesting(), bm0=CudaRbfBindMemoMissesForTesting();
    // Identical rebind on another stream: one probe, one hit, same bits.
    assert(mb.Bind({mr,size_t(mn)},0,s2)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+1 && CudaRbfBindMemoMissesForTesting()==bm0);
    assert(mb.Solve({mp,size_t(mn)},s2)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s2)==RbfStatus::Ok); assert(mb.Finish(s2)==RbfStatus::Ok);
    assert(msameBits(mreadback(),mout0));
    // A 1-ULP rest change misses; the miss path matches a clean binding
    // bitwise, and the rebind re-arms into a hit with the same bits.
    std::vector<float3> mrest2=mrest; mrest2[0].x=std::nextafterf(mrest2[0].x,2.f);
    check(cudaMemcpyAsync(mr,mrest2.data(),size_t(mn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+1 && CudaRbfBindMemoMissesForTesting()==bm0+1);
    assert(mb.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> mout2=mreadback();
    CudaRbfBinding mb2; assert(mb2.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok); assert(mb2.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb2.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb2.Finish(s)==RbfStatus::Ok);
    assert(msameBits(mreadback(),mout2));
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+2 && CudaRbfBindMemoMissesForTesting()==bm0+1);
    assert(mb.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb.Finish(s)==RbfStatus::Ok);
    assert(msameBits(mreadback(),mout2));
    // A smoothing change skips the probe (neither counter moves); the
    // rebound field still solves and evaluates.
    check(cudaMemcpyAsync(mr,mrest.data(),size_t(mn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(mb.Bind({mr,size_t(mn)},1e-6,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+2 && CudaRbfBindMemoMissesForTesting()==bm0+1);
    assert(mb.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb.Finish(s)==RbfStatus::Ok);
    // Seam-disabled binds never probe and match the memo path bitwise.
    TestDisableCudaRbfBindMemo(true);
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+2 && CudaRbfBindMemoMissesForTesting()==bm0+1);
    assert(mb.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb.Finish(s)==RbfStatus::Ok);
    assert(msameBits(mreadback(),mout0));
    TestDisableCudaRbfBindMemo(false);
    // Re-arm with a successful bind (a disarmed memo runs the full slice
    // without a probe and stores silently).
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+2 && CudaRbfBindMemoMissesForTesting()==bm0+1);
    // A failed bind disarms: the probe on the bad samples misses, the
    // recovery bind runs the full slice without a probe, and only the
    // rebind after it hits again.
    std::vector<float3> mnan=mrest; mnan[7].z=NAN;
    check(cudaMemcpyAsync(mr,mnan.data(),size_t(mn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::NonFiniteInput);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+2 && CudaRbfBindMemoMissesForTesting()==bm0+2);
    check(cudaMemcpyAsync(mr,mrest.data(),size_t(mn)*sizeof(float3),cudaMemcpyHostToDevice,s));
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+2 && CudaRbfBindMemoMissesForTesting()==bm0+2);
    assert(mb.Bind({mr,size_t(mn)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==bh0+3 && CudaRbfBindMemoMissesForTesting()==bm0+2);
    assert(mb.Solve({mp,size_t(mn)},s)==RbfStatus::Ok);
    assert(mb.Evaluate({mc,5},{mo,5},s)==RbfStatus::Ok); assert(mb.Finish(s)==RbfStatus::Ok);
    assert(msameBits(mreadback(),mout0));
    cudaFree(mr); cudaFree(mp); cudaFree(mc); cudaFree(mo);
  }
  // Bind-memo staging across shapes: a larger rebind grows the pinned
  // probe staging (no probe on a shape change), and the rebind after it
  // probes and hits; shrinking back re-arms the same way. All three
  // shapes solve and evaluate.
  {
    int const g0=64, g1=200;
    std::vector<float3> grest0(g0), grest1(g1);
    for(int i=0;i<g0;i++) grest0[size_t(i)]=f(float(i%8),float((i/8)%8),float(i/64)+.01f*float(i%5));
    for(int i=0;i<g1;i++) grest1[size_t(i)]=f(float(i%8),float((i/8)%8),float(i/64)+.01f*float(i%5));
    float3 *gr0,*gr1,*gp,*gc,*go;
    check(cudaMalloc(&gr0,size_t(g0)*sizeof(float3))); check(cudaMalloc(&gr1,size_t(g1)*sizeof(float3)));
    check(cudaMalloc(&gp,size_t(g1)*sizeof(float3))); check(cudaMalloc(&gc,4*sizeof(float3))); check(cudaMalloc(&go,4*sizeof(float3)));
    std::vector<float3> gcvs={f(.1f,.2f,.3f),f(.4f,.5f,.6f),f(.7f,.8f,.9f),f(1.1f,1.2f,1.3f)};
    check(cudaMemcpyAsync(gr0,grest0.data(),size_t(g0)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(gr1,grest1.data(),size_t(g1)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(gp,grest1.data(),size_t(g1)*sizeof(float3),cudaMemcpyHostToDevice,s));
    check(cudaMemcpyAsync(gc,gcvs.data(),4*sizeof(float3),cudaMemcpyHostToDevice,s));
    auto greadback = [&](int count){ std::vector<float3> v(count); check(cudaMemcpyAsync(v.data(),go,size_t(count)*sizeof(float3),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s)); return v; };
    CudaRbfBinding gb;
    uint64_t gh0=CudaRbfBindMemoHitsForTesting(), gm0=CudaRbfBindMemoMissesForTesting();
    assert(gb.Bind({gr0,size_t(g0)},0,s)==RbfStatus::Ok);
    assert(gb.Solve({gr0,size_t(g0)},s)==RbfStatus::Ok);
    assert(gb.Evaluate({gc,4},{go,4},s)==RbfStatus::Ok); assert(gb.Finish(s)==RbfStatus::Ok);
    std::vector<float3> gout0=greadback(4);
    assert(CudaRbfBindMemoHitsForTesting()==gh0 && CudaRbfBindMemoMissesForTesting()==gm0);
    // Grow past the staging: no probe (shape key mismatch), full slice.
    assert(gb.Bind({gr1,size_t(g1)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==gh0 && CudaRbfBindMemoMissesForTesting()==gm0);
    assert(gb.Solve({gp,size_t(g1)},s)==RbfStatus::Ok);
    assert(gb.Evaluate({gc,4},{go,4},s)==RbfStatus::Ok); assert(gb.Finish(s)==RbfStatus::Ok);
    // Identical rebind at the grown shape probes the grown staging: hit.
    assert(gb.Bind({gr1,size_t(g1)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==gh0+1 && CudaRbfBindMemoMissesForTesting()==gm0);
    assert(gb.Solve({gp,size_t(g1)},s)==RbfStatus::Ok);
    assert(gb.Evaluate({gc,4},{go,4},s)==RbfStatus::Ok); assert(gb.Finish(s)==RbfStatus::Ok);
    // Shrink back: no probe, then a hit again at the small shape.
    assert(gb.Bind({gr0,size_t(g0)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==gh0+1 && CudaRbfBindMemoMissesForTesting()==gm0);
    assert(gb.Bind({gr0,size_t(g0)},0,s)==RbfStatus::Ok);
    assert(CudaRbfBindMemoHitsForTesting()==gh0+2 && CudaRbfBindMemoMissesForTesting()==gm0);
    assert(gb.Solve({gr0,size_t(g0)},s)==RbfStatus::Ok);
    assert(gb.Evaluate({gc,4},{go,4},s)==RbfStatus::Ok); assert(gb.Finish(s)==RbfStatus::Ok);
    assert(greadback(4).size()==gout0.size() && std::memcmp(greadback(4).data(),gout0.data(),gout0.size()*sizeof(float3))==0);
    cudaFree(gr0); cudaFree(gr1); cudaFree(gp); cudaFree(gc); cudaFree(go);
  }
  cudaFree(dr);cudaFree(dp);cudaFree(do_);cudaStreamDestroy(s);cudaStreamDestroy(s2);
}
