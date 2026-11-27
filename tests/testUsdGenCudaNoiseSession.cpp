#include "utils/pxrInterop.h"
#include "pxr/base/gf/vec2f.h"
#include "testBackend.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/matrix4d.h"
#include "usdGen/exec/session.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/cuda/generation.h"
#include "usdGenCurves/cuda/cudaExecution.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <vector>
#include <future>

using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

static expr::ValueShape Scalar(expr::ScalarType type) { return {type, 1, 1, 1, 1, false}; }

static UsdGenExpressionBinding Bind(char const* expression, char const* destination,
                                    expr::Domain domain, expr::ScalarType type,
                                    char const* native, VtValue literal) {
    UsdGenExpressionBinding binding;
    binding.expression = SdfPath(expression);
    binding.destination = TfToken(destination);
    binding.domain = domain;
    binding.destinationShape = Scalar(type);
    binding.nativeType = TfToken(native);
    binding.literal = std::move(literal);
    return binding;
}

static UsdGenGraphDesc MakeDesc(bool withNoise) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Description");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    desc.defaultWidth = .02f;

    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.worldMatrix.SetIdentity();
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3};
    surface.faceVertexIndices = {0,1,2};
    desc.surfaces.push_back(surface);

    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Hair");
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.type = TfToken("cubic"); curves.basis = TfToken("bspline"); curves.wrap = TfToken("pinned");
    // Arrival order is deliberately not stable-ID order. ID 42 is the
    // source-sorted middle curve and has the short length that cull removes.
    curves.curveVertexCounts = {2,2,2};
    curves.curveId = {99,42,11};
    curves.points = {{9,0,0},{9,.5f,0}, {4,0,0},{4,.1f,0}, {1,0,0},{1,.5f,0}};
    curves.rest = {{90,0,0},{90,.5f,0}, {40,0,0},{40,.1f,0}, {10,0,0},{10,.5f,0}};
    curves.skinPrim = {0,0,0}; curves.skinPrimUv = {{.1f,.1f},{.2f,.2f},{.3f,.3f}};
    if (withNoise) for (int i = 0; i != 3; ++i) {
        GfMatrix4d frame(1.0);
        if (curves.curveId[i] == 42) {
            frame[0][0] = 0; frame[0][1] = 1;
            frame[1][0] = -1; frame[1][1] = 0;
        } else if (curves.curveId[i] == 99) {
            frame[0][0] = 0; frame[0][1] = 1; frame[0][2] = 0;
            frame[1][0] = 0; frame[1][1] = 0; frame[1][2] = 1;
            frame[2][0] = 1; frame[2][1] = 0; frame[2][2] = 0;
        }
        frame[3][0] = static_cast<double>(curves.curveId[i]);
        curves.rootFrame.push_back(frame);
    }
    desc.curveSets.push_back(curves);

    UsdGenNodeDesc source;
    source.path = SdfPath("/Ops/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path}; source.surfaces = {surface.path};
    source.params = {{TfToken("useRest"), VtValue(false), false}};
    UsdGenNodeDesc length;
    length.path = SdfPath("/Ops/Length"); length.type = TfToken("UsdGenLength"); length.inputs = {source.path};
    length.params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                     {TfToken("cullThreshold"), VtValue(.2f), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Ops/Width"); width.type = TfToken("UsdGenWidth");
    width.params = {{TfToken("width"), VtValue(.2f), false}};
    if (!withNoise) {
        width.inputs = {length.path}; desc.nodes = {source, length, width}; desc.terminal = width.path;
        return desc;
    }

    UsdGenNodeDesc noise;
    noise.path = SdfPath("/Ops/Noise"); noise.type = TfToken("UsdGenNoise"); noise.inputs = {length.path};
    noise.params = {{TfToken("enabled"), VtValue(true), false},
                    {TfToken("noise:frequency"), VtValue(1.0f), false}, {TfToken("noise:magnitude"), VtValue(.1f), false},
                    {TfToken("noise:seed"), VtValue(7), false}, {TfToken("noise:octaves"), VtValue(2), false}};
    width.inputs = {noise.path}; desc.nodes = {source, length, noise, width}; desc.terminal = width.path;

    UsdGenExpressionDesc frequency;
    frequency.path = SdfPath("/NoiseFrequency"); frequency.source = "$frame < 2 ? 1 : 3";
    frequency.outputs.push_back({TfToken("result"), TfToken("float"), Scalar(expr::ScalarType::Float32)});
    UsdGenExpressionDesc seed;
    seed.path = SdfPath("/NoiseSeed"); seed.source = "$primIndex + 7";
    seed.outputs.push_back({TfToken("result"), TfToken("int"), Scalar(expr::ScalarType::Int32)});
    UsdGenExpressionDesc magnitude;
    magnitude.path = SdfPath("/NoiseMagnitude"); magnitude.source = "$P[0] * 0 + .1";
    magnitude.outputs.push_back({TfToken("result"), TfToken("float"), Scalar(expr::ScalarType::Float32)});
    desc.expressions = {frequency, seed, magnitude};
    noise.expressionBindings.push_back(Bind("/NoiseFrequency", "noise:frequency", expr::Domain::Groom, expr::ScalarType::Float32, "float", VtValue(1.0f)));
    noise.expressionBindings.push_back(Bind("/NoiseSeed", "noise:seed", expr::Domain::Primitive, expr::ScalarType::Int32, "int", VtValue(7)));
    noise.expressionBindings.push_back(Bind("/NoiseMagnitude", "noise:magnitude", expr::Domain::Point, expr::ScalarType::Float32, "float", VtValue(.1f)));
    desc.nodes[2].expressionBindings = noise.expressionBindings;
    return desc;
}

int main() {
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    {
        // Staged Source -> Length(cull) -> Grow retains original frame IDs
        // while growing only the surviving curves.
        auto staged = MakeDesc(true);
        auto source = staged.nodes[0];
        auto length = staged.nodes[1];
        UsdGenNodeDesc grow; grow.type=TfToken("UsdGenGrow");
        grow.path=SdfPath("/Ops/Grow"); grow.inputs={length.path};
        grow.params={{TfToken("segments"),VtValue(6),false},{TfToken("length"),VtValue(2.0),false},
                     {TfToken("direction"),VtValue(TfToken("surfaceNormal")),false}};
        staged.nodes={source,length,grow}; staged.terminal=grow.path;
        UsdGenDiagnostics d; auto plan=CompileCudaGraph(usdGenUsd::Lower(staged),&d); CHECK(plan&&!d.HasErrors());
        auto ws=CreateCudaExecutionWorkspace(-1,&d); CHECK(ws); auto job=CreateCudaExecutionJob(plan,*ws,1,83,&d); CHECK(job);
        std::promise<bool> p; auto f=p.get_future(); CHECK(ExecuteCudaJobSourceAsync(job,[&](bool ok){p.set_value(ok);})&&f.wait_for(std::chrono::seconds(10))==std::future_status::ready&&f.get());
        for(size_t i=0;i<2;++i){std::promise<bool> q;auto r=q.get_future();CHECK(ExecuteCudaJobOperatorAsync(job,i,[&](bool ok){q.set_value(ok);})&&r.wait_for(std::chrono::seconds(10))==std::future_status::ready&&r.get());}
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> out;auto of=out.get_future();CHECK(FinalizeCudaExecutionJobAsync(job,[&](auto g){out.set_value(std::move(g));})&&of.wait_for(std::chrono::seconds(10))==std::future_status::ready);auto g=of.get();CHECK(g&&!d.HasErrors());
        auto l=gpu::AcquireGeometry(g,stream);CHECK(l&&l.Geometry().curveCount==2&&l.Geometry().pointCount==12&&l.RootT().size==2&&l.RootB().size==2&&l.RootN().size==2);
        std::vector<uint64_t> ids(2);CHECK(cudaMemcpyAsync(ids.data(),l.Geometry().stableIds.data,ids.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream)==cudaSuccess&&cudaStreamSynchronize(stream)==cudaSuccess&&ids==std::vector<uint64_t>({11,99}));
        std::vector<float3> t(2),b(2),n(2);
        CHECK(cudaMemcpyAsync(t.data(),l.RootT().data,2*sizeof(float3),cudaMemcpyDeviceToHost,stream)==cudaSuccess &&
              cudaMemcpyAsync(b.data(),l.RootB().data,2*sizeof(float3),cudaMemcpyDeviceToHost,stream)==cudaSuccess &&
              cudaMemcpyAsync(n.data(),l.RootN().data,2*sizeof(float3),cudaMemcpyDeviceToHost,stream)==cudaSuccess &&
              cudaStreamSynchronize(stream)==cudaSuccess);
        CHECK(t[0].x==1 && t[0].y==0 && t[0].z==0 &&
              b[0].x==0 && b[0].y==1 && b[0].z==0 &&
              n[0].x==0 && n[0].y==0 && n[0].z==1);
        CHECK(t[1].x==0 && t[1].y==1 && t[1].z==0 &&
              b[1].x==0 && b[1].y==0 && b[1].z==1 &&
              n[1].x==1 && n[1].y==0 && n[1].z==0);
    }
    {
        // Grow participates in the same serialized geometry lane as Length
        // and Noise when several direct-source branches are admitted.
        auto mixed = MakeDesc(true);
        auto source = mixed.nodes[0];
        UsdGenNodeDesc growA; growA.path=SdfPath("/Ops/GrowA"); growA.type=TfToken("UsdGenGrow");
        growA.inputs={source.path}; growA.seed=19;
        growA.params={{TfToken("segments"),VtValue(4),false},{TfToken("length"),VtValue(2.0),false},
                      {TfToken("lengthRandom"),VtValue(GfVec2f(.5f,1.5f)),false},
                      {TfToken("direction"),VtValue(TfToken("surfaceNormal")),false}};
        auto growB=growA; growB.path=SdfPath("/Ops/GrowB"); growB.seed=23;
        for(auto& parameter:growB.params) if(parameter.name==TfToken("segments"))
            parameter.value=VtValue(6);
        auto length=mixed.nodes[1]; length.path=SdfPath("/Ops/SourceLength"); length.inputs={source.path};
        auto noise=mixed.nodes[2]; noise.inputs={source.path};
        mixed.nodes={source,growA,growB,length,noise}; mixed.terminal=growB.path;
        UsdGenDiagnostics mixedDiagnostics; auto mixedPlan=CompileCudaGraph(usdGenUsd::Lower(mixed),&mixedDiagnostics);
        CHECK(mixedPlan&&!mixedDiagnostics.HasErrors());
        auto mixedWorkspace=CreateCudaExecutionWorkspace(-1,&mixedDiagnostics);CHECK(mixedWorkspace);
        auto mixedJob=CreateCudaExecutionJob(mixedPlan,*mixedWorkspace,1,82,&mixedDiagnostics);CHECK(mixedJob);
        std::promise<bool> sourceDone;auto sourceFuture=sourceDone.get_future();
        CHECK(ExecuteCudaJobSourceAsync(mixedJob,[&](bool ok){sourceDone.set_value(ok);})&&
              sourceFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready&&sourceFuture.get());
        armCudaOperatorAsyncContextCallbackGateForTesting();
        std::promise<bool> growADone;auto growAFuture=growADone.get_future();
        auto growALaunch=std::async(std::launch::async,[&]{return ExecuteCudaJobOperatorAsync(mixedJob,0,[&](bool ok){growADone.set_value(ok);});});
        struct ReleaseGrowGate { bool active=true; ~ReleaseGrowGate(){if(active)releaseCudaOperatorAsyncContextCallbackGateForTesting();} } releaseGrowGate;
        waitCudaOperatorAsyncContextCallbackGateForTesting();
        std::atomic<bool> growBCallback{false},lengthCallback{false},noiseCallback{false};
        CHECK(!ExecuteCudaJobOperatorAsync(mixedJob,1,[&](bool){growBCallback=true;}));
        CHECK(!ExecuteCudaJobOperatorAsync(mixedJob,2,[&](bool){lengthCallback=true;}));
        CHECK(!ExecuteCudaJobOperatorAsync(mixedJob,3,[&](bool){noiseCallback=true;}));
        releaseCudaOperatorAsyncContextCallbackGateForTesting();releaseGrowGate.active=false;
        CHECK(growALaunch.wait_for(std::chrono::seconds(10))==std::future_status::ready&&growALaunch.get()&&
              growAFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready&&growAFuture.get());
        for(size_t index=1;index!=4;++index){std::promise<bool> done;auto ready=done.get_future();
            CHECK(ExecuteCudaJobOperatorAsync(mixedJob,index,[&](bool ok){done.set_value(ok);})&&
                  ready.wait_for(std::chrono::seconds(10))==std::future_status::ready&&ready.get());}
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> finalPromise;auto finalFuture=finalPromise.get_future();
        CHECK(FinalizeCudaExecutionJobAsync(mixedJob,[&](auto result){finalPromise.set_value(std::move(result));})&&
              finalFuture.wait_for(std::chrono::seconds(10))==std::future_status::ready);
        auto final=finalFuture.get();CHECK(final&&!mixedDiagnostics.HasErrors()&&final->Geometry().elementCount==3&&
            final->Geometry().pointCount==18&&!growBCallback&&!lengthCallback&&!noiseCallback);
        auto finalLease=gpu::AcquireGeometry(final,stream); CHECK(finalLease);
        CHECK(finalLease.RootT().size==3&&finalLease.RootB().size==3&&finalLease.RootN().size==3);
    }
    {
        // Independent topology/value operators share one admitted geometry
        // lane: Length must exclude sibling Length and Noise work while held.
        auto mixed = MakeDesc(true);
        auto source = mixed.nodes[0];
        auto lengthA = mixed.nodes[1]; lengthA.path = SdfPath("/Ops/LengthA");
        auto lengthB = lengthA; lengthB.path = SdfPath("/Ops/LengthB");
        auto noise = mixed.nodes[2]; noise.inputs = {source.path};
        mixed.nodes = {source, lengthA, lengthB, noise}; mixed.terminal = noise.path;
        UsdGenDiagnostics mixedDiagnostics;
        auto mixedPlan = CompileCudaGraph(usdGenUsd::Lower(mixed), &mixedDiagnostics);
        CHECK(mixedPlan && !mixedDiagnostics.HasErrors());
        auto mixedWorkspace = CreateCudaExecutionWorkspace(-1, &mixedDiagnostics);
        CHECK(mixedWorkspace);
        auto mixedJob = CreateCudaExecutionJob(mixedPlan, *mixedWorkspace, 1, 80,
            &mixedDiagnostics);
        CHECK(mixedJob);
        std::promise<bool> sourceDone; auto sourceFuture = sourceDone.get_future();
        CHECK(ExecuteCudaJobSourceAsync(mixedJob, [&](bool ok) { sourceDone.set_value(ok); }));
        CHECK(sourceFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              sourceFuture.get());
        armCudaOperatorAsyncContextCallbackGateForTesting();
        std::promise<bool> lengthADone; auto lengthAFuture = lengthADone.get_future();
        auto lengthALaunch = std::async(std::launch::async, [&] {
            return ExecuteCudaJobOperatorAsync(mixedJob, 0,
                [&](bool ok) { lengthADone.set_value(ok); });
        });
        struct ReleaseMixedGate {
            bool active = true;
            ~ReleaseMixedGate() { if (active) releaseCudaOperatorAsyncContextCallbackGateForTesting(); }
        } releaseMixedGate;
        waitCudaOperatorAsyncContextCallbackGateForTesting();
        std::atomic<bool> lengthBCallback{false}, noiseCallback{false};
        CHECK(!ExecuteCudaJobOperatorAsync(mixedJob, 1, [&](bool) {
            lengthBCallback.store(true, std::memory_order_release);
        }));
        CHECK(!ExecuteCudaJobOperatorAsync(mixedJob, 2, [&](bool) {
            noiseCallback.store(true, std::memory_order_release);
        }));
        releaseCudaOperatorAsyncContextCallbackGateForTesting(); releaseMixedGate.active = false;
        CHECK(lengthALaunch.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              lengthALaunch.get() &&
              lengthAFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              lengthAFuture.get());
        std::promise<bool> lengthBDone; auto lengthBFuture = lengthBDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(mixedJob, 1, [&](bool ok) { lengthBDone.set_value(ok); }));
        CHECK(lengthBFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              lengthBFuture.get() && !lengthBCallback.load(std::memory_order_acquire));
        std::promise<bool> noiseDone; auto noiseFuture = noiseDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(mixedJob, 2, [&](bool ok) { noiseDone.set_value(ok); }));
        CHECK(noiseFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              noiseFuture.get() && !noiseCallback.load(std::memory_order_acquire));
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> mixedFinal;
        auto mixedFinalFuture = mixedFinal.get_future();
        CHECK(FinalizeCudaExecutionJobAsync(mixedJob, [&](auto result) {
            mixedFinal.set_value(std::move(result));
        }));
        CHECK(mixedFinalFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              mixedFinalFuture.get() && !mixedDiagnostics.HasErrors());
    }
    {
        // Reverse contention: a held Noise lane excludes a Length admission.
        auto mixed = MakeDesc(true);
        auto source = mixed.nodes[0];
        auto length = mixed.nodes[1]; length.path = SdfPath("/Ops/Length");
        auto noise = mixed.nodes[2]; noise.inputs = {source.path};
        mixed.nodes = {source, noise, length}; mixed.terminal = length.path;
        UsdGenDiagnostics mixedDiagnostics;
        auto mixedPlan = CompileCudaGraph(usdGenUsd::Lower(mixed), &mixedDiagnostics);
        CHECK(mixedPlan && !mixedDiagnostics.HasErrors());
        auto mixedWorkspace = CreateCudaExecutionWorkspace(-1, &mixedDiagnostics);
        CHECK(mixedWorkspace);
        auto mixedJob = CreateCudaExecutionJob(mixedPlan, *mixedWorkspace, 1, 81,
            &mixedDiagnostics);
        CHECK(mixedJob);
        std::promise<bool> sourceDone; auto sourceFuture = sourceDone.get_future();
        CHECK(ExecuteCudaJobSourceAsync(mixedJob, [&](bool ok) { sourceDone.set_value(ok); }));
        CHECK(sourceFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready && sourceFuture.get());
        armCudaOperatorAsyncContextCallbackGateForTesting();
        std::promise<bool> noiseDone; auto noiseFuture = noiseDone.get_future();
        auto noiseLaunch = std::async(std::launch::async, [&] {
            return ExecuteCudaJobOperatorAsync(mixedJob, 0,
                [&](bool ok) { noiseDone.set_value(ok); });
        });
        struct ReleaseReverseGate {
            bool active = true;
            ~ReleaseReverseGate() { if (active) releaseCudaOperatorAsyncContextCallbackGateForTesting(); }
        } releaseReverseGate;
        waitCudaOperatorAsyncContextCallbackGateForTesting();
        std::atomic<bool> lengthCallback{false};
        CHECK(!ExecuteCudaJobOperatorAsync(mixedJob, 1, [&](bool) {
            lengthCallback.store(true, std::memory_order_release);
        }));
        releaseCudaOperatorAsyncContextCallbackGateForTesting(); releaseReverseGate.active = false;
        CHECK(noiseLaunch.wait_for(std::chrono::seconds(10)) == std::future_status::ready && noiseLaunch.get() &&
              noiseFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready && noiseFuture.get());
        std::promise<bool> lengthDone; auto lengthFuture = lengthDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(mixedJob, 1, [&](bool ok) { lengthDone.set_value(ok); }));
        CHECK(lengthFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              lengthFuture.get() && !lengthCallback.load(std::memory_order_acquire));
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> mixedFinal;
        auto mixedFinalFuture = mixedFinal.get_future();
        CHECK(FinalizeCudaExecutionJobAsync(mixedJob, [&](auto result) { mixedFinal.set_value(std::move(result)); }));
        CHECK(mixedFinalFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              mixedFinalFuture.get() && !mixedDiagnostics.HasErrors());
    }
    {
        // A source-owned Width sibling must remain valid while a Length branch
        // commits its topology owner concurrently.
        auto branched = MakeDesc(false);
        auto source = branched.nodes[0];
        auto length = branched.nodes[1];
        auto width = branched.nodes[2];
        width.inputs = {source.path};
        branched.nodes = {source, length, width}; branched.terminal = width.path;
        UsdGenDiagnostics branchDiagnostics;
        auto branchPlan = CompileCudaGraph(usdGenUsd::Lower(branched), &branchDiagnostics);
        CHECK(branchPlan && !branchDiagnostics.HasErrors());
        auto branchWorkspace = CreateCudaExecutionWorkspace(-1, &branchDiagnostics);
        CHECK(branchWorkspace);
        auto branchJob = CreateCudaExecutionJob(branchPlan, *branchWorkspace,
            1, 79, &branchDiagnostics);
        CHECK(branchJob);
        std::promise<bool> branchSourceDone;
        auto branchSourceFuture = branchSourceDone.get_future();
        CHECK(ExecuteCudaJobSourceAsync(branchJob, [&](bool ok) {
            branchSourceDone.set_value(ok);
        }));
        CHECK(branchSourceFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              branchSourceFuture.get());
        armCudaOperatorAsyncWidthBranchGateForTesting(1);
        std::promise<bool> widthDone;
        auto widthFuture = widthDone.get_future();
        auto widthLaunch = std::async(std::launch::async, [&] {
            return ExecuteCudaJobOperatorAsync(branchJob, 1, [&](bool ok) {
                widthDone.set_value(ok);
            });
        });
        struct ReleaseWidthBranchGate {
            bool active = true;
            ~ReleaseWidthBranchGate() {
                if (active) releaseCudaOperatorAsyncWidthBranchGateForTesting();
            }
        } releaseWidthBranchGate;
        waitCudaOperatorAsyncWidthBranchGateForTesting();
        std::promise<bool> branchLengthDone;
        auto branchLengthFuture = branchLengthDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(branchJob, 0, [&](bool ok) {
            branchLengthDone.set_value(ok);
        }));
        CHECK(branchLengthFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              branchLengthFuture.get());
        releaseCudaOperatorAsyncWidthBranchGateForTesting();
        releaseWidthBranchGate.active = false;
        CHECK(widthLaunch.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              widthLaunch.get());
        CHECK(widthFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              widthFuture.get());
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> branchFinal;
        auto branchFinalFuture = branchFinal.get_future();
        CHECK(FinalizeCudaExecutionJobAsync(branchJob, [&](auto generation) {
            branchFinal.set_value(std::move(generation));
        }));
        auto final = branchFinalFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready
            ? branchFinalFuture.get() : std::shared_ptr<const UsdGenDeviceGeneration>{};
        CHECK(final && final->Geometry().elementCount == 3 && !branchDiagnostics.HasErrors());
    }
    {
        // Independent same-topology Noise siblings share frame-event state;
        // their operator admissions must serialize safely while one is held.
        auto sibling = MakeDesc(true);
        auto source = sibling.nodes[0];
        auto noiseA = sibling.nodes[2];
        noiseA.path = SdfPath("/Ops/NoiseA"); noiseA.inputs = {source.path};
        auto noiseB = noiseA;
        noiseB.path = SdfPath("/Ops/NoiseB");
        sibling.nodes = {source, noiseA, noiseB}; sibling.terminal = noiseB.path;
        UsdGenDiagnostics siblingDiagnostics;
        auto siblingPlan = CompileCudaGraph(usdGenUsd::Lower(sibling), &siblingDiagnostics);
        CHECK(siblingPlan && !siblingDiagnostics.HasErrors());
        auto siblingWorkspace = CreateCudaExecutionWorkspace(-1, &siblingDiagnostics);
        CHECK(siblingWorkspace);
        auto siblingJob = CreateCudaExecutionJob(siblingPlan, *siblingWorkspace,
            1, 77, &siblingDiagnostics);
        CHECK(siblingJob);
        std::promise<bool> sourceDone;
        auto sourceFuture = sourceDone.get_future();
        CHECK(ExecuteCudaJobSourceAsync(siblingJob, [&](bool ok) { sourceDone.set_value(ok); }));
        CHECK(sourceFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              sourceFuture.get());
        armCudaOperatorAsyncContextCallbackGateForTesting();
        std::promise<bool> noiseADone;
        auto noiseAFuture = noiseADone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(siblingJob, 0, [&](bool ok) { noiseADone.set_value(ok); }));
        waitCudaOperatorAsyncContextCallbackGateForTesting();
        std::atomic<bool> noiseBCallback{false};
        CHECK(!ExecuteCudaJobOperatorAsync(siblingJob, 1, [&](bool) {
            noiseBCallback.store(true, std::memory_order_release);
        }));
        releaseCudaOperatorAsyncContextCallbackGateForTesting();
        CHECK(noiseAFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              noiseAFuture.get());
        std::promise<bool> noiseBDone;
        auto noiseBFuture = noiseBDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(siblingJob, 1, [&](bool ok) { noiseBDone.set_value(ok); }));
        CHECK(noiseBFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              noiseBFuture.get() && !noiseBCallback.load(std::memory_order_acquire));
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> siblingFinal;
        auto finalFuture = siblingFinal.get_future();
        CHECK(FinalizeCudaExecutionJobAsync(siblingJob, [&](auto generation) {
            siblingFinal.set_value(std::move(generation));
        }));
        CHECK(finalFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              finalFuture.get() && !siblingDiagnostics.HasErrors() &&
              CudaExecutionJobOperatorCount(*siblingJob) == 2);
    }
    {
        // The same sibling guard must hold after a topology trunk has
        // published its compacted source value.
        auto sibling = MakeDesc(true);
        auto source = sibling.nodes[0];
        auto length = sibling.nodes[1];
        auto noiseA = sibling.nodes[2];
        noiseA.path = SdfPath("/Ops/NoiseA"); noiseA.inputs = {length.path};
        auto noiseB = noiseA;
        noiseB.path = SdfPath("/Ops/NoiseB");
        sibling.nodes = {source, length, noiseA, noiseB}; sibling.terminal = noiseB.path;
        UsdGenDiagnostics siblingDiagnostics;
        auto siblingPlan = CompileCudaGraph(usdGenUsd::Lower(sibling), &siblingDiagnostics);
        CHECK(siblingPlan && !siblingDiagnostics.HasErrors());
        auto siblingWorkspace = CreateCudaExecutionWorkspace(-1, &siblingDiagnostics);
        CHECK(siblingWorkspace);
        auto siblingJob = CreateCudaExecutionJob(siblingPlan, *siblingWorkspace,
            1, 78, &siblingDiagnostics);
        CHECK(siblingJob);
        std::promise<bool> sourceDone;
        auto sourceFuture = sourceDone.get_future();
        CHECK(ExecuteCudaJobSourceAsync(siblingJob, [&](bool ok) { sourceDone.set_value(ok); }));
        CHECK(sourceFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              sourceFuture.get());
        std::promise<bool> lengthDone;
        auto lengthFuture = lengthDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(siblingJob, 0, [&](bool ok) { lengthDone.set_value(ok); }));
        CHECK(lengthFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              lengthFuture.get());
        armCudaOperatorAsyncContextCallbackGateForTesting();
        std::promise<bool> noiseADone;
        auto noiseAFuture = noiseADone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(siblingJob, 1, [&](bool ok) { noiseADone.set_value(ok); }));
        waitCudaOperatorAsyncContextCallbackGateForTesting();
        std::atomic<bool> noiseBCallback{false};
        CHECK(!ExecuteCudaJobOperatorAsync(siblingJob, 2, [&](bool) {
            noiseBCallback.store(true, std::memory_order_release);
        }));
        releaseCudaOperatorAsyncContextCallbackGateForTesting();
        CHECK(noiseAFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              noiseAFuture.get());
        std::promise<bool> noiseBDone;
        auto noiseBFuture = noiseBDone.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(siblingJob, 2, [&](bool ok) { noiseBDone.set_value(ok); }));
        CHECK(noiseBFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
              noiseBFuture.get() && !noiseBCallback.load(std::memory_order_acquire));
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> siblingFinal;
        auto finalFuture = siblingFinal.get_future();
        CHECK(FinalizeCudaExecutionJobAsync(siblingJob, [&](auto generation) {
            siblingFinal.set_value(std::move(generation));
        }));
        auto final = finalFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready
            ? finalFuture.get() : std::shared_ptr<const UsdGenDeviceGeneration>{};
        CHECK(final && final->Geometry().elementCount == 2 && !siblingDiagnostics.HasErrors() &&
              CudaExecutionJobOperatorCount(*siblingJob) == 3);
    }
    {
        // Derived surface frames must drive the same native Noise calculation
        // as explicitly authored equivalent axes, including after Length cull.
        UsdGenSession framesSession;
        framesSession.SetDevicePublicationEnabled(true);
        auto derived = MakeDesc(true);
        derived.curveSets[0].rootFrame.clear();
        derived.curveSets[0].skinPrim.clear();
        derived.curveSets[0].skinPrimUv.clear();
        usdGenUsd::SetGraphDesc(framesSession, derived);
        auto generated = framesSession.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(generated && generated->device && !framesSession.LastDiagnostics().HasErrors());
        auto generatedLease = gpu::AcquireGeometry(generated->device, stream);
        CHECK(generatedLease && generatedLease.Geometry().pointCount == 4);
        std::vector<float3> derivedPoints(4), authoredPoints(4), retainedPoints(4);
        CHECK(cudaMemcpyAsync(derivedPoints.data(), generatedLease.Geometry().points.data,
            4*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess);
        auto authored = derived;
        authored.curveSets[0].rootFrame.assign(3, GfMatrix4d(1.0));
        usdGenUsd::SetGraphDesc(framesSession, authored);
        auto explicitGeneration = framesSession.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(explicitGeneration && explicitGeneration != generated &&
              !framesSession.LastDiagnostics().HasErrors());
        auto explicitLease = gpu::AcquireGeometry(explicitGeneration->device, stream);
        CHECK(explicitLease && explicitLease.Geometry().pointCount == 4);
        CHECK(cudaMemcpyAsync(authoredPoints.data(), explicitLease.Geometry().points.data,
            4*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(std::memcmp(authoredPoints.data(), derivedPoints.data(), 4*sizeof(float3)) == 0);
        // Missing real rest must not silently use posed points for Noise.
        auto missingRest = derived;
        missingRest.curveSets[0].rest.clear();
        usdGenUsd::SetGraphDesc(framesSession, missingRest);
        CHECK(framesSession.Commit(1, UsdGenCommitReason::SetTime) == explicitGeneration &&
              framesSession.LastDiagnostics().HasErrors());
        CHECK(cudaMemcpyAsync(retainedPoints.data(), generatedLease.Geometry().points.data,
            4*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(std::memcmp(retainedPoints.data(), derivedPoints.data(), 4*sizeof(float3)) == 0);
    }
    {
    UsdGenSession session;
    session.SetDevicePublicationEnabled(true);
    auto baseline = MakeDesc(false);
    usdGenUsd::SetGraphDesc(session, baseline);
    auto good = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(good && good->device && !session.LastDiagnostics().HasErrors());
    auto lease = gpu::AcquireGeometry(good->device, stream);
    CHECK(lease && lease.Geometry().curveCount == 2);
    std::vector<uint64_t> ids(2);
    CHECK(cudaMemcpyAsync(ids.data(), lease.Geometry().stableIds.data, ids.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(ids == std::vector<uint64_t>({11,99})); // ID 42 was the sorted-middle cull oracle.

    auto noise = MakeDesc(true);
    usdGenUsd::SetGraphDesc(session, noise);
    auto cooked = session.Commit(1, UsdGenCommitReason::SetTime);
    if (!cooked || cooked == good || !cooked->device ||
        session.LastDiagnostics().HasErrors())
        for (auto const& error : session.LastDiagnostics().errors)
            std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(cooked && cooked != good && cooked->device && !session.LastDiagnostics().HasErrors());
    auto cookedLease = gpu::AcquireGeometry(cooked->device, stream);
    CHECK(cookedLease && cookedLease.Geometry().curveCount == 2);
    std::vector<float3> cookedPoints(cookedLease.Geometry().pointCount);
    ids.resize(2);
    CHECK(cudaMemcpyAsync(cookedPoints.data(), cookedLease.Geometry().points.data,
                          cookedPoints.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaMemcpyAsync(ids.data(), cookedLease.Geometry().stableIds.data,
                          ids.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    // Root-frame gathering is by original IDs: Length removed 42, leaving
    // frames/points for 11 then 99 rather than source arrival order.
    CHECK(ids == std::vector<uint64_t>({11,99}) && cookedPoints.size() == 4);
    CHECK(std::memcmp(cookedPoints.data(),
                      std::vector<float3>{{1,0,0},{1,.5f,0},{9,0,0},{9,.5f,0}}.data(),
                      cookedPoints.size() * sizeof(float3)) != 0);

    // Editing the authored frame of the curve that Length removed must not
    // affect either survivor. This is the stable-ID lookup oracle; positional
    // indexing would incorrectly feed ID 42's edited frame to ID 99.
    auto culledFrameEdit = noise;
    culledFrameEdit.curveSets[0].rootFrame[1] = GfMatrix4d(1.0);
    usdGenUsd::SetGraphDesc(session, culledFrameEdit);
    auto culledFrameGeneration = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(culledFrameGeneration && !session.LastDiagnostics().HasErrors());
    auto culledFrameLease = gpu::AcquireGeometry(culledFrameGeneration->device, stream);
    std::vector<float3> culledFramePoints(culledFrameLease.Geometry().pointCount);
    CHECK(cudaMemcpyAsync(culledFramePoints.data(), culledFrameLease.Geometry().points.data,
                          culledFramePoints.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess &&
          std::memcmp(culledFramePoints.data(), cookedPoints.data(),
                      cookedPoints.size()*sizeof(float3)) == 0);

    // Editing ID 99's own frame must change only that survivor, while the old
    // generation and its device lease remain live as a COW snapshot.
    auto survivorFrameEdit = culledFrameEdit;
    survivorFrameEdit.curveSets[0].rootFrame[0] = GfMatrix4d(1.0);
    usdGenUsd::SetGraphDesc(session, survivorFrameEdit);
    auto survivorFrameGeneration = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(survivorFrameGeneration && !session.LastDiagnostics().HasErrors());
    auto survivorFrameLease = gpu::AcquireGeometry(survivorFrameGeneration->device, stream);
    std::vector<float3> survivorFramePoints(survivorFrameLease.Geometry().pointCount);
    CHECK(cudaMemcpyAsync(survivorFramePoints.data(), survivorFrameLease.Geometry().points.data,
                          survivorFramePoints.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(std::memcmp(survivorFramePoints.data(), cookedPoints.data(),
                      2*sizeof(float3)) == 0);
    CHECK(std::memcmp(survivorFramePoints.data() + 2, cookedPoints.data() + 2,
                      2*sizeof(float3)) != 0);

    usdGenUsd::SetGraphDesc(session, noise);
    auto frameTwo = session.Commit(2, UsdGenCommitReason::SetTime);
    CHECK(frameTwo && frameTwo != cooked && !session.LastDiagnostics().HasErrors());
    auto frameTwoLease = gpu::AcquireGeometry(frameTwo->device, stream);
    std::vector<float3> frameTwoPoints(frameTwoLease.Geometry().pointCount);
    CHECK(cudaMemcpyAsync(frameTwoPoints.data(), frameTwoLease.Geometry().points.data,
                          frameTwoPoints.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    bool frameChanged = false;
    for (size_t i = 0; i < cookedPoints.size(); ++i)
        frameChanged = frameChanged || frameTwoPoints[i].x != cookedPoints[i].x ||
            frameTwoPoints[i].y != cookedPoints[i].y || frameTwoPoints[i].z != cookedPoints[i].z;
    CHECK(frameChanged); // groom frequency is evaluated at execution time.

    std::vector<float3> const loaded{{1,0,0},{1,.5f,0},{9,0,0},{9,.5f,0}};
    auto disabled = noise;
    disabled.nodes[2].params[0].value = VtValue(false);
    usdGenUsd::SetGraphDesc(session, disabled);
    auto disabledGeneration = session.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(disabledGeneration && !session.LastDiagnostics().HasErrors());
    auto disabledLease = gpu::AcquireGeometry(disabledGeneration->device, stream);
    std::vector<float3> identityPoints(disabledLease.Geometry().pointCount);
    CHECK(cudaMemcpyAsync(identityPoints.data(), disabledLease.Geometry().points.data,
                          identityPoints.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess &&
          std::memcmp(identityPoints.data(), loaded.data(), loaded.size()*sizeof(float3)) == 0);
    auto blendZero = noise;
    blendZero.nodes[2].params.push_back({TfToken("mask"), VtValue(0.0f), false});
    usdGenUsd::SetGraphDesc(session, blendZero);
    auto identityGeneration = session.Commit(4, UsdGenCommitReason::SetTime);
    CHECK(identityGeneration && !session.LastDiagnostics().HasErrors());
    auto identityLease = gpu::AcquireGeometry(identityGeneration->device, stream);
    identityPoints.resize(identityLease.Geometry().pointCount);
    CHECK(cudaMemcpyAsync(identityPoints.data(), identityLease.Geometry().points.data,
                          identityPoints.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess &&
          std::memcmp(identityPoints.data(), loaded.data(), loaded.size()*sizeof(float3)) == 0);

    auto badType = noise;
    badType.nodes[2].expressionBindings[1].nativeType = TfToken("float");
    usdGenUsd::SetGraphDesc(session, badType);
    CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == identityGeneration &&
          session.LastDiagnostics().HasErrors());
    lease = {}; cookedLease = {}; culledFrameLease = {}; survivorFrameLease = {};
    frameTwoLease = {}; disabledLease = {}; identityLease = {};
    }
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("testUsdGenCudaNoiseSession: PASS");
    return 0;
}
