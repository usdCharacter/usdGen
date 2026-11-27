#include "utils/pxrInterop.h"
#include "cudaExecutionQueueFixture.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/exec/executionPipeline.h"
#include "usdGen/exec/device/executionResources.h"
#include "usdGenCurves/cuda/generation.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static UsdGenGraphDesc Desc(float translation = 1.f) {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/Hair"); d.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenSurfaceDesc s; s.path = SdfPath("/Scalp");
    s.restPoints = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
    s.points = s.restPoints; for (auto& p : s.points) p[0] += translation;
    s.faceVertexCounts = {3,3,3}; s.faceVertexIndices = {0,1,2, 0,1,3, 1,2,4};
    d.surfaces.push_back(s);
    UsdGenCurveSetDesc c; c.path = SdfPath("/Curves"); c.role = UsdGenLane::Chunked;
    c.curveRole = TfToken("hair"); c.curveVertexCounts = {2};
    c.points = {{.2f,.2f,0},{.2f,.4f,0}}; c.rest = c.points;
    c.curveId = {42}; c.skinPrim = {0}; c.skinPrimUv = {{.2f,.2f}};
    d.curveSets.push_back(c);
    UsdGenNodeDesc source; source.path = SdfPath("/Groom/Hair/Ops/Source");
    source.type = TfToken("UsdGenCurveSource"); source.curves = {c.path}; source.surfaces = {s.path};
    UsdGenNodeDesc deform; deform.path = SdfPath("/Groom/Hair/Ops/Deform");
    deform.type = TfToken("UsdGenDeform"); deform.inputs = {source.path}; deform.surfaces = {s.path};
    deform.params.push_back({TfToken("rbfSamples"), VtValue(5), false});
    UsdGenExpressionDesc e; e.path = SdfPath("/Blend"); e.source = "$frame == 1 ? 0.25 : 1";
    const expr::ValueShape shape{expr::ScalarType::Float32,1,1,1,1,false};
    e.outputs.push_back({TfToken("result"), TfToken("float"), shape}); d.expressions.push_back(e);
    UsdGenExpressionBinding b; b.expression = e.path; b.destination = TfToken("mask");
    b.domain = expr::Domain::Point; b.nativeType = TfToken("float"); b.destinationShape = shape;
    b.literal = VtValue(1.f); deform.expressionBindings.push_back(b);
    d.nodes = {source,deform}; d.terminal = deform.path;
    return d;
}

static UsdGenGraphDesc WidthDesc() {
    auto desc = Desc();
    UsdGenNodeDesc width;
    width.path = SdfPath("/Groom/Hair/Ops/Width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {desc.nodes.front().path};
    width.params.push_back({TfToken("width"), VtValue(.03f), false});
    desc.nodes = {desc.nodes.front(), std::move(width)};
    desc.terminal = desc.nodes.back().path;
    return desc;
}

static UsdGenGraphDesc LengthDesc() {
    auto desc = Desc();
    UsdGenAuthoredPlaneDesc density;
    density.name = TfToken("density");
    density.type = UsdGenAuthoredPlaneType::Float32;
    density.domain = UsdGenAuthoredPlaneDomain::Point;
    density.arity = 1;
    density.floatValues = {1.f, 2.f};
    desc.curveSets.front().authoredPlanes = {std::move(density)};
    UsdGenNodeDesc length;
    length.path = SdfPath("/Groom/Hair/Ops/Length");
    length.type = TfToken("UsdGenLength");
    length.inputs = {desc.nodes.front().path};
    length.params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                     {TfToken("length:value"), VtValue(.5f), false}};
    desc.nodes = {desc.nodes.front(), std::move(length)};
    desc.terminal = desc.nodes.back().path;
    return desc;
}

static bool Root(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                 float x, cudaStream_t stream) {
    auto lease = gpu::AcquireGeometry(generation,stream);
    if (!lease || lease.Geometry().pointCount != 2) return false;
    float3 point{};
    if (cudaMemcpyAsync(&point,lease.Geometry().points.data,sizeof(point),cudaMemcpyDeviceToHost,stream) != cudaSuccess)
        return false;
    return cudaStreamSynchronize(stream) == cudaSuccess && std::fabs(point.x-x) < 2e-3f &&
        std::fabs(point.y-.2f) < 2e-3f && std::fabs(point.z) < 2e-3f;
}

static bool Wait(std::atomic<bool> const& value) {
    auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (!value.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
        std::this_thread::yield();
    return value.load(std::memory_order_acquire);
}

static size_t ResourceKindBytes(UsdGenExecutionResourceSnapshot const& snapshot,
                                UsdGenExecutionResourceKind kind) {
    return snapshot.byKind[static_cast<size_t>(kind)];
}

struct ReleaseSourceGate {
    ~ReleaseSourceGate() { releaseCudaSourceAsyncCallbackGateForTesting(); }
};
struct ReleaseResampleGate {
    ~ReleaseResampleGate() { releaseCudaSourceAsyncResampleCallbackGateForTesting(); }
};
struct ReleaseOperatorGate {
    void (*release)() = nullptr;
    ~ReleaseOperatorGate() { if (release) release(); }
};
struct ResetFinalizationRelayCapacity {
    ~ResetFinalizationRelayCapacity() { setCudaFinalizationRelayCapacityForTesting(1024); }
};

int main() {
    UsdGenDiagnostics diagnostics;
    auto desc = Desc(); auto plan = CompileCudaGraph(usdGenUsd::Lower(desc),&diagnostics);
    CHECK(plan && !diagnostics.HasErrors());
    auto planMetadata = GetCudaExecutionPlanMetadata(*plan);
    // Length/RBF has only partial compile-time byte accounting (runtime CUB
    // and cuSolver workspaces remain unknown), so queue admission must retain
    // the legacy allocation path rather than inventing a reservation.
    CHECK(planMetadata && !planMetadata->MemoryEstimate().memoryAvailable);
    // Plan owns the authored snapshot; later edits to the caller's descriptor
    // cannot change its literal/pose inputs or compiled expression programs.
    desc.surfaces[0].points[0][0] = 100;
    desc.expressions[0].source = "0";
    UsdGenExecutionRuntime runtime(4);
    cudaStream_t reader = nullptr;
    auto const readerStatus = cudaStreamCreateWithFlags(&reader,cudaStreamNonBlocking);
    if (readerStatus != cudaSuccess)
        std::fprintf(stderr, "cudaStreamCreateWithFlags: %s (%s)\n",
            cudaGetErrorName(readerStatus), cudaGetErrorString(readerStatus));
    CHECK(readerStatus == cudaSuccess);
    // Source upload is asynchronous: a stream gate after Set holds the CUDA
    // callback, but the one-worker runtime must be free for unrelated CPU
    // pipeline work before the source task's terminal completion arrives.
    {
        UsdGenExecutionRuntime oneWorker(1);
        UsdGenCudaExecutionQueue held(oneWorker,-1,&diagnostics); CHECK(held.Valid());
        UsdGenExecutionPipeline probe(oneWorker);
        std::atomic<int> sourceCalls{0}; std::atomic<bool> sourceDone{false};
        armCudaSourceAsyncCallbackGateForTesting(); ReleaseSourceGate releaseGate;
        CHECK(held.Submit(plan,1,[&](uint64_t, auto, auto const&) {
            sourceCalls.fetch_add(1,std::memory_order_relaxed);
            sourceDone.store(true,std::memory_order_release);
        }));
        waitCudaSourceAsyncCallbackGateForTesting();
        std::atomic<bool> probeRan{false};
        CHECK(probe.Submit([&](auto const&) {
            probeRan.store(true,std::memory_order_release); return UsdGenExecutionPipeline::Publish{};
        }));
        CHECK(Wait(probeRan) && !sourceDone.load(std::memory_order_acquire) && sourceCalls == 0);
        releaseCudaSourceAsyncCallbackGateForTesting();
        held.Drain(); probe.Drain();
        CHECK(Wait(sourceDone) && sourceCalls == 1 && held.Snapshot() &&
            Root(held.Snapshot()->generation,.45f,reader));

        // Source controls have three additional asynchronous proofs before
        // upload.  Each held phase must release the sole runtime worker and
        // leave the public snapshot unchanged until its terminal callback.
        auto controlsDesc=Desc(); controlsDesc.nodes.resize(1); controlsDesc.terminal=controlsDesc.nodes.front().path;
        auto& controlCurves=controlsDesc.curveSets.front();
        controlCurves.points={{2.2f,.2f,0},{2.2f,.4f,0}}; controlCurves.rest={{.2f,.2f,0},{.2f,.4f,0}};
        auto& controlNode=controlsDesc.nodes.front();
        controlNode.params.push_back({TfToken("useRest"),VtValue(false),false});
        controlNode.params.push_back({TfToken("resampleTo"),VtValue(-1),false});
        auto addControl=[&](char const* path, char const* expression, char const* destination,
                            TfToken native, expr::ScalarType scalar, VtValue literal) {
            expr::ValueShape shape{scalar,1,1,1,1,false}; UsdGenExpressionDesc e;
            e.path=SdfPath(path); e.source=expression; e.outputs.push_back({TfToken("result"),native,shape});
            controlsDesc.expressions.push_back(std::move(e)); UsdGenExpressionBinding binding;
            binding.expression=SdfPath(path); binding.destination=TfToken(destination); binding.domain=expr::Domain::Groom;
            binding.nativeType=native; binding.destinationShape=shape; binding.literal=std::move(literal);
            controlNode.expressionBindings.push_back(std::move(binding));
        };
        addControl("/QueueSourceUseRest","$frame < 3","useRest",TfToken("bool"),expr::ScalarType::Bool,VtValue(true));
        addControl("/QueueSourceResample","$frame < 2 ? 0 : 4","resampleTo",TfToken("int"),expr::ScalarType::Int32,VtValue(0));
        auto controlsPlan=CompileCudaGraph(usdGenUsd::Lower(controlsDesc),&diagnostics); CHECK(controlsPlan);
        struct SourceGate { void (*arm)(); void (*wait)(); void (*release)(); };
        SourceGate const controlGates[] = {
            {armCudaSourceAsyncContextCallbackGateForTesting,waitCudaSourceAsyncContextCallbackGateForTesting,releaseCudaSourceAsyncContextCallbackGateForTesting},
            {armCudaSourceAsyncProgramCallbackGateForTesting,waitCudaSourceAsyncProgramCallbackGateForTesting,releaseCudaSourceAsyncProgramCallbackGateForTesting},
            {armCudaSourceAsyncScalarCallbackGateForTesting,waitCudaSourceAsyncScalarCallbackGateForTesting,releaseCudaSourceAsyncScalarCallbackGateForTesting},
        };
        for (auto const& gate : controlGates) {
            auto const lastGood=held.Snapshot(); std::atomic<bool> done{false}, probeDone{false};
            gate.arm(); ReleaseOperatorGate release{gate.release};
            CHECK(held.Submit(controlsPlan,1,[&](uint64_t, auto outcome, auto const&) {
                done.store(outcome==UsdGenExecutionPipeline::Outcome::Published,std::memory_order_release);
            }));
            gate.wait();
            CHECK(probe.Submit([&](auto const&) { probeDone.store(true,std::memory_order_release); return UsdGenExecutionPipeline::Publish{}; }));
            CHECK(Wait(probeDone) && !done.load(std::memory_order_acquire) && held.Snapshot()==lastGood);
            gate.release(); release.release=nullptr; held.Drain(); probe.Drain();
            CHECK(Wait(done) && held.Snapshot()!=lastGood && cudaSourceRelayOccupiedCountForTesting()==0);
        }
        // A terminally proven control semantic error publishes no candidate;
        // the queue retains last-good and the same workspace admits a valid
        // source-control retry.
        auto invalidControls=controlsDesc;
        for (auto& expression : invalidControls.expressions)
            if (expression.path==SdfPath("/QueueSourceResample")) expression.source="$frame / 0";
        auto invalidControlsPlan=CompileCudaGraph(usdGenUsd::Lower(invalidControls),&diagnostics); CHECK(invalidControlsPlan);
        auto const controlsLastGood=held.Snapshot(); std::atomic<int> invalidCalls{0}; std::atomic<bool> invalidDone{false};
        CHECK(held.Submit(invalidControlsPlan,1,[&](uint64_t, auto outcome, auto const& d) {
            if (outcome==UsdGenExecutionPipeline::Outcome::Failed && d.HasErrors()) ++invalidCalls;
            invalidDone.store(true,std::memory_order_release);
        }));
        held.Drain(); CHECK(Wait(invalidDone) && invalidCalls==1 && held.Snapshot()==controlsLastGood &&
            cudaSourceRelayOccupiedCountForTesting()==0);
        std::atomic<bool> controlsRetryDone{false};
        CHECK(held.Submit(controlsPlan,1,[&](uint64_t, auto outcome, auto const&) {
            controlsRetryDone.store(outcome==UsdGenExecutionPipeline::Outcome::Published,std::memory_order_release);
        }));
        held.Drain(); CHECK(Wait(controlsRetryDone) && held.Snapshot()!=controlsLastGood);

        auto resampleDesc=Desc();
        resampleDesc.nodes.front().params.push_back({TfToken("resampleTo"),VtValue(4),false});
        auto resamplePlan=CompileCudaGraph(usdGenUsd::Lower(resampleDesc),&diagnostics); CHECK(resamplePlan);
        std::atomic<int> resampleCalls{0}, resampleCompletionCalls{0}, resampleOutcome{-1};
        std::atomic<bool> resampleDone{false};
        armCudaSourceAsyncResampleCallbackGateForTesting(); ReleaseResampleGate releaseResampleGate;
        CHECK(held.Submit(resamplePlan,2,[&](uint64_t, auto outcome, auto const&) {
            resampleOutcome.store(int(outcome),std::memory_order_release);
            resampleCompletionCalls.fetch_add(1,std::memory_order_relaxed);
            if (outcome == UsdGenExecutionPipeline::Outcome::Published) ++resampleCalls;
            resampleDone.store(true,std::memory_order_release);
        }));
        waitCudaSourceAsyncResampleCallbackGateForTesting();
        std::atomic<bool> resampleProbeRan{false};
        bool const resampleProbeAccepted=probe.Submit([&](auto const&) {
            resampleProbeRan.store(true,std::memory_order_release); return UsdGenExecutionPipeline::Publish{};
        });
        CHECK(resampleProbeAccepted);
        auto const resampleProbeStart=std::chrono::steady_clock::now();
        bool const resampleProbeCompleted=Wait(resampleProbeRan);
        auto const resampleProbeElapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now()-resampleProbeStart).count();
        bool const resampleDoneBeforeRelease=resampleDone.load(std::memory_order_acquire);
        int const resamplePublishedBeforeRelease=resampleCalls.load(std::memory_order_acquire);
        int const resampleCompletionsBeforeRelease=resampleCompletionCalls.load(std::memory_order_acquire);
        int const resampleOutcomeBeforeRelease=resampleDoneBeforeRelease ?
            resampleOutcome.load(std::memory_order_acquire) : -1;
        size_t const resampleOccupiedBeforeRelease=cudaSourceRelayOccupiedCountForTesting();
        size_t const resampleQuarantinedBeforeRelease=cudaSourceRelayQuarantinedCountForTesting();
        if (!resampleProbeCompleted || resampleDoneBeforeRelease || resamplePublishedBeforeRelease != 0) {
            std::fprintf(stderr,
                "resample probe: accepted=%d ran=%d elapsedMs=%lld done=%d completions=%d published=%d outcome=%d sourceOccupied=%zu sourceQuarantined=%zu\n",
                resampleProbeAccepted,resampleProbeCompleted,static_cast<long long>(resampleProbeElapsed),
                resampleDoneBeforeRelease,resampleCompletionsBeforeRelease,resamplePublishedBeforeRelease,
                resampleOutcomeBeforeRelease,resampleOccupiedBeforeRelease,resampleQuarantinedBeforeRelease);
        }
        CHECK(resampleProbeCompleted && !resampleDoneBeforeRelease && resamplePublishedBeforeRelease == 0);
        releaseCudaSourceAsyncResampleCallbackGateForTesting();
        held.Drain(); probe.Drain();
        CHECK(Wait(resampleDone) && resampleCalls == 1 && held.Snapshot());

        auto const beforeResampleCancel=held.Snapshot();
        std::atomic<int> resampleCancelCalls{0}; std::atomic<bool> resampleCancelDone{false};
        UsdGenExecutionPipeline::Outcome resampleCancelOutcome=UsdGenExecutionPipeline::Outcome::Published;
        armCudaSourceAsyncResampleCallbackGateForTesting();
        CHECK(held.Submit(resamplePlan,3,[&](uint64_t, auto outcome, auto const&) {
            resampleCancelOutcome=outcome; ++resampleCancelCalls;
            resampleCancelDone.store(true,std::memory_order_release);
        }));
        waitCudaSourceAsyncResampleCallbackGateForTesting();
        CHECK(held.CancelPending());
        releaseCudaSourceAsyncResampleCallbackGateForTesting();
        held.Drain();
        CHECK(Wait(resampleCancelDone) && resampleCancelCalls == 1 &&
            resampleCancelOutcome == UsdGenExecutionPipeline::Outcome::Superseded &&
            held.Snapshot() == beforeResampleCancel &&
            cudaSourceRelayOccupiedCountForTesting() == 0);

        auto const lastGood=held.Snapshot();
        std::atomic<int> cancelledCalls{0};
        std::atomic<bool> cancelledDone{false};
        UsdGenExecutionPipeline::Outcome cancelledOutcome=UsdGenExecutionPipeline::Outcome::Published;
        armCudaSourceAsyncCallbackGateForTesting();
        CHECK(held.Submit(plan,2,[&](uint64_t, auto outcome, auto const&) {
            cancelledOutcome=outcome; cancelledCalls.fetch_add(1,std::memory_order_relaxed);
            cancelledDone.store(true,std::memory_order_release);
        }));
        waitCudaSourceAsyncCallbackGateForTesting();
        CHECK(held.CancelPending());
        releaseCudaSourceAsyncCallbackGateForTesting();
        held.Drain();
        CHECK(Wait(cancelledDone) && cancelledCalls == 1 &&
            cancelledOutcome == UsdGenExecutionPipeline::Outcome::Superseded &&
            held.Snapshot() == lastGood);

        // Callback installation failure is still an accepted terminal queue
        // outcome: the last good snapshot survives and the poisoned workspace
        // rejects a later request rather than reusing unproven source buffers.
        std::atomic<int> installCalls{0}; std::atomic<bool> installDone{false};
        failNextCudaSourceRelayCallbackInstallForTesting();
        CHECK(held.Submit(plan,3,[&](uint64_t, auto outcome, auto const& d) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Failed && d.HasErrors()) ++installCalls;
            installDone.store(true,std::memory_order_release);
        }));
        held.Drain();
        CHECK(Wait(installDone) && installCalls == 1 && held.Snapshot() == lastGood);
        std::atomic<bool> poisonedRejectDone{false};
        CHECK(held.Submit(plan,4,[&](uint64_t, auto outcome, auto const&) {
            poisonedRejectDone.store(outcome == UsdGenExecutionPipeline::Outcome::Failed,
                                     std::memory_order_release);
        }));
        held.Drain();
        CHECK(Wait(poisonedRejectDone) && held.Snapshot() == lastGood);
    }
    // Width has three terminal async phases (context, program, and kernel).
    // Holding each native callback must free the sole runtime worker for
    // unrelated CPU work without publishing the candidate early.
    {
        auto widthPlan = CompileCudaGraph(usdGenUsd::Lower(WidthDesc()), &diagnostics); CHECK(widthPlan);
        UsdGenExecutionRuntime oneWorker(1);
        UsdGenCudaExecutionQueue widthQueue(oneWorker, -1, &diagnostics); CHECK(widthQueue.Valid());
        UsdGenExecutionPipeline probe(oneWorker);
        std::atomic<bool> initialDone{false};
        CHECK(widthQueue.Submit(widthPlan, 1, [&](uint64_t, auto outcome, auto const&) {
            initialDone.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                              std::memory_order_release);
        }));
        widthQueue.Drain(); CHECK(Wait(initialDone) && widthQueue.Snapshot());

        // Known Source/Width metadata is admitted as one precharged graph
        // peak before any candidate buffer allocation. Leave just one byte
        // less than that peak available: individual source allocations would
        // fit, but the job ticket must reject and preserve last-good output.
        auto widthMetadata = GetCudaExecutionPlanMetadata(*widthPlan);
        CHECK(widthMetadata && widthMetadata->MemoryEstimate().memoryAvailable &&
              widthMetadata->MemoryEstimate().conservativeUpperBound);
        auto const graphPeak = widthMetadata->MemoryEstimate().concurrentPeakBytes;
        CHECK(graphPeak > widthMetadata->Tasks().front().estimate.retainedOutputBytes);
        int resourceDevice = -1;
        CHECK(cudaGetDevice(&resourceDevice) == cudaSuccess);
        auto resourcePool = FindUsdGenExecutionResourcePool(
            {usdGenTestUtils::Backend("cuda"), resourceDevice});
        CHECK(resourcePool);
        auto resourceBaseline = resourcePool->Snapshot();
        CHECK(resourceBaseline.usedBytes <= resourceBaseline.usableBytes);
        auto const available = resourceBaseline.usableBytes - resourceBaseline.usedBytes;
        CHECK(available > graphPeak);
        auto filler = resourcePool->TryReserve(
            available - graphPeak + 1, UsdGenExecutionResourceKind::Scratch);
        CHECK(filler);
        auto const beforeBudgetReject = resourcePool->Snapshot();
        auto const lastGoodBeforeBudgetReject = widthQueue.Snapshot();
        std::atomic<bool> budgetRejected{false};
        CHECK(widthQueue.Submit(widthPlan, 1.5,
            [&](uint64_t, auto outcome, auto const& failure) {
                budgetRejected.store(
                    outcome == UsdGenExecutionPipeline::Outcome::Failed &&
                        failure.errors.size() == 1 &&
                        failure.errors.front() ==
                            "CUDA: CUDA execution job memory reservation exceeds the available device budget",
                    std::memory_order_release);
            }));
        widthQueue.Drain();
        CHECK(Wait(budgetRejected) && widthQueue.Snapshot() == lastGoodBeforeBudgetReject &&
              resourcePool->Snapshot().usedBytes == beforeBudgetReject.usedBytes &&
              ResourceKindBytes(resourcePool->Snapshot(),
                  UsdGenExecutionResourceKind::Pending) ==
                  ResourceKindBytes(beforeBudgetReject,
                      UsdGenExecutionResourceKind::Pending));
        filler->Release();
        CHECK(resourcePool->Snapshot().usedBytes == resourceBaseline.usedBytes);

        // A successful gated source stage consumes child permits from the one
        // reservation without double-charging the pool total. Pending may be
        // partially transferred to active child buffers, but the total delta
        // remains exactly the graph peak.
        auto const beforeSuccessfulJob = resourcePool->Snapshot();
        std::atomic<bool> successfulDone{false};
        armCudaSourceAsyncCallbackGateForTesting();
        ReleaseSourceGate successfulSourceGate;
        CHECK(widthQueue.Submit(widthPlan, 1.75,
            [&](uint64_t, auto outcome, auto const&) {
                successfulDone.store(
                    outcome == UsdGenExecutionPipeline::Outcome::Published,
                    std::memory_order_release);
            }));
        waitCudaSourceAsyncCallbackGateForTesting();
        auto const duringSuccessfulSource = resourcePool->Snapshot();
        CHECK(duringSuccessfulSource.usedBytes ==
                  beforeSuccessfulJob.usedBytes + graphPeak &&
              ResourceKindBytes(duringSuccessfulSource,
                  UsdGenExecutionResourceKind::Pending) > 0);
        releaseCudaSourceAsyncCallbackGateForTesting();
        widthQueue.Drain();
        bool const successfulCompleted = Wait(successfulDone);
        auto const successfulSnapshot = widthQueue.Snapshot();
        auto const afterSuccessfulJob = resourcePool->Snapshot();
        if (!successfulCompleted || !successfulSnapshot ||
            ResourceKindBytes(afterSuccessfulJob,
                UsdGenExecutionResourceKind::Pending) != 0 ||
            ResourceKindBytes(afterSuccessfulJob,
                UsdGenExecutionResourceKind::Pinned) == 0 ||
            afterSuccessfulJob.usedBytes < beforeSuccessfulJob.usedBytes) {
            std::fprintf(stderr,
                "successful width: done=%d snapshot=%d used=%zu before=%zu "
                "pending=%zu pinned=%zu\n",
                int(successfulCompleted), int(bool(successfulSnapshot)),
                afterSuccessfulJob.usedBytes, beforeSuccessfulJob.usedBytes,
                ResourceKindBytes(afterSuccessfulJob,
                    UsdGenExecutionResourceKind::Pending),
                ResourceKindBytes(afterSuccessfulJob,
                    UsdGenExecutionResourceKind::Pinned));
            return 1;
        }

        struct Gate { void (*arm)(); void (*wait)(); void (*release)(); };
        Gate const gates[] = {
            {armCudaOperatorAsyncContextCallbackGateForTesting,
             waitCudaOperatorAsyncContextCallbackGateForTesting,
             releaseCudaOperatorAsyncContextCallbackGateForTesting},
            {armCudaOperatorAsyncProgramCallbackGateForTesting,
             waitCudaOperatorAsyncProgramCallbackGateForTesting,
             releaseCudaOperatorAsyncProgramCallbackGateForTesting},
            {armCudaOperatorAsyncWidthCallbackGateForTesting,
             waitCudaOperatorAsyncWidthCallbackGateForTesting,
             releaseCudaOperatorAsyncWidthCallbackGateForTesting},
        };
        for (auto const& gate : gates) {
            auto const lastGood = widthQueue.Snapshot();
            std::atomic<int> calls{0}; std::atomic<bool> done{false}, probeRan{false};
            gate.arm(); ReleaseOperatorGate releaseGate{gate.release};
            CHECK(widthQueue.Submit(widthPlan, 2, [&](uint64_t, auto outcome, auto const&) {
                if (outcome == UsdGenExecutionPipeline::Outcome::Published) ++calls;
                done.store(true, std::memory_order_release);
            }));
            gate.wait();
            CHECK(probe.Submit([&](auto const&) {
                probeRan.store(true, std::memory_order_release);
                return UsdGenExecutionPipeline::Publish{};
            }));
            CHECK(Wait(probeRan) && !done.load(std::memory_order_acquire) && calls == 0 &&
                  widthQueue.Snapshot() == lastGood);
            gate.release(); releaseGate.release = nullptr;
            widthQueue.Drain(); probe.Drain();
            CHECK(Wait(done) && calls == 1 && widthQueue.Snapshot() != lastGood &&
                  cudaOperatorRelayOccupiedCountForTesting() == 0);
        }

        auto const lastGood = widthQueue.Snapshot();
        auto const cancellationPending = ResourceKindBytes(
            resourcePool->Snapshot(), UsdGenExecutionResourceKind::Pending);
        std::atomic<int> cancelledCalls{0}; std::atomic<bool> cancelledDone{false};
        UsdGenExecutionPipeline::Outcome cancelled = UsdGenExecutionPipeline::Outcome::Published;
        armCudaOperatorAsyncWidthCallbackGateForTesting();
        ReleaseOperatorGate cancelRelease{releaseCudaOperatorAsyncWidthCallbackGateForTesting};
        CHECK(widthQueue.Submit(widthPlan, 3, [&](uint64_t, auto outcome, auto const&) {
            cancelled = outcome; ++cancelledCalls;
            cancelledDone.store(true, std::memory_order_release);
        }));
        waitCudaOperatorAsyncWidthCallbackGateForTesting();
        CHECK(widthQueue.CancelPending());
        releaseCudaOperatorAsyncWidthCallbackGateForTesting(); cancelRelease.release = nullptr;
        widthQueue.Drain();
        CHECK(Wait(cancelledDone) && cancelledCalls == 1 &&
              cancelled == UsdGenExecutionPipeline::Outcome::Superseded &&
              widthQueue.Snapshot() == lastGood && cudaOperatorRelayOccupiedCountForTesting() == 0 &&
              ResourceKindBytes(resourcePool->Snapshot(),
                  UsdGenExecutionResourceKind::Pending) == cancellationPending);
        std::atomic<int> supersededCalls{0}; std::atomic<bool> supersededDone{false}, replacementDone{false};
        UsdGenExecutionPipeline::Outcome superseded = UsdGenExecutionPipeline::Outcome::Published;
        armCudaOperatorAsyncWidthCallbackGateForTesting();
        ReleaseOperatorGate supersedeRelease{releaseCudaOperatorAsyncWidthCallbackGateForTesting};
        CHECK(widthQueue.Submit(widthPlan, 4, [&](uint64_t, auto outcome, auto const&) {
            superseded = outcome; ++supersededCalls;
            supersededDone.store(true, std::memory_order_release);
        }));
        waitCudaOperatorAsyncWidthCallbackGateForTesting();
        CHECK(widthQueue.Submit(widthPlan, 5, [&](uint64_t, auto outcome, auto const&) {
            replacementDone.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                                  std::memory_order_release);
        }));
        releaseCudaOperatorAsyncWidthCallbackGateForTesting(); supersedeRelease.release = nullptr;
        widthQueue.Drain();
        CHECK(Wait(supersededDone) && Wait(replacementDone) && supersededCalls == 1 &&
              superseded == UsdGenExecutionPipeline::Outcome::Superseded &&
              widthQueue.Snapshot() != lastGood && cudaOperatorRelayOccupiedCountForTesting() == 0);
        std::atomic<bool> reclaimed{false};
        CHECK(widthQueue.Submit(widthPlan, 6, [&](uint64_t, auto outcome, auto const&) {
            reclaimed.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                            std::memory_order_release);
        }));
        widthQueue.Drain();
        CHECK(Wait(reclaimed) && widthQueue.Snapshot() != lastGood &&
              cudaOperatorRelayOccupiedCountForTesting() == 0);
    }
    // Length has its own kernel terminal proof followed by exact survivor
    // count/scatter proofs.  Holding each one must leave the sole runtime
    // worker available and cannot publish a candidate before that proof.
    {
        auto lengthPlan = CompileCudaGraph(usdGenUsd::Lower(LengthDesc()), &diagnostics); CHECK(lengthPlan);
        auto lengthMetadata = GetCudaExecutionPlanMetadata(*lengthPlan);
        CHECK(lengthMetadata && !lengthMetadata->MemoryEstimate().memoryAvailable);
        UsdGenExecutionRuntime oneWorker(1);
        UsdGenCudaExecutionQueue lengthQueue(oneWorker, -1, &diagnostics); CHECK(lengthQueue.Valid());
        UsdGenExecutionPipeline probe(oneWorker);
        std::atomic<bool> initialDone{false};
        CHECK(lengthQueue.Submit(lengthPlan, 1, [&](uint64_t, auto outcome, auto const&) {
            initialDone.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                              std::memory_order_release);
        }));
        lengthQueue.Drain(); CHECK(Wait(initialDone) && lengthQueue.Snapshot());
        int lengthDevice = -1;
        CHECK(cudaGetDevice(&lengthDevice) == cudaSuccess);
        auto lengthResources = FindUsdGenExecutionResourcePool(
            {usdGenTestUtils::Backend("cuda"), lengthDevice});
        CHECK(lengthResources);
        auto const lengthPending = ResourceKindBytes(
            lengthResources->Snapshot(), UsdGenExecutionResourceKind::Pending);

        struct Gate { void (*arm)(); void (*wait)(); void (*release)(); };
        Gate const gates[] = {
            {armCudaOperatorAsyncLengthCallbackGateForTesting,
             waitCudaOperatorAsyncLengthCallbackGateForTesting,
             releaseCudaOperatorAsyncLengthCallbackGateForTesting},
            {armCudaOperatorAsyncCountsCallbackGateForTesting,
             waitCudaOperatorAsyncCountsCallbackGateForTesting,
             releaseCudaOperatorAsyncCountsCallbackGateForTesting},
            {armCudaOperatorAsyncScatterCallbackGateForTesting,
             waitCudaOperatorAsyncScatterCallbackGateForTesting,
             releaseCudaOperatorAsyncScatterCallbackGateForTesting},
            {armCudaOperatorAsyncNamedTopologyCallbackGateForTesting,
             waitCudaOperatorAsyncNamedTopologyCallbackGateForTesting,
             releaseCudaOperatorAsyncNamedTopologyCallbackGateForTesting},
        };
        for (auto const& gate : gates) {
            auto const lastGood = lengthQueue.Snapshot();
            std::atomic<int> calls{0}; std::atomic<bool> done{false}, probeRan{false};
            gate.arm(); ReleaseOperatorGate releaseGate{gate.release};
            CHECK(lengthQueue.Submit(lengthPlan, 2, [&](uint64_t, auto outcome, auto const&) {
                if (outcome == UsdGenExecutionPipeline::Outcome::Published) ++calls;
                done.store(true, std::memory_order_release);
            }));
            gate.wait();
            auto const heldLength = lengthResources->Snapshot();
            CHECK(heldLength.usedBytes > 0 &&
                  ResourceKindBytes(heldLength,
                      UsdGenExecutionResourceKind::Pending) > lengthPending);
            CHECK(probe.Submit([&](auto const&) {
                probeRan.store(true, std::memory_order_release);
                return UsdGenExecutionPipeline::Publish{};
            }));
            CHECK(Wait(probeRan) && !done.load(std::memory_order_acquire) && calls == 0 &&
                  lengthQueue.Snapshot() == lastGood);
            gate.release(); releaseGate.release = nullptr;
            lengthQueue.Drain(); probe.Drain();
            auto const afterLength = lengthResources->Snapshot();
            bool const lengthDone = Wait(done);
            auto const lengthSnapshot = lengthQueue.Snapshot();
            size_t const afterLengthPending = ResourceKindBytes(
                afterLength, UsdGenExecutionResourceKind::Pending);
            size_t const occupied = cudaOperatorRelayOccupiedCountForTesting();
            if (!lengthDone || calls != 1 || lengthSnapshot == lastGood ||
                afterLengthPending != lengthPending || occupied != 0) {
                std::fprintf(stderr,
                    "length gate completion: done=%d calls=%d snapshotChanged=%d "
                    "pending=%zu expected=%zu occupied=%zu\n",
                    int(lengthDone), calls.load(), int(lengthSnapshot != lastGood),
                    afterLengthPending, lengthPending, occupied);
                return 1;
            }
        }

        auto const lastGood = lengthQueue.Snapshot();
        std::atomic<int> cancelledCalls{0}; std::atomic<bool> cancelledDone{false};
        UsdGenExecutionPipeline::Outcome cancelled = UsdGenExecutionPipeline::Outcome::Published;
        armCudaOperatorAsyncScatterCallbackGateForTesting();
        ReleaseOperatorGate cancelRelease{releaseCudaOperatorAsyncScatterCallbackGateForTesting};
        CHECK(lengthQueue.Submit(lengthPlan, 3, [&](uint64_t, auto outcome, auto const&) {
            cancelled = outcome; ++cancelledCalls;
            cancelledDone.store(true, std::memory_order_release);
        }));
        waitCudaOperatorAsyncScatterCallbackGateForTesting();
        CHECK(lengthQueue.CancelPending());
        cancelRelease.release = nullptr;
        releaseCudaOperatorAsyncScatterCallbackGateForTesting();
        lengthQueue.Drain();
        CHECK(Wait(cancelledDone) && cancelledCalls == 1 &&
              cancelled == UsdGenExecutionPipeline::Outcome::Superseded &&
              lengthQueue.Snapshot() == lastGood && cudaOperatorRelayOccupiedCountForTesting() == 0);

        std::atomic<int> supersededCalls{0}; std::atomic<bool> supersededDone{false}, replacementDone{false};
        UsdGenExecutionPipeline::Outcome superseded = UsdGenExecutionPipeline::Outcome::Published;
        armCudaOperatorAsyncLengthCallbackGateForTesting();
        ReleaseOperatorGate supersedeRelease{releaseCudaOperatorAsyncLengthCallbackGateForTesting};
        CHECK(lengthQueue.Submit(lengthPlan, 4, [&](uint64_t, auto outcome, auto const&) {
            superseded = outcome; ++supersededCalls;
            supersededDone.store(true, std::memory_order_release);
        }));
        waitCudaOperatorAsyncLengthCallbackGateForTesting();
        CHECK(lengthQueue.Submit(lengthPlan, 5, [&](uint64_t, auto outcome, auto const&) {
            replacementDone.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                                  std::memory_order_release);
        }));
        supersedeRelease.release = nullptr;
        releaseCudaOperatorAsyncLengthCallbackGateForTesting();
        lengthQueue.Drain();
        CHECK(Wait(supersededDone) && Wait(replacementDone) && supersededCalls == 1 &&
              superseded == UsdGenExecutionPipeline::Outcome::Superseded &&
              lengthQueue.Snapshot() != lastGood && cudaOperatorRelayOccupiedCountForTesting() == 0);
        std::atomic<bool> reclaimed{false};
        CHECK(lengthQueue.Submit(lengthPlan, 6, [&](uint64_t, auto outcome, auto const&) {
            reclaimed.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                            std::memory_order_release);
        }));
        lengthQueue.Drain();
        CHECK(Wait(reclaimed) && cudaOperatorRelayOccupiedCountForTesting() == 0);

        // A phase-5 native failure cannot expose its private named outputs or
        // mutate a lease retained from the last accepted publication. Keep
        // this last because an unproved callback deliberately quarantines its
        // permanent relay slot.
        UsdGenExecutionRuntime failureRuntime(1);
        UsdGenCudaExecutionQueue failureQueue(failureRuntime, -1, &diagnostics);
        CHECK(failureQueue.Valid());
        std::atomic<bool> seeded{false};
        CHECK(failureQueue.Submit(lengthPlan, 20,
            [&](uint64_t, auto outcome, auto const&) {
                seeded.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                    std::memory_order_release);
            }));
        failureQueue.Drain();
        auto const retained = failureQueue.Snapshot();
        CHECK(Wait(seeded) && retained && retained->generation);
        cudaStream_t namedReader = nullptr;
        CHECK(cudaStreamCreateWithFlags(&namedReader, cudaStreamNonBlocking) == cudaSuccess);
        {
            auto retainedDensity = gpu::AcquireNamedChannel(
                retained->generation, TfToken("density"), namedReader);
            std::vector<float> before(2), after(2);
            CHECK(retainedDensity &&
                  cudaMemcpyAsync(before.data(), retainedDensity.Bytes().data,
                      retainedDensity.Bytes().size, cudaMemcpyDeviceToHost,
                      namedReader) == cudaSuccess &&
                  cudaStreamSynchronize(namedReader) == cudaSuccess);
            auto const quarantined = cudaOperatorRelayQuarantinedCountForTesting();
            std::atomic<bool> failedDone{false};
            std::atomic<int> failedOutcome{-1};
            failNextCudaOperatorRelayNamedTopologyNativeCallbackForTesting();
            CHECK(failureQueue.Submit(lengthPlan, 21,
                [&](uint64_t, auto outcome, auto const&) {
                    failedOutcome.store(int(outcome), std::memory_order_release);
                    failedDone.store(true, std::memory_order_release);
                }));
            failureQueue.Drain();
            CHECK(Wait(failedDone) && failedOutcome.load() != int(
                      UsdGenExecutionPipeline::Outcome::Published) &&
                  failureQueue.Snapshot() == retained &&
                  cudaOperatorRelayQuarantinedCountForTesting() == quarantined + 1);
            CHECK(cudaMemcpyAsync(after.data(), retainedDensity.Bytes().data,
                      retainedDensity.Bytes().size, cudaMemcpyDeviceToHost,
                      namedReader) == cudaSuccess &&
                  cudaStreamSynchronize(namedReader) == cudaSuccess && after == before);
        }
        CHECK(cudaStreamDestroy(namedReader) == cudaSuccess);
    }
    // Finalization has three asynchronous terminal proofs: tile status,
    // bounds status, then tile metadata/topology readback. Each releases the
    // single runtime worker and must retain the previous public snapshot.
    {
        auto finalPlan = CompileCudaGraph(usdGenUsd::Lower(LengthDesc()), &diagnostics); CHECK(finalPlan);
        UsdGenExecutionRuntime oneWorker(1);
        UsdGenCudaExecutionQueue finalQueue(oneWorker, -1, &diagnostics); CHECK(finalQueue.Valid());
        UsdGenExecutionPipeline probe(oneWorker);
        std::atomic<bool> initialDone{false};
        CHECK(finalQueue.Submit(finalPlan, 1, [&](uint64_t, auto outcome, auto const&) {
            initialDone.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                              std::memory_order_release);
        }));
        finalQueue.Drain(); CHECK(Wait(initialDone) && finalQueue.Snapshot());

        struct FinalGate { void (*arm)(); void (*wait)(); void (*release)(); };
        FinalGate const gates[] = {
            {armCudaFinalizationAsyncTileCallbackGateForTesting,
             waitCudaFinalizationAsyncTileCallbackGateForTesting,
             releaseCudaFinalizationAsyncTileCallbackGateForTesting},
            {armCudaFinalizationAsyncBoundsCallbackGateForTesting,
             waitCudaFinalizationAsyncBoundsCallbackGateForTesting,
             releaseCudaFinalizationAsyncBoundsCallbackGateForTesting},
            {armCudaFinalizationAsyncMetadataCallbackGateForTesting,
             waitCudaFinalizationAsyncMetadataCallbackGateForTesting,
             releaseCudaFinalizationAsyncMetadataCallbackGateForTesting},
        };
        for (auto const& gate : gates) {
            auto const lastGood = finalQueue.Snapshot();
            std::atomic<int> calls{0}; std::atomic<bool> done{false}, probeRan{false};
            gate.arm(); ReleaseOperatorGate releaseGate{gate.release};
            CHECK(finalQueue.Submit(finalPlan, 2, [&](uint64_t, auto outcome, auto const&) {
                if (outcome == UsdGenExecutionPipeline::Outcome::Published) ++calls;
                done.store(true, std::memory_order_release);
            }));
            gate.wait();
            CHECK(probe.Submit([&](auto const&) {
                probeRan.store(true, std::memory_order_release);
                return UsdGenExecutionPipeline::Publish{};
            }));
            CHECK(Wait(probeRan) && !done.load(std::memory_order_acquire) && calls == 0 &&
                  finalQueue.Snapshot() == lastGood);
            gate.release(); releaseGate.release = nullptr;
            finalQueue.Drain(); probe.Drain();
            CHECK(Wait(done) && calls == 1 && finalQueue.Snapshot() != lastGood &&
                  cudaFinalizationRelayOccupiedCountForTesting() == 0);
        }

        auto const lastGood = finalQueue.Snapshot();
        std::atomic<int> cancelledCalls{0}; std::atomic<bool> cancelledDone{false};
        UsdGenExecutionPipeline::Outcome cancelled = UsdGenExecutionPipeline::Outcome::Published;
        armCudaFinalizationAsyncMetadataCallbackGateForTesting();
        ReleaseOperatorGate cancelRelease{releaseCudaFinalizationAsyncMetadataCallbackGateForTesting};
        CHECK(finalQueue.Submit(finalPlan, 3, [&](uint64_t, auto outcome, auto const&) {
            cancelled = outcome; ++cancelledCalls;
            cancelledDone.store(true, std::memory_order_release);
        }));
        waitCudaFinalizationAsyncMetadataCallbackGateForTesting();
        CHECK(finalQueue.CancelPending());
        releaseCudaFinalizationAsyncMetadataCallbackGateForTesting(); cancelRelease.release = nullptr;
        finalQueue.Drain();
        CHECK(Wait(cancelledDone) && cancelledCalls == 1 &&
              cancelled == UsdGenExecutionPipeline::Outcome::Superseded &&
              finalQueue.Snapshot() == lastGood && cudaFinalizationRelayOccupiedCountForTesting() == 0);

        std::atomic<int> supersededCalls{0}; std::atomic<bool> supersededDone{false}, replacementDone{false};
        UsdGenExecutionPipeline::Outcome superseded = UsdGenExecutionPipeline::Outcome::Published;
        armCudaFinalizationAsyncTileCallbackGateForTesting();
        ReleaseOperatorGate supersedeRelease{releaseCudaFinalizationAsyncTileCallbackGateForTesting};
        CHECK(finalQueue.Submit(finalPlan, 4, [&](uint64_t, auto outcome, auto const&) {
            superseded = outcome; ++supersededCalls;
            supersededDone.store(true, std::memory_order_release);
        }));
        waitCudaFinalizationAsyncTileCallbackGateForTesting();
        CHECK(finalQueue.Submit(finalPlan, 5, [&](uint64_t, auto outcome, auto const&) {
            replacementDone.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                                  std::memory_order_release);
        }));
        releaseCudaFinalizationAsyncTileCallbackGateForTesting(); supersedeRelease.release = nullptr;
        finalQueue.Drain();
        CHECK(Wait(supersededDone) && Wait(replacementDone) && supersededCalls == 1 &&
              superseded == UsdGenExecutionPipeline::Outcome::Superseded &&
              finalQueue.Snapshot() != lastGood && cudaFinalizationRelayOccupiedCountForTesting() == 0);

        // A shared cap rejects a second finalization while a first is held;
        // release returns the one normal slot for a later request.
        setCudaFinalizationRelayCapacityForTesting(1); ResetFinalizationRelayCapacity resetCapacity;
        UsdGenCudaExecutionQueue otherQueue(oneWorker, -1, &diagnostics); CHECK(otherQueue.Valid());
        int capDevice = -1; CHECK(cudaGetDevice(&capDevice) == cudaSuccess);
        auto capGraph = UsdGenExecutionTaskGraph::GetOrCreate(oneWorker, "cuda", capDevice); CHECK(capGraph);
        // Hold only after metadata's native terminal has run, so the first
        // candidate retains its admission without blocking unrelated setup
        // allocations on the shared CUDA stream.
        armCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
        ReleaseOperatorGate capRelease{releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting};
        std::atomic<bool> capFirstDone{false}, capSecondDone{false};
        std::atomic<int> capFirstOutcome{-1}, capSecondOutcome{-1};
        CHECK(finalQueue.Submit(finalPlan, 6, [&](uint64_t, auto outcome, auto const&) {
            capFirstOutcome.store(static_cast<int>(outcome), std::memory_order_relaxed);
            capFirstDone.store(true, std::memory_order_release);
        }));
        waitCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
        CHECK(otherQueue.Submit(finalPlan, 7, [&](uint64_t, auto outcome, auto const&) {
            capSecondOutcome.store(static_cast<int>(outcome), std::memory_order_relaxed);
            capSecondDone.store(true, std::memory_order_release);
        }));
        bool const capSecondObserved = Wait(capSecondDone);
        if (!capSecondObserved || capFirstDone.load(std::memory_order_acquire) ||
            capSecondOutcome.load(std::memory_order_acquire) != static_cast<int>(UsdGenExecutionPipeline::Outcome::Failed)) {
            auto const usage = capGraph->GetAdmissionUsage();
            std::fprintf(stderr,
                "cap-one queue: first done=%d outcome=%d second done=%d outcome=%d graph=%u/%u/%u source=%zu operator=%zu final=%zu\n",
                int(capFirstDone.load(std::memory_order_acquire)), capFirstOutcome.load(std::memory_order_acquire),
                int(capSecondDone.load(std::memory_order_acquire)), capSecondOutcome.load(std::memory_order_acquire),
                usage.jobs, usage.tasks, usage.dependencyEdges,
                cudaSourceRelayOccupiedCountForTesting(), cudaOperatorRelayOccupiedCountForTesting(),
                cudaFinalizationRelayOccupiedCountForTesting());
        }
        CHECK(capSecondObserved && !capFirstDone.load(std::memory_order_acquire) &&
              capSecondOutcome.load(std::memory_order_acquire) == static_cast<int>(UsdGenExecutionPipeline::Outcome::Failed));
        releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting(); capRelease.release = nullptr;
        finalQueue.Drain(); otherQueue.Drain();
        CHECK(Wait(capFirstDone) && capFirstOutcome.load(std::memory_order_acquire) ==
              static_cast<int>(UsdGenExecutionPipeline::Outcome::Published) &&
              cudaFinalizationRelayOccupiedCountForTesting() == 0);
        std::atomic<bool> reclaimed{false};
        CHECK(otherQueue.Submit(finalPlan, 8, [&](uint64_t, auto outcome, auto const&) {
            reclaimed.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                            std::memory_order_release);
        }));
        otherQueue.Drain();
        CHECK(Wait(reclaimed) && cudaFinalizationRelayOccupiedCountForTesting() == 0);
        // The cap-one fixture is complete.  Independent unsafe-fault cases
        // below intentionally quarantine a slot, so restore the normal pool
        // before establishing their separate last-good generation.
        setCudaFinalizationRelayCapacityForTesting(1024);

        // A terminal callback-install failure is still one accepted failure:
        // it cannot replace a completed snapshot and keeps unsafe work charged.
        auto const beforeFault = otherQueue.Snapshot();
        std::atomic<int> faultCalls{0}; std::atomic<bool> faultDone{false};
        failNextCudaFinalizationRelayMetadataCallbackInstallForTesting();
        CHECK(otherQueue.Submit(finalPlan, 9, [&](uint64_t, auto outcome, auto const& diagnostics) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Failed && diagnostics.HasErrors())
                ++faultCalls;
            faultDone.store(true, std::memory_order_release);
        }));
        otherQueue.Drain();
        CHECK(Wait(faultDone) && faultCalls == 1 && otherQueue.Snapshot() == beforeFault &&
              cudaFinalizationRelayQuarantinedCountForTesting() != 0);

        UsdGenCudaExecutionQueue nativeFaultQueue(oneWorker, -1, &diagnostics);
        CHECK(nativeFaultQueue.Valid());
        std::atomic<bool> nativeGood{false};
        CHECK(nativeFaultQueue.Submit(finalPlan, 10, [&](uint64_t, auto outcome, auto const&) {
            nativeGood.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                             std::memory_order_release);
        }));
        nativeFaultQueue.Drain(); CHECK(Wait(nativeGood) && nativeFaultQueue.Snapshot());
        auto const beforeNativeFault = nativeFaultQueue.Snapshot();
        std::atomic<int> nativeFaultCalls{0}; std::atomic<bool> nativeFaultDone{false};
        failNextCudaFinalizationRelayMetadataNativeCallbackForTesting();
        CHECK(nativeFaultQueue.Submit(finalPlan, 11, [&](uint64_t, auto outcome, auto const& diagnostics) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Failed && diagnostics.HasErrors())
                ++nativeFaultCalls;
            nativeFaultDone.store(true, std::memory_order_release);
        }));
        nativeFaultQueue.Drain();
        CHECK(Wait(nativeFaultDone) && nativeFaultCalls == 1 &&
              nativeFaultQueue.Snapshot() == beforeNativeFault &&
              cudaFinalizationRelayQuarantinedCountForTesting() != 0);
    }
    // Phase-two (resample callback) failures are distinct from source upload
    // failures: each is terminal exactly once, retains last-good output, and
    // poisons only its own workspace for later rejection.
    {
        auto resampleDesc=Desc();
        resampleDesc.nodes.front().params.push_back({TfToken("resampleTo"),VtValue(4),false});
        auto resamplePlan=CompileCudaGraph(usdGenUsd::Lower(resampleDesc),&diagnostics); CHECK(resamplePlan);
        UsdGenExecutionRuntime faultRuntime(1);
        auto exerciseResampleFault=[&](bool nativeCallback) -> int {
            UsdGenCudaExecutionQueue fault(faultRuntime,-1,&diagnostics); CHECK(fault.Valid());
            std::atomic<bool> goodDone{false};
            CHECK(fault.Submit(resamplePlan,1,[&](uint64_t, auto outcome, auto const&) {
                goodDone.store(outcome==UsdGenExecutionPipeline::Outcome::Published,std::memory_order_release);
            }));
            fault.Drain(); CHECK(Wait(goodDone) && fault.Snapshot());
            auto const lastGood=fault.Snapshot();
            std::atomic<int> failureCalls{0}; std::atomic<bool> failureDone{false};
            if (nativeCallback) failNextCudaSourceRelayResampleNativeCallbackForTesting();
            else failNextCudaSourceRelayResampleCallbackInstallForTesting();
            CHECK(fault.Submit(resamplePlan,2,[&](uint64_t, auto outcome, auto const& d) {
                if (outcome==UsdGenExecutionPipeline::Outcome::Failed && d.HasErrors()) ++failureCalls;
                failureDone.store(true,std::memory_order_release);
            }));
            fault.Drain();
            CHECK(Wait(failureDone) && failureCalls==1 && fault.Snapshot()==lastGood);
            std::atomic<bool> retryRejected{false};
            CHECK(fault.Submit(resamplePlan,3,[&](uint64_t, auto outcome, auto const&) {
                retryRejected.store(outcome==UsdGenExecutionPipeline::Outcome::Failed,std::memory_order_release);
            }));
            fault.Drain(); CHECK(Wait(retryRejected) && fault.Snapshot()==lastGood);
            return 0;
        };
        CHECK(exerciseResampleFault(false)==0);
        CHECK(exerciseResampleFault(true)==0);
    }
    {
        // A queue releases its last snapshot's generation when it is
        // destroyed, even while another thread holds an epoch pin.
        std::weak_ptr<const UsdGenDeviceGeneration> weakGeneration;
        std::atomic<int> pin{0};
        std::thread pinned;
        {
            UsdGenCudaExecutionQueue q(runtime,-1,&diagnostics);
            CHECK(q.Valid() && q.Submit(plan,1,[](uint64_t, auto, auto const&) {}));
            q.Drain();
            {
                auto s = q.Snapshot(); CHECK(s && s->generation);
                weakGeneration = s->generation;
            }
            pinned = std::thread([&] {
                UsdGenEpochGuard guard;
                pin.store(1);
                while (pin.load() != 2) std::this_thread::yield();
            });
            while (pin.load() != 1) std::this_thread::yield();
        }
        bool const released = weakGeneration.expired();
        pin.store(2); pinned.join();
        CHECK(released);
    }
    std::shared_ptr<const UsdGenCudaQueueSnapshot> retained;
    {
        UsdGenCudaExecutionQueue a(runtime,-1,&diagnostics), b(runtime,-1,&diagnostics);
        CHECK(a.Valid() && b.Valid() && !a.Snapshot() && !b.Snapshot());
        std::atomic<int> completed{0}, failed{0};
        auto completion = [&](uint64_t, UsdGenExecutionPipeline::Outcome outcome, UsdGenDiagnostics const& d) {
            ++completed; if (outcome == UsdGenExecutionPipeline::Outcome::Failed || d.HasErrors()) ++failed;
        };
        auto firstA = a.Submit(plan,1,completion);
        auto firstB = b.Submit(plan,2,completion,
            UsdGenExecutionTaskGraph::RequestClass::Background);
        CHECK(firstA && firstB);
        a.Drain(); b.Drain();
        auto a1 = a.Snapshot(), b1 = b.Snapshot();
        CHECK(completed == 2 && failed == 0 && a1 && b1);
        CHECK(a1->epoch == firstA && b1->epoch == firstB);
        CHECK(Root(a1->generation,.45f,reader) && Root(b1->generation,1.2f,reader));
        CHECK(a1->bindings.size() == 1 && b1->bindings.size() == 1 &&
            a1->bindings[0].identity != b1->bindings[0].identity &&
            a1->bindings[0].bindCount == 1 && b1->bindings[0].bindCount == 1);
        retained = a1;

        auto posed = CompileCudaGraph(usdGenUsd::Lower(Desc(2)),&diagnostics); CHECK(posed);
        CHECK(a.Submit(posed,2,completion)); a.Drain();
        auto a2 = a.Snapshot(); CHECK(a2 != a1 && Root(a2->generation,2.2f,reader));
        CHECK(a2->bindings[0].identity == a1->bindings[0].identity &&
            a2->bindings[0].bindCount == 1 && a2->bindings[0].solveCount == 2);
        CHECK(Root(a1->generation,.45f,reader) && Root(b1->generation,1.2f,reader));

        // DeviceBuffer allocation admission uses the same registry pool as
        // queue execution. Saturating its remaining tracked bytes rejects a
        // replacement without disturbing a published generation or its lease.
        int device = -1; CHECK(cudaGetDevice(&device) == cudaSuccess);
        auto resources = FindUsdGenExecutionResourcePool(
            {usdGenTestUtils::Backend("cuda"), device});
        CHECK(resources);
        auto state = resources->Snapshot();
        auto filler = resources->TryReserve(state.usableBytes - state.usedBytes,
                                            UsdGenExecutionResourceKind::Scratch);
        CHECK(filler);
        bool budgetRejected = false;
        CHECK(a.Submit(posed,3,[&](uint64_t, auto outcome, auto const& d) {
            budgetRejected = outcome == UsdGenExecutionPipeline::Outcome::Failed && d.HasErrors();
        }));
        a.Drain(); CHECK(budgetRejected && a.Snapshot() == a2);
        CHECK(Root(a2->generation,2.2f,reader));
        filler.reset();
        CHECK(a.Submit(plan,2,completion)); a.Drain();
        CHECK(Root(a.Snapshot()->generation,1.2f,reader));

        // Execution failure cannot replace the last good published snapshot.
        auto beforeInvalid = a.Snapshot();
        auto const beforeInvalidResources = resources->Snapshot();
        bool rejected = false;
        CHECK(!a.Submit(posed,std::numeric_limits<double>::quiet_NaN()));
        auto invalidValues = Desc();
        invalidValues.expressions[0].source = "$P[0] / ($P[0] - $P[0])";
        auto invalidPlan = CompileCudaGraph(usdGenUsd::Lower(invalidValues),&diagnostics); CHECK(invalidPlan);
        CHECK(a.Submit(invalidPlan,2,
            [&](uint64_t, auto outcome, auto const& d) {
                rejected = outcome == UsdGenExecutionPipeline::Outcome::Failed && d.HasErrors();
            }));
        a.Drain(); CHECK(rejected && a.Snapshot() == beforeInvalid &&
            ResourceKindBytes(resources->Snapshot(), UsdGenExecutionResourceKind::Pending) ==
                ResourceKindBytes(beforeInvalidResources, UsdGenExecutionResourceKind::Pending));
        CHECK(a.Submit(plan,2,completion)); a.Drain();
        CHECK(Root(a.Snapshot()->generation,1.2f,reader) &&
            a.Snapshot()->bindings[0].identity == a1->bindings[0].identity);

        // Multiple external producers enqueue immutable requests. The final
        // accepted request wins without a per-plan or per-cache mutex.
        std::vector<std::thread> producers;
        for (int thread=0;thread<4;++thread) producers.emplace_back([&] {
            for (int i=0;i<4;++i) a.Submit(plan,2,completion);
        });
        for (auto& thread : producers) thread.join();
        auto latest = a.Submit(posed,3,completion); CHECK(latest);
        a.Drain();
        CHECK(a.Snapshot()->epoch == latest && Root(a.Snapshot()->generation,2.2f,reader));
        CHECK(completed == 22 && failed == 0);
        auto snapshot = a.Snapshot(); CHECK(a.CancelPending()); a.Drain();
        CHECK(a.Snapshot() == snapshot);

        // Completion callbacks may enqueue another cook, never wait on the
        // owner task. Request reports remain paired with their generations.
        CHECK(b.Submit(plan,1,[&](uint64_t, auto outcome, auto const&) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Published)
                b.Submit(posed,2,completion);
        }));
        b.Drain(); CHECK(Root(b.Snapshot()->generation,2.2f,reader));

        // Keep the callback-reentry case explicit at the smallest runtime
        // concurrency.  The successor is submitted by the completion of the
        // first graph, then must make progress through the shared CUDA task
        // graph before the external Drain returns.
        {
            UsdGenExecutionRuntime serialRuntime(1);
            UsdGenCudaExecutionQueue serialQueue(serialRuntime,-1,&diagnostics);
            CHECK(serialQueue.Valid());
            std::atomic<bool> firstDone{false}, successorDone{false};
            std::atomic<uint64_t> successorTicket{0};
            CHECK(serialQueue.Submit(plan,1,[&](uint64_t, auto outcome, auto const&) {
                if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                    successorTicket.store(serialQueue.Submit(posed,2,
                        [&](uint64_t, auto successorOutcome, auto const&) {
                            successorDone.store(
                                successorOutcome == UsdGenExecutionPipeline::Outcome::Published,
                                std::memory_order_release);
                        }), std::memory_order_release);
                }
                firstDone.store(true,std::memory_order_release);
            }));
            serialQueue.Drain();
            auto const serialSnapshot = serialQueue.Snapshot();
            CHECK(Wait(firstDone) && Wait(successorDone) && successorTicket != 0 &&
                  serialSnapshot && serialSnapshot->epoch == successorTicket &&
                  Root(serialSnapshot->generation,2.2f,reader));
        }

        // Independent runtimes still share the per-device CUDA dispatcher.
        // A completion on one runtime may submit work to another queue on the
        // same device without losing the second queue's task-graph progress.
        {
            UsdGenExecutionRuntime runtimeA(1), runtimeB(1);
            UsdGenCudaExecutionQueue queueA(runtimeA,-1,&diagnostics);
            UsdGenCudaExecutionQueue queueB(runtimeB,-1,&diagnostics);
            CHECK(queueA.Valid() && queueB.Valid());
            std::atomic<bool> firstDone{false}, secondDone{false};
            std::atomic<uint64_t> secondTicket{0};
            CHECK(queueA.Submit(plan,3,[&](uint64_t, auto outcome, auto const&) {
                if (outcome == UsdGenExecutionPipeline::Outcome::Published)
                    secondTicket.store(queueB.Submit(posed,4,
                        [&](uint64_t, auto successorOutcome, auto const&) {
                            secondDone.store(
                                successorOutcome == UsdGenExecutionPipeline::Outcome::Published,
                                std::memory_order_release);
                        }), std::memory_order_release);
                firstDone.store(true,std::memory_order_release);
            }));
            queueA.Drain(); queueB.Drain();
            auto const secondSnapshot = queueB.Snapshot();
            CHECK(Wait(firstDone) && Wait(secondDone) && secondTicket != 0 &&
                  secondSnapshot && secondSnapshot->epoch == secondTicket &&
                  Root(secondSnapshot->generation,2.2f,reader));
        }

        // Producers can race on one queue, but a request submitted after all
        // producers have joined has the greatest ticket and is the published
        // latest-wins result. Every accepted producer request still receives
        // exactly one completion report.
        {
            UsdGenExecutionRuntime producerRuntime(2);
            UsdGenCudaExecutionQueue producerQueue(producerRuntime,-1,&diagnostics);
            CHECK(producerQueue.Valid());
            std::atomic<int> accepted{0}, reports{0}, failures{0};
            std::atomic<uint64_t> producerMax{0};
            auto producerCompletion = [&](uint64_t, auto outcome, auto const&) {
                reports.fetch_add(1,std::memory_order_relaxed);
                if (outcome == UsdGenExecutionPipeline::Outcome::Failed)
                    failures.fetch_add(1,std::memory_order_relaxed);
            };
            std::vector<std::thread> producerThreads;
            for (int producer=0; producer<3; ++producer)
                producerThreads.emplace_back([&, producer] {
                    for (int request=0; request<3; ++request) {
                        auto const ticket = producerQueue.Submit(
                            plan,10.0 + producer * 3 + request,producerCompletion);
                        if (ticket != 0) {
                            accepted.fetch_add(1,std::memory_order_relaxed);
                            auto previous = producerMax.load(std::memory_order_relaxed);
                            while (previous < ticket && !producerMax.compare_exchange_weak(
                                       previous,ticket,std::memory_order_relaxed)) {}
                        }
                    }
                });
            for (auto& producer : producerThreads) producer.join();
            auto const producerLatest = producerQueue.Submit(posed,20,producerCompletion);
            CHECK(producerLatest != 0 && producerLatest > producerMax.load(std::memory_order_relaxed));
            producerQueue.Drain();
            auto const latestSnapshot = producerQueue.Snapshot();
            CHECK(latestSnapshot && latestSnapshot->epoch == producerLatest &&
                  Root(latestSnapshot->generation,2.2f,reader) &&
                  failures == 0 && reports == accepted + 1);
        }
    }
    // Shutdown retires accepted/queued requests while their callbacks and
    // publication storage remain alive. No external Drain is needed here.
    std::atomic<int> retired{0};
    {
        UsdGenCudaExecutionQueue closing(runtime,-1,&diagnostics); CHECK(closing.Valid());
        for (int i=0;i<32;++i)
            CHECK(closing.Submit(plan,2,[&](uint64_t, auto, auto const&) { ++retired; }));
    }
    CHECK(retired == 32);
    plan.reset();
    CHECK(Root(retained->generation,.45f,reader)); // queue/workspace/plan gone
    retained.reset();
    CHECK(cudaStreamDestroy(reader) == cudaSuccess);
    return 0;
}
