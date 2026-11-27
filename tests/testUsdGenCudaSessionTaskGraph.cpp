#include "utils/pxrInterop.h"
#include "usdGenHydraCurves/publication.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "testBackend.h"
#include "usdGen/exec/executionTaskGraph.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGenCurves/cuda/generation.h"
#include "usdGen/exec/session.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)
constexpr auto kTimeout = std::chrono::seconds(10);

static UsdGenGraphDesc Desc() {
    UsdGenGraphDesc desc; desc.description = SdfPath("/SessionStages");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenSurfaceDesc surface; surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0}}; surface.points = surface.restPoints;
    surface.faceVertexCounts = {3}; surface.faceVertexIndices = {0,1,2}; desc.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves; curves.path = SdfPath("/Curves"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair"); curves.curveVertexCounts = {2}; curves.points = {{0,0,0},{0,1,0}};
    curves.rest = curves.points; curves.curveId = {1}; curves.skinPrim = {0}; curves.skinPrimUv = {{.2f,.2f}};
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source; source.path = SdfPath("/SessionStages/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path}; source.surfaces = {surface.path};
    UsdGenNodeDesc length; length.path = SdfPath("/SessionStages/Length"); length.type = TfToken("UsdGenLength");
    length.inputs = {source.path};
    // CUDA's length contract uses namespaced length tokens, not generic names.
    length.params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                     {TfToken("length:value"), VtValue(.5f), false}};
    UsdGenNodeDesc width; width.path = SdfPath("/SessionStages/Width"); width.type = TfToken("UsdGenWidth");
    width.inputs = {length.path}; width.params = {{TfToken("width"), VtValue(.03f), false}};
    desc.nodes = {source,length,width}; desc.terminal = width.path; return desc;
}

static UsdGenGraphDesc WidthDagDesc() {
    auto desc = Desc();
    auto source = desc.nodes.front();
    UsdGenNodeDesc terminal;
    terminal.path = SdfPath("/SessionStages/TerminalWidth");
    terminal.type = TfToken("UsdGenWidth");
    terminal.inputs = {source.path};
    terminal.params = {{TfToken("width"), VtValue(.02f), false}};
    UsdGenNodeDesc sibling = terminal;
    sibling.path = SdfPath("/SessionStages/SiblingWidth");
    sibling.params = {{TfToken("width"), VtValue(.07f), false}};
    desc.nodes = {sibling, source, terminal};
    desc.terminal = terminal.path;
    return desc;
}

struct Result {
    std::atomic<int> calls{0}; std::atomic<bool> done{false}; UsdGenSession::SnapshotPtr snapshot;
    UsdGenExecutionPipeline::Outcome outcome = UsdGenExecutionPipeline::Outcome::Superseded;
};
static std::shared_ptr<Result> Submit(UsdGenSession& session, UsdGenSession::CommitRequest request) {
    auto result = std::make_shared<Result>();
    if (!session.CommitAsync(std::move(request), [result](auto snapshot, auto outcome) {
        result->snapshot = std::move(snapshot); result->outcome = outcome;
        result->calls.fetch_add(1, std::memory_order_relaxed); result->done.store(true, std::memory_order_release);
    })) return {};
    return result;
}
static bool Wait(std::atomic<bool> const& done) {
    auto const deadline = std::chrono::steady_clock::now() + kTimeout;
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    return done.load(std::memory_order_acquire);
}
static bool WaitForUsage(std::shared_ptr<UsdGenExecutionTaskGraph> const& graph,
                         uint32_t jobs, uint32_t tasks, uint32_t edges) {
    auto const deadline = std::chrono::steady_clock::now() + kTimeout;
    for (;;) {
        auto const usage = graph->GetAdmissionUsage();
        if (usage.jobs == jobs && usage.tasks == tasks && usage.dependencyEdges == edges) return true;
        if (std::chrono::steady_clock::now() >= deadline) {
            auto const finalUsage = graph->GetAdmissionUsage();
            std::fprintf(stderr, "usage wanted %u/%u/%u, got %u/%u/%u\n",
                jobs, tasks, edges, finalUsage.jobs, finalUsage.tasks, finalUsage.dependencyEdges);
            return false;
        }
        std::this_thread::yield();
    }
}

// This state outlives all task lambdas.  Ready and released are a two-phase
// gate: whichever side arrives second invokes the held completion exactly once.
struct HeldRoots {
    struct Slot {
        UsdGenExecutionTaskGraph::TaskCompletion done;
        std::atomic<unsigned> flags{0};
    };
    std::array<Slot, 8> slots;

    void Hold(uint32_t index, UsdGenExecutionTaskGraph::TaskCompletion done) {
        auto& slot = slots[index];
        slot.done = std::move(done); // Immutable before Ready publishes it.
        auto const prior = slot.flags.fetch_or(1, std::memory_order_acq_rel);
        if (prior & 2) slot.done([] {}, {});
    }

    bool WaitFor(uint32_t roots) const {
        auto const deadline = std::chrono::steady_clock::now() + kTimeout;
        while (std::chrono::steady_clock::now() < deadline) {
            uint32_t ready = 0;
            for (auto const& slot : slots)
                ready += (slot.flags.load(std::memory_order_acquire) & 1) != 0;
            if (ready == roots) return true;
            std::this_thread::yield();
        }
        return false;
    }

    void Release() {
        for (auto& slot : slots) {
            auto const prior = slot.flags.fetch_or(2, std::memory_order_acq_rel);
            if ((prior & 1) && !(prior & 2)) slot.done([] {}, {});
        }
    }
};
struct ReleaseHeldRoots { std::shared_ptr<HeldRoots> roots; ~ReleaseHeldRoots() { roots->Release(); } };
static bool HoldDispatcher(std::shared_ptr<UsdGenExecutionTaskGraph> const& graph, std::shared_ptr<HeldRoots> const& roots) {
    UsdGenExecutionTaskGraph::Job sentinel;
    for (uint32_t i = 0; i != 8; ++i)
        sentinel.tasks.push_back({{}, [roots, i](auto const&, auto done) {
            roots->Hold(i, std::move(done));
        }});
    sentinel.tasks.push_back({{0,1,2,3,4,5,6,7}, [](auto const&, auto done) { done([] {}, {}); }});
    sentinel.finalTask = 8; sentinel.completion = [](auto, auto) {};
    return graph->Submit(std::move(sentinel)) && roots->WaitFor(8);
}
static bool ReadGeometry(UsdGenSession::SnapshotPtr const& snapshot, std::vector<float3>* points,
                         std::vector<float>* widths, cudaStream_t stream) {
    if (!snapshot || !snapshot->generation || !snapshot->generation->device) return false;
    auto lease = gpu::AcquireGeometry(snapshot->generation->device, stream); if (!lease) return false;
    auto geometry = lease.Geometry(); points->resize(geometry.pointCount); widths->resize(geometry.pointCount);
    return cudaMemcpyAsync(points->data(), geometry.points.data, points->size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaMemcpyAsync(widths->data(), geometry.widths.data, widths->size()*sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
static bool HasDiagnostic(UsdGenSession::SnapshotPtr const& snapshot, char const* text) {
    if (!snapshot) return false;
    for (auto const& error : snapshot->diagnostics.errors)
        if (error.find(text) != std::string::npos) return true;
    return false;
}
static bool SubmitNoop(std::shared_ptr<UsdGenExecutionTaskGraph> const& graph) {
    UsdGenExecutionTaskGraph::Job job;
    job.tasks.push_back({{}, [](auto const&, auto done) { done([] {}, {}); }});
    job.finalTask = 0;
    job.completion = [](auto, auto) {};
    return graph->Submit(std::move(job));
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
struct ReleaseWidthBranches {
    bool armed = true;
    ~ReleaseWidthBranches() {
        if (armed) releaseCudaOperatorAsyncWidthBranchGateForTesting();
    }
    void Release() {
        if (armed) releaseCudaOperatorAsyncWidthBranchGateForTesting();
        armed = false;
    }
};
static bool WaitForWidthBranches() {
    try {
        waitCudaOperatorAsyncWidthBranchGateForTesting();
        return true;
    } catch (...) {
        return false;
    }
}

int main() {
    usdGenTest::EnsureHydraPublishers();
    int device = -1; CHECK(cudaGetDevice(&device) == cudaSuccess);
    cudaStream_t reader = nullptr; CHECK(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking) == cudaSuccess);
    UsdGenExecutionRuntime runtime(1);
    auto graph = UsdGenExecutionTaskGraph::GetOrCreate(runtime, "cuda", device); CHECK(graph);
    CHECK(WaitForUsage(graph, 0, 0, 0));

    // Synchronous validation rejection never queues a completion callback.
    std::atomic<int> rejectedCallbacks{0}; UsdGenExecutionTaskGraph::Job malformed;
    malformed.tasks.push_back({{1}, [](auto const&, auto done) { done([] {}, {}); }});
    malformed.completion = [&](auto, auto) { ++rejectedCallbacks; };
    CHECK(!graph->Submit(std::move(malformed)));
    graph->Drain();
    CHECK(rejectedCallbacks.load() == 0 && WaitForUsage(graph, 0, 0, 0));

    auto roots = std::make_shared<HeldRoots>();
    // This guard is declared after the sessions: on a CHECK return it releases
    // external task holds before either Session destructor drains accepted work.
    UsdGenSession a(1), b(1); ReleaseHeldRoots releaseOnExit{roots};
    CHECK(HoldDispatcher(graph, roots));
    usdGenUsd::SetGraphDesc(a, Desc()); usdGenUsd::SetGraphDesc(b, Desc()); a.SetDevicePublicationEnabled(true); b.SetDevicePublicationEnabled(true);
    UsdGenSession::CommitRequest requestA; requestA.frame = 1; requestA.reason = UsdGenCommitReason::SetTime;
    UsdGenSession::CommitRequest requestB; requestB.frame = 2; requestB.reason = UsdGenCommitReason::SetTime;
    auto firstA = Submit(a, std::move(requestA)), firstB = Submit(b, std::move(requestB));
    CHECK(firstA && firstB);
    // Sentinel: 9 tasks/8 edges.  Each real Source->Length->Width->Finish
    // session job contributes 4 tasks/5 edges: the publication task carries
    // the compiler's explicit all-stage join in addition to operator dataflow.
    CHECK(WaitForUsage(graph, 3, 17, 18));
    CHECK(!firstA->done.load(std::memory_order_acquire) && !firstB->done.load(std::memory_order_acquire));
    roots->Release();
    CHECK(Wait(firstA->done) && Wait(firstB->done));
    CHECK(firstA->calls == 1 && firstB->calls == 1);
    CHECK(firstA->outcome == UsdGenExecutionPipeline::Outcome::Published && firstB->outcome == UsdGenExecutionPipeline::Outcome::Published);
    CHECK(firstA->snapshot && firstB->snapshot && firstA->snapshot->generation->device && firstB->snapshot->generation->device);
    CHECK(WaitForUsage(graph, 0, 0, 0));
    std::vector<float3> points; std::vector<float> widths;
    CHECK(ReadGeometry(firstA->snapshot, &points, &widths, reader)); CHECK(points.size() == 2 && widths.size() == 2);
    CHECK(std::fabs(points[0].x) < 1e-5f && std::fabs(points[0].y) < 1e-5f && std::fabs(points[0].z) < 1e-5f);
    CHECK(std::fabs(points[1].x) < 1e-5f && std::fabs(points[1].y - .5f) < 2e-3f && std::fabs(points[1].z) < 1e-5f);
    CHECK(std::fabs(widths[0] - .03f) < 1e-5f && std::fabs(widths[1] - .03f) < 1e-5f);

    // Session cooks must consume the compiled task dependencies too.  Both
    // source-rooted Width siblings become ready together; publication remains
    // held until the metadata join completes and selects the declared terminal.
    UsdGenSession dagSession(1);
    usdGenUsd::SetGraphDesc(dagSession, WidthDagDesc());
    dagSession.SetDevicePublicationEnabled(true);
    armCudaOperatorAsyncWidthBranchGateForTesting(2);
    ReleaseWidthBranches releaseWidthBranches;
    UsdGenSession::CommitRequest dagRequest;
    dagRequest.frame = 2.25;
    dagRequest.reason = UsdGenCommitReason::SetTime;
    auto dagResult = Submit(dagSession, std::move(dagRequest));
    CHECK(dagResult && WaitForWidthBranches());
    CHECK(cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting() == 2 &&
          !dagResult->done.load(std::memory_order_acquire));
    releaseWidthBranches.Release();
    CHECK(Wait(dagResult->done) && dagResult->calls == 1 &&
          dagResult->outcome == UsdGenExecutionPipeline::Outcome::Published);
    CHECK(ReadGeometry(dagResult->snapshot, &points, &widths, reader) &&
          widths.size() == 2 && std::fabs(widths[0] - .02f) < 1e-5f &&
          std::fabs(widths[1] - .02f) < 1e-5f);
    CHECK(WaitForUsage(graph, 0, 0, 0));

    // The source stream callback is held after Set.  Runtime concurrency is
    // one, so an independent CPU pipeline completion proves the staged
    // Session source task returned instead of waiting for GPU upload.
    UsdGenExecutionPipeline cpuProbe(runtime);
    armCudaSourceAsyncCallbackGateForTesting();
    ReleaseSourceGate releaseSourceGate;
    UsdGenSession::CommitRequest heldSourceRequest;
    heldSourceRequest.frame = 2.5; heldSourceRequest.reason = UsdGenCommitReason::SetTime;
    auto heldSource = Submit(a, std::move(heldSourceRequest)); CHECK(heldSource);
    waitCudaSourceAsyncCallbackGateForTesting();
    std::atomic<bool> cpuProbeDone{false};
    CHECK(cpuProbe.Submit([&](auto const&) {
        cpuProbeDone.store(true,std::memory_order_release);
        return UsdGenExecutionPipeline::Publish{};
    }));
    CHECK(Wait(cpuProbeDone) && !heldSource->done.load(std::memory_order_acquire) &&
          heldSource->calls == 0);
    releaseCudaSourceAsyncCallbackGateForTesting();
    CHECK(Wait(heldSource->done) && heldSource->calls == 1 &&
          heldSource->outcome == UsdGenExecutionPipeline::Outcome::Published);
    cpuProbe.Drain();
    CHECK(WaitForUsage(graph, 0, 0, 0));

    // Context/program/scalar source-control proofs keep the Session task
    // completion alive but do not occupy the sole runtime worker or publish
    // candidate state before their native terminal callback.
    auto controlsDesc=Desc(); controlsDesc.nodes.resize(1); controlsDesc.terminal=controlsDesc.nodes.front().path;
    auto& controlCurves=controlsDesc.curveSets.front();
    controlCurves.points={{2,0,0},{2,1,0}}; controlCurves.rest={{0,0,0},{0,1,0}};
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
    addControl("/SessionSourceUseRest","$frame < 3","useRest",TfToken("bool"),expr::ScalarType::Bool,VtValue(true));
    addControl("/SessionSourceResample","$frame < 2 ? 0 : 4","resampleTo",TfToken("int"),expr::ScalarType::Int32,VtValue(0));
    struct SourceGate { void (*arm)(); void (*wait)(); void (*release)(); };
    SourceGate const sourceControlGates[] = {
        {armCudaSourceAsyncContextCallbackGateForTesting,waitCudaSourceAsyncContextCallbackGateForTesting,releaseCudaSourceAsyncContextCallbackGateForTesting},
        {armCudaSourceAsyncProgramCallbackGateForTesting,waitCudaSourceAsyncProgramCallbackGateForTesting,releaseCudaSourceAsyncProgramCallbackGateForTesting},
        {armCudaSourceAsyncScalarCallbackGateForTesting,waitCudaSourceAsyncScalarCallbackGateForTesting,releaseCudaSourceAsyncScalarCallbackGateForTesting},
    };
    size_t sourceControlGateIndex = 0;
    for (auto const& gate : sourceControlGates) {
        auto const lastGood=a.Snapshot(); UsdGenSession::CommitRequest request;
        // Accepted Session cooks now seed the execution cache. Give each gate
        // a distinct semantic tuple so every iteration reaches its intended
        // native source-control phase instead of replaying the first result.
        request.frame=1.0 + 0.125 * sourceControlGateIndex++;
        request.reason=UsdGenCommitReason::SetTime;
        request.spec = usdGenUsd::LowerShared(controlsDesc);
        gate.arm(); ReleaseOperatorGate release{gate.release}; auto heldControl=Submit(a,std::move(request)); CHECK(heldControl);
        gate.wait(); std::atomic<bool> probeDone{false};
        CHECK(cpuProbe.Submit([&](auto const&) { probeDone.store(true,std::memory_order_release); return UsdGenExecutionPipeline::Publish{}; }));
        CHECK(Wait(probeDone) && !heldControl->done.load(std::memory_order_acquire) && heldControl->calls==0 && a.Snapshot()==lastGood);
        gate.release(); release.release=nullptr;
        CHECK(Wait(heldControl->done) && heldControl->calls==1 && heldControl->outcome==UsdGenExecutionPipeline::Outcome::Published &&
              a.Snapshot()==heldControl->snapshot && cudaSourceRelayOccupiedCountForTesting()==0);
        cpuProbe.Drain(); CHECK(WaitForUsage(graph,0,0,0));
    }
    // A proved source-control semantic failure follows the Session failure
    // contract: publish diagnostics with the previous generation, preserve
    // commit stats/dirty state, then admit a valid retry.
    auto invalidControls=controlsDesc;
    for (auto& expression : invalidControls.expressions)
        if (expression.path==SdfPath("/SessionSourceResample")) expression.source="$frame / 0";
    auto const controlsLastGood=a.Snapshot(); UsdGenSession::CommitRequest invalidControlRequest;
    invalidControlRequest.frame=1; invalidControlRequest.reason=UsdGenCommitReason::SetTime;
    invalidControlRequest.spec = usdGenUsd::LowerShared(std::move(invalidControls));
    auto invalidControl=Submit(a,std::move(invalidControlRequest)); CHECK(invalidControl && Wait(invalidControl->done));
    CHECK(invalidControl->calls==1 && invalidControl->outcome==UsdGenExecutionPipeline::Outcome::Failed &&
          invalidControl->snapshot && invalidControl->snapshot->generation==controlsLastGood->generation &&
          invalidControl->snapshot->stats.commits==controlsLastGood->stats.commits &&
          invalidControl->snapshot->diagnostics.HasErrors() && a.NeedsCommit() && cudaSourceRelayOccupiedCountForTesting()==0);
    UsdGenSession::CommitRequest controlRetryRequest;
    controlRetryRequest.frame=1; controlRetryRequest.reason=UsdGenCommitReason::SetTime;
    controlRetryRequest.spec = usdGenUsd::LowerShared(controlsDesc);
    auto controlRetry=Submit(a,std::move(controlRetryRequest)); CHECK(controlRetry && Wait(controlRetry->done));
    CHECK(controlRetry->calls==1 && controlRetry->outcome==UsdGenExecutionPipeline::Outcome::Published &&
          controlRetry->snapshot->generation!=controlsLastGood->generation && !a.NeedsCommit());
    CHECK(WaitForUsage(graph,0,0,0));

    auto resampleDesc=Desc();
    resampleDesc.nodes.front().params.push_back({TfToken("resampleTo"),VtValue(4),false});
    armCudaSourceAsyncResampleCallbackGateForTesting();
    ReleaseResampleGate releaseResampleGate;
    UsdGenSession::CommitRequest heldResampleRequest;
    heldResampleRequest.frame = 2.75; heldResampleRequest.reason = UsdGenCommitReason::SetTime;
    heldResampleRequest.spec = usdGenUsd::LowerShared(std::move(resampleDesc));
    auto heldResample = Submit(a, std::move(heldResampleRequest)); CHECK(heldResample);
    waitCudaSourceAsyncResampleCallbackGateForTesting();
    std::atomic<bool> resampleProbeDone{false};
    bool const resampleProbeAccepted=cpuProbe.Submit([&](auto const&) {
        resampleProbeDone.store(true,std::memory_order_release); return UsdGenExecutionPipeline::Publish{};
    });
    CHECK(resampleProbeAccepted);
    auto const resampleProbeStart=std::chrono::steady_clock::now();
    bool const resampleProbeCompleted=Wait(resampleProbeDone);
    auto const resampleProbeElapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-resampleProbeStart).count();
    bool const resampleDoneBeforeRelease=heldResample->done.load(std::memory_order_acquire);
    int const resampleCallsBeforeRelease=heldResample->calls.load(std::memory_order_acquire);
    int const resampleOutcomeBeforeRelease=resampleDoneBeforeRelease ? int(heldResample->outcome) : -1;
    size_t const resampleOccupiedBeforeRelease=cudaSourceRelayOccupiedCountForTesting();
    size_t const resampleQuarantinedBeforeRelease=cudaSourceRelayQuarantinedCountForTesting();
    if (!resampleProbeCompleted || resampleDoneBeforeRelease || resampleCallsBeforeRelease != 0) {
        std::fprintf(stderr,
            "session resample probe: accepted=%d ran=%d elapsedMs=%lld done=%d calls=%d outcome=%d sourceOccupied=%zu sourceQuarantined=%zu\n",
            resampleProbeAccepted,resampleProbeCompleted,static_cast<long long>(resampleProbeElapsed),
            resampleDoneBeforeRelease,resampleCallsBeforeRelease,resampleOutcomeBeforeRelease,
            resampleOccupiedBeforeRelease,resampleQuarantinedBeforeRelease);
    }
    CHECK(resampleProbeCompleted && !resampleDoneBeforeRelease && resampleCallsBeforeRelease == 0);
    releaseCudaSourceAsyncResampleCallbackGateForTesting();
    CHECK(Wait(heldResample->done) && heldResample->calls == 1 &&
          heldResample->outcome == UsdGenExecutionPipeline::Outcome::Published);
    cpuProbe.Drain();
    CHECK(WaitForUsage(graph, 0, 0, 0));

    // Operator phases keep their Session TaskCompletion alive through each
    // terminal callback. With one runtime worker, unrelated CPU work still
    // progresses and the Session must not publish a candidate before release.
    struct OperatorGate { void (*arm)(); void (*wait)(); void (*release)(); size_t (*occupied)(); };
    OperatorGate const operatorGates[] = {
        {armCudaOperatorAsyncContextCallbackGateForTesting,
         waitCudaOperatorAsyncContextCallbackGateForTesting,
         releaseCudaOperatorAsyncContextCallbackGateForTesting,
         cudaOperatorRelayOccupiedCountForTesting},
        {armCudaOperatorAsyncProgramCallbackGateForTesting,
         waitCudaOperatorAsyncProgramCallbackGateForTesting,
         releaseCudaOperatorAsyncProgramCallbackGateForTesting,
         cudaOperatorRelayOccupiedCountForTesting},
        {armCudaOperatorAsyncWidthCallbackGateForTesting,
         waitCudaOperatorAsyncWidthCallbackGateForTesting,
         releaseCudaOperatorAsyncWidthCallbackGateForTesting,
         cudaOperatorRelayOccupiedCountForTesting},
        {armCudaOperatorAsyncLengthCallbackGateForTesting,
         waitCudaOperatorAsyncLengthCallbackGateForTesting,
         releaseCudaOperatorAsyncLengthCallbackGateForTesting,
         cudaOperatorRelayOccupiedCountForTesting},
        {armCudaOperatorAsyncCountsCallbackGateForTesting,
         waitCudaOperatorAsyncCountsCallbackGateForTesting,
         releaseCudaOperatorAsyncCountsCallbackGateForTesting,
         cudaOperatorRelayOccupiedCountForTesting},
        {armCudaOperatorAsyncScatterCallbackGateForTesting,
         waitCudaOperatorAsyncScatterCallbackGateForTesting,
         releaseCudaOperatorAsyncScatterCallbackGateForTesting,
         cudaOperatorRelayOccupiedCountForTesting},
        {armCudaFinalizationAsyncTileCallbackGateForTesting,
         waitCudaFinalizationAsyncTileCallbackGateForTesting,
         releaseCudaFinalizationAsyncTileCallbackGateForTesting,
         cudaFinalizationRelayOccupiedCountForTesting},
        {armCudaFinalizationAsyncBoundsCallbackGateForTesting,
         waitCudaFinalizationAsyncBoundsCallbackGateForTesting,
         releaseCudaFinalizationAsyncBoundsCallbackGateForTesting,
         cudaFinalizationRelayOccupiedCountForTesting},
        {armCudaFinalizationAsyncMetadataCallbackGateForTesting,
         waitCudaFinalizationAsyncMetadataCallbackGateForTesting,
         releaseCudaFinalizationAsyncMetadataCallbackGateForTesting,
         cudaFinalizationRelayOccupiedCountForTesting},
    };
    for (auto const& gate : operatorGates) {
        auto const lastGood = a.Snapshot();
        UsdGenSession::CommitRequest heldWidthRequest;
        heldWidthRequest.frame = lastGood->generation->frame + .125;
        heldWidthRequest.reason = UsdGenCommitReason::SetTime;
        std::atomic<bool> widthProbeDone{false};
        gate.arm(); ReleaseOperatorGate releaseGate{gate.release};
        auto heldWidth = Submit(a, std::move(heldWidthRequest)); CHECK(heldWidth);
        gate.wait();
        CHECK(cpuProbe.Submit([&](auto const&) {
            widthProbeDone.store(true, std::memory_order_release);
            return UsdGenExecutionPipeline::Publish{};
        }));
        CHECK(Wait(widthProbeDone) && !heldWidth->done.load(std::memory_order_acquire) &&
              heldWidth->calls == 0 && a.Snapshot() == lastGood);
        gate.release(); releaseGate.release = nullptr;
        CHECK(Wait(heldWidth->done) && heldWidth->calls == 1 &&
              heldWidth->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              a.Snapshot() == heldWidth->snapshot &&
              gate.occupied() == 0);
        cpuProbe.Drain();
        CHECK(WaitForUsage(graph, 0, 0, 0));
    }

    // Superseding a Session request while final metadata/topology proof is held
    // retains the shared TaskCompletion until exactly one terminal outcome;
    // the replacement publishes only after that held callback is released.
    auto const lengthLastGood = a.Snapshot();
    UsdGenSession::CommitRequest heldLengthRequest;
    heldLengthRequest.frame = lengthLastGood->generation->frame + .25;
    heldLengthRequest.reason = UsdGenCommitReason::SetTime;
    armCudaFinalizationAsyncMetadataCallbackGateForTesting();
    ReleaseOperatorGate releaseHeldLength{releaseCudaFinalizationAsyncMetadataCallbackGateForTesting};
    auto heldLength = Submit(a, std::move(heldLengthRequest)); CHECK(heldLength);
    waitCudaFinalizationAsyncMetadataCallbackGateForTesting();
    UsdGenSession::CommitRequest replacementLengthRequest;
    replacementLengthRequest.frame = lengthLastGood->generation->frame + .5;
    replacementLengthRequest.reason = UsdGenCommitReason::SetTime;
    auto replacementLength = Submit(a, std::move(replacementLengthRequest)); CHECK(replacementLength);
    releaseCudaFinalizationAsyncMetadataCallbackGateForTesting(); releaseHeldLength.release = nullptr;
    CHECK(Wait(heldLength->done) && Wait(replacementLength->done) &&
          heldLength->calls == 1 && replacementLength->calls == 1 &&
          heldLength->outcome == UsdGenExecutionPipeline::Outcome::Superseded &&
          replacementLength->outcome == UsdGenExecutionPipeline::Outcome::Published &&
          a.Snapshot() == replacementLength->snapshot && a.Snapshot() != lengthLastGood &&
          cudaFinalizationRelayOccupiedCountForTesting() == 0 && WaitForUsage(graph, 0, 0, 0));

    // Finalization admission is shared across sessions. Hold after metadata's
    // native terminal, so A retains the cap-one admission without blocking B's
    // unrelated CUDA setup allocations. B's accepted failure publishes fresh
    // diagnostics but retains its last-good generation; releasing A returns
    // the slot and lets B publish normally.
    setCudaFinalizationRelayCapacityForTesting(1); ResetFinalizationRelayCapacity resetFinalCapacity;
    auto const aBeforeCap = a.Snapshot(), bBeforeCap = b.Snapshot();
    UsdGenSession::CommitRequest capARequest;
    capARequest.frame = aBeforeCap->generation->frame + .125;
    capARequest.reason = UsdGenCommitReason::SetTime;
    armCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
    ReleaseOperatorGate releaseCapTile{releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting};
    auto capA = Submit(a, std::move(capARequest)); CHECK(capA);
    waitCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
    UsdGenSession::CommitRequest capBRequest;
    capBRequest.frame = bBeforeCap->generation->frame + .125;
    capBRequest.reason = UsdGenCommitReason::SetTime;
    auto capB = Submit(b, std::move(capBRequest)); CHECK(capB);
    bool const capBCompleted = Wait(capB->done);
    bool const capACompleted = capA->done.load(std::memory_order_acquire);
    int const capAOutcome = capACompleted ? int(capA->outcome) : -1;
    int const capBOutcome = capBCompleted ? int(capB->outcome) : -1;
    auto const bAfterCap = b.Snapshot();
    bool const capBExpected = capBCompleted && capB->calls.load(std::memory_order_acquire) == 1 &&
        capBOutcome == int(UsdGenExecutionPipeline::Outcome::Failed) && bAfterCap &&
        bAfterCap == capB->snapshot && bAfterCap != bBeforeCap &&
        bAfterCap->generation == bBeforeCap->generation &&
        bAfterCap->stats.commits == bBeforeCap->stats.commits &&
        bAfterCap->diagnostics.HasErrors() && b.NeedsCommit() &&
        !capACompleted && a.Snapshot() == aBeforeCap;
    if (!capBExpected) {
        auto const usage = graph->GetAdmissionUsage();
        std::fprintf(stderr,
            "cap-one session: A calls=%d done=%d outcome=%d B calls=%d done=%d outcome=%d BsnapshotSame=%d BgenerationSame=%d commits=%llu/%llu errors=%d needs=%d frames A=%g Bbefore=%g Bafter=%g graph=%u/%u/%u source=%zu operator=%zu final=%zu\n",
            capA->calls.load(std::memory_order_acquire), int(capACompleted), capAOutcome,
            capB->calls.load(std::memory_order_acquire), int(capBCompleted), capBOutcome,
            int(bAfterCap == bBeforeCap), int(bAfterCap && bAfterCap->generation == bBeforeCap->generation),
            static_cast<unsigned long long>(bAfterCap ? bAfterCap->stats.commits : 0),
            static_cast<unsigned long long>(bBeforeCap->stats.commits),
            int(bAfterCap && bAfterCap->diagnostics.HasErrors()), int(b.NeedsCommit()),
            aBeforeCap->generation->frame, bBeforeCap->generation->frame,
            bAfterCap && bAfterCap->generation ? bAfterCap->generation->frame : -1.0,
            usage.jobs, usage.tasks, usage.dependencyEdges,
            cudaSourceRelayOccupiedCountForTesting(), cudaOperatorRelayOccupiedCountForTesting(),
            cudaFinalizationRelayOccupiedCountForTesting());
    }
    CHECK(capBExpected);
    releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting(); releaseCapTile.release = nullptr;
    CHECK(Wait(capA->done) && capA->calls == 1 &&
          capA->outcome == UsdGenExecutionPipeline::Outcome::Published &&
          cudaFinalizationRelayOccupiedCountForTesting() == 0);
    UsdGenSession::CommitRequest capRetryRequest;
    capRetryRequest.frame = bBeforeCap->generation->frame + .25;
    capRetryRequest.reason = UsdGenCommitReason::SetTime;
    auto capRetry = Submit(b, std::move(capRetryRequest));
    CHECK(capRetry && Wait(capRetry->done) && capRetry->calls == 1 &&
          capRetry->outcome == UsdGenExecutionPipeline::Outcome::Published &&
          b.Snapshot() == capRetry->snapshot && cudaFinalizationRelayOccupiedCountForTesting() == 0 &&
          WaitForUsage(graph, 0, 0, 0));

    // The older admitted cook loses to a newer request while all task slots are held.
    roots = std::make_shared<HeldRoots>(); releaseOnExit.roots = roots; CHECK(HoldDispatcher(graph, roots));
    // Earlier callback-gate coverage advances in .125-frame steps and now
    // seeds the execution cache, including frame 3. Use an otherwise unseen
    // tuple so this block continues to exercise admitted task supersession
    // rather than the intentional zero-execution cache-hit path.
    UsdGenSession::CommitRequest oldRequest; oldRequest.frame = 13; oldRequest.reason = UsdGenCommitReason::SetTime;
    auto superseded = Submit(a, std::move(oldRequest));
    CHECK(superseded && WaitForUsage(graph, 2, 13, 13));
    UsdGenSession::CommitRequest intermediateRequest;
    intermediateRequest.frame = 13.5; intermediateRequest.reason = UsdGenCommitReason::SetTime;
    auto intermediate = Submit(a, std::move(intermediateRequest)); CHECK(intermediate);
    UsdGenSession::CommitRequest newRequest; newRequest.frame = 14; newRequest.reason = UsdGenCommitReason::SetTime;
    auto newest = Submit(a, std::move(newRequest)); CHECK(newest && Wait(intermediate->done));
    CHECK(intermediate->calls == 1 && intermediate->outcome == UsdGenExecutionPipeline::Outcome::Superseded);
    CHECK(!superseded->done.load(std::memory_order_acquire) && !newest->done.load(std::memory_order_acquire));
    roots->Release(); CHECK(Wait(superseded->done) && Wait(newest->done));
    CHECK(superseded->calls == 1 && newest->calls == 1);
    CHECK(superseded->outcome == UsdGenExecutionPipeline::Outcome::Superseded);
    CHECK(newest->outcome == UsdGenExecutionPipeline::Outcome::Published);
    CHECK(newest->snapshot && newest->snapshot->generation && newest->snapshot->generation->frame == 14);
    CHECK(a.Snapshot() == newest->snapshot && WaitForUsage(graph, 0, 0, 0));

    auto beforeBad = a.Snapshot(); CHECK(beforeBad && beforeBad->generation);
    auto malformedDesc = Desc();
    UsdGenExpressionDesc nonFinite;
    nonFinite.path = SdfPath("/SessionStages/NonFiniteWidth");
    nonFinite.source = "$frame / ($frame - 5)";
    const expr::ValueShape scalar{expr::ScalarType::Float32, 1, 1, 1, 1, false};
    nonFinite.outputs.push_back({TfToken("result"), TfToken("float"), scalar});
    malformedDesc.expressions.push_back(nonFinite);
    UsdGenExpressionBinding nonFiniteBinding;
    nonFiniteBinding.expression = nonFinite.path; nonFiniteBinding.destination = TfToken("width");
    nonFiniteBinding.nativeType = TfToken("float"); nonFiniteBinding.destinationShape = scalar;
    nonFiniteBinding.domain = expr::Domain::Point; nonFiniteBinding.literal = VtValue(.03f);
    malformedDesc.nodes[2].expressionBindings.push_back(nonFiniteBinding);
    UsdGenSession::CommitRequest badRequest; badRequest.frame = 5; badRequest.reason = UsdGenCommitReason::SetTime;
    badRequest.spec = usdGenUsd::LowerShared(std::move(malformedDesc));
    roots = std::make_shared<HeldRoots>(); releaseOnExit.roots = roots;
    CHECK(HoldDispatcher(graph, roots));
    auto bad = Submit(a, std::move(badRequest));
    // Admission proves compilation succeeded and the expression reaches the
    // real CUDA Width stage rather than failing during descriptor preparation.
    CHECK(bad && WaitForUsage(graph, 2, 13, 13));
    roots->Release();
    CHECK(Wait(bad->done) && bad->calls == 1);
    CHECK(bad->outcome == UsdGenExecutionPipeline::Outcome::Failed);
    CHECK(bad->snapshot && bad->snapshot->generation == beforeBad->generation);
    CHECK(bad->snapshot->stats.commits == beforeBad->stats.commits);
    CHECK(usdGen::UsdGenCurveDirtyReportOf(bad->snapshot->report).tiles.empty() &&
          !usdGen::UsdGenCurveDirtyReportOf(bad->snapshot->report).surfaceXformDirty &&
          bad->snapshot->diagnostics.HasErrors() && a.NeedsCommit());
    UsdGenSession::CommitRequest retryRequest; retryRequest.frame = 6; retryRequest.reason = UsdGenCommitReason::SetTime;
    retryRequest.spec = usdGenUsd::LowerShared(Desc());
    auto retry = Submit(a, std::move(retryRequest)); CHECK(retry && Wait(retry->done) && retry->calls == 1);
    CHECK(retry->outcome == UsdGenExecutionPipeline::Outcome::Published);
    CHECK(retry->snapshot && retry->snapshot->generation != beforeBad->generation);
    CHECK(retry->snapshot->stats.commits == beforeBad->stats.commits + 1 && !a.NeedsCommit());
    CHECK(WaitForUsage(graph, 0, 0, 0));

    // Saturate the real shared dispatcher, then submit a Session request.  A
    // rejected task-graph Submit must still deliver the Session's terminal
    // callback; otherwise this request would silently hang forever.
    auto beforeRejected = a.Snapshot();
    CHECK(beforeRejected && beforeRejected->generation);
    roots = std::make_shared<HeldRoots>(); releaseOnExit.roots = roots;
    CHECK(HoldDispatcher(graph, roots));
    for (int i = 0; i != 255; ++i) CHECK(SubmitNoop(graph));
    CHECK(WaitForUsage(graph, 256, 264, 8));
    UsdGenSession::CommitRequest rejectedRequest;
    rejectedRequest.frame = 7; rejectedRequest.reason = UsdGenCommitReason::SetTime;
    auto rejected = Submit(a, std::move(rejectedRequest));
    CHECK(rejected && Wait(rejected->done) && rejected->calls == 1);
    CHECK(rejected->outcome == UsdGenExecutionPipeline::Outcome::Failed);
    CHECK(rejected->snapshot && rejected->snapshot->generation == beforeRejected->generation);
    CHECK(rejected->snapshot->stats.commits == beforeRejected->stats.commits && a.NeedsCommit());
    CHECK(HasDiagnostic(rejected->snapshot, "CUDA dispatcher rejected job"));
    roots->Release();
    CHECK(WaitForUsage(graph, 0, 0, 0));
    UsdGenSession::CommitRequest finalRetry;
    finalRetry.frame = 8; finalRetry.reason = UsdGenCommitReason::SetTime;
    auto final = Submit(a, std::move(finalRetry));
    CHECK(final && Wait(final->done) && final->calls == 1);
    CHECK(final->outcome == UsdGenExecutionPipeline::Outcome::Published);
    CHECK(final->snapshot && final->snapshot->generation != beforeRejected->generation);
    CHECK(final->snapshot->stats.commits == beforeRejected->stats.commits + 1 && !a.NeedsCommit());
    CHECK(WaitForUsage(graph, 0, 0, 0));
    releaseOnExit.roots->Release(); CHECK(cudaStreamDestroy(reader) == cudaSuccess);
    return 0;
}
