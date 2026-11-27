#include "usdGenCurves/cuda/cudaExecution.h"
#include "cudaPlanRunner.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "cudaExecutionQueueFixture.h"
#include "usdGen/executionResources.h"
#include "usdGenCurves/cuda/generation.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static size_t ResourceKindBytes(UsdGenExecutionResourceSnapshot const& snapshot,
                                UsdGenExecutionResourceKind kind) {
    return snapshot.byKind[static_cast<size_t>(kind)];
}

static bool ParseLiteralRbfPeak(UsdGenDiagnostics const& diagnostics,
                                uint64_t* peak) {
    if (!peak || diagnostics.errors.size() != 1 || !diagnostics.warnings.empty())
        return false;
    std::string const marker = "literal-RBF memory reservation of ";
    auto const begin = diagnostics.errors.front().find(marker);
    if (begin == std::string::npos) return false;
    auto const number = begin + marker.size();
    auto const end = diagnostics.errors.front().find(" bytes", number);
    if (end == std::string::npos || end == number) return false;
    try {
        size_t consumed = 0;
        auto const value = std::stoull(diagnostics.errors.front().substr(
            number, end - number), &consumed);
        if (consumed != end - number || value == 0) return false;
        *peak = static_cast<uint64_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

namespace {

UsdGenGraphDesc RbfDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/RbfAtomic/Hair");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/RbfAtomic/Scalp");
    surface.restPoints = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,1}};
    surface.points = surface.restPoints;
    for (auto& point : surface.points) point[0] += .25f;
    surface.faceVertexCounts = {3,3,3};
    surface.faceVertexIndices = {0,1,2, 0,1,3, 1,2,4};
    desc.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/RbfAtomic/Curves");
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{.2f,.2f,0}, {.2f,.4f,0}};
    curves.rest = curves.points;
    curves.curveId = {42};
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.2f,.2f}};
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/RbfAtomic/Source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.surfaces = {surface.path};
    UsdGenNodeDesc deform;
    deform.path = SdfPath("/RbfAtomic/Deform");
    deform.type = TfToken("UsdGenDeform");
    deform.inputs = {source.path};
    deform.surfaces = {surface.path};
    deform.params.push_back({TfToken("rbfSamples"), VtValue(5), false});
    desc.nodes = {source, deform};
    desc.terminal = deform.path;
    return desc;
}

UsdGenGraphDesc ChangedRestDesc() {
    auto desc = RbfDesc();
    desc.surfaces.front().restPoints.back()[2] += .25f;
    desc.surfaces.front().points.back()[2] += .25f;
    return desc;
}

UsdGenGraphDesc ChangedBudgetDesc() {
    auto desc = ChangedRestDesc();
    desc.nodes.back().params.front().value = VtValue(4);
    return desc;
}

bool SameStats(std::vector<UsdGenCudaBindingStats> const& a,
               std::vector<UsdGenCudaBindingStats> const& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i != a.size(); ++i)
        if (a[i].key != b[i].key || a[i].identity != b[i].identity ||
            a[i].bindCount != b[i].bindCount ||
            a[i].solveCount != b[i].solveCount ||
            a[i].sampleCount != b[i].sampleCount) return false;
    return true;
}

bool SameStatsAccounting(std::vector<UsdGenCudaBindingStats> const& a,
                         std::vector<UsdGenCudaBindingStats> const& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i != a.size(); ++i)
        if (a[i].key != b[i].key || a[i].bindCount != b[i].bindCount ||
            a[i].solveCount != b[i].solveCount ||
            a[i].sampleCount != b[i].sampleCount) return false;
    return true;
}

bool SameGeneration(std::shared_ptr<const UsdGenDeviceGeneration> const& a,
                    std::shared_ptr<const UsdGenDeviceGeneration> const& b) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return false;
    bool same = false;
    {
        auto aa = gpu::AcquireGeometry(a, stream);
        auto bb = gpu::AcquireGeometry(b, stream);
        if (aa && bb && aa.Geometry().pointCount == bb.Geometry().pointCount &&
            aa.Geometry().curveCount == bb.Geometry().curveCount) {
            std::vector<float3> ap(aa.Geometry().pointCount), bp(bb.Geometry().pointCount);
            std::vector<float> aw(aa.Geometry().pointCount), bw(bb.Geometry().pointCount);
            auto const pointBytes = ap.size() * sizeof(float3);
            auto const widthBytes = aw.size() * sizeof(float);
            bool copied = true;
            if (pointBytes)
                copied = cudaMemcpyAsync(ap.data(), aa.Geometry().points.data, pointBytes,
                                         cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                         cudaMemcpyAsync(bp.data(), bb.Geometry().points.data, pointBytes,
                                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
            if (copied && widthBytes)
                copied = cudaMemcpyAsync(aw.data(), aa.Geometry().widths.data, widthBytes,
                                         cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                         cudaMemcpyAsync(bw.data(), bb.Geometry().widths.data, widthBytes,
                                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
            same = copied && cudaStreamSynchronize(stream) == cudaSuccess;
            for (size_t i = 0; same && i != ap.size(); ++i)
                same = std::fabs(ap[i].x - bp[i].x) < 1e-5f &&
                       std::fabs(ap[i].y - bp[i].y) < 1e-5f &&
                       std::fabs(ap[i].z - bp[i].z) < 1e-5f &&
                       std::fabs(aw[i] - bw[i]) < 1e-5f;
            auto equalView = [&](auto av, auto bv) {
                if (av.size != bv.size) return false;
                size_t const bytes = av.size * sizeof(*av.data);
                if (!bytes) return true;
                if (!av.data || !bv.data) return false;
                std::vector<unsigned char> left(bytes), right(bytes);
                return cudaMemcpyAsync(left.data(), av.data, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaMemcpyAsync(right.data(), bv.data, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess && left == right;
            };
            same = same && equalView(aa.Geometry().restPoints, bb.Geometry().restPoints) &&
                equalView(aa.Geometry().curveOffsets, bb.Geometry().curveOffsets) &&
                equalView(aa.Geometry().stableIds, bb.Geometry().stableIds) &&
                equalView(aa.HairT(), bb.HairT()) && equalView(aa.RootPrim(), bb.RootPrim()) &&
                equalView(aa.RootUV(), bb.RootUV()) && equalView(aa.RootT(), bb.RootT()) &&
                equalView(aa.RootB(), bb.RootB()) && equalView(aa.RootN(), bb.RootN()) &&
                a->Geometry().alreadyDeformed == b->Geometry().alreadyDeformed;
            for (auto const& channel : a->Channels()) {
                if (!same || channel.semantic != UsdGenDeviceChannelSemantic::Generic) continue;
                auto left = gpu::AcquireNamedChannel(a, channel.name, stream);
                auto right = gpu::AcquireNamedChannel(b, channel.name, stream);
                same = left && right && left.Metadata()->type == right.Metadata()->type &&
                    left.Metadata()->domain == right.Metadata()->domain &&
                    left.Metadata()->arity == right.Metadata()->arity && equalView(left.Bytes(), right.Bytes());
            }
        }
    }
    return cudaStreamDestroy(stream) == cudaSuccess && same;
}

struct Attempts {
    uint64_t rbfAccept = 0, rbfRollback = 0;
    uint64_t surfaceAccept = 0, surfaceRollback = 0;
};

Attempts GetAttempts() {
    return {cudaRbfFieldAcceptAttemptCountForTesting(),
            cudaRbfFieldRollbackAttemptCountForTesting(),
            cudaRbfSurfaceAcceptAttemptCountForTesting(),
            cudaRbfSurfaceRollbackAttemptCountForTesting()};
}

enum class Resolution { AcceptPreflight, RollbackPreflight,
                        AcceptCommit, RollbackCommit };
enum class Side { Rbf, Surface };

bool FailureCase(std::shared_ptr<const UsdGenCudaExecutionPlan> const& plan,
                 Resolution resolution, Side side) {
    auto require = [&](bool condition, char const* stage) {
        if (!condition)
            std::fprintf(stderr, "transaction case failed at %s (resolution=%d side=%d)\n",
                         stage, int(resolution), int(side));
        return condition;
    };
    UsdGenExecutionRuntime runtime(4);
    UsdGenDiagnostics diagnostics;
    UsdGenCudaExecutionQueue queue(runtime, -1, &diagnostics);
    if (!require(queue.Valid(), "queue creation")) return false;
    std::atomic<int> initialCalls{0};
    std::atomic<bool> initialPublished{false};
    if (!require(queue.Submit(plan, 1.0, [&](uint64_t, auto outcome, auto const&) {
            initialPublished.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
            initialCalls.fetch_add(1, std::memory_order_release);
        }), "initial submit")) return false;
    queue.Drain();
    auto const lastGood = queue.Snapshot();
    if (!require(initialCalls.load(std::memory_order_acquire) == 1 &&
        initialPublished.load(std::memory_order_acquire) && lastGood &&
        lastGood->generation && lastGood->bindings.size() == 1 &&
        lastGood->bindings[0].identity,
        "initial publication")) return false;

    auto const acceptedStats = lastGood->bindings;
    auto const before = GetAttempts();
    auto const quarantined = cudaFinalizationRelayQuarantinedCountForTesting();
    bool const rollback = resolution == Resolution::RollbackPreflight ||
                          resolution == Resolution::RollbackCommit;
    bool const preflight = resolution == Resolution::AcceptPreflight ||
                           resolution == Resolution::RollbackPreflight;
    if (rollback) failNextCudaFinalizationRelayAllocationForTesting();
    if (side == Side::Rbf) {
        if (preflight) failNextCudaRbfFieldResolvePreflightForTesting();
        else failNextCudaRbfFieldResolveCommitForTesting();
    } else {
        if (preflight) failNextCudaRbfSurfaceResolvePreflightForTesting();
        else failNextCudaRbfSurfaceResolveCommitForTesting();
    }

    std::atomic<int> failureCalls{0};
    std::atomic<bool> failed{false};
    if (!require(queue.Submit(plan, 2.0, [&](uint64_t, auto outcome, auto const&) {
            failed.store(outcome == UsdGenExecutionPipeline::Outcome::Failed,
                         std::memory_order_release);
            failureCalls.fetch_add(1, std::memory_order_release);
        }), "fault submit")) return false;
    queue.Drain();
    auto const after = GetAttempts();
    bool const failureOutcomeOk = failureCalls.load(std::memory_order_acquire) == 1 &&
        failed.load(std::memory_order_acquire) && queue.Snapshot() == lastGood &&
        SameStats(queue.Snapshot()->bindings, acceptedStats) &&
        cudaFinalizationRelayQuarantinedCountForTesting() == quarantined + 1;
    if (!failureOutcomeOk)
        std::fprintf(stderr,
            "  calls=%d failed=%d sameSnapshot=%d sameStats=%d quarantine=%zu->%zu\n",
            failureCalls.load(std::memory_order_acquire),
            int(failed.load(std::memory_order_acquire)),
            int(queue.Snapshot() == lastGood),
            int(queue.Snapshot() && SameStats(queue.Snapshot()->bindings, acceptedStats)),
            quarantined, cudaFinalizationRelayQuarantinedCountForTesting());
    if (!require(failureOutcomeOk,
        "failure outcome/retention/quarantine")) return false;

    if (preflight) {
        if (!require(after.rbfAccept == before.rbfAccept &&
            after.rbfRollback == before.rbfRollback &&
            after.surfaceAccept == before.surfaceAccept &&
            after.surfaceRollback == before.surfaceRollback,
            "preflight made no concrete calls")) return false;
    } else if (rollback) {
        if (!require(after.rbfRollback == before.rbfRollback + 1 &&
            after.surfaceRollback == before.surfaceRollback + 1 &&
            after.rbfAccept == before.rbfAccept &&
            after.surfaceAccept == before.surfaceAccept,
            "rollback attempted both resources")) return false;
    } else {
        if (!require(after.rbfAccept == before.rbfAccept + 1 &&
            after.surfaceAccept == before.surfaceAccept + 1 &&
            after.rbfRollback == before.rbfRollback &&
            after.surfaceRollback == before.surfaceRollback,
            "accept attempted both resources")) return false;
    }

    // Resolution failure poisons the workspace. A later request reports one
    // failure and cannot replace the exact last-good queue snapshot.
    std::atomic<int> retryCalls{0};
    std::atomic<bool> retryFailed{false};
    if (!require(queue.Submit(plan, 3.0, [&](uint64_t, auto outcome, auto const&) {
            retryFailed.store(outcome == UsdGenExecutionPipeline::Outcome::Failed,
                              std::memory_order_release);
            retryCalls.fetch_add(1, std::memory_order_release);
        }), "poisoned retry submit")) return false;
    queue.Drain();
    return require(retryCalls.load(std::memory_order_acquire) == 1 &&
        retryFailed.load(std::memory_order_acquire) &&
        queue.Snapshot() == lastGood &&
        SameStats(queue.Snapshot()->bindings, acceptedStats),
        "poisoned retry outcome");
}

int SynchronousTransactions(
    std::shared_ptr<const UsdGenCudaExecutionPlan> const& plan) {
    UsdGenDiagnostics diagnostics;
    auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(workspace);
    int resourceDevice = -1;
    CHECK(cudaGetDevice(&resourceDevice) == cudaSuccess);
    auto resources = FindUsdGenExecutionResourcePool(
        {usdGenTestUtils::Backend("cuda"), resourceDevice});
    CHECK(resources);
    auto const workspaceBaseline = resources->Snapshot();
    auto first = ExecuteCudaPlanForTesting(plan, *workspace, 20.0, 200, &diagnostics);
    CHECK(first && !diagnostics.HasErrors());
    auto const firstStats = GetCudaBindingStats(*workspace);
    CHECK(firstStats.size() == 1 && firstStats.front().identity &&
          firstStats.front().bindCount == 1 && firstStats.front().solveCount == 1 &&
          firstStats.front().sampleCount == 5);
    auto const firstAcceptedResources = resources->Snapshot();
    CHECK(firstAcceptedResources.usedBytes > workspaceBaseline.usedBytes &&
          ResourceKindBytes(firstAcceptedResources,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(workspaceBaseline,
                  UsdGenExecutionResourceKind::Pending));

    // The literal cold Source->RBF plan is admitted with one precharged
    // execution ticket. Saturate the baseline once to learn the exact bound
    // from the production diagnostic, then prove a peak-minus-one filler
    // rejects before source/device work or cache mutation.
    auto const rbfBaseline = resources->Snapshot();
    CHECK(rbfBaseline.usableBytes >= rbfBaseline.usedBytes);
    auto const rbfAvailable = rbfBaseline.usableBytes - rbfBaseline.usedBytes;
    auto fullFiller = resources->TryReserve(
        rbfAvailable, UsdGenExecutionResourceKind::Active);
    CHECK(fullFiller);
    auto const fullSaturated = resources->Snapshot();
    UsdGenDiagnostics fullRejectDiagnostics;
    auto fullReject = ExecuteCudaPlanForTesting(plan, *workspace, 20.5, 2001,
                                       &fullRejectDiagnostics, first);
    uint64_t rbfPeak = 0;
    CHECK(!fullReject && ParseLiteralRbfPeak(fullRejectDiagnostics, &rbfPeak));
    auto const afterFullReject = resources->Snapshot();
    CHECK(afterFullReject.usedBytes == fullSaturated.usedBytes &&
          ResourceKindBytes(afterFullReject,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(fullSaturated,
                  UsdGenExecutionResourceKind::Pending) &&
          SameStats(GetCudaBindingStats(*workspace), firstStats));
    fullFiller->Release();
    CHECK(resources->Snapshot().usedBytes == rbfBaseline.usedBytes &&
          rbfPeak <= rbfAvailable);

    auto peakFiller = resources->TryReserve(
        rbfAvailable - static_cast<size_t>(rbfPeak) + 1,
        UsdGenExecutionResourceKind::Active);
    CHECK(peakFiller);
    auto const peakSaturated = resources->Snapshot();
    UsdGenDiagnostics peakRejectDiagnostics;
    auto peakReject = ExecuteCudaPlanForTesting(plan, *workspace, 20.75, 2002,
                                       &peakRejectDiagnostics, first);
    uint64_t repeatedPeak = 0;
    CHECK(!peakReject && ParseLiteralRbfPeak(peakRejectDiagnostics, &repeatedPeak) &&
          repeatedPeak == rbfPeak);
    auto const afterPeakReject = resources->Snapshot();
    CHECK(afterPeakReject.usedBytes == peakSaturated.usedBytes &&
          ResourceKindBytes(afterPeakReject,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(peakSaturated,
                  UsdGenExecutionResourceKind::Pending) &&
          SameStats(GetCudaBindingStats(*workspace), firstStats));
    peakFiller->Release();
    CHECK(resources->Snapshot().usedBytes == rbfBaseline.usedBytes);

    // Releasing the filler admits the warm update. Its cache identity remains
    // stable, the output owner is replaced privately, and Pending is closed
    // before the synchronous call returns.
    UsdGenDiagnostics warmDiagnostics;
    auto warm = ExecuteCudaPlanForTesting(plan, *workspace, 20.75, 2003,
                                 &warmDiagnostics, first);
    auto const warmStats = GetCudaBindingStats(*workspace);
    CHECK(warm && !warmDiagnostics.HasErrors() && warm->Owner() &&
          warm->Owner() != first->Owner() && warmStats.size() == 1 &&
          warmStats.front().identity == firstStats.front().identity &&
          warmStats.front().solveCount == firstStats.front().solveCount + 1 &&
          ResourceKindBytes(resources->Snapshot(),
              UsdGenExecutionResourceKind::Pending) == 0 &&
          resources->Snapshot().usedBytes > rbfBaseline.usedBytes &&
          resources->Snapshot().usedBytes <= rbfBaseline.usedBytes + rbfPeak);

    // The asynchronous job ticket uses the same cold bound and is acquired
    // before source submission. A peak-minus-one filler therefore rejects
    // without constructing a cache candidate or changing Pending.
    auto asyncWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(asyncWorkspace);
    auto const asyncBaseline = resources->Snapshot();
    CHECK(asyncBaseline.usableBytes >= asyncBaseline.usedBytes + rbfPeak);
    auto asyncFiller = resources->TryReserve(
        asyncBaseline.usableBytes - asyncBaseline.usedBytes -
            static_cast<size_t>(rbfPeak) + 1,
        UsdGenExecutionResourceKind::Active);
    CHECK(asyncFiller);
    auto const asyncSaturated = resources->Snapshot();
    UsdGenDiagnostics asyncRejectDiagnostics;
    auto asyncRejected = CreateCudaExecutionJob(
        plan, *asyncWorkspace, 20.75, 2004, &asyncRejectDiagnostics, warm);
    uint64_t asyncPeak = 0;
    CHECK(!asyncRejected && ParseLiteralRbfPeak(asyncRejectDiagnostics, &asyncPeak) &&
          asyncPeak == rbfPeak && GetCudaBindingStats(*asyncWorkspace).empty());
    auto const afterAsyncReject = resources->Snapshot();
    CHECK(afterAsyncReject.usedBytes == asyncSaturated.usedBytes &&
          ResourceKindBytes(afterAsyncReject,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(asyncSaturated,
                  UsdGenExecutionResourceKind::Pending));
    asyncFiller->Release();
    CHECK(resources->Snapshot().usedBytes == asyncBaseline.usedBytes);
    UsdGenDiagnostics asyncAdmissionDiagnostics;
    auto asyncAdmitted = CreateCudaExecutionJob(
        plan, *asyncWorkspace, 20.75, 2005, &asyncAdmissionDiagnostics, warm);
    CHECK(asyncAdmitted);
    auto const asyncHeld = resources->Snapshot();
    CHECK(ResourceKindBytes(asyncHeld, UsdGenExecutionResourceKind::Pending) >
          ResourceKindBytes(asyncBaseline, UsdGenExecutionResourceKind::Pending));
    CHECK(RunCudaJobSourceForTesting(asyncAdmitted));
    for (size_t i = 0; i != CudaExecutionJobOperatorCount(*asyncAdmitted); ++i)
        CHECK(RunCudaJobOperatorForTesting(asyncAdmitted, i));
    auto asyncGeneration = FinalizeCudaJobForTesting(asyncAdmitted);
    CHECK(asyncGeneration && !asyncAdmissionDiagnostics.HasErrors());
    auto const afterAsyncAdmission = resources->Snapshot();
    CHECK(ResourceKindBytes(afterAsyncAdmission,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(asyncBaseline,
                  UsdGenExecutionResourceKind::Pending) &&
          GetCudaBindingStats(*asyncWorkspace).size() == 1);
    asyncAdmitted.reset();
    // The compatibility entry point publishes the same geometry and binding
    // accounting as the queued path for the same graph and frame.
    UsdGenExecutionRuntime runtime(4);
    UsdGenDiagnostics asyncDiagnostics;
    UsdGenCudaExecutionQueue queue(runtime, -1, &asyncDiagnostics);
    CHECK(queue.Valid() && queue.Submit(plan, 20.0));
    queue.Drain();
    auto const async = queue.Snapshot();
    CHECK(async && async->generation && SameGeneration(first, async->generation) &&
          SameStatsAccounting(firstStats, async->bindings));

    // A failure after the fresh field and surface solves committed must roll
    // both back. The caller's exact prior publication and the workspace's
    // accepted cache identity/counters remain unchanged.
    auto const firstPointer = first;
    auto const beforeFailureAttempts = GetAttempts();
    auto const beforeFailureResources = resources->Snapshot();
    failNextCudaRbfFinalizationForTesting();
    UsdGenDiagnostics failedDiagnostics;
    auto failed = ExecuteCudaPlanForTesting(plan, *workspace, 21.0, 201,
                                   &failedDiagnostics, firstPointer);
    auto const afterFailureAttempts = GetAttempts();
    auto const afterFailureResources = resources->Snapshot();
    if (afterFailureResources.usedBytes != beforeFailureResources.usedBytes ||
        ResourceKindBytes(afterFailureResources, UsdGenExecutionResourceKind::Pending) !=
            ResourceKindBytes(beforeFailureResources, UsdGenExecutionResourceKind::Pending)) {
        std::fprintf(stderr,
            "  rollback resources used=%zu->%zu pending=%zu->%zu cache=%zu->%zu scratch=%zu->%zu active=%zu->%zu\n",
            beforeFailureResources.usedBytes, afterFailureResources.usedBytes,
            ResourceKindBytes(beforeFailureResources, UsdGenExecutionResourceKind::Pending),
            ResourceKindBytes(afterFailureResources, UsdGenExecutionResourceKind::Pending),
            ResourceKindBytes(beforeFailureResources, UsdGenExecutionResourceKind::Cache),
            ResourceKindBytes(afterFailureResources, UsdGenExecutionResourceKind::Cache),
            ResourceKindBytes(beforeFailureResources, UsdGenExecutionResourceKind::Scratch),
            ResourceKindBytes(afterFailureResources, UsdGenExecutionResourceKind::Scratch),
            ResourceKindBytes(beforeFailureResources, UsdGenExecutionResourceKind::Active),
            ResourceKindBytes(afterFailureResources, UsdGenExecutionResourceKind::Active));
    }
    if (failed || firstPointer != first || !SameGeneration(firstPointer, first) ||
        !SameStats(GetCudaBindingStats(*workspace), warmStats) ||
        afterFailureAttempts.rbfAccept != beforeFailureAttempts.rbfAccept ||
        afterFailureAttempts.surfaceAccept != beforeFailureAttempts.surfaceAccept ||
        afterFailureAttempts.rbfRollback != beforeFailureAttempts.rbfRollback + 1 ||
        afterFailureAttempts.surfaceRollback != beforeFailureAttempts.surfaceRollback + 1) {
        std::fprintf(stderr,
            "  rollback state result=%d pointer=%d generation=%d stats=%d attempts rbfA=%llu->%llu surfaceA=%llu->%llu rbfR=%llu->%llu surfaceR=%llu->%llu\n",
            int(bool(failed)), int(firstPointer == first), int(SameGeneration(firstPointer, first)),
            int(SameStats(GetCudaBindingStats(*workspace), firstStats)),
            static_cast<unsigned long long>(beforeFailureAttempts.rbfAccept),
            static_cast<unsigned long long>(afterFailureAttempts.rbfAccept),
            static_cast<unsigned long long>(beforeFailureAttempts.surfaceAccept),
            static_cast<unsigned long long>(afterFailureAttempts.surfaceAccept),
            static_cast<unsigned long long>(beforeFailureAttempts.rbfRollback),
            static_cast<unsigned long long>(afterFailureAttempts.rbfRollback),
            static_cast<unsigned long long>(beforeFailureAttempts.surfaceRollback),
            static_cast<unsigned long long>(afterFailureAttempts.surfaceRollback));
    }
    CHECK(!failed && firstPointer == first && SameGeneration(firstPointer, first) &&
          SameStats(GetCudaBindingStats(*workspace), warmStats) &&
          afterFailureResources.usedBytes == beforeFailureResources.usedBytes &&
          ResourceKindBytes(afterFailureResources,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(beforeFailureResources,
                  UsdGenExecutionResourceKind::Pending) &&
          afterFailureAttempts.rbfAccept == beforeFailureAttempts.rbfAccept &&
          afterFailureAttempts.surfaceAccept == beforeFailureAttempts.surfaceAccept &&
          afterFailureAttempts.rbfRollback == beforeFailureAttempts.rbfRollback + 1 &&
          afterFailureAttempts.surfaceRollback == beforeFailureAttempts.surfaceRollback + 1);

    UsdGenDiagnostics retryDiagnostics;
    auto retry = ExecuteCudaPlanForTesting(plan, *workspace, 21.0, 202,
                                  &retryDiagnostics, firstPointer);
    auto const retryStats = GetCudaBindingStats(*workspace);
    CHECK(retry && !retryDiagnostics.HasErrors() && retryStats.size() == 1 &&
          retryStats.front().identity == warmStats.front().identity &&
          retryStats.front().bindCount == warmStats.front().bindCount &&
          retryStats.front().solveCount == warmStats.front().solveCount + 1 &&
          retryStats.front().sampleCount == warmStats.front().sampleCount);

    // A rejected changed-rest transaction cannot replace the accepted key.
    // Its clean retry installs a fresh identity and resets that binding's
    // counters, exactly as the queued transaction does.
    UsdGenDiagnostics changedRestCompileDiagnostics;
    auto changedRestPlan = CompileCudaGraph(usdGenUsd::Lower(ChangedRestDesc()),
                                            &changedRestCompileDiagnostics);
    CHECK(changedRestPlan && !changedRestCompileDiagnostics.HasErrors());
    failNextCudaRbfFinalizationForTesting();
    UsdGenDiagnostics changedRestFailureDiagnostics;
    CHECK(!ExecuteCudaPlanForTesting(changedRestPlan, *workspace, 22.0, 203,
                            &changedRestFailureDiagnostics, retry) &&
          SameStats(GetCudaBindingStats(*workspace), retryStats));
    UsdGenDiagnostics changedRestRetryDiagnostics;
    auto changedRestRetry = ExecuteCudaPlanForTesting(changedRestPlan, *workspace, 22.0, 204,
                                             &changedRestRetryDiagnostics, retry);
    auto const changedRestStats = GetCudaBindingStats(*workspace);
    CHECK(changedRestRetry && changedRestStats.size() == 1 &&
          changedRestStats.front().identity != retryStats.front().identity &&
          changedRestStats.front().bindCount == 1 &&
          changedRestStats.front().solveCount == 1 &&
          changedRestStats.front().sampleCount == 5);

    // A budget change is likewise part of the cache key. Failed finalization
    // retains the old identity; retry accepts the four-sample binding once.
    UsdGenDiagnostics budgetCompileDiagnostics;
    auto budgetPlan = CompileCudaGraph(usdGenUsd::Lower(ChangedBudgetDesc()), &budgetCompileDiagnostics);
    CHECK(budgetPlan && !budgetCompileDiagnostics.HasErrors());
    failNextCudaRbfFinalizationForTesting();
    UsdGenDiagnostics budgetFailureDiagnostics;
    CHECK(!ExecuteCudaPlanForTesting(budgetPlan, *workspace, 23.0, 205,
                            &budgetFailureDiagnostics, changedRestRetry) &&
          SameStats(GetCudaBindingStats(*workspace), changedRestStats));
    UsdGenDiagnostics budgetRetryDiagnostics;
    auto budgetRetry = ExecuteCudaPlanForTesting(budgetPlan, *workspace, 23.0, 206,
                                        &budgetRetryDiagnostics, changedRestRetry);
    auto const budgetStats = GetCudaBindingStats(*workspace);
    if (!budgetRetry || budgetStats.size() != 1 ||
        budgetStats.front().identity == changedRestStats.front().identity ||
        budgetStats.front().sampleCount != 4 ||
        budgetStats.front().bindCount != changedRestStats.front().bindCount + 1 ||
        budgetStats.front().solveCount != changedRestStats.front().solveCount + 1) {
        std::fprintf(stderr,
            "  budget retry result=%d errors=%zu stats=%zu old[id=%llu bind=%llu solve=%llu samples=%zu]",
            int(bool(budgetRetry)), budgetRetryDiagnostics.errors.size(), budgetStats.size(),
            static_cast<unsigned long long>(changedRestStats.front().identity),
            static_cast<unsigned long long>(changedRestStats.front().bindCount),
            static_cast<unsigned long long>(changedRestStats.front().solveCount),
            changedRestStats.front().sampleCount);
        if (!budgetStats.empty()) std::fprintf(stderr,
            " new[id=%llu bind=%llu solve=%llu samples=%zu]",
            static_cast<unsigned long long>(budgetStats.front().identity),
            static_cast<unsigned long long>(budgetStats.front().bindCount),
            static_cast<unsigned long long>(budgetStats.front().solveCount),
            budgetStats.front().sampleCount);
        if (!budgetRetryDiagnostics.errors.empty())
            std::fprintf(stderr, " error=%s", budgetRetryDiagnostics.errors.front().c_str());
        std::fprintf(stderr, "\n");
    }
    CHECK(budgetRetry && budgetStats.size() == 1 &&
          budgetStats.front().identity != changedRestStats.front().identity &&
          budgetStats.front().sampleCount == 4 &&
          budgetStats.front().bindCount == changedRestStats.front().bindCount + 1 &&
          budgetStats.front().solveCount == changedRestStats.front().solveCount + 1);

    // Accepted cache state is owned by the workspace, not by a completed
    // execution job. Published geometry remains charged by its generation,
    // while destroying the workspace releases the cache's child permits.
    auto cacheWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(cacheWorkspace);
    auto const cacheBaseline = resources->Snapshot();
    auto cacheGeneration = ExecuteCudaPlanForTesting(plan, *cacheWorkspace, 24.0, 207,
                                            &diagnostics);
    CHECK(cacheGeneration && !diagnostics.HasErrors());
    auto const cacheHeld = resources->Snapshot();
    CHECK(cacheHeld.usedBytes > cacheBaseline.usedBytes &&
          ResourceKindBytes(cacheHeld, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(cacheBaseline, UsdGenExecutionResourceKind::Pending));
    cacheWorkspace.reset();
    auto const cacheReleased = resources->Snapshot();
    CHECK(cacheReleased.usedBytes < cacheHeld.usedBytes &&
          ResourceKindBytes(cacheReleased, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(cacheBaseline, UsdGenExecutionResourceKind::Pending));
    cacheGeneration.reset();

    // Once fresh solve work was submitted but its completion cannot be
    // proven, rollback is unsafe. The workspace is quarantined: its accepted
    // stats and retained generation remain readable and every retry fails.
    auto poisonedWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(poisonedWorkspace);
    UsdGenDiagnostics poisonBaseDiagnostics;
    auto poisonBase = ExecuteCudaPlanForTesting(plan, *poisonedWorkspace, 30.0, 300,
                                       &poisonBaseDiagnostics);
    CHECK(poisonBase);
    auto const poisonStats = GetCudaBindingStats(*poisonedWorkspace);
    failNextCudaRbfSolveAfterInputSubmitForTesting();
    UsdGenDiagnostics poisonDiagnostics;
    CHECK(!ExecuteCudaPlanForTesting(plan, *poisonedWorkspace, 31.0, 301,
                            &poisonDiagnostics, poisonBase) &&
          SameStats(GetCudaBindingStats(*poisonedWorkspace), poisonStats) &&
          SameGeneration(poisonBase, poisonBase));
    UsdGenDiagnostics poisonedRetryDiagnostics;
    CHECK(!ExecuteCudaPlanForTesting(plan, *poisonedWorkspace, 32.0, 302,
                            &poisonedRetryDiagnostics, poisonBase) &&
          SameStats(GetCudaBindingStats(*poisonedWorkspace), poisonStats));
    return 0;
}

} // namespace

static int ValueDagTransactions() {
    auto width = [](char const* path, SdfPath input, float factor) {
        UsdGenNodeDesc node;
        node.path = SdfPath(path); node.type = TfToken("UsdGenWidth"); node.inputs = {input};
        node.params = {{TfToken("width"), VtValue(factor), false},
                       {TfToken("replace"), VtValue(false), false}};
        return node;
    };
    for (int variant = 0; variant != 8; ++variant) {
        std::fprintf(stderr, "RBF value DAG variant %d\n", variant);
        auto dag = RbfDesc();
        auto& curves = dag.curveSets.front();
        curves.curveVertexCounts = {2,3}; curves.curveId = {42,7};
        curves.points = {{.2f,.2f,0},{.2f,.4f,0},{.3f,.2f,0},{.3f,.7f,0},{.3f,1.2f,0}};
        curves.rest = curves.points; curves.skinPrim = {0,0};
        curves.skinPrimUv = {{.2f,.2f},{.3f,.2f}};
        UsdGenAuthoredPlaneDesc named;
        named.name = TfToken("rbfPointTag"); named.floatValues = {42,43,7,8,9};
        curves.authoredPlanes = {named};
        if (variant == 7) {
            curves.curveVertexCounts.clear(); curves.curveId.clear();
            curves.points.clear(); curves.rest.clear(); curves.skinPrim.clear(); curves.skinPrimUv.clear();
            curves.authoredPlanes.front().floatValues.clear();
        }
        auto source = dag.nodes.front();
        auto left = dag.nodes.back(); left.path = SdfPath("/RbfAtomic/LeftDeform");
        auto right = left; right.path = SdfPath("/RbfAtomic/RightDeform");
        auto sibling = width("/RbfAtomic/SiblingWidth", source.path, 3.f);
        auto reference = dag;
        reference.nodes = {source, left}; reference.terminal = left.path;
        dag.nodes = {source, left, sibling}; dag.terminal = left.path;
        if (variant == 1) {
            dag.terminal = source.path;
            reference.nodes = {source}; reference.terminal = source.path;
        } else if (variant == 2) {
            dag.terminal = sibling.path;
            reference.nodes = {source, sibling}; reference.terminal = sibling.path;
        } else if (variant == 3 || variant == 4 || variant == 6) {
            auto predecessor = width("/RbfAtomic/Predecessor", source.path, 2.f);
            if (variant != 3) {
                predecessor.type = TfToken("UsdGenLength");
                predecessor.params = variant == 6
                    ? std::vector<UsdGenParamValue>{{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                                                  {TfToken("cullThreshold"), VtValue(.3f), false}}
                    : std::vector<UsdGenParamValue>{{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                                                  {TfToken("length:value"), VtValue(.5f), false}};
            }
            left.inputs = {predecessor.path};
            dag.nodes = {source, predecessor, left, sibling};
            reference.nodes = {source, predecessor, left};
        } else if (variant == 5) {
            auto wl = width("/RbfAtomic/LeftWidth", left.path, 2.f);
            auto wr = width("/RbfAtomic/RightWidth", right.path, 3.f);
            UsdGenNodeDesc blend;
            blend.path = SdfPath("/RbfAtomic/Blend"); blend.type = TfToken("UsdGenWidthBlend");
            blend.inputs = {wl.path, wr.path};
            blend.params.push_back({TfToken("widthBlend:weight"), VtValue(.25f), false});
            dag.nodes = {source, left, right, wl, wr, blend}; dag.terminal = blend.path;
            auto expectedWidth = width("/RbfAtomic/ExpectedWidth", left.path, 2.25f);
            reference.nodes = {source, left, expectedWidth}; reference.terminal = expectedWidth.path;
        }
        UsdGenDiagnostics diagnostics;
        auto plan = CompileCudaGraph(usdGenUsd::Lower(dag), &diagnostics);
        if (!plan) for (auto const& error : diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(plan && !diagnostics.HasErrors());
        auto referencePlan = CompileCudaGraph(usdGenUsd::Lower(reference), &diagnostics); CHECK(referencePlan);
        auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        auto referenceWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(workspace && referenceWorkspace);
        auto expected = ExecuteCudaPlanForTesting(referencePlan, *referenceWorkspace, 1.0, 1000, &diagnostics);
        auto first = ExecuteCudaPlanForTesting(plan, *workspace, 1.0, 1001, &diagnostics);
        CHECK(expected && first && !diagnostics.HasErrors() && SameGeneration(expected, first));
        if (variant == 7) {
            CHECK(first->Geometry().pointCount == 0 && first->Geometry().elementCount == 0 &&
                  first->Geometry().alreadyDeformed);
            UsdGenExecutionRuntime runtime(4);
            UsdGenCudaExecutionQueue queue(runtime, -1, &diagnostics);
            CHECK(queue.Valid() && queue.Submit(plan, 1.0)); queue.Drain();
            auto snapshot = queue.Snapshot();
            CHECK(snapshot && snapshot->generation && SameGeneration(expected, snapshot->generation));
            continue;
        }
        auto stats = GetCudaBindingStats(*workspace);
        CHECK(stats.size() == (variant == 5 ? 2 : 1));
        auto attempts = GetAttempts();
        failNextCudaRbfFinalizationForTesting();
        UsdGenDiagnostics failedDiagnostics;
        CHECK(!ExecuteCudaPlanForTesting(plan, *workspace, 2.0, 1002, &failedDiagnostics, first));
        CHECK(failedDiagnostics.HasErrors() && SameStats(stats, GetCudaBindingStats(*workspace)) &&
              SameGeneration(expected, first));
        auto rejected = GetAttempts();
        if (rejected.rbfRollback != attempts.rbfRollback + stats.size() ||
            rejected.surfaceRollback != attempts.surfaceRollback + stats.size()) {
            std::fprintf(stderr, "rollback delta field=%llu surface=%llu expected=%zu\n",
                static_cast<unsigned long long>(rejected.rbfRollback - attempts.rbfRollback),
                static_cast<unsigned long long>(rejected.surfaceRollback - attempts.surfaceRollback), stats.size());
            for (auto const& error : failedDiagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
        }
        CHECK(rejected.rbfAccept == attempts.rbfAccept && rejected.surfaceAccept == attempts.surfaceAccept);
        CHECK(rejected.rbfRollback == attempts.rbfRollback + stats.size() &&
              rejected.surfaceRollback == attempts.surfaceRollback + stats.size());
        auto retry = ExecuteCudaPlanForTesting(plan, *workspace, 3.0, 1003, &diagnostics, first);
        CHECK(retry && SameGeneration(expected, retry) && SameGeneration(expected, first));
        auto after = GetCudaBindingStats(*workspace);
        CHECK(after.size() == stats.size());
        for (size_t i = 0; i != after.size(); ++i)
            CHECK(after[i].identity == stats[i].identity && after[i].solveCount == stats[i].solveCount + 1);
        UsdGenExecutionRuntime runtime(4);
        UsdGenCudaExecutionQueue queue(runtime, -1, &diagnostics);
        CHECK(queue.Valid() && queue.Submit(plan, 1.0)); queue.Drain();
        auto async = queue.Snapshot();
        CHECK(async && async->generation && async->bindings.size() == stats.size() &&
              SameGeneration(expected, async->generation));
        if (variant == 5) {
            auto beforeAsyncFailure = GetAttempts();
            failNextCudaFinalizationRelayAllocationForTesting();
            std::atomic<unsigned> callbacks{0};
            std::atomic<bool> failed{false};
            CHECK(queue.Submit(plan, 2.0, [&](uint64_t, auto outcome, auto const&) {
                ++callbacks;
                failed = outcome == UsdGenExecutionPipeline::Outcome::Failed;
            }));
            queue.Drain();
            CHECK(callbacks == 1 && failed && queue.Snapshot() == async &&
                  SameGeneration(expected, async->generation));
            auto failedAttempts = GetAttempts();
            CHECK(failedAttempts.rbfAccept == beforeAsyncFailure.rbfAccept &&
                  failedAttempts.surfaceAccept == beforeAsyncFailure.surfaceAccept &&
                  failedAttempts.rbfRollback == beforeAsyncFailure.rbfRollback + 2 &&
                  failedAttempts.surfaceRollback == beforeAsyncFailure.surfaceRollback + 2);
            CHECK(queue.Submit(plan, 3.0)); queue.Drain();
            CHECK(queue.Snapshot() != async && queue.Snapshot()->generation &&
                  SameGeneration(expected, queue.Snapshot()->generation) &&
                  SameGeneration(expected, async->generation));
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--value-dag") return ValueDagTransactions();
    CHECK(argc == 1);
    UsdGenDiagnostics diagnostics;
    auto plan = CompileCudaGraph(usdGenUsd::Lower(RbfDesc()), &diagnostics);
    CHECK(plan && !diagnostics.HasErrors());
    auto metadata = GetCudaExecutionPlanMetadata(*plan);
    CHECK(metadata && !metadata->MemoryEstimate().memoryAvailable &&
          !metadata->MemoryEstimate().conservativeUpperBound &&
          metadata->MemoryEstimate().runtimeRefinementAvailable);
    CHECK(FailureCase(plan, Resolution::AcceptPreflight, Side::Rbf));
    CHECK(FailureCase(plan, Resolution::AcceptPreflight, Side::Surface));
    CHECK(FailureCase(plan, Resolution::RollbackPreflight, Side::Rbf));
    CHECK(FailureCase(plan, Resolution::RollbackPreflight, Side::Surface));
    CHECK(FailureCase(plan, Resolution::AcceptCommit, Side::Rbf));
    CHECK(FailureCase(plan, Resolution::AcceptCommit, Side::Surface));
    CHECK(FailureCase(plan, Resolution::RollbackCommit, Side::Rbf));
    CHECK(FailureCase(plan, Resolution::RollbackCommit, Side::Surface));

    UsdGenExecutionRuntime runtime(4);
    UsdGenCudaExecutionQueue queue(runtime, -1, &diagnostics);
    CHECK(queue.Valid() && queue.Submit(plan, 10.0));
    queue.Drain();
    auto const first = queue.Snapshot();
    CHECK(first && first->generation && first->bindings.size() == 1);
    auto const before = GetAttempts();
    CHECK(queue.Submit(plan, 11.0));
    queue.Drain();
    auto const second = queue.Snapshot();
    auto const after = GetAttempts();
    CHECK(second && second != first && second->generation != first->generation &&
          second->bindings.size() == 1 &&
          second->bindings[0].identity == first->bindings[0].identity &&
          second->bindings[0].bindCount == first->bindings[0].bindCount &&
          second->bindings[0].solveCount == first->bindings[0].solveCount + 1 &&
          after.rbfAccept == before.rbfAccept + 1 &&
          after.surfaceAccept == before.surfaceAccept + 1 &&
          after.rbfRollback == before.rbfRollback &&
          after.surfaceRollback == before.surfaceRollback);
    CHECK(SynchronousTransactions(plan) == 0);
    return 0;
}
