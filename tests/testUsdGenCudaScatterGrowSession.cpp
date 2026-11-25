#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"
#include "usdGen/session.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/imagePayload.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static UsdGenGraphDesc Desc(bool nonYAxisBinormal = false) {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/ScatterGrow");
    d.executionBackend = UsdGenExecutionBackend::Cuda;
    d.defaultWidth = .025f;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    // The ordinary XY patch has B=+Y.  The XZ variant deliberately makes B
    // non-Y so an angular Grow lift cannot accidentally be implemented as a
    // Y-axis/global-space rotation.
    if (nonYAxisBinormal)
        surface.restPoints = {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}};
    else
        surface.restPoints = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}};
    surface.points = surface.restPoints;
    surface.uv = {{0,0}, {1,0}, {1,1}, {0,1}};
    surface.faceVertexCounts = {4};
    surface.faceVertexIndices = {0,1,2,3};
    d.surfaces.push_back(surface);
    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/Ops/Scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.seed = 41;
    scatter.surfaces = {surface.path};
    scatter.params = {{TfToken("density"), VtValue(300.0f), false}};
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Ops/Grow");
    grow.type = TfToken("UsdGenGrow");
    grow.seed = 19;
    grow.inputs = {scatter.path};
    grow.params = {{TfToken("segments"), VtValue(5), false},
                   {TfToken("length"), VtValue(2.0f), false},
                   {TfToken("lengthRandom"), VtValue(GfVec2f(.5f, 1.5f)), false}};
    d.nodes = {scatter, grow};
    d.terminal = grow.path;
    return d;
}

static bool CpuReference(UsdGenGraphDesc desc, UsdGenCurveBuffer* output) {
    desc.executionBackend = UsdGenExecutionBackend::CpuReference;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    if (!compiler.Compile(desc, &graph).ok) return false;
    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    auto result = scheduler.Run(graph, context, 1);
    for (auto const& error : result.diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
    if (result.diagnostics.HasErrors()) return false;
    *output = NodeBuffer(graph, graph.NodeIdForPath(desc.terminal));
    return true;
}

template<class T> static bool Read(gpu::DeviceView<const T> input,
                                   std::vector<T>* output, cudaStream_t stream) {
    output->resize(input.size);
    if (input.size && cudaMemcpyAsync(output->data(), input.data,
            input.size * sizeof(T), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

static bool Near(float a, float b) { return std::fabs(a - b) < 2e-5f; }

static bool CheckGrowParity(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                            UsdGenCurveBuffer const& reference, cudaStream_t stream,
                            std::vector<float>* endpointLengths = nullptr,
                            bool requireNonYAxisBinormal = false) {
    auto lease = gpu::AcquireGeometry(generation, stream);
    if (!lease || lease.Geometry().curveCount != reference.totalCurves ||
        lease.Geometry().pointCount != reference.totalCvs) return false;
    std::vector<float3> points, rest, rootB;
    std::vector<float> widths, hairT;
    std::vector<uint64_t> ids;
    std::vector<uint32_t> offsets;
    if (!Read(lease.Geometry().points, &points, stream) ||
        !Read(lease.Geometry().restPoints, &rest, stream) ||
        !Read(lease.Geometry().widths, &widths, stream) ||
        !Read(lease.HairT(), &hairT, stream) ||
        !Read(lease.Geometry().stableIds, &ids, stream) ||
        !Read(lease.Geometry().curveOffsets, &offsets, stream) ||
        !Read(lease.RootB(), &rootB, stream)) return false;
    if (ids.size() != reference.curveId.size() ||
        offsets.size() != ids.size() + 1 || rootB.size() != ids.size()) return false;
    for (size_t curve = 0; curve != ids.size(); ++curve) {
        uint32_t const begin = offsets[curve], end = offsets[curve + 1];
        if (ids[curve] != reference.curveId[curve] || begin >= end || end > points.size()) return false;
        if (requireNonYAxisBinormal &&
            !(std::fabs(rootB[curve].y) < .5f &&
              (std::fabs(rootB[curve].x) > .5f || std::fabs(rootB[curve].z) > .5f))) return false;
    }
    if (offsets.back() != points.size()) return false;
    for (size_t cv = 0; cv != points.size(); ++cv) {
        if (!Near(points[cv].x, reference.px[cv]) ||
            !Near(points[cv].y, reference.py[cv]) ||
            !Near(points[cv].z, reference.pz[cv]) ||
            !Near(rest[cv].x, reference.rest[cv][0]) ||
            !Near(rest[cv].y, reference.rest[cv][1]) ||
            !Near(rest[cv].z, reference.rest[cv][2]) ||
            !Near(widths[cv], reference.width[cv]) || !Near(hairT[cv], reference.hairT[cv])) return false;
    }
    if (endpointLengths) {
        endpointLengths->clear(); endpointLengths->reserve(ids.size());
        for (size_t curve = 0; curve != ids.size(); ++curve) {
            auto const first = points[offsets[curve]], last = points[offsets[curve + 1] - 1];
            float const x = last.x - first.x, y = last.y - first.y, z = last.z - first.z;
            endpointLengths->push_back(std::sqrt(x*x + y*y + z*z));
        }
    }
    return true;
}

struct ReleaseGate {
    void (*release)();
    ~ReleaseGate() { if (release) release(); }
};

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    {
        auto desc = Desc();
        UsdGenCurveBuffer reference;
        CHECK(CpuReference(desc, &reference));
        CHECK(reference.totalCurves == 300 && reference.totalCvs == 1500);
        CHECK(!std::is_sorted(reference.curveId.cbegin(), reference.curveId.cend()));
        UsdGenDiagnostics diagnostics;
        auto plan = CompileCudaGraph(desc, &diagnostics);
        for (auto const& error : diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(plan && !diagnostics.HasErrors());
        if (argc == 2) {
            bool const installFailure = std::strcmp(argv[1], "--source-install-failure") == 0;
            bool const nativeFailure = std::strcmp(argv[1], "--source-native-failure") == 0;
            CHECK(installFailure || nativeFailure);
            auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
            CHECK(workspace);
            auto job = CreateCudaExecutionJob(plan, *workspace, 1, 600, &diagnostics);
            CHECK(job);
            size_t const quarantined = cudaSourceRelayQuarantinedCountForTesting();
            std::promise<bool> promise;
            auto future = promise.get_future();
            if (installFailure) failNextCudaSourceRelayCallbackInstallForTesting();
            else failNextCudaSourceRelayNativeCallbackForTesting();
            CHECK(ExecuteCudaJobSourceAsync(job, [&](bool ok) { promise.set_value(ok); }));
            CHECK(future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
            CHECK(!future.get());
            CHECK(workspace->IsPoisoned());
            CHECK(cudaSourceRelayQuarantinedCountForTesting() == quarantined + 1);
            CHECK(!CreateCudaExecutionJob(plan, *workspace, 1, 601, &diagnostics));
            CHECK(!ExecuteCudaGraph(*plan, *workspace, 1, 602, &diagnostics));
            CHECK(cudaStreamDestroy(stream) == cudaSuccess);
            std::puts("testUsdGenCudaScatterGrowSourceFailure: PASS");
            return 0;
        }
        auto metadata = GetCudaExecutionPlanMetadata(*plan);
        CHECK(metadata && metadata->MemoryEstimate().memoryAvailable &&
              metadata->MemoryEstimate().conservativeUpperBound);
        CHECK(metadata->Tasks().size() == 2 && metadata->Operators().size() == 2);
        CHECK(metadata->TerminalTask() == 1);
        CHECK(metadata->Tasks()[0].type == TfToken("CudaScatterGrow"));
        CHECK(metadata->Tasks()[0].estimate.steadyBytes == 2 * sizeof(int));
        CHECK(metadata->Tasks()[1].dependencies == std::vector<uint32_t>{0});
        CHECK(metadata->Tasks()[1].estimate.scratchPeakBytes > 0);
        for (auto const& value : metadata->Values())
            CHECK(value.storage == (value.producerTask == UINT32_MAX
                ? UsdGenExecutionValueStorage::ExternalImmutable
                : value.producerTask == 0
                ? UsdGenExecutionValueStorage::JobOwnedImmutable
                : UsdGenExecutionValueStorage::PublishedImmutable));
        {
            UsdGenDiagnostics lostDiagnostics;
            auto lostWorkspace = CreateCudaExecutionWorkspace(-1, &lostDiagnostics);
            CHECK(lostWorkspace);
            auto pending = CreateCudaExecutionJob(plan, *lostWorkspace, 1, 500, &lostDiagnostics);
            CHECK(pending);
            lostWorkspace->MarkContextLost();
            CHECK(lostWorkspace->IsPoisoned());
            CHECK(!ExecuteCudaJobSource(*pending));
            CHECK(lostDiagnostics.HasErrors());
            CHECK(!CreateCudaExecutionJob(plan, *lostWorkspace, 1, 501, &lostDiagnostics));
            CHECK(!ExecuteCudaGraph(*plan, *lostWorkspace, 1, 502, &lostDiagnostics));
        }
        auto rejects = [](UsdGenGraphDesc const& candidate) {
            UsdGenDiagnostics errors;
            auto rejected = CompileCudaGraph(candidate, &errors);
            return !rejected && errors.HasErrors();
        };
        auto malformed = desc;
        malformed.nodes[1].params[0].value = VtValue(5.5f);
        CHECK(rejects(malformed));
        malformed = desc;
        malformed.nodes[1].params[1].value = VtValue(std::string("2"));
        CHECK(rejects(malformed));
        malformed = desc;
        malformed.nodes[1].params[1].value = VtValue(std::numeric_limits<double>::max());
        CHECK(rejects(malformed));
        malformed = desc;
        malformed.nodes[1].params.push_back(malformed.nodes[1].params[1]);
        CHECK(rejects(malformed));
        malformed = desc;
        malformed.nodes[1].mode = TfToken("unimplemented");
        CHECK(rejects(malformed));
        malformed = desc;
        malformed.nodes[1].params.push_back({TfToken("lift"), VtValue(std::string("1")), false});
        CHECK(rejects(malformed));
        for (double lift : {-90.01, 90.01,
                            std::numeric_limits<double>::quiet_NaN()}) {
            malformed = desc;
            malformed.nodes[1].params.push_back({TfToken("lift"), VtValue(lift), false});
            CHECK(rejects(malformed));
        }
        for (char const* name : {"azimuth", "azimuthRandom"}) {
            for (double value : {-361.0, 361.0, std::numeric_limits<double>::quiet_NaN()}) {
                malformed = desc;
                malformed.nodes[1].params.push_back({TfToken(name), VtValue(value), false});
                CHECK(rejects(malformed));
            }
            malformed = desc;
            malformed.nodes[1].params.push_back({TfToken(name), VtValue(std::string("1")), false});
            CHECK(rejects(malformed));
        }
        UsdGenSession session;
        session.SetDevicePublicationEnabled(true);
        session.SetGraphDesc(desc);
        auto first = session.Commit(1, UsdGenCommitReason::SetTime);
        for (auto const& error : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
        auto retained = gpu::AcquireGeometry(first->device, stream);
        CHECK(retained && retained.Geometry().curveCount == reference.totalCurves);
        std::vector<float3> points, rest, frames, tangents, bitangents;
        std::vector<float2> rootUV;
        std::vector<int32_t> rootPrim;
        std::vector<float> hairT, widths;
        std::vector<uint64_t> ids;
        std::vector<uint32_t> offsets;
        CHECK(Read(retained.Geometry().points, &points, stream));
        CHECK(Read(retained.Geometry().restPoints, &rest, stream));
        CHECK(Read(retained.Geometry().stableIds, &ids, stream));
        CHECK(Read(retained.Geometry().curveOffsets, &offsets, stream));
        CHECK(Read(retained.HairT(), &hairT, stream));
        CHECK(Read(retained.Geometry().widths, &widths, stream));
        CHECK(Read(retained.RootN(), &frames, stream));
        CHECK(Read(retained.RootT(), &tangents, stream));
        CHECK(Read(retained.RootB(), &bitangents, stream));
        CHECK(Read(retained.RootUV(), &rootUV, stream));
        CHECK(Read(retained.RootPrim(), &rootPrim, stream));
        CHECK(points.size() == reference.totalCvs && frames.size() == ids.size());
        CHECK(reference.width.size() == points.size());
        CHECK(tangents.size() == ids.size() && bitangents.size() == ids.size());
        CHECK(rootUV.size() == ids.size() && rootPrim.size() == ids.size());
        for (size_t c = 0; c < ids.size(); ++c) {
            CHECK(ids[c] == reference.curveId[c] && offsets[c] == c * 5);
            CHECK(Near(frames[c].x, reference.rootN[c][0]) &&
                  Near(frames[c].y, reference.rootN[c][1]) && Near(frames[c].z, reference.rootN[c][2]));
            CHECK(Near(tangents[c].x, reference.rootT[c][0]) &&
                  Near(tangents[c].y, reference.rootT[c][1]) && Near(tangents[c].z, reference.rootT[c][2]));
            CHECK(Near(bitangents[c].x, reference.rootB[c][0]) &&
                  Near(bitangents[c].y, reference.rootB[c][1]) && Near(bitangents[c].z, reference.rootB[c][2]));
            CHECK(rootPrim[c] == reference.rootPrim[c]);
            CHECK(Near(rootUV[c].x, reference.rootUV[c][0]) && Near(rootUV[c].y, reference.rootUV[c][1]));
        }
        CHECK(offsets.back() == points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            CHECK(Near(points[i].x, reference.px[i]) && Near(points[i].y, reference.py[i]) &&
                  Near(points[i].z, reference.pz[i]));
            CHECK(Near(rest[i].x, points[i].x) && Near(rest[i].y, points[i].y) && Near(rest[i].z, points[i].z));
            CHECK(Near(hairT[i], reference.hairT[i]) && Near(widths[i], desc.defaultWidth));
            CHECK(Near(widths[i], reference.width[i]));
        }
        CHECK(!first->device->Geometry().tiles.empty());
        uint64_t nextCurve = 0, nextPoint = 0;
        for (auto const& tile : first->device->Geometry().tiles) {
            CHECK(tile.firstElement == nextCurve && tile.firstPoint == nextPoint);
            CHECK(tile.elementCount > 0 && tile.pointCount == tile.elementCount * 5);
            nextCurve += tile.elementCount;
            nextPoint += tile.pointCount;
        }
        CHECK(nextCurve == ids.size() && nextPoint == points.size());
        // The direct entry point must produce the same immutable payload and
        // publication metadata as the asynchronous Session path.
        auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(workspace);
        auto direct = ExecuteCudaGraph(*plan, *workspace, 1, 101, &diagnostics);
        CHECK(direct && !diagnostics.HasErrors());
        auto directLease = gpu::AcquireGeometry(direct, stream);
        CHECK(directLease && directLease.Geometry().points.data != retained.Geometry().points.data);
        std::vector<float3> directPoints;
        CHECK(Read(directLease.Geometry().points, &directPoints, stream));
        CHECK(directPoints.size() == points.size());
        CHECK(std::memcmp(directPoints.data(), points.data(), points.size() * sizeof(float3)) == 0);
        CHECK(direct->Geometry().tiles.size() == first->device->Geometry().tiles.size());

        // Lift is an angular rotation about the per-root B frame.  The four
        // signed extrema cover the conventional 30-degree style adjustment
        // and the +/-90-degree axis exchange without changing random strand
        // lengths.  Keep the first published lease alive across the next
        // lift edit to prove COW retirement does not rewrite a prior result.
        UsdGenSession angularSession;
        angularSession.SetDevicePublicationEnabled(true);
        std::vector<float> baselineLengths;
        gpu::CudaGeometryLease angularRetained;
        std::vector<float3> angularRetainedPoints;
        std::array<float,4> const lifts{{-90.f,-30.f,30.f,90.f}};
        for (size_t variant = 0; variant != lifts.size(); ++variant) {
            float const lift = lifts[variant];
            auto angular = Desc();
            angular.nodes[1].params.push_back({TfToken("lift"),VtValue(lift),false});
            angular.nodes[1].params.push_back({TfToken("azimuth"),VtValue(float(variant)*30.f),false});
            angular.nodes[1].params.push_back({TfToken("azimuthRandom"),VtValue(float(variant)/3.f),false});
            UsdGenCurveBuffer angularReference;
            CHECK(CpuReference(angular,&angularReference));
            UsdGenDiagnostics angularDiagnostics;
            auto angularPlan=CompileCudaGraph(angular,&angularDiagnostics);
            CHECK(angularPlan&&!angularDiagnostics.HasErrors());
            auto angularWorkspace=CreateCudaExecutionWorkspace(-1,&angularDiagnostics);
            CHECK(angularWorkspace);
            auto angularDirect=ExecuteCudaGraph(*angularPlan,*angularWorkspace,
                                                static_cast<double>(10+variant),200+variant,
                                                &angularDiagnostics);
            std::vector<float> lengths;
            CHECK(angularDirect&&!angularDiagnostics.HasErrors()&&
                  CheckGrowParity(angularDirect,angularReference,stream,&lengths));
            if (variant == 0) baselineLengths=lengths;
            else {
                CHECK(lengths.size()==baselineLengths.size());
                for(size_t curve=0;curve!=lengths.size();++curve)
                    CHECK(Near(lengths[curve],baselineLengths[curve]));
            }
            angularSession.SetGraphDesc(angular);
            auto angularPublished=angularSession.Commit(static_cast<double>(10+variant),UsdGenCommitReason::SetTime);
            CHECK(angularPublished&&angularPublished->device&&
                  !angularSession.LastDiagnostics().HasErrors()&&
                  CheckGrowParity(angularPublished->device,angularReference,stream));
            if (variant == 0) {
                angularRetained=gpu::AcquireGeometry(angularPublished->device,stream);
                CHECK(angularRetained&&Read(angularRetained.Geometry().points,&angularRetainedPoints,stream));
            } else if (variant == 1) {
                auto angularCurrent=gpu::AcquireGeometry(angularPublished->device,stream);
                CHECK(angularCurrent&&angularCurrent.Geometry().points.data!=angularRetained.Geometry().points.data);
                CHECK(Read(angularRetained.Geometry().points,&directPoints,stream));
                CHECK(directPoints.size()==angularRetainedPoints.size()&&
                      std::memcmp(directPoints.data(),angularRetainedPoints.data(),
                          directPoints.size()*sizeof(float3))==0);
            }
        }

        // A held native callback cannot authorize publication. Both entry
        // points must return while proof is held, retaining their own state.
        auto asyncWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        CHECK(asyncWorkspace);
        auto job = CreateCudaExecutionJob(plan, *asyncWorkspace, 1, 102, &diagnostics, first->device);
        CHECK(job);
        std::promise<bool> sourcePromise;
        auto sourceFuture = sourcePromise.get_future();
        armCudaSourceAsyncCallbackGateForTesting();
        ReleaseGate releaseSource{releaseCudaSourceAsyncCallbackGateForTesting};
        CHECK(ExecuteCudaJobSourceAsync(job, [&](bool ok) { sourcePromise.set_value(ok); }));
        waitCudaSourceAsyncCallbackGateForTesting();
        CHECK(sourceFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
        releaseSource.release(); releaseSource.release = nullptr;
        CHECK(sourceFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
        CHECK(sourceFuture.get());
        for (size_t i = 0; i < CudaExecutionJobOperatorCount(*job); ++i)
            CHECK(ExecuteCudaJobOperator(*job, i));
        CHECK(Read(retained.Geometry().points, &directPoints, stream));
        CHECK(std::memcmp(directPoints.data(), points.data(), points.size() * sizeof(float3)) == 0);
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> finalPromise;
        auto finalFuture = finalPromise.get_future();
        armCudaFinalizationAsyncMetadataCallbackGateForTesting();
        ReleaseGate releaseFinal{releaseCudaFinalizationAsyncMetadataCallbackGateForTesting};
        CHECK(FinalizeCudaExecutionJobAsync(job, [&](auto result) { finalPromise.set_value(std::move(result)); }));
        waitCudaFinalizationAsyncMetadataCallbackGateForTesting();
        CHECK(finalFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
        CHECK(!FinalizeCudaExecutionJobAsync(job, [](auto) {}));
        // Do not enter CUDA while deliberately holding a stream callback:
        // instrumentation may serialize even a different stream's readback
        // behind that callback. Validate old bytes before and after instead.
        releaseFinal.release(); releaseFinal.release = nullptr;
        CHECK(finalFuture.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
        auto asyncResult = finalFuture.get();
        CHECK(asyncResult && asyncResult->Geometry().topologyVersion == first->device->Geometry().topologyVersion);
        CHECK(Read(retained.Geometry().points, &directPoints, stream));
        CHECK(std::memcmp(directPoints.data(), points.data(), points.size() * sizeof(float3)) == 0);
        auto oldPoints = points;
        auto oldTopology = first->device->Geometry().topologyVersion;
        desc.nodes[1].params[1].value = VtValue(4.0f);
        session.SetGraphDesc(desc);
        auto second = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(second && second != first && second->device && !session.LastDiagnostics().HasErrors());
        auto current = gpu::AcquireGeometry(second->device, stream);
        CHECK(current && current.Geometry().points.data != retained.Geometry().points.data);
        CHECK(second->device->Geometry().topologyVersion == oldTopology);
        CHECK(Read(retained.Geometry().points, &points, stream));
        CHECK(std::memcmp(points.data(), oldPoints.data(), points.size() * sizeof(float3)) == 0);

        // Rejected authored controls must preserve the accepted generation.
        auto invalid = desc;
        invalid.nodes[1].params.push_back({TfToken("lift"), VtValue(91.0f), false});
        session.SetGraphDesc(invalid);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());

        // Failure after source production must not replace last-good or
        // mutate leases. Use a fresh descriptor to avoid a completed cache hit.
        auto failedPublication = desc;
        failedPublication.nodes[1].params[1].value = VtValue(6.0f);
        session.SetGraphDesc(failedPublication);
        failNextCudaFinalizationRelayAllocationForTesting();
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        CHECK(Read(retained.Geometry().points, &points, stream));
        CHECK(std::memcmp(points.data(), oldPoints.data(), points.size() * sizeof(float3)) == 0);

        desc.nodes[0].params[0].value = VtValue(0.0f);
        auto emptyPlan = CompileCudaGraph(desc, &diagnostics);
        CHECK(emptyPlan);
        auto emptyMetadata = GetCudaExecutionPlanMetadata(*emptyPlan);
        CHECK(emptyMetadata && emptyMetadata->MemoryEstimate().memoryAvailable);
        CHECK(emptyMetadata->Tasks()[0].estimate.steadyBytes == 2 * sizeof(int));
        CHECK(emptyMetadata->Tasks()[0].estimate.retainedOutputBytes == sizeof(uint32_t));
        // Even with no roots, publication retains the native device/pinned
        // status pair alongside the single zero offset.
        CHECK(emptyMetadata->MemoryEstimate().concurrentPeakBytes >=
              sizeof(uint32_t) + 2 * sizeof(int) +
              emptyMetadata->Tasks()[1].estimate.scratchPeakBytes);
        session.SetGraphDesc(desc);
        auto empty = session.Commit(5, UsdGenCommitReason::SetTime);
        CHECK(empty && empty != second && empty->device && !session.LastDiagnostics().HasErrors());
        auto emptyLease = gpu::AcquireGeometry(empty->device, stream);
        CHECK(emptyLease && emptyLease.Geometry().curveCount == 0 && emptyLease.Geometry().pointCount == 0);
        CHECK(Read(emptyLease.Geometry().curveOffsets, &offsets, stream));
        CHECK(offsets == std::vector<uint32_t>{0});
    }
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("testUsdGenCudaScatterGrowSession: PASS");
    return 0;
}
