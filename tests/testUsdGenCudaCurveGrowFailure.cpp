#include "usdGen/cudaExecution.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGen/executionResources.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/imagePayload.h"

#include <chrono>
#include <cstdio>
#include <future>
#include <string>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

int main(int argc, char** argv) {
    bool const sourceTerminal = argc > 2 && std::string(argv[2]) == "--source-terminal";
    bool const mixed = argc > 2 && std::string(argv[2]) == "--mixed";
    bool const multiGrow = argc > 2 && std::string(argv[2]) == "--multi-grow";
    bool const noiseGrow = argc > 2 && std::string(argv[2]) == "--noise-grow";
    bool const culledNoiseGrow = argc > 2 && std::string(argv[2]) == "--culled-noise-grow";
    bool const predecessorGrow = noiseGrow || culledNoiseGrow;
    bool const withLength = mixed || (argc > 2 && std::string(argv[2]) == "--length");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(gpu::ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    auto pool = FindUsdGenExecutionResourcePool({usdGenTestUtils::Backend("cuda"), device});
    CHECK(pool);
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/GrowFailure");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Hair"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair"); curves.curveVertexCounts = {2};
    curves.points = {{1,2,3},{1,2,4}}; curves.rest = curves.points;
    curves.curveId = {7}; curves.skinPrim = {0}; curves.skinPrimUv = {{0,0}};
    curves.rootFrame = {GfMatrix4d(1.0)};
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken("value"); plane.type = UsdGenAuthoredPlaneType::Float32;
    plane.domain = UsdGenAuthoredPlaneDomain::Point; plane.arity = 1;
    plane.floatValues = {1,2}; curves.authoredPlanes = {plane};
    if (predecessorGrow) {
        curves.curveVertexCounts = {2,2,2};
        // Middle curve is deliberately below the cull threshold; source
        // IDs, frames, and named values make survivor/frame lookup visible.
        curves.points = {{1,2,3},{1,2,4}, {3,2,3},{3,2,3.1f}, {5,2,3},{5,2,5}};
        curves.rest = curves.points;
        curves.curveId = {7,11,19}; curves.skinPrim = {0,1,2};
        curves.skinPrimUv = {{0,0},{.25f,.5f},{.75f,.25f}};
        curves.rootFrame = {GfMatrix4d(1.0), GfMatrix4d(1.0), GfMatrix4d(1.0)};
        curves.rootFrame[1][0][0] = -1; curves.rootFrame[1][1][1] = -1;
        curves.rootFrame[2][0][0] = -1; curves.rootFrame[2][2][2] = -1;
        plane.floatValues = {1,2,3,4,5,6}; curves.authoredPlanes = {plane};
    }
    desc.curveSets = {curves};
    UsdGenNodeDesc source;
    source.path = SdfPath("/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Grow"); grow.type = TfToken("UsdGenGrow");
    grow.inputs = {source.path}; grow.params = {{TfToken("segments"), VtValue(4), false}};
    desc.nodes = {source, grow}; desc.terminal = sourceTerminal ? source.path : grow.path;
    if (predecessorGrow) {
        UsdGenNodeDesc noise;
        noise.path=SdfPath("/Noise"); noise.type=TfToken("UsdGenNoise");
        noise.params={{TfToken("noise:magnitude"),VtValue(.05f),false},
                      {TfToken("preserveLength"),VtValue(0.f),false}};
        if (culledNoiseGrow) {
            UsdGenNodeDesc length;
            length.path=SdfPath("/Length"); length.type=TfToken("UsdGenLength");
            length.inputs={source.path};
            length.params={{TfToken("length:mode"),VtValue(TfToken("cull")),false},
                           {TfToken("cullThreshold"),VtValue(.5f),false}};
            noise.inputs={length.path}; grow.inputs={noise.path};
            desc.nodes={source,length,noise,grow};
        } else {
            noise.inputs={source.path}; grow.inputs={noise.path};
            desc.nodes={source,noise,grow};
        }
        desc.terminal=grow.path;
    }
    if (multiGrow) {
        grow.path = SdfPath("/GrowA");
        auto growB = grow;
        growB.path = SdfPath("/GrowB");
        for (auto& parameter : growB.params)
            if (parameter.name == TfToken("segments")) parameter.value = VtValue(6);
        desc.nodes = {source, grow, growB}; desc.terminal = growB.path;
    }
    if (withLength) {
        UsdGenNodeDesc length;
        length.path=SdfPath("/Length"); length.type=TfToken("UsdGenLength");
        length.inputs={grow.path};
        length.params={{TfToken("length:value"),VtValue(.5f),false}};
        desc.nodes.push_back(length); desc.terminal=length.path;
        if(mixed) {
            UsdGenNodeDesc noise;noise.path=SdfPath("/Noise");noise.type=TfToken("UsdGenNoise");
            noise.inputs={length.path};
            noise.params={{TfToken("noise:magnitude"),VtValue(.05f),false},
                          {TfToken("preserveLength"),VtValue(0.f),false}};
            desc.nodes.push_back(noise);desc.terminal=noise.path;
        }
    }
    UsdGenDiagnostics diagnostics;
    auto plan = CompileCudaGraph(usdGenUsd::Lower(desc), &diagnostics);
    CHECK(plan);
    auto metadata = GetCudaExecutionPlanMetadata(*plan);
    CHECK(metadata && ((withLength || culledNoiseGrow) ? metadata->MemoryEstimate().runtimeRefinementAvailable
                                  : metadata->MemoryEstimate().memoryAvailable));
    auto oldWorkspace = CreateCudaExecutionWorkspace(device, &diagnostics);
    if (!oldWorkspace)
        for (auto const& error : diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(oldWorkspace);
    auto previous = ExecuteCudaGraph(*plan, *oldWorkspace, 1, 1, &diagnostics);
    CHECK(previous);
    auto old = gpu::AcquireGeometry(previous, nullptr);
    CHECK(old);
    CHECK(old.Geometry().pointCount == (sourceTerminal ? 2u : multiGrow ? 6u :
        noiseGrow ? 12u : culledNoiseGrow ? 8u : 4u));
    auto const* oldPointer = old.Geometry().points.data;
    float3 oldRoot{};
    CHECK(cudaMemcpy(&oldRoot,oldPointer,sizeof(oldRoot),cudaMemcpyDeviceToHost)==cudaSuccess);
    if(!mixed && !predecessorGrow) CHECK(oldRoot.x==1&&oldRoot.y==2&&oldRoot.z==3);
    auto copyBytes = [](void const* source, size_t bytes) {
        std::vector<unsigned char> result(bytes);
        if (bytes && cudaMemcpy(result.data(), source, bytes, cudaMemcpyDeviceToHost)!=cudaSuccess)
            result.clear();
        return result;
    };
    auto const oldPoints = copyBytes(old.Geometry().points.data, old.Geometry().pointCount*sizeof(float3));
    auto const oldRest = copyBytes(old.Geometry().restPoints.data, old.Geometry().pointCount*sizeof(float3));
    auto const oldWidths = copyBytes(old.Geometry().widths.data, old.Geometry().pointCount*sizeof(float));
    auto const oldIds = copyBytes(old.Geometry().stableIds.data, old.Geometry().curveCount*sizeof(uint64_t));
    auto const oldOffsets = copyBytes(old.Geometry().curveOffsets.data,
        old.Geometry().curveOffsets.size*sizeof(uint32_t));
    auto const oldHairT = copyBytes(old.HairT().data, old.HairT().size*sizeof(float));
    auto const oldRootPrim = copyBytes(old.RootPrim().data, old.RootPrim().size*sizeof(int32_t));
    auto const oldRootUV = copyBytes(old.RootUV().data, old.RootUV().size*sizeof(float2));
    std::vector<uint64_t> oldIdValues(old.Geometry().curveCount);
    CHECK(!oldIdValues.size() || cudaMemcpy(oldIdValues.data(), old.Geometry().stableIds.data,
          oldIdValues.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost)==cudaSuccess);
    if (noiseGrow) CHECK(oldIdValues == std::vector<uint64_t>({7,11,19}));
    if (culledNoiseGrow) CHECK(oldIdValues == std::vector<uint64_t>({7,19}));
    auto const oldT = copyBytes(old.RootT().data, old.RootT().size*sizeof(float3));
    auto const oldB = copyBytes(old.RootB().data, old.RootB().size*sizeof(float3));
    auto const oldN = copyBytes(old.RootN().data, old.RootN().size*sizeof(float3));
    auto oldNamedLease = gpu::AcquireNamedChannel(previous, "value", nullptr);
    CHECK(oldNamedLease);
    auto const oldNamed = copyBytes(oldNamedLease.Bytes().data, oldNamedLease.Bytes().size);
    CHECK(oldPoints.size() == old.Geometry().pointCount*sizeof(float3) &&
          oldRest.size() == old.Geometry().pointCount*sizeof(float3) &&
          oldWidths.size() == old.Geometry().pointCount*sizeof(float) &&
          oldIds.size() == old.Geometry().curveCount*sizeof(uint64_t) &&
          oldOffsets.size() == old.Geometry().curveOffsets.size*sizeof(uint32_t) &&
          oldHairT.size() == old.HairT().size*sizeof(float) &&
          oldRootPrim.size() == old.RootPrim().size*sizeof(int32_t) &&
          oldRootUV.size() == old.RootUV().size*sizeof(float2) &&
          oldT.size() == old.RootT().size*sizeof(float3) &&
          oldB.size() == old.RootB().size*sizeof(float3) &&
          oldN.size() == old.RootN().size*sizeof(float3) &&
          oldNamed.size() == oldNamedLease.Bytes().size);

    // Keep this predecessor witness alive through the cap-ledger assertions:
    // retiring its CUDA lease immediately before the baseline can otherwise
    // make a later asynchronous retirement look like a failed-job mutation.
    UsdGenDiagnostics predecessorDiagnostics;
    std::shared_ptr<const UsdGenCudaExecutionPlan> predecessorPlan;
    std::unique_ptr<UsdGenCudaExecutionWorkspace> predecessorWorkspace;
    std::shared_ptr<const UsdGenDeviceGeneration> predecessor;
    gpu::CudaGeometryLease predecessorLease;
    // Exercise the actual predecessor Noise output, rather than inferring it
    // from Grow's generated shape.
    if (predecessorGrow) {
        auto predecessorDesc = desc;
        predecessorDesc.terminal = SdfPath("/Noise");
        predecessorPlan = CompileCudaGraph(usdGenUsd::Lower(predecessorDesc), &predecessorDiagnostics);
        CHECK(predecessorPlan);
        predecessorWorkspace = CreateCudaExecutionWorkspace(device, &predecessorDiagnostics);
        CHECK(predecessorWorkspace);
        predecessor = ExecuteCudaGraph(*predecessorPlan, *predecessorWorkspace, 1, 17,
                                        &predecessorDiagnostics);
        CHECK(predecessor);
        predecessorLease = gpu::AcquireGeometry(predecessor, nullptr);
        CHECK(predecessorLease && predecessorLease.Geometry().pointCount >= 2);
        float3 nonroot{};
        CHECK(cudaMemcpy(&nonroot, predecessorLease.Geometry().points.data + 1,
                         sizeof(nonroot), cudaMemcpyDeviceToHost) == cudaSuccess);
        auto const& original = curves.points[1];
        CHECK(nonroot.x != original[0] || nonroot.y != original[1] || nonroot.z != original[2]);
    }

    // Cap rejection is pre-submit and cannot replace the accepted owner or
    // change Pending. No physical free-memory heuristics enter this check.
    auto workspace = CreateCudaExecutionWorkspace(device, &diagnostics);
    CHECK(workspace);
    {
        auto before = pool->Snapshot();
        auto filler = pool->TryReserve(before.usableBytes - before.usedBytes,
                                       UsdGenExecutionResourceKind::Active);
        CHECK(filler);
        auto full = pool->Snapshot();
        CHECK(!CreateCudaExecutionJob(plan, *workspace, 2, 2, &diagnostics, previous));
        auto rejected = pool->Snapshot();
        CHECK(rejected.usedBytes == full.usedBytes && rejected.byKind == full.byKind);
        CHECK(!workspace->IsPoisoned());
    }
    diagnostics = {};
    auto job = CreateCudaExecutionJob(plan, *workspace, 2, 2, &diagnostics, previous);
    CHECK(job);
    CHECK(pool->Snapshot().byKind[size_t(UsdGenExecutionResourceKind::Pending)]>0);
    std::promise<bool> sourceDone;
    auto sourceReady = sourceDone.get_future();
    CHECK(ExecuteCudaJobSourceAsync(job, [&](bool ok) { sourceDone.set_value(ok); }));
    CHECK(sourceReady.wait_for(std::chrono::seconds(10)) == std::future_status::ready && sourceReady.get());
    if (predecessorGrow) {
        size_t const finalGrow = culledNoiseGrow ? 2u : 1u;
        for (size_t operatorIndex = 0; operatorIndex != finalGrow; ++operatorIndex) {
            std::promise<bool> accepted;
            auto ready = accepted.get_future();
            CHECK(ExecuteCudaJobOperatorAsync(job, operatorIndex,
                  [&](bool ok){accepted.set_value(ok);}) &&
                  ready.wait_for(std::chrono::seconds(10))==std::future_status::ready && ready.get());
        }
    } else if (withLength) {
        std::promise<bool> acceptedGrow;
        auto ready=acceptedGrow.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(job,0,[&](bool ok){acceptedGrow.set_value(ok);}));
        CHECK(ready.wait_for(std::chrono::seconds(10))==std::future_status::ready&&ready.get());
    }
    if (multiGrow) {
        std::promise<bool> acceptedGrow;
        auto ready=acceptedGrow.get_future();
        CHECK(ExecuteCudaJobOperatorAsync(job,0,[&](bool ok){acceptedGrow.set_value(ok);})&&
              ready.wait_for(std::chrono::seconds(10))==std::future_status::ready&&ready.get());
    }
    // These existing hooks address native relay phase 2 (Grow completion)
    // and phase 3 (the following named-plane topology completion).
    std::string const mode = argc > 1 ? argv[1] : "";
    if (mode == "--native")
        failNextCudaOperatorRelayWidthNativeCallbackForTesting();
    else if (mode == "--named-native")
        failNextCudaOperatorRelayCountsNativeCallbackForTesting();
    else if (mode == "--named")
        failNextCudaOperatorRelayCountsCallbackInstallForTesting();
    else
        failNextCudaOperatorRelayWidthCallbackInstallForTesting();
    auto beforeFailure = pool->Snapshot();
    std::promise<bool> growDone;
    auto growReady = growDone.get_future();
    size_t const failingGrow = predecessorGrow ? (culledNoiseGrow ? 2u : 1u) :
        multiGrow ? 1u : withLength ? 1u : 0u;
    CHECK(ExecuteCudaJobOperatorAsync(job, failingGrow, [&](bool ok) { growDone.set_value(ok); }));
    CHECK(growReady.wait_for(std::chrono::seconds(10)) == std::future_status::ready && !growReady.get());
    CHECK(workspace->IsPoisoned());
    job.reset();
    auto quarantined = pool->Snapshot();
    CHECK(quarantined.usedBytes > beforeFailure.usedBytes -
              beforeFailure.byKind[size_t(UsdGenExecutionResourceKind::Pending)] &&
          quarantined.byKind[size_t(UsdGenExecutionResourceKind::Pending)] == 0);
    CHECK(beforeFailure.byKind[size_t(UsdGenExecutionResourceKind::Pending)] > 0);
    float3 root{};
    CHECK(old.Geometry().points.data == oldPointer && previous->Identity().generation == 1);
    CHECK(cudaMemcpy(&root, oldPointer, sizeof(root), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(root.x == oldRoot.x && root.y == oldRoot.y && root.z == oldRoot.z);
    auto held = gpu::AcquireGeometry(previous, nullptr); CHECK(held);
    CHECK(copyBytes(held.Geometry().points.data, held.Geometry().pointCount*sizeof(float3)) == oldPoints);
    CHECK(copyBytes(held.Geometry().restPoints.data, held.Geometry().pointCount*sizeof(float3)) == oldRest);
    CHECK(copyBytes(held.Geometry().widths.data, held.Geometry().pointCount*sizeof(float)) == oldWidths);
    CHECK(copyBytes(held.Geometry().stableIds.data, held.Geometry().curveCount*sizeof(uint64_t)) == oldIds);
    CHECK(copyBytes(held.Geometry().curveOffsets.data, held.Geometry().curveOffsets.size*sizeof(uint32_t)) == oldOffsets);
    CHECK(copyBytes(held.HairT().data, held.HairT().size*sizeof(float)) == oldHairT);
    CHECK(copyBytes(held.RootPrim().data, held.RootPrim().size*sizeof(int32_t)) == oldRootPrim);
    CHECK(copyBytes(held.RootUV().data, held.RootUV().size*sizeof(float2)) == oldRootUV);
    CHECK(copyBytes(held.RootT().data, held.RootT().size*sizeof(float3)) == oldT);
    CHECK(copyBytes(held.RootB().data, held.RootB().size*sizeof(float3)) == oldB);
    CHECK(copyBytes(held.RootN().data, held.RootN().size*sizeof(float3)) == oldN);
    auto heldNamedLease = gpu::AcquireNamedChannel(previous, "value", nullptr); CHECK(heldNamedLease);
    CHECK(copyBytes(heldNamedLease.Bytes().data, heldNamedLease.Bytes().size) == oldNamed);
    // The failed job's owners intentionally remain quarantined to process
    // teardown; a later device sync is not substituted for its lost callback.
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    return 0;
}
