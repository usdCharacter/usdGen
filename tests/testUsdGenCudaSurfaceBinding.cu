#include "usdGen/gpu/surfaceBinding.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace usdGen::gpu;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (false)

template <class T>
static bool Upload(DeviceBuffer<T>& device, std::vector<T> const& values) {
    if (device.reset(values.size()) != cudaSuccess) return false;
    return values.empty() || cudaMemcpy(device.data(), values.data(),
                                        values.size() * sizeof(T),
                                        cudaMemcpyHostToDevice) == cudaSuccess;
}

template <class T>
static DeviceView<const T> ConstView(DeviceBuffer<T> const& device) {
    return device.view();
}

static bool UploadUv(float2** device, std::vector<float2> const& values) {
    if (cudaMalloc(reinterpret_cast<void**>(device), values.size() * sizeof(float2)) != cudaSuccess)
        return false;
    return values.empty() || cudaMemcpy(*device, values.data(),
                                        values.size() * sizeof(float2),
                                        cudaMemcpyHostToDevice) == cudaSuccess;
}

template <class T>
static bool Download(DeviceView<const T> view, std::vector<T>* values) {
    values->resize(view.size);
    return view.size == 0 || cudaMemcpy(values->data(), view.data,
                                        view.size * sizeof(T),
                                        cudaMemcpyDeviceToHost) == cudaSuccess;
}

int main() {
    DeviceBuffer<float3> rest, current;
    DeviceBuffer<uint32_t> offsets, indices, badOffsets, badIndices;
    DeviceBuffer<int> skinPrim;
    float2* skinUv = nullptr;
    float2* badUv = nullptr;
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

    std::vector<float3> restHost{
        make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0),
        make_float3(1, 1, 0), make_float3(2, 0, 0)};
    std::vector<uint32_t> offsetsHost{0, 3, 7};
    // Face 0 is a triangle; face 1 is the quad 0,1,3,2.
    std::vector<uint32_t> indicesHost{0, 1, 2, 0, 1, 3, 2};
    std::vector<float3> currentHost{
        make_float3(0, 0, 1), make_float3(1, 0, 2), make_float3(0, 1, 3),
        make_float3(1, 1, 4), make_float3(2, 0, 5)};
    std::vector<int> rootsHost{0, 1};
    std::vector<float2> uvHost{make_float2(.25f, .25f), make_float2(.5f, .5f)};
    CHECK(Upload(rest, restHost) && Upload(current, currentHost) &&
          Upload(offsets, offsetsHost) && Upload(indices, indicesHost) &&
          Upload(skinPrim, rootsHost) && UploadUv(&skinUv, uvHost));

    CudaSurfaceBinding binding;
    CHECK(binding.Bind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) ==
          SurfaceBindingStatus::Ok);
    CHECK(binding.pending());
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::Ok);
    CHECK(!binding.pending() && binding.sampleCount() == 4);
    int currentDevice = -1;
    CHECK(cudaGetDevice(&currentDevice) == cudaSuccess &&
          binding.deviceIndex() == currentDevice);
    std::vector<uint32_t> selected;
    CHECK(Download(binding.sampleIndices(), &selected));
    // FPS starts at vertex zero, then picks vertex four, then the lowest-index
    // tie between vertices one and two.
    CHECK(selected == std::vector<uint32_t>({0, 4, 3, 1}));
    std::vector<float3> restSamples;
    CHECK(Download(binding.restSamples(), &restSamples) && restSamples.size() == 4);
    CHECK(restSamples[1].x == 2.0f && restSamples[2].x == 1.0f && restSamples[3].x == 1.0f);

    CHECK(binding.Update(ConstView(current), ConstView(skinPrim),
                         {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::Ok);
    std::vector<float3> currentSamples, targets;
    CHECK(Download(binding.currentSamples(), &currentSamples) && currentSamples.size() == 4);
    CHECK(currentSamples[0].z == 1.0f && currentSamples[1].z == 5.0f &&
          currentSamples[2].z == 4.0f && currentSamples[3].z == 2.0f);
    CHECK(Download(binding.rootTargets(), &targets) && targets.size() == 2);
    CHECK(std::fabs(targets[0].x - .25f) < 1e-6f &&
          std::fabs(targets[0].y - .25f) < 1e-6f &&
          std::fabs(targets[0].z - 1.75f) < 1e-6f);
    CHECK(std::fabs(targets[1].x - .5f) < 1e-6f &&
          std::fabs(targets[1].y - .5f) < 1e-6f &&
          std::fabs(targets[1].z - 2.5f) < 1e-6f);


    // A pose update gathers new samples but never changes the persistent FPS
    // result.  Bind in a second object also proves deterministic selection.
    auto changed = currentHost;
    for (auto& p : changed) p.x += 10.0f;
    CHECK(Upload(current, changed));
    CHECK(binding.Update(ConstView(current), ConstView(skinPrim),
                         {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::Ok);
    std::vector<uint32_t> selectedAfter;
    CHECK(Download(binding.sampleIndices(), &selectedAfter) && selectedAfter == selected);
    CudaSurfaceBinding second;
    CHECK(second.Bind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) ==
          SurfaceBindingStatus::Ok && second.Finish(stream) == SurfaceBindingStatus::Ok);
    std::vector<uint32_t> selectedSecond;
    CHECK(Download(second.sampleIndices(), &selectedSecond) && selectedSecond == selected);

    // Invalid topology is diagnosed by Finish and cannot replace a valid
    // binding.  Both unsupported polygon size and out-of-range indices are
    // tested without allowing a gather kernel to dereference them.
    DeviceBuffer<float3> badRest;
    auto nonFiniteRest = restHost;
    nonFiniteRest[0].z = std::numeric_limits<float>::infinity();
    CHECK(Upload(badRest, nonFiniteRest));
    CHECK(binding.Bind(ConstView(badRest), ConstView(offsets), 2, ConstView(indices), 4, stream) ==
          SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::NonFiniteInput);
    CHECK(Upload(badOffsets, std::vector<uint32_t>{0, 2, 7}));
    CHECK(binding.Bind(ConstView(rest), ConstView(badOffsets), 2, ConstView(indices), 4, stream) ==
          SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::InvalidTopology);
    std::vector<uint32_t> stillSelected;
    CHECK(Download(binding.sampleIndices(), &stillSelected) && stillSelected == selected);
    CHECK(Upload(badIndices, std::vector<uint32_t>{0, 1, 99, 0, 1, 3, 2}));
    CHECK(binding.Bind(ConstView(rest), ConstView(offsets), 2, ConstView(badIndices), 4, stream) ==
          SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::InvalidTopology);

    // Root UV and face validation are separate from surface topology.
    std::vector<float2> invalidUv{make_float2(1.1f, 0), make_float2(.5f, .5f)};
    CHECK(UploadUv(&badUv, invalidUv));
    // Fresh proof is parent-driven. It exposes only compact status/sample
    // copies; Finish is rejected while the parent owns terminal proof.
    CudaSurfaceBinding fresh;
    CHECK(!fresh.CanAcceptFreshUpdate() && !fresh.CanRollbackFreshUpdate());
    CHECK(fresh.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::Ok);
    CHECK(fresh.HasUnprovenFreshWork() && fresh.Finish(stream) == SurfaceBindingStatus::InvalidArgument);
    CHECK(fresh.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::InvalidArgument);
    CHECK(fresh.Bind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::InvalidArgument);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(fresh.CommitFreshBind() == SurfaceBindingStatus::Ok && fresh.sampleCount() == 4);
    std::vector<float3> freshPrior, freshAfter;
    CHECK(Download(fresh.currentSamples(), &freshPrior));
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {badUv, invalidUv.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(fresh.CommitFreshUpdate() == SurfaceBindingStatus::InvalidRootBinding);
    CHECK(!fresh.CanAcceptFreshUpdate() && !fresh.CanRollbackFreshUpdate());
    CHECK(Download(fresh.currentSamples(), &freshAfter) && freshAfter.size() == freshPrior.size());
    for (size_t i = 0; i < freshPrior.size(); ++i)
        CHECK(freshAfter[i].x == freshPrior[i].x && freshAfter[i].y == freshPrior[i].y && freshAfter[i].z == freshPrior[i].z);
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && fresh.CommitFreshUpdate() == SurfaceBindingStatus::Ok);
    // Reference outputs before any retired-storage adoption (same inputs as
    // the adopted update below, so the comparison is bitwise).
    std::vector<float3> preAdoptSamples, preAdoptTargets;
    CHECK(Download(fresh.currentSamples(), &preAdoptSamples) && preAdoptSamples.size() == 4);
    CHECK(Download(fresh.rootTargets(), &preAdoptTargets) && preAdoptTargets.size() == 2);
    CHECK(fresh.CanAcceptFreshUpdate() && fresh.CanRollbackFreshUpdate());
    // A proved update is visible to a following fresh solve but remains a
    // transaction: a second update/rebind cannot overtake it, and rollback
    // restores both samples and roots without CUDA work.
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::InvalidArgument);
    std::vector<float3> provisional;
    CHECK(Download(fresh.currentSamples(), &provisional) && provisional.size() == freshPrior.size());
    CHECK(fresh.RollbackFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(!fresh.CanAcceptFreshUpdate() && !fresh.CanRollbackFreshUpdate());
    CHECK(Download(fresh.currentSamples(), &freshAfter) && freshAfter.size() == freshPrior.size());
    for (size_t i = 0; i < freshPrior.size(); ++i)
        CHECK(freshAfter[i].x == freshPrior[i].x && freshAfter[i].y == freshPrior[i].y && freshAfter[i].z == freshPrior[i].z);
    CHECK(fresh.AcceptFreshUpdate() == SurfaceBindingStatus::InvalidArgument);
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && fresh.CommitFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(fresh.CanAcceptFreshUpdate() && fresh.CanRollbackFreshUpdate());
    CHECK(fresh.AcceptFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(!fresh.CanAcceptFreshUpdate() && !fresh.CanRollbackFreshUpdate());
    // The update above ran on adopted retired storage: its outputs must be
    // bitwise identical to the pre-adoption reference over the same inputs.
    std::vector<float3> adoptedSamples, adoptedTargets;
    CHECK(Download(fresh.currentSamples(), &adoptedSamples) && adoptedSamples.size() == preAdoptSamples.size());
    CHECK(Download(fresh.rootTargets(), &adoptedTargets) && adoptedTargets.size() == preAdoptTargets.size());
    for (size_t i = 0; i < preAdoptSamples.size(); ++i)
        CHECK(adoptedSamples[i].x == preAdoptSamples[i].x && adoptedSamples[i].y == preAdoptSamples[i].y &&
              adoptedSamples[i].z == preAdoptSamples[i].z);
    for (size_t i = 0; i < preAdoptTargets.size(); ++i)
        CHECK(adoptedTargets[i].x == preAdoptTargets[i].x && adoptedTargets[i].y == preAdoptTargets[i].y &&
              adoptedTargets[i].z == preAdoptTargets[i].z);
    // A root-count change falls back to release/realloc and still publishes
    // exact outputs; returning to the prior count re-adopts retired storage.
    CHECK(fresh.BeginFreshUpdate(ConstView(current), {}, {}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && fresh.CommitFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(fresh.AcceptFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(fresh.rootTargets().size == 0 && fresh.currentSamples().size == 4);
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && fresh.CommitFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(fresh.AcceptFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(Download(fresh.currentSamples(), &adoptedSamples) && adoptedSamples.size() == preAdoptSamples.size());
    CHECK(Download(fresh.rootTargets(), &adoptedTargets) && adoptedTargets.size() == preAdoptTargets.size());
    for (size_t i = 0; i < preAdoptSamples.size(); ++i)
        CHECK(adoptedSamples[i].x == preAdoptSamples[i].x && adoptedSamples[i].y == preAdoptSamples[i].y &&
              adoptedSamples[i].z == preAdoptSamples[i].z);
    for (size_t i = 0; i < preAdoptTargets.size(); ++i)
        CHECK(adoptedTargets[i].x == preAdoptTargets[i].x && adoptedTargets[i].y == preAdoptTargets[i].y &&
              adoptedTargets[i].z == preAdoptTargets[i].z);
    // Fresh error paths through the fused update kernel: rejection statuses
    // match the unfused sequence and the publication is retained.
    DeviceBuffer<float3> freshBadCurrent;
    auto freshNonFinite = changed;
    freshNonFinite[1].z = std::numeric_limits<float>::infinity();
    CHECK(Upload(freshBadCurrent, freshNonFinite));
    CHECK(fresh.BeginFreshUpdate(ConstView(freshBadCurrent), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(fresh.CommitFreshUpdate() == SurfaceBindingStatus::NonFiniteInput);
    DeviceBuffer<int> freshBadPrim;
    CHECK(Upload(freshBadPrim, std::vector<int>{0, 99}));
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(freshBadPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(fresh.CommitFreshUpdate() == SurfaceBindingStatus::InvalidRootBinding);
    float2* freshBadUv = nullptr;
    CHECK(UploadUv(&freshBadUv, std::vector<float2>{make_float2(.25f, .25f),
                    make_float2(std::numeric_limits<float>::quiet_NaN(), .5f)}));
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {freshBadUv, 2}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(fresh.CommitFreshUpdate() == SurfaceBindingStatus::InvalidRootBinding);
    CHECK(cudaFree(freshBadUv) == cudaSuccess);
    CHECK(Download(fresh.currentSamples(), &adoptedSamples) && adoptedSamples.size() == preAdoptSamples.size());
    for (size_t i = 0; i < preAdoptSamples.size(); ++i)
        CHECK(adoptedSamples[i].x == preAdoptSamples[i].x && adoptedSamples[i].y == preAdoptSamples[i].y &&
              adoptedSamples[i].z == preAdoptSamples[i].z);
    CHECK(fresh.BeginFreshUpdate({}, ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::InvalidArgument);
    CHECK(!fresh.HasUnprovenFreshWork());
    cudaGraph_t captured = nullptr;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::InvalidArgument);
    CHECK(cudaStreamEndCapture(stream, &captured) == cudaSuccess);
    if (captured) CHECK(cudaGraphDestroy(captured) == cudaSuccess);
    CHECK(fresh.BeginFreshUpdate(ConstView(current), ConstView(skinPrim), {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && fresh.CommitFreshUpdate() == SurfaceBindingStatus::Ok);
    CHECK(fresh.AcceptFreshUpdate() == SurfaceBindingStatus::Ok);
    // Clean preenqueue allocation rejection is retryable. Conversely, an
    // injected proof enqueue failure occurs after Bind submitted work: commit
    // must reject and the parent explicitly abandons the charged candidate.
    CudaSurfaceBinding freshRetry;
    failNextCudaSurfaceFreshProofAllocationForTesting();
    CHECK(freshRetry.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::CudaError);
    CHECK(!freshRetry.HasUnprovenFreshWork());
    CHECK(freshRetry.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && freshRetry.CommitFreshBind() == SurfaceBindingStatus::Ok);
    CudaSurfaceBinding freshSemantic;
    CHECK(freshSemantic.BeginFreshBind(ConstView(badRest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && freshSemantic.CommitFreshBind() == SurfaceBindingStatus::NonFiniteInput);
    CHECK(freshSemantic.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && freshSemantic.CommitFreshBind() == SurfaceBindingStatus::Ok);
    CudaSurfaceBinding freshUnsafe;
    failNextCudaSurfaceFreshProofEnqueueForTesting();
    CHECK(freshUnsafe.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::CudaError);
    CHECK(freshUnsafe.HasUnprovenFreshWork() && freshUnsafe.CommitFreshBind() == SurfaceBindingStatus::InvalidArgument);
    freshUnsafe.AbandonFresh();
    CudaSurfaceBinding freshPartial;
    failNextCudaSurfaceFreshProofAfterStatusForTesting();
    CHECK(freshPartial.BeginFreshBind(ConstView(rest), ConstView(offsets), 2, ConstView(indices), 4, stream) == SurfaceBindingStatus::CudaError);
    CHECK(freshPartial.HasUnprovenFreshWork() && freshPartial.CommitFreshBind() == SurfaceBindingStatus::InvalidArgument);
    freshPartial.AbandonFresh();
    CHECK(binding.Update(ConstView(current), ConstView(skinPrim),
                         {badUv, invalidUv.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::InvalidRootBinding);
    std::vector<float3> priorSamples;
    CHECK(Download(binding.currentSamples(), &priorSamples) && priorSamples.size() == 4);
    std::vector<float3> nonFinite = changed;
    nonFinite[2].y = std::numeric_limits<float>::quiet_NaN();
    CHECK(Upload(current, nonFinite));
    CHECK(binding.Update(ConstView(current), ConstView(skinPrim),
                         {skinUv, uvHost.size()}, stream) == SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::NonFiniteInput);
    CHECK(Download(binding.currentSamples(), &currentSamples) &&
          currentSamples[0].x == priorSamples[0].x);

    // Zero roots is a valid update and publishes an empty target view.
    CHECK(Upload(current, changed));
    CHECK(binding.Update(ConstView(current), {}, {}, stream) == SurfaceBindingStatus::Ok);
    CHECK(binding.Finish(stream) == SurfaceBindingStatus::Ok);
    CHECK(binding.rootCount() == 0 && binding.rootTargets().size == 0);

    // Coincident vertices are not emitted as duplicate RBF centers.  The
    // requested budget is five, but only four unique spatial centers exist.
    DeviceBuffer<float3> duplicateRest;
    auto duplicate = restHost;
    duplicate[1] = duplicate[0];
    CHECK(Upload(duplicateRest, duplicate));
    CudaSurfaceBinding duplicateBinding;
    CHECK(duplicateBinding.Bind(ConstView(duplicateRest), ConstView(offsets), 2,
                                ConstView(indices), 5, stream) == SurfaceBindingStatus::Ok);
    CHECK(duplicateBinding.Finish(stream) == SurfaceBindingStatus::Ok &&
          duplicateBinding.sampleCount() == 4);
    std::vector<uint32_t> duplicateIndices;
    CHECK(Download(duplicateBinding.sampleIndices(), &duplicateIndices));
    for (size_t i = 0; i < duplicateIndices.size(); ++i)
        for (size_t j = i + 1; j < duplicateIndices.size(); ++j)
            CHECK(duplicateIndices[i] != duplicateIndices[j]);
    CudaSurfaceBinding freshDuplicate;
    CHECK(freshDuplicate.BeginFreshBind(ConstView(duplicateRest), ConstView(offsets), 2, ConstView(indices), 5, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && freshDuplicate.CommitFreshBind() == SurfaceBindingStatus::Ok &&
          freshDuplicate.sampleCount() == 4);
    auto tiny = restHost;
    for (size_t i = 0; i < tiny.size(); ++i)
        tiny[i] = make_float3(static_cast<float>(i) * 1.0e-20f, 0, 0);
    CHECK(Upload(duplicateRest, tiny));
    CudaSurfaceBinding tinyBinding;
    CHECK(tinyBinding.Bind(ConstView(duplicateRest), ConstView(offsets), 2,
                           ConstView(indices), 5, stream) == SurfaceBindingStatus::Ok);
    CHECK(tinyBinding.Finish(stream) == SurfaceBindingStatus::Ok &&
          tinyBinding.sampleCount() == 5);

    // A completely empty surface and zero sample budget are valid.
    DeviceBuffer<float3> emptyVertices;
    DeviceBuffer<uint32_t> emptyOffsets;
    CHECK(Upload(emptyVertices, {}) && Upload(emptyOffsets, {0}));
    CudaSurfaceBinding empty;
    CHECK(empty.Bind(ConstView(emptyVertices), ConstView(emptyOffsets), 0, {}, 20, stream) ==
          SurfaceBindingStatus::Ok);
    CHECK(empty.Finish(stream) == SurfaceBindingStatus::Ok &&
          empty.sampleCount() == 0 && empty.restSamples().size == 0);
    CHECK(empty.Update(ConstView(emptyVertices), {}, {}, stream) == SurfaceBindingStatus::Ok);
    CHECK(empty.Finish(stream) == SurfaceBindingStatus::Ok && empty.rootTargets().size == 0);
    CudaSurfaceBinding freshEmpty;
    CHECK(freshEmpty.BeginFreshBind(ConstView(emptyVertices), ConstView(emptyOffsets), 0, {}, 20, stream) == SurfaceBindingStatus::Ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess && freshEmpty.CommitFreshBind() == SurfaceBindingStatus::Ok &&
          freshEmpty.sampleCount() == 0);

    cudaFree(skinUv);
    cudaFree(badUv);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("testUsdGenCudaSurfaceBinding: PASS");
    return 0;
}
