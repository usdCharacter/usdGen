#ifndef USDGEN_GPU_SURFACE_BINDING_H
#define USDGEN_GPU_SURFACE_BINDING_H

#include "deviceBuffer.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace usdGen { namespace gpu {

// Device-only rest-surface binding.  Face offsets are zero-based and contain
// faceCount+1 entries; every face must have exactly three or four vertices.
// skinPrim indexes a face and skinPrimUv is (u,v): barycentric on triangles
// and bilinear on quads.  The caller owns every input view until Finish.
enum class SurfaceBindingStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidTopology,
    InvalidRootBinding,
    CudaError
};

class CudaSurfaceBinding {
public:
    CudaSurfaceBinding() = default;
    ~CudaSurfaceBinding();
    CudaSurfaceBinding(const CudaSurfaceBinding&) = delete;
    CudaSurfaceBinding& operator=(const CudaSurfaceBinding&) = delete;

    // Selects persistent rest-space samples with deterministic spatial
    // farthest-point sampling.  The first sample and all exact distance ties
    // choose the lowest vertex index.  A budget larger than the vertex count
    // selects every distinct position; a zero budget is valid and selects none.
    SurfaceBindingStatus Bind(DeviceView<const float3> restVertices,
                              DeviceView<const uint32_t> faceOffsets,
                              size_t faceCount,
                              DeviceView<const uint32_t> faceIndices,
                              size_t sampleBudget, cudaStream_t stream);

    // Gathers current samples and current root targets.  Binding topology and
    // sample indices are persistent: Update never re-samples the surface.
    SurfaceBindingStatus Update(DeviceView<const float3> currentVertices,
                                DeviceView<const int32_t> skinPrim,
                                DeviceView<const float2> skinPrimUv,
                                cudaStream_t stream);

    // Completes the one pending Bind or Update.  Only the scalar validation
    // flag and actual sample count are copied to the host; no surface or root
    // geometry is read back.
    SurfaceBindingStatus Finish(cudaStream_t stream);

    // Fresh paths enqueue only producer work plus compact pinned D2H proof.
    // The parent owns the terminal CUDA callback/event and must call exactly
    // one matching CommitFresh* only after it has proved that terminal
    // success and launcher return.  These calls perform no CUDA work, wait,
    // event query, callback installation, or device allocation.
    SurfaceBindingStatus BeginFreshBind(DeviceView<const float3> restVertices,
                                       DeviceView<const uint32_t> faceOffsets,
                                       size_t faceCount,
                                       DeviceView<const uint32_t> faceIndices,
                                       size_t sampleBudget, cudaStream_t stream,
                                       UsdGenExecutionMemoryReservation* reservation = nullptr);
    SurfaceBindingStatus BeginFreshUpdate(DeviceView<const float3> currentVertices,
                                         DeviceView<const int32_t> skinPrim,
                                         DeviceView<const float2> skinPrimUv,
                                         cudaStream_t stream,
                                         UsdGenExecutionMemoryReservation* reservation = nullptr);
    SurfaceBindingStatus CommitFreshBind();
    SurfaceBindingStatus CommitFreshUpdate();
    // A fresh update is provisionally visible after its proof so a following
    // fresh RBF solve can consume its samples/roots.  The parent must resolve
    // it after downstream publication proof.  These are host-only ownership
    // exchanges: they do not submit CUDA, wait, or free device storage.
    SurfaceBindingStatus AcceptFreshUpdate();
    SurfaceBindingStatus RollbackFreshUpdate();
    // Pure transaction preflight; see CudaRbfBinding's matching predicates.
    bool CanAcceptFreshUpdate() const noexcept;
    bool CanRollbackFreshUpdate() const noexcept;
    // Parent terminal failure, callback-install failure, or a lost device
    // permanently abandons candidate storage; the owning relay keeps this
    // binding charged rather than freeing memory with no native proof.
    void AbandonFresh() noexcept;
    bool HasUnprovenFreshWork() const noexcept { return freshPending_ || freshUnproven_; }

    bool pending() const { return pending_; }
    size_t sampleCount() const { return sampleCount_; }
    size_t rootCount() const { return rootCount_; }
    size_t vertexCount() const { return vertexCount_; }
    size_t faceCount() const { return faceCount_; }
    int deviceIndex() const { return deviceIndex_; }
    DeviceView<const uint32_t> sampleIndices() const { return {sampleIndices_.data(), sampleCount_}; }
    DeviceView<const float3> restSamples() const { return {restSamples_.data(), sampleCount_}; }
    DeviceView<const float3> currentSamples() const { return {currentSamples_.data(), sampleCount_}; }
    DeviceView<const float3> rootTargets() const { return rootTargets_.view(); }
    const char* diagnostic() const { return diagnostic_.c_str(); }

private:
    SurfaceBindingStatus fail(SurfaceBindingStatus status, const char* message);
    SurfaceBindingStatus validateCommon(DeviceView<const float3> vertices,
                                        DeviceView<const uint32_t> offsets,
                                        size_t faces,
                                        DeviceView<const uint32_t> indices) const;
    SurfaceBindingStatus validateStream(cudaStream_t stream) const;
    void discardPending();
    SurfaceBindingStatus beginFreshProof(cudaStream_t stream, bool bind);
    SurfaceBindingStatus commitFresh(bool bind);
    void discardFreshProof(bool abandon) noexcept;

    DeviceBuffer<int> error_;
    DeviceBuffer<uint32_t> sampleIndices_, pendingSampleIndices_;
    DeviceBuffer<double> pendingNearest_;
    DeviceBuffer<uint32_t> pendingActualSampleCount_;
    DeviceBuffer<float3> restSamples_, pendingRestSamples_;
    DeviceBuffer<float3> currentSamples_, pendingCurrentSamples_;
    DeviceBuffer<float3> rootTargets_, pendingRootTargets_;
    DeviceBuffer<uint32_t> faceOffsets_, pendingFaceOffsets_;
    DeviceBuffer<uint32_t> faceIndices_, pendingFaceIndices_;
    // Fresh commits first park the prior publication here, then release this
    // ordinary-worker retirement set.  This avoids move assignment freeing a
    // live published buffer while installing a proved candidate.
    DeviceBuffer<uint32_t> retiredSampleIndices_, retiredFaceOffsets_, retiredFaceIndices_;
    DeviceBuffer<float3> retiredRestSamples_, retiredCurrentSamples_, retiredRootTargets_;
    cudaEvent_t ready_ = nullptr;
    int* freshCode_ = nullptr;
    uint32_t* freshActual_ = nullptr;
    UsdGenExecutionResourcePermit freshCodePermit_, freshActualPermit_;
    // Only set while a fresh launcher delegates to Bind/Update.  Accepted
    // candidate buffers retain their child permits after this pointer clears.
    UsdGenExecutionMemoryReservation* freshReservation_ = nullptr;

    size_t vertexCount_ = 0, faceCount_ = 0, indexCount_ = 0;
    size_t sampleCount_ = 0, rootCount_ = 0;
    size_t pendingVertexCount_ = 0, pendingFaceCount_ = 0;
    size_t pendingIndexCount_ = 0, pendingSampleCount_ = 0;
    size_t pendingRootCount_ = 0;
    size_t retiredRootCount_ = 0;
    bool pending_ = false;
    bool pendingBind_ = false;
    bool freshMode_ = false;
    bool freshLaunching_ = false;
    bool latestFresh_ = false;
    bool freshPending_ = false;
    bool freshUnproven_ = false;
    bool freshUpdatePendingAcceptance_ = false;
    bool bound_ = false;
    int deviceIndex_ = -1;
    std::string diagnostic_;
};

// Test-only fresh failure seams. They affect no legacy operation.
void failNextCudaSurfaceFreshProofAllocationForTesting();
void failNextCudaSurfaceFreshProofEnqueueForTesting();
void failNextCudaSurfaceFreshProofAfterStatusForTesting();
void TestFailNextFreshSurfaceResolvePreflight() noexcept;
void TestFailNextFreshSurfaceResolveCommit() noexcept;
uint64_t FreshSurfaceAcceptAttemptCountForTesting() noexcept;
uint64_t FreshSurfaceRollbackAttemptCountForTesting() noexcept;

}} // namespace usdGen::gpu

#endif
