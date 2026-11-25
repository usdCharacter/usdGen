#ifndef USDGEN_GPU_NAMED_CHANNEL_TOPOLOGY_H
#define USDGEN_GPU_NAMED_CHANNEL_TOPOLOGY_H

#include "generation.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen { class UsdGenExecutionMemoryReservation; }
namespace usdGen::gpu {

// The topology transform is deliberately split at its terminal stream
// callback.  Begin owns the source/topology leases, private output planes and
// validation status until the callback proves completion; Commit is then the
// sole host-only publication transition.  Callback code must only relay its
// status -- it must not call Commit, destroy the candidate, or make CUDA API
// calls.  A candidate that loses its callback/proof quarantines its private
// CUDA and pinned-status storage rather than freeing potentially in-flight
// memory.
enum class CudaNamedChannelTopologyMode : uint8_t {
    Resample,
    Compaction
};

class CudaNamedChannelTopologyCandidate {
public:
    // A raw input belongs to an execution-job candidate rather than a
    // published generation. `bytes` and both raw geometry views passed to
    // BeginRaw must remain valid until this candidate reaches CommitPlanes or
    // Quarantine. `lifetime` can retain an immutable job owner when available;
    // it supplements, but cannot infer, the caller's raw-buffer ownership.
    struct RawInputPlane {
        UsdGenDeviceChannelMetadata metadata;
        DeviceView<const unsigned char> bytes;
        // Optional producer-event carrier for `bytes`. The caller retains
        // this buffer (normally through `lifetime`) until the candidate is
        // committed or quarantined; BeginRaw enqueues its ready-event wait on
        // the transform stream before any kernel can read the view.
        DeviceBuffer<unsigned char> const* readyBuffer = nullptr;
    };

    CudaNamedChannelTopologyCandidate();
    ~CudaNamedChannelTopologyCandidate();
    CudaNamedChannelTopologyCandidate(CudaNamedChannelTopologyCandidate const&) = delete;
    CudaNamedChannelTopologyCandidate& operator=(CudaNamedChannelTopologyCandidate const&) = delete;

    // Validates and enqueues all transforms without synchronizing.  It must
    // run on the shared CUDA device and outside stream capture.  No output is
    // visible through either input generation before Commit succeeds.
    cudaError_t Begin(std::shared_ptr<const UsdGenDeviceGeneration> source,
                      std::shared_ptr<const UsdGenDeviceGeneration> topology,
                      uint64_t generation, cudaStream_t stream,
                      CudaNamedChannelTopologyMode mode,
                      std::string* reason = nullptr,
                      UsdGenExecutionMemoryReservation* reservation = nullptr);

    // Job-facing counterpart to Begin. It never creates a generation or
    // retains an old published output: on proof, CommitPlanes transfers only
    // the private rebuilt planes for a later final-publication transaction.
    cudaError_t BeginRaw(DeviceCurveGeometryView source,
                         DeviceCurveGeometryView target,
                         std::vector<RawInputPlane> inputs,
                         cudaStream_t stream,
                         CudaNamedChannelTopologyMode mode,
                         std::shared_ptr<const void> lifetime = {},
                         std::string* reason = nullptr,
                         UsdGenExecutionMemoryReservation* reservation = nullptr);

    // Enqueues the validation-status readback and exactly one terminal native
    // callback. `callback` receives the stream terminal status.  A host relay
    // must subsequently call Commit with that exact status.
    cudaError_t FinishAsync(
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept,
        void* userdata);

    // The callback/proof relay calls this once.  A cudaSuccess status plus a
    // zero device validation result publishes an immutable COW revision;
    // otherwise this returns null and leaves both input generations intact.
    std::shared_ptr<const UsdGenDeviceGeneration> Commit(
        cudaError_t completionStatus, std::string* reason = nullptr);

    // Raw-mode publication transfer. `planes` must be empty; it receives the
    // candidate's unique buffers only after the exact native callback status
    // has been observed. Generation mode rejects this entry point.
    bool CommitPlanes(cudaError_t completionStatus,
                      std::vector<CudaNamedChannelPlane>* planes,
                      std::string* reason = nullptr);

    // Explicit fail-closed escape hatch for a lost relay/device.  It is safe
    // to call repeatedly and makes this candidate permanently unusable.
    void Quarantine() noexcept;
    bool HasUnprovenWork() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Rebuild every Generic named channel from `source` against a completed
// topology generation.  Point/Float32 channels use the same indexed-CV
// interpolation as CudaCurveResample, Point/Int32 channels choose the nearest
// source CV, Primitive channels follow stable curve ids, and Groom channels
// (constant primvars) are copied once.  The returned revision owns private
// output planes over `topology`; source overlays are read-only inputs.
//
// A null result leaves both input generations untouched and writes `reason`.
// Unsupported metadata is rejected rather than retyped, truncated, or
// inherited across the changed topology. This compatibility wrapper is built
// on CudaNamedChannelTopologyCandidate and synchronizes only after its
// terminal callback has been installed.
std::shared_ptr<const UsdGenDeviceGeneration>
TransformCudaNamedChannelsForResample(
    std::shared_ptr<const UsdGenDeviceGeneration> const& source,
    std::shared_ptr<const UsdGenDeviceGeneration> const& topology,
    uint64_t generation, cudaStream_t stream, std::string* reason = nullptr,
    UsdGenExecutionMemoryReservation* reservation = nullptr);

// As above, but `topology` is a completed stable compaction result.  Output
// curves are located by stable id in `source`, which makes survivor mapping
// explicit and rejects duplicate/missing ids transactionally.
std::shared_ptr<const UsdGenDeviceGeneration>
TransformCudaNamedChannelsForCompaction(
    std::shared_ptr<const UsdGenDeviceGeneration> const& source,
    std::shared_ptr<const UsdGenDeviceGeneration> const& topology,
    uint64_t generation, cudaStream_t stream, std::string* reason = nullptr,
    UsdGenExecutionMemoryReservation* reservation = nullptr);

} // namespace usdGen::gpu

#endif
