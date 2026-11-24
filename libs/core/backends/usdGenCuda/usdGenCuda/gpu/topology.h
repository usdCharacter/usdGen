#ifndef USDGEN_GPU_TOPOLOGY_H
#define USDGEN_GPU_TOPOLOGY_H

#include "curveGeometry.h"

namespace usdGen::gpu {

struct CurveTopologyCompareResult {
    int error = 0;
    int equal = 1;
};

// Compares only the layout identity of two curve sets.  Point positions and
// all other per-point channels are deliberately not inspected.  The function
// synchronously publishes only the scalar result; geometry never crosses to
// the host.  cudaErrorInvalidValue denotes a malformed topology (including
// bad offset sentinels), while a well-formed but different topology returns
// cudaSuccess and sets equal=false.
cudaError_t CompareCurveTopology(DeviceCurveGeometryView a,
                                 DeviceCurveGeometryView b,
                                 cudaStream_t stream, bool* equal,
                                 UsdGenExecutionMemoryReservation* reservation = nullptr);

// Fresh comparator for a parent-owned finalization relay.  BeginFresh and
// EnqueueFreshStatus enqueue only; the parent batches this scalar D2H with its
// tile metadata copies under one native callback.  CommitFreshFinish is
// host-only and is valid only after that callback proved cudaSuccess and the
// launcher returned.  A valid unequal result is cudaSuccess with equal=false.
// Bad offsets or ID shape/provenance are rejected (arbitrary ID values are
// compared, not validated as an ordering); a proven device semantic error is
// cudaErrorInvalidValue and leaves the caller's output unchanged.
class CudaCurveTopologyCompare {
public:
    CudaCurveTopologyCompare() = default;
    ~CudaCurveTopologyCompare();
    CudaCurveTopologyCompare(CudaCurveTopologyCompare const&) = delete;
    CudaCurveTopologyCompare& operator=(CudaCurveTopologyCompare const&) = delete;

    cudaError_t BeginFresh(DeviceCurveGeometryView, DeviceCurveGeometryView,
                           cudaStream_t,
                           UsdGenExecutionMemoryReservation* reservation = nullptr);
    cudaError_t EnqueueFreshStatus(cudaStream_t);
    cudaError_t CommitFreshFinish(bool* equal);
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }

private:
    DeviceBuffer<unsigned char> result_;
    CurveTopologyCompareResult* hostResult_ = nullptr;
    UsdGenExecutionResourcePermit hostResultPermit_;
    cudaEvent_t ready_ = nullptr;
    int deviceIndex_ = -1;
    bool freshRunning_ = false;
    bool freshStatusEnqueued_ = false;
    bool freshFailed_ = false;
    bool unprovenWork_ = false;
};

} // namespace usdGen::gpu

#endif // USDGEN_GPU_TOPOLOGY_H
