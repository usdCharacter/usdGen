#ifndef USDGEN_GPU_CURVE_GROW_H
#define USDGEN_GPU_CURVE_GROW_H

#include "curveGeometry.h"
#include "imageSampler.h"

#include <cstdint>
#include <memory>

namespace usdGen { class UsdGenExecutionMemoryReservation; }
namespace usdGen::gpu {

namespace detail {
// Valid for a nonempty source span and u in [0,1]. Float conversion of a
// large last index can round upward (even to 2^32); clamp before the cast.
__host__ __device__ inline uint32_t CurveGrowWidthLowerIndex(float u, uint32_t count) {
    float const position = u * float(count - 1);
    return position >= float(count - 1) ? count - 1 : uint32_t(position);
}
}

// Immutable device-side C3 input. The caller must have already ordered each
// producer-ready view onto the BeginFresh stream; `lifetime` must keep every
// view (including the frame owners) alive until the terminal callback is
// committed or quarantined.
// Named extension planes intentionally remain an executor-level topology
// transform and are not part of this producer.
struct CurveGrowInput {
    DeviceCurveGeometryView geometry;
    DeviceView<const int32_t> rootPrim;
    DeviceView<const float2> rootUV;
    DeviceView<const float3> rootT, rootB, rootN;
    // Optional sorted source-frame ID domain.  When present, rootT/B/N are
    // indexed by this immutable domain rather than by the active geometry's
    // compacted curve ordinal.  Empty preserves the original ordinal ABI.
    DeviceView<const uint64_t> frameStableIds;
};

enum class CurveGrowDirection : uint8_t { RootNormal, RootTangent, Literal };

struct CurveGrowControls {
    uint32_t cvCount = 2;
    int seed = 0;
    // CPU Grow computes the random target in double then stores float.  Keep
    // these authored controls double so a CUDA lowering does not lose parity
    // before its kernel sees them.
    double length = 1.0;
    double randomLo = 1.0, randomHi = 1.0;
    // Degrees rotated about root-frame B, constrained to [-90, 90] by
    // BeginFresh (plan/04 §2.2).
    float lift = 0.0f;
    float fallbackWidth = 0.01f;
    CurveGrowDirection direction = CurveGrowDirection::RootNormal;
    float3 literalDirection = {0.0f, 1.0f, 0.0f};
    // Root-normal rotation after lift; random=1 spans a full circle.
    // Appended to preserve aggregate initialization order.
    float azimuth = 0.0f;
    float azimuthRandom = 0.0f;
};

enum class CurveGrowStatus {
    Ok, InvalidArgument, NonFiniteInput, InvalidTopology, NoPendingUpdate, CudaError
};

struct CurveGrowRequirements {
    size_t pointCount = 0;
    size_t outputBytes = 0;
    size_t statusBytes = 0;
    // Optional length-map texels plus one sampled scalar per input curve.
    size_t peakBytes = 0;
};

CurveGrowStatus GetCurveGrowRequirements(size_t curveCount, uint32_t cvCount,
                                         CurveGrowRequirements* result);

// Fresh C3 topology producer.  It borrows already-uploaded C3 source planes,
// but writes every output topology/value plane into private storage.  The
// caller supplies an immutable owner for those borrowed views; losing native
// completion proof intentionally abandons that owner with quarantined work.
class CudaCurveGrow {
public:
    CudaCurveGrow() = default;
    ~CudaCurveGrow();
    CudaCurveGrow(CudaCurveGrow const&) = delete;
    CudaCurveGrow& operator=(CudaCurveGrow const&) = delete;

    CurveGrowStatus BeginFresh(CurveGrowInput input,
        std::shared_ptr<const void> inputLifetime, CurveGrowControls controls,
        cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation = nullptr);
    CurveGrowStatus FinishFreshAsync(cudaStream_t stream,
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata);
    // Call only after FinishFreshAsync returned Ok *and* its exact native
    // callback reported cudaSuccess. A pending host status is deliberately
    // not a completion proof and leaves this owner conservatively pending.
    // Run on a host completion relay, never inside the CUDA callback: retiring
    // the input owner or a rejected candidate may invoke CUDA release APIs.
    CurveGrowStatus CommitFreshFinish();
    void MarkUnprovenWork() noexcept { unprovenWork_ = true; }
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    bool pending() const noexcept { return pendingWork_; }
    uint64_t generation() const noexcept { return generation_; }
    int deviceIndex() const noexcept { return deviceIndex_; }
    size_t curveCount() const noexcept { return curves_; }
    size_t pointCount() const noexcept { return points_; }
    DeviceCurveGeometryView view() const noexcept;
    DeviceView<const float> hairT() const noexcept;
    DeviceView<const int32_t> rootPrim() const noexcept;
    DeviceView<const float2> rootUV() const noexcept;
    DeviceView<const float3> rootT() const noexcept;
    DeviceView<const float3> rootB() const noexcept;
    DeviceView<const float3> rootN() const noexcept;
    CurveGrowStatus recordUse(cudaStream_t stream);
    CurveGrowStatus waitOn(cudaStream_t stream) const;
    void ReclassifyPublishedGeneration() noexcept;
    size_t ExclusiveRetainedBytes() const noexcept;

private:
    struct Storage {
        DeviceBuffer<float3> points, restPoints, rootT, rootB, rootN;
        DeviceBuffer<float> widths, hairT;
        DeviceBuffer<uint32_t> offsets;
        DeviceBuffer<uint64_t> stableIds;
        DeviceBuffer<int32_t> rootPrim;
        DeviceBuffer<float2> rootUV;
        void quarantine() noexcept;
        void reclassify(UsdGenExecutionResourceKind) noexcept;
        size_t bytes() const noexcept;
        CurveGrowStatus recordUse(cudaStream_t);
        CurveGrowStatus waitOn(cudaStream_t) const;
        CurveGrowStatus synchronizeUse() const;
    } active_, pending_;
    CurveGrowStatus validate(CurveGrowInput const&, std::shared_ptr<const void> const&,
                             CurveGrowControls const&, size_t*) const;
    CurveGrowStatus validateStream(cudaStream_t) const;
    void discardPending() noexcept;
    std::shared_ptr<const void> inputOwner_;
    std::unique_ptr<std::shared_ptr<const void>> inputQuarantineOwner_;
    CurveGrowInput input_{};
    DeviceBuffer<int> error_;
    int* hostError_ = nullptr;
    UsdGenExecutionResourcePermit hostErrorPermit_;
    cudaEvent_t ready_ = nullptr;
    cudaStream_t producerStream_ = nullptr;
    size_t curves_ = 0, points_ = 0, pendingCurves_ = 0, pendingPoints_ = 0;
    uint64_t generation_ = 0;
    int deviceIndex_ = -1;
    bool pendingWork_ = false, finishScheduled_ = false, unprovenWork_ = false;
};

} // namespace usdGen::gpu
#endif
