#ifndef USDGEN_GPU_NON_WIDTH_COMPARE_H
#define USDGEN_GPU_NON_WIDTH_COMPARE_H

#include "curveGeometry.h"
#include "rootFrames.h"
#include "usdGen/deviceGeneration.h"

#include <memory>
#include <vector>

namespace usdGen { class UsdGenExecutionMemoryReservation; }
namespace usdGen::gpu {

// Borrowed, ordered Generic-channel view.  This is deliberately a view type,
// not the executor's unique-owner CudaNamedChannelPlane.
struct NonWidthNamedPlaneView {
    UsdGenDeviceChannelMetadata metadata;
    DeviceView<const unsigned char> bytes;
    DeviceBuffer<unsigned char>* readyBuffer = nullptr;
};
struct NonWidthChunkDesc {
    uint64_t firstCurve = 0, curveCount = 0, liveCount = 0;
    uint64_t firstCv = 0, cvCount = 0;
    uint32_t tile = 0, surface = 0;
};
struct CurveFullNonWidthInput {
    DeviceCurveGeometryView geometry; // widths are intentionally excluded
    DeviceView<const float> hairT;
    DeviceView<const int32_t> rootPrim;
    DeviceView<const float2> rootUV;
    RestRootFrames frames;
    std::vector<NonWidthChunkDesc> chunks;
    std::vector<NonWidthNamedPlaneView> named;
};

struct CurveFullNonWidthCompareResult {
    int error = 0;
    int equal = 1;
};

// Async proof that two WidthBlend inputs differ only by width. Width is never
// read. Optional rest/hair/root-frame/mask views compare absent-vs-present as
// unequal; supplied T/B/N must be a complete ordinal-aligned trio whose
// stableIds are absent or exactly the geometry IDs. Named input is initially
// restricted to tightly packed typed scalar storage (stride=arity*sizeof(T));
// descriptor disagreement is a valid proved unequal result, while malformed
// views/offsets are cudaErrorInvalidValue. Curve type/basis/wrap are not part
// of this buffer primitive and remain executor-level policy.
class CudaCurveFullNonWidthCompare {
public:
    CudaCurveFullNonWidthCompare() = default;
    ~CudaCurveFullNonWidthCompare();
    CudaCurveFullNonWidthCompare(CudaCurveFullNonWidthCompare const&) = delete;
    CudaCurveFullNonWidthCompare& operator=(CudaCurveFullNonWidthCompare const&) = delete;
    cudaError_t BeginFresh(CurveFullNonWidthInput const&, CurveFullNonWidthInput const&,
                           std::shared_ptr<const void> lifetime, cudaStream_t,
                           UsdGenExecutionMemoryReservation* reservation = nullptr);
    cudaError_t EnqueueFreshStatus(cudaStream_t);
    cudaError_t CommitFreshFinish(bool* equal);
    void Quarantine() noexcept;
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    size_t ExclusiveRetainedBytes() const noexcept;
private:
    DeviceBuffer<unsigned char> result_;
    CurveFullNonWidthCompareResult* hostResult_ = nullptr;
    UsdGenExecutionResourcePermit hostPermit_;
    cudaEvent_t ready_ = nullptr;
    std::shared_ptr<const void> lifetime_;
    std::unique_ptr<std::shared_ptr<const void>> quarantineLifetime_;
    int device_ = -1;
    bool running_ = false, statusEnqueued_ = false, failed_ = false, unprovenWork_ = false;
};
} // namespace usdGen::gpu
#endif
