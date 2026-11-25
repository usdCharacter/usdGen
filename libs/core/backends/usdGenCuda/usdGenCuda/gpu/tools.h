#ifndef USDGEN_GPU_TOOLS_H
#define USDGEN_GPU_TOOLS_H

#include "generation.h"
#include "curveTileBounds.h"
#include "picking.h"
#include "pointOverride.h"
#include "usdGen/session.h"

#include <vector>

namespace usdGen::gpu {

// One caller-serialized stroke, tied to an engine and a borrowed CUDA stream.
// The engine and stream must outlive this object. No method edits USD or
// commits the engine: callers use the imaging Commit(LiveOverride) path to
// publish a staged move, retaining its existing notification ordering.
class CudaToolSession {
public:
    CudaToolSession(UsdGenSession& engine, cudaStream_t stream)
        : engine_(engine), stream_(stream) {}
    ~CudaToolSession();
    CudaToolSession(CudaToolSession const&) = delete;
    CudaToolSession& operator=(CudaToolSession const&) = delete;

    bool Begin(uint64_t graphVersion = 0);
    // Absolute replacements from the press-time base, never accumulation
    // from the previous move. Inputs stay on GPU and are consumed before return.
    bool UpdateIndexed(DeviceView<const int32_t> indices,
                       DeviceView<const float3> absolutePositions);
    // Escape: stage the original base for the next LiveOverride commit.
    bool Cancel();
    // Release: close without republishing. Return the last complete edit for
    // explicit commit/bake; authoring and undo are the app's responsibility.
    std::shared_ptr<const UsdGenDeviceGeneration> Close();
    bool Pick(PickQuery query, PickResult* result);
    bool Footprint(PickQuery query);
    // Read-only indices on this tool's stream, valid until the next successful
    // footprint or destruction. Complete downstream uses before either event.
    DeviceView<const int32_t> FootprintIndices() const { return picking_.footprint(); }
    bool active() const { return bool(base_); }
    char const* diagnostic() const { return diagnostic_.c_str(); }
private:
    bool Fail(char const* message);
    bool CheckCurrent(UsdGenGenerationConstPtr const& current);
    bool RefreshTileBounds(CudaGeometryLease const& lease,
                           DeviceView<const float3> revisedPoints,
                           std::vector<UsdGenDeviceTileMetadata>* tiles,
                           bool* boundsReady);
    UsdGenSession& engine_;
    cudaStream_t stream_;
    UsdGenGenerationConstPtr base_;
    UsdGenGenerationConstPtr expected_;
    std::shared_ptr<const UsdGenDeviceGeneration> latest_;
    std::shared_ptr<const UsdGenDeviceGeneration> staged_;
    uint64_t graphVersion_ = 0;
    uint64_t editToken_ = 0;
    CudaPicking picking_;
    CudaPointOverride override_;
    // Reused exclusively by this caller-serialized tool session. They are
    // always enqueued on stream_, never on an implicit/default stream.
    DeviceBuffer<CurveTileSpan> spans_;
    DeviceBuffer<CurveTileBoundsScratch> boundsScratch_;
    DeviceBuffer<float3> boundsMinimums_;
    DeviceBuffer<float3> boundsMaximums_;
    DeviceBuffer<uint32_t> boundsStatus_;
    std::string diagnostic_;
};

} // namespace usdGen::gpu
#endif
