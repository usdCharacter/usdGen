#ifndef USDGEN_GPU_ROOT_FRAMES_H
#define USDGEN_GPU_ROOT_FRAMES_H

#include "deviceBuffer.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace usdGen::gpu {

// The authored normal layout follows the corresponding USD primvar
// interpolation.  FaceVarying has one value per entry in faceIndices.
enum class RootFrameNormalDomain : uint8_t {
    None,
    Constant,
    Uniform,
    Vertex,
    FaceVarying
};

enum class RootFrameStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidTopology,
    InvalidRootBinding,
    InvalidAuthoredFrame,
    CudaError,
    NoPendingUpdate
};

// The axis ordering is intentionally the Noise contract: local displacement
// (x,y,z) maps to tangent*x + binormal*y + normal*z.  origin is included for
// transport clients.  All views name immutable published storage.
struct RestRootFrames {
    DeviceView<const float3> tangent;
    DeviceView<const float3> binormal;
    DeviceView<const float3> normal;
    // Optional source-order stable IDs. Consumers such as Noise use these to
    // resolve frames after a topology-preserving reorder/compaction without
    // copying frame planes on the host.
    DeviceView<const uint64_t> stableIds;
};

// Computes rest-space root frames wholly on the owning CUDA device. Inputs are
// borrowed until Finish. Optional authoredFrames is 16 doubles per curve in
// USD/Gf row-major, row-vector layout: TransformDir(e_x/e_y/e_z) reads rows
// 0/1/2 and Transform(0) reads row 3. It is validated affine/orthonormal and
// converted to the published float views without resampling or renormalizing.
// recordUse stores one last-use event per buffer, so the owning scheduler must
// serialize consumers (or hold an owning snapshot); independent concurrent
// consumers are not aggregated by this helper.
class CudaRestRootFrames {
public:
    CudaRestRootFrames() = default;
    ~CudaRestRootFrames();
    CudaRestRootFrames(CudaRestRootFrames const&) = delete;
    CudaRestRootFrames& operator=(CudaRestRootFrames const&) = delete;

    RootFrameStatus Apply(
        DeviceView<const float3> restVertices,
        DeviceView<const uint32_t> faceOffsets, size_t faceCount,
        DeviceView<const uint32_t> faceIndices,
        DeviceView<const int32_t> rootPrim,
        DeviceView<const float2> rootUV,
        DeviceView<const float3> authoredNormals,
        RootFrameNormalDomain normalDomain,
        DeviceView<const double> authoredFrames,
        cudaStream_t stream);
    RootFrameStatus Finish(cudaStream_t stream);

    RestRootFrames frames() const noexcept;
    DeviceView<const float3> origins() const noexcept;
    DeviceView<const uint8_t> valid() const noexcept;
    DeviceView<const uint8_t> dropMask() const noexcept;
    uint32_t badRootCount() const noexcept { return badRootCount_; }
    uint64_t generation() const noexcept { return generation_; }
    int deviceIndex() const noexcept { return deviceIndex_; }
    bool pending() const noexcept { return pending_; }
    cudaError_t recordUse(cudaStream_t stream);
    cudaError_t waitOn(cudaStream_t stream) const;
    char const* diagnostic() const noexcept { return diagnostic_.c_str(); }

private:
    struct Storage {
        DeviceBuffer<float3> origin, tangent, binormal, normal;
        DeviceBuffer<uint8_t> valid, drop;
        size_t curveCount = 0;
        void clear() noexcept;
        void quarantine() noexcept;
        void swap(Storage&) noexcept;
        cudaError_t recordUse(cudaStream_t);
        cudaError_t waitOn(cudaStream_t) const;
    };

    RootFrameStatus fail(RootFrameStatus, char const*);
    cudaError_t validateStream(cudaStream_t) const;
    void discardPending() noexcept;
    void quarantinePending() noexcept;

    Storage active_, pendingStorage_;
    DeviceBuffer<int> error_;
    DeviceBuffer<uint32_t> pendingBadRootCount_;
    cudaEvent_t ready_ = nullptr;
    uint32_t badRootCount_ = 0;
    uint64_t generation_ = 0;
    int deviceIndex_ = -1;
    bool pending_ = false;
    std::string diagnostic_;
};

} // namespace usdGen::gpu

#endif
