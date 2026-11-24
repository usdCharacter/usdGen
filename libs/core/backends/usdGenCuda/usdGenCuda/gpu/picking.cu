#include "picking.h"
#include "cudaCompletionLane.h"
#include "cudaCompat.h"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/version.cuh>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <limits>

namespace usdGen::gpu {
namespace {
constexpr int kBadGeometry = 1;
constexpr int kNonFinite = 2;
constexpr int kBadQuery = 3;

__device__ void Error(int *error, int code) { atomicCAS(error, 0, code); }
__device__ bool Finite(float3 p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}
__device__ int ErrorLoad(int const *error) {
    return atomicAdd(const_cast<int *>(error), 0);
}

// CUB 2.8 (CUDA 12.8) added the ArgMin overload that writes the extremum and
// its index to separate outputs. Older CUB only reduces to a
// KeyValuePair<int, float>: park that in the 8-byte index slot and split it in
// place, so FinalizePick reads the same (minimum, index) pair either way.
#if CUB_VERSION < 200800
__global__ void SplitArgMin(float *minimum, int64_t *index) {
    auto const pair = *reinterpret_cast<cub::KeyValuePair<int, float> const *>(index);
    *minimum = pair.value;
    *index = pair.key;
}
#endif

cudaError_t ArgMinDistance(void *temp, size_t &bytes, float const *distances,
                           float *minimum, int64_t *index, size_t count,
                           cudaStream_t stream) {
#if CUB_VERSION >= 200800
    return cub::DeviceReduce::ArgMin(temp, bytes, distances, minimum, index,
                                     static_cast<int64_t>(count), stream);
#else
    static_assert(sizeof(cub::KeyValuePair<int, float>) <= sizeof(int64_t),
                  "the argmin pair must fit the index slot");
    if (count > size_t(INT_MAX)) return cudaErrorInvalidValue;
    auto *pair = reinterpret_cast<cub::KeyValuePair<int, float> *>(index);
    cudaError_t const status = cub::DeviceReduce::ArgMin(
        temp, bytes, distances, pair, static_cast<int>(count), stream);
    if (status != cudaSuccess || !temp) return status;
    SplitArgMin<<<1,1,0,stream>>>(minimum, index);
    return cudaGetLastError();
#endif
}

struct DeviceQuery {
    float matrix[16];
    int width, height;
    float x, y, radius;
};
struct DevicePickResult {
    int hit;
    uint32_t curve, cv, flat;
    uint64_t stableId;
    float distance;
};

__device__ bool Project(float3 p, DeviceQuery const& q, float2 *screen) {
    // Row-vector convention: clip[j] = sum_i p[i] * M[i*4+j].
    const float in[4] = {p.x, p.y, p.z, 1.0f};
    float clip[4] = {0, 0, 0, 0};
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            clip[j] += in[i] * q.matrix[i*4+j];
    if (!isfinite(clip[0]) || !isfinite(clip[1]) || !isfinite(clip[2]) ||
        !isfinite(clip[3]) || clip[3] <= 0.0f ||
        clip[2] < -clip[3] || clip[2] > clip[3]) return false;
    const float nx = clip[0] / clip[3];
    const float ny = clip[1] / clip[3];
    if (!isfinite(nx) || !isfinite(ny)) return false;
    screen->x = (nx + 1.0f) * 0.5f * float(q.width);
    screen->y = (1.0f - ny) * 0.5f * float(q.height);
    return isfinite(screen->x) && isfinite(screen->y);
}

__global__ void ValidateGeometry(DeviceCurveGeometryView g, int *error) {
    size_t first = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x)*blockDim.x;
    for (size_t i = first; i <= g.curveCount; i += stride) {
        uint32_t o = g.curveOffsets.data[i];
        if ((i == 0 && o != 0u) ||
            (i == g.curveCount && o != g.pointCount) ||
            (i > 0 && i < g.curveCount && o <= g.curveOffsets.data[i-1u]) ||
            (i < g.curveCount && o >= g.curveOffsets.data[i+1u]))
            Error(error, kBadGeometry);
    }
    for (size_t i = first; i < g.pointCount; i += stride)
        if (!Finite(g.points.data[i])) Error(error, kNonFinite);
}

__global__ void PickCandidates(DeviceCurveGeometryView g, DeviceQuery q,
                               float *distances, int *error) {
    const size_t first = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x)*blockDim.x;
    for (size_t i = first; i < g.pointCount; i += stride) {
        distances[i] = INFINITY;
        if (ErrorLoad(error) != 0) continue;
        float2 screen;
        if (!Project(g.points.data[i], q, &screen)) continue;
        const float dx = screen.x-q.x, dy = screen.y-q.y;
        const float d = dx*dx + dy*dy;
        if (!isfinite(d)) { Error(error, kNonFinite); continue; }
        if (d <= q.radius*q.radius) distances[i] = d;
    }
}

__global__ void FinalizePick(DeviceCurveGeometryView g, DeviceQuery q,
                             float const *minimum, int64_t const *index,
                             DevicePickResult *result, int *error) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || ErrorLoad(error) != 0) return;
    result->hit = 0;
    result->curve = result->cv = result->flat = UINT32_MAX;
    result->stableId = 0;
    result->distance = INFINITY;
    if (*index < 0 || uint64_t(*index) >= g.pointCount ||
        !isfinite(*minimum) || *minimum > q.radius*q.radius) return;
    const uint32_t flat = uint32_t(*index);
    size_t lo = 0, hi = g.curveCount;
    while (lo + 1 < hi) {
        const size_t mid = (lo + hi) / 2;
        if (g.curveOffsets.data[mid] <= flat) lo = mid;
        else hi = mid;
    }
    const uint32_t begin = g.curveOffsets.data[lo];
    result->hit = 1; result->curve = uint32_t(lo); result->cv = flat-begin;
    result->flat = flat;
    result->stableId = g.stableIds.data ? g.stableIds.data[lo] : uint64_t(lo);
    result->distance = sqrtf(*minimum);
}

__global__ void InitPick(DevicePickResult *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result->hit = 0;
        result->curve = result->cv = result->flat = UINT32_MAX;
        result->stableId = 0;
        result->distance = 0.0f;
    }
}

__global__ void MarkFootprint(DeviceCurveGeometryView g, DeviceQuery q,
                              uint32_t *flags, int *error) {
    size_t first = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x)*blockDim.x;
    for (size_t i = first; i < g.pointCount; i += stride) {
        float2 screen;
        bool hit = Project(g.points.data[i], q, &screen);
        if (hit) {
            const float dx = screen.x-q.x, dy = screen.y-q.y;
            const float d = dx*dx + dy*dy;
            if (!isfinite(d)) { Error(error, kNonFinite); hit = false; }
            else hit = d <= q.radius*q.radius;
        }
        flags[i] = hit ? 1u : 0u;
}
}

__global__ void ScatterFootprint(size_t count, uint32_t const *flags,
                                 uint32_t const *prefix, int32_t *output) {
    size_t first = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x)*blockDim.x;
    for (size_t i = first; i < count; i += stride)
        if (flags[i]) output[prefix[i]] = int32_t(i);
}

__global__ void FootprintCount(size_t count, uint32_t const *flags,
                               uint32_t const *prefix, uint32_t *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *result = count ? prefix[count-1u] + uint32_t(flags[count-1u]) : 0u;
}

unsigned Blocks(size_t n) {
    size_t b = (n + 255u) / 256u;
    return unsigned(std::max<size_t>(1u, std::min<size_t>(65535u, b)));
}
PickingStatus Decode(int code) {
    return code == kBadGeometry ? PickingStatus::InvalidArgument :
           code == kNonFinite ? PickingStatus::NonFiniteInput :
           code == kBadQuery ? PickingStatus::InvalidValue : PickingStatus::CudaError;
}
} // namespace

CudaPicking::~CudaPicking() {
    int previous = -1;
    const bool got = cudaGetDevice(&previous) == cudaSuccess;
    const bool selected = deviceIndex_ >= 0 && cudaSetDevice(deviceIndex_) == cudaSuccess;
    const bool done = selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess);
    if (!selected || !done) {
        error_.quarantine(); flags_.quarantine(); prefix_.quarantine();
        pendingFootprint_.quarantine(); scanTemp_.quarantine();
        pendingPickBytes_.quarantine(); pendingCount_.quarantine();
        pickDistances_.quarantine(); pickMinimum_.quarantine(); pickIndexBytes_.quarantine();
        footprint_.quarantine(); ready_ = nullptr;
        if (selected && got && previous != deviceIndex_) cudaSetDevice(previous);
        return;
    }
    if (ready_) cudaEventDestroy(ready_);
    ready_ = nullptr;
    error_.release(); flags_.release(); prefix_.release(); pendingFootprint_.release();
    scanTemp_.release(); pendingPickBytes_.release(); pendingCount_.release();
    pickDistances_.release(); pickMinimum_.release(); pickIndexBytes_.release();
    footprint_.release();
    if (got && previous != deviceIndex_) cudaSetDevice(previous);
}

PickingStatus CudaPicking::fail(PickingStatus status, const char *message) {
    diagnostic_ = message; return status;
}

PickingStatus CudaPicking::validateGeometry(DeviceCurveGeometryView g) const {
    if (g.curveCount > size_t(UINT32_MAX) || g.pointCount > size_t(INT32_MAX) ||
        g.curveCount == std::numeric_limits<size_t>::max() ||
        (g.curveCount == 0 && g.pointCount != 0) ||
        g.points.size != g.pointCount || g.curveOffsets.size != g.curveCount + 1u ||
        !g.curveOffsets.data || (g.pointCount && !g.points.data) ||
        (g.stableIds.data && g.stableIds.size != g.curveCount) ||
        (!g.stableIds.data && g.stableIds.size != 0)) return PickingStatus::InvalidArgument;
    auto isDevice = [](const void *pointer) {
        cudaPointerAttributes attributes{};
        return cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
               attributes.type == cudaMemoryTypeDevice;
    };
    if (!isDevice(g.curveOffsets.data) ||
        (g.pointCount && !isDevice(g.points.data)) ||
        (g.stableIds.data && g.curveCount && !isDevice(g.stableIds.data)))
        return PickingStatus::InvalidArgument;
    return PickingStatus::Ok;
}

PickingStatus CudaPicking::validateQuery(PickQuery const& q) const {
    if (q.width <= 0 || q.height <= 0 || !std::isfinite(q.x) || !std::isfinite(q.y) ||
        !std::isfinite(q.radiusPx) || q.radiusPx < 0 ||
        q.radiusPx > std::sqrt(std::numeric_limits<float>::max()))
        return PickingStatus::InvalidValue;
    for (float value : q.viewProj)
        if (!std::isfinite(value)) return PickingStatus::NonFiniteInput;
    return PickingStatus::Ok;
}

PickingStatus CudaPicking::begin(DeviceCurveGeometryView g, PickQuery query,
                                 bool footprint, cudaStream_t stream) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return fail(PickingStatus::CudaError, "cannot identify CUDA device");
    if (deviceIndex_ >= 0 && deviceIndex_ != current) return fail(PickingStatus::InvalidArgument, "picking device mismatch");
    if (stream) {
        int sd = -1;
        if (cudaStreamGetDevice(stream, &sd) != cudaSuccess) return fail(PickingStatus::CudaError, "cannot identify picking stream");
        if (sd != current) return fail(PickingStatus::InvalidArgument, "picking stream device mismatch");
    }
    auto onCurrentDevice = [current](const void *pointer) {
        if (!pointer) return true;
        cudaPointerAttributes attributes{};
        return cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
               attributes.type == cudaMemoryTypeDevice && attributes.device == current;
    };
    if (!onCurrentDevice(g.curveOffsets.data) ||
        (g.pointCount && !onCurrentDevice(g.points.data)) ||
        (g.stableIds.data && g.curveCount && !onCurrentDevice(g.stableIds.data)))
        return fail(PickingStatus::InvalidArgument, "picking view belongs to another device");
    if (deviceIndex_ < 0) deviceIndex_ = current;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return fail(PickingStatus::CudaError, "picking event creation failed");
    if (error_.reset(1) != cudaSuccess) return fail(PickingStatus::CudaError, "picking error allocation failed");
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return abortBegin(stream, PickingStatus::CudaError, "picking error reset failed");
    if (!footprint) {
        if (pendingPickBytes_.reset(sizeof(DevicePickResult)) != cudaSuccess ||
            pickDistances_.reset(g.pointCount) != cudaSuccess ||
            pickMinimum_.reset(1) != cudaSuccess ||
            pickIndexBytes_.reset(sizeof(int64_t)) != cudaSuccess)
            return abortBegin(stream, PickingStatus::CudaError, "pick result allocation failed");
        if (cudaMemsetAsync(pendingPickBytes_.data(), 0, sizeof(DevicePickResult), stream) != cudaSuccess)
            return abortBegin(stream, PickingStatus::CudaError, "pick result reset failed");
        InitPick<<<1,1,0,stream>>>(reinterpret_cast<DevicePickResult*>(pendingPickBytes_.data()));
        if (cudaGetLastError() != cudaSuccess)
            return abortBegin(stream, PickingStatus::CudaError, "pick result initialization failed");
    }
    const size_t n = g.pointCount;
    ValidateGeometry<<<Blocks(std::max(g.curveCount, n)),256,0,stream>>>(g,error_.data());
    if (cudaGetLastError() != cudaSuccess)
        return abortBegin(stream, PickingStatus::CudaError, "picking validation launch failed");
    DeviceQuery dq{};
    std::memcpy(dq.matrix, query.viewProj, sizeof(dq.matrix));
    dq.width=query.width; dq.height=query.height; dq.x=query.x; dq.y=query.y; dq.radius=query.radiusPx;
    if (footprint) {
        if (flags_.reset(n) != cudaSuccess || prefix_.reset(n) != cudaSuccess ||
            pendingFootprint_.reset(n) != cudaSuccess || pendingCount_.reset(1) != cudaSuccess)
            return abortBegin(stream, PickingStatus::CudaError, "footprint allocation failed");
        if (n) {
            MarkFootprint<<<Blocks(n),256,0,stream>>>(g,dq,flags_.data(),error_.data());
            if (cudaGetLastError() != cudaSuccess)
                return abortBegin(stream, PickingStatus::CudaError, "footprint mark failed");
            size_t bytes = 0;
            if (cub::DeviceScan::ExclusiveSum(nullptr, bytes, flags_.data(), prefix_.data(), n, stream) != cudaSuccess ||
                scanTemp_.reset(bytes) != cudaSuccess ||
                cub::DeviceScan::ExclusiveSum(scanTemp_.data(), bytes, flags_.data(), prefix_.data(), n, stream) != cudaSuccess)
                return abortBegin(stream, PickingStatus::CudaError, "footprint scan failed");
            ScatterFootprint<<<Blocks(n),256,0,stream>>>(n,flags_.data(),prefix_.data(),pendingFootprint_.data());
            if (cudaGetLastError() != cudaSuccess)
                return abortBegin(stream, PickingStatus::CudaError, "footprint scatter failed");
        } else if (cudaMemsetAsync(pendingCount_.data(), 0, sizeof(uint32_t), stream) != cudaSuccess)
            return abortBegin(stream, PickingStatus::CudaError, "empty footprint reset failed");
        if (n) {
            FootprintCount<<<1,1,0,stream>>>(n,flags_.data(),prefix_.data(),pendingCount_.data());
            if (cudaGetLastError() != cudaSuccess)
                return abortBegin(stream, PickingStatus::CudaError, "footprint count failed");
        }
    } else {
        if (n) {
            PickCandidates<<<Blocks(n),256,0,stream>>>(g,dq,pickDistances_.data(),error_.data());
            if (cudaGetLastError() != cudaSuccess)
                return abortBegin(stream, PickingStatus::CudaError, "pick candidate launch failed");
            size_t bytes = 0;
            if (ArgMinDistance(nullptr, bytes, pickDistances_.data(), pickMinimum_.data(),
                               reinterpret_cast<int64_t*>(pickIndexBytes_.data()), n, stream) != cudaSuccess ||
                scanTemp_.reset(bytes) != cudaSuccess ||
                ArgMinDistance(scanTemp_.data(), bytes, pickDistances_.data(), pickMinimum_.data(),
                               reinterpret_cast<int64_t*>(pickIndexBytes_.data()), n, stream) != cudaSuccess)
                return abortBegin(stream, PickingStatus::CudaError, "pick reduction failed");
            FinalizePick<<<1,1,0,stream>>>(g,dq,pickMinimum_.data(),
                reinterpret_cast<int64_t*>(pickIndexBytes_.data()),
                reinterpret_cast<DevicePickResult*>(pendingPickBytes_.data()),error_.data());
        }
        if (cudaGetLastError() != cudaSuccess)
            return abortBegin(stream, PickingStatus::CudaError, "pick finalize launch failed");
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess)
        return abortBegin(stream, PickingStatus::CudaError, "picking event record failed");
    pendingPoints_=n; pendingIsFootprint_=footprint; pending_=true;
    return PickingStatus::Ok;
}

PickingStatus CudaPicking::abortBegin(cudaStream_t stream, PickingStatus status,
                                      const char* message) {
    // A launch/allocation failure after async work has been queued cannot use
    // the previous ready event as a fence. Synchronize this operation's stream
    // before allowing buffers to be reused; quarantine if that proof fails.
    if (CudaFenceStream(stream) != cudaSuccess) {
        error_.quarantine(); flags_.quarantine(); prefix_.quarantine();
        pendingFootprint_.quarantine(); scanTemp_.quarantine();
        pendingPickBytes_.quarantine(); pendingCount_.quarantine();
        pickDistances_.quarantine(); pickMinimum_.quarantine(); pickIndexBytes_.quarantine();
        ready_ = nullptr;
    }
    pending_ = false;
    return fail(status, message);
}

PickingStatus CudaPicking::ApplyPick(DeviceCurveGeometryView g, PickQuery q, cudaStream_t stream) {
    if (pending_)
        return fail(PickingStatus::InvalidArgument,
                    "Finish is required before another pick");
    PickingStatus status = validateGeometry(g);
    if (status != PickingStatus::Ok)
        return fail(status, "invalid picking geometry");
    status = validateQuery(q);
    if (status != PickingStatus::Ok)
        return fail(status, "invalid pick query");
    return begin(g, q, false, stream);
}
PickingStatus CudaPicking::ApplyFootprint(DeviceCurveGeometryView g, PickQuery q, cudaStream_t stream) {
    if (pending_)
        return fail(PickingStatus::InvalidArgument,
                    "Finish is required before another pick");
    PickingStatus status = validateGeometry(g);
    if (status != PickingStatus::Ok)
        return fail(status, "invalid picking geometry");
    status = validateQuery(q);
    if (status != PickingStatus::Ok)
        return fail(status, "invalid footprint query");
    return begin(g, q, true, stream);
}

PickingStatus CudaPicking::Finish(cudaStream_t stream) {
    if (!pending_)
        return PickingStatus::NoPendingOperation;

    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_)
        return PickingStatus::InvalidArgument;
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
            return PickingStatus::CudaError;
        if (streamDevice != deviceIndex_)
            return PickingStatus::InvalidArgument;
    }

    int code = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess)
        return fail(PickingStatus::CudaError, "picking diagnostic readback failed");
    // Fence the producer before using stack-backed scalar destinations. The
    // copies below are synchronous and therefore cannot outlive Finish.
    if (CudaFenceStream(stream) != cudaSuccess)
        return abortBegin(stream, PickingStatus::CudaError, "picking synchronization failed");
    if (cudaMemcpy(&code, error_.data(), sizeof(code), cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail(PickingStatus::CudaError, "picking diagnostic readback failed");

    DevicePickResult deviceResult{};
    uint32_t count = 0;
    if (!pendingIsFootprint_ &&
        cudaMemcpy(&deviceResult, pendingPickBytes_.data(), sizeof(deviceResult),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail(PickingStatus::CudaError, "pick result readback failed");
    if (pendingIsFootprint_ &&
        cudaMemcpy(&count, pendingCount_.data(), sizeof(count),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail(PickingStatus::CudaError, "footprint count readback failed");
    if (code) {
        pending_ = false;
        return fail(Decode(code), "device picking validation failed");
    }

    if (pendingIsFootprint_) {
        if (count > pendingPoints_) {
            pending_ = false;
            return fail(PickingStatus::CudaError, "invalid footprint count");
        }
        // Move the complete pending allocation into the published slot. This
        // keeps the previous successful footprint intact until all validation
        // and count readback has succeeded, and avoids a publication copy.
        footprint_ = std::move(pendingFootprint_);
        footprintCount_ = count;
    } else {
        PickResult next{};
        next.hit = deviceResult.hit != 0;
        next.curve = deviceResult.curve;
        next.cv = deviceResult.cv;
        next.flatIndex = deviceResult.flat;
        next.stableId = deviceResult.stableId;
        next.distancePx = deviceResult.distance;
        result_ = next;
    }
    pending_ = false;
    pendingPoints_ = 0;
    return PickingStatus::Ok;
}
} // namespace usdGen::gpu
