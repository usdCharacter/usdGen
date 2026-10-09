#include "surfaceBinding.h"
#include "cudaCompat.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <limits>

namespace usdGen { namespace gpu {
namespace {
std::atomic<bool> s_failFreshProofAllocation{false}, s_failFreshProofEnqueue{false},
    s_failFreshProofAfterStatus{false};
std::atomic<bool> s_failFreshResolvePreflight{false}, s_failFreshResolveCommit{false};
std::atomic<uint64_t> s_freshAcceptAttempts{0}, s_freshRollbackAttempts{0};

constexpr int kBadTopology = 1 << 0;
constexpr int kNonFinite = 1 << 1;
constexpr int kBadRoot = 1 << 2;

__device__ inline bool finite3(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__device__ inline void mark(int* error, int bit) { atomicOr(error, bit); }

__global__ void ValidateSurface(DeviceView<const float3> vertices,
                                DeviceView<const uint32_t> offsets,
                                size_t faceCount,
                                DeviceView<const uint32_t> indices,
                                int* error) {
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vertices.size; i += stride) {
        if (!finite3(vertices.data[i])) mark(error, kNonFinite);
    }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < indices.size; i += stride) {
        if (indices.data[i] >= vertices.size) mark(error, kBadTopology);
    }
    for (size_t face = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         face < faceCount; face += stride) {
        uint32_t begin = offsets.data[face];
        uint32_t end = offsets.data[face + 1];
        size_t count = end >= begin ? size_t(end - begin) : 0;
        if (end < begin || size_t(end) > indices.size ||
            (count != 3 && count != 4))
            mark(error, kBadTopology);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        (offsets.data[0] != 0 || size_t(offsets.data[faceCount]) != indices.size))
        mark(error, kBadTopology);
}

__global__ void InitializeNearest(DeviceView<double> nearest) {
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nearest.size; i += stride)
        nearest.data[i] = INFINITY;
}

__global__ void SelectFirst(DeviceView<uint32_t> samples,
                            DeviceView<uint32_t> actual, int* error) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (atomicAdd(error, 0) != 0) return;
        if (samples.size) {
            samples.data[0] = 0;
            actual.data[0] = 1;
        } else {
            actual.data[0] = 0;
        }
    }
}

__global__ void UpdateNearest(DeviceView<const float3> vertices,
                              DeviceView<const uint32_t> samples,
                              size_t selected, DeviceView<double> nearest,
                              DeviceView<uint32_t> actual, int* error) {
    if (atomicAdd(error, 0) != 0 || atomicAdd(actual.data, 0) != selected)
        return;
    uint32_t selectedIndex = samples.data[selected - 1];
    float3 q = vertices.data[selectedIndex];
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vertices.size; i += stride) {
        float3 p = vertices.data[i];
        double dx = double(p.x) - double(q.x);
        double dy = double(p.y) - double(q.y);
        double dz = double(p.z) - double(q.z);
        double distance = dx * dx + dy * dy + dz * dz;
        if (distance < nearest.data[i]) nearest.data[i] = distance;
    }
}

__global__ void SelectNext(DeviceView<const double> nearest,
                           DeviceView<uint32_t> samples,
                           size_t selected, DeviceView<uint32_t> actual,
                           int* error) {
    if (blockIdx.x != 0 || atomicAdd(error, 0) != 0 ||
        atomicAdd(actual.data, 0) != selected)
        return;
    double bestDistance = -1.0;
    uint32_t best = UINT32_MAX;
    bool found = false;
    for (size_t candidate = threadIdx.x; candidate < nearest.size;
         candidate += blockDim.x) {
        double distance = nearest.data[candidate];
        uint32_t index = static_cast<uint32_t>(candidate);
        if (!found || distance > bestDistance ||
            (distance == bestDistance && index < best)) {
            bestDistance = distance;
            best = index;
            found = true;
        }
    }
    __shared__ double distances[256];
    __shared__ uint32_t indices[256];
    __shared__ int valid[256];
    distances[threadIdx.x] = bestDistance;
    indices[threadIdx.x] = best;
    valid[threadIdx.x] = found ? 1 : 0;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride && valid[threadIdx.x + stride]) {
            if (!valid[threadIdx.x] ||
                distances[threadIdx.x + stride] > distances[threadIdx.x] ||
                (distances[threadIdx.x + stride] == distances[threadIdx.x] &&
                 indices[threadIdx.x + stride] < indices[threadIdx.x])) {
                distances[threadIdx.x] = distances[threadIdx.x + stride];
                indices[threadIdx.x] = indices[threadIdx.x + stride];
                valid[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0 || !valid[0]) return;
    bestDistance = distances[0];
    best = indices[0];
    // A zero nearest distance means every remaining point is coincident with
    // an existing sample.  Do not emit duplicate RBF centers.
    if (!(bestDistance > 0.0)) return;
    samples.data[selected] = static_cast<uint32_t>(best);
    actual.data[0] = static_cast<uint32_t>(selected + 1);
}

__global__ void GatherSamples(DeviceView<const float3> vertices,
                              DeviceView<const uint32_t> indices,
                              DeviceView<float3> output, int* error,
                              const uint32_t* actualCount = nullptr) {
    if (atomicAdd(error, 0) != 0) return;
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < output.size; i += stride) {
        // Deduplication can stop before the requested capacity. Those index
        // slots were never written and must not be read, even transiently.
        if (actualCount && i >= *actualCount) {
            output.data[i] = make_float3(0, 0, 0);
            continue;
        }
        uint32_t index = indices.data[i];
        if (size_t(index) >= vertices.size) {
            mark(error, kBadTopology);
            continue;
        }
        float3 value = vertices.data[index];
        if (!finite3(value)) mark(error, kNonFinite);
        else output.data[i] = value;
    }
}

__device__ float3 RootPosition(DeviceView<const float3> vertices,
                               DeviceView<const uint32_t> offsets,
                               DeviceView<const uint32_t> indices,
                               int32_t face, float2 uv) {
    uint32_t begin = offsets.data[face];
    uint32_t end = offsets.data[size_t(face) + 1];
    size_t count = size_t(end - begin);
    float3 a = vertices.data[indices.data[begin]];
    float3 b = vertices.data[indices.data[begin + 1]];
    float3 c = vertices.data[indices.data[begin + 2]];
    if (count == 3) {
        float w = 1.0f - uv.x - uv.y;
        return make_float3(a.x * w + b.x * uv.x + c.x * uv.y,
                           a.y * w + b.y * uv.x + c.y * uv.y,
                           a.z * w + b.z * uv.x + c.z * uv.y);
    }
    float3 d = vertices.data[indices.data[begin + 3]];
    float u0 = 1.0f - uv.x, v0 = 1.0f - uv.y;
    return make_float3(a.x * u0 * v0 + b.x * uv.x * v0 +
                           c.x * uv.x * uv.y + d.x * u0 * uv.y,
                       a.y * u0 * v0 + b.y * uv.x * v0 +
                           c.y * uv.x * uv.y + d.y * u0 * uv.y,
                       a.z * u0 * v0 + b.z * uv.x * v0 +
                           c.z * uv.x * uv.y + d.z * u0 * uv.y);
}

// Single-launch update validation+gather. One launch replaces the
// ValidateSurface + GatherSamples + ValidateRoots + GatherRoots sequence
// in Update: four grid-teardown round-trips become one, and the roots
// pass reads skinPrim/uv/offsets once instead of twice. Bit-identity:
// every mark is an atomicOr of a deterministic per-element predicate, so
// the final flag is the same union whatever order blocks run in; when
// the flag is zero no whole-kernel early-out could have triggered, so
// every output element is written with the same formula. Roots are
// validated per element before their own dereference (no cross-block
// early-out exists inside one launch), so invalid roots still never
// dereference, exactly as before.
__global__ void ValidateAndGatherUpdate(DeviceView<const float3> vertices,
                                        DeviceView<const uint32_t> sampleIndices,
                                        DeviceView<float3> samplesOut,
                                        DeviceView<const int32_t> skinPrim,
                                        DeviceView<const float2> uv,
                                        DeviceView<const uint32_t> offsets,
                                        size_t faceCount,
                                        DeviceView<const uint32_t> indices,
                                        DeviceView<float3> rootsOut, int* error) {
    size_t stride = size_t(blockDim.x) * gridDim.x;
    size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (size_t i = lane; i < vertices.size; i += stride) {
        if (!finite3(vertices.data[i])) mark(error, kNonFinite);
    }
    for (size_t i = lane; i < indices.size; i += stride) {
        if (indices.data[i] >= vertices.size) mark(error, kBadTopology);
    }
    for (size_t face = lane; face < faceCount; face += stride) {
        uint32_t begin = offsets.data[face];
        uint32_t end = offsets.data[face + 1];
        size_t count = end >= begin ? size_t(end - begin) : 0;
        if (end < begin || size_t(end) > indices.size ||
            (count != 3 && count != 4))
            mark(error, kBadTopology);
    }
    if (lane == 0 &&
        (offsets.data[0] != 0 || size_t(offsets.data[faceCount]) != indices.size))
        mark(error, kBadTopology);
    for (size_t i = lane; i < samplesOut.size; i += stride) {
        uint32_t index = sampleIndices.data[i];
        if (size_t(index) >= vertices.size) {
            mark(error, kBadTopology);
            continue;
        }
        float3 value = vertices.data[index];
        if (!finite3(value)) mark(error, kNonFinite);
        else samplesOut.data[i] = value;
    }
    for (size_t root = lane; root < skinPrim.size; root += stride) {
        int32_t face = skinPrim.data[root];
        float2 value = uv.data[root];
        if (face < 0 || size_t(face) >= faceCount ||
            !isfinite(value.x) || !isfinite(value.y)) {
            mark(error, kBadRoot);
            continue;
        }
        uint32_t begin = offsets.data[face];
        uint32_t end = offsets.data[size_t(face) + 1];
        size_t count = end >= begin ? size_t(end - begin) : 0;
        if ((count != 3 && count != 4) || end < begin ||
            size_t(end) > indices.size || value.x < 0.0f || value.x > 1.0f ||
            value.y < 0.0f || value.y > 1.0f ||
            (count == 3 && value.x + value.y > 1.0f)) {
            mark(error, kBadRoot);
            continue;
        }
        float3 gathered = RootPosition(vertices, offsets, indices, face, value);
        if (!finite3(gathered)) mark(error, kNonFinite);
        else rootsOut.data[root] = gathered;
    }
}

unsigned Blocks(size_t count) {
    size_t needed = (count + 255) / 256;
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(needed, 65535)));
}

SurfaceBindingStatus Decode(int code) {
    if (code & kBadRoot) return SurfaceBindingStatus::InvalidRootBinding;
    if (code & kBadTopology) return SurfaceBindingStatus::InvalidTopology;
    if (code & kNonFinite) return SurfaceBindingStatus::NonFiniteInput;
    return SurfaceBindingStatus::Ok;
}
bool FreshDeviceInput(void const* pointer, size_t count) {
    if (!count) return true;
    if (!pointer) return false;
    cudaPointerAttributes attributes{};
    int device = -1;
    return cudaGetDevice(&device) == cudaSuccess &&
        cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
        attributes.type == cudaMemoryTypeDevice && attributes.device == device;
}
}

CudaSurfaceBinding::~CudaSurfaceBinding() {
    // Fresh work has no parent terminal proof at destruction.  This branch is
    // intentionally before every CUDA call, including cudaGetDevice.
    if (HasUnprovenFreshWork()) {
        error_.quarantine(); sampleIndices_.quarantine(); pendingSampleIndices_.quarantine();
        pendingNearest_.quarantine(); pendingActualSampleCount_.quarantine();
        restSamples_.quarantine(); pendingRestSamples_.quarantine();
        currentSamples_.quarantine(); pendingCurrentSamples_.quarantine();
        rootTargets_.quarantine(); pendingRootTargets_.quarantine();
        faceOffsets_.quarantine(); pendingFaceOffsets_.quarantine(); faceIndices_.quarantine(); pendingFaceIndices_.quarantine();
        retiredSampleIndices_.quarantine(); retiredFaceOffsets_.quarantine(); retiredFaceIndices_.quarantine();
        retiredRestSamples_.quarantine(); retiredCurrentSamples_.quarantine(); retiredRootTargets_.quarantine();
        freshCodePermit_.Abandon(); freshActualPermit_.Abandon(); freshCode_ = nullptr; freshActual_ = nullptr; ready_ = nullptr;
        return;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    const bool selected = deviceIndex_ >= 0 &&
        cudaSetDevice(deviceIndex_) == cudaSuccess;
    auto quarantine = [&]() {
        // DeviceBuffer cannot safely free an allocation when its CUDA context
        // is lost or completion cannot be established.  Null every handle so
        // the member destructors do not attempt a foreign-context free.
        error_.quarantine();
        sampleIndices_.quarantine();
        pendingSampleIndices_.quarantine();
        pendingNearest_.quarantine();
        pendingActualSampleCount_.quarantine();
        restSamples_.quarantine();
        pendingRestSamples_.quarantine();
        currentSamples_.quarantine();
        pendingCurrentSamples_.quarantine();
        rootTargets_.quarantine();
        pendingRootTargets_.quarantine();
        faceOffsets_.quarantine();
        pendingFaceOffsets_.quarantine();
        faceIndices_.quarantine();
        pendingFaceIndices_.quarantine();
        retiredSampleIndices_.quarantine(); retiredFaceOffsets_.quarantine(); retiredFaceIndices_.quarantine();
        retiredRestSamples_.quarantine(); retiredCurrentSamples_.quarantine(); retiredRootTargets_.quarantine();
        freshCodePermit_.Abandon(); freshActualPermit_.Abandon();
        freshCode_ = nullptr; freshActual_ = nullptr;
        ready_ = nullptr;
    };
    if (!selected) {
        quarantine();
        return;
    }
    // A fresh caller has no terminal proof.  Do not turn destruction into a
    // synchronizing cleanup or free pageable/pinned staging behind its back.
    if (HasUnprovenFreshWork()) {
        quarantine();
        if (previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
        return;
    }
    if (ready_) {
        if (!latestFresh_ && cudaEventSynchronize(ready_) != cudaSuccess) {
            quarantine();
            if (previous >= 0 && previous != deviceIndex_)
                cudaSetDevice(previous);
            return;
        }
        cudaEventDestroy(ready_);
        ready_ = nullptr;
    }
    error_.release();
    sampleIndices_.release();
    pendingSampleIndices_.release();
    pendingNearest_.release();
    pendingActualSampleCount_.release();
    restSamples_.release();
    pendingRestSamples_.release();
    currentSamples_.release();
    pendingCurrentSamples_.release();
    rootTargets_.release();
    pendingRootTargets_.release();
    faceOffsets_.release();
    pendingFaceOffsets_.release();
    faceIndices_.release();
    pendingFaceIndices_.release();
    retiredSampleIndices_.release(); retiredFaceOffsets_.release(); retiredFaceIndices_.release();
    retiredRestSamples_.release(); retiredCurrentSamples_.release(); retiredRootTargets_.release();
    discardFreshProof(false);
    if (selected && previous >= 0 && previous != deviceIndex_)
        cudaSetDevice(previous);
}

SurfaceBindingStatus CudaSurfaceBinding::fail(SurfaceBindingStatus status,
                                              const char* message) {
    diagnostic_ = message;
    return status;
}

SurfaceBindingStatus CudaSurfaceBinding::validateCommon(
    DeviceView<const float3> vertices, DeviceView<const uint32_t> offsets,
    size_t faces, DeviceView<const uint32_t> indices) const {
    if ((vertices.size && !vertices.data) ||
        (offsets.size && !offsets.data) || (indices.size && !indices.data))
        return SurfaceBindingStatus::InvalidArgument;
    if (faces == std::numeric_limits<size_t>::max() ||
        offsets.size != faces + 1 ||
        faces > std::numeric_limits<uint32_t>::max() ||
        vertices.size > std::numeric_limits<uint32_t>::max() ||
        indices.size > std::numeric_limits<uint32_t>::max())
        return SurfaceBindingStatus::InvalidArgument;
    return SurfaceBindingStatus::Ok;
}

SurfaceBindingStatus CudaSurfaceBinding::validateStream(cudaStream_t stream) const {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
        return SurfaceBindingStatus::CudaError;
    if (deviceIndex_ >= 0 && current != deviceIndex_)
        return SurfaceBindingStatus::InvalidArgument;
    if (!stream) return SurfaceBindingStatus::Ok;
    int streamDevice = -1;
    if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
        return SurfaceBindingStatus::CudaError;
    if (deviceIndex_ >= 0 && streamDevice != deviceIndex_)
        return SurfaceBindingStatus::InvalidArgument;
    return streamDevice == current ? SurfaceBindingStatus::Ok
                                   : SurfaceBindingStatus::InvalidArgument;
}

void CudaSurfaceBinding::discardPending() {
    pendingSampleIndices_.reset(0);
    pendingNearest_.reset(0);
    pendingActualSampleCount_.reset(0);
    pendingRestSamples_.reset(0);
    pendingCurrentSamples_.reset(0);
    pendingRootTargets_.reset(0);
    pendingFaceOffsets_.reset(0);
    pendingFaceIndices_.reset(0);
    pendingVertexCount_ = pendingFaceCount_ = pendingIndexCount_ = 0;
    pendingSampleCount_ = pendingRootCount_ = 0;
    pending_ = false;
    pendingBind_ = false;
}

void CudaSurfaceBinding::discardFreshProof(bool abandon) noexcept {
    if (freshCode_) {
        if (abandon) freshCodePermit_.Abandon();
        else if (cudaFreeHost(freshCode_) == cudaSuccess) freshCodePermit_.Release();
        else freshCodePermit_.Abandon();
        freshCode_ = nullptr;
    }
    if (freshActual_) {
        if (abandon) freshActualPermit_.Abandon();
        else if (cudaFreeHost(freshActual_) == cudaSuccess) freshActualPermit_.Release();
        else freshActualPermit_.Abandon();
        freshActual_ = nullptr;
    }
    freshMode_ = freshPending_ = freshUnproven_ = false;
}

SurfaceBindingStatus CudaSurfaceBinding::beginFreshProof(cudaStream_t stream, bool bind) {
    if (s_failFreshProofEnqueue.exchange(false, std::memory_order_acq_rel)) {
        freshUnproven_ = true;
        return fail(SurfaceBindingStatus::CudaError, "injected surface fresh proof enqueue failure");
    }
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(freshCode_, error_.data(), sizeof(*freshCode_), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        freshUnproven_ = true;
        return fail(SurfaceBindingStatus::CudaError, "surface fresh proof enqueue failed");
    }
    if (s_failFreshProofAfterStatus.exchange(false, std::memory_order_acq_rel) ||
        (bind && cudaMemcpyAsync(freshActual_, pendingActualSampleCount_.data(), sizeof(*freshActual_), cudaMemcpyDeviceToHost, stream) != cudaSuccess)) {
        freshUnproven_ = true;
        return fail(SurfaceBindingStatus::CudaError, "surface fresh proof actual-count enqueue failed");
    }
    freshUnproven_ = false; // parent still owns proof; commit remains forbidden until it arrives.
    return SurfaceBindingStatus::Ok;
}

SurfaceBindingStatus CudaSurfaceBinding::BeginFreshBind(
    DeviceView<const float3> vertices, DeviceView<const uint32_t> offsets, size_t faces,
    DeviceView<const uint32_t> indices, size_t budget, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!stream || cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Bind rejects captured or null streams");
    if (!FreshDeviceInput(vertices.data, vertices.size) || !FreshDeviceInput(offsets.data, offsets.size) ||
        !FreshDeviceInput(indices.data, indices.size))
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Bind requires views on the selected CUDA device");
    if (pending_ || freshPending_ || freshMode_ || freshUpdatePendingAcceptance_) return fail(SurfaceBindingStatus::InvalidArgument, "fresh Bind requires no pending operation");
    if (validateCommon(vertices, offsets, faces, indices) != SurfaceBindingStatus::Ok ||
        validateStream(stream) != SurfaceBindingStatus::Ok)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Bind preflight failed");
    discardPending();
    retiredSampleIndices_.release(); retiredRestSamples_.release(); retiredCurrentSamples_.release();
    retiredRootTargets_.release(); retiredRootCount_ = 0; retiredFaceOffsets_.release(); retiredFaceIndices_.release();
    // The previous pose's proof packets are pooled, not freed: their words
    // are fully overwritten below (proof D2Hs), so the new candidate adopts
    // them instead of freeing and re-allocating every pose. Only the flags
    // reset here; later failures still discard (free) as before, and the
    // forced-allocation seam still fires first, as before. Permits attach
    // to the storage, not the operation, so a reused packet keeps its
    // original permit and consumes nothing further.
    freshMode_ = freshPending_ = freshUnproven_ = false;
    if (s_failFreshProofAllocation.exchange(false, std::memory_order_acq_rel))
        return fail(SurfaceBindingStatus::CudaError, "surface fresh proof allocation failed");
    if (!freshCode_) {
        auto codePermit = TryReserveCudaExecutionBytes(sizeof(int), UsdGenExecutionResourceKind::Cache, reservation);
        if (!codePermit || cudaHostAlloc(reinterpret_cast<void**>(&freshCode_), sizeof(int), cudaHostAllocDefault) != cudaSuccess)
            return fail(SurfaceBindingStatus::CudaError, "surface fresh proof allocation failed");
        freshCodePermit_ = std::move(*codePermit);
    }
    if (!freshActual_) {
        auto actualPermit = TryReserveCudaExecutionBytes(sizeof(uint32_t), UsdGenExecutionResourceKind::Cache, reservation);
        if (!actualPermit || cudaHostAlloc(reinterpret_cast<void**>(&freshActual_), sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess) {
            discardFreshProof(false);
            return fail(SurfaceBindingStatus::CudaError, "surface fresh proof allocation failed");
        }
        freshActualPermit_ = std::move(*actualPermit);
    }
    freshReservation_ = reservation;
    freshMode_ = true; freshLaunching_ = true; freshPending_ = true; freshUnproven_ = true;
    auto status = Bind(vertices, offsets, faces, indices, budget, stream);
    freshLaunching_ = false; freshReservation_ = nullptr;
    if (status != SurfaceBindingStatus::Ok) {
        if (!pending_) discardFreshProof(false);
        return status;
    }
    return beginFreshProof(stream, true);
}

SurfaceBindingStatus CudaSurfaceBinding::BeginFreshUpdate(
    DeviceView<const float3> vertices, DeviceView<const int32_t> roots,
    DeviceView<const float2> uv, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!stream || cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Update rejects captured or null streams");
    if (!FreshDeviceInput(vertices.data, vertices.size) || !FreshDeviceInput(roots.data, roots.size) ||
        !FreshDeviceInput(uv.data, uv.size))
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Update requires views on the selected CUDA device");
    if (pending_ || freshPending_ || freshMode_ || freshUpdatePendingAcceptance_ || !bound_) return fail(SurfaceBindingStatus::InvalidArgument, "fresh Update requires no pending operation");
    if (validateStream(stream) != SurfaceBindingStatus::Ok || vertices.size != vertexCount_ || roots.size != uv.size)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Update preflight failed");
    discardPending();
    // The retired publication is stale garbage here (the prologue rejects
    // any acceptance-pending update), and Update fully overwrites both
    // buffers below (GatherSamples/GatherRoots cover every element), so a
    // size-matching retired buffer is adopted as the new pending storage
    // instead of freed and re-malloc'd. reset() still re-sizes on groom
    // or budget changes, so rebinds and quarantines behave as before.
    if (retiredCurrentSamples_.size() == sampleCount_)
        pendingCurrentSamples_ = std::move(retiredCurrentSamples_);
    else
        retiredCurrentSamples_.release();
    if (retiredRootTargets_.size() == roots.size)
        pendingRootTargets_ = std::move(retiredRootTargets_);
    else
        retiredRootTargets_.release();
    retiredRootCount_ = 0;
    // Same proof-packet pooling as BeginFreshBind (see above): only the
    // flags reset here, the seam fires first, and storage is adopted.
    freshMode_ = freshPending_ = freshUnproven_ = false;
    if (s_failFreshProofAllocation.exchange(false, std::memory_order_acq_rel))
        return fail(SurfaceBindingStatus::CudaError, "surface fresh proof allocation failed");
    if (!freshCode_) {
        auto codePermit = TryReserveCudaExecutionBytes(sizeof(int), UsdGenExecutionResourceKind::Cache, reservation);
        if (!codePermit || cudaHostAlloc(reinterpret_cast<void**>(&freshCode_), sizeof(int), cudaHostAllocDefault) != cudaSuccess)
            return fail(SurfaceBindingStatus::CudaError, "surface fresh proof allocation failed");
        freshCodePermit_ = std::move(*codePermit);
    }
    freshReservation_ = reservation;
    freshMode_ = true; freshLaunching_ = true; freshPending_ = true; freshUnproven_ = true;
    auto status = Update(vertices, roots, uv, stream);
    freshLaunching_ = false; freshReservation_ = nullptr;
    if (status != SurfaceBindingStatus::Ok) {
        if (!pending_) discardFreshProof(false);
        return status;
    }
    return beginFreshProof(stream, false);
}

void CudaSurfaceBinding::AbandonFresh() noexcept {
    if (!freshPending_ && !pending_) return;
    discardFreshProof(true);
    pendingSampleIndices_.quarantine(); pendingNearest_.quarantine(); pendingActualSampleCount_.quarantine();
    pendingRestSamples_.quarantine(); pendingCurrentSamples_.quarantine(); pendingRootTargets_.quarantine();
    pendingFaceOffsets_.quarantine(); pendingFaceIndices_.quarantine();
    pending_ = pendingBind_ = false;
    // Retain the abandonment state through destruction; all pending handles
    // are already quarantined and their charged permits remain abandoned.
    freshPending_ = true; freshUnproven_ = true;
}

SurfaceBindingStatus CudaSurfaceBinding::Bind(
    DeviceView<const float3> restVertices,
    DeviceView<const uint32_t> faceOffsets, size_t faceCount,
    DeviceView<const uint32_t> faceIndices, size_t sampleBudget,
    cudaStream_t stream) {
    if ((freshMode_ || freshPending_) && !freshLaunching_) return fail(SurfaceBindingStatus::InvalidArgument,
                              "fresh Bind requires parent proof before legacy reentry");
    if (!freshLaunching_) latestFresh_ = false;
    if (pending_) return fail(SurfaceBindingStatus::InvalidArgument,
                              "Finish is required before another binding operation");
    auto valid = validateCommon(restVertices, faceOffsets, faceCount, faceIndices);
    if (valid != SurfaceBindingStatus::Ok) return fail(valid, "invalid rest surface views");
    if (sampleBudget > restVertices.size)
        sampleBudget = restVertices.size;
    auto streamStatus = validateStream(stream);
    if (streamStatus != SurfaceBindingStatus::Ok)
        return fail(streamStatus, "CUDA stream is in an error state");
    if (deviceIndex_ < 0) {
        if (cudaGetDevice(&deviceIndex_) != cudaSuccess)
            return fail(SurfaceBindingStatus::CudaError, "unable to identify CUDA device");
    }
    diagnostic_.clear();
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return fail(SurfaceBindingStatus::CudaError, "surface binding event creation failed");
    auto* const reservation = freshLaunching_ ? freshReservation_ : nullptr;
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingFaceOffsets_.reset(faceOffsets.size, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingFaceIndices_.reset(faceIndices.size, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingSampleIndices_.reset(sampleBudget, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingRestSamples_.reset(sampleBudget, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingCurrentSamples_.reset(sampleBudget, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingNearest_.reset(restVertices.size, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingActualSampleCount_.reset(1, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess)
        return fail(SurfaceBindingStatus::CudaError, "surface binding allocation failed");
    pendingVertexCount_ = restVertices.size;
    pendingFaceCount_ = faceCount;
    pendingIndexCount_ = faceIndices.size;
    pendingSampleCount_ = sampleBudget;
    pendingRootCount_ = 0;
    pendingBind_ = true;
    pending_ = true;
    auto abort = [&](SurfaceBindingStatus status, const char* message) {
        if (freshMode_) {
            freshUnproven_ = true;
            return fail(status, message);
        }
        cudaStreamSynchronize(stream);
        discardPending();
        return fail(status, message);
    };
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(pendingActualSampleCount_.data(), 0, sizeof(uint32_t), stream) != cudaSuccess ||
        (sampleBudget && cudaMemsetAsync(pendingRestSamples_.data(), 0,
                        sampleBudget * sizeof(float3), stream) != cudaSuccess) ||
        (faceOffsets.size && cudaMemcpyAsync(pendingFaceOffsets_.data(), faceOffsets.data,
                        faceOffsets.size * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) ||
        (faceIndices.size && cudaMemcpyAsync(pendingFaceIndices_.data(), faceIndices.data,
                        faceIndices.size * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess))
        return abort(SurfaceBindingStatus::CudaError, "surface topology copy failed");
    size_t work = std::max(restVertices.size, std::max(faceCount, faceIndices.size));
    ValidateSurface<<<Blocks(work), 256, 0, stream>>>(
        restVertices, faceOffsets, faceCount, faceIndices, error_.data());
    if (cudaGetLastError() != cudaSuccess)
        return abort(SurfaceBindingStatus::CudaError, "surface validation launch failed");
    SelectFirst<<<1, 1, 0, stream>>>(pendingSampleIndices_.view(),
                                     pendingActualSampleCount_.view(), error_.data());
    if (cudaGetLastError() != cudaSuccess)
        return abort(SurfaceBindingStatus::CudaError, "first sample selection launch failed");
    if (sampleBudget) {
        InitializeNearest<<<Blocks(restVertices.size), 256, 0, stream>>>(
            pendingNearest_.view());
        if (cudaGetLastError() != cudaSuccess)
            return abort(SurfaceBindingStatus::CudaError, "sample distance initialization failed");
        for (size_t selected = 1; selected < sampleBudget; ++selected) {
            UpdateNearest<<<Blocks(restVertices.size), 256, 0, stream>>>(
                restVertices, {pendingSampleIndices_.data(), pendingSampleIndices_.size()},
                selected, pendingNearest_.view(), pendingActualSampleCount_.view(), error_.data());
            if (cudaGetLastError() != cudaSuccess)
                return abort(SurfaceBindingStatus::CudaError, "sample distance update launch failed");
            SelectNext<<<1, 256, 0, stream>>>(
                {pendingNearest_.data(), pendingNearest_.size()},
                pendingSampleIndices_.view(), selected,
                pendingActualSampleCount_.view(), error_.data());
            if (cudaGetLastError() != cudaSuccess)
                return abort(SurfaceBindingStatus::CudaError, "sample selection launch failed");
        }
        GatherSamples<<<Blocks(sampleBudget), 256, 0, stream>>>(
            restVertices, {pendingSampleIndices_.data(), pendingSampleIndices_.size()},
            pendingRestSamples_.view(), error_.data(), pendingActualSampleCount_.data());
        if (cudaGetLastError() != cudaSuccess)
            return abort(SurfaceBindingStatus::CudaError, "rest sample gather launch failed");
        if (cudaMemcpyAsync(pendingCurrentSamples_.data(), pendingRestSamples_.data(),
                            sampleBudget * sizeof(float3), cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return abort(SurfaceBindingStatus::CudaError, "initial current sample copy failed");
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess)
        return abort(SurfaceBindingStatus::CudaError, "surface binding event record failed");
    return SurfaceBindingStatus::Ok;
}

SurfaceBindingStatus CudaSurfaceBinding::Update(
    DeviceView<const float3> currentVertices,
    DeviceView<const int32_t> skinPrim, DeviceView<const float2> skinPrimUv,
    cudaStream_t stream) {
    if ((freshMode_ || freshPending_) && !freshLaunching_)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh Update requires parent proof before legacy reentry");
    if (!freshLaunching_) latestFresh_ = false;
    if (!bound_)
        return fail(SurfaceBindingStatus::InvalidArgument, "Bind must finish before Update");
    if (pending_) return fail(SurfaceBindingStatus::InvalidArgument,
                              "Finish is required before another binding operation");
    if ((currentVertices.size && !currentVertices.data) ||
        (skinPrim.size && !skinPrim.data) || (skinPrimUv.size && !skinPrimUv.data) ||
        currentVertices.size != vertexCount_ || skinPrim.size != skinPrimUv.size ||
        currentVertices.size > std::numeric_limits<uint32_t>::max() ||
        skinPrim.size > std::numeric_limits<uint32_t>::max())
        return fail(SurfaceBindingStatus::InvalidArgument, "invalid current surface or root views");
    auto streamStatus = validateStream(stream);
    if (streamStatus != SurfaceBindingStatus::Ok)
        return fail(streamStatus, "CUDA stream is in an error state");
    diagnostic_.clear();
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return fail(SurfaceBindingStatus::CudaError, "surface binding event creation failed");
    auto* const reservation = freshLaunching_ ? freshReservation_ : nullptr;
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingCurrentSamples_.reset(sampleCount_, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess ||
        pendingRootTargets_.reset(skinPrim.size, reservation, UsdGenExecutionResourceKind::Cache) != cudaSuccess)
        return fail(SurfaceBindingStatus::CudaError, "surface update allocation failed");
    pendingSampleCount_ = sampleCount_;
    pendingRootCount_ = skinPrim.size;
    pendingBind_ = false;
    pending_ = true;
    auto abort = [&](SurfaceBindingStatus status, const char* message) {
        if (freshMode_) {
            freshUnproven_ = true;
            return fail(status, message);
        }
        cudaStreamSynchronize(stream);
        discardPending();
        return fail(status, message);
    };
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return abort(SurfaceBindingStatus::CudaError, "surface update ordering failed");
    size_t work = currentVertices.size;
    if (sampleCount_ > work) work = sampleCount_;
    if (skinPrim.size > work) work = skinPrim.size;
    ValidateAndGatherUpdate<<<Blocks(work), 256, 0, stream>>>(
        currentVertices, {sampleIndices_.data(), sampleIndices_.size()},
        pendingCurrentSamples_.view(), skinPrim, skinPrimUv,
        {faceOffsets_.data(), faceOffsets_.size()}, faceCount_,
        {faceIndices_.data(), faceIndices_.size()},
        pendingRootTargets_.view(), error_.data());
    if (cudaGetLastError() != cudaSuccess)
        return abort(SurfaceBindingStatus::CudaError, "surface update validation launch failed");
    if (cudaEventRecord(ready_, stream) != cudaSuccess)
        return abort(SurfaceBindingStatus::CudaError, "surface update event record failed");
    return SurfaceBindingStatus::Ok;
}

SurfaceBindingStatus CudaSurfaceBinding::commitFresh(bool bind) {
    if (!freshPending_ || !pending_ || pendingBind_ != bind || freshUnproven_ || !freshCode_)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh surface commit requires parent terminal proof");
    // Parent proof covers both publication and semantic rejection; neither
    // path may make a later destructor synchronize this fresh operation.
    latestFresh_ = true;
    auto status = Decode(*freshCode_);
    if (status != SurfaceBindingStatus::Ok) {
        // Host-only semantic rejection: leave candidate buffers and proof
        // packet parked for a later ordinary-worker Begin cleanup.
        pending_ = pendingBind_ = false;
        freshMode_ = freshPending_ = false;
        return fail(status, status == SurfaceBindingStatus::InvalidTopology ? "invalid rest-surface topology" :
                            status == SurfaceBindingStatus::InvalidRootBinding ? "invalid skinPrim/skinPrimUv root binding" :
                            "non-finite surface binding input");
    }
    // Parent proof establishes that old buffers are no longer read by this
    // producer.  Park the prior publication before moving the candidate so no
    // DeviceBuffer move assignment frees an active buffer during install.
    if (bind) {
        retiredSampleIndices_ = std::move(sampleIndices_); retiredRestSamples_ = std::move(restSamples_);
        retiredCurrentSamples_ = std::move(currentSamples_); retiredRootTargets_ = std::move(rootTargets_);
        retiredRootCount_ = rootCount_;
        retiredFaceOffsets_ = std::move(faceOffsets_); retiredFaceIndices_ = std::move(faceIndices_);
        sampleIndices_ = std::move(pendingSampleIndices_); restSamples_ = std::move(pendingRestSamples_);
        currentSamples_ = std::move(pendingCurrentSamples_); faceOffsets_ = std::move(pendingFaceOffsets_);
        faceIndices_ = std::move(pendingFaceIndices_);
        vertexCount_ = pendingVertexCount_; faceCount_ = pendingFaceCount_; indexCount_ = pendingIndexCount_;
        sampleCount_ = std::min<size_t>(*freshActual_, pendingSampleCount_); rootCount_ = 0;
        bound_ = true;
        freshUpdatePendingAcceptance_ = false;
        // Scratch buffers remain parked; the next Begin releases them before
        // any new submission on its selected ordinary worker.
    } else {
        retiredCurrentSamples_ = std::move(currentSamples_); retiredRootTargets_ = std::move(rootTargets_);
        currentSamples_ = std::move(pendingCurrentSamples_); rootTargets_ = std::move(pendingRootTargets_);
        rootCount_ = pendingRootCount_;
        // `currentSamples_`/`rootTargets_` now deliberately expose the
        // provisional pose to the following RBF solve.  Keep the prior
        // buffers parked until the parent accepts or rolls it back.
        freshUpdatePendingAcceptance_ = true;
    }
    pendingVertexCount_ = pendingFaceCount_ = pendingIndexCount_ = 0;
    pendingSampleCount_ = pendingRootCount_ = 0;
    pending_ = pendingBind_ = false;
    freshMode_ = freshPending_ = freshUnproven_ = false;
    diagnostic_.clear();
    return SurfaceBindingStatus::Ok;
}

SurfaceBindingStatus CudaSurfaceBinding::CommitFreshBind() { return commitFresh(true); }
SurfaceBindingStatus CudaSurfaceBinding::CommitFreshUpdate() { return commitFresh(false); }
SurfaceBindingStatus CudaSurfaceBinding::AcceptFreshUpdate() {
    s_freshAcceptAttempts.fetch_add(1, std::memory_order_relaxed);
    if (!CanAcceptFreshUpdate())
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh update acceptance requires terminal downstream proof");
    if (s_failFreshResolveCommit.exchange(false, std::memory_order_acq_rel))
        return fail(SurfaceBindingStatus::CudaError,
                    "forced fresh surface resolve commit failure");
    freshUpdatePendingAcceptance_ = false;
    return SurfaceBindingStatus::Ok;
}
SurfaceBindingStatus CudaSurfaceBinding::RollbackFreshUpdate() {
    s_freshRollbackAttempts.fetch_add(1, std::memory_order_relaxed);
    if (!CanRollbackFreshUpdate())
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh update rollback requires terminal downstream proof");
    if (s_failFreshResolveCommit.exchange(false, std::memory_order_acq_rel))
        return fail(SurfaceBindingStatus::CudaError,
                    "forced fresh surface resolve commit failure");
    // Both buffer sets are host-owned and terminally proven.  Swap rather
    // than assign so this host-only decision cannot free device storage.
    std::swap(currentSamples_, retiredCurrentSamples_);
    std::swap(rootTargets_, retiredRootTargets_);
    std::swap(rootCount_, retiredRootCount_);
    freshUpdatePendingAcceptance_ = false;
    return SurfaceBindingStatus::Ok;
}
bool CudaSurfaceBinding::CanAcceptFreshUpdate() const noexcept {
    return !s_failFreshResolvePreflight.exchange(false, std::memory_order_acq_rel) &&
        freshUpdatePendingAcceptance_ && !pending_ && !freshPending_ &&
        !freshUnproven_;
}
bool CudaSurfaceBinding::CanRollbackFreshUpdate() const noexcept {
    return !s_failFreshResolvePreflight.exchange(false, std::memory_order_acq_rel) &&
        freshUpdatePendingAcceptance_ && !pending_ && !freshPending_ &&
        !freshUnproven_;
}

SurfaceBindingStatus CudaSurfaceBinding::Finish(cudaStream_t stream) {
    if (freshMode_ || freshPending_)
        return fail(SurfaceBindingStatus::InvalidArgument, "fresh surface operation requires parent proof commit");
    if (!pending_) return SurfaceBindingStatus::Ok;
    auto streamStatus = validateStream(stream);
    if (streamStatus != SurfaceBindingStatus::Ok)
        return fail(streamStatus, "surface completion stream is on the wrong CUDA device");
    int code = 0;
    uint32_t actual = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&code, error_.data(), sizeof(code),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        (pendingBind_ && cudaMemcpyAsync(&actual, pendingActualSampleCount_.data(),
                         sizeof(actual), cudaMemcpyDeviceToHost, stream) != cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        discardPending();
        return fail(SurfaceBindingStatus::CudaError, "surface binding completion failed");
    }
    auto status = Decode(code);
    if (status != SurfaceBindingStatus::Ok) {
        discardPending();
        return fail(status, status == SurfaceBindingStatus::InvalidTopology
                              ? "invalid rest-surface topology"
                              : status == SurfaceBindingStatus::InvalidRootBinding
                                  ? "invalid skinPrim/skinPrimUv root binding"
                                  : "non-finite surface binding input");
    }
    if (pendingBind_) {
        sampleIndices_ = std::move(pendingSampleIndices_);
        restSamples_ = std::move(pendingRestSamples_);
        currentSamples_ = std::move(pendingCurrentSamples_);
        faceOffsets_ = std::move(pendingFaceOffsets_);
        faceIndices_ = std::move(pendingFaceIndices_);
        vertexCount_ = pendingVertexCount_;
        faceCount_ = pendingFaceCount_;
        indexCount_ = pendingIndexCount_;
        sampleCount_ = std::min<size_t>(actual, pendingSampleCount_);
        rootCount_ = 0;
        rootTargets_.reset(0);
        bound_ = true;
        pendingNearest_.reset(0);
        pendingActualSampleCount_.reset(0);
    } else {
        currentSamples_ = std::move(pendingCurrentSamples_);
        rootTargets_ = std::move(pendingRootTargets_);
        rootCount_ = pendingRootCount_;
    }
    pendingVertexCount_ = pendingFaceCount_ = pendingIndexCount_ = 0;
    pendingSampleCount_ = pendingRootCount_ = 0;
    pending_ = false;
    pendingBind_ = false;
    diagnostic_.clear();
    return SurfaceBindingStatus::Ok;
}

void failNextCudaSurfaceFreshProofAllocationForTesting() {
    s_failFreshProofAllocation.store(true, std::memory_order_release);
}
void failNextCudaSurfaceFreshProofEnqueueForTesting() {
    s_failFreshProofEnqueue.store(true, std::memory_order_release);
}
void failNextCudaSurfaceFreshProofAfterStatusForTesting() {
    s_failFreshProofAfterStatus.store(true, std::memory_order_release);
}
void TestFailNextFreshSurfaceResolvePreflight() noexcept {
    s_failFreshResolvePreflight.store(true, std::memory_order_release);
}
void TestFailNextFreshSurfaceResolveCommit() noexcept {
    s_failFreshResolveCommit.store(true, std::memory_order_release);
}
uint64_t FreshSurfaceAcceptAttemptCountForTesting() noexcept {
    return s_freshAcceptAttempts.load(std::memory_order_acquire);
}
uint64_t FreshSurfaceRollbackAttemptCountForTesting() noexcept {
    return s_freshRollbackAttempts.load(std::memory_order_acquire);
}

}} // namespace usdGen::gpu
