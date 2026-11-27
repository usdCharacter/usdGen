#include "usdGenCurves/cuda/namedChannelTopology.h"
#include "usdGenCuda/backendId.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <vector>

namespace usdGen::gpu {
namespace {

enum class TransformMode : int { Resample = 0, Compact = 1 };
enum class ChannelDomain : int { Point = 0, Primitive = 1, Constant = 2 };
constexpr int kBadInput = 1;
constexpr int kBadStableId = 2;

void Fail(std::string const& message, std::string* reason)
{
    if (reason) *reason = message;
}

bool MetadataDomain(UsdGenDeviceChannelMetadata const& metadata,
                    ChannelDomain* domain)
{
    if (metadata.domain == UsdGenDeviceDomain::Point) {
        *domain = ChannelDomain::Point;
        return true;
    }
    if (metadata.domain == UsdGenDeviceDomain::Primitive) {
        *domain = ChannelDomain::Primitive;
        return true;
    }
    if (metadata.domain == UsdGenDeviceDomain::Groom) {
        *domain = ChannelDomain::Constant;
        return true;
    }
    return false;
}

bool ValidateMetadata(UsdGenDeviceChannelMetadata const& metadata,
                      DeviceCurveGeometryView const& geometry,
                      ChannelDomain* domain, std::string* reason)
{
    if (metadata.name.IsEmpty() || metadata.semantic != UsdGenDeviceChannelSemantic::Generic ||
        !metadata.readOnly || metadata.arity == 0 || metadata.arity > 16 ||
        (metadata.type != UsdGenDeviceValueType::Float32 &&
         metadata.type != UsdGenDeviceValueType::Int32) ||
        metadata.strideBytes != metadata.arity * sizeof(uint32_t) ||
        !MetadataDomain(metadata, domain)) {
        Fail("CUDA named topology transform has unsupported channel metadata", reason);
        return false;
    }
    uint64_t const wanted = *domain == ChannelDomain::Point ? geometry.pointCount :
        *domain == ChannelDomain::Primitive ? geometry.curveCount : 1;
    if (metadata.elementCount != wanted) {
        Fail("CUDA named topology transform channel cardinality disagrees with source geometry", reason);
        return false;
    }
    return true;
}

bool ValidateGeometry(DeviceCurveGeometryView const& geometry, std::string* reason)
{
    if (geometry.curveCount > UINT32_MAX || geometry.pointCount > UINT32_MAX ||
        !geometry.curveOffsets.data || geometry.curveOffsets.size != geometry.curveCount + 1 ||
        (!geometry.stableIds.data && geometry.curveCount != 0) ||
        geometry.stableIds.size != geometry.curveCount) {
        Fail("CUDA named topology transform has invalid curve topology", reason);
        return false;
    }
    return true;
}

bool SameDevice(std::shared_ptr<const UsdGenDeviceGeneration> const& source,
                std::shared_ptr<const UsdGenDeviceGeneration> const& topology,
                uint64_t generation, std::string* reason)
{
    if (!source || !topology || source->Identity().backend != UsdGenCudaBackendId() ||
        topology->Identity().backend != UsdGenCudaBackendId() ||
        source->Identity().deviceIndex != topology->Identity().deviceIndex ||
        generation <= topology->Identity().generation ||
        generation <= source->Identity().generation) {
        Fail("CUDA named topology transform requires ordered local CUDA generations", reason);
        return false;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != source->Identity().deviceIndex) {
        Fail("CUDA named topology transform requires the source device", reason);
        return false;
    }
    return true;
}

bool SameRawDevice(DeviceCurveGeometryView const& source,
                   DeviceCurveGeometryView const& target, int device,
                   std::string* reason)
{
    auto local = [device](auto view) {
        if (!view.size) return true;
        cudaPointerAttributes attributes{};
        return view.data && cudaPointerGetAttributes(&attributes, view.data) == cudaSuccess &&
            attributes.type == cudaMemoryTypeDevice && attributes.device == device;
    };
    if (!local(source.curveOffsets) || !local(source.stableIds) ||
        !local(target.curveOffsets) || !local(target.stableIds)) {
        Fail("CUDA named topology transform raw geometry belongs to another device", reason);
        return false;
    }
    return true;
}

__device__ void SetError(int* error, int code) { atomicCAS(error, 0, code); }

__device__ int FindStableId(DeviceView<const uint64_t> ids, uint64_t id)
{
    int found = -1;
    for (size_t i = 0; i != ids.size; ++i) {
        if (ids.data[i] != id) continue;
        if (found >= 0) return -2;
        found = static_cast<int>(i);
    }
    return found;
}

__global__ void ValidateStableIds(DeviceCurveGeometryView source,
                                  DeviceCurveGeometryView target,
                                  TransformMode mode, int* error)
{
    size_t const first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = size_t(blockDim.x) * gridDim.x;
    for (size_t outputCurve = first; outputCurve < target.curveCount;
         outputCurve += stride) {
        uint64_t const id = target.stableIds.data[outputCurve];
        int const sourceCurve = FindStableId(
            {source.stableIds.data, source.stableIds.size}, id);
        if (sourceCurve < 0 ||
            (mode == TransformMode::Resample &&
             sourceCurve != static_cast<int>(outputCurve))) {
            SetError(error, kBadStableId);
            continue;
        }
        // Compaction is a stable survivor subset, never a fan-out operation.
        // Reject a duplicate target id even when it maps uniquely in source.
        for (size_t earlier = 0; earlier != outputCurve; ++earlier) {
            if (target.stableIds.data[earlier] == id) {
                SetError(error, kBadStableId);
                break;
            }
        }
    }
}

template <class T>
__device__ void CopyElement(T const* source, T* target, uint32_t arity,
                            size_t sourceIndex, size_t targetIndex)
{
    for (uint32_t component = 0; component != arity; ++component)
        target[targetIndex * arity + component] = source[sourceIndex * arity + component];
}

__global__ void TransformFloat(DeviceCurveGeometryView source,
                               DeviceCurveGeometryView target,
                               DeviceView<const float> input,
                               DeviceView<float> output, uint32_t arity,
                               ChannelDomain domain, TransformMode mode, int* error)
{
    if (domain == ChannelDomain::Constant) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            CopyElement(input.data, output.data, arity, 0, 0);
        return;
    }
    size_t const first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = size_t(blockDim.x) * gridDim.x;
    for (size_t outputCurve = first; outputCurve < target.curveCount; outputCurve += stride) {
        int sourceCurve = FindStableId({source.stableIds.data, source.stableIds.size},
                                       target.stableIds.data[outputCurve]);
        if (mode == TransformMode::Resample && sourceCurve != static_cast<int>(outputCurve))
            sourceCurve = -1;
        if (sourceCurve < 0) { SetError(error, kBadStableId); continue; }
        if (domain == ChannelDomain::Primitive) {
            CopyElement(input.data, output.data, arity, sourceCurve, outputCurve);
            continue;
        }
        uint32_t const sourceBegin = source.curveOffsets.data[sourceCurve];
        uint32_t const sourceEnd = source.curveOffsets.data[sourceCurve + 1];
        uint32_t const targetBegin = target.curveOffsets.data[outputCurve];
        uint32_t const targetEnd = target.curveOffsets.data[outputCurve + 1];
        if (sourceEnd < sourceBegin || targetEnd < targetBegin || sourceEnd > source.pointCount ||
            targetEnd > target.pointCount || (sourceEnd == sourceBegin && targetEnd != targetBegin)) {
            SetError(error, kBadInput); continue;
        }
        if (targetEnd == targetBegin) continue;
        uint32_t const sourceCount = sourceEnd - sourceBegin;
        uint32_t const targetCount = targetEnd - targetBegin;
        for (uint32_t point = 0; point != targetCount; ++point) {
            double const t = targetCount <= 1 || sourceCount <= 1 ? 0.0 :
                double(point) * double(sourceCount - 1) / double(targetCount - 1);
            uint32_t const lower = min(static_cast<uint32_t>(t), sourceCount - 1);
            uint32_t const upper = min(lower + 1, sourceCount - 1);
            float const fraction = static_cast<float>(t - double(lower));
            for (uint32_t component = 0; component != arity; ++component) {
                float const a = input.data[(size_t(sourceBegin + lower) * arity) + component];
                float const b = input.data[(size_t(sourceBegin + upper) * arity) + component];
                output.data[(size_t(targetBegin + point) * arity) + component] =
                    static_cast<float>(double(a) + (double(b) - double(a)) * double(fraction));
            }
        }
    }
}

__global__ void TransformInt(DeviceCurveGeometryView source,
                             DeviceCurveGeometryView target,
                             DeviceView<const int> input,
                             DeviceView<int> output, uint32_t arity,
                             ChannelDomain domain, TransformMode mode, int* error)
{
    if (domain == ChannelDomain::Constant) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            CopyElement(input.data, output.data, arity, 0, 0);
        return;
    }
    size_t const first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = size_t(blockDim.x) * gridDim.x;
    for (size_t outputCurve = first; outputCurve < target.curveCount; outputCurve += stride) {
        int sourceCurve = FindStableId({source.stableIds.data, source.stableIds.size},
                                       target.stableIds.data[outputCurve]);
        if (mode == TransformMode::Resample && sourceCurve != static_cast<int>(outputCurve))
            sourceCurve = -1;
        if (sourceCurve < 0) { SetError(error, kBadStableId); continue; }
        if (domain == ChannelDomain::Primitive) {
            CopyElement(input.data, output.data, arity, sourceCurve, outputCurve);
            continue;
        }
        uint32_t const sourceBegin = source.curveOffsets.data[sourceCurve];
        uint32_t const sourceEnd = source.curveOffsets.data[sourceCurve + 1];
        uint32_t const targetBegin = target.curveOffsets.data[outputCurve];
        uint32_t const targetEnd = target.curveOffsets.data[outputCurve + 1];
        if (sourceEnd < sourceBegin || targetEnd < targetBegin || sourceEnd > source.pointCount ||
            targetEnd > target.pointCount || (sourceEnd == sourceBegin && targetEnd != targetBegin)) {
            SetError(error, kBadInput); continue;
        }
        if (targetEnd == targetBegin) continue;
        uint32_t const sourceCount = sourceEnd - sourceBegin;
        uint32_t const targetCount = targetEnd - targetBegin;
        for (uint32_t point = 0; point != targetCount; ++point) {
            double const t = targetCount <= 1 || sourceCount <= 1 ? 0.0 :
                double(point) * double(sourceCount - 1) / double(targetCount - 1);
            uint32_t const nearest = min(static_cast<uint32_t>(floor(t + 0.5)), sourceCount - 1);
            CopyElement(input.data, output.data, arity, sourceBegin + nearest, targetBegin + point);
        }
    }
}

unsigned Blocks(size_t count)
{
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>((count + 255) / 256, 65535)));
}

} // namespace

struct CudaNamedChannelTopologyCandidate::Impl {
    enum class Phase : uint8_t { Idle, Submitted, Armed, Finished, Quarantined };
    struct InputChannel {
        UsdGenDeviceChannelMetadata metadata;
        ChannelDomain domain;
        CudaNamedChannelLease lease;
        DeviceView<const unsigned char> rawBytes;
        DeviceView<const unsigned char> Bytes() const noexcept {
            return rawBytes.data || rawBytes.size ? rawBytes : lease.Bytes();
        }
    };
    struct CallbackState {
        std::atomic<int> status{std::numeric_limits<int>::min()};
        std::atomic<bool> called{false};
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept = nullptr;
        void* userdata = nullptr;
    };
    static void Callback(cudaStream_t stream, cudaError_t status,
                         void* userdata) noexcept;
    DeviceCurveGeometryView SourceGeometry() const noexcept {
        return raw ? sourceRaw : sourceGeometry.Geometry();
    }
    DeviceCurveGeometryView TargetGeometry() const noexcept {
        return raw ? targetRaw : targetGeometry.Geometry();
    }
    cudaError_t Submit(std::string* reason,
                       UsdGenExecutionMemoryReservation* reservation);
    bool Prove(cudaError_t completionStatus, std::string* reason);

    ~Impl() {
        if (unproven) Quarantine();
        else ReleaseHostStatus();
    }

    void ReleaseHostStatus() noexcept {
        if (!hostError) return;
        if (cudaFreeHost(hostError) == cudaSuccess) hostErrorPermit.Release();
        else hostErrorPermit.Abandon();
        hostError = nullptr;
    }

    void Quarantine() noexcept {
        if (phase == Phase::Quarantined) return;
        // Preserve the fact that this candidate had borrowed raw inputs when
        // Quarantine clears `unproven` below.  Callers may need that sticky
        // proof-loss bit after BeginRaw has already performed its own
        // fail-closed cleanup; otherwise a queued validation kernel could let
        // the raw owners destruct normally.
        bool const quarantinedUnproven = unproven;
        for (CudaNamedChannelPlane& output : outputs)
            if (output.bytes) output.bytes->quarantine();
        error.quarantine();
        // The status copy is ordered before the callback.  Without that
        // callback proof, pinned host memory can still be a DMA destination.
        if (hostError) {
            hostErrorPermit.Abandon();
            hostError = nullptr; // deliberately leaked with its retained charge
        }
        // CUDA retains this pointer until its native callback returns.  A
        // caller may destroy a failed/unproved candidate before then, so keep
        // the tiny relay state alive as part of the quarantine too.
        if (callbackArmed) callbackState.release();
        else callbackState.reset();
        inputs.clear();
        sourceGeometry = {};
        targetGeometry = {};
        sourceRaw = {};
        targetRaw = {};
        // In raw mode this token can be the only owner of input geometry and
        // plane buffers.  Lost proof therefore requires retaining the token,
        // just like private outputs and callback userdata.  The allocation is
        // made before submission so this noexcept path only relinquishes it.
        if (rawLifetime) {
            if (unproven) rawLifetime.release();
            else rawLifetime.reset();
        }
        source.reset();
        topology.reset();
        if (quarantinedUnproven) quarantinedWithUnprovenWork = true;
        unproven = false;
        phase = Phase::Quarantined;
    }

    Phase phase = Phase::Idle;
    bool unproven = false;
    bool noOp = false;
    bool raw = false;
    bool quarantinedWithUnprovenWork = false;
    TransformMode mode = TransformMode::Resample;
    uint64_t generation = 0;
    cudaStream_t stream = nullptr;
    std::shared_ptr<const UsdGenDeviceGeneration> source;
    std::shared_ptr<const UsdGenDeviceGeneration> topology;
    CudaGeometryLease sourceGeometry;
    CudaGeometryLease targetGeometry;
    DeviceCurveGeometryView sourceRaw{};
    DeviceCurveGeometryView targetRaw{};
    std::unique_ptr<std::shared_ptr<const void>> rawLifetime;
    std::vector<InputChannel> inputs;
    std::vector<CudaNamedChannelPlane> outputs;
    DeviceBuffer<int> error;
    int* hostError = nullptr;
    UsdGenExecutionResourcePermit hostErrorPermit;
    std::unique_ptr<CallbackState> callbackState;
    bool callbackArmed = false;
};

CudaNamedChannelTopologyCandidate::CudaNamedChannelTopologyCandidate()
    : impl_(std::make_unique<Impl>()) {}
CudaNamedChannelTopologyCandidate::~CudaNamedChannelTopologyCandidate() = default;

void CudaNamedChannelTopologyCandidate::Impl::Callback(
    cudaStream_t stream, cudaError_t status, void* userdata) noexcept {
    auto* callback = static_cast<CallbackState*>(userdata);
    callback->status.store(static_cast<int>(status), std::memory_order_relaxed);
    callback->called.store(true, std::memory_order_release);
    callback->callback(stream, status, callback->userdata);
}

cudaError_t CudaNamedChannelTopologyCandidate::Begin(
    std::shared_ptr<const UsdGenDeviceGeneration> source,
    std::shared_ptr<const UsdGenDeviceGeneration> topology,
    uint64_t generation, cudaStream_t stream, CudaNamedChannelTopologyMode mode,
    std::string* reason, UsdGenExecutionMemoryReservation* reservation)
{
    if (!impl_ || impl_->phase != Impl::Phase::Idle) {
        Fail("CUDA named topology transform candidate is not idle", reason);
        return cudaErrorInvalidResourceHandle;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        Fail("CUDA named topology transform cannot run under stream capture", reason);
        impl_->phase = Impl::Phase::Finished;
        return cudaErrorStreamCaptureUnsupported;
    }
    Impl& state = *impl_;
    state.mode = mode == CudaNamedChannelTopologyMode::Resample
        ? TransformMode::Resample : TransformMode::Compact;
    if (!SameDevice(source, topology, generation, reason)) {
        state.phase = Impl::Phase::Finished;
        return cudaErrorInvalidDevice;
    }
    state.source = std::move(source);
    state.topology = std::move(topology);
    state.generation = generation;
    state.stream = stream;
    state.sourceGeometry = AcquireGeometry(state.source, stream);
    state.targetGeometry = AcquireGeometry(state.topology, stream);
    if (!state.sourceGeometry || !state.targetGeometry ||
        !ValidateGeometry(state.sourceGeometry.Geometry(), reason) ||
        !ValidateGeometry(state.targetGeometry.Geometry(), reason) ||
        (state.mode == TransformMode::Resample &&
         state.sourceGeometry.Geometry().curveCount != state.targetGeometry.Geometry().curveCount)) {
        if (reason && reason->empty())
            *reason = "CUDA named topology transform cannot acquire compatible geometry";
        state.phase = Impl::Phase::Finished;
        return cudaErrorInvalidValue;
    }

    std::vector<UsdGenToken> names;
    for (UsdGenDeviceChannelMetadata const& metadata : state.source->Channels()) {
        if (metadata.semantic != UsdGenDeviceChannelSemantic::Generic) continue;
        ChannelDomain domain;
        if (!ValidateMetadata(metadata, state.sourceGeometry.Geometry(), &domain, reason) ||
            metadata.elementCount > std::numeric_limits<size_t>::max() / metadata.strideBytes ||
            std::find(names.begin(), names.end(), metadata.name) != names.end()) {
            if (reason && reason->empty())
                *reason = "CUDA named topology transform has duplicate or oversized input";
            state.phase = Impl::Phase::Finished;
            return cudaErrorInvalidValue;
        }
        CudaNamedChannelLease input = AcquireNamedChannel(state.source, metadata.name, stream);
        if (!input || input.Bytes().size != size_t(metadata.elementCount) * metadata.strideBytes) {
            Fail("CUDA named topology transform cannot acquire named input", reason);
            state.phase = Impl::Phase::Finished;
            return cudaErrorInvalidValue;
        }
        names.push_back(metadata.name);
        state.inputs.push_back({metadata, domain, std::move(input)});
    }
    // A Generic plane inherited only from topology would have source-sized
    // cardinality after the topology change. Never carry it through a COW
    // revision without rebuilding it from a corresponding source plane.
    for (UsdGenDeviceChannelMetadata const& metadata : state.topology->Channels()) {
        if (metadata.semantic == UsdGenDeviceChannelSemantic::Generic &&
            std::find(names.begin(), names.end(), metadata.name) == names.end()) {
            Fail("CUDA named topology transform refuses an unrelated target overlay", reason);
            state.phase = Impl::Phase::Finished;
            return cudaErrorInvalidValue;
        }
    }
    return state.Submit(reason, reservation);
}

cudaError_t CudaNamedChannelTopologyCandidate::BeginRaw(
    DeviceCurveGeometryView source, DeviceCurveGeometryView target,
    std::vector<RawInputPlane> inputs, cudaStream_t stream,
    CudaNamedChannelTopologyMode mode, std::shared_ptr<const void> lifetime,
    std::string* reason, UsdGenExecutionMemoryReservation* reservation)
{
    if (!impl_ || impl_->phase != Impl::Phase::Idle) {
        Fail("CUDA named topology transform candidate is not idle", reason);
        return cudaErrorInvalidResourceHandle;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    int device = -1;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone || cudaGetDevice(&device) != cudaSuccess) {
        Fail("CUDA named topology transform cannot run raw mode under capture or without a device", reason);
        impl_->phase = Impl::Phase::Finished;
        return cudaErrorInvalidDevice;
    }
    Impl& state = *impl_;
    state.raw = true;
    state.mode = mode == CudaNamedChannelTopologyMode::Resample
        ? TransformMode::Resample : TransformMode::Compact;
    state.stream = stream;
    state.sourceRaw = source;
    state.targetRaw = target;
    try {
        state.rawLifetime =
            std::make_unique<std::shared_ptr<const void>>(std::move(lifetime));
    } catch (...) {
        state.phase = Impl::Phase::Finished;
        Fail("CUDA named topology transform cannot retain raw input lifetime", reason);
        return cudaErrorMemoryAllocation;
    }
    if (!ValidateGeometry(source, reason) || !ValidateGeometry(target, reason) ||
        !SameRawDevice(source, target, device, reason) ||
        (state.mode == TransformMode::Resample && source.curveCount != target.curveCount)) {
        if (reason && reason->empty())
            *reason = "CUDA named topology transform raw inputs are incompatible";
        state.phase = Impl::Phase::Finished;
        return cudaErrorInvalidValue;
    }
    std::vector<UsdGenToken> names;
    for (RawInputPlane const& input : inputs) {
        ChannelDomain domain;
        if (!ValidateMetadata(input.metadata, source, &domain, reason) ||
            input.metadata.elementCount >
                std::numeric_limits<size_t>::max() / input.metadata.strideBytes ||
            (!input.bytes.data && input.bytes.size != 0) || input.bytes.size !=
                size_t(input.metadata.elementCount) * input.metadata.strideBytes ||
            std::find(names.begin(), names.end(), input.metadata.name) != names.end()) {
            if (reason && reason->empty())
                *reason = "CUDA named topology transform has invalid raw named input";
            state.phase = Impl::Phase::Finished;
            return cudaErrorInvalidValue;
        }
        cudaPointerAttributes attributes{};
        if (input.bytes.size &&
            (cudaPointerGetAttributes(&attributes, input.bytes.data) != cudaSuccess ||
             attributes.type != cudaMemoryTypeDevice || attributes.device != device)) {
            Fail("CUDA named topology transform raw named input belongs to another device", reason);
            state.phase = Impl::Phase::Finished;
            return cudaErrorInvalidDevice;
        }
        if (input.readyBuffer) {
            auto const ready =
                static_cast<DeviceBuffer<unsigned char> const&>(*input.readyBuffer).view();
            if (ready.data != input.bytes.data || ready.size != input.bytes.size ||
                input.readyBuffer->waitOn(stream) != cudaSuccess) {
                Fail("CUDA named topology transform cannot wait for raw named input", reason);
                state.phase = Impl::Phase::Finished;
                return cudaErrorInvalidResourceHandle;
            }
        }
        names.push_back(input.metadata.name);
        state.inputs.push_back({input.metadata, domain, {}, input.bytes});
    }
    return state.Submit(reason, reservation);
}

cudaError_t CudaNamedChannelTopologyCandidate::Impl::Submit(
    std::string* reason, UsdGenExecutionMemoryReservation* reservation)
{
    if (inputs.empty()) {
        noOp = true;
        phase = Phase::Submitted;
        return cudaSuccess;
    }
    if (error.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess) {
        Fail("CUDA named topology transform cannot allocate validation status", reason);
        phase = Phase::Finished;
        return cudaErrorMemoryAllocation;
    }
    auto statusPermit = TryReserveCudaExecutionBytes(sizeof(int),
        UsdGenExecutionResourceKind::Scratch, reservation);
    if (!statusPermit || cudaHostAlloc(reinterpret_cast<void**>(&hostError),
                                       sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
        Fail("CUDA named topology transform cannot allocate pinned validation status", reason);
        phase = Phase::Finished;
        return cudaErrorMemoryAllocation;
    }
    hostErrorPermit = std::move(*statusPermit);
    *hostError = std::numeric_limits<int>::min();
    unproven = true;
    cudaError_t result = cudaMemsetAsync(error.data(), 0, sizeof(int), stream);
    if (result != cudaSuccess) {
        Quarantine();
        Fail("CUDA named topology transform validation submission failed", reason);
        return result;
    }
    auto const source = SourceGeometry();
    auto const target = TargetGeometry();
    ValidateStableIds<<<Blocks(target.curveCount), 256, 0, stream>>>(
        source, target, mode, error.data());
    if (cudaGetLastError() != cudaSuccess) {
        Quarantine();
        Fail("CUDA named topology stable-id validation launch failed", reason);
        return cudaErrorLaunchFailure;
    }
    for (InputChannel const& input : inputs) {
        auto const bytes = input.Bytes();
        UsdGenDeviceChannelMetadata const& metadata = input.metadata;
        uint64_t const targetElements = input.domain == ChannelDomain::Point
            ? target.pointCount : input.domain == ChannelDomain::Primitive
            ? target.curveCount : 1;
        if (targetElements > std::numeric_limits<size_t>::max() / metadata.strideBytes) {
            Quarantine();
            Fail("CUDA named topology transform output size overflows", reason);
            return cudaErrorInvalidValue;
        }
        CudaNamedChannelPlane output;
        output.metadata = metadata;
        output.metadata.elementCount = targetElements;
        output.bytes = std::make_unique<DeviceBuffer<unsigned char>>();
        if (output.bytes->reset(size_t(targetElements) * metadata.strideBytes,
                                reservation, UsdGenExecutionResourceKind::Active) != cudaSuccess) {
            Quarantine();
            Fail("CUDA named topology transform cannot allocate private output", reason);
            return cudaErrorMemoryAllocation;
        }
        if (metadata.type == UsdGenDeviceValueType::Float32) {
            TransformFloat<<<Blocks(target.curveCount), 256, 0, stream>>>(
                source, target,
                {reinterpret_cast<float const*>(bytes.data), bytes.size / sizeof(float)},
                {reinterpret_cast<float*>(output.bytes->data()),
                 output.bytes->size() / sizeof(float)}, metadata.arity, input.domain,
                mode, error.data());
        } else {
            TransformInt<<<Blocks(target.curveCount), 256, 0, stream>>>(
                source, target,
                {reinterpret_cast<int const*>(bytes.data), bytes.size / sizeof(int)},
                {reinterpret_cast<int*>(output.bytes->data()),
                 output.bytes->size() / sizeof(int)}, metadata.arity, input.domain,
                mode, error.data());
        }
        if (cudaGetLastError() != cudaSuccess) {
            Quarantine();
            Fail("CUDA named topology transform launch failed", reason);
            return cudaErrorLaunchFailure;
        }
        outputs.push_back(std::move(output));
    }
    phase = Phase::Submitted;
    return cudaSuccess;
}

cudaError_t CudaNamedChannelTopologyCandidate::FinishAsync(
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata)
{
    if (!impl_ || impl_->phase != Impl::Phase::Submitted || !callback)
        return cudaErrorInvalidResourceHandle;
    Impl& state = *impl_;
    try {
        state.callbackState = std::make_unique<Impl::CallbackState>();
    } catch (...) {
        if (state.unproven) state.Quarantine();
        else state.phase = Impl::Phase::Finished;
        return cudaErrorMemoryAllocation;
    }
    state.callbackState->callback = callback;
    state.callbackState->userdata = userdata;
    if (!state.noOp &&
        cudaMemcpyAsync(state.hostError, state.error.data(), sizeof(int),
                        cudaMemcpyDeviceToHost, state.stream) != cudaSuccess) {
        state.Quarantine();
        return cudaErrorUnknown;
    }
    // Record every private plane after its final transform/status dependency.
    // A published owner can subsequently make a consumer stream wait on these
    // events without relying on the submission stream's lifetime.
    for (CudaNamedChannelPlane& output : state.outputs)
        if (!output.bytes || output.bytes->recordUse(state.stream) != cudaSuccess) {
            state.Quarantine();
            return cudaErrorUnknown;
        }
    if (!state.noOp && state.error.recordUse(state.stream) != cudaSuccess) {
        state.Quarantine();
        return cudaErrorUnknown;
    }
    // A no-op transform still needs callback proof before it releases its
    // immutable source/topology leases.  In both paths this is set before the
    // callback is armed so destruction remains fail-closed.
    state.unproven = true;
    if (cudaStreamAddCallback(state.stream, Impl::Callback,
                              state.callbackState.get(), 0) != cudaSuccess) {
        state.Quarantine();
        return cudaErrorUnknown;
    }
    state.callbackArmed = true;
    state.phase = Impl::Phase::Armed;
    return cudaSuccess;
}

std::shared_ptr<const UsdGenDeviceGeneration>
CudaNamedChannelTopologyCandidate::Commit(cudaError_t completionStatus,
                                           std::string* reason)
{
    if (!impl_ || impl_->raw) {
        Fail("CUDA named topology transform raw candidate requires CommitPlanes", reason);
        return {};
    }
    Impl& state = *impl_;
    if (!state.Prove(completionStatus, reason)) return {};
    if (state.noOp) {
        state.phase = Impl::Phase::Finished;
        return state.topology;
    }
    auto result = MakeNamedChannelRevisionGeneration(state.topology, state.generation,
                                                      std::move(state.outputs), {}, reason);
    state.phase = Impl::Phase::Finished;
    return result;
}

bool CudaNamedChannelTopologyCandidate::CommitPlanes(
    cudaError_t completionStatus, std::vector<CudaNamedChannelPlane>* planes,
    std::string* reason)
{
    if (!impl_ || !impl_->raw) {
        Fail("CUDA named topology transform generation candidate requires Commit", reason);
        return false;
    }
    if (!planes || !planes->empty()) {
        Fail("CUDA named topology transform CommitPlanes requires an empty output", reason);
        return false;
    }
    Impl& state = *impl_;
    if (!state.Prove(completionStatus, reason)) return false;
    state.phase = Impl::Phase::Finished;
    if (!state.noOp) *planes = std::move(state.outputs);
    return true;
}

bool CudaNamedChannelTopologyCandidate::Impl::Prove(
    cudaError_t completionStatus, std::string* reason)
{
    if (phase != Phase::Armed) {
        Fail("CUDA named topology transform candidate has no proved completion", reason);
        return false;
    }
    if (!callbackState || !callbackState->called.load(std::memory_order_acquire)) {
        Fail("CUDA named topology transform callback has not proved completion", reason);
        return false;
    }
    auto const callbackStatus = static_cast<cudaError_t>(
        callbackState->status.load(std::memory_order_acquire));
    if (completionStatus != callbackStatus) {
        Fail("CUDA named topology transform callback status mismatch", reason);
        return false;
    }
    if (completionStatus != cudaSuccess) {
        Quarantine();
        Fail("CUDA named topology transform completion is unproven", reason);
        return false;
    }
    unproven = false;
    if (!noOp && (!hostError || *hostError != 0)) {
        int const status = hostError ? *hostError : kBadInput;
        phase = Phase::Finished;
        Fail(status == kBadStableId
                 ? "CUDA named topology transform has missing or duplicate survivor stable ids"
                 : "CUDA named topology transform rejected malformed topology", reason);
        return false;
    }
    return true;
}

void CudaNamedChannelTopologyCandidate::Quarantine() noexcept {
    if (impl_) impl_->Quarantine();
}

bool CudaNamedChannelTopologyCandidate::HasUnprovenWork() const noexcept {
    return impl_ && (impl_->unproven || impl_->quarantinedWithUnprovenWork);
}

namespace {

struct SynchronousCompletion {
    std::atomic<int> status{std::numeric_limits<int>::min()};
};

void SynchronousCallback(cudaStream_t, cudaError_t status, void* userdata) noexcept {
    static_cast<SynchronousCompletion*>(userdata)->status.store(
        static_cast<int>(status), std::memory_order_release);
}

std::shared_ptr<const UsdGenDeviceGeneration> Transform(
    std::shared_ptr<const UsdGenDeviceGeneration> const& source,
    std::shared_ptr<const UsdGenDeviceGeneration> const& topology,
    uint64_t generation, cudaStream_t stream, TransformMode mode, std::string* reason,
    UsdGenExecutionMemoryReservation* reservation)
{
    CudaNamedChannelTopologyCandidate candidate;
    if (candidate.Begin(source, topology, generation, stream,
            mode == TransformMode::Resample ? CudaNamedChannelTopologyMode::Resample
                                             : CudaNamedChannelTopologyMode::Compaction,
            reason, reservation) != cudaSuccess)
        return {};
    auto completion = std::make_unique<SynchronousCompletion>();
    if (candidate.FinishAsync(SynchronousCallback, completion.get()) != cudaSuccess) {
        candidate.Quarantine();
        Fail("CUDA named topology transform callback installation failed", reason);
        return {};
    }
    if (CudaFenceStream(stream) != cudaSuccess) {
        candidate.Quarantine();
        // CUDA may still deliver an already-armed callback after a failed
        // synchronization. Preserve its userdata for exactly the same reason
        // the candidate preserves its native CallbackState.
        completion.release();
        Fail("CUDA named topology transform completion is unproven", reason);
        return {};
    }
    int const status = completion->status.load(std::memory_order_acquire);
    if (status == std::numeric_limits<int>::min()) {
        candidate.Quarantine();
        Fail("CUDA named topology transform callback did not run", reason);
        return {};
    }
    return candidate.Commit(static_cast<cudaError_t>(status), reason);
}

} // namespace

std::shared_ptr<const UsdGenDeviceGeneration>
TransformCudaNamedChannelsForResample(
    std::shared_ptr<const UsdGenDeviceGeneration> const& source,
    std::shared_ptr<const UsdGenDeviceGeneration> const& topology,
    uint64_t generation, cudaStream_t stream, std::string* reason,
    UsdGenExecutionMemoryReservation* reservation)
{
    return Transform(source, topology, generation, stream, TransformMode::Resample, reason, reservation);
}

std::shared_ptr<const UsdGenDeviceGeneration>
TransformCudaNamedChannelsForCompaction(
    std::shared_ptr<const UsdGenDeviceGeneration> const& source,
    std::shared_ptr<const UsdGenDeviceGeneration> const& topology,
    uint64_t generation, cudaStream_t stream, std::string* reason,
    UsdGenExecutionMemoryReservation* reservation)
{
    return Transform(source, topology, generation, stream, TransformMode::Compact, reason, reservation);
}

} // namespace usdGen::gpu
