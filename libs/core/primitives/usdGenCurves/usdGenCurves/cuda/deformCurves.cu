#include "usdGenCurves/cuda/deformCurves.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace usdGen { namespace gpu { namespace {

constexpr int kBadOffsets = 1;
constexpr int kNonFinite = 2;
constexpr int kBadValue = 3;

__device__ void SetError(int *error, int code) { atomicCAS(error, 0, code); }
__device__ bool Finite(float value) { return isfinite(value); }

__device__ bool ReadScalar(ScalarField field, size_t curve, size_t point,
                           float *value, int *error) {
    size_t index = 0;
    if (!field.data) {
        if (field.count != 0 || field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        *value = field.literal;
    } else {
        if (field.domain == expr::Domain::Primitive) index = curve;
        else if (field.domain == expr::Domain::Point) index = point;
        else if (field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= field.count) { SetError(error, kBadValue); return false; }
        *value = field.data[index];
    }
    if (!Finite(*value)) { SetError(error, kNonFinite); return false; }
    if (*value < 0.0f || *value > 1.0f) {
        SetError(error, kBadValue); return false;
    }
    return true;
}

__device__ bool ReadBool(BoolField field, size_t curve, size_t point,
                         bool *value, int *error) {
    size_t index = 0;
    uint8_t raw = field.literal ? 1u : 0u;
    if (field.data) {
        if (field.domain == expr::Domain::Primitive) index = curve;
        else if (field.domain == expr::Domain::Point) index = point;
        else if (field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= field.count) { SetError(error, kBadValue); return false; }
        raw = field.data[index];
    } else if (field.count != 0 || field.domain != expr::Domain::Groom) {
        SetError(error, kBadValue); return false;
    }
    if (raw > 1u) { SetError(error, kBadValue); return false; }
    *value = raw != 0;
    return true;
}

__global__ void ValidateShape(const uint32_t *offsets, int curves, int points,
                              const float3 *roots, int *error) {
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        (offsets[0] != 0 || offsets[curves] != uint32_t(points)))
        SetError(error, kBadOffsets);
    size_t curve = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (curve >= size_t(curves)) return;
    uint32_t begin = offsets[curve], end = offsets[curve + 1];
    float3 root = roots[curve];
    if (begin >= end || end > uint32_t(points) || !Finite(root.x) ||
        !Finite(root.y) || !Finite(root.z)) SetError(error, kBadOffsets);
}

__global__ void ApplyDeformation(
    const float3 *input, const float3 *warped, const uint32_t *offsets,
    const float3 *targets, int curves, float legacyGroom,
    const float *legacyPrimitive, const float *legacyPoint,
    ScalarField mask, BoolField enabled,
    BoolField lockRoots,
    float3 *output, int *error) {
    int curve = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (curve >= curves) return;
    uint32_t begin = offsets[curve], end = offsets[curve + 1];
    bool groomEnabled = true;
    if (!ReadBool(enabled, size_t(curve), begin, &groomEnabled, error)) return;
    bool lock = true;
    if (!ReadBool(lockRoots, size_t(curve), begin, &lock, error)) return;
    float primitive = 1.0f;
    if (legacyPrimitive) {
        primitive = legacyPrimitive[curve];
        if (!Finite(primitive)) { SetError(error, kNonFinite); return; }
    }
    if (!Finite(legacyGroom)) { SetError(error, kNonFinite); return; }
    float3 correction = make_float3(0, 0, 0);
    if (lock) {
        correction = make_float3(targets[curve].x - warped[begin].x,
                                 targets[curve].y - warped[begin].y,
                                 targets[curve].z - warped[begin].z);
        if (!Finite(correction.x) || !Finite(correction.y) ||
            !Finite(correction.z)) { SetError(error, kNonFinite); return; }
    }
    for (uint32_t point = begin; point < end; ++point) {
        float3 source = input[point], evaluated = warped[point];
        if (!Finite(source.x) || !Finite(source.y) || !Finite(source.z) ||
            !Finite(evaluated.x) || !Finite(evaluated.y) ||
            !Finite(evaluated.z)) { SetError(error, kNonFinite); continue; }
        float maskValue = 1.0f;
        if (!ReadScalar(mask, size_t(curve), point, &maskValue, error)) continue;
        float pointEnvelope = 1.0f;
        if (legacyPoint) {
            pointEnvelope = legacyPoint[point];
            if (!Finite(pointEnvelope)) { SetError(error, kNonFinite); continue; }
        }
        // usdGen:mask IS the operator envelope (02 §2.13).
        float envelope = legacyGroom * primitive * pointEnvelope * maskValue;
        if (!Finite(envelope)) { SetError(error, kNonFinite); continue; }
        envelope = fminf(1.0f, fmaxf(0.0f, envelope));
        float3 result = source;
        if (groomEnabled && envelope > 0.0f) {
            float3 destination = lock
                ? make_float3(evaluated.x + correction.x,
                              evaluated.y + correction.y,
                              evaluated.z + correction.z)
                : evaluated;
            result = make_float3(source.x + (destination.x - source.x) * envelope,
                                 source.y + (destination.y - source.y) * envelope,
                                 source.z + (destination.z - source.z) * envelope);
        }
        if (!Finite(result.x) || !Finite(result.y) || !Finite(result.z)) {
            SetError(error, kNonFinite); continue;
        }
        output[point] = result;
    }
}

bool IsDomain(expr::Domain domain) {
    return domain == expr::Domain::Groom || domain == expr::Domain::Primitive ||
           domain == expr::Domain::Point;
}

RbfStatus ValidateScalar(ScalarField field, DeviceCurveGeometryView geometry) {
    if (!IsDomain(field.domain)) return RbfStatus::InvalidArgument;
    size_t expected = field.domain == expr::Domain::Groom ? 1 :
        (field.domain == expr::Domain::Primitive ? geometry.curveCount : geometry.pointCount);
    if (!field.data) {
        if (field.count != 0 || field.domain != expr::Domain::Groom)
            return RbfStatus::InvalidArgument;
        if (!std::isfinite(field.literal)) return RbfStatus::NonFiniteInput;
        return field.literal < 0.0f || field.literal > 1.0f
            ? RbfStatus::InvalidArgument : RbfStatus::Ok;
    }
    return field.count == expected &&
                   field.count <= size_t(std::numeric_limits<int>::max())
               ? RbfStatus::Ok : RbfStatus::InvalidArgument;
}

RbfStatus ValidateBool(BoolField field, DeviceCurveGeometryView geometry,
                       bool allowPrimitive) {
    if (!IsDomain(field.domain) ||
        (allowPrimitive && field.domain == expr::Domain::Point) ||
        (!allowPrimitive && field.domain != expr::Domain::Groom))
        return RbfStatus::InvalidArgument;
    size_t expected = field.domain == expr::Domain::Groom ? 1 :
        (field.domain == expr::Domain::Primitive ? geometry.curveCount : geometry.pointCount);
    if (!field.data)
        return field.count == 0 && field.domain == expr::Domain::Groom
            ? RbfStatus::Ok : RbfStatus::InvalidArgument;
    return field.count == expected &&
                   field.count <= size_t(std::numeric_limits<int>::max())
               ? RbfStatus::Ok : RbfStatus::InvalidArgument;
}

RbfStatus CheckDevice(int *deviceIndex, cudaStream_t stream) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return RbfStatus::CudaError;
    int streamDevice = current;
    if (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
        return RbfStatus::CudaError;
    if (streamDevice != current || (*deviceIndex >= 0 && *deviceIndex != current))
        return RbfStatus::InvalidArgument;
    if (*deviceIndex < 0) *deviceIndex = current;
    return RbfStatus::Ok;
}

bool DevicePointer(const void* pointer, size_t count, int device) {
    if (!count) return true;
    cudaPointerAttributes attributes{};
    return pointer && cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
        attributes.type == cudaMemoryTypeDevice && attributes.device == device;
}

bool FreshProvenance(DeviceCurveGeometryView geometry, DeviceView<const float3> roots,
                     DeformParameters parameters, DeviceView<float3> output, int device) {
    auto scalar = [device](ScalarField field) { return !field.data || DevicePointer(field.data, field.count, device); };
    auto boolean = [device](BoolField field) { return !field.data || DevicePointer(field.data, field.count, device); };
    return DevicePointer(geometry.points.data, geometry.points.size, device) &&
        DevicePointer(geometry.curveOffsets.data, geometry.curveOffsets.size, device) &&
        DevicePointer(roots.data, roots.size, device) && DevicePointer(output.data, output.size, device) &&
        scalar(parameters.mask) && boolean(parameters.enabled) && boolean(parameters.lockRoots);
}

} // namespace anonymous

CudaRbfCurveDeformer::~CudaRbfCurveDeformer() {
    // A fresh terminal callback proves work later than ready_.  Never use the
    // legacy event to infer that a fresh D2H/copy callback has completed.
    if (freshUnproven_ || freshFailed_) {
        warped_.quarantine(); staged_.quarantine(); flags_.quarantine();
        ready_ = nullptr; freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
        return;
    }
    // DeviceBuffer::release uses the current CUDA context.  Select the owner
    // before synchronizing/destroying anything, and never free through a
    // foreign or lost context.  A deliberately leaked quarantine is safer
    // than handing an opaque allocation to the wrong context.
    const bool ownsBuffers = warped_.size() || staged_.size() || flags_.size() || ready_ || freshHostError_;
    int previous = -1;
    const bool selected = !ownsBuffers ||
        (cudaGetDevice(&previous) == cudaSuccess && deviceIndex_ >= 0 &&
         cudaSetDevice(deviceIndex_) == cudaSuccess);
    const bool synchronized = !ownsBuffers || freshUsed_ ||
        (selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess));
    if (!selected || !synchronized) {
        if (ownsBuffers) {
            warped_.quarantine();
            staged_.quarantine();
            flags_.quarantine();
            ready_ = nullptr;
            freshHostError_ = nullptr;
            freshHostErrorPermit_.Abandon();
        }
        if (selected && previous >= 0 && previous != deviceIndex_)
            cudaSetDevice(previous);
        return;
    }
    if (ready_) { cudaEventDestroy(ready_); ready_ = nullptr; }
    warped_.reset(0);
    staged_.reset(0);
    flags_.reset(0);
    if (freshHostError_) {
        if (cudaFreeHost(freshHostError_) == cudaSuccess) freshHostErrorPermit_.Release();
        else freshHostErrorPermit_.Abandon();
        freshHostError_ = nullptr;
    }
    if (previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
}

RbfStatus CudaRbfCurveDeformer::Deform(
    CudaRbfBinding& rbf, DeviceCurveGeometryView geometry,
    DeviceView<const float3> rootTargets, float groomEnvelope,
    DeviceView<const float> primitiveEnvelope,
    DeviceView<const float> pointEnvelope, DeviceView<float3> output,
    cudaStream_t stream) {
    // Preserve the legacy contract: its restPoints member is the source.
    if (geometry.restPoints.size != geometry.pointCount ||
        (geometry.pointCount && !geometry.restPoints.data))
        return RbfStatus::InvalidArgument;
    DeviceCurveGeometryView legacy = geometry;
    legacy.points = geometry.restPoints;
    DeformParameters parameters;
    return deformImpl(rbf, legacy, rootTargets, parameters, groomEnvelope,
                      primitiveEnvelope, pointEnvelope, output, stream);
}

RbfStatus CudaRbfCurveDeformer::Deform(
    CudaRbfBinding& rbf, DeviceCurveGeometryView geometry,
    DeviceView<const float3> rootTargets, DeformParameters parameters,
    DeviceView<float3> output, cudaStream_t stream) {
    return deformImpl(rbf, geometry, rootTargets, parameters, 1.0f, {}, {},
                      output, stream);
}

RbfStatus CudaRbfCurveDeformer::deformImpl(
    CudaRbfBinding& rbf, DeviceCurveGeometryView geometry,
    DeviceView<const float3> rootTargets, DeformParameters parameters,
    float groomEnvelope, DeviceView<const float> primitiveEnvelope,
    DeviceView<const float> pointEnvelope, DeviceView<float3> output,
    cudaStream_t stream) {
    if (pending_ || poisoned_ || freshPhase_ != FreshPhase::None || freshFailed_ || freshUsed_)
        return RbfStatus::InvalidArgument;
    legacyUsed_ = true;
    RbfStatus status = CheckDevice(&deviceIndex_, stream);
    if (status != RbfStatus::Ok) return status;
    if (geometry.curveCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<uint32_t>::max()) ||
        (geometry.curveCount == 0 && geometry.pointCount != 0) ||
        (geometry.curveCount && geometry.pointCount == 0) ||
        (geometry.curveCount && (!geometry.curveOffsets.data ||
         geometry.curveOffsets.size != geometry.curveCount + 1)) ||
        (!geometry.curveCount && geometry.curveOffsets.size != 0 &&
         geometry.curveOffsets.size != 1) ||
        (!geometry.curveCount && geometry.curveOffsets.size == 1 &&
         !geometry.curveOffsets.data) ||
        (geometry.pointCount && (!geometry.points.data ||
         geometry.points.size != geometry.pointCount)) ||
        (!geometry.pointCount && geometry.points.size != 0) ||
        (geometry.curveCount && (!rootTargets.data ||
         rootTargets.size != geometry.curveCount)) ||
        (!geometry.curveCount && rootTargets.size != 0) ||
        output.size != geometry.pointCount ||
        (geometry.pointCount && !output.data) ||
        (!geometry.pointCount && output.size != 0) ||
        (!primitiveEnvelope.data && primitiveEnvelope.size != 0) ||
        (primitiveEnvelope.data && primitiveEnvelope.size != geometry.curveCount) ||
        (!pointEnvelope.data && pointEnvelope.size != 0) ||
        (pointEnvelope.data && pointEnvelope.size != geometry.pointCount) ||
        !std::isfinite(groomEnvelope))
        return RbfStatus::InvalidArgument;
    status = ValidateScalar(parameters.mask, geometry);
    if (status != RbfStatus::Ok) return status;
    status = ValidateBool(parameters.enabled, geometry, false);
    if (status != RbfStatus::Ok) return status;
    status = ValidateBool(parameters.lockRoots, geometry, true);
    if (status != RbfStatus::Ok) return status;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return RbfStatus::CudaError;
    if (flags_.reset(1) != cudaSuccess || warped_.reset(geometry.pointCount) != cudaSuccess ||
        staged_.reset(geometry.pointCount) != cudaSuccess ||
        cudaMemsetAsync(flags_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return RbfStatus::CudaError;
    if (geometry.curveCount || geometry.curveOffsets.data) {
        const size_t blocks = geometry.curveCount
            ? (geometry.curveCount + 255) / 256 : 1;
        ValidateShape<<<blocks, 256, 0, stream>>>(
            geometry.curveOffsets.data, int(geometry.curveCount),
            int(geometry.pointCount), rootTargets.data, flags_.data());
        if (cudaGetLastError() != cudaSuccess) return RbfStatus::CudaError;
    }
    int shapeError = 0;
    if (cudaMemcpyAsync(&shapeError, flags_.data(), sizeof(shapeError),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess)
        return RbfStatus::CudaError;
    if (shapeError == kNonFinite) return RbfStatus::NonFiniteInput;
    if (shapeError) return RbfStatus::InvalidArgument;
    if (!geometry.pointCount) return rbf.Finish(stream);
    status = rbf.Evaluate(geometry.points, warped_.view(), stream);
    if (status != RbfStatus::Ok) return status;
    ApplyDeformation<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
        geometry.points.data, warped_.data(), geometry.curveOffsets.data,
        rootTargets.data, int(geometry.curveCount), groomEnvelope,
        primitiveEnvelope.data, pointEnvelope.data,
        parameters.mask, parameters.enabled, parameters.lockRoots,
        staged_.data(), flags_.data());
    if (cudaGetLastError() != cudaSuccess || cudaEventRecord(ready_, stream) != cudaSuccess)
        return RbfStatus::CudaError;
    output_ = output;
    outputCount_ = geometry.pointCount;
    pending_ = true;
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::Finish(CudaRbfBinding& rbf, cudaStream_t stream) {
    if (freshPhase_ != FreshPhase::None || freshUnproven_ || freshFailed_ || freshUsed_)
        return RbfStatus::InvalidArgument;
    RbfStatus status = CheckDevice(&deviceIndex_, stream);
    if (status != RbfStatus::Ok) return status;
    if (!pending_) return rbf.Finish(stream);
    int error = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&error, flags_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess)
        return RbfStatus::CudaError;
    pending_ = false;
    RbfStatus bindingStatus = rbf.Finish(stream);
    DeviceView<float3> destination = output_;
    output_ = {};
    outputCount_ = 0;
    if (bindingStatus != RbfStatus::Ok) return bindingStatus;
    if (error == kBadOffsets || error == kBadValue) return RbfStatus::InvalidArgument;
    if (error == kNonFinite) { poisoned_ = true; return RbfStatus::NonFiniteInput; }
    if (error != 0) return RbfStatus::CudaError;
    if (cudaMemcpyAsync(destination.data, staged_.data(),
                        staged_.size() * sizeof(float3), cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess || CudaFenceStream(stream) != cudaSuccess)
        return RbfStatus::CudaError;
    return RbfStatus::Ok;
}

bool CudaRbfCurveDeformer::HasUnprovenWork() const noexcept {
    return freshUnproven_ || freshFailed_;
}

RbfStatus CudaRbfCurveDeformer::BeginFreshShape(
    DeviceCurveGeometryView geometry, DeviceView<const float3> rootTargets,
    DeformParameters parameters, DeviceView<float3> output, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation) {
    if (pending_ || poisoned_ || legacyUsed_ || freshPhase_ != FreshPhase::None || freshFailed_)
        return RbfStatus::InvalidArgument;
    // Capture must be checked before stream-device queries or allocations.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return RbfStatus::InvalidArgument;
    RbfStatus status = CheckDevice(&deviceIndex_, stream);
    if (status != RbfStatus::Ok) return status;
    if (geometry.curveCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<uint32_t>::max()) ||
        (geometry.curveCount == 0 && geometry.pointCount != 0) ||
        (geometry.curveCount && geometry.pointCount == 0) ||
        (geometry.curveCount && (!geometry.curveOffsets.data ||
            geometry.curveOffsets.size != geometry.curveCount + 1)) ||
        (!geometry.curveCount && geometry.curveOffsets.size != 0 &&
            geometry.curveOffsets.size != 1) ||
        (!geometry.curveCount && geometry.curveOffsets.size == 1 &&
            !geometry.curveOffsets.data) ||
        (geometry.pointCount && (!geometry.points.data ||
            geometry.points.size != geometry.pointCount)) ||
        (!geometry.pointCount && geometry.points.size != 0) ||
        (geometry.curveCount && (!rootTargets.data ||
            rootTargets.size != geometry.curveCount)) ||
        (!geometry.curveCount && rootTargets.size != 0) ||
        output.size != geometry.pointCount ||
        (geometry.pointCount && !output.data) ||
        (!geometry.pointCount && output.size != 0))
        return RbfStatus::InvalidArgument;
    status = ValidateScalar(parameters.mask, geometry);
    if (status != RbfStatus::Ok) return status;
    status = ValidateBool(parameters.enabled, geometry, false);
    if (status != RbfStatus::Ok) return status;
    status = ValidateBool(parameters.lockRoots, geometry, true);
    if (status != RbfStatus::Ok) return status;
    if (!FreshProvenance(geometry, rootTargets, parameters, output, deviceIndex_))
        return RbfStatus::InvalidArgument;

    // Allocate every deformer-owned candidate before its first submission.
    if (flags_.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        warped_.reset(geometry.pointCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        staged_.reset(geometry.pointCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
        return RbfStatus::CudaError;
    if (!freshHostError_) {
        auto permit = TryReserveCudaExecutionBytes(sizeof(int), UsdGenExecutionResourceKind::Scratch, reservation);
        int* hostError = nullptr;
        if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&hostError), sizeof(int),
                                     cudaHostAllocDefault) != cudaSuccess)
            return RbfStatus::CudaError;
        freshHostError_ = hostError;
        freshHostErrorPermit_ = std::move(*permit);
    }
    freshGeometry_ = geometry;
    freshRoots_ = rootTargets;
    freshParameters_ = parameters;
    output_ = output;
    outputCount_ = geometry.pointCount;
    *freshHostError_ = std::numeric_limits<int>::min();
    freshPhase_ = FreshPhase::Shape;
    freshUsed_ = true;
    freshUnproven_ = true;
    freshArmed_ = false;
    if (cudaMemsetAsync(flags_.data(), 0, sizeof(int), stream) != cudaSuccess) {
        freshFailed_ = true;
        return RbfStatus::CudaError;
    }
    if (geometry.curveCount || geometry.curveOffsets.data) {
        size_t const blocks = geometry.curveCount ? (geometry.curveCount + 255) / 256 : 1;
        ValidateShape<<<blocks, 256, 0, stream>>>(geometry.curveOffsets.data,
            int(geometry.curveCount), int(geometry.pointCount), rootTargets.data, flags_.data());
        if (cudaGetLastError() != cudaSuccess) {
            freshFailed_ = true;
            return RbfStatus::CudaError;
        }
    }
    if (cudaMemcpyAsync(freshHostError_, flags_.data(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        freshFailed_ = true;
        return RbfStatus::CudaError;
    }
    freshArmed_ = true;
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::CommitFreshShape() {
    if (freshPhase_ != FreshPhase::Shape || !freshArmed_ || freshFailed_ || !freshHostError_)
        return RbfStatus::InvalidArgument;
    freshUnproven_ = false;
    freshArmed_ = false;
    int const error = *freshHostError_;
    if (error != 0) {
        freshPhase_ = FreshPhase::None;
        output_ = {}; outputCount_ = 0; freshGeometry_ = {}; freshRoots_ = {}; freshParameters_ = {};
        return error == kNonFinite ? RbfStatus::NonFiniteInput : RbfStatus::InvalidArgument;
    }
    freshPhase_ = FreshPhase::ShapeReady;
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::BeginFreshEvaluate(CudaRbfBinding& rbf, cudaStream_t stream,
                                                    UsdGenExecutionMemoryReservation* reservation) {
    if (freshPhase_ != FreshPhase::ShapeReady || freshFailed_) return RbfStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return RbfStatus::InvalidArgument;
    if (CheckDevice(&deviceIndex_, stream) != RbfStatus::Ok) return RbfStatus::InvalidArgument;
    if (!freshGeometry_.pointCount) {
        freshPhase_ = FreshPhase::EvaluateReady;
        return RbfStatus::Ok;
    }
    freshPhase_ = FreshPhase::Evaluate;
    freshUnproven_ = true;
    freshRbf_ = &rbf;
    RbfStatus const status = rbf.BeginFreshEvaluate(freshGeometry_.points, warped_.view(), stream, reservation);
    if (status != RbfStatus::Ok) {
        if (rbf.HasUnprovenWork()) freshFailed_ = true;
        else { freshPhase_ = FreshPhase::ShapeReady; freshUnproven_ = false; }
    }
    return status;
}

RbfStatus CudaRbfCurveDeformer::CommitFreshEvaluate(CudaRbfBinding& rbf) {
    if (freshPhase_ == FreshPhase::EvaluateReady && !freshGeometry_.pointCount)
        return RbfStatus::Ok;
    if (freshPhase_ != FreshPhase::Evaluate || freshFailed_ || freshRbf_ != &rbf)
        return RbfStatus::InvalidArgument;
    RbfStatus const status = rbf.CommitFreshEvaluate();
    if (status != RbfStatus::Ok && rbf.HasUnprovenWork()) {
        freshFailed_ = true;
        return status;
    }
    freshUnproven_ = false;
    freshRbf_ = nullptr;
    if (status != RbfStatus::Ok) {
        freshPhase_ = FreshPhase::None;
        output_ = {}; outputCount_ = 0; freshGeometry_ = {}; freshRoots_ = {}; freshParameters_ = {};
        return status;
    }
    freshPhase_ = FreshPhase::EvaluateReady;
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::BeginFreshApply(cudaStream_t stream) {
    if (freshPhase_ != FreshPhase::EvaluateReady || freshFailed_ || !freshHostError_)
        return RbfStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return RbfStatus::InvalidArgument;
    if (CheckDevice(&deviceIndex_, stream) != RbfStatus::Ok) return RbfStatus::InvalidArgument;
    *freshHostError_ = std::numeric_limits<int>::min();
    freshPhase_ = FreshPhase::Apply;
    freshUnproven_ = true;
    freshArmed_ = false;
    if (cudaMemsetAsync(flags_.data(), 0, sizeof(int), stream) != cudaSuccess) {
        freshFailed_ = true;
        return RbfStatus::CudaError;
    }
    auto const& geometry = freshGeometry_;
    if (geometry.curveCount)
        ApplyDeformation<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry.points.data, warped_.data(), geometry.curveOffsets.data, freshRoots_.data,
            int(geometry.curveCount), 1.0f, nullptr, nullptr,
            freshParameters_.mask, freshParameters_.enabled, freshParameters_.lockRoots,
            staged_.data(), flags_.data());
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(freshHostError_, flags_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        freshFailed_ = true;
        return RbfStatus::CudaError;
    }
    freshArmed_ = true;
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::CommitFreshApply() {
    if (freshPhase_ != FreshPhase::Apply || !freshArmed_ || freshFailed_ || !freshHostError_)
        return RbfStatus::InvalidArgument;
    freshUnproven_ = false;
    freshArmed_ = false;
    int const error = *freshHostError_;
    if (error != 0) {
        freshPhase_ = FreshPhase::None;
        output_ = {}; outputCount_ = 0; freshGeometry_ = {}; freshRoots_ = {}; freshParameters_ = {};
        return error == kNonFinite ? RbfStatus::NonFiniteInput : RbfStatus::InvalidArgument;
    }
    freshPhase_ = FreshPhase::ApplyReady;
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::BeginFreshCopy(cudaStream_t stream) {
    if (freshPhase_ != FreshPhase::ApplyReady || freshFailed_ ||
        (outputCount_ && !output_.data) || outputCount_ != staged_.size())
        return RbfStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return RbfStatus::InvalidArgument;
    if (CheckDevice(&deviceIndex_, stream) != RbfStatus::Ok) return RbfStatus::InvalidArgument;
    freshPhase_ = FreshPhase::Copy;
    freshUnproven_ = true;
    if (staged_.size() && cudaMemcpyAsync(output_.data, staged_.data(), staged_.size() * sizeof(float3),
                                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        freshFailed_ = true;
        return RbfStatus::CudaError;
    }
    return RbfStatus::Ok;
}

RbfStatus CudaRbfCurveDeformer::CommitFreshFinish() {
    if (freshPhase_ != FreshPhase::Copy || freshFailed_) return RbfStatus::InvalidArgument;
    freshUnproven_ = false;
    freshPhase_ = FreshPhase::None;
    output_ = {}; outputCount_ = 0; freshGeometry_ = {}; freshRoots_ = {}; freshParameters_ = {};
    return RbfStatus::Ok;
}

}} // namespace usdGen::gpu
