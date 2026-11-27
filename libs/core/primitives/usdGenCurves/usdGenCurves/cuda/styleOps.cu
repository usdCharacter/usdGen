#include "usdGenCurves/cuda/styleOps.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace usdGen::gpu {
namespace {

constexpr int kBadOffsets = 1;
constexpr int kNonFinite = 2;
constexpr int kBadValue = 3;

__device__ void SetError(int *error, int code) {
    atomicCAS(error, 0, code);
}

__device__ int LoadError(const int *error) {
    // The diagnostic is written with atomics by peer threads. An atomic load
    // is required here too; a plain read would race those writes.
    return atomicAdd(const_cast<int *>(error), 0);
}

__device__ bool Finite(float3 const &v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

// This is deliberately a separate pass. Every operation waits for it in the
// same stream and exits before reading an offset-derived point index if a
// malformed offset was found.
__global__ void ValidateOffsets(const uint32_t *offsets, size_t curves,
                                size_t points, int *error) {
    if (blockIdx.x == 0 && blockDim.x != 0 && threadIdx.x == 0 &&
        (offsets[0] != 0 || offsets[curves] != points))
        SetError(error, kBadOffsets);
    size_t c = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= curves) return;
    const uint32_t begin = offsets[c];
    const uint32_t end = offsets[c + 1];
    if (begin >= end || end > points) SetError(error, kBadOffsets);
}

__device__ bool ReadField(ScalarField field, size_t curve, size_t point,
                          float *value, int *error) {
    if (!field.data) {
        *value = field.literal;
    } else {
        size_t index = 0;
        if (field.domain == expr::Domain::Primitive) index = curve;
        else if (field.domain == expr::Domain::Point) index = point;
        else if (field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue);
            return false;
        }
        // Host validation guarantees this bound. Keep the device check as a
        // final guard for callers constructing a view concurrently.
        if (!field.data || index >= field.count) {
            SetError(error, kBadValue);
            return false;
        }
        *value = field.data[index];
    }
    if (!isfinite(*value)) {
        SetError(error, kNonFinite);
        return false;
    }
    return true;
}

__device__ bool ReadHairT(const float *hairT, size_t point, float fallback,
                          float *t, int *error) {
    *t = hairT ? hairT[point] : fallback;
    if (!isfinite(*t)) {
        SetError(error, kNonFinite);
        return false;
    }
    if (*t < 0.0f || *t > 1.0f) {
        SetError(error, kBadValue);
        return false;
    }
    return true;
}

__global__ void WidthRampKernel(DeviceCurveGeometryView geometry,
                                ScalarField root, ScalarField tip,
                                const float *hairT, float *output, int *error) {
    size_t curve = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (curve >= geometry.curveCount || LoadError(error) != 0) return;
    const uint32_t begin = geometry.curveOffsets.data[curve];
    const uint32_t end = geometry.curveOffsets.data[curve + 1];
    const float denominator = float(end - begin - 1u);
    for (uint32_t point = begin; point < end; ++point) {
        const float fallback = denominator > 0.0f
            ? float(point - begin) / denominator : 0.0f;
        float t = 0.0f;
        if (!ReadHairT(hairT, point, fallback, &t, error)) continue;
        float rootWidth = 0.0f, tipWidth = 0.0f;
        if (!ReadField(root, curve, point, &rootWidth, error) ||
            !ReadField(tip, curve, point, &tipWidth, error)) continue;
        if (rootWidth < 0.0f || tipWidth < 0.0f) {
            SetError(error, kBadValue);
            continue;
        }
        const float value = rootWidth + (tipWidth - rootWidth) * t;
        if (!isfinite(value)) SetError(error, kNonFinite);
        else output[point] = value;
    }
}

__global__ void LengthKernel(DeviceCurveGeometryView geometry, ScalarField scale,
                             float3 *output, int *error) {
    size_t curve = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (curve >= geometry.curveCount || LoadError(error) != 0) return;
    const uint32_t begin = geometry.curveOffsets.data[curve];
    const uint32_t end = geometry.curveOffsets.data[curve + 1];
    const float3 root = geometry.points.data[begin];
    if (!Finite(root)) {
        SetError(error, kNonFinite);
        return;
    }
    for (uint32_t point = begin; point < end; ++point) {
        const float3 p = geometry.points.data[point];
        float factor = 0.0f;
        if (!Finite(p)) {
            SetError(error, kNonFinite);
            continue;
        }
        if (!ReadField(scale, curve, point, &factor, error)) continue;
        const float3 value = make_float3(
            root.x + (p.x - root.x) * factor,
            root.y + (p.y - root.y) * factor,
            root.z + (p.z - root.z) * factor);
        if (!Finite(value)) SetError(error, kNonFinite);
        else output[point] = value;
    }
}

__global__ void GrowKernel(DeviceCurveGeometryView geometry,
                           DeviceView<const float3> roots,
                           DeviceView<const float3> normals, ScalarField length,
                           const float *hairT, float3 *output, int *error) {
    size_t curve = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (curve >= geometry.curveCount || LoadError(error) != 0) return;
    const uint32_t begin = geometry.curveOffsets.data[curve];
    const uint32_t end = geometry.curveOffsets.data[curve + 1];
    const float3 root = roots.data[curve];
    const float3 inputNormal = normals.data[curve];
    if (!Finite(root) || !Finite(inputNormal)) {
        SetError(error, kNonFinite);
        return;
    }
    const float normalLength = sqrtf(inputNormal.x * inputNormal.x +
                                     inputNormal.y * inputNormal.y +
                                     inputNormal.z * inputNormal.z);
    if (!isfinite(normalLength) || normalLength == 0.0f) {
        SetError(error, kBadValue);
        return;
    }
    const float3 normal = make_float3(inputNormal.x / normalLength,
                                     inputNormal.y / normalLength,
                                     inputNormal.z / normalLength);
    const float denominator = float(end - begin - 1u);
    for (uint32_t point = begin; point < end; ++point) {
        const float fallback = denominator > 0.0f
            ? float(point - begin) / denominator : 0.0f;
        float t = 0.0f, amount = 0.0f;
        if (!ReadHairT(hairT, point, fallback, &t, error) ||
            !ReadField(length, curve, point, &amount, error)) continue;
        const float distance = amount * t;
        const float3 value = make_float3(root.x + normal.x * distance,
                                         root.y + normal.y * distance,
                                         root.z + normal.z * distance);
        if (!Finite(value)) SetError(error, kNonFinite);
        else output[point] = value;
    }
}

bool IsDomain(expr::Domain domain) {
    return domain == expr::Domain::Groom || domain == expr::Domain::Primitive ||
           domain == expr::Domain::Point;
}

StyleStatus DecodeError(int code) {
    if (code == kBadOffsets) return StyleStatus::InvalidArgument;
    if (code == kNonFinite) return StyleStatus::NonFiniteInput;
    if (code == kBadValue) return StyleStatus::InvalidValue;
    return StyleStatus::CudaError;
}

} // namespace

CudaStyleOps::~CudaStyleOps() {
    if (ready_) {
        // shutdown wait: destructor.
        cudaEventSynchronize(ready_);
        cudaEventDestroy(ready_);
    }
}

StyleStatus CudaStyleOps::validateField(ScalarField field,
                                        DeviceCurveGeometryView geometry) const {
    if (!IsDomain(field.domain)) return StyleStatus::InvalidArgument;
    if (!field.data) {
        if (field.domain != expr::Domain::Groom || field.count != 0)
            return StyleStatus::InvalidArgument;
        return std::isfinite(field.literal)
            ? StyleStatus::Ok : StyleStatus::NonFiniteInput;
    }
    const size_t expected = field.domain == expr::Domain::Groom
        ? 1 : (field.domain == expr::Domain::Primitive
            ? geometry.curveCount : geometry.pointCount);
    if (field.count != expected ||
        field.count > size_t(std::numeric_limits<int>::max()))
        return StyleStatus::InvalidArgument;
    return StyleStatus::Ok;
}

StyleStatus CudaStyleOps::begin(DeviceCurveGeometryView geometry, size_t outputSize,
                                const void *output, cudaStream_t stream,
                                DeviceView<const float> hairT, bool vectorOutput,
                                bool requirePoints) {
    if (pending_) return StyleStatus::InvalidArgument;
    if (geometry.curveCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<uint32_t>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<int>::max()) ||
        (geometry.curveCount == 0 && geometry.pointCount != 0) ||
        (requirePoints && geometry.pointCount && !geometry.points.data) ||
        (requirePoints && geometry.points.size != geometry.pointCount) ||
        outputSize != geometry.pointCount ||
        (geometry.pointCount && !output) ||
        (!hairT.data && hairT.size != 0) ||
        (hairT.data && hairT.size != geometry.pointCount))
        return StyleStatus::InvalidArgument;
    if (geometry.curveCount != 0 &&
        (!geometry.curveOffsets.data ||
         geometry.curveOffsets.size != geometry.curveCount + 1))
        return StyleStatus::InvalidArgument;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return StyleStatus::CudaError;
    if (error_.reset(1) != cudaSuccess ||
        (vectorOutput ? vectorStaging_.reset(geometry.pointCount) != cudaSuccess
                      : scalarStaging_.reset(geometry.pointCount) != cudaSuccess) ||
        cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return StyleStatus::CudaError;
    pointCount_ = geometry.pointCount;
    vectorPending_ = vectorOutput;
    return StyleStatus::Ok;
}

StyleStatus CudaStyleOps::WidthRamp(DeviceCurveGeometryView geometry,
                                    ScalarField root, ScalarField tip,
                                    DeviceView<float> output, cudaStream_t stream,
                                    DeviceView<const float> hairT) {
    if (pending_) return StyleStatus::InvalidArgument;
    StyleStatus status = validateField(root, geometry);
    if (status != StyleStatus::Ok) return status;
    status = validateField(tip, geometry);
    if (status != StyleStatus::Ok) return status;
    status = begin(geometry, output.size, output.data, stream, hairT, false, false);
    if (status != StyleStatus::Ok) return status;
    if (geometry.curveCount) {
        ValidateOffsets<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry.curveOffsets.data, geometry.curveCount, geometry.pointCount,
            error_.data());
        if (cudaGetLastError() != cudaSuccess) return StyleStatus::CudaError;
        WidthRampKernel<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry, root, tip, hairT.data, scalarStaging_.data(), error_.data());
        if (cudaGetLastError() != cudaSuccess) return StyleStatus::CudaError;
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) return StyleStatus::CudaError;
    scalarOutput_ = output;
    pending_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaStyleOps::WidthRamp(DeviceCurveGeometryView geometry,
                                    ScalarField root, ScalarField tip,
                                    DeviceView<const float> hairT,
                                    DeviceView<float> output, cudaStream_t stream) {
    return WidthRamp(geometry, root, tip, output, stream, hairT);
}

StyleStatus CudaStyleOps::WidthRamp(DeviceCurveGeometryView geometry, float root, float tip,
                                    DeviceView<float> output, cudaStream_t stream,
                                    DeviceView<const float> hairT) {
    return WidthRamp(geometry, ScalarField::Literal(root), ScalarField::Literal(tip),
                     output, stream, hairT);
}

StyleStatus CudaStyleOps::Length(DeviceCurveGeometryView geometry, ScalarField scale,
                                 DeviceView<float3> output, cudaStream_t stream) {
    if (pending_) return StyleStatus::InvalidArgument;
    StyleStatus status = validateField(scale, geometry);
    if (status != StyleStatus::Ok) return status;
    status = begin(geometry, output.size, output.data, stream, {}, true, true);
    if (status != StyleStatus::Ok) return status;
    if (geometry.curveCount) {
        ValidateOffsets<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry.curveOffsets.data, geometry.curveCount, geometry.pointCount,
            error_.data());
        if (cudaGetLastError() != cudaSuccess) return StyleStatus::CudaError;
        LengthKernel<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry, scale, vectorStaging_.data(), error_.data());
        if (cudaGetLastError() != cudaSuccess) return StyleStatus::CudaError;
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) return StyleStatus::CudaError;
    vectorOutput_ = output;
    pending_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaStyleOps::Length(DeviceCurveGeometryView geometry, float scale,
                                 DeviceView<float3> output, cudaStream_t stream) {
    return Length(geometry, ScalarField::Literal(scale), output, stream);
}

StyleStatus CudaStyleOps::Grow(DeviceCurveGeometryView geometry,
                               DeviceView<const float3> roots,
                               DeviceView<const float3> normals, ScalarField length,
                               DeviceView<float3> output, cudaStream_t stream,
                               DeviceView<const float> hairT) {
    if (pending_) return StyleStatus::InvalidArgument;
    if (roots.size != geometry.curveCount || normals.size != geometry.curveCount ||
        (geometry.curveCount != 0 && (!roots.data || !normals.data)))
        return StyleStatus::InvalidArgument;
    StyleStatus status = validateField(length, geometry);
    if (status != StyleStatus::Ok) return status;
    status = begin(geometry, output.size, output.data, stream, hairT, true, false);
    if (status != StyleStatus::Ok) return status;
    if (geometry.curveCount) {
        ValidateOffsets<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry.curveOffsets.data, geometry.curveCount, geometry.pointCount,
            error_.data());
        if (cudaGetLastError() != cudaSuccess) return StyleStatus::CudaError;
        GrowKernel<<<(geometry.curveCount + 255) / 256, 256, 0, stream>>>(
            geometry, roots, normals, length, hairT.data, vectorStaging_.data(),
            error_.data());
        if (cudaGetLastError() != cudaSuccess) return StyleStatus::CudaError;
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) return StyleStatus::CudaError;
    vectorOutput_ = output;
    pending_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaStyleOps::Grow(DeviceCurveGeometryView geometry,
                               DeviceView<const float3> roots,
                               DeviceView<const float3> normals, ScalarField length,
                               DeviceView<const float> hairT,
                               DeviceView<float3> output, cudaStream_t stream) {
    return Grow(geometry, roots, normals, length, output, stream, hairT);
}

StyleStatus CudaStyleOps::Grow(DeviceCurveGeometryView geometry,
                               DeviceView<const float3> roots,
                               DeviceView<const float3> normals, float length,
                               DeviceView<float3> output, cudaStream_t stream,
                               DeviceView<const float> hairT) {
    return Grow(geometry, roots, normals, ScalarField::Literal(length), output,
                stream, hairT);
}

StyleStatus CudaStyleOps::Noise(DeviceCurveGeometryView, float, uint32_t,
                                DeviceView<float3>, cudaStream_t) {
    if (pending_) return StyleStatus::InvalidArgument;
    return StyleStatus::NotSupported;
}

StyleStatus CudaStyleOps::finishPublication(cudaStream_t stream) {
    if (pointCount_ == 0) {
        // shutdown wait: test hook (CudaStyleOps has no cook caller; kernel tests only).
        if (CudaFenceStream(stream) != cudaSuccess) return StyleStatus::CudaError;
        pending_ = false;
        scalarOutput_ = {};
        vectorOutput_ = {};
        return StyleStatus::Ok;
    }
    cudaError_t result = cudaSuccess;
    if (vectorPending_) {
        result = cudaMemcpyAsync(vectorOutput_.data, vectorStaging_.data(),
                                 pointCount_ * sizeof(float3), cudaMemcpyDeviceToDevice,
                                 stream);
    } else {
        result = cudaMemcpyAsync(scalarOutput_.data, scalarStaging_.data(),
                                 pointCount_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                 stream);
    }
    // shutdown wait: test hook (CudaStyleOps has no cook caller; kernel tests only).
    if (result != cudaSuccess || CudaFenceStream(stream) != cudaSuccess)
        return StyleStatus::CudaError;
    pending_ = false;
    scalarOutput_ = {};
    vectorOutput_ = {};
    pointCount_ = 0;
    return StyleStatus::Ok;
}

StyleStatus CudaStyleOps::Finish(cudaStream_t stream) {
    if (!pending_) return StyleStatus::InvalidArgument;
    int code = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&code, error_.data(), sizeof(code), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        // shutdown wait: test hook (CudaStyleOps has no cook caller; kernel tests only).
        CudaFenceStream(stream) != cudaSuccess)
        return StyleStatus::CudaError;
    if (code != 0) {
        pending_ = false;
        scalarOutput_ = {};
        vectorOutput_ = {};
        pointCount_ = 0;
        return DecodeError(code);
    }
    return finishPublication(stream);
}

} // namespace usdGen::gpu
