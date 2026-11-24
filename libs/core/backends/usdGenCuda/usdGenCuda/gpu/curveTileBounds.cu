#include "curveTileBounds.h"
#include "cudaCompat.h"

#include <math_constants.h>
#include <climits>
#include <cfloat>
#include <cmath>

namespace usdGen::gpu {
namespace {

constexpr uint32_t kThreads = 256;

__device__ void Fail(uint32_t* status) { atomicCAS(status, 0u, 1u); }

__device__ uint32_t Load(uint32_t const* status) {
    return atomicAdd(const_cast<uint32_t*>(status), 0u);
}

__device__ bool Finite(float3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ float3 Min(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ float3 Max(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// Directed conversion preserves an outward finite bound without turning an
// exactly representable FLT_MAX endpoint into infinity.
__device__ bool LowerFloat(double value, float* result) {
    if (!isfinite(value) || value < -double(FLT_MAX) || value > double(FLT_MAX))
        return false;
    float rounded = __double2float_rd(value);
    if (!isfinite(rounded)) return false;
    *result = rounded;
    return true;
}

__device__ bool UpperFloat(double value, float* result) {
    if (!isfinite(value) || value < -double(FLT_MAX) || value > double(FLT_MAX))
        return false;
    float rounded = __double2float_ru(value);
    if (!isfinite(rounded)) return false;
    *result = rounded;
    return true;
}

__device__ bool FinalizeBounds(CurveTileBoundsBasis basis, float3 minimum,
    float3 maximum, float maxWidth, float3* resultMinimum,
    float3* resultMaximum) {
    if (!Finite(minimum) || !Finite(maximum) || !isfinite(maxWidth) ||
        maxWidth < 0.0f)
        return false;

    // Uniform Catmull-Rom's negative weights sum to -u(1-u)/2, whose
    // magnitude is at most 1/8. Its signed width interpolation has absolute
    // maximum at most 9/8 of the nonnegative CV maximum. B-spline and linear
    // bases are convex, so only their renderer width/2 needs expansion.
    double const widthScale = basis == CurveTileBoundsBasis::CatmullRom
        ? 9.0 / 16.0 : 1.0 / 2.0;
    double const widthPad = __dmul_ru(double(maxWidth), widthScale);
    double const positionScale = basis == CurveTileBoundsBasis::CatmullRom
        ? 1.0 / 8.0 : 0.0;
    double const mn[3] = {minimum.x, minimum.y, minimum.z};
    double const mx[3] = {maximum.x, maximum.y, maximum.z};
    float lo[3], hi[3];
    for (int axis = 0; axis != 3; ++axis) {
        double const range = __dsub_ru(mx[axis], mn[axis]);
        double const positionPad = __dmul_ru(range, positionScale);
        double const lower = __dsub_rd(__dsub_rd(mn[axis], positionPad), widthPad);
        double const upper = __dadd_ru(__dadd_ru(mx[axis], positionPad), widthPad);
        if (!LowerFloat(lower, &lo[axis]) || !UpperFloat(upper, &hi[axis]))
            return false;
    }
    *resultMinimum = make_float3(lo[0], lo[1], lo[2]);
    *resultMaximum = make_float3(hi[0], hi[1], hi[2]);
    return true;
}

__global__ void ValidateKernel(CurveTileBoundsInput input, uint32_t* status) {
    size_t const tile = blockIdx.x;
    CurveTileSpan const span = input.spans.data[tile];
    uint64_t const first = span.firstPoint;
    uint64_t const count = span.pointCount;
    bool const inRange = first + count >= first &&
        first + count <= input.points.size && first + count <= input.widths.size;
    if (!inRange) {
        if (threadIdx.x == 0) Fail(status);
        return;
    }
    for (uint64_t point = threadIdx.x; point < count; point += blockDim.x) {
        float3 const position = input.points.data[first + point];
        float const width = input.widths.data[first + point];
        if (!Finite(position) || !isfinite(width) || width < 0.0f) Fail(status);
    }
}

__global__ void ReduceKernel(CurveTileBoundsInput input,
    CurveTileBoundsWorkspace workspace, uint32_t const* status) {
    size_t const tile = blockIdx.x;
    if (*status) return;
    CurveTileSpan const span = input.spans.data[tile];
    if (span.pointCount == 0) {
        if (threadIdx.x == 0) {
            workspace.scratch.data[tile] = {make_float3(0.0f, 0.0f, 0.0f),
                                            make_float3(0.0f, 0.0f, 0.0f), 0.0f};
        }
        return;
    }

    float3 minimum = make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F);
    float3 maximum = make_float3(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
    float maxWidth = 0.0f;
    size_t const first = span.firstPoint;
    for (size_t point = threadIdx.x; point < span.pointCount; point += blockDim.x) {
        float3 const position = input.points.data[first + point];
        float const width = input.widths.data[first + point];
        minimum = Min(minimum, position);
        maximum = Max(maximum, position);
        maxWidth = fmaxf(maxWidth, width);
    }

    __shared__ float3 sharedMinimum[kThreads];
    __shared__ float3 sharedMaximum[kThreads];
    __shared__ float sharedWidth[kThreads];
    sharedMinimum[threadIdx.x] = minimum;
    sharedMaximum[threadIdx.x] = maximum;
    sharedWidth[threadIdx.x] = maxWidth;
    __syncthreads();
    for (uint32_t stride = kThreads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedMinimum[threadIdx.x] = Min(sharedMinimum[threadIdx.x],
                                               sharedMinimum[threadIdx.x + stride]);
            sharedMaximum[threadIdx.x] = Max(sharedMaximum[threadIdx.x],
                                               sharedMaximum[threadIdx.x + stride]);
            sharedWidth[threadIdx.x] = fmaxf(sharedWidth[threadIdx.x],
                                               sharedWidth[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        workspace.scratch.data[tile] = {sharedMinimum[0], sharedMaximum[0],
                                        sharedWidth[0]};
    }
}

__global__ void FinalizeKernel(CurveTileBoundsOptions options, size_t tileCount,
    CurveTileBoundsWorkspace workspace, uint32_t* status) {
    size_t const tile = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile >= tileCount || Load(status)) return;
    CurveTileBoundsScratch& scratch = workspace.scratch.data[tile];
    float3 minimum, maximum;
    if (!FinalizeBounds(options.basis, scratch.minimum, scratch.maximum,
            scratch.maximumWidth, &minimum, &maximum)) {
        Fail(status);
        return;
    }
    scratch.minimum = minimum;
    scratch.maximum = maximum;
}

// This pass runs after every finalizer, so one invalid tile prevents all
// externally visible writes.
__global__ void PublishKernel(size_t tileCount, CurveTileBoundsWorkspace workspace,
    CurveTileBoundsOutput output) {
    if (output.status.data[0] != 0) return;
    size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = size_t(gridDim.x) * blockDim.x;
    for (size_t tile = index; tile < tileCount; tile += stride) {
        output.minimums.data[tile] = workspace.scratch.data[tile].minimum;
        output.maximums.data[tile] = workspace.scratch.data[tile].maximum;
    }
}

bool Valid(CurveTileBoundsBasis basis) {
    return basis == CurveTileBoundsBasis::Linear ||
        basis == CurveTileBoundsBasis::BSpline ||
        basis == CurveTileBoundsBasis::CatmullRom;
}

} // namespace

cudaError_t BuildCurveTileBounds(CurveTileBoundsOptions options,
    CurveTileBoundsInput input, CurveTileBoundsWorkspace workspace,
    CurveTileBoundsOutput output, cudaStream_t stream) {
    if (!Valid(options.basis) || input.points.size != input.widths.size ||
        (input.points.size && (!input.points.data || !input.widths.data)) ||
        (input.spans.size && !input.spans.data) ||
        workspace.scratch.size < input.spans.size ||
        (input.spans.size && !workspace.scratch.data) ||
        output.minimums.size < input.spans.size ||
        output.maximums.size < input.spans.size ||
        (input.spans.size && (!output.minimums.data || !output.maximums.data)) ||
        !output.status.data || output.status.size != 1 ||
        input.spans.size > size_t(INT_MAX))
        return cudaErrorInvalidValue;

    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (stream) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        status = cudaStreamIsCapturing(stream, &capture);
        if (status != cudaSuccess) return status;
        // cudaStreamGetDevice is not capture-safe on every supported CUDA
        // runtime. During capture the caller's active execution device owns
        // the stream; outside capture, reject a cross-device stream eagerly.
        if (capture != cudaStreamCaptureStatusNone) {
            // No stream-introspection call is permitted while capturing.
        } else {
            int streamDevice = -1;
            status = cudaStreamGetDevice(stream, &streamDevice);
            if (status != cudaSuccess) return status;
            if (streamDevice != device) return cudaErrorInvalidResourceHandle;
        }
    }
    status = cudaMemsetAsync(output.status.data, 0, sizeof(uint32_t), stream);
    if (status != cudaSuccess) return status;
    if (!input.spans.size) return cudaSuccess;
    unsigned const blocks = static_cast<unsigned>(input.spans.size);
    ValidateKernel<<<blocks, kThreads, 0, stream>>>(input, output.status.data);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    ReduceKernel<<<blocks, kThreads, 0, stream>>>(input, workspace, output.status.data);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    unsigned const finalizeBlocks =
        static_cast<unsigned>((input.spans.size + kThreads - 1) / kThreads);
    FinalizeKernel<<<finalizeBlocks, kThreads, 0, stream>>>(options,
        input.spans.size, workspace, output.status.data);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    PublishKernel<<<finalizeBlocks, kThreads, 0, stream>>>(input.spans.size,
        workspace, output);
    return cudaGetLastError();
}

} // namespace usdGen::gpu
