#include "curveTiles.h"
#include "cudaCompat.h"

#include <algorithm>
#include <limits>

namespace usdGen::gpu {
namespace {

constexpr uint32_t kThreads = 256;

__device__ void Fail(uint32_t* status) { atomicCAS(status, 0u, 1u); }

__device__ size_t LowerBound(DeviceView<const uint64_t> values, uint64_t key) {
    size_t first = 0;
    size_t count = values.size;
    while (count) {
        size_t const step = count / 2;
        size_t const probe = first + step;
        if (values.data[probe] < key) {
            first = probe + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

__host__ __device__ bool ValidOrder(CurveTileOrder order) {
    return order == CurveTileOrder::SortedSurvivorSubset ||
        order == CurveTileOrder::IdentityCaptureOrder ||
        order == CurveTileOrder::CaptureOrderSurvivorSubset;
}

__device__ size_t CaptureOrdinal(CurveTileInput const& input, uint64_t id) {
    size_t const found = LowerBound(input.sortedCaptureStableIds, id);
    if (found == input.captureCurveCount || input.sortedCaptureStableIds.data[found] != id)
        return input.captureCurveCount;
    return input.sortedCaptureOrdinals.data[found];
}

__device__ size_t SurvivorOrdinalLowerBound(CurveTileInput const& input, size_t ordinal) {
    size_t first = 0, count = input.survivorCurveCount;
    while (count) {
        size_t const step = count / 2, probe = first + step;
        if (CaptureOrdinal(input, input.survivorStableIds.data[probe]) < ordinal) {
            first = probe + 1; count -= step + 1;
        } else count = step;
    }
    return first;
}

__global__ void ValidateKernel(CurveTileInput input, uint32_t* status) {
    size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = size_t(blockDim.x) * gridDim.x;
    if (index == 0) {
        if (input.survivorCurveOffsets.data[0] != 0 ||
            input.survivorCurveOffsets.data[input.survivorCurveCount] !=
                input.survivorPointCount) {
            Fail(status);
        }
    }
    if (input.order == CurveTileOrder::SortedSurvivorSubset) {
        for (size_t curve = index; curve < input.captureCurveCount; curve += stride) {
            if (curve && input.captureStableIds.data[curve] <=
                             input.captureStableIds.data[curve - 1]) {
                Fail(status);
            }
        }
    } else if (input.order == CurveTileOrder::CaptureOrderSurvivorSubset) {
        for (size_t curve = index; curve < input.captureCurveCount; curve += stride) {
            uint64_t const id = input.sortedCaptureStableIds.data[curve];
            uint32_t const ordinal = input.sortedCaptureOrdinals.data[curve];
            if ((curve && id <= input.sortedCaptureStableIds.data[curve - 1]) ||
                ordinal >= input.captureCurveCount || input.captureStableIds.data[ordinal] != id)
                Fail(status);
        }
    }
    for (size_t curve = index; curve < input.survivorCurveCount; curve += stride) {
        uint32_t const first = input.survivorCurveOffsets.data[curve];
        uint32_t const last = input.survivorCurveOffsets.data[curve + 1];
        if (last <= first || last > input.survivorPointCount) Fail(status);
        uint64_t const id = input.survivorStableIds.data[curve];
        if (input.order == CurveTileOrder::IdentityCaptureOrder) {
            if (input.captureCurveCount != input.survivorCurveCount ||
                id != input.captureStableIds.data[curve]) {
                Fail(status);
            }
        } else if (input.order == CurveTileOrder::CaptureOrderSurvivorSubset) {
            size_t const ordinal = CaptureOrdinal(input, id);
            if (ordinal >= input.captureCurveCount ||
                (curve && ordinal <= CaptureOrdinal(input, input.survivorStableIds.data[curve - 1])))
                Fail(status);
        } else {
            if (curve && id <= input.survivorStableIds.data[curve - 1]) Fail(status);
            size_t const capture = LowerBound(input.captureStableIds, id);
            if (capture == input.captureCurveCount ||
                input.captureStableIds.data[capture] != id) {
                Fail(status);
            }
        }
    }
}

__global__ void EmitKernel(CurveTileInput input, CurveTileRequirements requirements,
    CurveTileSpan* spans, uint32_t const* status) {
    size_t const tile = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile >= requirements.tileCount || *status) return;
    // Reserve one capture chunk for every remaining tile.  This avoids
    // phantom initial tiles when the clamp floor raises tileCount above the
    // raw ceil(chunkCount / chunksPerTile) value.
    size_t const startChunk = min(tile * requirements.chunksPerTile,
        requirements.chunkCount - (requirements.tileCount - tile));
    size_t const endChunk = min((tile + 1) * requirements.chunksPerTile,
        requirements.chunkCount - (requirements.tileCount - (tile + 1)));
    size_t const start = min(startChunk * size_t(requirements.chunkSize),
                             input.captureCurveCount);
    size_t const end = min(endChunk * size_t(requirements.chunkSize),
                           input.captureCurveCount);
    size_t firstCurve;
    size_t lastCurve;
    if (input.order == CurveTileOrder::IdentityCaptureOrder) {
        firstCurve = start;
        lastCurve = end;
    } else if (input.order == CurveTileOrder::CaptureOrderSurvivorSubset) {
        firstCurve = SurvivorOrdinalLowerBound(input, start);
        lastCurve = SurvivorOrdinalLowerBound(input, end);
    } else {
        firstCurve = start == input.captureCurveCount
            ? input.survivorCurveCount
            : LowerBound(input.survivorStableIds, input.captureStableIds.data[start]);
        lastCurve = end == input.captureCurveCount
            ? input.survivorCurveCount
            : LowerBound(input.survivorStableIds, input.captureStableIds.data[end]);
    }
    uint32_t const firstPoint = input.survivorCurveOffsets.data[firstCurve];
    uint32_t const lastPoint = input.survivorCurveOffsets.data[lastCurve];
    spans[tile] = {static_cast<uint32_t>(tile), static_cast<uint32_t>(firstCurve),
                   static_cast<uint32_t>(lastCurve - firstCurve), firstPoint,
                   lastPoint - firstPoint};
}

uint32_t Clamp(uint32_t value, uint32_t low, uint32_t high) {
    return std::max(low, std::min(value, high));
}

bool Same(CurveTileRequirements const& a, CurveTileRequirements const& b) {
    return a.chunkCount == b.chunkCount && a.tileCount == b.tileCount &&
        a.chunksPerTile == b.chunksPerTile &&
        a.curvesPerTile == b.curvesPerTile && a.chunkSize == b.chunkSize &&
        a.tileTarget == b.tileTarget && a.deviceIndex == b.deviceIndex &&
        a.stream == b.stream;
}

cudaError_t Requirements(CurveTileOptions options, size_t captureCurves,
    size_t survivorCurves, size_t survivorPoints, CurveTileRequirements* result,
    cudaStream_t stream, bool prepare) {
    if (!result || captureCurves > UINT32_MAX || survivorCurves > captureCurves ||
        survivorPoints > UINT32_MAX || (survivorCurves == 0 && survivorPoints != 0) ||
        survivorPoints < survivorCurves) {
        return cudaErrorInvalidValue;
    }
    uint32_t const chunk = options.chunkSize
        ? Clamp(options.chunkSize, 128, 1024) : 512;
    uint32_t const target = options.tileTarget
        ? Clamp(options.tileTarget, 32, 256) : 64;
    size_t const chunks = captureCurves ? (captureCurves + chunk - 1) / chunk : 0;
    size_t const chunksPerTile = chunks ? std::max<size_t>(1, (chunks + target - 1) / target) : 0;
    size_t const rawTiles = chunks ? (chunks + chunksPerTile - 1) / chunksPerTile : 0;
    size_t const tileCount = chunks
        ? std::min(chunks, std::max<size_t>(32, std::min<size_t>(rawTiles, 256)))
        : 0;
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (prepare && stream) {
        int streamDevice = -1;
        status = cudaStreamGetDevice(stream, &streamDevice);
        if (status != cudaSuccess) return status;
        if (streamDevice != device) return cudaErrorInvalidResourceHandle;
    }
    *result = {chunks, tileCount, chunksPerTile, size_t(chunk) * chunksPerTile,
               chunk, target, device, stream};
    return cudaSuccess;
}

unsigned Blocks(size_t count) {
    size_t const needed = (count + kThreads - 1) / kThreads;
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(needed, 65535)));
}

} // namespace

cudaError_t GetCurveTileRequirements(CurveTileOptions options,
    size_t captureCurveCount, size_t survivorCurveCount, size_t survivorPointCount,
    CurveTileRequirements* result, cudaStream_t stream) {
    return Requirements(options, captureCurveCount, survivorCurveCount,
                        survivorPointCount, result, stream, true);
}

cudaError_t BuildCurveTiles(CurveTileInput input,
    CurveTileRequirements const& requirements, CurveTileOutput output,
    cudaStream_t stream) {
    CurveTileRequirements expected;
    CurveTileOptions const options{requirements.chunkSize, requirements.tileTarget};
    cudaError_t status = Requirements(options, input.captureCurveCount,
        input.survivorCurveCount, input.survivorPointCount, &expected, stream, false);
    if (status != cudaSuccess) return status;
    if (!ValidOrder(input.order) ||
        !Same(requirements, expected) ||
        input.captureStableIds.size != input.captureCurveCount ||
        input.survivorStableIds.size != input.survivorCurveCount ||
        input.survivorCurveOffsets.size != input.survivorCurveCount + 1 ||
        (input.captureCurveCount && !input.captureStableIds.data) ||
        (input.survivorCurveCount && !input.survivorStableIds.data) ||
        !input.survivorCurveOffsets.data || !output.status.data || output.status.size != 1 ||
        output.spans.size < requirements.tileCount ||
        (requirements.tileCount && !output.spans.data) ||
        (input.order == CurveTileOrder::CaptureOrderSurvivorSubset &&
            (input.sortedCaptureStableIds.size != input.captureCurveCount ||
             input.sortedCaptureOrdinals.size != input.captureCurveCount ||
             (input.captureCurveCount && (!input.sortedCaptureStableIds.data ||
                                         !input.sortedCaptureOrdinals.data)))) ||
        (input.order == CurveTileOrder::IdentityCaptureOrder &&
            input.survivorCurveCount != input.captureCurveCount)) {
        return cudaErrorInvalidValue;
    }
    status = cudaMemsetAsync(output.status.data, 0, sizeof(uint32_t), stream);
    if (status != cudaSuccess) return status;
    size_t const work = std::max(input.captureCurveCount, input.survivorCurveCount);
    ValidateKernel<<<Blocks(work), kThreads, 0, stream>>>(input, output.status.data);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    if (requirements.tileCount) {
        EmitKernel<<<Blocks(requirements.tileCount), kThreads, 0, stream>>>(
            input, requirements, output.spans.data, output.status.data);
        if ((status = cudaGetLastError()) != cudaSuccess) return status;
    }
    return cudaSuccess;
}

} // namespace usdGen::gpu
