// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// CudaFenceStream lives in usdGenGpu with the kernels that call it; it needs
// only the parking word, not the scheduler (the lane is in usdGen).
#include "usdGen/gpu/cudaCompletionLane.h"

#include "usdGen/parkingWord.h"

#include <atomic>

namespace usdGen::gpu {
namespace {

struct Fence {
    std::atomic<uint32_t> done{0};
    cudaError_t status = cudaErrorUnknown;
};

void CUDART_CB FenceDone(cudaStream_t, cudaError_t status, void *data) noexcept
{
    auto *fence = static_cast<Fence *>(data);
    fence->status = status;
    // The store is the last touch of *fence; the wake is keyed by address.
    std::atomic<uint32_t> *word = &fence->done;
    word->store(1, std::memory_order_release);
    UsdGenUnparkAll(*word);
}

}  // namespace

// shutdown wait: the blocking primitive itself.
cudaError_t CudaFenceStream(cudaStream_t stream) noexcept
{
    // Already idle (common after small work): no callback round trip.
    cudaError_t const now = cudaStreamQuery(stream);
    if (now != cudaErrorNotReady) return now;
    Fence fence;
    cudaError_t const added = cudaStreamAddCallback(stream, FenceDone, &fence, 0);
    if (added != cudaSuccess) {
        // Device loss or an invalid stream: report what the stream reports.
        // shutdown wait: device-loss fallback of the primitive.
        cudaError_t const e = cudaStreamSynchronize(stream);
        return e != cudaSuccess ? e : added;
    }
    while (!fence.done.load(std::memory_order_acquire)) UsdGenParkWhileEqual(fence.done, 0);
    return fence.status;
}

}  // namespace usdGen::gpu
