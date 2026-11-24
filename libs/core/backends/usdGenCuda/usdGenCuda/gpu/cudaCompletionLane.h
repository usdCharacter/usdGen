// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — CUDA completion without blocking a thread in the driver.
//
// - CudaCompletionSubmit: records an event on a stream; the completion lane
//   polls cudaEventQuery over its pending ring and runs the continuation once
//   the event finished or failed. The lane is a serial task on the process
//   scheduler, kicked by a stream callback behind each event, so it runs only
//   when some GPU work just finished and holds no thread while idle.
// - CudaFenceStream: the synchronous executor's stand-in for
//   cudaStreamSynchronize. A stream callback flips a word and the caller parks
//   on it (parkingWord), so the wait spins nowhere and needs no worker.
//
// Continuations run on the lane and must not block; post heavier work.
// cudaStreamSynchronize is left only at shutdown, device loss, destructors
// and in tests.
#pragma once

#include "usdGen/taskScheduler.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace usdGen::gpu {

/// One pending completion, owned by the submitter; it must stay alive until
/// its taskRun starts. `event` is the caller's (made once, reused).
struct CudaCompletion : UsdGenTask {
    cudaEvent_t event = nullptr;
    /// cudaSuccess or the event's error, written before taskRun.
    cudaError_t status = cudaErrorNotReady;
    CudaCompletion *laneNext = nullptr;
};

/// Records `completion->event` on `stream` and queues it on the lane. Never
/// blocks. On false nothing was queued and `completion->status` says why.
bool CudaCompletionSubmit(CudaCompletion *completion, cudaStream_t stream) noexcept;

/// Parks until every operation queued on `stream` so far finished; returns
/// its status (cudaStreamSynchronize's contract). Never from a CUDA callback.
cudaError_t CudaFenceStream(cudaStream_t stream) noexcept;

/// Test observation: lane passes run, and completions still pending.
uint64_t CudaCompletionLanePassesForTesting() noexcept;
size_t CudaCompletionLanePendingForTesting() noexcept;

}  // namespace usdGen::gpu
