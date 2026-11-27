// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/gpu/cudaCompletionLane.h"

#include "usdGen/epoch.h"
#include "usdGen/exec/serialLane.h"

#include <atomic>

namespace usdGen::gpu {
namespace {

// The lane: submitters push on `incoming`; the one runner moves them into
// its private ring (`pending`), queries each event and runs the finished
// ones. A UsdGenSerialLane keeps it serial with no kick lost.
struct Lane {
    UsdGenTreiberStack<CudaCompletion, &CudaCompletion::laneNext> incoming;
    CudaCompletion *pending = nullptr;  // runner only
    std::atomic<size_t> count{0};
    UsdGenSerialLane serial{&Lane::Pass, nullptr};

    void Kick() noexcept { serial.Kick(); }

    static void Pass(void *) noexcept;
};

// Constant-initialized and never destroyed: a late stream callback during
// process teardown still finds it.
Lane g_lane;

void Lane::Pass(void *) noexcept
{
    Lane &lane = g_lane;
    if (CudaCompletion *in = lane.incoming.PopAll()) {
        CudaCompletion *last = in;
        while (last->laneNext) last = last->laneNext;
        last->laneNext = lane.pending;
        lane.pending = in;
    }
    CudaCompletion **link = &lane.pending;
    while (CudaCompletion *c = *link) {
        cudaError_t const e = cudaEventQuery(c->event);
        if (e == cudaErrorNotReady) {
            link = &c->laneNext;
            continue;
        }
        *link = c->laneNext;
        c->laneNext = nullptr;
        c->status = e;
        lane.count.fetch_sub(1, std::memory_order_relaxed);
        c->taskRun(c);  // may free c
    }
}

void CUDART_CB LaneKick(cudaStream_t, cudaError_t, void *) noexcept { g_lane.Kick(); }

}  // namespace

bool CudaCompletionSubmit(CudaCompletion *c, cudaStream_t stream) noexcept
{
    c->laneNext = nullptr;
    c->status = cudaEventRecord(c->event, stream);
    if (c->status != cudaSuccess) return false;
    c->status = cudaErrorNotReady;
    g_lane.count.fetch_add(1, std::memory_order_relaxed);
    g_lane.incoming.Push(c);
    // The callback runs after the event completes (or with the error that
    // stopped the stream) and kicks a pass. A completion that is already done
    // is found by the pass this kick starts.
    if (cudaStreamAddCallback(stream, LaneKick, nullptr, 0) != cudaSuccess) {
        // The stream cannot carry a callback: the device is being lost. Wait
        // here, in the device-loss category, so the pass below sees it done.
        // shutdown wait: device loss: the stream cannot carry a callback.
        (void)cudaEventSynchronize(c->event);
    }
    g_lane.Kick();
    return true;
}

uint64_t CudaCompletionLanePassesForTesting() noexcept
{
    return g_lane.serial.Passes();
}

size_t CudaCompletionLanePendingForTesting() noexcept
{
    return g_lane.count.load(std::memory_order_relaxed);
}

}  // namespace usdGen::gpu
