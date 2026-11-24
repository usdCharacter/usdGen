// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/gpu/cudaCompletionLane.h"

#include "usdGen/epoch.h"
#include "usdGen/parkingWord.h"

#include <atomic>

namespace usdGen::gpu {
namespace {

// The lane: submitters push on `incoming`; the one runner moves them into
// its private ring (`pending`), queries each event and runs the finished
// ones. `state` makes it serial: SCHEDULED while a pass is queued or running,
// DIRTY when a kick arrived since the pass began, so no kick is lost.
struct Lane;
extern Lane g_lane;

struct Lane {
    static constexpr uint32_t kScheduled = 1u, kDirty = 2u;
    UsdGenTreiberStack<CudaCompletion, &CudaCompletion::laneNext> incoming;
    CudaCompletion *pending = nullptr;  // runner only
    std::atomic<uint32_t> state{0};
    std::atomic<uint64_t> passes{0};
    std::atomic<size_t> count{0};
    UsdGenTask task{&Lane::Run};


    void Kick() noexcept
    {
        uint32_t const old = state.fetch_or(kScheduled | kDirty, std::memory_order_acq_rel);
        if (!(old & kScheduled)) UsdGenTaskScheduler::Process().Post(&task);
    }

    void Pass() noexcept
    {
        passes.fetch_add(1, std::memory_order_relaxed);
        if (CudaCompletion *in = incoming.PopAll()) {
            CudaCompletion *last = in;
            while (last->laneNext) last = last->laneNext;
            last->laneNext = pending;
            pending = in;
        }
        CudaCompletion **link = &pending;
        while (CudaCompletion *c = *link) {
            cudaError_t const e = cudaEventQuery(c->event);
            if (e == cudaErrorNotReady) {
                link = &c->laneNext;
                continue;
            }
            *link = c->laneNext;
            c->laneNext = nullptr;
            c->status = e;
            count.fetch_sub(1, std::memory_order_relaxed);
            c->taskRun(c);  // may free c
        }
    }

    static void Run(UsdGenTask *t) noexcept
    {
        (void)t;
        Lane &lane = g_lane;
        uint32_t s = lane.state.fetch_and(~kDirty, std::memory_order_acq_rel);
        for (;;) {
            lane.Pass();
            s = kScheduled;
            if (lane.state.compare_exchange_strong(s, 0, std::memory_order_acq_rel,
                                                   std::memory_order_acquire))
                return;
            lane.state.fetch_and(~kDirty, std::memory_order_acq_rel);  // kicked meanwhile
        }
    }
};

// Constant-initialized and never destroyed: a late stream callback during
// process teardown still finds it.
Lane g_lane;

void CUDART_CB LaneKick(cudaStream_t, cudaError_t, void *) noexcept { g_lane.Kick(); }

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
        (void)cudaEventSynchronize(c->event);
    }
    g_lane.Kick();
    return true;
}

cudaError_t CudaFenceStream(cudaStream_t stream) noexcept
{
    Fence fence;
    cudaError_t const added = cudaStreamAddCallback(stream, FenceDone, &fence, 0);
    if (added != cudaSuccess) {
        // Device loss or an invalid stream: report what the stream reports.
        cudaError_t const e = cudaStreamSynchronize(stream);
        return e != cudaSuccess ? e : added;
    }
    while (!fence.done.load(std::memory_order_acquire)) UsdGenParkWhileEqual(fence.done, 0);
    return fence.status;
}

uint64_t CudaCompletionLanePassesForTesting() noexcept
{
    return g_lane.passes.load(std::memory_order_relaxed);
}

size_t CudaCompletionLanePendingForTesting() noexcept
{
    return g_lane.count.load(std::memory_order_relaxed);
}

}  // namespace usdGen::gpu
