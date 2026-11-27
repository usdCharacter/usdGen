// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// The CUDA completion lane and stream fence: results visible after a fence,
// continuations run once with the event status, many threads at once, and an
// idle lane runs no passes (nothing polls).
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGen/exec/taskScheduler.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

constexpr size_t kWords = 1 << 18;

struct Job : gpu::CudaCompletion {
    UsdGenPendingCount* pending = nullptr;
    uint32_t* host = nullptr;
    uint32_t expect = 0;
    std::atomic<int> runs{0};
    std::atomic<bool> sawData{false};
    cudaError_t seen = cudaErrorUnknown;
    Job() { taskRun = &Job::Done; }
    static void Done(UsdGenTask* t) noexcept {
        auto* job = static_cast<Job*>(t);
        job->seen = job->status;
        // The event covers the D2H copy, so the bytes are already here.
        job->sawData.store(job->host[0] == job->expect && job->host[kWords - 1] == job->expect,
                           std::memory_order_relaxed);
        job->runs.fetch_add(1, std::memory_order_acq_rel);
        job->pending->Release();
    }
};

// One stream's worth of work: fill a device buffer, copy it to pinned memory.
struct Lane {
    cudaStream_t stream = nullptr;
    uint32_t* device = nullptr;
    uint32_t* host = nullptr;
    bool Make() {
        return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
               cudaMalloc(&device, kWords * sizeof(uint32_t)) == cudaSuccess &&
               cudaMallocHost(&host, kWords * sizeof(uint32_t)) == cudaSuccess;
    }
    bool Issue(uint32_t byte) {
        host[0] = host[kWords - 1] = 0;
        return cudaMemsetAsync(device, int(byte), kWords * sizeof(uint32_t), stream) == cudaSuccess &&
               cudaMemcpyAsync(host, device, kWords * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               stream) == cudaSuccess;
    }
    void Free() {
        if (host) cudaFreeHost(host);
        if (device) cudaFree(device);
        if (stream) cudaStreamDestroy(stream);
    }
};

uint32_t Splat(uint32_t byte) { return byte * 0x01010101u; }

}  // namespace

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return 77;

    // Fence on an idle stream, then after real work.
    {
        Lane lane;
        CHECK(lane.Make());
        CHECK(gpu::CudaFenceStream(lane.stream) == cudaSuccess);
        for (uint32_t round = 1; round != 20; ++round) {
            CHECK(lane.Issue(round));
            CHECK(gpu::CudaFenceStream(lane.stream) == cudaSuccess);
            CHECK(lane.host[0] == Splat(round) && lane.host[kWords - 1] == Splat(round));
        }
        lane.Free();
    }

    // Stress: many threads fence their own streams concurrently, more threads
    // than the process scheduler has workers so no fence may need a worker.
    {
        int const threads = UsdGenParallelForConcurrency() + 8;
        std::atomic<int> failures{0};
        std::vector<std::thread> pool;
        for (int t = 0; t != threads; ++t) {
            pool.emplace_back([t, &failures, device] {
                cudaSetDevice(device);
                Lane lane;
                if (!lane.Make()) { failures.fetch_add(1); lane.Free(); return; }
                for (uint32_t round = 0; round != 60; ++round) {
                    uint32_t const byte = 1 + uint32_t((t * 61 + int(round)) % 250);
                    if (!lane.Issue(byte) || gpu::CudaFenceStream(lane.stream) != cudaSuccess ||
                        lane.host[0] != Splat(byte) || lane.host[kWords - 1] != Splat(byte))
                        failures.fetch_add(1);
                }
                lane.Free();
            });
        }
        for (auto& th : pool) th.join();
        CHECK(failures.load() == 0);
    }

    // Lane: completions submitted from many threads, every continuation runs
    // exactly once with cudaSuccess after its copy landed.
    {
        int const threads = 16;
        int const perThread = 40;
        UsdGenPendingCount pending;
        std::vector<Lane> lanes(threads);
        std::vector<Job> jobs(size_t(threads) * perThread);
        std::vector<cudaEvent_t> events(jobs.size(), nullptr);
        for (auto& lane : lanes) CHECK(lane.Make());
        for (size_t i = 0; i != jobs.size(); ++i)
            CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming) == cudaSuccess);
        std::atomic<int> submitFailures{0};
        std::vector<std::thread> pool;
        for (int t = 0; t != threads; ++t) {
            pool.emplace_back([&, t] {
                cudaSetDevice(device);
                for (int k = 0; k != perThread; ++k) {
                    Job& job = jobs[size_t(t) * perThread + k];
                    // One job in flight per stream: its pinned buffer is reused.
                    if (k) {
                        Job& prior = jobs[size_t(t) * perThread + k - 1];
                        while (!prior.runs.load(std::memory_order_acquire))
                            std::this_thread::yield();
                    }
                    uint32_t const byte = 1 + uint32_t((t * 7 + k) % 250);
                    job.event = events[size_t(t) * perThread + k];
                    job.pending = &pending;
                    job.host = lanes[size_t(t)].host;
                    job.expect = Splat(byte);
                    pending.Add();
                    if (!lanes[size_t(t)].Issue(byte) || !gpu::CudaCompletionSubmit(&job, lanes[size_t(t)].stream)) {
                        submitFailures.fetch_add(1);
                        pending.Release();
                    }
                }
            });
        }
        for (auto& th : pool) th.join();
        pending.Wait();
        CHECK(submitFailures.load() == 0);
        for (auto& job : jobs) {
            CHECK(job.runs.load() == 1);
            CHECK(job.seen == cudaSuccess);
            CHECK(job.sawData.load());
        }
        CHECK(gpu::CudaCompletionLanePendingForTesting() == 0);
        // Idle: no pass runs while nothing is pending (no timer, no polling).
        uint64_t const passes = gpu::CudaCompletionLanePassesForTesting();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(gpu::CudaCompletionLanePassesForTesting() == passes);
        for (auto e : events) cudaEventDestroy(e);
        for (auto& lane : lanes) lane.Free();
    }
    std::printf("PASS\n");
    return 0;
}
